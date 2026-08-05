# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
    _batch_sampling_input,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3VisualEncodingStage(PipelineStage):
    def __init__(
        self,
        video_vae,
        vae_arch_config,
    ) -> None:
        super().__init__()
        self.video_vae = video_vae
        self.vae_arch_config = vae_arch_config

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "video_vae")]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_cleanup_temp_dirs,
        )

        try:
            return self._forward(batch, server_args)
        except Exception:
            # No later stage will run after an encoder failure.
            minimax_h3_cleanup_temp_dirs(batch)
            raise
        finally:
            # Qwen and the visual tokenizer have both consumed the derived
            # reference-video files by this point. Original localized media
            # remains alive for the following audio stage.
            minimax_h3_cleanup_temp_dirs(batch, owners=("reference_video",))

    def _forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is not None:
            routed = plan.encoders.get("visual")
            if not routed:
                return batch
            self._encode_keyframes_from_plan(batch, plan, routed)
            return batch
        if _batch_sampling_input(batch, "image_path") is not None:
            raise NotImplementedError(
                "MiniMaxH3VisualEncodingStage direct visual tokenizer encode "
                "requires a canonical minimax_h3 request (resolved plan); "
                "legacy image_path-only requests are unsupported."
            )
        return batch

    def _encode_keyframes_from_plan(self, batch: Req, plan, routed) -> None:
        """Direct keyframe encode: seeded sampled encode_images ->
        normalized [n,96] cond rows in batch.extra."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            minimax_h3_encode_keyframe_cond_rows,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.sp_broadcast import (
            minimax_h3_sp_broadcast_extra,
            minimax_h3_sp_ctx,
        )

        materials = [m for m in plan.materials if m.condition_index in set(routed)]
        chains = {m.material_chain for m in materials}
        keyframe_materials = [
            material
            for material in materials
            if material.material_chain == "image.target_canvas"
        ]
        if str(plan.task) == "fl2va":
            frame_indices = tuple(
                material.frame_index for material in keyframe_materials
            )
            if frame_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
                raise ValueError(
                    "fl2va visual encoding requires an ordered keyframe signature "
                    f"in {MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got "
                    f"{frame_indices!r}"
                )
        elif keyframe_materials:
            raise ValueError(
                f"task {plan.task!r} cannot carry image.target_canvas materials"
            )
        if MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY in batch.extra:
            return
        if chains == {"image.reference_preserve"}:
            self._encode_reference_image(batch, plan)
            return
        video_chains = {
            "video.reference_preserve",
            "video_audio.reference_preserve",
        }
        if chains and chains <= {"image.reference_preserve", *video_chains}:
            if "image.reference_preserve" in chains:
                self._encode_reference_image(batch, plan)
            if chains & video_chains:
                self._encode_reference_video(batch, plan)
            return
        unsupported = [
            m.material_chain
            for m in materials
            if m.material_chain != "image.target_canvas"
        ]
        if unsupported:
            raise NotImplementedError(
                "MiniMaxH3VisualEncodingStage direct encode only supports "
                f"image.target_canvas / image.reference_preserve, got {unsupported}"
            )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.canvas import (
            minimax_h3_prepared_keyframes,
        )

        _, sp_rank = minimax_h3_sp_ctx()
        if sp_rank == 0:
            prepared = minimax_h3_prepared_keyframes(batch, plan)
            prepared_indices = tuple(prepared.get("semantic_frame_indices") or ())
            if prepared_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES or len(
                prepared.get("images") or ()
            ) != len(prepared_indices):
                raise ValueError(
                    "fl2va visual preparation requires one or two ordered images "
                    "with a supported semantic_frame_indices signature"
                )
            encoded = []
            rows_list = []
            for item in prepared["images"]:
                image = item["image"]
                width, height = item["canvas_width"], item["canvas_height"]
                # The encode sampling seed is pinned at 42 (the VAE sample
                # seed is part of the contract), independent of the
                # request seed.
                rows = minimax_h3_encode_keyframe_cond_rows(
                    self.video_vae,
                    image,
                    self.vae_arch_config,
                )
                encoded.append(
                    {
                        "rows": rows,
                        "latent_h": height // 16,
                        "latent_w": width // 16,
                        "canvas_height": height,
                        "canvas_width": width,
                        "frame_index": item.get("frame_index"),
                        "resolved_frame_index": item.get("resolved_frame_index"),
                        "condition_index": item.get("condition_index"),
                    }
                )
                rows_list.append(rows)
            rows = torch.cat(rows_list, dim=0)
            first = encoded[0]
            batch.extra[MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY] = {
                "rows": rows,
                "latent_h": first["latent_h"],
                "latent_w": first["latent_w"],
                "canvas_height": first["canvas_height"],
                "canvas_width": first["canvas_width"],
                "keyframes": encoded,
                "semantic_frame_indices": prepared.get("semantic_frame_indices"),
                "pixel_frame_indices": prepared.get("pixel_frame_indices"),
                "frame_count": prepared.get("frame_count"),
            }
        minimax_h3_sp_broadcast_extra(batch, MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY)

    def _encode_reference_video(self, batch: Req, plan) -> None:
        """ref2va video/video_audio encode: prepare chain (fps
        normalize + cap_resize + truncate, all libx264 re-encodes) + the
        encode_videos recipe; rows use the video's OWN latent grid."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_encode_reference_video_rows,
            minimax_h3_prepared_reference_videos,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.sp_broadcast import (
            minimax_h3_sp_broadcast_extra,
            minimax_h3_sp_ctx,
        )

        if MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY in batch.extra:
            return
        _, sp_rank = minimax_h3_sp_ctx()
        if sp_rank == 0:
            prepared = minimax_h3_prepared_reference_videos(batch, plan)
            videos = prepared.get("videos")
            if not isinstance(videos, list) or not videos:
                raise ValueError(
                    "prepared reference videos payload must carry a non-empty "
                    f"'videos' list, got {videos!r}"
                )
            entries = []
            for item in videos:
                rows, latent_t, latent_h, latent_w = (
                    minimax_h3_encode_reference_video_rows(
                        self.video_vae,
                        item["prepared_path"],
                        self.vae_arch_config,
                    )
                )
                entries.append(
                    {
                        "rows": rows,
                        "latent_t": latent_t,
                        "latent_h": latent_h,
                        "latent_w": latent_w,
                        "condition_index": int(item["condition_index"]),
                        "material_chain": str(item["material_chain"]),
                    }
                )
            payload = dict(entries[0])
            payload["videos"] = entries
            batch.extra[MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY] = payload
        minimax_h3_sp_broadcast_extra(batch, MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY)

    def _encode_reference_image(self, batch: Req, plan) -> None:
        """ref2va reference image encode: cap_resize (intrinsic
        geometry) + the verified keyframe recipe; rows use the image's OWN
        latent grid, not the target canvas."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            minimax_h3_encode_keyframe_cond_rows,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_prepared_reference_image,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.sp_broadcast import (
            minimax_h3_sp_broadcast_extra,
            minimax_h3_sp_ctx,
        )

        if MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY in batch.extra:
            return
        _, sp_rank = minimax_h3_sp_ctx()
        if sp_rank == 0:
            prepared = minimax_h3_prepared_reference_image(batch, plan)
            entries = []
            for item in prepared["images"]:
                image = item["image"]
                rows = minimax_h3_encode_keyframe_cond_rows(
                    self.video_vae,
                    image,
                    self.vae_arch_config,
                )
                width, height = image.size
                entries.append(
                    {
                        "rows": rows,
                        "latent_h": height // 16,
                        "latent_w": width // 16,
                        "condition_index": int(item["condition_index"]),
                        "material_chain": "image.reference_preserve",
                    }
                )
            payload = dict(entries[0])  # single-image consumers keep the keys
            payload["images"] = entries
            batch.extra[MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY] = payload
        minimax_h3_sp_broadcast_extra(batch, MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY)


__all__ = ["MiniMaxH3VisualEncodingStage"]
