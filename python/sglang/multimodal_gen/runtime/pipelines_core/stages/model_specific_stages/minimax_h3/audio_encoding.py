# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
    _batch_sampling_input,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3AudioEncodingStage(PipelineStage):
    def __init__(self, audio_vae, vae_arch_config) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.vae_arch_config = vae_arch_config

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "audio_vae")]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_cleanup_temp_dirs,
        )

        try:
            return self._forward(batch, server_args)
        finally:
            # Audio encoding is the final material consumer in the MiniMax H3
            # encoder pipeline, including requests with no routed audio.
            minimax_h3_cleanup_temp_dirs(batch, owners=("material",))

    def _forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is not None:
            routed = plan.encoders.get("audio")
            if not routed:
                return batch
            self._encode_references_from_plan(batch, plan, routed)
            return batch
        if _batch_sampling_input(batch, "audio_path") is not None:
            raise NotImplementedError(
                "MiniMaxH3AudioEncodingStage direct audio tokenizer encode "
                "requires a canonical minimax_h3 request (resolved plan); "
                "legacy audio_path-only requests are unsupported."
            )
        return batch

    def _encode_references_from_plan(self, batch: Req, plan, routed) -> None:
        """Direct reference-audio encode: audio VAE posterior mean ->
        normalized channel-major rows in batch.extra."""
        routed_set = set(routed)
        routed_materials = [
            material
            for material in plan.materials
            if material.condition_index in routed_set
        ]
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_localize_material_uri,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_encode_reference_audio_rows,
            minimax_h3_reference_video_has_audio,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.sp_broadcast import (
            minimax_h3_sp_broadcast_extra,
            minimax_h3_sp_ctx,
        )

        if MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY in batch.extra:
            return
        _, sp_rank = minimax_h3_sp_ctx()
        if sp_rank == 0:
            materials = routed_materials
            if not materials:
                raise ValueError("ref2va audio routing selected no reference materials")
            entries = []
            for material in materials:
                audio_path = minimax_h3_localize_material_uri(
                    batch,
                    material.uri,
                    condition_type=material.condition_type,
                    condition_index=int(material.condition_index),
                )
                material_chain = str(material.material_chain)
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
                    MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
                )

                source_facts = batch.extra.get(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, {}).get(
                    int(material.condition_index)
                )
                input_has_audio = True
                if material_chain == "video.reference_preserve":
                    input_has_audio = (
                        bool(source_facts.get("has_audio"))
                        if isinstance(source_facts, dict)
                        else minimax_h3_reference_video_has_audio(audio_path)
                    )
                if material_chain == "video.reference_preserve" and not input_has_audio:
                    # Keep the visual reference block in request order while
                    # representing the absent soundtrack as a zero-length
                    # audio condition. Videos whose probe reports
                    # metadata.has_audio=false contribute no audio condition.
                    out = {
                        "rows": torch.empty((0, 32), dtype=torch.float32),
                        "ref_audio_t": 0,
                        "duration_seconds": 0.0,
                    }
                else:
                    out = minimax_h3_encode_reference_audio_rows(
                        self.audio_vae,
                        audio_path,
                        self.vae_arch_config,
                        material_chain=material_chain,
                    )
                entries.append(
                    {
                        **out,
                        "condition_index": int(material.condition_index),
                        "material_chain": material_chain,
                    }
                )
            # Single-entry payloads keep the entry's fields at the top level
            # for backward compatibility; "audios" always carries the full list.
            payload = dict(entries[0]) if len(entries) == 1 else {}
            payload["audios"] = entries
            batch.extra[MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY] = payload
        minimax_h3_sp_broadcast_extra(batch, MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY)


__all__ = ["MiniMaxH3AudioEncodingStage"]
