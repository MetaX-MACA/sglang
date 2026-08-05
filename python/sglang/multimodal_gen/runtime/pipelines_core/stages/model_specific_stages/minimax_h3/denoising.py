# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 denoise sink for packed-token DiT stepping, CFG-distilled
single-branch execution, and payload validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)

_REF2VA_VIDEO_CHAINS = {
    "video.reference_preserve",
    "video_audio.reference_preserve",
}


def minimax_h3_condition_noise_aug(sampling: Any) -> tuple[float, float]:
    """Resolve condition timesteps using the model defaults."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )

    imgvid_noise_aug = getattr(
        sampling,
        "imgvid_cond_noise_aug_for_inference",
        None,
    )
    if imgvid_noise_aug is None:
        # The model default uses imgvid noise aug 0.999. A request selecting
        # imgvid noise aug 1.0 still overrides this.
        imgvid_noise_aug = MINIMAX_H3_IMGVID_COND_TIMESTEP
    audio_noise_aug = getattr(
        sampling,
        "audio_cond_noise_aug_for_inference",
        None,
    )
    if audio_noise_aug is None:
        audio_noise_aug = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP
    return float(imgvid_noise_aug), float(audio_noise_aug)


def _validate_fl2va_keyframe_payload(plan: Any, keyframe: Any) -> None:
    """Reject stale/middle/reordered keyframe payloads at the DiT sink."""

    task = None if plan is None else str(plan.task)
    if task != "fl2va":
        if keyframe is not None:
            raise ValueError(
                "keyframe condition rows are only valid for plan.task='fl2va'"
            )
        return
    if not isinstance(keyframe, Mapping):
        raise ValueError("fl2va denoising requires encoded keyframe condition rows")

    semantic_indices = tuple(keyframe.get("semantic_frame_indices") or ())
    if semantic_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "fl2va denoising requires semantic_frame_indices in "
            f"{MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, "
            f"got {semantic_indices!r}"
        )
    frame_count = keyframe.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("fl2va keyframe payload requires an integer frame_count")
    if frame_count <= 1:
        raise ValueError("fl2va keyframe payload frame_count must be greater than one")
    pixel_indices = keyframe.get("pixel_frame_indices")
    expected_pixel_indices = [
        frame_count - 1 if index == -1 else index for index in semantic_indices
    ]
    if pixel_indices != expected_pixel_indices:
        raise ValueError(
            "fl2va denoising requires pixel_frame_indices resolved from the "
            "semantic anchors, "
            f"got {pixel_indices!r} for frame_count={frame_count}"
        )

    entries = keyframe.get("keyframes")
    if (
        not isinstance(entries, list)
        or len(entries) != len(semantic_indices)
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        raise ValueError(
            "fl2va denoising requires one encoded keyframe per semantic anchor"
        )
    if [entry.get("frame_index") for entry in entries] != list(semantic_indices):
        raise ValueError("fl2va encoded keyframes must remain in semantic anchor order")
    if [
        entry.get("resolved_frame_index") for entry in entries
    ] != expected_pixel_indices:
        raise ValueError(
            "fl2va encoded keyframes must carry matching resolved_frame_index values"
        )

    latent_h = int(keyframe.get("latent_h") or 0)
    latent_w = int(keyframe.get("latent_w") or 0)
    rows = keyframe.get("rows")
    expected_rows = len(semantic_indices) * (latent_h // 2) * (latent_w // 2)
    if (
        latent_h <= 0
        or latent_w <= 0
        or not isinstance(rows, torch.Tensor)
        or int(rows.shape[0]) != expected_rows
    ):
        actual_rows = None if not isinstance(rows, torch.Tensor) else int(rows.shape[0])
        raise ValueError(
            "fl2va encoded keyframe rows do not match target-canvas blocks: "
            f"expected={expected_rows}, actual={actual_rows}"
        )


def _imgvid_condition_shapes(
    *,
    ref2va_blocks: list[dict[str, int | str]] | None,
    keyframe: Any,
    is_ref2va: bool,
) -> list[tuple[int, int, int]]:
    """Return visual-condition ``(T,H,W)`` in packed anchor-row order."""

    if ref2va_blocks is not None:
        shapes = []
        for block in ref2va_blocks:
            kind = str(block["kind"])
            if kind == "image":
                shapes.append((1, int(block["latent_h"]), int(block["latent_w"])))
            elif kind in {"video", "video_audio"}:
                shapes.append(
                    (
                        int(block["latent_t"]),
                        int(block["latent_h"]),
                        int(block["latent_w"]),
                    )
                )
        return shapes

    if is_ref2va:
        # ref2va always carries a resolved plan, so ordered blocks are
        # supplied above; reaching here indicates an upstream bug.
        raise ValueError("ref2va visual-condition shapes require ordered blocks")

    if not isinstance(keyframe, Mapping):
        return []
    entries = keyframe.get("keyframes")
    if isinstance(entries, list) and entries:
        return [
            (1, int(entry["latent_h"]), int(entry["latent_w"])) for entry in entries
        ]

    latent_h = int(keyframe["latent_h"])
    latent_w = int(keyframe["latent_w"])
    frame_rows = (latent_h // 2) * (latent_w // 2)
    rows = keyframe["rows"]
    if frame_rows <= 0 or int(rows.shape[0]) % frame_rows:
        raise ValueError(
            "legacy keyframe rows cannot be split into visual-condition frames"
        )
    return [(1, latent_h, latent_w)] * (int(rows.shape[0]) // frame_rows)


def _ref2va_payload_entry(
    payload: Any,
    *,
    list_key: str,
    condition_index: int,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} is required for ref2va condition rows")
    entries = payload.get(list_key)
    if isinstance(entries, list):
        for entry in entries:
            if (
                isinstance(entry, Mapping)
                and entry.get("condition_index") is not None
                and int(entry["condition_index"]) == int(condition_index)
            ):
                return entry
        if len(entries) == 1 and isinstance(entries[0], Mapping):
            return entries[0]
        raise ValueError(
            f"{path}.{list_key} missing entry for condition_index={condition_index}"
        )
    if payload.get("condition_index") is None or int(payload["condition_index"]) == int(
        condition_index
    ):
        return payload
    raise ValueError(
        f"{path}.{list_key} missing entry for condition_index={condition_index}"
    )


def _cat_optional(rows: list[torch.Tensor]) -> torch.Tensor | None:
    if not rows:
        return None
    return torch.cat(rows, dim=0)


def _ref2va_ordered_blocks_and_rows(
    *,
    plan: Any,
    ref_image: Any,
    ref_audio: Any,
    ref_video: Any,
) -> tuple[list[dict[str, int | str]], torch.Tensor | None, torch.Tensor | None]:
    blocks: list[dict[str, int | str]] = []
    visual_rows: list[torch.Tensor] = []
    audio_rows: list[torch.Tensor] = []
    for material in plan.materials:
        chain = str(material.material_chain)
        condition_index = int(material.condition_index)
        if chain == "image.reference_preserve":
            entry = _ref2va_payload_entry(
                ref_image,
                list_key="images",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_image_rows",
            )
            blocks.append(
                {
                    "kind": "image",
                    "latent_h": int(entry["latent_h"]),
                    "latent_w": int(entry["latent_w"]),
                }
            )
            visual_rows.append(entry["rows"])
        elif chain == "audio":
            entry = _ref2va_payload_entry(
                ref_audio,
                list_key="audios",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_audio_rows",
            )
            ref_audio_t = int(entry["ref_audio_t"])
            blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
            if ref_audio_t > 0:
                audio_rows.append(entry["rows"])
        elif chain in _REF2VA_VIDEO_CHAINS:
            video_entry = _ref2va_payload_entry(
                ref_video,
                list_key="videos",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_video_rows",
            )
            audio_entry = _ref2va_payload_entry(
                ref_audio,
                list_key="audios",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_audio_rows",
            )
            ref_audio_t = int(audio_entry["ref_audio_t"])
            blocks.append(
                {
                    "kind": (
                        "video_audio"
                        if chain == "video_audio.reference_preserve"
                        else "video"
                    ),
                    "ref_audio_t": ref_audio_t,
                    "latent_t": int(video_entry["latent_t"]),
                    "latent_h": int(video_entry["latent_h"]),
                    "latent_w": int(video_entry["latent_w"]),
                }
            )
            visual_rows.append(video_entry["rows"])
            if ref_audio_t > 0:
                audio_rows.append(audio_entry["rows"])
        else:
            raise ValueError(f"unsupported ref2va material chain {chain!r}")
    return blocks, _cat_optional(visual_rows), _cat_optional(audio_rows)


def _resolve_denoise_model(transformer: Any, device: torch.device) -> Any:
    """Resolve the DiT module and place it for denoise.

    FSDP-managed modules own their device placement; move only plain modules.
    """
    model = getattr(transformer, "model", transformer)
    if is_fsdp_managed_module(model):
        return model.eval()
    return model.to(device).eval()


class MiniMaxH3DenoisingStage(PipelineStage):
    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                memory_intensive=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
            MINIMAX_H3_SIGMAS_EXTRA_KEY,
            MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
        )

        if MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY in batch.extra:
            for required in (
                MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
                MINIMAX_H3_SIGMAS_EXTRA_KEY,
            ):
                if required not in batch.extra:
                    raise ValueError(
                        f"direct full-loop denoise requires batch.extra[{required!r}]"
                    )
            self._run_full_loop(batch, server_args)
            return batch
        if (
            batch.latents is not None
            or batch.audio_latents is not None
            or batch.timestep is not None
            or batch.timesteps is not None
        ):
            raise NotImplementedError(
                "MiniMaxH3DenoisingStage requires the canonical direct pipeline "
                "to populate text embeddings, denoise state, and sigma schedules."
            )
        return batch

    def _run_full_loop(self, batch: Req, server_args: ServerArgs) -> None:
        """Assemble the cfg-distilled positive input and run the full loop.

        The heavy lifting is decomposed into per-phase helpers:
        context/payload resolution, condition-row assembly, packed-layout
        construction, condition noise augmentation, initial-row expansion,
        the denoise loop itself, and output publication.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
            minimax_h3_denoise_loop,
        )

        ctx = _resolve_full_loop_context(batch)

        if not torch.cuda.is_available():
            raise RuntimeError("MiniMax H3 full-loop denoise requires CUDA")
        device = torch.device("cuda")
        model = _resolve_denoise_model(self.transformer, device)

        _assemble_condition_rows(ctx)

        emb = ctx.embeddings["positive"]
        packed = _build_packed_layout(ctx, emb)
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"].view(-1)] = (
            emb["text_token_tags"].view(-1).to(torch.long)
        )
        positive = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=emb["hidden_states"],
            token_tags=tags,
            device=device,
        )

        sampling = batch.sampling_params
        imgvid_noise_aug, audio_noise_aug = minimax_h3_condition_noise_aug(sampling)
        _apply_condition_noise_aug(
            ctx,
            sampling=sampling,
            imgvid_noise_aug=imgvid_noise_aug,
            audio_noise_aug=audio_noise_aug,
        )

        initial_video, initial_audio = _expand_initial_rows(ctx, positive)

        sigmas_video = [float(v) for v in ctx.sigmas["video"]]
        with self.progress_bar(
            total=len(sigmas_video) - 1, batch=batch, desc="minimax_h3 denoise"
        ) as progress_bar:
            video_rows, audio_rows = minimax_h3_denoise_loop(
                model=model,
                positive=positive,
                initial_video_rows=initial_video,
                initial_audio_rows=initial_audio,
                keyframe_cond_rows=ctx.cond_rows,
                audio_ref_rows=ctx.audio_ref_rows,
                sigmas_video=sigmas_video,
                sigmas_audio=[float(v) for v in ctx.sigmas["audio"]],
                device=device,
                imgvid_cond_noise_aug_for_inference=float(imgvid_noise_aug),
                audio_cond_noise_aug_for_inference=float(audio_noise_aug),
                on_step=lambda step, v, a: progress_bar.update(),
                step_profiler=lambda step_index: StageProfiler(
                    f"denoising_step_{step_index}",
                    logger=logger,
                    metrics=batch.metrics,
                    perf_dump_path_provided=batch.perf_dump_path is not None,
                    record_as_step=True,
                ),
            )
        _publish_full_loop_outputs(
            ctx,
            batch=batch,
            positive=positive,
            video_rows=video_rows,
            audio_rows=audio_rows,
        )


class _FullLoopContext:
    """Mutable per-request state threaded through the full-loop phases."""

    __slots__ = (
        "plan",
        "keyframe",
        "ref_image",
        "ref_audio",
        "ref_video",
        "is_ref2va",
        "embeddings",
        "state",
        "sigmas",
        "latent_t",
        "latent_h",
        "latent_w",
        "audio_t",
        "ref2va_positive_blocks",
        "cond_rows",
        "audio_ref_rows",
        "include_cond",
        "keyframe_frame_indices",
        "keyframe_frame_count",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, None)
        self.is_ref2va = False
        self.include_cond = False


def _resolve_full_loop_context(batch: Req) -> _FullLoopContext:
    """Read/validate extras and the denoise state into a loop context.

    Enforces the task-payload exclusivity rules (keyframe vs reference
    exclusivity) and cross-checks the resolved latent dims.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
        MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
        MINIMAX_H3_SIGMAS_EXTRA_KEY,
        MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        minimax_h3_plan_from_batch,
    )

    ctx = _FullLoopContext()
    ctx.embeddings = batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]
    ctx.state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
    ctx.sigmas = batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
    ctx.plan = minimax_h3_plan_from_batch(batch)
    ctx.keyframe = batch.extra.get(MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY)
    ctx.ref_image = batch.extra.get(MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY)
    ctx.ref_audio = batch.extra.get(MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY)
    ctx.ref_video = batch.extra.get(MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY)
    ctx.is_ref2va = (
        ctx.ref_image is not None
        or ctx.ref_audio is not None
        or ctx.ref_video is not None
    )
    if ctx.is_ref2va and ctx.keyframe is not None:
        raise ValueError("keyframe and reference extras are mutually exclusive")
    _validate_fl2va_keyframe_payload(ctx.plan, ctx.keyframe)

    ctx.latent_t = int(ctx.state["latent_t"])
    ctx.latent_h = int(ctx.state["latent_h"])
    ctx.latent_w = int(ctx.state["latent_w"])
    ctx.audio_t = int(ctx.state["audio_t"])
    return ctx


def _assemble_condition_rows(ctx: _FullLoopContext) -> None:
    """Populate cond/audio reference rows and keyframe metadata per task."""

    ctx.include_cond = (
        ctx.keyframe is not None
        or ctx.ref_image is not None
        or ctx.ref_video is not None
    )
    if ctx.is_ref2va:
        if ctx.plan is None:
            raise ValueError(
                "ref2va reference extras require a resolved plan; "
                "plan-less ref2va requests are unsupported"
            )
        ctx.ref2va_positive_blocks, ctx.cond_rows, ctx.audio_ref_rows = (
            _ref2va_ordered_blocks_and_rows(
                plan=ctx.plan,
                ref_image=ctx.ref_image,
                ref_audio=ctx.ref_audio,
                ref_video=ctx.ref_video,
            )
        )
        ctx.include_cond = ctx.cond_rows is not None
    else:
        ctx.cond_rows = ctx.keyframe["rows"] if ctx.include_cond else None
        ctx.audio_ref_rows = (
            ctx.ref_audio["rows"] if ctx.ref_audio is not None else None
        )
    if ctx.keyframe is not None:
        raw_indices = ctx.keyframe.get("semantic_frame_indices")
        ctx.keyframe_frame_indices = [int(v) for v in raw_indices]
        ctx.keyframe_frame_count = int(ctx.keyframe["frame_count"])


def _build_packed_layout(
    ctx: _FullLoopContext,
    emb: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build the per-task packed layout for the positive branch."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    if ctx.is_ref2va:
        if ctx.ref2va_positive_blocks is None:
            raise ValueError("ref2va ordered reference blocks missing")
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=int(emb["text_len"]),
            latent_t=ctx.latent_t,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
            audio_t=ctx.audio_t,
            ref_blocks=ctx.ref2va_positive_blocks,
        )
    else:
        packed = minimax_h3_packed_sequence(
            text_len=int(emb["text_len"]),
            latent_t=ctx.latent_t,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
            audio_t=ctx.audio_t,
            include_keyframe_cond=ctx.include_cond,
            keyframe_frame_indices=(
                ctx.keyframe_frame_indices if ctx.include_cond else None
            ),
            frame_count=ctx.keyframe_frame_count,
        )
    return packed


def _condition_audio_lengths(ctx: _FullLoopContext) -> list[int]:
    """Per-task condition audio T list for the noise-aug recipe."""

    condition_audio_t: list[int] = []
    if ctx.ref2va_positive_blocks is not None:
        for block in ctx.ref2va_positive_blocks:
            if str(block["kind"]) in {"audio", "video", "video_audio"}:
                ref_audio_t = int(block["ref_audio_t"])
                if ref_audio_t > 0:
                    condition_audio_t.append(ref_audio_t)
    elif isinstance(ctx.ref_audio, Mapping):
        entries = ctx.ref_audio.get("audios")
        if isinstance(entries, list):
            condition_audio_t.extend(
                int(entry["ref_audio_t"])
                for entry in entries
                if int(entry["ref_audio_t"]) > 0
            )
        elif ctx.ref_audio.get("ref_audio_t") is not None:
            ref_audio_t = int(ctx.ref_audio["ref_audio_t"])
            if ref_audio_t > 0:
                condition_audio_t.append(ref_audio_t)
    return condition_audio_t


def _apply_condition_noise_aug(
    ctx: _FullLoopContext,
    *,
    sampling: Any,
    imgvid_noise_aug: float,
    audio_noise_aug: float,
) -> None:
    """Apply condition noise augmentation to cond rows."""

    noise_visual_conditions = (
        ctx.cond_rows is not None and float(imgvid_noise_aug) < 1.0
    )
    noise_audio_conditions = (
        ctx.audio_ref_rows is not None and float(audio_noise_aug) < 1.0
    )
    if not (noise_visual_conditions or noise_audio_conditions):
        return

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )

    noise_seed = getattr(ctx.plan, "seed", None) if ctx.plan is not None else None
    if noise_seed is None:
        noise_seed = getattr(sampling, "seed", None)
    if noise_seed is None:
        noise_seed = 42

    if noise_visual_conditions:
        condition_shapes = _imgvid_condition_shapes(
            ref2va_blocks=ctx.ref2va_positive_blocks,
            keyframe=ctx.keyframe,
            is_ref2va=ctx.is_ref2va,
        )
        imgvid_cond_num_frames = len(condition_shapes)
        if not condition_shapes:
            raise ValueError("imgvid condition rows are missing shape metadata")
        # Imgvid conditions (ref2va blocks / keyframes) contribute one frame
        # count per condition entry.
        ctx.cond_rows = minimax_h3_imgvid_cond_noise_aug_rows(
            ctx.cond_rows,
            condition_shapes=condition_shapes,
            target_latent_t=ctx.latent_t,
            imgvid_cond_num_frames=imgvid_cond_num_frames,
            seed=int(noise_seed),
            noise_aug=float(imgvid_noise_aug),
        )

    if noise_audio_conditions:
        condition_audio_t = _condition_audio_lengths(ctx)
        if not condition_audio_t:
            raise ValueError("audio condition rows are missing length metadata")
        ctx.audio_ref_rows = minimax_h3_audio_cond_noise_aug_rows(
            ctx.audio_ref_rows,
            condition_audio_t=condition_audio_t,
            seed=int(noise_seed),
            noise_aug=float(audio_noise_aug),
        )


def _expand_initial_rows(
    ctx: _FullLoopContext,
    positive: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter target-row noise into the full packed layout when cond rows exist."""

    initial_video = ctx.state["initial_video_rows"]
    if ctx.include_cond:
        # layout target rows = noise; cond anchors appended by the loop
        n_video = positive.img_pos.shape[0]
        full = torch.zeros(int(n_video), initial_video.shape[1], dtype=torch.float32)
        full[positive.update_mask] = initial_video
        initial_video = full

    initial_audio = ctx.state["initial_audio_rows"]
    if ctx.audio_ref_rows is not None:
        n_audio = positive.audio_pos.shape[0]
        full_audio = torch.zeros(
            int(n_audio), initial_audio.shape[1], dtype=torch.float32
        )
        full_audio[positive.audio_update_mask] = initial_audio
        initial_audio = full_audio
    return initial_video, initial_audio


def _publish_full_loop_outputs(
    ctx: _FullLoopContext,
    *,
    batch: Req,
    positive: Any,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
) -> None:
    """Compose the generated target latents onto the batch."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_unpack_audio_tokens,
        minimax_h3_unpatchify_video_tokens,
    )

    target_rows = video_rows[positive.update_mask_dev]
    # keep latents on the CUDA device: DecodingStage enables its VAE
    # autocast only for cuda inputs (fp16/bf16 VAE weights).
    batch.latents = minimax_h3_unpatchify_video_tokens(
        target_rows,
        latent_shape=[ctx.latent_t, ctx.latent_h // 2, ctx.latent_w // 2, 24],
        patch_size=[1, 2, 2],
    )
    audio_target_rows = audio_rows[positive.audio_update_mask_dev]
    batch.audio_latents = minimax_h3_unpack_audio_tokens(
        audio_target_rows, audio_t=ctx.audio_t * 2, audio_channel=2
    )


__all__ = [
    "MiniMaxH3DenoisingStage",
    "minimax_h3_condition_noise_aug",
]
