# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
    _required_tensor,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_enabled,
    resolve_precision,
)


def _reverse_normalize_latents(
    latents: torch.Tensor,
    *,
    mean_values,
    std_values,
    name: str,
) -> torch.Tensor:
    mean = torch.as_tensor(mean_values, device=latents.device, dtype=latents.dtype)
    std = torch.as_tensor(std_values, device=latents.device, dtype=latents.dtype)
    if mean.ndim != 1:
        raise ValueError(f"{name}.latents_mean must be 1-D, got {tuple(mean.shape)}")
    if std.ndim != 1:
        raise ValueError(f"{name}.latents_std must be 1-D, got {tuple(std.shape)}")
    if mean.shape != std.shape:
        raise ValueError(
            f"{name} latent normalization shape mismatch: "
            f"mean={tuple(mean.shape)} std={tuple(std.shape)}"
        )
    if latents.ndim < 2:
        raise ValueError(f"{name} latents must have a channel dimension")
    if int(latents.shape[1]) != int(mean.shape[0]):
        raise ValueError(
            f"{name} latent normalization channel mismatch: "
            f"latents.shape[1]={int(latents.shape[1])} mean_len={int(mean.shape[0])}"
        )
    view_shape = [1] * latents.ndim
    view_shape[1] = int(mean.shape[0])
    return latents * std.view(*view_shape) + mean.view(*view_shape)


def _crop_to_target_canvas(batch: Req, frames: torch.Tensor) -> torch.Tensor:
    """Crop decoded frames [B,C,T,H,W] back to the target canvas.

    The visual VAE pads the latent grid to its tile multiples (padding lands
    at the bottom/right), so a non-tile-aligned geometry decodes larger than the
    requested canvas (e.g. 1344x768 for a 1280x704 target). Target dims come
    from the direct-mode denoise state (latent_h/w * 16); requests without
    that state keep the raw decode.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )

    state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY)
    if state is None:
        return frames
    target_h = int(state["latent_h"]) * 16
    target_w = int(state["latent_w"]) * 16
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h < target_h or w < target_w:
        raise ValueError(
            f"decoded frames {h}x{w} smaller than target canvas {target_h}x{target_w}"
        )
    if h == target_h and w == target_w:
        return frames
    return frames[..., :target_h, :target_w].contiguous()


def _canonical_visual_video_frames(
    frames: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    if frames.ndim == 4:
        if int(frames.shape[0]) % batch_size != 0:
            raise ValueError(
                f"Decoded visual video shape {tuple(frames.shape)} is incompatible "
                f"with batch_size={batch_size}"
            )
        frames = frames.reshape(
            batch_size, int(frames.shape[0]) // batch_size, *frames.shape[1:]
        )
        frames = frames.transpose(1, 2).contiguous()
    elif frames.ndim == 5:
        if int(frames.shape[0]) != batch_size:
            raise ValueError(
                f"Decoded visual video batch mismatch: frames.shape[0]={int(frames.shape[0])} "
                f"batch_size={batch_size}"
            )
        frames = frames.contiguous()
    else:
        raise ValueError(
            f"Decoded visual video shape {tuple(frames.shape)} is not supported"
        )
    return frames.to(torch.float32)


def _canonical_output_audio_waveform(
    audio_waveform: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    """Project audio-VAE-native ``[C, 1, L]`` audio to output ``[1, C, L]``.

    The audio VAE treats stereo channels as its decoder batch and returns
    ``[2, 1, samples]`` for MiniMax H3's one generated sample.  The generic output
    path instead selects generated samples along dimension zero.  Keep the audio VAE
    tensor unchanged for decoder artifacts, then make the singleton generated-
    sample dimension explicit only at the ``OutputBatch`` boundary.
    """
    if audio_waveform.ndim != 3:
        raise ValueError(
            "Decoded audio VAE waveform must be [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    if batch_size != 1:
        raise ValueError(
            "MiniMax H3 audio VAE output only supports one generated sample, "
            f"got visual batch_size={batch_size}"
        )
    if int(audio_waveform.shape[1]) != 1:
        raise ValueError(
            "Decoded audio VAE waveform must have shape [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    return audio_waveform.permute(1, 0, 2).contiguous()


_MINIMAX_H3_DECODER_TASKS = frozenset({"t2va", "fl2va", "ref2va"})
_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY = "minimax_h3_canonical_request"
_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY = "minimax_h3_resolved_plan"


def _minimax_h3_decoder_task(batch: Req) -> str | None:
    """Return the validated request task used for output-decoder routing.

    Debug requests have no canonical task and retain the
    generic decoder.
    """

    extra = getattr(batch, "extra", None)
    if not isinstance(extra, Mapping):
        return None
    canonical = extra.get(_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY)
    if canonical is not None and not isinstance(canonical, Mapping):
        raise ValueError("minimax_h3_canonical_request must be a mapping")
    canonical_task = canonical.get("task") if isinstance(canonical, Mapping) else None
    resolved = extra.get(_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY)
    resolved_task = getattr(resolved, "task", None) if resolved is not None else None
    if canonical_task is not None and resolved_task is not None:
        if str(canonical_task) != str(resolved_task):
            raise ValueError(
                "MiniMax H3 decoder task mismatch between canonical request and "
                "resolved plan"
            )
    task_value = resolved_task if resolved_task is not None else canonical_task
    if task_value is None:
        return None
    if not isinstance(task_value, str) or task_value not in _MINIMAX_H3_DECODER_TASKS:
        raise ValueError(f"unsupported MiniMax H3 decoder task {task_value!r}")
    return task_value


class MiniMaxH3DecodingStage(PipelineStage):
    def __init__(self, video_vae, audio_vae) -> None:
        super().__init__()
        self.video_vae = video_vae
        self.audio_vae = audio_vae

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        video_vae_dtype = resolve_precision(
            server_args, "video_vae", precision_attr="vae_precision"
        )
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        uses = [
            ComponentUse(stage_name, "video_vae", target_dtype=video_vae_dtype),
        ]
        uses.append(ComponentUse(stage_name, "audio_vae", target_dtype=audio_vae_dtype))
        return uses

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        _minimax_h3_decoder_task(batch)
        visual_latent = _required_tensor(batch.latents, "batch.latents")
        audio_latent = _required_tensor(batch.audio_latents, "batch.audio_latents")
        if visual_latent.ndim != 5:
            raise ValueError("batch.latents must be [B, C, T, H, W]")
        if audio_latent.ndim != 3:
            raise ValueError(
                "batch.audio_latents must be [audio_channel, latent_dim, T]"
            )

        if self.video_vae is None:
            raise RuntimeError("MiniMax H3 tasks require the video_vae output decoder")
        with self.use_declared_component(
            component_name="video_vae",
            module=self.video_vae,
        ) as selected_video_vae:
            if selected_video_vae is None:
                raise RuntimeError("video_vae became unavailable during decode")
            self.video_vae = selected_video_vae
            selected_video_vae.eval()
            visual_arch_config = server_args.pipeline_config.vae_config.arch_config
            visual_decode_latent = _reverse_normalize_latents(
                visual_latent,
                mean_values=visual_arch_config.latents_mean,
                std_values=visual_arch_config.latents_std,
                name="video_vae",
            )
            video_vae_dtype = resolve_precision(
                server_args,
                "video_vae",
                precision_attr="vae_precision",
            )
            if video_vae_dtype == torch.float32:
                # Decode compute runs in fp16 by default. Weights stay
                # fp32 for the keyframe-encode recipe; autocast casts per-op.
                video_vae_dtype = torch.float16
            visual_autocast_enabled = (
                visual_latent.device.type == "cuda"
                and autocast_enabled(video_vae_dtype, server_args.disable_autocast)
            )
            with torch.autocast(
                device_type=visual_latent.device.type,
                dtype=video_vae_dtype,
                enabled=visual_autocast_enabled,
            ):
                visual_frames = selected_video_vae.decode_base(visual_decode_latent)
                visual_frames = selected_video_vae.model.processor.revert_tensor(
                    visual_frames
                )
                visual_frames = _required_tensor(
                    visual_frames,
                    "video_vae.model.processor.revert_tensor",
                )
                visual_frames = _canonical_visual_video_frames(
                    visual_frames, batch_size=int(visual_latent.shape[0])
                )
                visual_frames = _crop_to_target_canvas(batch, visual_frames)

        with self.use_declared_component(
            component_name="audio_vae",
            module=self.audio_vae,
        ) as audio_vae:
            assert audio_vae is not None
            self.audio_vae = audio_vae
            self.audio_vae.eval()
            audio_arch_config = server_args.pipeline_config.audio_vae_config.arch_config
            audio_decode_latent = _reverse_normalize_latents(
                audio_latent,
                mean_values=audio_arch_config.latents_mean,
                std_values=audio_arch_config.latents_std,
                name="audio_vae",
            )
            audio_vae_dtype = resolve_precision(
                server_args, "audio_vae", precision_attr="audio_vae_precision"
            )
            audio_autocast_enabled = (
                audio_latent.device.type == "cuda"
                and autocast_enabled(audio_vae_dtype, server_args.disable_autocast)
            )
            with torch.autocast(
                device_type=audio_latent.device.type,
                dtype=audio_vae_dtype,
                enabled=audio_autocast_enabled,
            ):
                audio_waveform = _required_tensor(
                    audio_vae.decode(audio_decode_latent), "audio_vae.decode"
                )
            audio_sample_rate = int(audio_vae.sample_rate)

        output_audio_waveform = _canonical_output_audio_waveform(
            audio_waveform, batch_size=int(visual_frames.shape[0])
        )
        return OutputBatch(
            output=visual_frames,
            audio=output_audio_waveform,
            audio_sample_rate=audio_sample_rate,
            metrics=batch.metrics,
        )


__all__ = [
    "MiniMaxH3DecodingStage",
]
