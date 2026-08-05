# SPDX-License-Identifier: Apache-2.0
"""Generic lowering and delivery hooks for the asynchronous video API."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_SECONDS,
    build_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import expand_request_outputs

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


COMMON_MULTIPART_FORM_FIELDS = (
    "use_duration_template",
    "use_resolution_template",
    "use_system_prompt",
    "use_guardrails",
    "guardrails",
    "generate_sound",
    "sound_duration",
    "action_mode",
    "condition_frame_indexes_vision",
    "condition_video_keep",
)


def _extra_value(request: VideoGenerationsRequest, name: str) -> Any:
    return (request.model_extra or {}).get(name)


def _parse_extra_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        return value


def _format_video_seconds(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return str(int(rounded))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


class BaseVideoModelAdapter:
    """Default no-op hooks for models using the generic video API contract."""

    model_specific_fields: frozenset[str] = frozenset()
    strict_file_delivery = False

    def _common_sampling_kwargs(
        self,
        request_id: str,
        request: VideoGenerationsRequest,
    ) -> dict[str, Any]:
        seconds = (
            request.seconds if request.seconds is not None else DEFAULT_VIDEO_SECONDS
        )
        fps = request.fps if request.fps is not None else DEFAULT_FPS
        num_frames = (
            request.num_frames if request.num_frames is not None else fps * seconds
        )
        num_outputs = request.num_outputs_per_prompt
        if num_outputs is None:
            num_outputs = request.n or 1
        num_outputs = int(num_outputs)
        if not 1 <= num_outputs <= 10:
            raise ValueError("num_outputs_per_prompt must be between 1 and 10")

        return {
            "prompt": request.prompt,
            "num_outputs_per_prompt": num_outputs,
            "size": request.size,
            "width": request.width,
            "height": request.height,
            "num_frames": num_frames,
            "fps": fps,
            "image_path": request.input_reference,
            "output_file_name": request_id,
            "seed": request.seed,
            "generator_device": request.generator_device,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "guidance_scale_2": request.guidance_scale_2,
            "negative_prompt": request.negative_prompt,
            "max_sequence_length": request.max_sequence_length,
            "flow_shift": request.flow_shift,
            "use_duration_template": _extra_value(request, "use_duration_template"),
            "use_resolution_template": _extra_value(request, "use_resolution_template"),
            "use_system_prompt": _extra_value(request, "use_system_prompt"),
            "use_guardrails": _extra_value(request, "use_guardrails"),
            "enable_teacache": request.enable_teacache,
            "enable_frame_interpolation": request.enable_frame_interpolation,
            "frame_interpolation_exp": request.frame_interpolation_exp,
            "frame_interpolation_scale": request.frame_interpolation_scale,
            "frame_interpolation_model_path": request.frame_interpolation_model_path,
            "enable_upscaling": request.enable_upscaling,
            "upscaling_model_path": request.upscaling_model_path,
            "upscaling_scale": request.upscaling_scale,
            "output_path": request.output_path,
            "output_compression": request.output_compression,
            "output_quality": request.output_quality,
            "perf_dump_path": request.perf_dump_path,
            "diffusers_kwargs": request.diffusers_kwargs,
        }

    def lower_sampling_params(
        self,
        request_id: str,
        request: VideoGenerationsRequest,
    ) -> SamplingParams:
        """Lower a transport request to the deployment SamplingParams type."""

        return build_sampling_params(
            request_id,
            **self._common_sampling_kwargs(request_id, request),
        )

    def validate_transport_options(
        self,
        request: VideoGenerationsRequest,
        *,
        model_path: str | None,
    ) -> None:
        """Validate non-security transport options before a job is queued."""

        if "cosmos3" not in (model_path or "").lower():
            return
        extra = request.model_extra or {}
        if extra.get("generate_sound"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cosmos3 video-with-sound is not supported by SGLang yet; "
                    "omit generate_sound for video-only generation."
                ),
            )
        if extra.get("action_mode"):
            raise HTTPException(
                status_code=400,
                detail="Cosmos3 action generation is not supported by SGLang yet.",
            )
        if extra.get("condition_frame_indexes_vision") or extra.get(
            "condition_video_keep"
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cosmos3 video-to-video conditioning is not supported by "
                    "SGLang yet."
                ),
            )

    def validate_task_gate(self, task: Any, *, provided: bool) -> None:
        """Validate a raw model task before request-owned resources exist."""

        return None

    def validate_sampling_params(self, sampling_params: SamplingParams) -> None:
        """Validate offline parameters before request resources exist."""

        return None

    def prepare_for_queue_sync(self, batch: Req) -> None:
        """Synchronous admission hook used by in-process/offline callers."""

        return None

    async def prepare_for_queue(self, batch: Req) -> None:
        """Resolve model-owned admission facts before publishing a job."""

        hook = self.prepare_for_queue_sync
        if (
            getattr(hook, "__func__", hook)
            is BaseVideoModelAdapter.prepare_for_queue_sync
        ):
            return None
        await asyncio.to_thread(hook, batch)

    def cleanup_request_sync(self, batch: Req) -> None:
        """Synchronous request-resource cleanup hook."""

        return None

    async def cleanup_request(self, batch: Req) -> None:
        """Release adapter-owned request resources after rejection/dispatch."""

        hook = self.cleanup_request_sync
        if (
            getattr(hook, "__func__", hook)
            is BaseVideoModelAdapter.cleanup_request_sync
        ):
            return None
        await asyncio.to_thread(hook, batch)

    def expand_for_dispatch(
        self,
        batch: Req,
        *,
        num_prompts: int = 1,
        prompt_index: int = 0,
    ) -> Req | list[Req]:
        """HTTP dispatch hook; generic adapters retain native ``n`` handling."""

        del num_prompts, prompt_index
        return batch

    def expand_for_offline_dispatch(
        self,
        batch: Req,
        *,
        num_prompts: int = 1,
        prompt_index: int = 0,
    ) -> Req | list[Req]:
        """Preserve DiffGenerator's historical generic output expansion."""

        dispatch_batch = self.expand_for_dispatch(
            batch,
            num_prompts=num_prompts,
            prompt_index=prompt_index,
        )
        if dispatch_batch is not batch:
            return dispatch_batch
        expanded = expand_request_outputs(
            batch,
            num_prompts=num_prompts,
            prompt_index=prompt_index,
        )
        return expanded if len(expanded) > 1 else expanded[0]

    def project_queued_job_fields(self, batch: Req) -> dict[str, str]:
        """Project request-derived metadata onto the queued job record."""

        return {}

    def validate_final_outputs_sync(
        self,
        output_paths: list[str],
        batch: Req,
    ) -> dict[str, str]:
        del output_paths, batch
        return {}

    async def validate_final_outputs(
        self,
        output_paths: list[str],
        batch: Req,
    ) -> dict[str, str]:
        hook = self.validate_final_outputs_sync
        if (
            getattr(hook, "__func__", hook)
            is BaseVideoModelAdapter.validate_final_outputs_sync
        ):
            return {}
        return await asyncio.to_thread(
            hook,
            output_paths,
            batch,
        )


_VIDEO_ADAPTER_REGISTRY: dict[type, type[BaseVideoModelAdapter]] = {}


def register_video_model_adapter(
    pipeline_config_cls: type,
    adapter_cls: type[BaseVideoModelAdapter],
) -> None:
    _VIDEO_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls


def _load_model_adapter_for_config(pipeline_config: PipelineConfig) -> None:
    config_type = type(pipeline_config)
    if (
        config_type.__name__ == "MiniMaxH3PipelineConfig"
        and config_type.__module__.endswith("pipeline_configs.minimax_h3")
    ):
        importlib.import_module(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.minimax_h3.video_adapter"
        )


def get_video_model_adapter(
    pipeline_config: PipelineConfig,
) -> BaseVideoModelAdapter:
    """Resolve the nearest registered adapter through pipeline-config MRO."""

    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _VIDEO_ADAPTER_REGISTRY.get(config_cls)
        if adapter_cls is not None:
            return adapter_cls()
    _load_model_adapter_for_config(pipeline_config)
    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _VIDEO_ADAPTER_REGISTRY.get(config_cls)
        if adapter_cls is not None:
            return adapter_cls()
    return BaseVideoModelAdapter()


def known_video_model_fields() -> frozenset[str]:
    fields: set[str] = set()
    for adapter_cls in _VIDEO_ADAPTER_REGISTRY.values():
        fields.update(adapter_cls.model_specific_fields)
    return frozenset(fields)


def validate_adapter_field_claims(
    request: VideoGenerationsRequest,
    adapter: BaseVideoModelAdapter,
) -> None:
    """Reject fields recognized by another adapter instead of ignoring them."""

    extras = request.model_extra or {}
    known_fields = known_video_model_fields()
    provided = set(extras).intersection(known_fields)
    provided.update(request.model_fields_set.intersection(known_fields))
    unsupported = provided.difference(adapter.model_specific_fields)
    if unsupported:
        raise ValueError(
            "unsupported model-specific video request field(s) for "
            f"{type(adapter).__name__}: " + ", ".join(sorted(unsupported))
        )


__all__ = [
    "BaseVideoModelAdapter",
    "COMMON_MULTIPART_FORM_FIELDS",
    "get_video_model_adapter",
    "known_video_model_fields",
    "register_video_model_adapter",
    "validate_adapter_field_claims",
]
