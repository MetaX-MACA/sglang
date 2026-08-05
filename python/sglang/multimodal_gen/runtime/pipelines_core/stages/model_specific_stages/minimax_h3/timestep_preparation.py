# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Mapping

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
    _not_implemented,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3TimestepPreparationStage(PipelineStage):
    def __init__(self, sigma_shift_scales=None) -> None:
        super().__init__()
        # Per-model sigma shift override (model_index.json "_minimax_h3" release
        # block, sigma_shift_scales): the schedule constants are a MODEL
        # serving contract — fl2va and ref2va use video 12 / audio 3 by default.
        self.sigma_shift_scales = sigma_shift_scales

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            _not_implemented(self.__class__.__name__)
        self._generate_sigmas_from_plan(batch, plan)
        return batch

    def _generate_sigmas_from_plan(self, batch: Req, plan) -> None:
        """Generate the fixed per-modality float32 time-shift schedules."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_SIGMAS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_time_shift_sigmas,
        )

        if MINIMAX_H3_SIGMAS_EXTRA_KEY in batch.extra:
            return
        requested_num_steps = getattr(batch, "num_inference_steps", None)
        if requested_num_steps is None:
            sampling = getattr(batch, "sampling_params", None)
            requested_num_steps = getattr(sampling, "num_inference_steps", None)
        if requested_num_steps is None:
            requested_num_steps = 50
        if (
            isinstance(requested_num_steps, bool)
            or not isinstance(requested_num_steps, int)
            or requested_num_steps <= 0
        ):
            raise ValueError(
                "num_inference_steps must be a positive integer, got "
                f"{requested_num_steps!r}"
            )

        model_scales = self.sigma_shift_scales
        if model_scales is not None and not isinstance(model_scales, Mapping):
            raise ValueError("model sigma_shift_scales must be an object")

        def resolved_scale(
            *, modality: str, request_value, task_default: float
        ) -> float:
            value = request_value
            source = (
                "request "
                f"{'flow_shift' if modality == 'video' else 'audio_flow_shift'}"
            )
            if value is None and model_scales is not None:
                value = model_scales.get(modality)
                source = f"model sigma_shift_scales.{modality}"
            if value is None:
                value = task_default
                source = (
                    "task default "
                    f"{'flow_shift' if modality == 'video' else 'audio_flow_shift'}"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{source} must be a positive finite number")
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{source} must be a positive finite number")
            return scale

        scales = {
            "video": resolved_scale(
                modality="video",
                request_value=plan.flow_shift,
                task_default=plan.default_flow_shift,
            ),
            "audio": resolved_scale(
                modality="audio",
                request_value=plan.audio_flow_shift,
                task_default=plan.default_audio_flow_shift,
            ),
        }
        sigmas: dict[str, list[float]] = {}
        for modality in ("video", "audio"):
            sigmas[modality] = minimax_h3_time_shift_sigmas(
                num_steps=requested_num_steps,
                shift_scale=scales[modality],
            )
        batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = sigmas


__all__ = ["MiniMaxH3TimestepPreparationStage"]
