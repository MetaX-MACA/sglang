# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_adapter import (
    BaseVideoModelAdapter,
    get_video_model_adapter,
    register_video_model_adapter,
    validate_adapter_field_claims,
)


class _CustomPipelineConfig(PipelineConfig):
    pass


class _DerivedCustomPipelineConfig(_CustomPipelineConfig):
    pass


class _CustomVideoAdapter(BaseVideoModelAdapter):
    model_specific_fields = frozenset({"custom_mode"})


def test_video_adapter_registry_uses_mro_and_has_generic_default() -> None:
    register_video_model_adapter(_CustomPipelineConfig, _CustomVideoAdapter)

    assert type(get_video_model_adapter(PipelineConfig())) is BaseVideoModelAdapter
    assert isinstance(
        get_video_model_adapter(_CustomPipelineConfig()), _CustomVideoAdapter
    )
    assert isinstance(
        get_video_model_adapter(_DerivedCustomPipelineConfig()), _CustomVideoAdapter
    )


def test_json_known_model_fields_require_an_adapter_claim() -> None:
    register_video_model_adapter(_CustomPipelineConfig, _CustomVideoAdapter)
    request = VideoGenerationsRequest(prompt="p", custom_mode="fast")

    with pytest.raises(ValueError, match="unsupported model-specific"):
        validate_adapter_field_claims(request, BaseVideoModelAdapter())

    validate_adapter_field_claims(request, _CustomVideoAdapter())


def test_raw_multipart_fields_are_collected_then_claim_checked() -> None:
    register_video_model_adapter(_CustomPipelineConfig, _CustomVideoAdapter)
    extra_from_form: dict = {}
    video_api._merge_multipart_extra_form_fields(
        {"custom_mode": "fast"},
        extra_from_form,
    )
    request = VideoGenerationsRequest(prompt="p", **extra_from_form)

    assert extra_from_form == {"custom_mode": "fast"}
    with pytest.raises(ValueError, match="custom_mode"):
        validate_adapter_field_claims(request, BaseVideoModelAdapter())
    validate_adapter_field_claims(request, _CustomVideoAdapter())


def test_multipart_declared_generic_fields_are_preserved_and_unknown_dropped() -> None:
    extras: dict = {}
    video_api._merge_multipart_extra_form_fields(
        {
            "width": "640",
            "height": "360",
            "guidance_scale": "2.5",
            "negative_prompt": "avoid blur",
            "not_a_declared_video_field": "drop me",
        },
        extras,
    )

    assert extras == {
        "width": 640,
        "height": 360,
        "guidance_scale": 2.5,
        "negative_prompt": "avoid blur",
    }


def test_base_offline_expansion_normalizes_single_output() -> None:
    request = SimpleNamespace(
        num_outputs_per_prompt=1,
        seed=7,
        seeds=None,
        generator=SimpleNamespace(),
        request_id="request",
        output_file_name="output.mp4",
        sampling_params=SimpleNamespace(
            refresh_request_extra_after_output_expansion=lambda _request: None
        ),
        validate=lambda: None,
    )

    expanded = BaseVideoModelAdapter().expand_for_offline_dispatch(request)

    assert expanded.seed == 7
    assert expanded.generator is None
