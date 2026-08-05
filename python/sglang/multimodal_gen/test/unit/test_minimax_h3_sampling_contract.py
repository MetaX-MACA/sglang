from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_adapter import (
    validate_adapter_field_claims,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    video_adapter as minimax_h3_video_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    MiniMaxH3ShapePlanner,
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
    MiniMaxH3VideoModelAdapter,
)

_MINIMAX_H3_VIDEO_ADAPTER = MiniMaxH3VideoModelAdapter()


def _target() -> dict:
    return {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 8.7,
    }


def _minimax_h3_factory_params(
    monkeypatch: pytest.MonkeyPatch, **kwargs
) -> MiniMaxH3SamplingParams:
    """Exercise the same generic SamplingParams boundary as DiffGenerator."""

    monkeypatch.setattr(
        SamplingParams,
        "from_pretrained",
        lambda *_args, **_kwargs: MiniMaxH3SamplingParams(),
    )
    server_args = SimpleNamespace(
        pipeline_class_name=None,
        backend="sglang",
        model_id=None,
        pipeline_config=MiniMaxH3PipelineConfig(),
        output_path="/tmp",
        num_gpus=1,
        comfyui_mode=False,
    )
    return SamplingParams.from_user_sampling_params_args(
        "minimax_h3-test-model",
        server_args=server_args,
        **kwargs,
    )


def test_minimax_h3_canonical_request_accepts_independent_flow_shift_overrides() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="a calm lake",
        conditions=[],
        target=_target(),
        flow_shift=12,
        audio_flow_shift=3,
        seed=123,
    )

    assert canonical["flow_shift"] == 12.0
    assert canonical["audio_flow_shift"] == 3.0
    assert canonical["seed"] == 123


def test_minimax_h3_audio_shape_rounds_at_40hz_boundary() -> None:
    # 124 frames at 24 fps is 206.666... audio tokens; the canonical rule uses
    # round, not floor, so the shape must be 207.
    assert MiniMaxH3ShapePlanner().audio_latent_t(124 / 24) == 207


@pytest.mark.parametrize(
    ("overrides", "expected_video_shift", "expected_audio_shift"),
    [
        ({}, 12.0, 3.0),
        ({"flow_shift": 8.5}, 8.5, 3.0),
        ({"audio_flow_shift": 2.5}, 12.0, 2.5),
    ],
)
def test_minimax_h3_resolved_plan_keeps_request_flow_shift_overrides(
    overrides: dict[str, float],
    expected_video_shift: float,
    expected_audio_shift: float,
) -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="a calm lake",
        conditions=[],
        target=_target(),
        **overrides,
    )

    plan = minimax_h3_resolve_plan(canonical)

    assert plan.default_flow_shift == 12.0
    assert plan.default_audio_flow_shift == 3.0
    assert plan.flow_shift == overrides.get("flow_shift")
    assert plan.audio_flow_shift == overrides.get("audio_flow_shift")
    # The effective value is request > task default; model config is applied
    # by the timestep stage and is tested below.
    assert (
        plan.flow_shift if plan.flow_shift is not None else plan.default_flow_shift
    ) == expected_video_shift
    assert (
        plan.audio_flow_shift if plan.audio_flow_shift is not None else plan.default_audio_flow_shift
    ) == expected_audio_shift


@pytest.mark.parametrize(
    (
        "request_flow_shift",
        "request_audio_flow_shift",
        "model_scales",
        "expected_video_shift",
        "expected_audio_shift",
    ),
    [
        (None, None, None, 12.0, 3.0),
        (None, None, {"video": 9.0, "audio": 4.0}, 9.0, 4.0),
        (8.0, None, {"video": 9.0, "audio": 4.0}, 8.0, 4.0),
        (None, 2.0, {"video": 9.0, "audio": 4.0}, 9.0, 2.0),
        (8.0, 2.0, {"video": 9.0, "audio": 4.0}, 8.0, 2.0),
    ],
)
def test_minimax_h3_timestep_stage_applies_flow_shift_precedence(
    request_flow_shift,
    request_audio_flow_shift,
    model_scales,
    expected_video_shift,
    expected_audio_shift,
) -> None:
    stage = MiniMaxH3TimestepPreparationStage(sigma_shift_scales=model_scales)
    batch = SimpleNamespace(extra={}, num_inference_steps=3)
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="a calm lake",
        conditions=[],
        target=_target(),
        flow_shift=request_flow_shift,
        audio_flow_shift=request_audio_flow_shift,
    )
    stage._generate_sigmas_from_plan(batch, minimax_h3_resolve_plan(canonical))

    assert batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] == {
        "video": minimax_h3_time_shift_sigmas(
            num_steps=3, shift_scale=expected_video_shift
        ),
        "audio": minimax_h3_time_shift_sigmas(
            num_steps=3, shift_scale=expected_audio_shift
        ),
    }


@pytest.mark.parametrize("num_steps", [1, 2, 7, 50])
def test_minimax_h3_timestep_stage_uses_only_num_inference_steps(num_steps: int) -> None:
    stage = MiniMaxH3TimestepPreparationStage()
    batch = SimpleNamespace(extra={}, num_inference_steps=num_steps)
    canonical = minimax_h3_validate_canonical_request(
        task="t2va", prompt="a calm lake", conditions=[], target=_target()
    )
    stage._generate_sigmas_from_plan(batch, minimax_h3_resolve_plan(canonical))

    assert len(batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]["video"]) == num_steps
    assert len(batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]["audio"]) == num_steps


def test_minimax_h3_timestep_stage_defaults_to_50_sigma_points() -> None:
    stage = MiniMaxH3TimestepPreparationStage()
    batch = SimpleNamespace(extra={}, num_inference_steps=None)
    canonical = minimax_h3_validate_canonical_request(
        task="t2va", prompt="a calm lake", conditions=[], target=_target()
    )
    stage._generate_sigmas_from_plan(batch, minimax_h3_resolve_plan(canonical))
    assert len(batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]["video"]) == 50
    assert len(batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]["audio"]) == 50


@pytest.mark.parametrize("field_name", ["flow_shift", "audio_flow_shift"])
@pytest.mark.parametrize("value", [True, "3", 0, -1, float("nan"), float("inf")])
def test_minimax_h3_sampling_params_rejects_invalid_flow_shifts(
    field_name, value
) -> None:
    with pytest.raises((ValueError, TypeError)):
        MiniMaxH3SamplingParams(
            prompt="a calm lake",
            task="t2va",
            conditions=[],
            target=_target(),
            **{field_name: value},
        ).build_request_extra()


@pytest.mark.parametrize("field_name", ["time_spec", "add_keyframe_instruction"])
def test_minimax_h3_sampling_factory_rejects_unknown_time_and_instruction_fields(
    field_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        _minimax_h3_factory_params(monkeypatch, prompt="p", **{field_name: {}})


@pytest.mark.parametrize("seed", [0, (1 << 63) - 1])
def test_minimax_h3_sampling_params_accepts_non_negative_signed_int64_range(
    seed: int,
) -> None:
    params = MiniMaxH3SamplingParams(
        prompt="a calm lake",
        task="t2va",
        conditions=[],
        target=_target(),
        seed=seed,
    )

    extra = params.build_request_extra()

    assert params.seed == seed
    assert extra["minimax_h3_canonical_request"]["seed"] == seed


@pytest.mark.parametrize("seed", [-1, [7, -1]])
def test_minimax_h3_sampling_params_rejects_negative_seed(
    seed: int | list[int],
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MiniMaxH3SamplingParams(seed=seed)


def test_minimax_h3_sampling_params_rejects_seed_above_signed_int64() -> None:
    with pytest.raises(ValueError, match="signed int64"):
        MiniMaxH3SamplingParams(seed=1 << 63)


def test_minimax_h3_sampling_params_rejects_output_seed_overflow() -> None:
    with pytest.raises(ValueError, match="seed plus output index"):
        MiniMaxH3SamplingParams(seed=(1 << 63) - 1, num_outputs_per_prompt=2)


@pytest.mark.parametrize(
    "field_name", ["enable_frame_interpolation", "enable_upscaling"]
)
def test_minimax_h3_sampling_params_rejects_unaccepted_postprocessing(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=f"does not support {field_name}"):
        MiniMaxH3SamplingParams(**{field_name: True})


@pytest.mark.parametrize(
    "field_name",
    [
        "cfg_distilled",
        "initial_latent_rows_path",
        "qwen_hidden_inject_paths",
        "qwen3vl_encoder_output",
        "qwen3vl_runtime_config",
        "tokenizer_artifacts",
        "video_tokenizer_tensors",
        "audio_tokenizer_tensors",
        "latent_preparation_input",
        "timestep_preparation_input",
        "denoising_tensor_input",
        "denoising_artifact_input",
    ],
)
def test_unknown_request_fields_are_rejected_by_python_sampling_factory(
    field_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        _minimax_h3_factory_params(monkeypatch, prompt="p", **{field_name: {}})


@pytest.mark.parametrize("transport", ["json", "multipart"])
def test_video_api_rejects_negative_seed_at_sampling_boundary(
    transport: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_build_sampling_params(request_id: str, **kwargs):
        captured.update(kwargs)
        return MiniMaxH3SamplingParams(
            request_id=request_id,
            prompt=kwargs["prompt"],
            task=kwargs["task"],
            conditions=kwargs["conditions"],
            target=kwargs["target"],
            seed=kwargs["seed"],
        )

    monkeypatch.setattr(
        minimax_h3_video_adapter, "build_sampling_params", fake_build_sampling_params
    )
    payload = {
        "prompt": "a calm lake",
        "task": "t2va",
        "conditions": [],
        "target": _target(),
        "seed": -7,
    }
    if transport == "json":
        request = VideoGenerationsRequest(**payload)
    else:
        # FastAPI's ``Form(int)`` parser has already converted the multipart
        # text field before create_video constructs this protocol model.
        request = VideoGenerationsRequest(**{**payload, "seed": int("-7")})

    with pytest.raises(ValueError, match="non-negative"):
        _MINIMAX_H3_VIDEO_ADAPTER.lower_sampling_params(
            f"req-negative-seed-{transport}", request
        )

    assert captured["seed"] == -7


def test_rollout_generate_maps_negative_seed_validation_to_http_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_sampling_params(_request_id: str, **kwargs):
        return MiniMaxH3SamplingParams(**kwargs)

    monkeypatch.setattr(
        rollout_api,
        "build_sampling_params",
        build_sampling_params,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            rollout_api.rollout_generate(RolloutRequest(prompt="a calm lake", seed=-1))
        )

    assert error.value.status_code == 400
    assert "non-negative" in error.value.detail


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("imgvid_cond_noise_aug_for_inference", 1.2),
        ("audio_cond_noise_aug_for_inference", -0.1),
        ("imgvid_cond_noise_aug_for_inference", True),
    ],
)
def test_minimax_h3_sampling_params_rejects_invalid_noise_aug(
    field_name: str,
    value,
) -> None:
    params = MiniMaxH3SamplingParams(**{field_name: value})

    with pytest.raises(ValueError, match=field_name):
        params.build_request_extra()


@pytest.mark.parametrize(("frame_index", "expected_pixel"), [(0, 0), (-1, 208)])
def test_minimax_h3_sampling_params_accepts_single_i2va_or_l2va_keyframe(
    frame_index: int,
    expected_pixel: int,
) -> None:
    params = MiniMaxH3SamplingParams(
        prompt="animate the frame",
        task="fl2va",
        conditions=[
            {
                "type": "image",
                "uri": "file:///anchor.png",
                "role": "keyframe",
                "frame_index": frame_index,
            }
        ],
        target={
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 8.7,
        },
    )

    canonical = params.build_request_extra()["minimax_h3_canonical_request"]
    plan = minimax_h3_resolve_plan(canonical)

    assert [condition["frame_index"] for condition in canonical["conditions"]] == [
        frame_index
    ]
    assert plan.condition_mask["semantic_frame_indices"] == [frame_index]
    assert plan.condition_mask["pixel_frame_indices"] == [expected_pixel]
    assert len(plan.materials) == 1


def test_minimax_h3_ref2va_accepts_audio_only_reference() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="keep the subject",
        conditions=[{"type": "audio", "uri": "file:///voice.wav", "role": "reference"}],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8.7},
    )

    assert [c["type"] for c in canonical["conditions"]] == ["audio"]


def test_minimax_h3_ref2va_accepts_video_without_image_reference() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="keep the subject",
        conditions=[{"type": "video", "uri": "file:///ref.mp4", "role": "reference"}],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8.7},
    )

    assert [c["type"] for c in canonical["conditions"]] == ["video"]


def test_minimax_h3_ref2va_accepts_video_audio_families() -> None:
    rva = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="follow the reference motion and sound",
        conditions=[
            {"type": "video_audio", "uri": "file:///ref.mp4", "role": "reference"}
        ],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8.7},
    )
    assert [c["type"] for c in rva["conditions"]] == ["video_audio"]

    rirva = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="keep the subject and follow the reference motion",
        conditions=[
            {"type": "image", "uri": "file:///subject.png", "role": "reference"},
            {"type": "video_audio", "uri": "file:///ref.mp4", "role": "reference"},
        ],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8.7},
    )
    assert [c["type"] for c in rirva["conditions"]] == ["image", "video_audio"]


def test_minimax_h3_ref2va_accepts_multi_audio_with_explicit_target() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="blend the references",
        conditions=[
            {"type": "image", "uri": "file:///subject.png", "role": "reference"},
            {"type": "audio", "uri": "file:///a.wav", "role": "reference"},
            {"type": "audio", "uri": "file:///b.wav", "role": "reference"},
        ],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8.7},
    )

    assert [c["type"] for c in canonical["conditions"]] == [
        "image",
        "audio",
        "audio",
    ]


def test_minimax_h3_ref2va_accepts_supported_reference_combinations() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="keep the subject",
        conditions=[
            {"type": "image", "uri": "file:///subject-a.png", "role": "reference"},
            {"type": "image", "uri": "file:///subject-b.png", "role": "reference"},
            {"type": "video", "uri": "file:///ref-a.mp4", "role": "reference"},
            {"type": "video", "uri": "file:///ref-b.mp4", "role": "reference"},
            {"type": "video_audio", "uri": "file:///ref-c.mp4", "role": "reference"},
            {"type": "audio", "uri": "file:///voice.wav", "role": "reference"},
        ],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8.7},
    )

    assert [c["type"] for c in canonical["conditions"]] == [
        "image",
        "image",
        "video",
        "video",
        "video_audio",
        "audio",
    ]


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("guidance_scale", 1.0),
        ("guidance_scale_2", 1.0),
        ("true_cfg_scale", 1.0),
        ("negative_prompt", "avoid blur"),
        ("audio_guidance_scale", 1.0),
        ("uncond_drop_conditions", ["text"]),
        ("ref2va_negative_drop_audio", False),
    ],
)
def test_minimax_h3_sampling_factory_rejects_retired_cfg_fields(
    field_name: str,
    value,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        _minimax_h3_factory_params(monkeypatch, prompt="p", **{field_name: value})


def test_offline_factory_projects_canonical_target_and_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _minimax_h3_factory_params(
        monkeypatch,
        prompt="a calm lake",
        task="t2va",
        conditions=[],
        target={**_target(), "unknown_target_value": 99},
    )

    assert params.fps == 24
    assert params.num_frames == 1
    assert "fps" not in params._explicit_fields
    assert "num_frames" not in params._explicit_fields
    assert params.target == _target()

    extra = params.build_request_extra()
    canonical = extra["minimax_h3_canonical_request"]
    assert canonical["target"] == _target()

    batch = Req(sampling_params=params)
    params.apply_request_extra(batch)
    assert batch.fps == 24
    assert batch.num_frames == 1
    assert batch.target == _target()


@pytest.mark.parametrize("field_name", ["fps", "num_frames"])
def test_offline_factory_rejects_internal_temporal_fields(
    field_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        _minimax_h3_factory_params(monkeypatch, prompt="p", **{field_name: 120})


@pytest.mark.parametrize("output_mode", ["decoded_files_with_latents", "latents"])
def test_video_api_rejects_unsupported_minimax_h3_output_modes(output_mode: str) -> None:
    request = VideoGenerationsRequest(
        prompt="a calm lake",
        task="t2va",
        conditions=[],
        target=_target(),
        output_mode=output_mode,
    )

    with pytest.raises(ValueError, match="only supports output_mode='decoded_files'"):
        _MINIMAX_H3_VIDEO_ADAPTER.validate_transport_options(
            request, model_path="minimax_h3"
        )


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("enable_frame_interpolation", "does not support enable_frame_interpolation"),
        ("enable_upscaling", "does not support enable_upscaling"),
    ],
)
def test_video_api_rejects_unaccepted_minimax_h3_postprocessing(
    field_name: str, message: str
) -> None:
    request = VideoGenerationsRequest(
        prompt="a calm lake",
        task="t2va",
        conditions=[],
        target=_target(),
        **{field_name: True},
    )

    with pytest.raises(ValueError, match=message):
        _MINIMAX_H3_VIDEO_ADAPTER.validate_transport_options(
            request, model_path="minimax_h3"
        )


def test_offline_factory_keeps_capability_boundaries_visible_to_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _minimax_h3_factory_params(
        monkeypatch,
        prompt="a calm lake",
        task="t2va",
        conditions=[],
        target=_target(),
        output_mode="latents",
    )
    with pytest.raises(ValueError, match="only supports output_mode='decoded_files'"):
        _MINIMAX_H3_VIDEO_ADAPTER.validate_sampling_params(params)


def test_minimax_h3_unknown_transport_fields_are_ignored():
    for field_name, value in (
        ("time_spec", {}),
        ("initial_latent_rows_path", {}),
        ("arbitrary_legacy_field", {}),
        ("cfg_distilled", True),
        # Undeclared fields fall through to the silently-ignored
        # unknown-field path.
        ("unknown_string_field", "/tmp/unused"),
        ("unknown_mapping_field", {"profile": "strict"}),
    ):
        request = VideoGenerationsRequest(
            prompt="a calm lake",
            task="t2va",
            conditions=[],
            target=_target(),
            **{field_name: value},
        )
        validate_adapter_field_claims(request, _MINIMAX_H3_VIDEO_ADAPTER)


def test_video_api_forwards_minimax_h3_sampling_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_build_sampling_params(request_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(request_id=request_id)

    monkeypatch.setattr(
        minimax_h3_video_adapter, "build_sampling_params", fake_build_sampling_params
    )
    request = VideoGenerationsRequest(
        prompt="animate the frame",
        task="fl2va",
        conditions=[
            {
                "type": "image",
                "uri": "file:///first.png",
                "role": "keyframe",
                "frame_index": 0,
            },
            {
                "type": "image",
                "uri": "file:///last.png",
                "role": "keyframe",
                "frame_index": -1,
            },
        ],
        target={
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 8.7,
        },
        flow_shift=8.0,
        audio_flow_shift=2.0,
        imgvid_cond_noise_aug_for_inference=1.0,
        audio_cond_noise_aug_for_inference=0.75,
    )

    _MINIMAX_H3_VIDEO_ADAPTER.lower_sampling_params("req-2", request)

    assert captured["flow_shift"] == 8.0
    assert captured["audio_flow_shift"] == 2.0
    assert "guidance_scale" not in captured
    assert "guidance_scale_2" not in captured
    assert "audio_guidance_scale" not in captured
    assert "negative_prompt" not in captured
    assert "uncond_drop_conditions" not in captured
    assert captured["imgvid_cond_noise_aug_for_inference"] == 1.0
    assert captured["audio_cond_noise_aug_for_inference"] == 0.75
    assert len(captured["conditions"]) == 2


def test_video_api_collects_minimax_h3_raw_multipart_extra_fields() -> None:
    extra_from_form = {"task": "t2va"}
    raw_form = {
        "task": "t2va",
        "conditions": "[]",
        "target": '{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":8.7}',
        "flow_shift": "8.0",
        "audio_flow_shift": "2.0",
        "imgvid_cond_noise_aug_for_inference": "1.0",
        "audio_cond_noise_aug_for_inference": "0.75",
    }

    video_api._merge_multipart_extra_form_fields(raw_form, extra_from_form)

    assert extra_from_form["task"] == "t2va"
    assert extra_from_form["conditions"] == []
    assert extra_from_form["target"]["duration_seconds"] == 8.7
    # flow_shift is a declared request/Form field: the merge collects it like
    # any declared form field, and the endpoint later strips declared names
    # from the extras splat so it only travels the declared-field path.
    assert extra_from_form["flow_shift"] == 8.0
    assert extra_from_form["audio_flow_shift"] == 2.0
    assert extra_from_form["imgvid_cond_noise_aug_for_inference"] == 1.0
    assert extra_from_form["audio_cond_noise_aug_for_inference"] == 0.75

    declared = video_api._multipart_declared_request_fields(
        raw_form, extra_from_form, {"flow_shift": 8.0}
    )
    extra_request_fields = {
        key: value
        for key, value in extra_from_form.items()
        if key not in VideoGenerationsRequest.model_fields
    }
    request = VideoGenerationsRequest(prompt="p", **declared, **extra_request_fields)
    _MINIMAX_H3_VIDEO_ADAPTER.validate_transport_options(request, model_path="minimax_h3")
    assert request.flow_shift == 8.0
    assert request.audio_flow_shift == 2.0


def test_video_api_collects_minimax_h3_multipart_extra_wrapper() -> None:
    extras = video_api._multipart_video_extras(
        {
            "task": "t2va",
            "conditions": "[]",
            "target": '{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":8.7}',
        },
        extra_body=json.dumps(
            {
                "flow_shift": 7.0,
                "audio_flow_shift": 1.5,
                "arbitrary_compatibility_value": True,
            }
        ),
        extra_params=None,
    )

    assert extras["flow_shift"] == 7.0
    assert extras["audio_flow_shift"] == 1.5
    assert "arbitrary_compatibility_value" not in extras
    request = VideoGenerationsRequest(prompt="p", **extras)
    _MINIMAX_H3_VIDEO_ADAPTER.validate_transport_options(request, model_path="minimax_h3")


@pytest.mark.parametrize(
    ("field_name", "detail"),
    [
        ("extra_body", "extra_body is not valid JSON"),
        ("extra_params", "extra_params is not valid JSON"),
    ],
)
def test_video_api_rejects_invalid_multipart_extra_json(
    field_name: str,
    detail: str,
) -> None:
    kwargs = {"extra_body": None, "extra_params": None}
    kwargs[field_name] = "{"

    with pytest.raises(HTTPException) as exc_info:
        video_api._multipart_video_extras({}, **kwargs)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == detail


@pytest.mark.parametrize("field_name", ["time_spec", "add_keyframe_instruction"])
def test_video_api_drops_unknown_raw_multipart_request_fields(field_name: str) -> None:
    extra_from_form: dict = {}
    video_api._merge_multipart_extra_form_fields(
        {field_name: "{}" if field_name == "time_spec" else "false"},
        extra_from_form,
    )
    assert field_name not in extra_from_form


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("initial_latent_rows_path", "/tmp/removed.safetensors"),
        ("cfg_distilled", "true"),
    ],
)
def test_video_api_drops_removed_raw_multipart_request_field(
    field_name: str,
    value: str,
) -> None:
    extra_from_form: dict = {}
    raw_form = {field_name: value}

    video_api._merge_multipart_extra_form_fields(raw_form, extra_from_form)

    assert field_name not in extra_from_form
