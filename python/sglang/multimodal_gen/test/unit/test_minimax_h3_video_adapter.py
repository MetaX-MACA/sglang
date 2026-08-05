# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints import utils as entrypoint_utils
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_adapter import (
    BaseVideoModelAdapter,
    get_video_model_adapter,
    validate_adapter_field_claims,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    video_adapter as minimax_h3_video_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    partition_for_task,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
    MiniMaxH3VideoModelAdapter,
)


def _target() -> dict:
    return {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
    }


def _minimax_h3_request(**overrides) -> VideoGenerationsRequest:
    values = {
        "prompt": "a calm lake",
        "task": "t2va",
        "conditions": [],
        "target": _target(),
    }
    values.update(overrides)
    return VideoGenerationsRequest(**values)


def _write_small_av_file(path, *, frame_count: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the final-media frame-count test")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=24",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=32000:cl=stereo",
            "-frames:v",
            str(frame_count),
            "-t",
            str(frame_count / 24),
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "32000",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_video_adapter_registry_uses_mro_and_has_generic_default() -> None:
    class DerivedMiniMaxH3Config(MiniMaxH3PipelineConfig):
        pass

    assert type(get_video_model_adapter(PipelineConfig())) is BaseVideoModelAdapter
    assert isinstance(
        get_video_model_adapter(MiniMaxH3PipelineConfig()),
        MiniMaxH3VideoModelAdapter,
    )
    assert isinstance(
        get_video_model_adapter(DerivedMiniMaxH3Config()),
        MiniMaxH3VideoModelAdapter,
    )


@pytest.mark.parametrize(
    ("task", "partition"),
    [
        ("t2va", "fl2va"),
        ("fl2va", "fl2va"),
        ("ref2va", "ref2va"),
    ],
)
def test_minimax_h3_task_partition_contract(task: str, partition: str) -> None:
    assert partition_for_task(task) == partition


@pytest.mark.parametrize(
    "payload",
    [
        {"task": "t2va"},
        {"flow_shift": 8.0, "audio_flow_shift": 2.0},
    ],
)
def test_json_known_model_fields_require_an_adapter_claim(payload) -> None:
    request = VideoGenerationsRequest(prompt="p", **payload)

    with pytest.raises(ValueError, match="unsupported model-specific"):
        validate_adapter_field_claims(request, BaseVideoModelAdapter())

    validate_adapter_field_claims(request, MiniMaxH3VideoModelAdapter())


def test_raw_multipart_fields_are_collected_then_claim_checked() -> None:
    extra_from_form: dict = {}
    video_api._merge_multipart_extra_form_fields(
        {
            "task": "t2va",
            "conditions": "[]",
            "target": (
                '{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0}'
            ),
        },
        extra_from_form,
    )
    request = VideoGenerationsRequest(prompt="p", **extra_from_form)

    with pytest.raises(ValueError, match="conditions.*target.*task|task.*conditions"):
        validate_adapter_field_claims(request, BaseVideoModelAdapter())
    validate_adapter_field_claims(request, MiniMaxH3VideoModelAdapter())


def test_multipart_declared_generic_fields_are_preserved_and_unknown_fields_dropped():
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

    assert extras["width"] == 640
    assert extras["height"] == 360
    assert extras["guidance_scale"] == 2.5
    assert extras["negative_prompt"] == "avoid blur"
    assert "not_a_declared_video_field" not in extras

    request = VideoGenerationsRequest(prompt="p", **extras)
    assert request.width == 640
    assert request.height == 360
    assert request.guidance_scale == 2.5
    assert request.negative_prompt == "avoid blur"


def test_multipart_optional_generic_fields_do_not_change_minimax_h3_validation() -> None:
    values = {
        "audio_guidance_scale": None,
        "guidance_scale": None,
        "guidance_scale_2": None,
        "negative_prompt": None,
        "true_cfg_scale": None,
        "flow_shift": None,
        "audio_flow_shift": None,
    }
    assert video_api._multipart_declared_request_fields({}, {}, values) == {}

    normal = _minimax_h3_request()
    MiniMaxH3VideoModelAdapter().validate_transport_options(
        normal, model_path="minimax_h3"
    )

    explicit = video_api._multipart_declared_request_fields(
        {"guidance_scale": None}, {}, values
    )
    request = _minimax_h3_request(**explicit)
    MiniMaxH3VideoModelAdapter().validate_transport_options(
        request, model_path="minimax_h3"
    )


def test_minimax_h3_sampling_lowering_preserves_json_fields(monkeypatch) -> None:
    captured = {}

    def fake_build_sampling_params(request_id: str, **kwargs):
        captured["request_id"] = request_id
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "build_sampling_params",
        fake_build_sampling_params,
    )
    request = _minimax_h3_request(
        flow_shift=8.0,
        audio_flow_shift=2.0,
        imgvid_cond_noise_aug_for_inference=0.75,
        audio_cond_noise_aug_for_inference=0.5,
    )

    MiniMaxH3VideoModelAdapter().lower_sampling_params("json-request", request)

    assert captured["request_id"] == "json-request"
    assert captured["task"] == "t2va"
    assert captured["target"] == _target()
    assert captured["flow_shift"] == 8.0
    assert captured["audio_flow_shift"] == 2.0
    assert captured["imgvid_cond_noise_aug_for_inference"] == 0.75
    assert captured["audio_cond_noise_aug_for_inference"] == 0.5
    assert "num_frames" not in captured
    assert "fps" not in captured
    assert "guidance_scale" not in captured
    assert "guidance_scale_2" not in captured
    assert "negative_prompt" not in captured


@pytest.mark.parametrize(
    ("overrides", "expected_flow_shift", "expected_audio_flow_shift"),
    [({"flow_shift": 8.0}, 8.0, None), ({"audio_flow_shift": 2.0}, None, 2.0)],
)
def test_minimax_h3_sampling_lowering_preserves_single_sided_flow_shifts(
    monkeypatch, overrides, expected_flow_shift, expected_audio_flow_shift
) -> None:
    captured = {}
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "build_sampling_params",
        lambda _request_id, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    MiniMaxH3VideoModelAdapter().lower_sampling_params(
        "single-sided", _minimax_h3_request(**overrides)
    )

    assert captured["flow_shift"] == expected_flow_shift
    assert captured["audio_flow_shift"] == expected_audio_flow_shift


@pytest.mark.parametrize(
    "wrapper",
    [
        {
            "extra_body": {
                "flow_shift": 7.0,
                "audio_flow_shift": 1.5,
                "arbitrary_compatibility_value": True,
            }
        },
        {
            "extra_json": {
                "flow_shift": 6.0,
                "audio_flow_shift": 4.0,
                "arbitrary_compatibility_value": True,
            }
        },
    ],
)
def test_json_extra_wrappers_lower_flow_shifts(monkeypatch, wrapper) -> None:
    captured = {}

    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "build_sampling_params",
        lambda _request_id, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    payload = {
        "prompt": "a calm lake",
        "task": "t2va",
        "conditions": [],
        "target": _target(),
        **wrapper,
    }
    flattened = video_api._json_video_payload(payload)
    request = VideoGenerationsRequest(**flattened)
    MiniMaxH3VideoModelAdapter().lower_sampling_params("wrapped", request)

    expected = next(iter(wrapper.values()))
    assert captured["flow_shift"] == expected["flow_shift"]
    assert captured["audio_flow_shift"] == expected["audio_flow_shift"]
    assert "arbitrary_compatibility_value" not in captured


def test_multipart_direct_flow_shift_lowering(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "build_sampling_params",
        lambda _request_id, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    raw_form = {
        "task": "t2va",
        "conditions": "[]",
        "target": json.dumps(_target()),
        "flow_shift": "9.0",
        "audio_flow_shift": "3.5",
    }
    extras: dict = {}
    video_api._merge_multipart_extra_form_fields(raw_form, extras)
    # flow_shift is a declared request/Form field; the endpoint passes it via
    # the declared-field path and strips declared names from the extras splat.
    declared = video_api._multipart_declared_request_fields(
        raw_form, extras, {"flow_shift": 9.0}
    )
    extra_request_fields = {
        key: value
        for key, value in extras.items()
        if key not in VideoGenerationsRequest.model_fields
    }
    request = VideoGenerationsRequest(
        prompt="a calm lake", **declared, **extra_request_fields
    )
    MiniMaxH3VideoModelAdapter().lower_sampling_params("multipart", request)

    assert captured["flow_shift"] == 9.0
    assert captured["audio_flow_shift"] == 3.5


@pytest.mark.parametrize("field_name", ["time_spec", "add_keyframe_instruction"])
def test_unknown_minimax_h3_fields_are_ignored_by_transport_claims(
    field_name: str,
) -> None:
    request = _minimax_h3_request(**{field_name: {} if field_name == "time_spec" else True})
    validate_adapter_field_claims(request, MiniMaxH3VideoModelAdapter())


@pytest.mark.parametrize(
    ("transport", "frame_index"),
    (("json", 0), ("multipart", -1), ("offline", 0)),
)
def test_single_keyframe_interfaces_reach_prequeue_resolution(
    transport: str,
    frame_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MiniMaxH3VideoModelAdapter()
    conditions = [
        {
            "type": "image",
            "uri": "anchor.png",
            "role": "keyframe",
            "frame_index": frame_index,
        }
    ]
    target = {
        "short_edge": 768,
        "aspect_ratio": "auto",
        "duration_seconds": 5.0,
    }

    def sampling_from_kwargs(request_id: str, **kwargs):
        return MiniMaxH3SamplingParams(
            request_id=request_id,
            prompt=kwargs["prompt"],
            task=kwargs["task"],
            conditions=kwargs["conditions"],
            target=kwargs["target"],
            seed=42 if kwargs.get("seed") is None else kwargs["seed"],
            save_output=True,
            output_path="/tmp",
        )

    if transport == "offline":
        sampling = sampling_from_kwargs(
            "offline-single-keyframe",
            prompt="animate the anchor",
            task="fl2va",
            conditions=conditions,
            target=target,
            seed=42,
        )
    else:
        monkeypatch.setattr(
            minimax_h3_video_adapter, "build_sampling_params", sampling_from_kwargs
        )
        if transport == "json":
            request = VideoGenerationsRequest(
                prompt="animate the anchor",
                task="fl2va",
                conditions=conditions,
                target=target,
                seed=42,
            )
        else:
            extras: dict = {}
            video_api._merge_multipart_extra_form_fields(
                {
                    "task": "fl2va",
                    "conditions": json.dumps(conditions),
                    "target": json.dumps(target),
                    "seed": "42",
                },
                extras,
            )
            request = VideoGenerationsRequest(
                prompt="animate the anchor",
                **extras,
            )
        validate_adapter_field_claims(request, adapter)
        adapter.validate_transport_options(request, model_path="minimax_h3")
        sampling = adapter.lower_sampling_params(
            f"{transport}-single-keyframe",
            request,
        )

    adapter.validate_sampling_params(sampling)
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
        enable_trace=False,
    )
    batch = entrypoint_utils.prepare_request(server_args, sampling)

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        prequeue,
    )

    monkeypatch.setattr(
        prequeue,
        "minimax_h3_probe_material",
        lambda *_args, **_kwargs: {
            "local_path": "/localized/anchor.png",
            "display_width": 9,
            "display_height": 16,
        },
    )
    adapter.prepare_for_queue_sync(batch)

    plan = batch.extra["minimax_h3_resolved_plan"]
    assert plan.condition_mask["semantic_frame_indices"] == [frame_index]
    assert plan.condition_mask["pixel_frame_indices"] == [
        0 if frame_index == 0 else int(plan.shape["frame_count"]) - 1
    ]
    assert (plan.shape["width"], plan.shape["height"]) == (768, 1344)


@pytest.mark.parametrize(
    "payload, field_name",
    [
        ({"guidance_scale": None}, "guidance_scale"),
        ({"negative_prompt": None}, "negative_prompt"),
        ({"audio_guidance_scale": 1.0}, "audio_guidance_scale"),
        ({"true_cfg_scale": 1.0}, "true_cfg_scale"),
        ({"uncond_drop_conditions": ["text"]}, "uncond_drop_conditions"),
        ({"ref2va_negative_drop_audio": False}, "ref2va_negative_drop_audio"),
        ({"drop_cond": True}, "drop_cond"),
    ],
)
def test_minimax_h3_ignores_retired_cfg_fields(
    payload, field_name, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _minimax_h3_request(**payload)
    adapter = MiniMaxH3VideoModelAdapter()
    captured = {}
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "build_sampling_params",
        lambda _request_id, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    adapter.validate_transport_options(request, model_path="minimax_h3")
    adapter.lower_sampling_params("request-id", request)

    assert field_name not in captured


@pytest.mark.parametrize(
    "payload",
    [
        {"guidance_scale": 7.5},
        {"guidance_scale": 1.0},
        {"guidance_scale_2": 3.0},
        {"guidance_scale_2": 1.0},
        {"negative_prompt": "avoid blur"},
        {"negative_prompt": ""},
    ],
)
def test_minimax_h3_rejects_retired_cfg_fields(
    payload, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _minimax_h3_request(**payload)
    adapter = MiniMaxH3VideoModelAdapter()
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "build_sampling_params",
        lambda _request_id, **kwargs: SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="is not supported"):
        adapter.lower_sampling_params("request-id", request)


def test_minimax_h3_rejects_explicit_frames_and_fps_on_http_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MiniMaxH3VideoModelAdapter()
    captured = {}
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "build_sampling_params",
        lambda _request_id, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    for overrides in (
        {"num_frames": 121},
        {"fps": 12},
        {"fps": 24},
        {"fps": 24, "num_frames": 121},
    ):
        request = _minimax_h3_request(**overrides)
        adapter.validate_transport_options(request, model_path="minimax_h3")
        with pytest.raises(
            ValueError,
            match="derives the temporal shape from target.duration_seconds",
        ):
            adapter.lower_sampling_params("request-id", request)
    assert captured == {}

    # Default clients leave the protocol fields at None; lowering still strips
    # the transport-synthesized timing defaults before SamplingParams.
    adapter.lower_sampling_params("request-id", _minimax_h3_request())
    assert "num_frames" not in captured
    assert "fps" not in captured

    sampling = MiniMaxH3SamplingParams(
        prompt="p",
        task="t2va",
        conditions=[],
        target=_target(),
        save_output=True,
        output_path="/tmp",
    )
    sampling._explicit_fields = {"prompt", "task", "target"}
    adapter.validate_sampling_params(sampling)
    assert sampling.num_frames == 1
    assert sampling.fps == 24


def test_prepare_builds_canonical_extra_once_and_projection_reuses_it(
    monkeypatch,
) -> None:
    sampling = MiniMaxH3SamplingParams(
        request_id="canonical-once",
        prompt="p",
        task="t2va",
        conditions=[],
        target=_target(),
    )
    original = sampling.build_request_extra
    calls = 0

    def counted_build_request_extra():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(sampling, "build_request_extra", counted_build_request_extra)
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
        enable_trace=False,
    )

    batch = entrypoint_utils.prepare_request(server_args, sampling)
    fields = MiniMaxH3VideoModelAdapter().project_queued_job_fields(batch)

    assert calls == 1
    assert batch.extra["minimax_h3_canonical_request"]["seed"] == sampling.seed
    assert fields == {"size": "1344x768", "seconds": "5.166667"}


def test_minimax_h3_single_output_seed_list_refreshes_canonical_extra() -> None:
    sampling = MiniMaxH3SamplingParams(
        request_id="single-seed-list",
        prompt="p",
        task="t2va",
        conditions=[],
        target=_target(),
        seed=[7],
    )
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
        enable_trace=False,
    )
    batch = entrypoint_utils.prepare_request(server_args, sampling)

    dispatch_batch = MiniMaxH3VideoModelAdapter().expand_for_dispatch(batch)

    assert isinstance(dispatch_batch, list)
    assert dispatch_batch == [batch]
    assert batch.seed == 7
    assert batch.extra["minimax_h3_canonical_request"]["seed"] == 7


def test_minimax_h3_final_validation_uses_original_multi_output_count() -> None:
    batch = SimpleNamespace(extra={}, num_outputs_per_prompt=2)

    with pytest.raises(RuntimeError, match="1 output files, expected 2"):
        MiniMaxH3VideoModelAdapter().validate_final_outputs_sync(["only-one.mp4"], batch)


def test_minimax_h3_final_validation_async_wrapper_uses_one_sync_hook(
    monkeypatch,
) -> None:
    calls = []
    adapter = MiniMaxH3VideoModelAdapter()
    batch = SimpleNamespace(extra={}, num_outputs_per_prompt=2)

    def fake_sync(output_paths, request_batch):
        calls.append((output_paths, request_batch))
        return {"size": "1344x768", "seconds": "8.708333"}

    async def fake_to_thread(function, *args, **kwargs):
        calls.append((function, args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(adapter, "validate_final_outputs_sync", fake_sync)
    monkeypatch.setattr(minimax_h3_video_adapter.asyncio, "to_thread", fake_to_thread)
    fields = asyncio.run(adapter.validate_final_outputs(["zero.mp4", "one.mp4"], batch))

    assert fields == {"size": "1344x768", "seconds": "8.708333"}
    assert calls[0][0] is fake_sync
    assert calls[0][1][0] == ["zero.mp4", "one.mp4"]
    assert calls[0][1][1] is batch


@pytest.mark.parametrize("actual_frame_count", [9, 11])
def test_probe_rejects_real_mp4_one_frame_short_or_long(
    tmp_path,
    actual_frame_count,
) -> None:
    output_path = tmp_path / f"actual-{actual_frame_count}.mp4"
    _write_small_av_file(output_path, frame_count=actual_frame_count)

    with pytest.raises(
        RuntimeError,
        match=rf"frame count.*expected 10, got {actual_frame_count}",
    ):
        minimax_h3_video_adapter._probe_minimax_h3_output_fields(
            str(output_path),
            expected_frame_count=10,
        )


def test_minimax_h3_probe_has_30_second_timeout_and_reports_timeout(monkeypatch) -> None:
    captured = {}

    def timeout(_command, **kwargs):
        captured.update(kwargs)
        raise subprocess.TimeoutExpired("ffprobe", kwargs["timeout"])

    monkeypatch.setattr(minimax_h3_video_adapter.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="timed out after 30 seconds"):
        minimax_h3_video_adapter._probe_minimax_h3_output_fields("generated.mp4")
    assert captured["timeout"] == 30


def test_minimax_h3_probe_reports_ffprobe_failure(monkeypatch) -> None:
    def failed(command, **_kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="invalid media",
        )

    monkeypatch.setattr(minimax_h3_video_adapter.subprocess, "run", failed)

    with pytest.raises(RuntimeError, match="ffprobe failed.*invalid media"):
        minimax_h3_video_adapter._probe_minimax_h3_output_fields("generated.mp4")


@pytest.mark.parametrize(
    ("seed", "expected_seeds"),
    [
        (17, [17, 18, 19]),
        ([17, 18, 19], [17, 18, 19]),
    ],
)
def test_minimax_h3_multi_output_refreshes_each_scalar_canonical_seed(
    seed, expected_seeds
) -> None:
    sampling = MiniMaxH3SamplingParams(
        request_id="multi-seed-refresh",
        prompt="p",
        task="t2va",
        conditions=[],
        target=_target(),
        seed=seed,
        num_outputs_per_prompt=3,
    )
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
        enable_trace=False,
    )
    batch = entrypoint_utils.prepare_request(server_args, sampling)

    dispatch_batch = MiniMaxH3VideoModelAdapter().expand_for_dispatch(batch)

    assert [item.seed for item in dispatch_batch] == expected_seeds
    assert [
        item.extra["minimax_h3_canonical_request"]["seed"] for item in dispatch_batch
    ] == expected_seeds


@pytest.mark.parametrize("bad_seed", [None, "17", 1.5, True])
def test_minimax_h3_multi_output_expansion_requires_integer_child_seed(
    bad_seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = [
        SimpleNamespace(seed=bad_seed, extra={"minimax_h3_resolved_plan": object()}),
        SimpleNamespace(seed=18, extra={}),
    ]
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "expand_request_outputs",
        lambda _batch, **_kwargs: children,
    )
    batch = SimpleNamespace(num_outputs_per_prompt=2, seed=17, extra={})

    with pytest.raises(ValueError, match="output 0 must carry an integer seed"):
        MiniMaxH3VideoModelAdapter().expand_for_dispatch(batch)


def test_minimax_h3_multi_output_plan_seed_refresh_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resolved plan that is not a msgspec struct is a producer bug; the
    # msgspec.structs.replace failure must surface instead of silently
    # keeping the shared parent seed.
    children = [
        SimpleNamespace(seed=17, extra={"minimax_h3_resolved_plan": object()}),
        SimpleNamespace(seed=18, extra={}),
    ]
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "expand_request_outputs",
        lambda _batch, **_kwargs: children,
    )
    batch = SimpleNamespace(num_outputs_per_prompt=2, seed=17, extra={})

    with pytest.raises(TypeError):
        MiniMaxH3VideoModelAdapter().expand_for_dispatch(batch)
