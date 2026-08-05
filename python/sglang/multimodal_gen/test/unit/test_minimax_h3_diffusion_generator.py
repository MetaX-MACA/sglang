# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle regressions for MiniMax H3's offline DiffGenerator path."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MiniMaxH3ResolvedPlan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
    MiniMaxH3VideoModelAdapter,
)

_TASKS = ("t2va", "fl2va", "ref2va")


def _resolved_plan(
    task: str,
    seed: int,
    *,
    flow_shift: float | None = None,
    audio_flow_shift: float | None = None,
) -> MiniMaxH3ResolvedPlan:
    default_flow_shift, default_audio_flow_shift = (12.0, 3.0)
    return MiniMaxH3ResolvedPlan(
        task=task,
        prompt="prompt",
        seed=seed,
        materials=(),
        encoders={},
        branches=(),
        default_flow_shift=default_flow_shift,
        default_audio_flow_shift=default_audio_flow_shift,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        shape={
            "geometry": "resolved_v2",
            "width": 1344,
            "height": 768,
            "fps": 24,
            "frame_count": 121,
        },
        condition_mask={},
    )


def _sampling_params(
    *,
    task: str,
    output_path: Path,
    output_file_name: str,
    num_outputs: int,
    seed: int | list[int],
    flow_shift: float | None = None,
    audio_flow_shift: float | None = None,
) -> MiniMaxH3SamplingParams:
    conditions_by_task = {
        "t2va": [],
        "fl2va": [
            {"role": "keyframe", "type": "image", "uri": "first.png", "frame_index": 0},
            {"role": "keyframe", "type": "image", "uri": "last.png", "frame_index": -1},
        ],
        "ref2va": [
            {"role": "reference", "type": "image", "uri": "reference.png"},
        ],
    }
    target = {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
    }
    return MiniMaxH3SamplingParams(
        request_id=f"request-{task}",
        prompt="prompt",
        output_path=str(output_path),
        output_file_name=output_file_name,
        save_output=True,
        num_outputs_per_prompt=num_outputs,
        seed=seed,
        task=task,
        conditions=conditions_by_task.get(task, []),
        target=target,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
    )


def _make_req(sampling_params: MiniMaxH3SamplingParams, task: str) -> Req:
    req = Req(sampling_params=sampling_params)
    req.extra.update(
        {
            "minimax_h3_canonical_request": {
                "schema": "minimax_h3.request/v1",
                "task": task,
                "prompt": sampling_params.prompt,
                "conditions": sampling_params.conditions,
                "target": sampling_params.target,
                "seed": sampling_params.seed,
                "flow_shift": sampling_params.flow_shift,
                "audio_flow_shift": sampling_params.audio_flow_shift,
            },
            "minimax_h3_resolved_plan": _resolved_plan(
                task,
                (
                    sampling_params.seed
                    if isinstance(sampling_params.seed, int)
                    else int(sampling_params.seed[0])
                ),
                flow_shift=sampling_params.flow_shift,
                audio_flow_shift=sampling_params.audio_flow_shift,
            ),
        }
    )
    return req


@pytest.fixture
def lifecycle_harness(tmp_path, monkeypatch):
    """Build a real DiffGenerator with transport/scheduler boundaries mocked."""

    events: list[tuple] = []
    state: dict[str, object] = {}
    state["task"] = "t2va"
    adapter = MiniMaxH3VideoModelAdapter()
    original_validate_sampling = adapter.validate_sampling_params

    def validate_sampling(params):
        events.append(("sampling_validate", params.task))
        return original_validate_sampling(params)

    def prepare(parent):
        events.append(("prequeue", parent))
        if state.get("fail_prequeue"):
            raise RuntimeError("prequeue failed")
        parent.extra["prequeued"] = True
        parent.extra.setdefault("minimax_h3_request_temp_dirs", {})["prequeue"] = []

    original_expand = adapter.expand_for_dispatch

    def expand(parent, *, num_prompts=1, prompt_index=0):
        events.append(("expand", parent))
        if state.get("fail_expand"):
            raise RuntimeError("expansion failed")
        return original_expand(
            parent, num_prompts=num_prompts, prompt_index=prompt_index
        )

    def final_validate(paths, parent):
        events.append(("final_validate", list(paths), parent))
        if state.get("fail_final"):
            raise RuntimeError("final validation failed")
        return {"size": "1344x768", "seconds": "5.041667"}

    def cleanup(parent):
        events.append(("cleanup", parent))
        parent.extra.pop("minimax_h3_request_temp_dirs", None)

    adapter.validate_sampling_params = validate_sampling
    adapter.prepare_for_queue_sync = prepare
    adapter.expand_for_dispatch = expand
    adapter.validate_final_outputs_sync = final_validate
    adapter.cleanup_request_sync = cleanup

    monkeypatch.setattr(dg, "get_video_model_adapter", lambda _config: adapter)

    def fake_from_user(_model_path, server_args, *args, **kwargs):
        del server_args, args
        task = str(state["task"])
        return _sampling_params(
            task=task,
            output_path=tmp_path,
            output_file_name=f"{task}.mp4",
            num_outputs=int(
                kwargs.get("num_outputs_per_prompt", kwargs.get("n", 1)) or 1
            ),
            seed=kwargs.get("seed", 7),
            flow_shift=kwargs.get("flow_shift"),
            audio_flow_shift=kwargs.get("audio_flow_shift"),
        )

    monkeypatch.setattr(
        dg.SamplingParams,
        "from_user_sampling_params_args",
        staticmethod(fake_from_user),
    )

    def fake_prepare(server_args, sampling_params, external_trace_header=None):
        del server_args, external_trace_header
        return _make_req(sampling_params, str(state["task"]))

    monkeypatch.setattr(dg, "prepare_request", fake_prepare)

    @contextlib.contextmanager
    def fake_trace(_trace_ctx):
        yield

    monkeypatch.setattr(dg, "trace_req", fake_trace)

    class _Timer:
        duration = 0.25

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(dg, "log_generation_timer", lambda *_args: _Timer())

    server_args = SimpleNamespace(
        model_path="fake-model",
        prompt_file_path=None,
        pipeline_config=SimpleNamespace(
            requires_audio_output=True,
            # Strict AV save reads the delivery contract from the pipeline
            # config; mirror MiniMaxH3PipelineConfig defaults.
            output_audio_sample_rate=32000,
            output_audio_channels=2,
            output_av_drift_tolerance_s=0.25,
        ),
        batching_max_size=1,
        warmup=False,
    )
    generator = object.__new__(DiffGenerator)
    generator.server_args = server_args

    def run(
        task: str,
        *,
        num_outputs: int = 1,
        seed: int | list[int] = 7,
        fail_dispatch: bool = False,
        fail_final: bool = False,
        fail_prequeue: bool = False,
        fail_expand: bool = False,
        output_paths: list[Path] | None = None,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
    ):
        state.update(
            task=task,
            fail_final=fail_final,
            fail_prequeue=fail_prequeue,
            fail_expand=fail_expand,
            output_paths=output_paths,
        )
        paths = output_paths or [
            tmp_path / f"{task}_{i}.mp4" for i in range(num_outputs)
        ]

        def dispatch(requests):
            events.append(("dispatch", list(requests)))
            if fail_dispatch:
                raise RuntimeError("dispatch failed")
            for path in paths:
                path.write_bytes(b"generated")
            return OutputBatch(
                output_file_paths=[str(path) for path in paths],
            )

        generator._send_to_scheduler_and_wait_for_response = dispatch
        return generator.generate(
            {
                "prompt": "prompt",
                "task": task,
                "output_path": str(tmp_path),
                "output_file_name": f"{task}.mp4",
                "num_outputs_per_prompt": num_outputs,
                "seed": seed,
                "flow_shift": flow_shift,
                "audio_flow_shift": audio_flow_shift,
            }
        )

    return SimpleNamespace(
        generator=generator,
        adapter=adapter,
        events=events,
        state=state,
        run=run,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("task", _TASKS)
def test_diff_generator_supports_all_canonical_minimax_h3_tasks(lifecycle_harness, task):
    harness = lifecycle_harness
    result = harness.run(task)

    assert result is not None
    assert result.output_file_path is not None
    assert Path(result.output_file_path).is_file()
    names = [event[0] for event in harness.events]
    assert names == [
        "sampling_validate",
        "prequeue",
        "expand",
        "dispatch",
        "final_validate",
        "cleanup",
    ]
    final_event = harness.events[4]
    assert final_event[2] is harness.events[1][1]


def test_invalid_minimax_h3_task_fails_before_prequeue(lifecycle_harness):
    harness = lifecycle_harness
    # "i2va" is a first-frame fl2va variant, not a task name.
    with pytest.raises(ValueError, match="unsupported MiniMax H3 task"):
        harness.run("i2va")
    assert [event[0] for event in harness.events] == ["sampling_validate"]


def test_diff_generator_forwards_flow_shifts_to_canonical_request(
    lifecycle_harness,
):
    harness = lifecycle_harness
    harness.run("t2va", flow_shift=7.5, audio_flow_shift=2.25)

    child = next(event for event in harness.events if event[0] == "dispatch")[1][0]
    canonical = child.extra["minimax_h3_canonical_request"]
    assert child.sampling_params.flow_shift == 7.5
    assert child.sampling_params.audio_flow_shift == 2.25
    assert canonical["flow_shift"] == 7.5
    assert canonical["audio_flow_shift"] == 2.25
    assert child.extra["minimax_h3_resolved_plan"].flow_shift == 7.5
    assert child.extra["minimax_h3_resolved_plan"].audio_flow_shift == 2.25


def test_multi_output_prequeues_once_and_isolates_child_seed_and_resources(
    lifecycle_harness,
):
    harness = lifecycle_harness
    result = harness.run("t2va", num_outputs=3, seed=41)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(Path(item.output_file_path).is_file() for item in result)
    assert [event[0] for event in harness.events].count("prequeue") == 1
    children = harness.events[3][1]
    assert [child.seed for child in children] == [41, 42, 43]
    assert len({child.request_id for child in children}) == 3
    # Cleanup runs for every expanded child; an empty registry is removed by
    # the adapter's idempotent cleanup hook.
    assert all(
        child.extra.get("minimax_h3_request_temp_dirs", {}) == {} for child in children
    )
    assert [
        child.extra["minimax_h3_canonical_request"]["seed"] for child in children
    ] == [41, 42, 43]
    assert [child.extra["minimax_h3_resolved_plan"].seed for child in children] == [
        41,
        42,
        43,
    ]


@pytest.mark.parametrize("failure", ["prequeue", "expand", "dispatch", "final"])
def test_failed_generation_raises_cleans_new_outputs_and_publishes_no_partial_result(
    lifecycle_harness, failure
):
    harness = lifecycle_harness
    paths = [harness.tmp_path / "failure_0.mp4", harness.tmp_path / "failure_1.mp4"]
    with pytest.raises(RuntimeError, match="no successful outputs"):
        harness.run(
            "t2va",
            num_outputs=2,
            fail_dispatch=failure == "dispatch",
            fail_final=failure == "final",
            fail_prequeue=failure == "prequeue",
            fail_expand=failure == "expand",
            output_paths=paths,
        )

    assert not any(path.exists() for path in paths)
    assert [event[0] for event in harness.events][-1] == "cleanup"


def test_all_prompts_failed_raises_with_first_error(lifecycle_harness):
    harness = lifecycle_harness
    harness.state["fail_prequeue"] = True

    with pytest.raises(RuntimeError, match=r"first error:.*prequeue failed"):
        harness.generator.generate(
            {
                "prompt": ["first prompt", "second prompt"],
                "task": "t2va",
                "output_path": str(harness.tmp_path),
                "num_outputs_per_prompt": 1,
                "seed": 7,
            }
        )
    assert [event[0] for event in harness.events].count("prequeue") == 2


def test_partial_prompt_failure_still_returns_surviving_results(lifecycle_harness):
    harness = lifecycle_harness
    calls = {"count": 0}
    original_prepare = harness.adapter.prepare_for_queue_sync

    def flaky_prepare(parent):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("prequeue failed for prompt 0")
        original_prepare(parent)

    harness.adapter.prepare_for_queue_sync = flaky_prepare

    def dispatch(requests):
        harness.events.append(("dispatch", list(requests)))
        paths = [Path(request.output_file_path(1, 0)) for request in requests]
        for path in paths:
            path.write_bytes(b"generated")
        return OutputBatch(output_file_paths=[str(path) for path in paths])

    harness.generator._send_to_scheduler_and_wait_for_response = dispatch
    result = harness.generator.generate(
        {
            "prompt": ["first prompt", "second prompt"],
            "task": "t2va",
            "output_path": str(harness.tmp_path),
            "num_outputs_per_prompt": 1,
            "seed": 7,
        }
    )

    assert result is not None
    assert not isinstance(result, list)
    assert result.prompt_index == 1
    assert Path(result.output_file_path).is_file()


def test_save_failure_cleans_outputs_and_preserves_preexisting_file(
    lifecycle_harness, monkeypatch
):
    harness = lifecycle_harness
    harness.state["task"] = "t2va"
    preexisting = harness.tmp_path / "t2va_0.mp4"
    new_path = harness.tmp_path / "t2va_1.mp4"
    preexisting.write_bytes(b"old")

    # Return raw output so the MiniMax H3 file-first branch invokes save_outputs.
    def dispatch(requests):
        harness.events.append(("dispatch", list(requests)))
        return OutputBatch(output=[object(), object()])

    harness.generator._send_to_scheduler_and_wait_for_response = dispatch

    def failing_save(_output, _data_type, _fps, _save_output, build_path, **_kwargs):
        Path(build_path(0)).write_bytes(b"rewritten")
        Path(build_path(1)).write_bytes(b"new")
        raise RuntimeError("save failed")

    monkeypatch.setattr(dg, "save_outputs", failing_save)
    with pytest.raises(RuntimeError, match=r"no successful outputs.*save failed"):
        harness.generator.generate(
            {
                "prompt": "prompt",
                "task": "t2va",
                "output_path": str(harness.tmp_path),
                "output_file_name": "t2va.mp4",
                "num_outputs_per_prompt": 2,
                "seed": 7,
            }
        )

    assert preexisting.read_bytes() == b"old"
    assert not new_path.exists()
