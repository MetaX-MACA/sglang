# SPDX-License-Identifier: Apache-2.0
"""Focused admission tests for MiniMax H3 probe -> resolve-once preparation."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _RawRequest:
    def __init__(self, payload: dict[str, Any], *, multipart: bool) -> None:
        self._payload = payload
        self.headers = {
            "content-type": (
                "multipart/form-data; boundary=test"
                if multipart
                else "application/json"
            )
        }

    async def form(self) -> dict[str, Any]:
        return self._payload

    async def json(self) -> dict[str, Any]:
        return self._payload


@pytest.mark.parametrize("multipart", [False, True], ids=["json", "multipart"])
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"task": None},
        {"task": ""},
        {"task": "   "},
        {"task": 7},
        {"task": ["fl2va"]},
        {"task": "unknown"},
        {"task": "foo2va"},
    ],
    ids=[
        "missing",
        "null",
        "empty",
        "blank",
        "integer",
        "list",
        "unknown",
        "unknown-2va",
    ],
)
def test_raw_task_gate_rejects_before_request_side_effects(
    payload: dict[str, Any],
    multipart: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
        MiniMaxH3VideoModelAdapter,
    )

    touched: list[str] = []

    def touched_sync(name: str):
        def record(*_args, **_kwargs):
            touched.append(name)
            return SimpleNamespace()

        return record

    async def touched_async(name: str, *_args, **_kwargs):
        touched.append(name)

    pipeline_config = SimpleNamespace(
        task_type=SimpleNamespace(requires_image_input=lambda: False)
    )
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(pipeline_config=pipeline_config),
    )
    monkeypatch.setattr(
        video_api,
        "get_video_model_adapter",
        lambda _config: MiniMaxH3VideoModelAdapter(),
    )
    monkeypatch.setattr(
        video_api.tempfile,
        "TemporaryDirectory",
        touched_sync("temporary-directory"),
    )
    monkeypatch.setattr(video_api, "prepare_request", touched_sync("scheduler-req"))
    monkeypatch.setattr(
        video_api,
        "_save_first_input_image",
        lambda *_args, **_kwargs: touched_async("upload"),
    )
    monkeypatch.setattr(
        video_api.VIDEO_STORE,
        "upsert",
        lambda *_args, **_kwargs: touched_async("store"),
    )
    monkeypatch.setattr(video_api.asyncio, "create_task", touched_sync("dispatch"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            video_api.create_video(
                _RawRequest(payload, multipart=multipart),
                extra_body=None,
                extra_params=None,
            )
        )

    assert exc_info.value.status_code == 400
    assert touched == []


def test_invalid_request_json_body_rejects_before_request_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
    from sglang.multimodal_gen.runtime.entrypoints.openai.video_adapter import (
        BaseVideoModelAdapter,
    )

    class InvalidJsonRequest:
        headers = {"content-type": "application/json"}

        async def json(self) -> dict[str, Any]:
            raise json.JSONDecodeError("Expecting value", "{", 1)

    pipeline_config = SimpleNamespace(
        task_type=SimpleNamespace(requires_image_input=lambda: False)
    )
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(pipeline_config=pipeline_config),
    )
    monkeypatch.setattr(
        video_api,
        "get_video_model_adapter",
        lambda _config: BaseVideoModelAdapter(),
    )
    monkeypatch.setattr(
        video_api.tempfile,
        "TemporaryDirectory",
        lambda *_args, **_kwargs: pytest.fail(
            "request resources must not be created for invalid JSON"
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            video_api.create_video(
                InvalidJsonRequest(),
                extra_body=None,
                extra_params=None,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "request body is not valid JSON"


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_task_gate_accepts_only_the_exact_wire_tasks(task: str) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
        MiniMaxH3VideoModelAdapter,
    )

    MiniMaxH3VideoModelAdapter().validate_task_gate(task, provided=True)


def _video_probe_payload(
    *,
    width: int,
    height: int,
    sample_aspect_ratio: str = "1:1",
    display_aspect_ratio: str | None = None,
    rotation: float | None = None,
) -> dict[str, Any]:
    stream: dict[str, Any] = {
        "codec_type": "video",
        "width": width,
        "height": height,
        "duration": "2.0",
        "avg_frame_rate": "24/1",
        "r_frame_rate": "24/1",
        "nb_read_frames": "48",
        "sample_aspect_ratio": sample_aspect_ratio,
    }
    if display_aspect_ratio is not None:
        stream["display_aspect_ratio"] = display_aspect_ratio
    if rotation is not None:
        stream["side_data_list"] = [{"rotation": rotation}]
    return {
        "streams": [stream],
        "format": {"format_name": "mp4", "duration": "2.0"},
    }


def test_image_probe_uses_exif_transposed_display_geometry(tmp_path: Path) -> None:
    from PIL import Image

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        material_io,
    )

    path = tmp_path / "portrait-by-exif.jpg"
    image = Image.new("RGB", (40, 20), "red")
    exif = image.getexif()
    exif[274] = 6  # 90 degrees clockwise
    image.save(path, exif=exif)

    facts = material_io._validate_localized_media(
        str(path),
        condition_type="image",
    )

    assert (facts["coded_width"], facts["coded_height"]) == (40, 20)
    assert (facts["display_width"], facts["display_height"]) == (20, 40)
    assert facts["exif_transposed"] is True


def test_video_probe_applies_rotation_to_display_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        material_io,
    )

    payload = _video_probe_payload(width=1920, height=1080, rotation=90)
    monkeypatch.setattr(
        material_io.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    facts = material_io._validate_localized_media(
        "rotated.mp4",
        condition_type="video",
    )

    assert facts["rotation_degrees"] == 90
    assert facts["display_width"] == pytest.approx(1080)
    assert facts["display_height"] == pytest.approx(1920)
    assert facts["display_aspect_ratio"] == pytest.approx(9 / 16)


def test_video_probe_uses_square_pixel_sar_and_dar_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        material_io,
    )

    payload = _video_probe_payload(
        width=720,
        height=576,
        sample_aspect_ratio="16:15",
        display_aspect_ratio="4:3",
    )
    monkeypatch.setattr(
        material_io.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    facts = material_io._validate_localized_media(
        "anamorphic.mp4",
        condition_type="video",
    )

    assert facts["coded_width"] == 720
    assert facts["sample_aspect_ratio"] == "16:15"
    assert facts["sample_aspect_ratio_value"] == pytest.approx(16 / 15)
    assert facts["display_width"] == pytest.approx(768)
    assert facts["display_height"] == pytest.approx(576)
    assert facts["display_aspect_ratio"] == pytest.approx(4 / 3)


def test_video_probe_keeps_video_and_audio_durations_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        material_io,
    )

    payload = _video_probe_payload(width=1280, height=720)
    payload["streams"][0]["duration"] = "2.0"
    payload["streams"].append(
        {
            "codec_type": "audio",
            "duration": "10.0",
            "sample_rate": "32000",
            "channels": 2,
        }
    )
    payload["format"]["duration"] = "10.0"
    monkeypatch.setattr(
        material_io.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    facts = material_io._validate_localized_media(
        "different-av-durations.mp4",
        condition_type="video",
    )

    assert facts["duration_seconds"] == 10.0
    assert facts["video_duration_seconds"] == 2.0
    assert facts["audio_duration_seconds"] == 10.0


def test_localized_source_probe_is_cached_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        material_io,
    )

    source = tmp_path / "source.png"
    source.write_bytes(b"bounded-image-source")
    batch = SimpleNamespace(extra={})
    calls: list[str] = []
    expected = {
        "condition_type": "image",
        "coded_width": 32,
        "coded_height": 64,
        "display_width": 32,
        "display_height": 64,
    }

    def fake_validate(path: str, **_kwargs) -> dict[str, Any]:
        calls.append(path)
        return expected

    monkeypatch.setattr(material_io, "_validate_localized_media", fake_validate)

    first = material_io.minimax_h3_probe_material(
        batch,
        str(source),
        condition_type="image",
        condition_index=0,
    )
    second = material_io.minimax_h3_probe_material(
        batch,
        str(source),
        condition_type="image",
        condition_index=1,
    )

    assert calls == [str(source)]
    assert first == second == {"local_path": str(source), **expected}


def _canonical_batch(
    *,
    task: str,
    conditions: list[dict[str, Any]] | None,
    target: dict[str, Any],
    steps: int = 1,
    outputs: int = 1,
    flow_shift: float | None = None,
    audio_flow_shift: float | None = None,
) -> SimpleNamespace:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
        minimax_h3_validate_canonical_request,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
    )

    canonical = minimax_h3_validate_canonical_request(
        task=task,
        prompt="probe once",
        conditions=conditions,
        target=target,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
    )
    return SimpleNamespace(
        extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical},
        num_inference_steps=steps,
        num_outputs_per_prompt=outputs,
    )


def _fake_probe(
    monkeypatch: pytest.MonkeyPatch,
    facts_by_uri: dict[str, dict[str, Any]],
) -> tuple[Any, list[tuple[str, str, int]]]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        prequeue,
    )

    calls: list[tuple[str, str, int]] = []

    def probe(
        _batch,
        uri: str,
        *,
        condition_type: str,
        condition_index: int,
    ) -> dict[str, Any]:
        calls.append((uri, condition_type, condition_index))
        return {"local_path": f"/localized/{condition_index}", **facts_by_uri[uri]}

    monkeypatch.setattr(prequeue, "minimax_h3_probe_material", probe)
    return prequeue, calls


def test_fl_auto_target_resolves_from_first_keyframe_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _canonical_batch(
        task="fl2va",
        conditions=[
            {
                "type": "image",
                "uri": "first.png",
                "role": "keyframe",
                "frame_index": 0,
            },
            {
                "type": "image",
                "uri": "last.png",
                "role": "keyframe",
                "frame_index": -1,
            },
        ],
        target={
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 5.0,
        },
    )
    prequeue, calls = _fake_probe(
        monkeypatch,
        {
            "first.png": {"display_width": 21, "display_height": 9},
            "last.png": {"display_width": 9, "display_height": 21},
        },
    )

    plan = prequeue.minimax_h3_prepare_for_queue(batch)

    assert (plan.shape["width"], plan.shape["height"]) == (1536, 672)
    assert plan.shape["geometry_source"] == "first_keyframe"
    assert plan.shape["geometry_source_condition_index"] == 0
    assert plan.shape["geometry_source_frame_index"] == 0
    assert plan.shape["size_mode"] == "area"
    assert plan.shape["shape_policy_version"] == "adapt_shape_v1"
    assert (batch.width, batch.height, batch.num_frames) == (1536, 672, 124)
    assert calls == [
        ("first.png", "image", 0),
        ("last.png", "image", 1),
    ]
    material_shapes = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY]
    assert {
        index: (shape["width"], shape["height"])
        for index, shape in material_shapes.items()
    } == {0: (1536, 672), 1: (1536, 672)}


@pytest.mark.parametrize(
    ("frame_index", "display_size", "expected_size", "geometry_source"),
    [
        (0, (4, 3), (1024, 768), "first_keyframe"),
        (-1, (1, 2), (704, 1440), "last_keyframe"),
    ],
)
def test_fl_auto_target_resolves_from_single_semantic_keyframe(
    frame_index: int,
    display_size: tuple[int, int],
    expected_size: tuple[int, int],
    geometry_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _canonical_batch(
        task="fl2va",
        conditions=[
            {
                "type": "image",
                "uri": "anchor.png",
                "role": "keyframe",
                "frame_index": frame_index,
            }
        ],
        target={
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 5.0,
        },
    )
    prequeue, calls = _fake_probe(
        monkeypatch,
        {
            "anchor.png": {
                "display_width": display_size[0],
                "display_height": display_size[1],
            }
        },
    )

    plan = prequeue.minimax_h3_prepare_for_queue(batch)

    assert (plan.shape["width"], plan.shape["height"]) == expected_size
    assert plan.shape["geometry_source"] == geometry_source
    assert plan.shape["geometry_source_condition_index"] == 0
    assert plan.shape["geometry_source_frame_index"] == frame_index
    assert calls == [("anchor.png", "image", 0)]
    material_shape = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY][
        0
    ]
    assert (material_shape["width"], material_shape["height"]) == expected_size


def test_reference_videos_resolve_independent_material_canvases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _canonical_batch(
        task="ref2va",
        conditions=[
            {"type": "video", "uri": "wide.mp4", "role": "reference"},
            {"type": "video", "uri": "tall.mp4", "role": "reference"},
        ],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
    )
    common = {
        "duration_seconds": 2.0,
        "fps": 24.0,
        "frame_count": 48,
        "has_audio": True,
    }
    prequeue, _calls = _fake_probe(
        monkeypatch,
        {
            "wide.mp4": {**common, "display_width": 21, "display_height": 9},
            "tall.mp4": {**common, "display_width": 9, "display_height": 21},
        },
    )

    plan = prequeue.minimax_h3_prepare_for_queue(batch)

    assert (plan.shape["width"], plan.shape["height"]) == (1344, 768)
    shapes = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY]
    assert (shapes[0]["width"], shapes[0]["height"]) == (1536, 672)
    assert (shapes[1]["width"], shapes[1]["height"]) == (672, 1536)
    assert shapes[0]["base_short_edge"] == shapes[1]["base_short_edge"] == 768
    assert (
        shapes[0]["shape_policy_version"]
        == shapes[1]["shape_policy_version"]
        == "adapt_shape_v1"
    )
    assert shapes[0]["rounding"] == shapes[1]["rounding"] == "nearest"


def test_reference_image_keeps_base_2048_when_target_effective_edge_is_672(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _canonical_batch(
        task="ref2va",
        conditions=[{"type": "image", "uri": "subject.png", "role": "reference"}],
        target={
            "short_edge": 768,
            "aspect_ratio": "21:9",
            "duration_seconds": 5.0,
        },
    )
    prequeue, _calls = _fake_probe(
        monkeypatch,
        {"subject.png": {"display_width": 1080, "display_height": 1440}},
    )

    plan = prequeue.minimax_h3_prepare_for_queue(batch)

    assert plan.shape["effective_short_edge"] == 672
    ref_shape = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY][0]
    assert ref_shape["base_short_edge"] == 2048
    assert ref_shape["effective_short_edge"] == 2048
    assert (ref_shape["width"], ref_shape["height"]) == (2048, 2720)
    assert ref_shape["shape_policy_version"] == "reference_image_short_edge_v1"
    assert ref_shape["rounding"] == "nearest"
    assert ref_shape["allow_upscale"] is True


def test_reference_image_uses_independent_2048_short_edge_and_allows_upscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _canonical_batch(
        task="ref2va",
        conditions=[{"type": "image", "uri": "small.png", "role": "reference"}],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
    )
    prequeue, _calls = _fake_probe(
        monkeypatch,
        {"small.png": {"display_width": 320, "display_height": 240}},
    )

    plan = prequeue.minimax_h3_prepare_for_queue(batch)

    assert (plan.shape["width"], plan.shape["height"]) == (1344, 768)
    ref_shape = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY][0]
    assert (ref_shape["width"], ref_shape["height"]) == (2720, 2048)
    assert ref_shape["base_short_edge"] == 2048
    assert ref_shape["multiple"] == 32
    assert ref_shape["rounding"] == "nearest"
    assert ref_shape["allow_upscale"] is True


def test_silent_video_cannot_supply_audio_derived_target_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _canonical_batch(
        task="ref2va",
        conditions=[{"type": "video", "uri": "silent.mp4", "role": "reference"}],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
        },
    )
    prequeue, _calls = _fake_probe(
        monkeypatch,
        {
            "silent.mp4": {
                "display_width": 16,
                "display_height": 9,
                "duration_seconds": 2.0,
                "video_duration_seconds": 2.0,
                "fps": 24.0,
                "frame_count": 48,
                "has_audio": False,
            }
        },
    )

    with pytest.raises(ValueError, match="condition with an audio stream"):
        prequeue.minimax_h3_prepare_for_queue(batch)


def test_reference_video_preparation_direct_scales_then_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        reference_encoding,
    )

    commands: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(),
    )
    probed = iter(
        [
            {
                "width": 1536,
                "height": 672,
                "fps": 24.0,
                "frame_count": 100,
                "sample_aspect_ratio": "1:1",
                "rotation_degrees": 0.0,
            },
            {
                "width": 1536,
                "height": 672,
                "fps": 24.0,
                "frame_count": 48,
                "sample_aspect_ratio": "1:1",
                "rotation_degrees": 0.0,
            },
        ]
    )
    monkeypatch.setattr(
        reference_encoding,
        "_ffprobe_video",
        lambda _path: next(probed),
    )

    prepared = reference_encoding.minimax_h3_prepare_reference_video(
        "/input/ref.mp4",
        target_width=1536,
        target_height=672,
        target_frame_count=48,
        workdir="/work",
        fps=24.0,
        source_facts={
            "coded_width": 1920,
            "coded_height": 1080,
            "fps": 30.0,
            "frame_count": 125,
            "sample_aspect_ratio": "16:15",
            "rotation_degrees": 90.0,
        },
    )

    assert prepared == "/work/refvid_frames48.mp4"
    assert len(commands) == 2
    resize_filter = commands[0][commands[0].index("-vf") + 1]
    assert resize_filter == "fps=24,scale=1536:672:flags=lanczos,setsar=1"
    assert "crop" not in resize_filter
    assert commands[1][commands[1].index("-frames:v") + 1] == "48"


@pytest.mark.parametrize("display_size", [(1000, 20), (20, 1000)])
def test_extreme_auto_shape_rejects_after_probe_before_ffmpeg_and_cleans_up(
    display_size: tuple[int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
        MINIMAX_H3_TEMP_DIRS_EXTRA_KEY,
    )

    batch = _canonical_batch(
        task="fl2va",
        conditions=[
            {
                "type": "image",
                "uri": "first.png",
                "role": "keyframe",
                "frame_index": 0,
            },
            {
                "type": "image",
                "uri": "last.png",
                "role": "keyframe",
                "frame_index": -1,
            },
        ],
        target={
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 5.0,
        },
    )
    material_dir = tmp_path / "localized"
    material_dir.mkdir()
    (material_dir / "source.bin").write_bytes(b"localized")
    batch.extra[MINIMAX_H3_TEMP_DIRS_EXTRA_KEY] = {"material": [str(material_dir)]}
    facts = {
        uri: {
            "display_width": display_size[0],
            "display_height": display_size[1],
        }
        for uri in ("first.png", "last.png")
    }
    prequeue, calls = _fake_probe(monkeypatch, facts)

    def forbidden_ffmpeg(*_args, **_kwargs):
        raise AssertionError("ffmpeg must not run during pre-queue resolution")

    monkeypatch.setattr(subprocess, "run", forbidden_ffmpeg)

    with pytest.raises(ValueError, match="inclusive range 1:4 to 4:1"):
        prequeue.minimax_h3_prepare_for_queue(batch)

    assert [call[0] for call in calls] == ["first.png", "last.png"]
    assert not material_dir.exists()
    assert MINIMAX_H3_TEMP_DIRS_EXTRA_KEY not in batch.extra
    assert prequeue.MINIMAX_H3_PROBE_FACTS_EXTRA_KEY not in batch.extra
    assert prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY not in batch.extra


def test_extreme_shape_api_rejection_precedes_store_pending_and_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        prequeue,
        video_adapter,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
        MiniMaxH3VideoModelAdapter,
    )

    uri = "data:image/png;base64,AA=="
    batch = _canonical_batch(
        task="fl2va",
        conditions=[
            {
                "type": "image",
                "uri": uri,
                "role": "keyframe",
                "frame_index": 0,
            },
            {
                "type": "image",
                "uri": uri,
                "role": "keyframe",
                "frame_index": -1,
            },
        ],
        target={
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 5.0,
        },
    )
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    pipeline_config = SimpleNamespace(
        task_type=SimpleNamespace(requires_image_input=lambda: False)
    )
    server_args = SimpleNamespace(
        pipeline_config=pipeline_config,
        model_path=None,
        output_path=str(output_root),
    )
    adapter = MiniMaxH3VideoModelAdapter()
    touched: list[str] = []

    # Keep this admission-order test independent of model-registry loading;
    # prepare_request is stubbed below and the real prequeue hook is the subject.
    monkeypatch.setattr(
        adapter,
        "lower_sampling_params",
        lambda *_args, **_kwargs: SimpleNamespace(
            output_path=str(output_root), save_output=True
        ),
    )

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(video_adapter.asyncio, "to_thread", fake_to_thread)

    def build_batch(**_kwargs):
        touched.append("batch-build")
        return batch

    async def record(name: str, *_args, **_kwargs):
        touched.append(name)

    monkeypatch.setattr(video_api, "get_global_server_args", lambda: server_args)
    monkeypatch.setattr(video_api, "get_video_model_adapter", lambda _cfg: adapter)
    monkeypatch.setattr(video_api, "prepare_request", build_batch)
    monkeypatch.setattr(
        prequeue,
        "minimax_h3_probe_material",
        lambda *_args, **_kwargs: {
            "local_path": "/localized.png",
            "display_width": 1000,
            "display_height": 20,
        },
    )
    monkeypatch.setattr(
        video_api.VIDEO_STORE,
        "upsert",
        lambda *_args, **_kwargs: record("store"),
    )
    monkeypatch.setattr(
        video_api.asyncio,
        "create_task",
        lambda *_args, **_kwargs: touched.append("dispatch"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ffmpeg must not run before admission")
        ),
    )
    payload = {
        "prompt": "extreme ratio",
        "task": "fl2va",
        "conditions": batch.extra["minimax_h3_canonical_request"]["conditions"],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 5.0,
        },
    }

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            video_api.create_video(
                _RawRequest(payload, multipart=False),
                extra_body=None,
                extra_params=None,
            )
        )

    assert exc_info.value.status_code == 400
    assert "inclusive range 1:4 to 4:1" in str(exc_info.value.detail)
    assert touched == ["batch-build"]


def test_multi_output_children_own_independent_derived_temp_registries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        video_adapter,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
        MINIMAX_H3_TEMP_DIRS_EXTRA_KEY,
        minimax_h3_cleanup_temp_dirs,
        minimax_h3_register_temp_dir,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
        MiniMaxH3VideoModelAdapter,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    parent_registry = {"prequeue_material": [str(source_dir)]}
    parent = SimpleNamespace(
        num_outputs_per_prompt=2,
        seed=7,
        extra={MINIMAX_H3_TEMP_DIRS_EXTRA_KEY: parent_registry},
    )
    children = [
        SimpleNamespace(extra={MINIMAX_H3_TEMP_DIRS_EXTRA_KEY: parent_registry})
        for _ in range(2)
    ]
    monkeypatch.setattr(
        video_adapter,
        "expand_request_outputs",
        lambda _batch: children,
    )

    expanded = MiniMaxH3VideoModelAdapter().expand_for_dispatch(parent)

    assert expanded == children
    assert parent.extra[MINIMAX_H3_TEMP_DIRS_EXTRA_KEY] is parent_registry
    assert all(child.extra[MINIMAX_H3_TEMP_DIRS_EXTRA_KEY] == {} for child in children)
    assert (
        children[0].extra[MINIMAX_H3_TEMP_DIRS_EXTRA_KEY]
        is not children[1].extra[MINIMAX_H3_TEMP_DIRS_EXTRA_KEY]
    )

    derived = [tmp_path / "derived-0", tmp_path / "derived-1"]
    for child, path in zip(children, derived):
        path.mkdir()
        minimax_h3_register_temp_dir(child, str(path), owner="reference_video")
    minimax_h3_cleanup_temp_dirs(children[0], owners=("reference_video",))

    assert not derived[0].exists()
    assert derived[1].exists()
    assert source_dir.exists()
