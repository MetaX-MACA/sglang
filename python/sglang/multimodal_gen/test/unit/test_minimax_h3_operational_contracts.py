import asyncio
import base64
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    material_io,
    reference_encoding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    video_adapter as minimax_h3_video_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
    MiniMaxH3VideoModelAdapter,
)

_MINIMAX_H3_VIDEO_ADAPTER = MiniMaxH3VideoModelAdapter()


def _batch():
    return SimpleNamespace(extra={})


def _use_minimax_h3_delivery_capability(monkeypatch) -> None:
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(
            pipeline_config=SimpleNamespace(requires_audio_output=True)
        ),
    )


class _HttpHeaders(dict):
    def get_content_type(self):
        value = self.get("Content-Type")
        return value.split(";", 1)[0] if value else "application/octet-stream"


class _HttpResponse:
    def __init__(self, payload: bytes, *, headers=None, max_chunk=None):
        self.payload = payload
        self.headers = _HttpHeaders(headers or {})
        self.max_chunk = max_chunk
        self.offset = 0
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        effective = size if self.max_chunk is None else min(size, self.max_chunk)
        chunk = self.payload[self.offset : self.offset + effective]
        self.offset += len(chunk)
        return chunk

    def iter_bytes(self, size: int):
        while True:
            chunk = self.read(size)
            if not chunk:
                return
            yield chunk


@pytest.fixture(autouse=True)
def _stub_material_metadata_probe(monkeypatch):
    # Localization tests use intentionally tiny non-media payloads; actual
    # format/probe behavior is covered by the dedicated prequeue tests.
    monkeypatch.setattr(
        material_io, "_validate_localized_media", lambda *_a, **_k: {}
    )

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    # torch-backed asyncio.run() waits forever for this environment's default
    # executor at loop shutdown; the sync hook behavior is tested inline here.
    monkeypatch.setattr(minimax_h3_video_adapter.asyncio, "to_thread", _inline_to_thread)


def test_data_uri_is_streamed_localized_once_and_cleaned(monkeypatch):
    monkeypatch.setattr(material_io, "MINIMAX_H3_BASE64_DECODE_CHUNK_CHARS", 4)
    batch = _batch()
    uri = "data:image/png;base64," + base64.b64encode(b"png-bytes").decode()

    first = material_io.minimax_h3_localize_material_uri(
        batch, uri, condition_type="image", condition_index=0
    )
    second = material_io.minimax_h3_localize_material_uri(
        batch, uri, condition_type="image", condition_index=0
    )

    assert first == second
    assert Path(first).read_bytes() == b"png-bytes"
    workdir = Path(first).parent
    material_io.minimax_h3_cleanup_temp_dirs(batch, owners=("material",))
    assert not workdir.exists()
    assert material_io.MINIMAX_H3_MATERIAL_CACHE_EXTRA_KEY not in batch.extra


def test_tar_member_uri_is_streamed_with_member_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(material_io, "MINIMAX_H3_HTTP_READ_CHUNK_BYTES", 3)
    payload = b"prefix" + b"image-payload" + b"suffix"
    tar_path = tmp_path / "fixture.tar"
    tar_path.write_bytes(payload)
    header = {
        "schema": "sglang.tar_member_ref/v1",
        "member": "nested/reference.webp",
        "offset_data": 6,
        "size": len(b"image-payload"),
    }
    encoded = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    uri = f"tar+offset://{tar_path}:{encoded}"
    batch = _batch()

    localized = material_io.minimax_h3_localize_material_uri(
        batch, uri, condition_type="image", condition_index=2
    )

    assert localized.endswith(".webp")
    assert Path(localized).read_bytes() == b"image-payload"
    material_io.minimax_h3_cleanup_temp_dirs(batch)


def test_file_uri_is_unquoted_without_copy(tmp_path):
    source = tmp_path / "image with spaces.png"
    source.write_bytes(b"image")

    localized = material_io.minimax_h3_localize_material_uri(
        _batch(),
        source.as_uri(),
        condition_type="image",
        condition_index=0,
    )

    assert Path(localized) == source


def test_reference_video_and_material_owners_cleanup_independently(tmp_path):
    batch = _batch()
    material_dir = tmp_path / "material"
    video_dir = tmp_path / "reference_video"
    material_dir.mkdir()
    video_dir.mkdir()
    material_io.minimax_h3_register_temp_dir(batch, str(material_dir), owner="material")
    material_io.minimax_h3_register_temp_dir(
        batch, str(video_dir), owner="reference_video"
    )

    material_io.minimax_h3_cleanup_temp_dirs(batch, owners=("reference_video",))
    assert material_dir.exists()
    assert not video_dir.exists()

    material_io.minimax_h3_cleanup_temp_dirs(batch, owners=("material",))
    assert not material_dir.exists()


def test_http_material_streams_in_chunks_and_is_cleaned(monkeypatch):
    response = _HttpResponse(
        b"chunked-image",
        headers={"Content-Type": "image/png"},
        max_chunk=3,
    )
    monkeypatch.setattr(
        material_io.urllib.request,
        "urlopen",
        lambda _url, timeout: response,
    )
    batch = _batch()

    localized = material_io.minimax_h3_localize_material_uri(
        batch,
        "https://example.invalid/reference",
        condition_type="image",
        condition_index=0,
    )

    assert Path(localized).read_bytes() == b"chunked-image"
    assert response.read_calls > 2
    workdir = Path(localized).parent
    material_io.minimax_h3_cleanup_temp_dirs(batch)
    assert not workdir.exists()


def test_audio_material_chain_preserves_rate_and_normalizes_channels(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout='{"streams":[{"channels":6}]}')
        return SimpleNamespace(stdout="")

    fake_torchaudio = SimpleNamespace(load=lambda _path: (torch.zeros((2, 4)), 48000))
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setattr(subprocess, "run", fake_run)

    waveform, source_rate = reference_encoding._load_waveform(
        "surround.wav", material_chain="audio"
    )

    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert waveform.shape == (2, 4)
    assert source_rate == 48000
    assert ffmpeg[ffmpeg.index("-ac") + 1] == "2"
    assert "-ar" not in ffmpeg


def test_video_audio_material_chain_extracts_44100hz_stereo(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="")

    fake_torchaudio = SimpleNamespace(load=lambda _path: (torch.zeros((2, 4)), 44100))
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setattr(subprocess, "run", fake_run)

    _, source_rate = reference_encoding._load_waveform(
        "reference.mp4", material_chain="video_audio.reference_preserve"
    )

    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert source_rate == 44100
    assert ffmpeg[ffmpeg.index("-ac") + 1] == "2"
    assert ffmpeg[ffmpeg.index("-ar") + 1] == "44100"


class _CanonicalSampling:
    task = "t2va"
    width = 896
    height = 512
    num_frames = 96
    fps = 24

    def output_file_path(self):
        return "/tmp/generated.mp4"

    def build_request_extra(self):
        return {
            "minimax_h3_canonical_request": {
                "schema": "minimax_h3.request/v1",
                "task": "t2va",
                "prompt": "test",
                "seed": 7,
                "flow_shift": 12.0,
                "audio_flow_shift": 3.0,
                "conditions": [],
                "target": {
                    "short_edge": 768,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 8.7,
                },
            }
        }


def _job_batch(sampling):
    return SimpleNamespace(
        width=sampling.width,
        height=sampling.height,
        num_frames=sampling.num_frames,
        fps=sampling.fps,
        extra=sampling.build_request_extra(),
        output_file_path=sampling.output_file_path,
    )


def test_video_job_uses_canonical_minimax_h3_target():
    job = video_api._video_job_from_batch(
        "video-1",
        SimpleNamespace(model="minimax_h3"),
        _job_batch(_CanonicalSampling()),
        _MINIMAX_H3_VIDEO_ADAPTER,
    )

    assert job["size"] == "1344x768"
    assert job["seconds"] == "8.708333"


def test_final_minimax_h3_probe_requires_audio_and_reports_actual_shape(monkeypatch):
    probe_payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "8.708333"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1344,
                "height": 768,
                "avg_frame_rate": "24/1",
                "nb_frames": "209",
                "duration": "8.708333",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": "8.708333",
                "sample_rate": "32000",
                "channels": 2,
            },
        ],
    }
    monkeypatch.setattr(
        minimax_h3_video_adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(probe_payload)),
    )

    fields = minimax_h3_video_adapter._probe_minimax_h3_output_fields(
        "generated.mp4",
        expected_frame_count=209,
        expected_size=(1344, 768),
    )

    assert fields == {"size": "1344x768", "seconds": "8.708333"}
    with pytest.raises(RuntimeError, match="size.*expected 768x1344"):
        minimax_h3_video_adapter._probe_minimax_h3_output_fields(
            "generated.mp4",
            expected_frame_count=209,
            expected_size=(768, 1344),
        )


@pytest.mark.parametrize(
    ("video_rate", "audio_sample_rate", "audio_channels", "message"),
    [
        ("25/1", "32000", 2, "frame rate must be 24 fps"),
        ("24/1", "44100", 2, "sample rate must be 32000 Hz"),
        ("24/1", "32000", 1, "audio must be stereo"),
    ],
)
def test_final_minimax_h3_probe_rejects_wrong_delivery_format(
    monkeypatch,
    video_rate,
    audio_sample_rate,
    audio_channels,
    message,
):
    probe_payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "8.708333"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1344,
                "height": 768,
                "avg_frame_rate": video_rate,
                "nb_frames": "209",
                "duration": "8.708333",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": "8.708333",
                "sample_rate": audio_sample_rate,
                "channels": audio_channels,
            },
        ],
    }
    monkeypatch.setattr(
        minimax_h3_video_adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(probe_payload)),
    )

    with pytest.raises(RuntimeError, match=message):
        minimax_h3_video_adapter._probe_minimax_h3_output_fields("generated.mp4")


def test_final_minimax_h3_probe_rejects_truncated_or_desynchronized_output(monkeypatch):
    probe_payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "8.708333"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1344,
                "height": 768,
                "avg_frame_rate": "24/1",
                "nb_read_frames": "208",
                "duration": "8.666667",
            },
            {
                "codec_type": "audio",
                "duration": "8.2",
                "sample_rate": "32000",
                "channels": 2,
            },
        ],
    }
    monkeypatch.setattr(
        minimax_h3_video_adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(probe_payload)),
    )

    with pytest.raises(RuntimeError, match="frame count.*expected 209, got 208"):
        minimax_h3_video_adapter._probe_minimax_h3_output_fields(
            "generated.mp4", expected_frame_count=209
        )

    probe_payload["streams"][0]["nb_read_frames"] = "209"
    with pytest.raises(RuntimeError, match="duration drift exceeds"):
        minimax_h3_video_adapter._probe_minimax_h3_output_fields(
            "generated.mp4", expected_frame_count=209
        )


def test_minimax_h3_missing_audio_marks_job_failed_and_cleans_api_dirs(
    tmp_path, monkeypatch
):
    _use_minimax_h3_delivery_capability(monkeypatch)
    output_path = tmp_path / "silent.mp4"
    output_path.write_bytes(b"silent")
    api_temp = tmp_path / "api_temp"
    api_temp.mkdir()
    updates = []

    async def fake_process(*_args, **_kwargs):
        return [str(output_path)], SimpleNamespace()

    async def fake_update(job_id, fields):
        updates.append((job_id, fields))

    monkeypatch.setattr(video_api, "process_generation_batch", fake_process)
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "_probe_minimax_h3_output_fields",
        lambda _path, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("generated MiniMax H3 MP4 has no audio stream")
        ),
    )
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update)
    batch = SimpleNamespace(extra={"minimax_h3_canonical_request": {}})

    asyncio.run(
        video_api._dispatch_job_async(
            "video-1",
            batch,
            adapter=_MINIMAX_H3_VIDEO_ADAPTER,
            temp_dirs=[str(api_temp)],
        )
    )

    assert not api_temp.exists()
    assert not output_path.exists()
    assert updates[0][0] == "video-1"
    assert updates[0][1]["status"] == "failed"
    assert updates[0][1]["error"] == {
        "message": "generated MiniMax H3 MP4 has no audio stream"
    }
    assert updates[0][1]["file_paths"] is None


def test_minimax_h3_video_job_probes_and_publishes_every_persistent_output(
    tmp_path, monkeypatch
):
    _use_minimax_h3_delivery_capability(monkeypatch)
    output_paths = [tmp_path / "video_0.mp4", tmp_path / "video_1.mp4"]
    for index, path in enumerate(output_paths):
        path.write_bytes(f"video-{index}".encode())
    probed = []
    dispatched = []
    updates = []
    expanded = [
        SimpleNamespace(extra={"minimax_h3_canonical_request": {}}),
        SimpleNamespace(extra={"minimax_h3_canonical_request": {}}),
    ]

    async def fake_process(_client, dispatch_batch):
        dispatched.append(dispatch_batch)
        return [str(path) for path in output_paths], SimpleNamespace(
            peak_memory_mb=None,
            metrics=None,
        )

    async def fake_update(job_id, fields):
        updates.append((job_id, fields))

    async def fake_cleanup(_batch):
        return None

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(video_api, "process_generation_batch", fake_process)
    monkeypatch.setattr(
        minimax_h3_video_adapter, "expand_request_outputs", lambda _batch: expanded
    )
    monkeypatch.setattr(
        minimax_h3_video_adapter,
        "_probe_minimax_h3_output_fields",
        lambda path, **kwargs: (
            probed.append((path, kwargs)) or {"size": "1344x768", "seconds": "7.875"}
        ),
    )
    monkeypatch.setattr(video_api.cloud_storage, "is_enabled", lambda: False)
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update)
    monkeypatch.setattr(_MINIMAX_H3_VIDEO_ADAPTER, "cleanup_request", fake_cleanup)
    monkeypatch.setattr(minimax_h3_video_adapter.asyncio, "to_thread", fake_to_thread)

    asyncio.run(
        video_api._dispatch_job_async(
            "video-many",
            SimpleNamespace(
                extra={"minimax_h3_canonical_request": {"task": "t2va"}},
                num_outputs_per_prompt=2,
            ),
            adapter=_MINIMAX_H3_VIDEO_ADAPTER,
            output_persistent=True,
        )
    )

    fields = updates[0][1]
    expected = [str(path.resolve()) for path in output_paths]
    assert dispatched == [expanded]
    assert [path for path, _ in probed] == [str(path) for path in output_paths]
    assert fields["status"] == "completed"
    assert fields["size"] == "1344x768"
    assert fields["seconds"] == "7.875"
    assert fields["file_path"] == expected[0]
    assert fields["file_paths"] == expected
    assert fields["num_outputs"] == 2
    assert fields["url"] is None
    assert fields["urls"] is None


def test_minimax_h3_video_job_fails_when_scheduler_drops_an_output(tmp_path, monkeypatch):
    output_path = tmp_path / "video_0.mp4"
    output_path.write_bytes(b"video")
    dispatched = []
    updates = []
    expanded = [
        SimpleNamespace(extra={"minimax_h3_canonical_request": {}}),
        SimpleNamespace(extra={"minimax_h3_canonical_request": {}}),
    ]

    async def fake_process(_client, dispatch_batch):
        dispatched.append(dispatch_batch)
        return [str(output_path)], SimpleNamespace()

    async def fake_update(job_id, fields):
        updates.append((job_id, fields))

    monkeypatch.setattr(video_api, "process_generation_batch", fake_process)
    monkeypatch.setattr(
        minimax_h3_video_adapter, "expand_request_outputs", lambda _batch: expanded
    )
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update)

    asyncio.run(
        video_api._dispatch_job_async(
            "video-missing-output",
            SimpleNamespace(
                extra={"minimax_h3_canonical_request": {}},
                num_outputs_per_prompt=2,
            ),
            adapter=_MINIMAX_H3_VIDEO_ADAPTER,
        )
    )

    assert dispatched == [expanded]
    assert updates[0][0] == "video-missing-output"
    assert updates[0][1]["status"] == "failed"
    assert updates[0][1]["error"] == {
        "message": "MiniMax H3 video generation produced 1 output files, expected 2"
    }
    assert updates[0][1]["file_paths"] is None


def test_minimax_h3_video_job_rejects_inconsistent_output_metadata(tmp_path, monkeypatch):
    _use_minimax_h3_delivery_capability(monkeypatch)
    output_paths = [tmp_path / "video_0.mp4", tmp_path / "video_1.mp4"]
    for path in output_paths:
        path.write_bytes(b"video")
    dispatched = []
    updates = []
    expanded = [
        SimpleNamespace(extra={"minimax_h3_canonical_request": {}}),
        SimpleNamespace(extra={"minimax_h3_canonical_request": {}}),
    ]

    async def fake_process(_client, dispatch_batch):
        dispatched.append(dispatch_batch)
        return [str(path) for path in output_paths], SimpleNamespace()

    async def fake_update(job_id, fields):
        updates.append((job_id, fields))

    def fake_probe(path, **_kwargs):
        size = "1344x768" if path.endswith("video_0.mp4") else "768x1344"
        return {"size": size, "seconds": "8.708333"}

    monkeypatch.setattr(video_api, "process_generation_batch", fake_process)
    monkeypatch.setattr(
        minimax_h3_video_adapter, "expand_request_outputs", lambda _batch: expanded
    )
    monkeypatch.setattr(
        minimax_h3_video_adapter, "_probe_minimax_h3_output_fields", fake_probe
    )
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update)

    asyncio.run(
        video_api._dispatch_job_async(
            "video-inconsistent-output",
            SimpleNamespace(
                extra={"minimax_h3_canonical_request": {}},
                num_outputs_per_prompt=2,
            ),
            adapter=_MINIMAX_H3_VIDEO_ADAPTER,
        )
    )

    assert dispatched == [expanded]
    assert updates[0][1]["status"] == "failed"
    assert "inconsistent media metadata" in updates[0][1]["error"]["message"]


def test_video_job_uploads_every_output(tmp_path, monkeypatch):
    output_paths = [tmp_path / "video_0.mp4", tmp_path / "video_1.mp4"]
    for path in output_paths:
        path.write_bytes(b"video")
    uploaded = []
    updates = []

    async def fake_process(*_args, **_kwargs):
        return [str(path) for path in output_paths], SimpleNamespace(
            peak_memory_mb=None,
            metrics=None,
        )

    async def fake_upload(path):
        uploaded.append(path)
        return f"https://storage.invalid/{Path(path).name}"

    async def fake_update(job_id, fields):
        updates.append((job_id, fields))

    monkeypatch.setattr(video_api, "process_generation_batch", fake_process)
    monkeypatch.setattr(video_api.cloud_storage, "is_enabled", lambda: True)
    monkeypatch.setattr(video_api.cloud_storage, "upload_and_cleanup", fake_upload)
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update)

    asyncio.run(
        video_api._dispatch_job_async(
            "video-cloud",
            SimpleNamespace(extra={}, num_outputs_per_prompt=1),
            output_persistent=False,
        )
    )

    urls = [f"https://storage.invalid/{path.name}" for path in output_paths]
    fields = updates[0][1]
    assert uploaded == [str(path) for path in output_paths]
    assert fields["url"] == urls[0]
    assert fields["urls"] == urls
    assert fields["file_path"] is None
    assert fields["file_paths"] is None


def test_video_job_falls_back_to_persistent_output_when_cloud_upload_fails(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "video.mp4"
    output_path.write_bytes(b"video")
    updates = []

    async def fake_process(*_args, **_kwargs):
        return [str(output_path)], SimpleNamespace(
            peak_memory_mb=None,
            metrics=None,
        )

    async def fake_upload(_path):
        return None

    async def fake_update(job_id, fields):
        updates.append((job_id, fields))

    monkeypatch.setattr(video_api, "process_generation_batch", fake_process)
    monkeypatch.setattr(video_api.cloud_storage, "is_enabled", lambda: True)
    monkeypatch.setattr(video_api.cloud_storage, "upload_and_cleanup", fake_upload)
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update)

    asyncio.run(
        video_api._dispatch_job_async(
            "video-cloud-fallback",
            SimpleNamespace(extra={}, num_outputs_per_prompt=1),
            output_persistent=True,
        )
    )

    fields = updates[0][1]
    assert fields["status"] == "completed"
    assert fields["url"] is None
    assert fields["file_path"] == str(output_path.resolve())
    assert fields["file_paths"] == [str(output_path.resolve())]


def test_video_job_fails_without_a_durable_destination(tmp_path, monkeypatch):
    api_temp = tmp_path / "api_temp"
    api_temp.mkdir()
    output_path = api_temp / "ephemeral.mp4"
    output_path.write_bytes(b"video")
    updates = []

    async def fake_process(*_args, **_kwargs):
        return [str(output_path)], SimpleNamespace()

    async def fake_update(job_id, fields):
        updates.append((job_id, fields))

    monkeypatch.setattr(video_api, "process_generation_batch", fake_process)
    monkeypatch.setattr(video_api.cloud_storage, "is_enabled", lambda: False)
    monkeypatch.setattr(video_api.VIDEO_STORE, "update_fields", fake_update)

    asyncio.run(
        video_api._dispatch_job_async(
            "video-ephemeral",
            SimpleNamespace(extra={}, num_outputs_per_prompt=1),
            temp_dirs=[str(api_temp)],
            output_persistent=False,
        )
    )

    assert not api_temp.exists()
    assert updates[0][0] == "video-ephemeral"
    assert updates[0][1]["status"] == "failed"
    assert updates[0][1]["error"] == {
        "message": (
            "generated video has no durable destination; configure an "
            "output path or enable cloud storage"
        )
    }
    assert updates[0][1]["file_paths"] is None


def test_video_content_selects_output_index(tmp_path, monkeypatch):
    output_paths = [tmp_path / "video_0.mp4", tmp_path / "video_1.mp4"]
    for path in output_paths:
        path.write_bytes(b"video")

    async def fake_get(_video_id):
        return {
            "status": "completed",
            "file_path": str(output_paths[0]),
            "file_paths": [str(path) for path in output_paths],
            "num_outputs": 2,
        }

    monkeypatch.setattr(video_api.VIDEO_STORE, "get", fake_get)

    response = asyncio.run(
        video_api.download_video_content("video-many", variant="video", output_index=1)
    )

    assert Path(response.path) == output_paths[1]
    with pytest.raises(video_api.HTTPException, match="Video output not found"):
        asyncio.run(
            video_api.download_video_content(
                "video-many", variant="video", output_index=2
            )
        )


def test_video_content_selects_cloud_url_and_rejects_unknown_variant(monkeypatch):
    urls = [
        "https://storage.invalid/video_0.mp4",
        "https://storage.invalid/video_1.mp4",
    ]

    async def fake_get(_video_id):
        return {"status": "completed", "url": urls[0], "urls": urls}

    monkeypatch.setattr(video_api.VIDEO_STORE, "get", fake_get)

    with pytest.raises(video_api.HTTPException) as cloud_error:
        asyncio.run(
            video_api.download_video_content(
                "video-cloud", variant="video", output_index=1
            )
        )
    assert cloud_error.value.status_code == 400
    assert urls[1] in cloud_error.value.detail

    with pytest.raises(video_api.HTTPException) as variant_error:
        asyncio.run(
            video_api.download_video_content(
                "video-cloud", variant="thumbnail", output_index=0
            )
        )
    assert variant_error.value.status_code == 400
    assert "Unsupported video content variant" in variant_error.value.detail


def test_video_content_preserves_in_progress_error_for_unwritten_path(monkeypatch):
    async def fake_get(_video_id):
        return {"status": "in_progress", "file_path": "/not-yet-written.mp4"}

    monkeypatch.setattr(video_api.VIDEO_STORE, "get", fake_get)

    with pytest.raises(video_api.HTTPException) as error:
        asyncio.run(
            video_api.download_video_content(
                "video-pending", variant=None, output_index=0
            )
        )
    assert error.value.status_code == 404
    assert error.value.detail == "Generation is still in-progress"


def test_video_content_does_not_report_failed_job_as_in_progress(monkeypatch):
    async def fake_get(_video_id):
        return {
            "status": "failed",
            "error": {"message": "inference failed"},
        }

    monkeypatch.setattr(video_api.VIDEO_STORE, "get", fake_get)

    with pytest.raises(video_api.HTTPException) as error:
        asyncio.run(
            video_api.download_video_content(
                "video-failed", variant=None, output_index=0
            )
        )
    assert error.value.status_code == 404
    assert error.value.detail == "Video output not found"
