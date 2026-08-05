import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    video_adapter,
)


def _valid_probe_payload() -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1344,
                "height": 768,
                "avg_frame_rate": "24/1",
                "nb_frames": "121",
                "nb_read_frames": "121",
                "duration": "5.041667",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "32000",
                "channels": 2,
                "duration": "5.041667",
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "5.041667",
        },
    }


def _install_probe(monkeypatch, payload: dict, captured: dict | None = None) -> None:
    def fake_run(command, **kwargs):
        if captured is not None:
            captured["command"] = command
            captured["kwargs"] = kwargs
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(video_adapter.subprocess, "run", fake_run)


def _probe(monkeypatch, payload: dict, **kwargs) -> dict[str, str]:
    _install_probe(monkeypatch, payload)
    return video_adapter._probe_minimax_h3_output_fields("generated.mp4", **kwargs)


def test_valid_minimax_h3_mp4_contract_passes_and_uses_probe_metadata(
    monkeypatch,
) -> None:
    captured = {}
    _install_probe(monkeypatch, _valid_probe_payload(), captured)

    fields = video_adapter._probe_minimax_h3_output_fields(
        "misleading-extension.mkv",
        expected_frame_count=121,
        expected_size=(1344, 768),
    )

    assert fields == {"size": "1344x768", "seconds": "5.041667"}
    show_entries = captured["command"][captured["command"].index("-show_entries") + 1]
    assert "codec_name" in show_entries
    assert "pix_fmt" in show_entries
    assert "format_name" in show_entries


@pytest.mark.parametrize("format_name", ["matroska,webm", "mov,m4a,3gp,3g2,mj2", ""])
def test_minimax_h3_output_rejects_non_mp4_container(monkeypatch, format_name) -> None:
    payload = _valid_probe_payload()
    payload["format"]["format_name"] = format_name

    with pytest.raises(RuntimeError, match="container must be MP4-family"):
        _probe(monkeypatch, payload)


@pytest.mark.parametrize("codec_name", ["hevc", "vp9", "mpeg4", ""])
def test_minimax_h3_output_rejects_non_h264_video(monkeypatch, codec_name) -> None:
    payload = _valid_probe_payload()
    payload["streams"][0]["codec_name"] = codec_name

    with pytest.raises(RuntimeError, match="video codec must be h264"):
        _probe(monkeypatch, payload)


@pytest.mark.parametrize("codec_name", ["opus", "pcm_s16le", "mp3", ""])
def test_minimax_h3_output_rejects_non_aac_audio(monkeypatch, codec_name) -> None:
    payload = _valid_probe_payload()
    payload["streams"][1]["codec_name"] = codec_name

    with pytest.raises(RuntimeError, match="audio codec must be aac"):
        _probe(monkeypatch, payload)


@pytest.mark.parametrize("pixel_format", ["yuv444p", "yuv422p", "nv12", ""])
def test_minimax_h3_output_rejects_non_yuv420p_video(monkeypatch, pixel_format) -> None:
    payload = _valid_probe_payload()
    payload["streams"][0]["pix_fmt"] = pixel_format

    with pytest.raises(RuntimeError, match="pixel format must be yuv420p"):
        _probe(monkeypatch, payload)


@pytest.mark.parametrize(
    "streams",
    [
        [],
        [_valid_probe_payload()["streams"][0]],
        [_valid_probe_payload()["streams"][1]],
        _valid_probe_payload()["streams"] + [_valid_probe_payload()["streams"][0]],
        _valid_probe_payload()["streams"] + [_valid_probe_payload()["streams"][1]],
        _valid_probe_payload()["streams"]
        + [{"codec_type": "subtitle", "codec_name": "mov_text"}],
    ],
)
def test_minimax_h3_output_requires_exactly_one_video_and_audio_stream(
    monkeypatch, streams
) -> None:
    payload = _valid_probe_payload()
    payload["streams"] = deepcopy(streams)

    with pytest.raises(RuntimeError, match="exactly one video stream and one audio"):
        _probe(monkeypatch, payload)


@pytest.mark.parametrize(
    ("mutate", "kwargs", "message"),
    [
        (
            lambda payload: payload["streams"][0].update(width=1280),
            {"expected_size": (1344, 768)},
            "size does not match",
        ),
        (
            lambda payload: payload["streams"][0].update(avg_frame_rate="25/1"),
            {},
            "must be 24 fps",
        ),
        (
            lambda payload: payload["streams"][1].update(sample_rate="44100"),
            {},
            "must be 32000 Hz",
        ),
        (
            lambda payload: payload["streams"][1].update(channels=1),
            {},
            "must be stereo",
        ),
        (
            lambda payload: payload["streams"][0].update(nb_read_frames="120"),
            {"expected_frame_count": 121},
            "frame count.*expected 121, got 120",
        ),
        (
            lambda payload: payload["streams"][1].update(duration="5.5"),
            {},
            "duration drift exceeds",
        ),
    ],
)
def test_minimax_h3_output_preserves_existing_media_contract_failures(
    monkeypatch, mutate, kwargs, message
) -> None:
    payload = _valid_probe_payload()
    mutate(payload)

    with pytest.raises(RuntimeError, match=message):
        _probe(monkeypatch, payload, **kwargs)
