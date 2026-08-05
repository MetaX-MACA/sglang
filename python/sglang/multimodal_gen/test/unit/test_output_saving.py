import json
import shutil
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image

import sglang.multimodal_gen.runtime.entrypoints.utils as output_utils
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.mova import MOVAPipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample


def _rgb_frame() -> np.ndarray:
    return np.array(
        [
            [[0, 32, 255], [64, 128, 192], [255, 224, 16]],
            [[9, 17, 33], [127, 128, 129], [240, 12, 88]],
        ],
        dtype=np.uint8,
    )


def test_batched_stereo_audio_selection_and_normalization_preserve_channels():
    left = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float32)
    right = torch.tensor([0.5, 0.25, 0.0, -0.25], dtype=torch.float32)
    batched_audio = torch.stack((left, right)).unsqueeze(0)  # [B, C, L]

    selected = output_utils.select_output_audio(batched_audio, output_idx=0)
    normalized = output_utils._normalize_audio_to_numpy(selected)

    assert tuple(selected.shape) == (2, 4)
    assert normalized.shape == (4, 2)
    np.testing.assert_array_equal(normalized[:, 0], left.numpy())
    np.testing.assert_array_equal(normalized[:, 1], right.numpy())


def test_required_audio_delivery_is_a_read_only_pipeline_capability():
    base = PipelineConfig()

    assert not base.requires_audio_output
    assert MiniMaxH3PipelineConfig().requires_audio_output
    assert not MOVAPipelineConfig().requires_audio_output
    assert not LTX2PipelineConfig().requires_audio_output
    with pytest.raises(AttributeError):
        base.requires_audio_output = True


def test_strict_audio_mux_propagates_failure(tmp_path, monkeypatch):
    output_path = tmp_path / "strict.mp4"
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    audio = torch.zeros((2, 320), dtype=torch.float32)

    def fake_video_write(path, *_args, **_kwargs):
        with open(path, "wb") as handle:
            handle.write(b"silent-video")

    monkeypatch.setattr(output_utils.imageio, "mimsave", fake_video_write)
    monkeypatch.setattr(
        output_utils,
        "_resolve_ffmpeg_exe",
        lambda: (_ for _ in ()).throw(RuntimeError("ffmpeg unavailable")),
    )

    with pytest.raises(RuntimeError, match="failed to mux generated audio"):
        post_process_sample(
            (frames, audio),
            DataType.VIDEO,
            fps=24,
            save_file_path=str(output_path),
            audio_sample_rate=32000,
            strict_audio_mux=True,
        )
    assert not output_path.exists()


def test_strict_audio_mux_rejects_missing_audio(tmp_path, monkeypatch):
    output_path = tmp_path / "missing-audio.mp4"
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    def fake_video_write(path, *_args, **_kwargs):
        with open(path, "wb") as handle:
            handle.write(b"silent-video")

    monkeypatch.setattr(output_utils.imageio, "mimsave", fake_video_write)

    with pytest.raises(RuntimeError, match="requires generated audio"):
        post_process_sample(
            frames,
            DataType.VIDEO,
            fps=24,
            save_file_path=str(output_path),
            audio_sample_rate=32000,
            strict_audio_mux=True,
        )
    assert not output_path.exists()


def test_strict_audio_mux_rejects_invalid_audio_and_removes_silent_video(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "invalid-audio.mp4"
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    def fake_video_write(path, *_args, **_kwargs):
        with open(path, "wb") as handle:
            handle.write(b"silent-video")

    monkeypatch.setattr(output_utils.imageio, "mimsave", fake_video_write)

    with pytest.raises(TypeError, match="cannot materialize generated audio"):
        post_process_sample(
            (frames, "not-audio"),
            DataType.VIDEO,
            fps=24,
            save_file_path=str(output_path),
            audio_sample_rate=32000,
            strict_audio_mux=True,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.empty((0, 2), dtype=np.float32), 32000, "at least one audio sample"),
        (np.zeros((2, 2, 320), dtype=np.float32), 32000, "samples x 2"),
        (np.zeros((320, 1), dtype=np.float32), 32000, "samples x 2"),
        (np.full((320, 2), np.nan, dtype=np.float32), 32000, "finite audio"),
        (np.full((320, 2), np.inf, dtype=np.float32), 32000, "finite audio"),
        (np.full((320, 2), -np.inf, dtype=np.float32), 32000, "finite audio"),
        (torch.full((2, 320), torch.inf), 32000, "finite audio"),
        (torch.full((2, 320), -torch.inf), 32000, "finite audio"),
        (np.zeros((320, 2), dtype=np.float32), 44100, "32000 Hz"),
        (np.zeros((320, 2), dtype=np.float32), 32000, "duration drift"),
    ],
    ids=[
        "empty",
        "three-dimensional",
        "mono",
        "numpy-nan",
        "numpy-positive-inf",
        "numpy-negative-inf",
        "torch-positive-inf",
        "torch-negative-inf",
        "sample-rate",
        "duration",
    ],
)
def test_strict_audio_mux_rejects_malformed_audio_without_publishing_mp4(
    tmp_path,
    monkeypatch,
    audio,
    sample_rate,
    message,
):
    output_path = tmp_path / "invalid-contract.mp4"
    frame_count = 121 if message == "duration drift" else 2
    frames = np.zeros((frame_count, 16, 16, 3), dtype=np.uint8)

    def fake_video_write(path, *_args, **_kwargs):
        with open(path, "wb") as handle:
            handle.write(b"silent-video")

    monkeypatch.setattr(output_utils.imageio, "mimsave", fake_video_write)

    with pytest.raises(ValueError, match=message):
        post_process_sample(
            (frames, audio),
            DataType.VIDEO,
            fps=24,
            save_file_path=str(output_path),
            audio_sample_rate=sample_rate,
            strict_audio_mux=True,
            output_audio_sample_rate=32000,
            output_audio_channels=2,
            output_av_drift_tolerance_s=0.25,
        )

    assert not output_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_strict_audio_failure_preserves_preexisting_output(tmp_path, monkeypatch):
    output_path = tmp_path / "existing.mp4"
    output_path.write_bytes(b"previous-valid-output")
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    audio = np.zeros((320, 2), dtype=np.float32)

    monkeypatch.setattr(
        output_utils.imageio,
        "mimsave",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("video writer failed before publish")
        ),
    )

    with pytest.raises(RuntimeError, match="video writer failed"):
        post_process_sample(
            (frames, audio),
            DataType.VIDEO,
            fps=24,
            save_file_path=str(output_path),
            audio_sample_rate=32000,
            strict_audio_mux=True,
        )

    assert output_path.read_bytes() == b"previous-valid-output"
    assert list(tmp_path.iterdir()) == [output_path]


def test_strict_audio_output_is_atomically_published(tmp_path, monkeypatch):
    output_path = tmp_path / "atomic.mp4"
    output_path.write_bytes(b"previous-output")
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    audio = np.zeros((320, 2), dtype=np.float32)

    def fake_video_write(path, *_args, **_kwargs):
        assert path != str(output_path)
        with open(path, "wb") as handle:
            handle.write(b"silent-video")

    def fake_mux(*, save_file_path, audio_np, sample_rate, ffmpeg_exe):
        assert save_file_path != str(output_path)
        assert audio_np.shape == (320, 2)
        assert sample_rate == 32000
        assert ffmpeg_exe == "ffmpeg"
        with open(save_file_path, "ab") as handle:
            handle.write(b"+stereo-audio")

    monkeypatch.setattr(output_utils.imageio, "mimsave", fake_video_write)
    monkeypatch.setattr(output_utils, "_resolve_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(output_utils, "_mux_audio_np_into_mp4", fake_mux)

    post_process_sample(
        (frames, audio),
        DataType.VIDEO,
        fps=24,
        save_file_path=str(output_path),
        audio_sample_rate=32000,
        strict_audio_mux=True,
    )

    assert output_path.read_bytes() == b"silent-video+stereo-audio"
    assert list(tmp_path.iterdir()) == [output_path]


@pytest.mark.parametrize(
    "pipeline_config",
    [MOVAPipelineConfig(), LTX2PipelineConfig()],
    ids=["mova", "ltx2"],
)
def test_non_minimax_h3_audio_mux_remains_best_effort(
    tmp_path, monkeypatch, pipeline_config
):
    output_path = tmp_path / "legacy.mp4"
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    audio = torch.zeros((2, 320), dtype=torch.float32)

    def fake_video_write(path, *_args, **_kwargs):
        with open(path, "wb") as handle:
            handle.write(b"silent-video")

    monkeypatch.setattr(output_utils.imageio, "mimsave", fake_video_write)
    monkeypatch.setattr(
        output_utils,
        "_resolve_ffmpeg_exe",
        lambda: (_ for _ in ()).throw(RuntimeError("ffmpeg unavailable")),
    )

    post_process_sample(
        (frames, audio),
        DataType.VIDEO,
        fps=24,
        save_file_path=str(output_path),
        audio_sample_rate=32000,
        strict_audio_mux=pipeline_config.requires_audio_output,
    )
    assert output_path.read_bytes() == b"silent-video"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or output_utils.scipy_wavfile is None,
    reason="ffmpeg, ffprobe, and scipy are required for the MP4 audio contract",
)
def test_video_mp4_mux_preserves_stereo_channels(tmp_path):
    sample_rate = 32000
    fps = 20
    sample_count = sample_rate // 10
    time = torch.arange(sample_count, dtype=torch.float32) / sample_rate
    stereo = torch.stack(
        (
            0.25 * torch.sin(2 * torch.pi * 440 * time),
            0.25 * torch.sin(2 * torch.pi * 660 * time),
        )
    )
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    output_path = tmp_path / "stereo.mp4"

    post_process_sample(
        (frames, stereo),
        DataType.VIDEO,
        fps=fps,
        save_file_path=str(output_path),
        audio_sample_rate=sample_rate,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]

    assert output_path.exists()
    assert streams == [{"channels": 2}]


@pytest.mark.parametrize("output_compression", [None, 0, 75])
def test_png_output_saving_preserves_pixels(tmp_path, output_compression):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    frames = post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert output_path.exists()
    np.testing.assert_array_equal(frames[0], frame)
    np.testing.assert_array_equal(np.array(Image.open(output_path)), frame)


@pytest.mark.parametrize(
    ("output_compression", "expected_compress_level"), [(None, 1), (0, 0), (75, 1)]
)
def test_png_output_saving_uses_fast_pillow_path(
    tmp_path, monkeypatch, output_compression, expected_compress_level
):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    def fail_imageio_imwrite(*args, **kwargs):
        raise AssertionError("PNG output should use Pillow's PNG fast path")

    original_save = Image.Image.save
    save_calls = []

    def save_spy(self, fp, format=None, **params):
        save_calls.append((format, params.get("compress_level")))
        return original_save(self, fp, format=format, **params)

    monkeypatch.setattr(output_utils.imageio, "imwrite", fail_imageio_imwrite)
    monkeypatch.setattr(Image.Image, "save", save_spy)

    post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert save_calls == [("PNG", expected_compress_level)]
