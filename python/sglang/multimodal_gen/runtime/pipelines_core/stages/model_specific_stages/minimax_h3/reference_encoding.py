# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 ref2va reference-material encoding.

Encoding recipes for user-provided reference materials:

- image reference: independent 2048px short-edge resize with upscale enabled,
  LANCZOS, and nearest-32 dimensions, then the SAME keyframe tokenizer recipe
  as fl2va (seed-42 sampled encode, normalize, [1,2,2] patchify);
- audio reference: the audio material chain (pure
  audio is losslessly normalized
  to stereo; video soundtracks are extracted as 44.1 kHz stereo), then a
  single resample to 32 kHz,
  audio VAE posterior MEAN (encoder -> optional pre_block -> mean_proj; no
  sampling), canonical [2, T, 32], normalize with loader-injected audio stats,
  channel-major rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEArchConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
    minimax_h3_scoped_encode_rng,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.media_prep import (
    minimax_h3_ffprobe_video,
    minimax_h3_materialize_normalized_canvas,
    minimax_h3_verify_normalized_video_meta,
    minimax_h3_x264_transcode,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SUPPORTED_FPS,
)

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_AUDIO_CHANNELS = 2


class _AudioVAEDeterminismContext:
    """Scoped determinism config for the audio encode.

    Disables TF32, forces deterministic algorithms, DISABLES cuDNN entirely
    for the encode (convs run on the fallback kernels), and pins SDP to the
    math backend. This configuration is required for a deterministic encode.
    Everything is restored on exit
    so the decode path keeps its own configuration.
    """

    def __enter__(self):
        b = torch.backends
        self._saved = (
            b.cuda.matmul.allow_tf32,
            b.cudnn.allow_tf32,
            b.cudnn.benchmark,
            b.cudnn.deterministic,
            b.cudnn.enabled,
            b.cuda.flash_sdp_enabled(),
            b.cuda.mem_efficient_sdp_enabled(),
            b.cuda.math_sdp_enabled(),
        )
        b.cuda.matmul.allow_tf32 = False
        b.cudnn.allow_tf32 = False
        b.cudnn.benchmark = False
        b.cudnn.deterministic = True
        b.cudnn.enabled = False
        b.cuda.enable_flash_sdp(False)
        b.cuda.enable_mem_efficient_sdp(False)
        b.cuda.enable_math_sdp(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        b = torch.backends
        (
            b.cuda.matmul.allow_tf32,
            b.cudnn.allow_tf32,
            b.cudnn.benchmark,
            b.cudnn.deterministic,
            b.cudnn.enabled,
            flash,
            mem_eff,
            math_sdp,
        ) = self._saved
        b.cuda.enable_flash_sdp(flash)
        b.cuda.enable_mem_efficient_sdp(mem_eff)
        b.cuda.enable_math_sdp(math_sdp)


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def minimax_h3_resolve_reference_image_shape(
    *,
    width: int | float,
    height: int | float,
) -> dict[str, Any]:
    """Resolve a ref2va image independently from the target canvas.

    The image keeps its display ratio, always targets a 2048px short edge (even
    when that requires upscaling), and rounds both dimensions independently to
    the nearest 32px grid. Unlike target/video ``adapt_shape_v1``, reference
    images have no area-cap branch.
    """

    try:
        source_width = float(width)
        source_height = float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference image width and height must be positive finite numbers"
        ) from exc
    if (
        not math.isfinite(source_width)
        or not math.isfinite(source_height)
        or source_width <= 0.0
        or source_height <= 0.0
    ):
        raise ValueError(
            "reference image width and height must be positive finite numbers"
        )
    if source_width > 4.0 * source_height or source_height > 4.0 * source_width:
        raise ValueError(
            "reference image ratio must be within the inclusive range "
            f"1:4 to 4:1, got {source_width:g}x{source_height:g}"
        )

    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(source_width, source_height)
    target_width = _nearest_multiple(
        source_width * scale, MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
    )
    target_height = _nearest_multiple(
        source_height * scale, MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
    )
    return {
        "geometry": "reference_image_resolved",
        "shape_policy_version": "reference_image_short_edge_v1",
        "base_short_edge": MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE,
        "effective_short_edge": min(target_width, target_height),
        "size_mode": "short_edge",
        "multiple": MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
        "rounding": "nearest",
        "allow_upscale": True,
        "width": target_width,
        "height": target_height,
    }


def minimax_h3_resize_reference_image(
    image: Any,
    *,
    target_width: int,
    target_height: int,
) -> Any:
    """Resize a reference image to the shape fixed by pre-queue admission."""

    from PIL import Image

    if target_width <= 0 or target_height <= 0:
        raise ValueError("reference image target dimensions must be positive")
    if (
        target_width % MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
        or target_height % MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
    ):
        raise ValueError(
            "reference image target dimensions must be aligned to "
            f"{MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE}"
        )
    image = image.convert("RGB")
    if (target_width, target_height) == image.size:
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _load_waveform(
    path: str, *, material_chain: str = "audio"
) -> tuple[torch.Tensor, int]:
    """Apply the audio material chain.

    Pure-audio references preserve their source rate while normalizing to
    stereo. Video-bearing references first extract 44.1 kHz stereo PCM. The
    audio VAE boundary then performs the single 32 kHz resample below.
    """

    import json
    import subprocess
    import tempfile
    from pathlib import Path

    import torchaudio

    with tempfile.TemporaryDirectory(prefix="minimax_h3_audio_") as tmpdir:
        if material_chain == "audio":
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
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout).get("streams") or []
            if not streams or not streams[0].get("channels"):
                raise ValueError(f"reference audio has no audio stream: {path}")
            channels = int(streams[0]["channels"])
            output_path = Path(tmpdir) / "audio_stereo.flac"
            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
            ]
            if channels > MINIMAX_H3_AUDIO_CHANNELS:
                command += ["-c:a", "flac", "-ac", "2"]
            elif channels == 1:
                command += [
                    "-af",
                    "pan=stereo|c0=c0|c1=c0",
                    "-c:a",
                    "flac",
                ]
            else:
                command += ["-c:a", "flac"]
            command.append(str(output_path))
        elif material_chain in {
            "video.reference_preserve",
            "video_audio.reference_preserve",
        }:
            output_path = Path(tmpdir) / "video_audio_44100hz.wav"
            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-f",
                "wav",
                str(output_path),
            ]
        else:
            raise ValueError(
                f"unsupported MiniMax H3 audio material chain {material_chain!r}"
            )
        subprocess.run(command, check=True)
        return torchaudio.load(str(output_path))


@torch.inference_mode()
def minimax_h3_encode_reference_audio_rows(
    audio_vae: Any,
    audio_path: str,
    arch_config: MiniMaxH3AudioVAEArchConfig,
    *,
    material_chain: str = "audio",
) -> dict[str, Any]:
    """Encode a reference audio file into normalized channel-major rows.

    Returns {"rows": [2*T, 32] fp32 cpu, "ref_audio_t": T,
    "duration_seconds": float}.
    """
    import torchaudio

    model = audio_vae.model
    device = next(model.parameters()).device
    waveform, source_rate = _load_waveform(audio_path, material_chain=material_chain)
    if waveform.numel() == 0:
        raise ValueError(f"reference audio is empty: {audio_path}")
    waveform = waveform.float()
    if int(source_rate) != MINIMAX_H3_AUDIO_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            int(source_rate), MINIMAX_H3_AUDIO_SAMPLE_RATE
        )(waveform)
    if waveform.shape[0] < MINIMAX_H3_AUDIO_CHANNELS:
        repeats = (MINIMAX_H3_AUDIO_CHANNELS + waveform.shape[0] - 1) // waveform.shape[0]
        waveform = waveform.repeat(repeats, 1)
    waveform = waveform[:MINIMAX_H3_AUDIO_CHANNELS].to(device)

    with _AudioVAEDeterminismContext():
        audio_data = model.preprocess(waveform.unsqueeze(1), MINIMAX_H3_AUDIO_SAMPLE_RATE)
        z = model.encoder(audio_data)
        if bool(getattr(model, "attn_proj", False)):
            z = model.pre_block(z.transpose(1, 2)).transpose(1, 2)
        if not hasattr(model, "mean_proj"):
            raise AttributeError(
                "audio VAE model must expose mean_proj for deterministic mean encoding"
            )
        latent = model.mean_proj(z).float().cpu()  # [2, 32, T] or [2, T, 32]
    if latent.ndim != 3:
        raise ValueError(f"expected 3D audio latent, got {list(latent.shape)}")
    latent_channels = arch_config.latent_channels
    if int(latent.shape[-1]) != latent_channels:
        if int(latent.shape[1]) != latent_channels:
            raise ValueError(f"cannot canonicalize audio latent {list(latent.shape)}")
        latent = latent.transpose(1, 2).contiguous()  # -> [2, T, 32]

    mean = torch.tensor(arch_config.latents_mean).view(1, 1, latent_channels)
    std = torch.tensor(arch_config.latents_std).view(1, 1, latent_channels)
    normalized = (latent - mean) / std
    rows = normalized.reshape(-1, latent_channels).to(torch.float32).contiguous()
    ref_audio_t = int(latent.shape[1])
    return {
        "rows": rows,
        "ref_audio_t": ref_audio_t,
        "duration_seconds": float(waveform.shape[-1])
        / float(MINIMAX_H3_AUDIO_SAMPLE_RATE),
    }


MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY = "minimax_h3_prepared_reference_image"


# Module-local binding so tests can monkeypatch
# reference_encoding._ffprobe_video.
_ffprobe_video = minimax_h3_ffprobe_video


def minimax_h3_reference_video_has_audio(path: str) -> bool:
    """Return whether a reference video contains an audio stream.

    Canonical ``type=video`` inputs are always visual references, but only
    videos that actually carry a soundtrack are audio references.  This is
    the media-probe fact used to skip
    ``audio.extract`` for silent videos.
    """
    import json as _json
    import subprocess

    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = _json.loads(out.stdout).get("streams") or []
    return bool(streams)


def minimax_h3_prepare_reference_video(
    video_path: str,
    *,
    target_width: int,
    target_height: int,
    target_frame_count: int,
    workdir: str,
    fps: float = MINIMAX_H3_SUPPORTED_FPS,
    source_facts: dict[str, Any] | None = None,
) -> str:
    """Prepare a reference video for the visual-condition tokenizer.

    The source display ratio has already been resolved to ``target_width`` x
    ``target_height`` by the pre-queue hook. Preparation performs a direct
    scale (no crop) while materializing square-pixel CFR-24 video, then keeps
    at most the target number of leading frames in a separate truncation pass.
    Qwen and the visual VAE reuse this exact prepared file.
    """
    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be positive")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target reference-video dimensions must be positive")

    current, meta = minimax_h3_materialize_normalized_canvas(
        str(video_path),
        target_width=target_width,
        target_height=target_height,
        workdir=workdir,
        fps=fps,
        output_stem="refvid",
        context="reference-video preparation",
        probe=lambda path: _ffprobe_video(path),
        probe_source=source_facts is None,
    )

    if meta["frame_count"] > target_frame_count:
        current = minimax_h3_x264_transcode(
            current,
            ["-map", "0:v:0", "-an", "-frames:v", str(target_frame_count)],
            str(Path(workdir) / f"refvid_frames{target_frame_count}.mp4"),
        )
        meta = _ffprobe_video(current)
        if int(meta["frame_count"]) != target_frame_count:
            raise ValueError(
                "reference-video truncation produced unexpected frame count: "
                f"expected={target_frame_count}, actual={meta['frame_count']}"
            )
        minimax_h3_verify_normalized_video_meta(
            meta,
            target_width=target_width,
            target_height=target_height,
            fps=fps,
            context="reference-video truncation",
        )

    return current


MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED = 42
MINIMAX_H3_REFERENCE_VIDEO_PATCH_SIZE = (1, 2, 2)


@torch.inference_mode()
def minimax_h3_encode_reference_video_rows(
    video_vae: Any,
    video_path: str,
    arch_config: MiniMaxH3VideoVAEArchConfig,
) -> tuple[torch.Tensor, int, int, int]:
    """Encode a (truncated) reference video into packed imgvid cond rows.

    Decord reads ALL
    frames as uint8 numpy, then the SAME ``encode_videos``
    recipe as the fl2va keyframe sink (fp32 weights,
    parallel tiling off, torch seed pinned at 42 because the encode
    SAMPLES the DiagonalGaussian, fp16 latent), then normalize
    and [1,2,2]-patchify. The VAE's clip_length=17 / token_drop=3 give the
    17-frames-per-5-latents temporal grouping (107 frames -> 32 latents).

    Returns (rows [n, 96] fp32 cpu, latent_t, latent_h, latent_w).
    """
    import decord
    import numpy as np

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )

    reader = decord.VideoReader(str(video_path))
    if len(reader) <= 0:
        raise ValueError(f"video has no frames: {video_path}")
    frames = reader.get_batch(list(range(len(reader))))
    frames = frames.asnumpy() if hasattr(frames, "asnumpy") else np.asarray(frames)

    prev_parallel_tiling = video_vae.model.parallel_tiling
    video_vae.model.parallel_tiling = False
    parameter = next(video_vae.parameters())
    prev_dtype = parameter.dtype
    if prev_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        with minimax_h3_scoped_encode_rng(
            MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED, parameter.device
        ):
            z = video_vae.model.encode_videos(frames, use_fp16_latent=True)[0]
    finally:
        video_vae.model.parallel_tiling = prev_parallel_tiling
        if prev_dtype != torch.float32:
            video_vae.to(prev_dtype)
    z = z.float().cpu()
    if z.dim() == 4:
        z = z[None]
    latent_channels = arch_config.latent_channels
    if z.dim() != 5 or int(z.shape[1]) != latent_channels:
        raise ValueError(f"unexpected reference video latent shape {list(z.shape)}")
    latent_t, latent_h, latent_w = int(z.shape[2]), int(z.shape[3]), int(z.shape[4])
    mean = torch.tensor(arch_config.latents_mean).view(1, latent_channels, 1, 1, 1)
    std = torch.tensor(arch_config.latents_std).view(1, latent_channels, 1, 1, 1)
    rows = minimax_h3_patchify_video_latent(
        (z - mean) / std, patch_size=list(MINIMAX_H3_REFERENCE_VIDEO_PATCH_SIZE)
    )
    return rows.to(torch.float32), latent_t, latent_h, latent_w


MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2


def minimax_h3_sample_reference_video_frames(
    prepared_path: str,
    *,
    workdir: str,
) -> dict[str, Any]:
    """Sample Qwen frames from the PREPARED reference video.

    Frame-sampling recipe (cursor += fps ratio,
    round, dedup; one ffmpeg select per frame -> png) plus the
    qwen3 timestamp rule (indices padded to the
    temporal patch size with the last frame, block ts = mean of the pair at
    sample fps; text is rendered later with ``f"<{ts:.1f} seconds>"``).

    Returns {"frames": [np.ndarray HxWx3 u8], "block_timestamps": [float]}.
    """
    import subprocess
    from pathlib import Path as _Path

    import numpy as np
    from PIL import Image

    meta = _ffprobe_video(prepared_path)
    ratio = MINIMAX_H3_SUPPORTED_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while True:
        idx = int(round(cursor))
        if idx >= meta["frame_count"]:
            break
        if not indices or idx > indices[-1]:
            indices.append(idx)
        cursor += ratio
    if not indices:
        raise ValueError(f"no frames sampled from {prepared_path}")
    frames = []
    frame_dir = _Path(workdir) / "qwen_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for out_i, src_i in enumerate(indices, start=1):
        p = frame_dir / f"frame_{out_i:06d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(prepared_path),
                "-vf",
                f"select=eq(n\\,{src_i})",
                "-vsync",
                "vfr",
                "-frames:v",
                "1",
                str(p),
            ],
            check=True,
        )
        frames.append(np.asarray(Image.open(p).convert("RGB")))
    ts = [i / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for i in range(len(indices))]
    pad = (-len(ts)) % MINIMAX_H3_QWEN_TEMPORAL_PATCH
    ts = ts + [ts[-1]] * pad
    block_timestamps = [
        (ts[i] + ts[i + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for i in range(0, len(ts), MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    ]
    return {"frames": frames, "block_timestamps": block_timestamps}


def _reference_video_materials(plan: Any) -> list[Any]:
    return [
        m
        for m in plan.materials
        if m.material_chain
        in {"video.reference_preserve", "video_audio.reference_preserve"}
    ]


def _reference_video_target_frame_count(
    *,
    plan: Any,
) -> int:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_align_frame_count,
        minimax_h3_frame_count_from_video_latent_t,
    )

    shape = plan.shape
    fps = int(shape["fps"])
    duration = shape.get("duration_seconds")
    if duration is not None:
        return minimax_h3_align_frame_count(int(round(float(duration) * fps)))
    if shape.get("video_latent_t") is not None:
        return minimax_h3_frame_count_from_video_latent_t(int(shape["video_latent_t"]))
    raise ValueError(
        "reference-video preparation requires pre-queue resolved temporal dimensions"
    )


def minimax_h3_prepared_reference_videos(batch: Any, plan: Any) -> dict[str, Any]:
    """Run the reference-video prepare chain once per request.

    The prepared (fps-normalized, cap-resized, truncated) mp4 is consumed by
    BOTH the visual-condition tokenizer (full-frame VAE encode) and the Qwen frame
    sampler; its target frame count comes from the resolved target duration
    (17n+5 rule), matching the material chain. The reference
    soundtrack is read from the ORIGINAL file (pre-truncation), so the
    original path travels alongside. Cached in batch.extra.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY,
    )

    cached = batch.extra.get(MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY)
    if cached is not None:
        return cached
    videos = _reference_video_materials(plan)
    if not videos:
        raise NotImplementedError(
            "ref2va video preparation requires a video or video_audio reference"
        )

    import tempfile

    prepared_videos = []
    for material in videos:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_localize_material_uri,
            minimax_h3_register_temp_dir,
        )

        video_path = minimax_h3_localize_material_uri(
            batch,
            material.uri,
            condition_type=material.condition_type,
            condition_index=int(material.condition_index),
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
        )

        condition_index = int(material.condition_index)
        source_facts = batch.extra.get(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, {}).get(
            condition_index
        )
        resolved_material_shape = batch.extra.get(
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY, {}
        ).get(condition_index)
        if not isinstance(source_facts, dict) or not isinstance(
            resolved_material_shape, dict
        ):
            raise ValueError(
                "reference-video preparation requires cached pre-queue probe "
                f"and shape facts for conditions[{condition_index}]"
            )
        input_has_audio = bool(source_facts.get("has_audio"))
        target_frames = _reference_video_target_frame_count(plan=plan)
        workdir = minimax_h3_register_temp_dir(
            batch,
            tempfile.mkdtemp(prefix="minimax_h3_refvid_"),
            owner="reference_video",
        )
        prepared_path = minimax_h3_prepare_reference_video(
            video_path,
            target_width=int(resolved_material_shape["width"]),
            target_height=int(resolved_material_shape["height"]),
            target_frame_count=target_frames,
            workdir=workdir,
            fps=float(plan.shape["fps"]),
            source_facts=source_facts,
        )
        prepared_videos.append(
            {
                "prepared_path": prepared_path,
                "original_path": video_path,
                "target_frame_count": target_frames,
                "condition_index": int(material.condition_index),
                "material_chain": str(material.material_chain),
                "input_has_audio": input_has_audio,
                "width": int(resolved_material_shape["width"]),
                "height": int(resolved_material_shape["height"]),
            }
        )
    prepared = dict(prepared_videos[0])
    prepared["videos"] = prepared_videos
    batch.extra[MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY] = prepared
    return prepared


def minimax_h3_prepared_reference_image(batch: Any, plan: Any) -> dict[str, Any]:
    """Resize ref2va image references to their pre-queue-resolved shapes.

    Qwen (pixel_values) and the visual-condition tokenizer consume the identical
    prepared image. The runtime never recomputes geometry from ``plan.shape``;
    it consumes the per-material width/height admitted before queueing.
    """
    cached = batch.extra.get(MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY)
    if cached is not None:
        return cached
    images = [
        m for m in plan.materials if m.material_chain == "image.reference_preserve"
    ]
    if not images:
        raise ValueError("ref2va requires at least one image reference")
    from PIL import Image, ImageOps

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
        minimax_h3_localize_material_uri,
    )

    prepared_images = []
    for material in images:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
        )

        condition_index = int(material.condition_index)
        source_facts = batch.extra.get(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, {}).get(
            condition_index
        )
        resolved_shape = batch.extra.get(
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY, {}
        ).get(condition_index)
        if not isinstance(source_facts, dict) or not isinstance(resolved_shape, dict):
            raise ValueError(
                "reference-image preparation requires cached pre-queue probe "
                f"and shape facts for conditions[{condition_index}]"
            )
        image_path = minimax_h3_localize_material_uri(
            batch,
            material.uri,
            condition_type=material.condition_type,
            condition_index=condition_index,
        )
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        expected_size = (
            int(resolved_shape["width"]),
            int(resolved_shape["height"]),
        )
        prepared_image = minimax_h3_resize_reference_image(
            image,
            target_width=expected_size[0],
            target_height=expected_size[1],
        )
        if prepared_image.size != expected_size:
            raise ValueError(
                "reference image preparation disagrees with pre-queue shape: "
                f"expected={expected_size}, actual={prepared_image.size}"
            )
        prepared_images.append(
            {
                "image": prepared_image,
                "condition_index": condition_index,
            }
        )
    prepared = {
        # single-image consumers keep the existing keys
        "image": prepared_images[0]["image"],
        "condition_index": prepared_images[0]["condition_index"],
        "images": prepared_images,
    }
    batch.extra[MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY] = prepared
    return prepared


__all__ = [
    "minimax_h3_encode_reference_audio_rows",
    "minimax_h3_encode_reference_video_rows",
    "minimax_h3_prepared_reference_image",
    "minimax_h3_prepared_reference_videos",
    "minimax_h3_reference_video_has_audio",
    "minimax_h3_resolve_reference_image_shape",
    "minimax_h3_sample_reference_video_frames",
]
