# SPDX-License-Identifier: Apache-2.0
"""Shared ffprobe/ffmpeg media preparation for MiniMax H3 condition inputs.

The ref2va reference recipe materializes a normalized intermediate: a CFR,
square-pixel, rotation-free libx264 stream at the target canvas. This module
holds the single implementation; the calling modules keep thin wrappers so
their module-local ``_ffprobe_video`` bindings stay monkeypatchable in tests.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def minimax_h3_ffprobe_video(path: str) -> dict[str, Any]:
    """Probe the primary video stream (geometry, fps, frames, SAR, rotation)."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames,nb_frames,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"media has no video stream: {path}")
    stream = streams[0]
    numerator, denominator = str(stream["r_frame_rate"]).split("/")
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in (None, "", "N/A"):
        raise ValueError(f"cannot determine video frame count: {path}")
    rotation = 0.0
    for item in stream.get("side_data_list") or ():
        if isinstance(item, Mapping) and item.get("rotation") is not None:
            rotation = float(item["rotation"]) % 360.0
            break
    if not rotation and isinstance(stream.get("tags"), Mapping):
        rotation = float(stream["tags"].get("rotate") or 0.0) % 360.0
    sar = str(stream.get("sample_aspect_ratio") or "1:1")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frame_count": int(raw_count),
        "sample_aspect_ratio": sar,
        "rotation_degrees": rotation,
    }


def minimax_h3_x264_transcode(source: str, args: list[str], output: str) -> str:
    """Re-encode ``source`` to rotation-free yuv420p libx264 at ``output``."""

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
        + args
        + [
            "-metadata:s:v:0",
            "rotate=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    return str(output)


def minimax_h3_verify_normalized_video_meta(
    meta: Mapping[str, Any],
    *,
    target_width: int,
    target_height: int,
    fps: float,
    context: str,
) -> None:
    """Fail closed when a materialized stream misses the normalized contract."""

    materialized_size = (int(meta["width"]), int(meta["height"]))
    if materialized_size != (target_width, target_height):
        raise ValueError(
            f"{context} produced unexpected geometry: "
            f"expected={target_width}x{target_height}, "
            f"actual={materialized_size[0]}x{materialized_size[1]}"
        )
    if str(meta.get("sample_aspect_ratio") or "1:1") not in {"1:1", "1/1"}:
        raise ValueError(f"{context} did not normalize SAR to 1:1")
    if abs(float(meta.get("rotation_degrees") or 0.0)) > 1e-6:
        raise ValueError(f"{context} retained rotation metadata")
    if abs(float(meta.get("fps") or 0.0) - fps) >= 1e-6:
        raise ValueError(
            f"{context} did not normalize frame rate: "
            f"expected={fps:g}, actual={meta.get('fps')!r}"
        )


def minimax_h3_materialize_normalized_canvas(
    source: str,
    *,
    target_width: int,
    target_height: int,
    workdir: str,
    fps: float,
    output_stem: str,
    context: str,
    probe: Callable[[str], Mapping[str, Any]],
    probe_source: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Direct-scale ``source`` onto the target canvas; never crop.

    Always materializes a CFR, square-pixel stream: nominal/average probe
    rates cannot prove that an input is CFR, and ffmpeg's fps filter also
    applies display rotation before the direct target resize.
    """

    if probe_source:
        probe(str(source))
    filters = [
        f"fps={fps:g}",
        f"scale={target_width}:{target_height}:flags=lanczos",
        "setsar=1",
    ]
    prepared = minimax_h3_x264_transcode(
        str(source),
        ["-map", "0:v:0", "-an", "-vf", ",".join(filters)],
        str(Path(workdir) / f"{output_stem}_{target_width}x{target_height}.mp4"),
    )
    metadata = dict(probe(prepared))
    minimax_h3_verify_normalized_video_meta(
        metadata,
        target_width=target_width,
        target_height=target_height,
        fps=fps,
        context=context,
    )
    return prepared, metadata


__all__ = [
    "minimax_h3_ffprobe_video",
    "minimax_h3_materialize_normalized_canvas",
    "minimax_h3_verify_normalized_video_meta",
    "minimax_h3_x264_transcode",
]
