# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 keyframe (imgvid) condition encoding.

Condition anchor row recipe:

- ``video_vae.model.encode_images(PIL, use_fp16_latent=True)`` under a
  scoped seed-42 RNG fork — the DiagonalGaussian is SAMPLED
  (use_mean=False) with seed 42, so the seed is part of
  the contract, not a convenience
- normalize ``(z - latents_mean) / latents_std`` with the loader-injected
  ``MiniMaxH3VideoVAEArchConfig`` values
- patchify [1, 2, 2] into packed cond rows, fp32
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.utils import (
    minimax_h3_scoped_encode_rng,
)

MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42
MINIMAX_H3_KEYFRAME_PATCH_SIZE = (1, 2, 2)


@torch.inference_mode()
def minimax_h3_encode_keyframe_cond_rows(
    video_vae: Any,
    image: Any,
    arch_config: MiniMaxH3VideoVAEArchConfig,
) -> torch.Tensor:
    """Encode a target-canvas PIL image into packed imgvid cond rows.

    Returns [n_rows, 24 * patch_h * patch_w] fp32 on CPU.
    """
    seed = MINIMAX_H3_KEYFRAME_ENCODE_SEED
    # Single-process encode probe semantics: parallel tiling off
    # for the encode call, restored afterwards (decode keeps its own config).
    prev_parallel_tiling = video_vae.model.parallel_tiling
    video_vae.model.parallel_tiling = False
    # The encode recipe runs on fp32 weights. The
    # pipeline loads the VAE in vae_precision (fp16) for decode; upcast for
    # the encode and restore (fp16 -> fp32 -> fp16 is bit-lossless).
    parameter = next(video_vae.parameters())
    prev_dtype = parameter.dtype
    if prev_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        with minimax_h3_scoped_encode_rng(seed, parameter.device):
            z = video_vae.model.encode_images(image, use_fp16_latent=True)[0]
    finally:
        video_vae.model.parallel_tiling = prev_parallel_tiling
        if prev_dtype != torch.float32:
            video_vae.to(prev_dtype)
    z = z.float().cpu()
    if z.dim() == 4:
        z = z[None]
    latent_channels = arch_config.latent_channels
    if z.dim() != 5 or int(z.shape[1]) != latent_channels:
        raise ValueError(f"unexpected imgvid latent shape {list(z.shape)}")
    mean = torch.tensor(arch_config.latents_mean).view(1, latent_channels, 1, 1, 1)
    std = torch.tensor(arch_config.latents_std).view(1, latent_channels, 1, 1, 1)
    rows = minimax_h3_patchify_video_latent(
        (z - mean) / std, patch_size=list(MINIMAX_H3_KEYFRAME_PATCH_SIZE)
    )
    return rows.to(torch.float32)


__all__ = [
    "MINIMAX_H3_KEYFRAME_ENCODE_SEED",
    "MINIMAX_H3_KEYFRAME_PATCH_SIZE",
    "minimax_h3_encode_keyframe_cond_rows",
]
