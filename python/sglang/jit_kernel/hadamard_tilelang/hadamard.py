# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#       Ported into sglang.  Public API of the TileLang composite Hadamard.

from __future__ import annotations

from functools import cache
from typing import Any

import torch


def _tilelang_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError("Hadamard supports bfloat16, float16, and float32 inputs")


@cache
def _get_hadamard_kernel(
    hidden_size: int,
    base_order: int,
    inverse: bool,
    dtype: str,
    num_threads: int,
    mma_rows: int,
):
    # Keep TileLang and the JIT kernel out of the module import path.
    from .tilelang_kernels import make_composite_hadamard_kernel

    return make_composite_hadamard_kernel(
        hidden_size,
        base_order,
        inverse,
        dtype,
        num_threads,
        mma_rows,
    )


def _hadamard_pow2(x: torch.Tensor) -> torch.Tensor:
    """Unscaled Sylvester Hadamard on the last, power-of-two dimension."""
    n = x.shape[-1]
    orig = x.shape
    x = x.contiguous().reshape(-1, n)
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2 * h)
        a = x[..., :h]
        b = x[..., h : 2 * h]
        x = torch.cat([a + b, a - b], dim=-1).reshape(-1, n)
        h *= 2
    return x.reshape(orig)


def _hadamard_torch(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h16: torch.Tensor,
    inverse: bool,
    base_order: int,
    power_of_two_part: int,
    num_threads: int,
    mma_rows: int,
) -> torch.Tensor:
    """Torch mirror of the TileLang composite Hadamard kernel (same math)."""
    k = base_order
    width = power_of_two_part
    hidden_size = x.shape[-1]
    scale = 1.0 / (hidden_size**0.5)
    out_dtype = x.dtype
    # The TileLang kernel computes entirely in float32; mirror that precision.
    x32 = x.reshape(-1, hidden_size).to(torch.float32)
    sign32 = sign.to(dtype=torch.float32)
    hk32 = hk.to(dtype=torch.float32)

    if inverse:
        hb = _hadamard_pow2(x32.reshape(-1, k, width))
    else:
        hb = _hadamard_pow2((x32 * sign32).reshape(-1, k, width))

    # mixed[ob, p] = sum_i hk[ob, i] * xp[i, p]      (inverse, H_K)
    # mixed[ob, p] = sum_i hk[i, ob] * xp[i, p]      (forward, H_K^T)
    mixed = torch.matmul(hk32, hb) if inverse else torch.matmul(hk32.t(), hb)

    out = (mixed * scale).reshape(-1, hidden_size)
    if inverse:
        out = out * sign32
    return out.to(out_dtype).reshape(x.shape)


def hadamard(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h16: torch.Tensor,
    *,
    inverse: bool = False,
    k: int | None = None,
    p: int | None = None,
    num_threads: int = 512,
    mma_rows: int = 128,
) -> torch.Tensor:
    """Composite Hadamard rotation, TileLang-backed with a torch fallback."""
    from .tilelang_kernels import _TILELANG_ERROR, tilelang
    from .utils import resolve_hadamard_dim

    hidden_size = x.shape[-1]
    resolved_k, resolved_p = resolve_hadamard_dim(hidden_size, k=k, p=p)

    if x.is_cuda and tilelang is not None:
        input_2d = x.reshape(-1, hidden_size).contiguous()
        output_2d = input_2d.new_empty(input_2d.shape)
        if input_2d.shape[0] != 0:
            kernel = _get_hadamard_kernel(
                hidden_size,
                resolved_k,
                inverse,
                _tilelang_dtype(x.dtype),
                num_threads,
                mma_rows,
            )
            kernel(input_2d, sign, hk, h16, output_2d)
        return output_2d.reshape(x.shape)

    return _hadamard_torch(
        x,
        sign,
        hk,
        h16,
        inverse,
        resolved_k,
        resolved_p,
        num_threads,
        mma_rows,
    )


__all__ = ["hadamard"]