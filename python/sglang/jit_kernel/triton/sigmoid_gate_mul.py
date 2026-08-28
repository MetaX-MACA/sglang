# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Kernel from upstream PR #27630, which only enables it under _is_hip. It is
# pure triton, so it is portable as is; only the launch config is retuned.
#
# Tuned on MetaX C600-U (T=4096, width=8192, bf16): latency tracks total warp
# count (grid * num_warps), flat between 16k and 32k warps and rising linearly
# above. 2048/2 sits on that plateau; upstream's 1024/4 is 1.24x slower.
_BLOCK_SIZE = int(os.environ.get("SGLANG_SIGMOID_GATE_MUL_BLOCK_SIZE", "2048"))
_NUM_WARPS = int(os.environ.get("SGLANG_SIGMOID_GATE_MUL_NUM_WARPS", "2"))


@triton.jit
def _sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    out = x * tl.sigmoid(g)
    tl.store(out_ptr + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute x * sigmoid(gate) in a single fused kernel."""
    # the kernel indexes flat, so a non-contiguous input would silently misread
    assert x.is_contiguous() and gate.is_contiguous(), "inputs must be contiguous"
    assert x.shape == gate.shape, f"shape mismatch: {x.shape} vs {gate.shape}"

    out = torch.empty_like(x)
    n = x.numel()
    _sigmoid_gate_mul_kernel[(triton.cdiv(n, _BLOCK_SIZE),)](
        x,
        gate,
        out,
        n,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return out
