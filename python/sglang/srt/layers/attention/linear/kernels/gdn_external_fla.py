# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in external FLA adapters for MACA Triton GDN.

The caller retains native ReplaySSM, verification and intermediate-state tracking.
External FLA imports and JIT compilation occur only when this path is selected.
"""

import inspect
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@lru_cache(maxsize=2)
def _external_kernel(prefill):
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    kernel = chunk_gated_delta_rule if prefill else fused_recurrent_gated_delta_rule
    parameters = inspect.signature(kernel).parameters
    if "use_qk_l2norm_in_kernel" not in parameters:
        raise RuntimeError("External FLA GDN requires use_qk_l2norm_in_kernel support")
    return kernel, parameters


def _prepare_external_state(initial_state, parameters):
    if "state_v_first" in parameters:
        return initial_state, {"state_v_first": True}, False
    if "transpose_state_layout" in parameters:
        return initial_state, {"transpose_state_layout": True}, False
    return initial_state.transpose(-1, -2).contiguous(), {}, True


@triton.jit
def _write_state_pool_kernel(
    pool, indices, states,
    PS0: tl.constexpr, PS1: tl.constexpr, PS2: tl.constexpr, PS3: tl.constexpr,
    IS0: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
    H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(indices + row * IS0).to(tl.int64)
    mask = (x < H * V * K) & (slot >= 0)
    h = x // (V * K)
    v = x // K % V
    k = x % K
    value = tl.load(states + row * SS0 + h * SS1 + v * SS2 + k * SS3, mask, other=0)
    tl.store(pool + slot * PS0 + h * PS1 + v * PS2 + k * PS3, value, mask)


def _write_state_pool(pool, indices, states):
    # Live rows have distinct slots. Mask negative graph-padding rows rather
    # than clamping writes to slot 0, which could overwrite a live request.
    h, v, k = pool.shape[-3:]
    _write_state_pool_kernel[(indices.numel(), triton.cdiv(h * v * k, 256))](
        pool, indices, states, *pool.stride(), indices.stride(0),
        *states.stride(), h, v, k, 256,
    )


def packed_decode_external_fla(
    mixed_qkv, a, b, *, A_log, dt_bias, scale, ssm_states, cache_indices,
    num_v_heads, head_v_dim, **kwargs,
):
    kernel, parameters = _external_kernel(False)
    batch = mixed_qkv.shape[0]
    hv, vdim, kdim = ssm_states.shape[-3:]
    qk_dim = mixed_qkv.shape[-1] - hv * vdim
    if qk_dim <= 0 or qk_dim % (2 * kdim):
        raise ValueError("Packed QKV dimensions do not match the GDN state pool")
    qdim = qk_dim // 2
    q, k, v = mixed_qkv.split([qdim, qdim, hv * vdim], dim=-1)
    q = q.view(batch, 1, qdim // kdim, kdim)
    k = k.view(batch, 1, qdim // kdim, kdim)
    v = v.view(batch, 1, hv, vdim)
    g = -A_log.float().exp().view(1, 1, hv) * F.softplus(
        a.float().view(batch, 1, hv) + dt_bias.float().view(1, 1, hv)
    )
    beta = b.sigmoid().view(batch, 1, hv)
    initial = ssm_states[cache_indices.clamp_min(0).long()]
    state, extra, transpose = _prepare_external_state(initial, parameters)
    output, final = kernel(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=state,
        output_final_state=True, use_qk_l2norm_in_kernel=True, **extra,
    )
    if final is not None:
        if transpose:
            final = final.transpose(-1, -2).contiguous()
        _write_state_pool(ssm_states, cache_indices, final)
    output = torch.where(
        (cache_indices >= 0).view(batch, 1, 1, 1), output.to(mixed_qkv.dtype), 0
    )
    return output.transpose(0, 1)


def extend_external_fla(
    q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, **kwargs,
):
    kernel, parameters = _external_kernel(True)
    initial = ssm_states[cache_indices.clamp_min(0).long()]
    state, extra, transpose = _prepare_external_state(initial, parameters)
    if "head_first" in parameters:
        extra["head_first"] = False
    output, final = kernel(
        q=q, k=k, v=v, g=g, beta=beta, scale=k.shape[-1] ** -0.5,
        initial_state=state, output_final_state=True,
        cu_seqlens=query_start_loc, use_qk_l2norm_in_kernel=True, **extra,
    )
    if final is not None:
        if transpose:
            final = final.transpose(-1, -2).contiguous()
        _write_state_pool(ssm_states, cache_indices, final)
    destination = kwargs.get("output")
    if destination is not None:
        destination.copy_(output)
        output = destination
    return output, None, None
