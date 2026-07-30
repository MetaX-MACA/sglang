"""Triton operators for DSpark compact ragged GDN target verification.

This module is intentionally the single home for DSpark-specific GDN Triton
operators. Triton 3.2.0 is the minimum supported version; older versions disable
the compact GDN path so its caller can fall back to DSpark static verify.

The main operator evaluates a depth-wise causal convolution directly over a
flattened ragged token buffer. Requests are delimited by ``query_start_loc``:
request ``r`` owns tokens ``[query_start_loc[r], query_start_loc[r + 1])``.
Unlike the rectangular target-verify kernel, it performs work for the physical
graph token count M rather than S * W.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

_MIN_TRITON = (3, 2)


def _triton_major_minor() -> Optional[tuple[int, int]]:
    match = re.match(r"^(\d+)\.(\d+)", str(getattr(triton, "__version__", "")))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


@lru_cache(maxsize=1)
def dspark_ragged_gdn_supported() -> tuple[bool, str]:
    """Return compact-ragged GDN capability and a stable diagnostic reason.

    The kernels use only long-standing primitives:
    program ids, ``static_range``, masked load/store, reductions, and elementary
    arithmetic.
    """
    version = _triton_major_minor()
    raw_version = str(getattr(triton, "__version__", "unknown"))
    if version is None:
        return False, f"unparseable_triton_version:{raw_version}"
    if version < _MIN_TRITON:
        return False, f"triton_{raw_version}_below_3.2"
    return True, f"triton_{raw_version}"


@triton.jit
def _dspark_ragged_token_rows_kernel(
    query_start_loc_ptr,
    token_request_ptr,
    token_within_ptr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Map each flattened token to its request and request-local position.

    Input is the int32 `[S+1]` cumulative offset array. Outputs are int32 `[M]`
    request ids and `[M]` local positions. A vectorized upper-bound search counts
    request ends less than or equal to the token index, correctly skipping
    repeated offsets from zero-length virtual rows. The launch is shape-static
    for CUDA Graph `(M,S)` capture, uses no atomics or host-visible values, and
    relies only on masked loads, comparisons, and `tl.sum` available throughout
    Triton 3.2-3.6.
    """
    token = tl.program_id(0)
    slot_offsets = tl.arange(0, BLOCK_S)
    slot_mask = slot_offsets < NUM_SLOTS
    request_ends = tl.load(
        query_start_loc_ptr + slot_offsets + 1,
        mask=slot_mask,
        other=0x7FFFFFFF,
    )
    request = tl.sum(tl.where(token >= request_ends, 1, 0))
    total_tokens = tl.load(query_start_loc_ptr + NUM_SLOTS)
    valid_token = (token < total_tokens) & (request < NUM_SLOTS)
    safe_request = tl.minimum(request, NUM_SLOTS - 1)
    request_start = tl.load(query_start_loc_ptr + safe_request)
    tl.store(token_request_ptr + token, tl.where(valid_token, request, -1))
    tl.store(
        token_within_ptr + token,
        tl.where(valid_token, token - request_start, -1),
    )


@triton.jit
def _dspark_ragged_causal_conv1d_update_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    token_request_ptr,
    token_within_ptr,
    cache_indices_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
    dim,
    state_len,
    stride_x_token,
    stride_x_dim,
    stride_state_seq,
    stride_state_dim,
    stride_state_win,
    stride_weight_dim,
    stride_weight_win,
    stride_inter_seq,
    stride_inter_step,
    stride_inter_dim,
    stride_inter_win,
    stride_out_token,
    stride_out_dim,
    NUM_SLOTS: tl.constexpr,
    NUM_CACHE_LINES: tl.constexpr,
    NUM_INTERMEDIATE_ROWS: tl.constexpr,
    MAX_INTERMEDIATE_STEPS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Evaluate one flattened token/channel tile of ragged causal convolution.

    Math for request-local token j and channel d is
    ``y[j,d] = act(bias[d] + sum_k weight[d,k] * source[j+k-K+1,d])``.
    Negative request-local source positions read the committed convolution
    state selected by ``cache_indices``; non-negative positions read ``x`` from
    the same ragged segment. The committed state is read-only. If requested,
    each token's resulting K-1 input window is written to the speculative
    intermediate cache for later acceptance/commit.

    Shapes and dtypes:
    - x/out: [M, D], fp16/bf16/fp32 with arbitrary leading stride.
    - conv_state: [N, D, K-1 or greater], same value dtype.
    - weight/bias: [D, K] / [D], value dtype; accumulation is fp32.
    - query_start_loc: [S+1], int32 cumulative ragged offsets.
    - cache_indices/intermediate_state_indices: [S], int32 or int64.
    - intermediate window: [N_i, W, D, K-1], arbitrary strides, including
      SGLang's overlapping as_strided representation.

    Zero-length virtual requests are represented by repeated offsets and launch
    no token programs. Request, cache-row, local-step, and intermediate-row
    bounds are masked in-device so stale or padded graph metadata cannot form an
    invalid address. The caller still leaves virtual rows empty until a scratch
    KV/state pool exists. All pointers and output shapes are stable for a captured
    (M,S) graph, and the kernel performs no host synchronization or
    data-dependent allocation.
    """
    token = tl.program_id(0)
    d_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < dim

    request = tl.load(token_request_ptr + token)
    local_token = tl.load(token_within_ptr + token)
    valid_request = (request >= 0) & (request < NUM_SLOTS)
    valid_token = (
        valid_request & (local_token >= 0) & (local_token < MAX_INTERMEDIATE_STEPS)
    )
    safe_request = tl.maximum(0, tl.minimum(request, NUM_SLOTS - 1))
    safe_local_token = tl.maximum(local_token, 0)
    request_start = token - safe_local_token

    cache_index = tl.load(
        cache_indices_ptr + safe_request, mask=valid_request, other=-1
    )
    safe_cache_index = tl.maximum(0, tl.minimum(cache_index, NUM_CACHE_LINES - 1))
    valid_cache = valid_token & (cache_index >= 0) & (cache_index < NUM_CACHE_LINES)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    for tap in tl.static_range(0, KERNEL_WIDTH):
        source_local = safe_local_token + tap - (KERNEL_WIDTH - 1)
        source_token = request_start + tl.maximum(source_local, 0)
        x_value = tl.load(
            x_ptr + source_token * stride_x_token + d_offsets * stride_x_dim,
            mask=d_mask & valid_token & (source_local >= 0),
            other=0.0,
        )
        state_offset = tl.maximum(state_len + source_local, 0)
        state_value = tl.load(
            conv_state_ptr
            + safe_cache_index * stride_state_seq
            + d_offsets * stride_state_dim
            + state_offset * stride_state_win,
            mask=d_mask & (source_local < 0) & (state_offset < state_len) & valid_cache,
            other=0.0,
        )
        source_value = tl.where(source_local >= 0, x_value, state_value)
        weight = tl.load(
            weight_ptr + d_offsets * stride_weight_dim + tap * stride_weight_win,
            mask=d_mask,
            other=0.0,
        )
        acc += source_value.to(tl.float32) * weight.to(tl.float32)

    if SILU_ACTIVATION:
        acc = acc / (1.0 + tl.exp(-acc))
    tl.store(
        out_ptr + token * stride_out_token + d_offsets * stride_out_dim,
        tl.where(valid_token, acc, 0.0),
        mask=d_mask,
    )

    if SAVE_INTERMEDIATE:
        intermediate_index = tl.load(
            intermediate_state_indices_ptr + safe_request,
            mask=valid_request,
            other=-1,
        )
        safe_intermediate_index = tl.maximum(
            0, tl.minimum(intermediate_index, NUM_INTERMEDIATE_ROWS - 1)
        )
        valid_intermediate = (
            valid_token
            & (intermediate_index >= 0)
            & (intermediate_index < NUM_INTERMEDIATE_ROWS)
        )
        for window_offset in tl.static_range(0, KERNEL_WIDTH - 1):
            source_local = safe_local_token + window_offset - (KERNEL_WIDTH - 2)
            source_token = request_start + tl.maximum(source_local, 0)
            x_value = tl.load(
                x_ptr + source_token * stride_x_token + d_offsets * stride_x_dim,
                mask=d_mask & valid_token & (source_local >= 0),
                other=0.0,
            )
            state_offset = tl.maximum(state_len + source_local, 0)
            state_value = tl.load(
                conv_state_ptr
                + safe_cache_index * stride_state_seq
                + d_offsets * stride_state_dim
                + state_offset * stride_state_win,
                mask=d_mask
                & (source_local < 0)
                & (state_offset < state_len)
                & valid_cache,
                other=0.0,
            )
            window_value = tl.where(source_local >= 0, x_value, state_value)
            tl.store(
                intermediate_conv_window_ptr
                + safe_intermediate_index * stride_inter_seq
                + safe_local_token * stride_inter_step
                + d_offsets * stride_inter_dim
                + window_offset * stride_inter_win,
                window_value,
                mask=d_mask & valid_intermediate,
            )


def dspark_ragged_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_conv_window: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    activation: Optional[str],
) -> torch.Tensor:
    """Launch graph-safe causal convolution over a flattened DSpark layout.

    ``x`` is `[M,D]`; request boundaries come exclusively from the device-side
    `[S+1]` ``query_start_loc`` buffer, whose contents may change before CUDA
    Graph replay while its pointer and shape remain fixed for a captured `(M,S)`
    tier. ``cache_indices`` selects committed `[N,D,state_len]` state rows.
    Intermediate state is `[N_i,W,D,K-1]` and records every request-local token
    window without updating committed state. This wrapper supports fp16, bf16,
    and fp32 values, int32/int64 indices, SiLU/swish or no activation, and Triton
    3.2-3.6 stable primitives. It never calls `.item()`, `.cpu()`, or `.tolist()`.
    """
    supported, reason = dspark_ragged_gdn_supported()
    if not supported:
        raise RuntimeError(f"DSpark ragged GDN is unavailable: {reason}")
    if x.ndim != 2:
        raise ValueError(f"x must have shape [M, D], got {tuple(x.shape)}")
    if conv_state.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "conv_state and weight must have shapes [N,D,state] and [D,K], "
            f"got {tuple(conv_state.shape)} and {tuple(weight.shape)}"
        )
    if x.shape[1] != weight.shape[0] or conv_state.shape[1] != x.shape[1]:
        raise ValueError(
            "ragged GDN channel dimensions disagree: "
            f"x={tuple(x.shape)}, state={tuple(conv_state.shape)}, "
            f"weight={tuple(weight.shape)}"
        )
    if weight.shape[1] < 2 or conv_state.shape[2] < weight.shape[1] - 1:
        raise ValueError(
            f"invalid convolution width/state: weight={tuple(weight.shape)}, "
            f"state={tuple(conv_state.shape)}"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError(
            f"query_start_loc must have shape [S+1], got {tuple(query_start_loc.shape)}"
        )
    if query_start_loc.dtype != torch.int32:
        raise ValueError(
            "query_start_loc must be int32 for the ragged GDN kernels, got "
            f"{query_start_loc.dtype}"
        )
    num_slots = query_start_loc.shape[0] - 1
    if cache_indices.ndim != 1 or cache_indices.shape[0] < num_slots:
        raise ValueError(
            f"cache_indices must contain at least {num_slots} rows, got "
            f"{tuple(cache_indices.shape)}"
        )
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation {activation!r}")
    save_intermediate = intermediate_conv_window is not None
    if save_intermediate != (intermediate_state_indices is not None):
        raise ValueError(
            "intermediate_conv_window and intermediate_state_indices must be "
            "provided together"
        )

    out = torch.empty_like(x)
    block_d = 256
    block_s = triton.next_power_of_2(num_slots)
    token_request = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    token_within = torch.empty_like(token_request)
    _dspark_ragged_token_rows_kernel[(x.shape[0],)](
        query_start_loc,
        token_request,
        token_within,
        NUM_SLOTS=num_slots,
        BLOCK_S=block_s,
    )
    grid = (x.shape[0], triton.cdiv(x.shape[1], block_d))
    if intermediate_conv_window is None:
        intermediate_conv_window = x
        intermediate_state_indices = cache_indices
        inter_strides = (0, 0, 0, 0)
        num_intermediate_rows = 1
        max_intermediate_steps = x.shape[0]
    else:
        if intermediate_conv_window.ndim != 4:
            raise ValueError(
                "intermediate_conv_window must have shape [N_i,W,D,K-1], "
                f"got {tuple(intermediate_conv_window.shape)}"
            )
        if (
            intermediate_conv_window.shape[2] != x.shape[1]
            or intermediate_conv_window.shape[3] < weight.shape[1] - 1
        ):
            raise ValueError(
                "intermediate_conv_window channel/window dimensions disagree: "
                f"intermediate={tuple(intermediate_conv_window.shape)}, "
                f"x={tuple(x.shape)}, weight={tuple(weight.shape)}"
            )
        if intermediate_state_indices.shape[0] < num_slots:
            raise ValueError(
                f"intermediate_state_indices must contain at least {num_slots} "
                f"rows, got {tuple(intermediate_state_indices.shape)}"
            )
        inter_strides = intermediate_conv_window.stride()
        num_intermediate_rows = intermediate_conv_window.shape[0]
        max_intermediate_steps = intermediate_conv_window.shape[1]

    _dspark_ragged_causal_conv1d_update_kernel[grid](
        x,
        conv_state,
        weight,
        bias if bias is not None else x,
        token_request,
        token_within,
        cache_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        out,
        x.shape[1],
        conv_state.shape[2],
        x.stride(0),
        x.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        weight.stride(1),
        inter_strides[0],
        inter_strides[1],
        inter_strides[2],
        inter_strides[3],
        out.stride(0),
        out.stride(1),
        NUM_SLOTS=num_slots,
        NUM_CACHE_LINES=conv_state.shape[0],
        NUM_INTERMEDIATE_ROWS=num_intermediate_rows,
        MAX_INTERMEDIATE_STEPS=max_intermediate_steps,
        KERNEL_WIDTH=weight.shape[1],
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        SAVE_INTERMEDIATE=save_intermediate,
        BLOCK_D=block_d,
    )
    return out
