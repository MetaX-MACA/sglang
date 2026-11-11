# SPDX-License-Identifier: Apache-2.0
"""GDN ReplaySSM fold-every-commit: replay the ring-written raw inputs of the
accepted draft prefix into the fp32 checkpoint on commit (replaces the
per-draft ``intermediate_ssm`` snapshots).

The fold is a BITWISE CLONE of ``fused_sigmoid_gating_delta_rule_update_kernel``'s
GDN branch (same tile, division-form L2 norm, op order). Do NOT reorder into
tl.dot / reciprocal-multiply; keep num_warps=1 so the reduction trees match.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import enable_specific_device_opt

_C600_DEVICE_OPT = enable_specific_device_opt()


@triton.jit
def gdn_replayssm_normalize_keys_kernel(
    rawk,
    normalized,
    indices,
    accept_lens,
    STRIDE_RAW_SLOT: tl.constexpr,
    STRIDE_RAW_LAYER: tl.constexpr,
    # Runtime on purpose: the scratch layer stride is B * H * RL * K, so keeping
    # it constexpr recompiles the kernel for every batch size.
    STRIDE_OUT_REQUEST: tl.constexpr,
    STRIDE_OUT_LAYER,
    STRIDE_INDICES: tl.constexpr,
    STRIDE_ACCEPT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    RL: tl.constexpr,
    BK: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    request = tl.program_id(0)
    head_layer = tl.program_id(1)
    layer = (head_layer // H).to(tl.int64)
    head = head_layer % H
    slot = tl.load(indices + request * STRIDE_INDICES).to(tl.int64)
    count = tl.load(accept_lens + request * STRIDE_ACCEPT)
    if slot <= NULL_BLOCK_ID or count <= 0:
        return
    # Keep the original one-dimensional reduction for each accepted token.
    # Processing the window here avoids launching a tiny program per token.
    key = tl.arange(0, BK)
    for step in range(0, tl.minimum(count, RL)):
        value = tl.load(
            rawk
            + layer * STRIDE_RAW_LAYER
            + slot * STRIDE_RAW_SLOT
            + (head * RL + step) * K
            + key,
            mask=key < K,
            other=0.0,
        ).to(tl.float32)
        # Preserve the recurrent baseline's division-form normalization.
        value = value / tl.sqrt(tl.sum(value * value) + 1e-6)
        tl.store(
            normalized
            + layer * STRIDE_OUT_LAYER
            + request * STRIDE_OUT_REQUEST
            + (head * RL + step) * K
            + key,
            value,
            mask=key < K,
        )


@triton.jit
def gdn_replayssm_exact_fold_kernel(
    h0,  # [num_slots, HV, K, V] fp32 checkpoint (folded in place)
    rawv_cache,  # [num_slots, HV, RL, V]  raw v
    rawk_cache,  # [num_slots, H,  RL, K]  raw pre-norm k
    g_cache,  # [num_slots, HV, RL]     fp32 scalar log-decay gate
    beta_cache,  # [num_slots, HV, RL]     fp32 beta
    ssm_state_indices,  # [B] int  physical slot per request
    accept_lens,  # [B] int  committed prefix length per request (incl. bonus)
    mamba_track_indices,  # [B] int  extra_buffer track slot (or NULL) per request
    mamba_steps_to_track,  # [B] int  crossing step (or -1) per request
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    # Runtime: points at the per-call normalized-key scratch in the precomputed
    # path, whose layer stride is B * H * RL * K.
    stride_rawk_layer,
    stride_g_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    PRECOMPUTED_KEYS: tl.constexpr = False,
):
    i_n = tl.program_id(1)
    if PRECOMPUTED_KEYS:
        # Group all heads of a state slot along the grid's contiguous axis.
        # Each value tile performs the same arithmetic as the original grid.
        i_v = tl.program_id(0) % tl.cdiv(V, BV)
        i_hv = tl.program_id(0) // tl.cdiv(V, BV)
        i_layer = tl.program_id(2).to(tl.int64)
    else:
        i_v = tl.program_id(0)
        i_hvl = tl.program_id(2)
        # int64: layer stride * i_layer can overflow an int32 index product.
        i_layer = (i_hvl // HV).to(tl.int64)
        i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    g_cache = g_cache + i_layer * stride_g_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    key_slot = i_n.to(tl.int64) if PRECOMPUTED_KEYS else state_idx
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return

    if HAS_TRACK:
        track_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        track_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    else:
        track_idx = NULL_BLOCK_ID
        track_step = -1

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Checkpoint-bandwidth bound. Tuning tried and rejected: num_stages (no-op),
    # num_warps > 1 (breaks the bitwise clone), evict_first (loses cold-L2).
    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, n_commit):
        phys = t.to(tl.int64)
        b_k = tl.load(
            rawk_cache
            + key_slot * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv_cache
            + state_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_g = tl.load(
            g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)
        b_beta = tl.load(
            beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        if HAS_TRACK:
            if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                tl.store(
                    h0
                    + track_idx * stride_state_slot
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None],
                    b_h.to(h0.dtype.element_ty),
                    mask=mask_h,
                )

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def commit_gdn_replayssm_fold_all_layers(
    checkpoint_state: torch.Tensor,  # [num_layers, num_slots, HV, K, V], in place
    rawv_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL, V]
    rawk_cache: torch.Tensor,  # [num_layers, num_slots, H,  RL, K]
    g_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL] fp32
    beta_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL] fp32
    ssm_state_indices: torch.Tensor,  # [B] int   (shared across layers)
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
    precompute_keys: bool = True,
) -> None:
    """Fold accepted windows across all layers, bit-identically to a per-layer
    loop. The device-optimized path first normalizes the accepted keys."""
    num_layers, num_slots, HV = checkpoint_state.shape[:3]
    K = rawk_cache.shape[-1]
    V = rawv_cache.shape[-1]
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    # opt for C600U
    BV = min(triton.next_power_of_2(V), 16)
    grid = (triton.cdiv(V, BV), B, HV * num_layers)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        # Unused when HAS_TRACK is False; pass a valid pointer.
        track_idx_t = ssm_state_indices
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0
    # Request-local scratch avoids allocating a normalized ring for every
    # physical state slot. PyTorch's allocator keeps it alive through the
    # current stream and owns its storage during CUDA Graph capture/replay.
    use_precomputed_keys = (
        precompute_keys
        and _C600_DEVICE_OPT
        and use_qk_l2norm_in_kernel
        and K == 128
        and V == 128
    )
    if use_precomputed_keys:
        grid = (triton.cdiv(V, BV) * HV, B, num_layers)
        normalized_keys = torch.empty(
            (num_layers, B, num_k_heads, max_cache_len, K),
            dtype=torch.float32,
            device=rawk_cache.device,
        )
        gdn_replayssm_normalize_keys_kernel[(B, num_layers * num_k_heads)](
            rawk_cache,
            normalized_keys,
            ssm_state_indices,
            accept_lens,
            rawk_cache.stride(1),
            rawk_cache.stride(0),
            normalized_keys.stride(1),
            normalized_keys.stride(0),
            ssm_state_indices.stride(0),
            accept_lens.stride(0),
            H=num_k_heads,
            K=K,
            RL=max_cache_len,
            BK=BK,
            NULL_BLOCK_ID=null_block_id,
            num_warps=1,
            num_stages=1,
        )
        rawk_cache = normalized_keys
    gdn_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        g_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(1),
        rawv_cache.stride(1),
        rawk_cache.stride(1),
        g_cache.stride(1),
        beta_cache.stride(1),
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        g_cache.stride(0),
        beta_cache.stride(0),
        ssm_state_indices.stride(0),
        accept_lens.stride(0),
        stride_track,
        stride_steps,
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel and not use_precomputed_keys,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        PRECOMPUTED_KEYS=use_precomputed_keys,
        num_warps=1,
        num_stages=1,
    )


def commit_gdn_replayssm_fold_after_verify(
    *,
    spec_state,  # MambaPool.SpeculativeState (all layers)
    state_batch_indices: torch.Tensor,  # [B] per-req mamba slot
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    last_correct_step_indices: torch.Tensor,  # [B] conv rollback target step
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:
    """Fold each layer's accepted window into ``temporal``, then do the usual
    conv accept-rollback (+ the track-slot conv scatter under extra_buffer)."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    max_cache_len = spec_state.replayssm_rawv.shape[-2]
    num_k_heads = spec_state.replayssm_rawk.shape[2]
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.replayssm_rawv,
        rawk_cache=spec_state.replayssm_rawk,
        g_cache=spec_state.replayssm_g,
        beta_cache=spec_state.replayssm_beta,
        ssm_state_indices=state_batch_indices,
        accept_lens=accept_lens,
        max_cache_len=max_cache_len,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
    )
    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states, interm_conv, state_batch_indices, last_correct_step_indices
        )
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            fused_conv_window_scatter_with_mask(
                conv_states, interm_conv, mamba_track_indices, mamba_steps_to_track
            )
