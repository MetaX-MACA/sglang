# Copyright 2025 XunhaoLai. All rights reserved.
"""Standalone decode sparse-attention kernels (bf16 / fp16 only).

Step 1 (lightning indexer): ``flash_decode_with_topk_idx`` — score every KV
block with the index head and select the top-k blocks (2-stage split-K Triton
topk). Optionally also produces the index-head attention output.

Step 3 (main sparse attention): ``flash_decode_with_gqa_share_sparse`` —
attend only to the selected top-k blocks with the main head.

No fp8, no sglang, no project utils. Paged KV via ``req_to_token`` / ``slot_ids``.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ..common.utils import (
    check_sparse_kv_fp8,
    sparse_out_dtype,
    unit_scale,
)


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
        "BATCH_SIZE_BUCKET": lambda args: triton.next_power_of_2(args["batch_size"]),
        "HAS_HISPARSE_SLOTS": lambda args: args["hisparse_slots_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8]
        for ns in [1, 2, 3]
    ],
    key=["BATCH_SIZE_BUCKET", "gqa_group_size", "head_dim", "BLOCK_SIZE_N", "HAS_SINK"],
)
@triton.jit
def _gqa_share_sparse_decode_kernel(
    q_ptr,  # Q: b x qh x d
    sink_ptr,  # Sink: qh x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    idx_ptr,  # topk index: qh x b x topk
    o_ptr,  # O partial: c x b x qh x d
    lse_ptr,  # lse partial: c x b x qh
    hisparse_slots_ptr,  # pre-resolved device slots: kh x b x (topk * block)
    seq_lens,
    slot_ids,
    # shape
    max_slots,
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    hisparse_slots_stride_h,
    hisparse_slots_stride_b,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # stride
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_r2t_b,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # META parameters
    BATCH_SIZE_BUCKET: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    block_size: tl.constexpr,
    IS_FP8: tl.constexpr,
    HAS_HISPARSE_SLOTS: tl.constexpr,
):
    # decode program ids: split-K over the topk dimension to give every SM
    # something to do at small batch. pid(0) folds (batch, chunk) together so
    # the grid size = batch_size * NUM_TOPK_CHUNKS.
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    # per-chunk topk range. chunk_size is *runtime* (depends on max_topk which
    # is a runtime arg, not constexpr), so don't annotate as tl.constexpr —
    # doing so produces undefined behavior in Triton.
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk_compiletime = chunk_start_topk + chunk_size_topk
    # get q k start and len after rmpad
    seq_len = tl.minimum(tl.load(seq_lens + pid_b), max_kv_len)
    # number of KV blocks; used to clamp stray topk_idx entries (see loop below).
    num_blocks = (seq_len + block_size - 1) // block_size
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # to avoid bugs when slot_ids is negative
    # get real topk
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b
    topk_idx = tl.load(idx_base + off_t * stride_ti_t, mask=off_t < max_topk, other=-1)
    valid_idx = tl.where(topk_idx >= 0, off_t, -1)
    real_topk = tl.sum((valid_idx != -1).to(tl.int32), axis=0)
    chunk_end_topk = tl.minimum(chunk_end_topk_compiletime, real_topk)
    # init pointer
    SUB_TILES_PER_BLOCK: tl.constexpr = block_size // BLOCK_SIZE_N
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
    # init statistics — kept at -inf so empty chunks (chunk_start >= real_topk)
    # naturally fall out as weight=0 in the merge step.
    # NOTE: nest `if HAS_SINK:` (constexpr, prunes sink_ptr use when False)
    # around the runtime `if pid_c == 0:` — a single `if HAS_SINK and pid_c == 0`
    # is NOT statically pruned by the compiler and would dereference sink_ptr
    # (None) even when HAS_SINK is False.
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK:
        if pid_c == 0:
            sink_ptrs = tl.make_block_ptr(
                base=sink_ptr + pid_h * stride_sink_h,
                shape=(gqa_group_size, head_dim),
                strides=(stride_sink_h, stride_sink_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
                order=(1, 0),
            )
            sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale  # (BLOCK_SIZE_H,)
            m_i = qsink
            lse_i = qsink
        else:
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # only iterate over this chunk's topk slice. the load must respect the
    # per-chunk start offset.
    cur_idx_ptr = idx_base + chunk_start_topk * stride_ti_t
    hisparse_topk_counter = chunk_start_topk
    for _ in tl.range(chunk_start_topk, chunk_end_topk, num_stages=2):
        # load index; c is the absolute start of this bsk-block (block_size
        # tokens). Clamp out-of-range ids to 0: under cuda-graph capture the
        # topk_idx buffer may be uninitialized, and a huge positive entry makes
        # `raw * block_size` overflow int32 to a negative c; then pos = c + ...
        # < 0 < seq_len is spuriously true and the masked req_to_token gather
        # dereferences an unmapped address (MetaX Xnack/ATU fault). Legal ids
        # are untouched on the real path.
        raw = tl.load(cur_idx_ptr).to(tl.int32)
        c = tl.where((raw >= 0) & (raw < num_blocks), raw, 0).to(tl.int32) * block_size
        cur_idx_ptr = cur_idx_ptr + stride_ti_t
        # split the bsk-block into sub-tiles of BLOCK_SIZE_N to shrink the K/V
        # shared-mem tile and raise occupancy (see prefill step3). Online
        # softmax accumulates across sub-tiles, so the split is exact.
        for s in tl.static_range(0, SUB_TILES_PER_BLOCK):
            cs = c + s * BLOCK_SIZE_N
            # resolve slots for this sub-tile via req_to_token
            pos = cs + off_n
            pos_mask = pos < seq_len
            if HAS_HISPARSE_SLOTS:
                slots = tl.load(
                    hisparse_slots_ptr
                    + pid_kh * hisparse_slots_stride_h
                    + pid_b * hisparse_slots_stride_b
                    + hisparse_topk_counter * block_size
                    + s * BLOCK_SIZE_N + off_n,
                    mask=pos_mask, other=0,
                ).to(tl.int64)
            else:
                slots = tl.load(
                    req_to_token_ptr + sid * stride_r2t_b + pos,
                    mask=pos_mask,
                    other=0,
                ).to(tl.int64)
            slots = (slots + max_slots) % max_slots  # safety against negative
            # load K as (head_dim, BLOCK_SIZE_N) via indirect addressing
            k_off = (
                slots[None, :] * stride_k_s
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_off,
                mask=dim_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            # load V as (BLOCK_SIZE_N, head_dim) via indirect addressing
            v_off = (
                slots[:, None] * stride_v_s
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d
            )
            v = tl.load(
                v_cache_ptr + v_off,
                mask=pos_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
            qk += tl.where(off_n[None, :] < seq_len - cs, 0, float("-inf"))
            # [H, D], [D, N] -> [H, N]
            if IS_FP8:
                k = k.to(q.dtype)
            qk += tl.dot(q, k) * (sm_scale * k_scale)
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            # load v and update acc_o
            if IS_FP8:
                v = v.to(q.dtype)
            # [H, N], [N, D] -> [H, D]
            acc_o += tl.dot(p.to(v.dtype), v) * v_scale
            # update statistics
            m_i = m_ij
            lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
        hisparse_topk_counter = hisparse_topk_counter + 1
    # final scale (matches the old non-split kernel for chunks where lse_i>-inf).
    # For empty chunks the inner loop never runs, so m_i = lse_i = -inf and naive
    # `tl.exp(m_i - lse_i)` would be NaN; gate with tl.where for a clean zero.
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]
    # save partial output and lse for the merge step
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH, D] — partials in, merged out at chunk 0
    lse_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    # standard flash-decoding merge in linear (not log2) space, matching the
    # decode kernel which uses tl.exp / tl.log.
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    o_out_ptrs = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h + off_d * stride_o_d
    tl.store(o_out_ptrs, o_merged.to(o_ptr.dtype.element_ty), mask=off_d < head_dim)


@torch.no_grad()
def flash_decode_with_gqa_share_sparse(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch_size, ]
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
    use_tma: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    hisparse_slots: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # triton.set_allocator(robust_allocator)
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="decode")
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    # shape
    batch_size, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    assert slot_ids.shape[0] == batch_size and seq_lens.shape[0] == batch_size
    assert topk_idx.shape[0] == num_kv_heads
    assert triton.next_power_of_2(block_size) == block_size, (
        f"block_size must be a power of 2, but got {block_size}"
    )
    # assert slot_ids.max() < max_slots, f"get slot_ids {slot_ids}, but kv_cache shape is {kv_cache.shape}"
    max_kv_len = req_to_token.shape[1]
    # gqa
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads
    max_topk = topk_idx.shape[2]
    # sm scale
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # q_scale multiplies every Q-side logit (QK dot and sink), so it folds into
    # sm_scale; k_scale must not touch the sink term and stays a kernel arg.
    sm_scale = sm_scale * unit_scale(q_scale)
    # Pick NUM_TOPK_CHUNKS so total grid ≈ TARGET_GRID. Same constraints as
    # flash_decode_with_topk_idx: must be power of 2 (Triton arange) and must
    # only depend on shape constants (so grid is fixed within a cuda graph).
    # Capped by max_topk because chunks beyond real_topk early-fall-through to
    # the merge-as-zero path; capping avoids wasting blocks at tiny topk.
    # Pick NUM_TOPK_CHUNKS so total grid ≈ TARGET_GRID. Must be power of 2 and
    # only depend on shape constants (cuda-graph friendly).
    TARGET_GRID = 256
    target = max(
        1,
        min(max_topk, TARGET_GRID // max(1, batch_size * num_kv_heads)),
    )
    NUM_TOPK_CHUNKS = 1 << (target.bit_length() - 1)
    # output tensor: split-K partials, merged into chunk 0 by the merge kernel
    o_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        head_dim,
        dtype=sparse_out_dtype(q),
        device=q.device,
    )
    lse_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        dtype=torch.float32,
        device=q.device,
    )
    # launch attention kernel
    grid = (batch_size * NUM_TOPK_CHUNKS, num_kv_heads)
    _gqa_share_sparse_decode_kernel[grid](
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        topk_idx,
        o_partial,
        lse_partial,
        hisparse_slots,
        seq_lens,
        slot_ids,
        max_slots,
        batch_size,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        hisparse_slots.stride(0) if hisparse_slots is not None else 0,
        hisparse_slots.stride(1) if hisparse_slots is not None else 0,
        sm_scale,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_size=block_size,
        BLOCK_SIZE_N=min(16, triton.next_power_of_2(block_size)),
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
        IS_FP8=is_fp8
    )
    # merge partials into chunk 0
    merge_grid = (batch_size, num_q_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
    )
    return o_partial[0].contiguous()
