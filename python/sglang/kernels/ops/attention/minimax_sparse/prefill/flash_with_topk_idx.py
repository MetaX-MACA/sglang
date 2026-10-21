# Copyright 2025 XunhaoLai. All rights reserved.
"""Standalone prefill sparse-attention kernels (bf16 / fp16 only).

Step 1 (lightning indexer): ``flash_prefill_with_topk_index`` — score every KV
block with the index head and select the top-k blocks (in-kernel bitonic top-k).
Optionally also produces the index-head attention output.

Step 3 (main sparse attention): ``flash_prefill_with_gqa_share_sparse`` —
attend only to the selected top-k blocks with the main head.

No fp8, no sglang, no project utils. Paged KV via ``req_to_token`` / ``slot_ids``.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ..common.utils import (
    _bitonic_merge,
    check_sparse_kv_fp8,
    get_cu_seqblocks,
    sparse_out_dtype,
    unit_scale,
)

# --------------------------------------------------------------------------- #
# Step 1: score + (optional) index-head attention, with in-kernel top-k score  #
# --------------------------------------------------------------------------- #
@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        # BLOCK_SIZE_K is now the K/V sub-tile (block_size=bsk is processed in
        # SUB_TILES_PER_BLOCK = block_size//BLOCK_SIZE_K sub-tiles). Small
        # sub-tile -> small smem -> high occupancy on 64KB-smem/SM. 16 is the
        # bf16 mma K-dim floor; 32 is the larger alternative. num_stages=1
        # fits <=64KB; larger stages are pruned where they exceed the limit.
        # Note: BSQ=32 raises occupancy (18% vs 6%) but step1 is compute-bound
        # (~16 TFLOPS), so halving the Q tile costs more arithmetic intensity
        # than occupancy gains —实测 BSQ32 慢一倍, 故不纳入。BSQ64 BSK32 最优。
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 16}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 16}, num_warps=2, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 32}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 16}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 32}, num_warps=8, num_stages=1
        ),
    ],
    key=[
        "qk_head_dim",
        "v_head_dim",
        "block_size",
        "use_gumbel_topk",
    ],
)
@triton.jit
def _flash_attn_fwd_with_block_score_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    o_ptr,  # O: n x h x d
    score_ptr,  # Score: h x n x max_seqblock
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    block_size: tl.constexpr,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # gumbel topk
    use_gumbel_topk: tl.constexpr,
    gumbel_seed,
    # stride
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    DISABLE_INDEX_VALUE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    # BLOCK_SIZE_K is the K/V sub-tile; block_size is the topk block size (bsk).
    # A selected block of `block_size` tokens is processed in sub-tiles of
    # BLOCK_SIZE_K (block_size must be a multiple of BLOCK_SIZE_K). Smaller
    # sub-tile -> smaller K/V shared-mem footprint -> higher occupancy on the
    # 64KB-smem/SM device (see step3 kernel for the same trick). Each bsk-block
    # writes exactly one score column, accumulated across its sub-tiles.
    tl.static_assert(block_size % BLOCK_SIZE_K == 0)
    SUB_TILES_PER_BLOCK: tl.constexpr = block_size // BLOCK_SIZE_K
    # get batch id and head id
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    block_num = (seq_len + block_size - 1) // block_size
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, qk_head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    s_ptrs = tl.make_block_ptr(
        base=score_ptr + seq_start * stride_s_q + pid_h * stride_s_h,
        shape=(q_len, block_num),
        strides=(stride_s_q, stride_s_k),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK:
        off_d = tl.arange(0, BLOCK_SIZE_KD)
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=off_d < qk_head_dim,
            other=0,
        )
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    if HAS_SINK:
        m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        lse_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        qsink = (
            tl.sum(q.to(tl.float32) * sink[None, :].to(tl.float32), axis=1)
            * sm_scale_log2e
        )  # (BLOCK_SIZE_Q,)
        m_i += qsink
        lse_i += qsink
    else:
        m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    # attention. Outer loop steps by block_size (one bsk-block -> one score
    # column); inner loop splits that block into SUB_TILES_PER_BLOCK sub-tiles
    # of BLOCK_SIZE_K so the K/V shared-mem tile stays small (high occupancy).
    # Online softmax (m/lse/acc_o) and the per-block score (max/lse) both
    # accumulate across sub-tiles, so the split is exact.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, block_size):
        # accumulate this bsk-block's score across its sub-tiles.
        # max: running max of per-token qk. lse: logsumexp via running (max, sum).
        blk_score_max = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        blk_lse_m = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        blk_lse_s = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        for s in tl.static_range(0, SUB_TILES_PER_BLOCK):
            ks = i + s * BLOCK_SIZE_K
            # paged load K via req_to_token: pos -> slot -> k_cache
            pos = ks + off_k
            pos_mask = pos < seq_len
            slots = tl.load(
                req_to_token_ptr + sid * stride_r2t_b + pos,
                mask=pos_mask,
                other=0,
            ).to(tl.int64)
            slots = (slots + max_slots) % max_slots  # safety against negative
            # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
            k = tl.load(
                k_cache_ptr
                + slots[None, :] * stride_k_s
                + pid_kh * stride_k_h
                + off_kd[:, None] * stride_k_d,
                mask=kd_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                k = k.to(q.dtype)
            # compute qk
            qk = tl.dot(q, k) * (sm_scale_log2e * k_scale)
            # causal mask: a query at off_q attends to keys at abs pos ks+off_k
            qk = tl.where(off_q[:, None] >= (ks + off_k)[None, :], qk, float("-inf"))
            # K boundary mask: positions beyond seq_len contribute -inf
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            # accumulate per-block score (max or lse) across sub-tiles
            sub_max = tl.max(qk, axis=1)  # (BLOCK_SIZE_Q,)
            if SCORE_TYPE == "max":
                blk_score_max = tl.maximum(blk_score_max, sub_max)
            else:  # "lse": logsumexp merge of this sub-tile into the block
                new_m = tl.maximum(blk_lse_m, sub_max)
                blk_lse_s = blk_lse_s * tl.exp2(blk_lse_m - new_m) + tl.sum(
                    tl.exp2(qk - new_m[:, None]), axis=1
                )
                blk_lse_m = new_m
            if not DISABLE_INDEX_VALUE:
                # compute m_ij and l_ij (online softmax over all sub-tiles)
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                # scale acc_o
                acc_o_scale = tl.exp2(m_i - m_ij)
                acc_o = acc_o * acc_o_scale[:, None]
                # paged load V (sub-tile, same slots)
                v = tl.load(
                    v_cache_ptr
                    + slots[:, None] * stride_v_s
                    + pid_kh * stride_v_h
                    + off_vd[None, :] * stride_v_d,
                    mask=pos_mask[:, None] & vd_mask[None, :],
                    other=0.0,
                )
                if IS_FP8:
                    v = v.to(q.dtype)
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v) * v_scale
                # update statistics
                m_i = m_ij
                lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # finalize and save this bsk-block's score (one column)
        if SCORE_TYPE == "max":
            score = blk_score_max
        else:  # "lse"
            score = blk_lse_m + tl.log2(blk_lse_s)
            # fully-masked block (all -inf) -> NaN; clamp to -inf sentinel
            score = tl.where(score != score, float("-inf"), score)
        if use_gumbel_topk:
            local_seed = (pid_h | (pid_b << 7) | (gumbel_seed << 19)).to(tl.int32)
            noise_offset = (off_q << 13) | (i // block_size)
            noise = tl.rand(local_seed, offset=noise_offset)
            noise = tl.clamp(noise, min=1e-9, max=1 - 1e-9)  # avoid log(0)
            noise = -tl.log(-tl.log(noise)) * 1.4426950409
            score = score + noise
        tl.store(s_ptrs, tl.reshape(score, (BLOCK_SIZE_Q, 1)).to(score_ptr.dtype.element_ty), boundary_check=(0, 1))
        # advance score ptr to next bsk-block column
        s_ptrs = tl.advance(s_ptrs, (0, 1))
    if not DISABLE_INDEX_VALUE:
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + seq_start * stride_o_n + pid_h * stride_o_h,
            shape=(q_len, v_head_dim),
            strides=(stride_o_n, stride_o_d),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=[
        "topk"
    ],  # topk gates BLOCK_SIZE_K validity (assert BLOCK_SIZE_K > topk)
)
@triton.jit
def _topk_index_kernel(
    s_ptr,  # Score: h x n x max_seqblock
    ti_ptr,  # topk_idx: h x n x topk
    # size
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    # seqlens
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    # shape
    topk,  # not constexpr to avoid recompilation when topk changes
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    # stride
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(
        BLOCK_SIZE_K > BLOCK_SIZE_T
    )  # use BLOCK_SIZE_T instead of topk (stricter but safe)
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    # offsets
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    # init qkv pointer
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    # init statistics
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    # compute topk
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        # masks
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        # load score
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        # handle NaN: NaN inputs cause bitonic sort to fail, resulting in invalid indices (-2)
        # appearing in the topk list. We replace NaN with -inf to maintain sort order.
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        # fill init and local part, make sure init part is always in topk
        # and at the first position. Note: must use causal_mask to protect
        # init_mask to avoid selecting blocks outside causal window
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        # bitonic merge
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # get topk, shape: [BLOCK_SIZE_T,]
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    # save topk
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    topk_mask = tl.arange(0, BLOCK_SIZE_T) < min(topk, valid_blocks)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=topk_mask)


@torch.no_grad()
def flash_prefill_with_topk_index(
    q: torch.Tensor,
    k_cache: torch.Tensor,  # paged
    v_cache: Optional[torch.Tensor],  # paged; ignored when disable_index_value=True
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
):
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"
    # triton.set_allocator(robust_allocator)
    # dtype check (v_cache is None under disable_index_value)
    is_fp8 = check_sparse_kv_fp8(
        q, k_cache, None if disable_index_value else v_cache, label="prefill indexer"
    )
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    assert cu_seqlens.dtype == torch.int32
    # shape
    total_q, num_heads, qk_head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    if disable_index_value:
        v_head_dim = qk_head_dim  # placeholder for BLOCK_SIZE_VD; V is never loaded
    else:
        assert v_cache is not None
        assert v_cache.shape[1] == k_cache.shape[1]
        check_sparse_kv_fp8(q, k_cache, v_cache)
        v_head_dim = v_cache.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, "head_dim must be less than 256"
    if sink is not None:
        assert sink.shape[0] == num_heads and sink.shape[1] == qk_head_dim
    assert (
        init_blocks + local_blocks <= topk
    ), "init_blocks + local_blocks must be less than topk"
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    # q_scale multiplies every Q-side logit (QK dot and sink), so it folds into
    # sm_scale; k_scale must not touch the sink term and stays a kernel arg.
    sm_scale = sm_scale * unit_scale(q_scale)
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    if disable_index_value:
        o = None
    else:
        o = torch.empty(
            total_q, num_heads, v_head_dim, dtype=sparse_out_dtype(q), device=q.device
        )
    score = torch.full(
        (num_heads, total_q, max_seqblock_k),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    # launch kernel
    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads)

    _flash_attn_fwd_with_block_score_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        o,
        score,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        block_size_k,
        sm_scale,
        k_scale,
        v_scale,
        False,
        1,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0) if v_cache is not None else 0,
        v_cache.stride(1) if v_cache is not None else 0,
        v_cache.stride(2) if v_cache is not None else 0,
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        o.stride(0) if o is not None else 0,
        o.stride(1) if o is not None else 0,
        o.stride(2) if o is not None else 0,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=disable_index_value,
        IS_FP8=is_fp8,
    )

    # topk extraction kernel
    topk_idx = torch.full(
        (num_heads, all_seqblock_q, topk),
        fill_value=-1,
        device=score.device,
        dtype=torch.int32,
    )
    # launch kernel
    grid = (max_seqblock_q, batch_size, num_heads)
    _topk_index_kernel[grid](
        score,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
    )
    return o, topk_idx
