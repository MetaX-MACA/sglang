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

from ..common.utils import _check_dtypes
from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu

from ..common.utils import (
    _bitonic_merge,
    _sort_ids_ascending,
    check_sparse_kv_fp8,
    robust_allocator,
    sparse_out_dtype,
    unit_scale,
)

# Must match _MAX_NUM_BLOCKS in ops/attention/minimax_decode_topk.py.
_JIT_TOPK_MAX_NUM_BLOCKS = 16384 if torch.version.hip else 4096


def _prune_decode_configs(configs, named_args, **kwargs):
    """Keep token sub-tiles that divide a sparse block.

    BLOCKS_PER_K_BLOCK = BLOCK_SIZE_N // block_size is 0 for those, so they
    cannot compile. Keep the full list if nothing survives (block_size larger
    than every tile) and let triton report the real failure.
    """
    block_size = named_args["block_size"]
    kept = [c for c in configs if c.kwargs["BLOCK_SIZE_N"] <= block_size and block_size % c.kwargs["BLOCK_SIZE_N"] == 0]
    return kept or list(configs)


_ON_HIP = torch.version.hip is not None


def _decode_score_block_n(args):
    # gfx950 pick: 512 for tiny batches (few CTAs), else 128.
    return min(64, args["block_size"])


# On ROCm the autotune sweep is replaced by a fixed config plus the
# BLOCK_SIZE_N heuristic above: runtime autotune can fire under CUDA-graph
# capture and mistune.
_DECODE_SCORE_CONFIGS = (
    [triton.Config({}, num_warps=4, num_stages=1)]
    if _ON_HIP
    else [
        triton.Config({"BLOCK_SIZE_N": BN}, num_warps=nw, num_stages=ns)
        for BN in [16, 32, 64]
        for nw in [4, 8, 16]
        for ns in [1, 2, 3]
    ]
)

_DECODE_SCORE_HEURISTICS = {
    "BLOCK_SIZE_H": lambda args: max(
        16, triton.next_power_of_2(args["gqa_group_size"])
    ),
    "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    "BATCH_SIZE_BUCKET": lambda args: triton.next_power_of_2(args["batch_size"]),
    **({"BLOCK_SIZE_N": _decode_score_block_n} if _ON_HIP else {}),
}


@triton.heuristics(_DECODE_SCORE_HEURISTICS)
@triton.autotune(
    configs=([triton.Config({"BLOCK_SIZE_N": BN}, num_warps=nw, num_stages=ns) for BN in [16, 32, 64] for nw in [4, 8, 16] for ns in [1, 2, 3]] if "metax" in torch.__version__ else _DECODE_SCORE_CONFIGS),
    key=[
        "gqa_group_size",
        "head_dim",
        "block_size",
    ],
    prune_configs_by=(
        None if _ON_HIP else {"early_config_prune": _prune_decode_configs}
    ),
)
@triton.jit
def _decode_score_kernel(
    q_ptr,  # Q: b x qh x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    score_ptr,  # Score: qh x b x max_seqblock
    seq_lens,
    slot_ids,
    # shape
    max_slots,
    batch_size,
    gqa_group_size,
    head_dim,
    # block size
    block_size: tl.constexpr,
    topk: tl.constexpr,
    # sm_scale
    sm_scale,
    # per-tensor K dequant scale (1.0 when the cache is unit-scaled)
    k_scale,
    # init and local blocks
    init_blocks,
    local_blocks,
    # stride
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_r2t_b,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    # META parameters
    BATCH_SIZE_BUCKET: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    # BLOCK_SIZE_N is the K/V sub-tile; block_size is the topk block size (bsk).
    # A chunk is processed in sub-tiles of BLOCK_SIZE_N (block_size must be a
    # multiple of BLOCK_SIZE_N). Smaller sub-tile -> smaller K/V shared-mem
    # tile -> higher occupancy; V load also tracks K sub-tile so its gather
    # overlaps the QK dot (see prefill step1 for the same trick). Each bsk-block
    # writes one score column, accumulated across its sub-tiles.
    tl.static_assert(block_size % BLOCK_SIZE_N == 0)
    SUB_TILES_PER_BLOCK: tl.constexpr = block_size // BLOCK_SIZE_N
    # get batch id and head id
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    # block-aligned fixed-count chunked decode (grid independent of seq_len for cuda graph)
    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = (seq_len + block_size - 1) // block_size
    if SKIP_TRIVIAL_TOPK_SCORE:
        if num_blocks <= topk:
            return
    chunk_size_blocks = tl.cdiv(num_blocks, NUM_KV_CHUNKS)
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    chunk_start = chunk_start_block * block_size
    chunk_end = tl.minimum(chunk_end_block * block_size, seq_len)
    if chunk_start_block >= chunk_end_block:
        return
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    s_ptrs = tl.make_block_ptr(
        base=score_ptr + pid_b * stride_s_b + pid_h * stride_s_h,
        shape=(gqa_group_size, chunk_end_block),
        strides=(stride_s_h, stride_s_n),
        offsets=(0, chunk_start_block),
        block_shape=(BLOCK_SIZE_H, 1),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
    # pre-compute local_start outside the loop (constant for entire batch)
    local_start = tl.maximum(0, num_blocks - local_blocks)
    # prefetch first sub-tile's slots/K so the paged gather overlaps the prologue.
    r2t_base = req_to_token_ptr + sid * stride_r2t_b
    prefetch_pos = chunk_start + off_n
    prefetch_mask = prefetch_pos < seq_len
    prefetched_slots = tl.load(
        r2t_base + prefetch_pos,
        mask=prefetch_mask,
        other=0,
    ).to(tl.int64)
    first_slots = (prefetched_slots + max_slots) % max_slots
    k = tl.load(
        k_cache_ptr
        + first_slots[None, :] * stride_k_s
        + pid_kh * stride_k_h
        + off_d[:, None] * stride_k_d,
        mask=dim_mask[:, None] & prefetch_mask[None, :],
        other=0.0,
    )
    # score-only: outer loop steps by block_size (one bsk-block -> one score
    # column); inner loop splits it into SUB_TILES_PER_BLOCK sub-tiles of
    # BLOCK_SIZE_N. Score (max/lse) accumulates across sub-tiles. K is
    # prefetched one sub-tile ahead to overlap gather with the QK dot.
    for blk in range(chunk_start_block, chunk_end_block):
        i = blk * block_size
        blk_score_max = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        blk_lse_m = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        blk_lse_s = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
        for s in tl.static_range(0, SUB_TILES_PER_BLOCK):
            ks = i + s * BLOCK_SIZE_N
            pos_mask = prefetch_mask
            slots = prefetched_slots
            # prefetch next sub-tile's slots+K
            next_ks = ks + BLOCK_SIZE_N
            if next_ks < chunk_end:
                next_pos = next_ks + off_n
                prefetch_mask = next_pos < seq_len
                prefetched_slots = tl.load(
                    r2t_base + next_pos,
                    mask=prefetch_mask,
                    other=0,
                ).to(tl.int64)
                next_slots = (prefetched_slots + max_slots) % max_slots
                k_next = tl.load(
                    k_cache_ptr
                    + next_slots[None, :] * stride_k_s
                    + pid_kh * stride_k_h
                    + off_d[:, None] * stride_k_d,
                    mask=dim_mask[:, None] & prefetch_mask[None, :],
                    other=0.0,
                )
            else:
                k_next = k
            slots = (slots + max_slots) % max_slots
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
            qk += tl.where(off_n[None, :] < chunk_end - ks, 0, float("-inf"))
            # [H, D], [D, N] -> [H, N]
            if IS_FP8:
                k = k.to(q.dtype)
            qk += tl.dot(q, k) * (sm_scale_log2e * k_scale)
            k = k_next
            # accumulate this sub-tile into the bsk-block score
            sub_max = tl.max(qk, axis=1)  # (BLOCK_SIZE_H,)
            if SCORE_TYPE == "max":
                blk_score_max = tl.maximum(blk_score_max, sub_max)
            else:  # "lse"
                new_m = tl.maximum(blk_lse_m, sub_max)
                blk_lse_s = blk_lse_s * tl.exp2(blk_lse_m - new_m) + tl.sum(
                    tl.exp2(qk - new_m[:, None]), axis=1
                )
                blk_lse_m = new_m
        # finalize bsk-block score
        if SCORE_TYPE == "max":
            score = blk_score_max
        else:
            score = blk_lse_m + tl.log2(blk_lse_s)
            score = tl.where(score != score, float("-inf"), score)
        # apply init_blocks and local_blocks (scalar, per bsk-block)
        curr_block_idx = blk
        is_init = curr_block_idx < init_blocks
        is_local = (curr_block_idx >= local_start) & (curr_block_idx < num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(s_ptrs, tl.reshape(score, (BLOCK_SIZE_H, 1)).to(score_ptr.dtype.element_ty), boundary_check=(0, 1))
        s_ptrs = tl.advance(s_ptrs, (0, 1))


# --------------------------------------------------------------------------- #
# Step 1b: score + index-head attention kernel                                #
# --------------------------------------------------------------------------- #
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": BN}, num_warps=nw, num_stages=ns)
        for BN in [16, 32, 64]
        for nw in [4, 8, 16]
        for ns in [1, 2, 3]
    ],
    key=[
        "gqa_group_size",
        "head_dim",
        "block_size",
    ],
    prune_configs_by={"early_config_prune": _prune_decode_configs},
)
@triton.jit
def _decode_score_attn_kernel(
    q_ptr,  # Q: b x qh x d
    sink_ptr,  # Sink: qh x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    o_ptr,  # O: c x b x qh x d
    lse_ptr,  # lse: c x b x qh
    score_ptr,  # Score: qh x b x max_seqblock
    seq_lens,
    slot_ids,
    # shape
    max_slots,
    batch_size,
    gqa_group_size,
    head_dim,
    # block size
    block_size: tl.constexpr,
    topk: tl.constexpr,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # init and local blocks
    init_blocks,
    local_blocks,
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
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    # META parameters
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    # BLOCK_SIZE_N is the K/V sub-tile; block_size is the topk block size (bsk).
    # A chunk is processed in sub-tiles of BLOCK_SIZE_N (block_size must be a
    # multiple of BLOCK_SIZE_N). Smaller sub-tile -> smaller K/V shared-mem
    # tile -> higher occupancy; V load also tracks K sub-tile so its gather
    # overlaps the QK dot (see prefill step1 for the same trick). Each bsk-block
    # writes one score column, accumulated across its sub-tiles.
    tl.static_assert(block_size % BLOCK_SIZE_N == 0)
    SUB_TILES_PER_BLOCK: tl.constexpr = block_size // BLOCK_SIZE_N
    # get batch id and head id
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    # block-aligned fixed-count chunked decode (grid independent of seq_len for cuda graph)
    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = (seq_len + block_size - 1) // block_size
    chunk_size_blocks = tl.cdiv(num_blocks, NUM_KV_CHUNKS)
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    chunk_start = chunk_start_block * block_size
    chunk_end = tl.minimum(chunk_end_block * block_size, seq_len)
    if chunk_start_block >= chunk_end_block:
        return
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # to avoid bugs when slot_ids is negative
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    s_ptrs = tl.make_block_ptr(
        base=score_ptr + pid_b * stride_s_b + pid_h * stride_s_h,
        shape=(gqa_group_size, chunk_end_block),
        strides=(stride_s_h, stride_s_n),
        offsets=(0, chunk_start_block),
        block_shape=(BLOCK_SIZE_H, 1),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
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
            qsink = (
                tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale_log2e
            )  # (BLOCK_SIZE_H,)
            m_i = qsink
            l_i = tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32)
        else:
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # pre-compute local_start outside the loop (constant for entire batch)
    local_start = tl.maximum(0, num_blocks - local_blocks)
    # prefetch first sub-tile's slots/K/V so the paged gather overlaps the prologue.
    r2t_base = req_to_token_ptr + sid * stride_r2t_b
    prefetch_pos = chunk_start + off_n
    prefetch_mask = prefetch_pos < seq_len
    prefetched_slots = tl.load(
        r2t_base + prefetch_pos, mask=prefetch_mask, other=0
    ).to(tl.int64)
    first_slots = (prefetched_slots + max_slots) % max_slots
    k = tl.load(
        k_cache_ptr
        + first_slots[None, :] * stride_k_s
        + pid_kh * stride_k_h
        + off_d[:, None] * stride_k_d,
        mask=dim_mask[:, None] & prefetch_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        v_cache_ptr
        + first_slots[:, None] * stride_v_s
        + pid_kh * stride_v_h
        + off_d[None, :] * stride_v_d,
        mask=prefetch_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    # outer loop steps by block_size (one bsk-block -> one score column); inner
    # loop splits it into SUB_TILES_PER_BLOCK sub-tiles of BLOCK_SIZE_N. Score
    # (max/lse) and the online softmax (m/l/acc_o) both accumulate across
    # sub-tiles, so the split is exact. K/V prefetched one sub-tile ahead.
    for blk in range(chunk_start_block, chunk_end_block):
        i = blk * block_size
        blk_score_max = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        blk_lse_m = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        blk_lse_s = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
        for s in tl.static_range(0, SUB_TILES_PER_BLOCK):
            ks = i + s * BLOCK_SIZE_N
            pos_mask = prefetch_mask
            slots = prefetched_slots
            # prefetch next sub-tile's slots/K/V
            next_ks = ks + BLOCK_SIZE_N
            if next_ks < chunk_end:
                next_pos = next_ks + off_n
                prefetch_mask = next_pos < seq_len
                prefetched_slots = tl.load(
                    r2t_base + next_pos, mask=prefetch_mask, other=0
                ).to(tl.int64)
                next_slots = (prefetched_slots + max_slots) % max_slots
                k_next = tl.load(
                    k_cache_ptr
                    + next_slots[None, :] * stride_k_s
                    + pid_kh * stride_k_h
                    + off_d[:, None] * stride_k_d,
                    mask=dim_mask[:, None] & prefetch_mask[None, :],
                    other=0.0,
                )
                v_next = tl.load(
                    v_cache_ptr
                    + next_slots[:, None] * stride_v_s
                    + pid_kh * stride_v_h
                    + off_d[None, :] * stride_v_d,
                    mask=prefetch_mask[:, None] & dim_mask[None, :],
                    other=0.0,
                )
            else:
                k_next = k
                v_next = v
            slots = (slots + max_slots) % max_slots  # safety against negative
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
            qk += tl.where(off_n[None, :] < chunk_end - ks, 0, float("-inf"))
            # [H, D], [D, N] -> [H, N]
            if IS_FP8:
                k = k.to(q.dtype)
            qk += tl.dot(q, k) * (sm_scale_log2e * k_scale)
            # accumulate this sub-tile into the bsk-block score
            sub_max = tl.max(qk, axis=1)  # (BLOCK_SIZE_H,)
            if SCORE_TYPE == "max":
                blk_score_max = tl.maximum(blk_score_max, sub_max)
            else:  # "lse"
                new_m = tl.maximum(blk_lse_m, sub_max)
                blk_lse_s = blk_lse_s * tl.exp2(blk_lse_m - new_m) + tl.sum(
                    tl.exp2(qk - new_m[:, None]), axis=1
                )
                blk_lse_m = new_m
            # online softmax attention update (per sub-tile)
            m_ij = tl.maximum(m_i, sub_max)
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            # [H, N], [N, D] -> [H, D]
            if IS_FP8:
                v = v.to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v) * v_scale
            m_i = m_ij
            l_i = l_i * acc_o_scale + l_ij
            # advance to prefetched K/V for the next sub-tile
            k = k_next
            v = v_next
        # finalize and save this bsk-block's score (one column)
        if SCORE_TYPE == "max":
            score = blk_score_max
        else:
            score = blk_lse_m + tl.log2(blk_lse_s)
            score = tl.where(score != score, float("-inf"), score)
        curr_block_idx = blk
        is_init = curr_block_idx < init_blocks
        is_local = (curr_block_idx >= local_start) & (curr_block_idx < num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(s_ptrs, tl.reshape(score, (BLOCK_SIZE_H, 1)).to(score_ptr.dtype.element_ty), boundary_check=(0, 1))
        s_ptrs = tl.advance(s_ptrs, (0, 1))
    # final scale
    acc_o = acc_o / l_i[:, None]
    # compute lse in log2 scale for merge kernel
    lse_i = m_i + tl.log2(l_i)
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    # save lse (in log2 scale, will be used directly in merge kernel)
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
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8]
        for ns in [2, 3, 4]
    ],
    key=["BLOCK_SIZE_D"],
    restore_value=["o_ptr"],
)
@triton.jit
def _merge_attn_out_kernel(
    o_ptr,
    lse_ptr,
    # shape
    seq_lens,
    head_dim,
    block_size: tl.constexpr,
    # stride
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    NUM_KV_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    # recompute chunk layout (must match _decode_score_attn_kernel)
    seq_len = tl.load(seq_lens + pid_b)
    num_blocks = (seq_len + block_size - 1) // block_size
    chunk_size_blocks = tl.cdiv(num_blocks, NUM_KV_CHUNKS)
    valid_chunks = tl.cdiv(num_blocks, chunk_size_blocks)
    # ptrs
    off_c = tl.arange(0, NUM_KV_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(valid_chunks, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_KV_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    # load o and lse
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs, mask=off_c < valid_chunks, other=float("-inf"))
    # merge o (lse is in log2 scale, use exp2/log2 for efficiency)
    lse_shifted = lse - tl.max(lse, axis=0)
    log_sum = tl.log2(tl.sum(tl.exp2(lse_shifted), axis=0))
    scale = tl.exp2(lse_shifted - log_sum)
    o = tl.sum(o * scale[:, None], axis=0)
    # save o
    o_ptrs = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h + off_d * stride_o_d
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=off_d < head_dim)


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["topk"],
)
@triton.jit
def _topk_index_partial_kernel(
    s_ptr,  # Score: [num_q_heads, BS, max_seqblock]
    ts_partial_ptr,  # Partial scores out: [NUM_TOPK_CHUNKS, num_q_heads, BS, BLOCK_SIZE_T]
    ti_partial_ptr,  # Partial idx out (1-indexed global, 0=invalid): same shape
    seq_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    chunk_blocks: tl.constexpr,  # how many score-blocks each chunk owns
    # strides
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    # META
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > topk)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    seq_len = tl.load(seq_lens + pid_b)
    num_blocks = (seq_len + block_size - 1) // block_size

    # Slice this chunk owns within [0, num_blocks).
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    chunk_actual = tl.maximum(chunk_end - chunk_start, 0)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)

    # Pointer starts at chunk_start.
    s_ptrs = (
        s_ptr
        + pid_b * stride_s_b
        + pid_h * stride_s_h
        + (chunk_start + off_k) * stride_s_k
    )

    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2

    # Streaming top-K within this chunk. tl.range(0, 0) is a no-op so empty
    # chunks (chunk_actual == 0) skip the body and store sentinel -1e30 / 0.
    for i in tl.range(0, chunk_actual, BLOCK_SIZE_K):
        mask = off_k < chunk_actual - i
        score = tl.load(s_ptrs, mask=mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (
            tl.where(mask, chunk_start + i + off_k + 1, 0),  # 1-indexed global
            topk_idx,
        )
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

    # Extract first BLOCK_SIZE_T entries (top-K of this chunk after descending sort).
    topk_mask_extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_score = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_score, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    final_idx = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_idx, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    # Store partial. Always write all BLOCK_SIZE_T slots — invalid slots carry
    # -1e30 / 0 sentinels and lose to real scores in the merge stage.
    ts_ptrs = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_b * stride_ts_b
        + pid_h * stride_ts_h
        + off_t * stride_ts_t
    )
    ti_ptrs = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_b * stride_ti_b
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    tl.store(ts_ptrs, final_score)
    tl.store(ti_ptrs, final_idx)


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["NUM_TOPK_CHUNKS"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit
def _topk_index_merge_kernel(
    ts_partial_ptr,  # Partial scores: [NUM_TOPK_CHUNKS, num_q_heads, BS, BLOCK_SIZE_T]
    ti_partial_ptr,  # Partial idx (1-indexed global, 0=invalid): same shape
    ti_final_ptr,  # Final idx (0-indexed global, -1=invalid): [num_q_heads, BS, topk]
    seq_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    # strides
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    # META
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_len = tl.load(seq_lens + pid_b)
    num_blocks = (seq_len + block_size - 1) // block_size

    off = tl.arange(0, BLOCK_SIZE_K)
    chunk_idx = off // BLOCK_SIZE_T
    in_chunk_idx = off % BLOCK_SIZE_T
    valid = chunk_idx < NUM_TOPK_CHUNKS

    score_offset = (
        chunk_idx * stride_ts_c
        + pid_h * stride_ts_h
        + pid_b * stride_ts_b
        + in_chunk_idx * stride_ts_t
    )
    idx_offset = (
        chunk_idx * stride_ti_c
        + pid_h * stride_ti_h
        + pid_b * stride_ti_b
        + in_chunk_idx * stride_ti_t
    )

    score = tl.load(ts_partial_ptr + score_offset, mask=valid, other=-1e30).to(
        tl.float32
    )
    score = tl.where(score != score, -1e30, score)
    idx = tl.load(ti_partial_ptr + idx_offset, mask=valid, other=0).to(tl.int32)

    # Full bitonic descending sort of BLOCK_SIZE_K items.
    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for j in tl.static_range(1, n_dims):
        score, idx = _bitonic_merge(score, idx.to(tl.int32), j, 2, n_dims)
    score, idx = _bitonic_merge(score, idx.to(tl.int32), n_dims, True, n_dims)

    # Extract first BLOCK_SIZE_T positions — these are the global top-K.
    extract_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx_final = tl.sum(
        extract_mask[:, None]
        * tl.reshape(idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    off_t = tl.arange(0, BLOCK_SIZE_T)
    tif_ptrs = (
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t
    )
    # Ascending by block id, -1 tail: the MSA fmha_sm100 consumer requires
    # sorted kv_block_indexes (the bitonic pass above orders by score).
    topk_idx_final = _sort_ids_ascending(
        topk_idx_final, tl.minimum(topk, num_blocks), BLOCK_SIZE_T
    )
    tl.store(
        tif_ptrs, topk_idx_final.to(ti_final_ptr.dtype.element_ty), mask=off_t < topk
    )


@torch.no_grad()
def flash_decode_with_topk_idx(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: Optional[torch.Tensor],  # paged; ignored when disable_index_value=True
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_dense_main_attn: bool = False,  # NOTE: need transform idx in this case
    page_size: int = 1,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"
    triton.set_allocator(robust_allocator)
    # dtype check (v_cache is None under disable_index_value)
    is_fp8 = check_sparse_kv_fp8(
        q, k_cache, None if disable_index_value else v_cache, label="decode indexer"
    )
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    if not disable_index_value:
        assert v_cache is not None
        _check_dtypes(q, k_cache, v_cache)
    batch_size, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    max_kv_len = req_to_token.shape[1]
    assert slot_ids.shape[0] == batch_size and seq_lens.shape[0] == batch_size
    # gqa
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads
    # sm scale
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # q_scale folds exactly into sm_scale: it multiplies every Q-side logit —
    # the QK dot AND the sink logit — unlike k_scale, which must not touch the
    # sink term and therefore stays a separate kernel argument.
    sm_scale = sm_scale * unit_scale(q_scale)
    # NUM_KV_CHUNKS controls how many parallel chunks each (batch, kv_head) gets.
    # Total CTAs = batch_size * NUM_KV_CHUNKS * num_kv_heads.
    # TARGET_GRID is the desired total CTA count; NUM_KV_CHUNKS is derived by:
    #   NUM_KV_CHUNKS = clamp(TARGET_GRID // (BS * num_kv_heads), 1, MAX_NUM_KV_CHUNKS)
    # then rounded down to the nearest power of 2 (merge kernel requires it).
    # Must only depend on cuda-graph-constant quantities (BS, num_kv_heads), not seq_len.
    # Empty chunks early-return cheaply, so over-chunking is nearly free.
    # E.g. with num_kv_heads=1: BS=1→NKC=256,CTAs=256; BS=32→NKC=128,CTAs=4096.
    # NUM_KV_CHUNKS: grid independent of seq_len (cuda-graph friendly). Empty
    # chunks early-return cheaply, so over-chunking is nearly free — BUT each
    # chunk must hold >= MIN_BLOCKS_PER_CHUNK blocks so the in-kernel K/V
    # prefetch (which overlaps the next block's gather with this block's dot)
    # actually fires. With 1 block/chunk the prefetch branch never runs and
    # every paged gather stall is exposed.
    #
    # So cap NUM_KV_CHUNKS by max_seqblock // MIN_BLOCKS_PER_CHUNK — but only
    # when batch*num_kv_heads is large enough that the grid fills the SMs
    # without relying on split-K. At tiny batch (bs=1) the grid is propped up
    # by split-K itself; capping there shrinks the grid below the SM count and
    # loses occupancy, so we leave it uncapped.
    TARGET_GRID = 4096
    MAX_NUM_KV_CHUNKS = 256
    MIN_BLOCKS_PER_CHUNK = 2
    MIN_BATCH_FOR_CAP = 8
    score_kv_len = min(max_seqlen, max_kv_len)
    max_seqblock = triton.cdiv(score_kv_len, block_size)
    target = max(
        1,
        min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, batch_size * num_kv_heads)),
    )
    NUM_KV_CHUNKS = 1 << (target.bit_length() - 1)
    # score_kv_len = min(max_seqlen, max_kv_len)
    # The score producers below write every valid block column
    # [0, ceil(seq_len / block_size)) for each (head, batch) row. All consumers
    # clamp their scan to the same per-row valid block count, so columns beyond
    # seq_len are never read. Avoid a full-tensor -inf memset here; on CUDA graph
    # capture the static score shape can be much larger than the live context.
    score = torch.empty(
        (num_q_heads, batch_size, triton.cdiv(max_kv_len, block_size)),
        dtype=torch.float32,
        device=q.device,
    )
    use_jit_topk = (
        envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.get()
        and score.shape[2] <= _JIT_TOPK_MAX_NUM_BLOCKS
        and topk <= 32
    )
    # If the live context has <= topk sparse blocks, the downstream dense
    # page-table/JIT top-k kernels select every block from seq_lens directly
    # without reading score. Keep this gate in sync with the consumers below:
    # _skip_block_topk/use_dense_main_attn and use_jit_topk special-case
    # num_blocks <= topk, while the Triton fallback reads score and must not
    # skip these writes.
    skip_trivial_topk_score = use_dense_main_attn or use_jit_topk

    grid = (batch_size * NUM_KV_CHUNKS, num_kv_heads)
    if disable_index_value:
        _decode_score_kernel[grid](
            q,
            k_cache,
            req_to_token,
            score,
            seq_lens,
            slot_ids,
            max_slots,
            batch_size,
            gqa_group_size,
            head_dim,
            block_size,
            topk,
            sm_scale,
            k_scale,
            init_blocks,
            local_blocks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            req_to_token.stride(0),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            NUM_KV_CHUNKS=NUM_KV_CHUNKS,
            SCORE_TYPE=score_type,
            SKIP_TRIVIAL_TOPK_SCORE=skip_trivial_topk_score,
            IS_FP8=is_fp8,
        )
    else:
        assert v_cache is not None
        o = torch.empty(
            NUM_KV_CHUNKS,
            batch_size,
            num_q_heads,
            head_dim,
            dtype=sparse_out_dtype(q),
            device=q.device,
        )
        lse = torch.empty(
            NUM_KV_CHUNKS, batch_size, num_q_heads, dtype=torch.float32, device=q.device
        )
        _decode_score_attn_kernel[grid](
            q,
            sink,
            k_cache,
            v_cache,
            req_to_token,
            o,
            lse,
            score,
            seq_lens,
            slot_ids,
            max_slots,
            batch_size,
            gqa_group_size,
            head_dim,
            block_size,
            topk,
            sm_scale,
            k_scale,
            v_scale,
            init_blocks,
            local_blocks,
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
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            NUM_KV_CHUNKS=NUM_KV_CHUNKS,
            SCORE_TYPE=score_type,
            SKIP_TRIVIAL_TOPK_SCORE=skip_trivial_topk_score,
            IS_FP8=is_fp8,
        )
    # Fused top-k + page-table transform: emit the dense backend's page table
    # directly (page-size-aware) instead of block ids, skipping a separate gather.
    # The page table + per-query effective KV length are allocated and returned.
    topk_idx = torch.empty(
        (num_q_heads, batch_size, topk),
        device=score.device,
        dtype=torch.int32,
        )
    real_seq_lens = None
    # if use_dense_main_attn:
    if use_jit_topk:
        from sglang.kernels.ops.attention.minimax_decode_topk import (
            # minimax_decode_topk_page_table,
            minimax_decode_topk
        )
        # topk_idx, real_seq_lens = minimax_decode_topk_page_table(
        #     score, seq_lens, req_to_token, slot_ids, block_size, topk, page_size
        # )
        topk_idx = minimax_decode_topk(
            score, seq_lens, block_size, topk, topk_idx
        )
        if disable_index_value:
            return None, topk_idx, real_seq_lens
        # fall through to the idx_o (index value) merge, then return
        _skip_block_topk = True
    else:
        _skip_block_topk = False

    # get topk index
    # if not _skip_block_topk:
    #     topk_idx = torch.empty(
    #         (num_q_heads, batch_size, topk),
    #         device=score.device,
    #         dtype=torch.int32,
    #     )

    if _skip_block_topk:
        pass
    elif use_jit_topk:
        # Single-stage JIT radix-select: one kernel, no intermediate buffers.
        # Equivalent output to the 2-stage path (set of block ids, front-packed,
        # -1 padded); ~2-16x faster for long context. See
        # sglang/kernels/ops/attention/minimax_decode_topk.py.
        if is_xpu():
            from sgl_kernel import minimax_decode_topk
        else:
            from sglang.kernels.ops.attention.minimax_decode_topk import (
                minimax_decode_topk,
            )

        minimax_decode_topk(score, seq_lens, block_size, topk, out=topk_idx)
    else:
        # 2-stage split-K Triton fallback.
        # Choose NUM_TOPK_CHUNKS to add parallelism over the seqblock dim.
        # Same constraints as flash_decode: must be deterministic from values
        # constant within a cuda graph (BS, num_q_heads), pow2, capped so the
        # merge kernel's BLOCK_SIZE_K = pow2(NUM_TOPK_CHUNKS * pow2(topk))
        # stays reasonable. With topk=32 and cap 16, merge sorts 512 items.
        TOPK_TARGET_GRID = 64
        MAX_NUM_TOPK_CHUNKS = 16
        topk_target = max(
            1,
            min(
                MAX_NUM_TOPK_CHUNKS,
                TOPK_TARGET_GRID // max(1, batch_size * num_q_heads),
            ),
        )
        NUM_TOPK_CHUNKS = 1 << (topk_target.bit_length() - 1)
        BLOCK_SIZE_T = triton.next_power_of_2(topk)
        max_seqblock = score.shape[2]
        chunk_blocks = (max_seqblock + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
        topk_score_partial = torch.empty(
            NUM_TOPK_CHUNKS,
            num_q_heads,
            batch_size,
            BLOCK_SIZE_T,
            dtype=torch.float32,
            device=score.device,
        )
        topk_idx_partial = torch.empty(
            NUM_TOPK_CHUNKS,
            num_q_heads,
            batch_size,
            BLOCK_SIZE_T,
            dtype=torch.int32,
            device=score.device,
        )
        # stream 0 (default): topk partial → topk merge
        grid = (batch_size, num_q_heads, NUM_TOPK_CHUNKS)
        _topk_index_partial_kernel[grid](
            score,
            topk_score_partial,
            topk_idx_partial,
            seq_lens,
            block_size,
            topk,
            chunk_blocks,
            score.stride(0),
            score.stride(1),
            score.stride(2),
            topk_score_partial.stride(0),
            topk_score_partial.stride(1),
            topk_score_partial.stride(2),
            topk_score_partial.stride(3),
            topk_idx_partial.stride(0),
            topk_idx_partial.stride(1),
            topk_idx_partial.stride(2),
            topk_idx_partial.stride(3),
        )
        grid = (batch_size, num_q_heads)
        _topk_index_merge_kernel[grid](
            topk_score_partial,
            topk_idx_partial,
            topk_idx,
            seq_lens,
            block_size,
            topk,
            topk_score_partial.stride(0),
            topk_score_partial.stride(1),
            topk_score_partial.stride(2),
            topk_score_partial.stride(3),
            topk_idx_partial.stride(0),
            topk_idx_partial.stride(1),
            topk_idx_partial.stride(2),
            topk_idx_partial.stride(3),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
        )
    if disable_index_value:
        return None, topk_idx, real_seq_lens
    # attn output merge (default stream)
    grid = (batch_size, num_q_heads)
    _merge_attn_out_kernel[grid](
        o,
        lse,
        seq_lens,
        head_dim,
        block_size,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        NUM_KV_CHUNKS=NUM_KV_CHUNKS,
    )
    o = o[0].contiguous()
    return o, topk_idx, real_seq_lens
