"""Correctness-first Triton replacement for quantized flash_mla_with_kvcache."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged_ref,
)
from sglang.srt.layers.attention.dsa.kv_cache_layout import (
    DSAKVCacheFormat,
    HEAD_DIM,
    NOPE_DIM,
    NUM_QUANT_GROUPS,
    PACKED_BYTES_PER_TOKEN,
    QUANT_GROUP_SIZE,
    ROPE_DIM,
    get_kv_cache_format,
)

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _dsa_quantized_sparse_decode_kernel(
    q_ptr,
    cache_fp8_ptr,
    cache_int8_ptr,
    cache_fp32_ptr,
    cache_bf16_ptr,
    indices_ptr,
    topk_len_ptr,
    out_ptr,
    lse_ptr,
    softmax_scale: tl.float32,
    cache_capacity: tl.int64,
    stride_qr: tl.int64,
    stride_qh: tl.int64,
    stride_ir: tl.int64,
    stride_or: tl.int64,
    stride_oh: tl.int64,
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
    NOPE_SIZE: tl.constexpr,
    ROPE_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
):
    row_id = tl.program_id(0)
    head_id = tl.program_id(1)
    nope_offs = tl.arange(0, NOPE_SIZE)
    rope_offs = tl.arange(0, ROPE_SIZE)
    group_ids = nope_offs // GROUP_SIZE
    q_base = row_id * stride_qr + head_id * stride_qh
    q_nope = tl.load(q_ptr + q_base + nope_offs).to(tl.float32) * softmax_scale
    q_rope = (
        tl.load(q_ptr + q_base + NOPE_SIZE + rope_offs).to(tl.float32)
        * softmax_scale
    )

    valid_topk = TOPK
    if HAS_TOPK_LENGTH:
        valid_topk = tl.minimum(tl.load(topk_len_ptr + row_id), TOPK)
    running_max: tl.float32 = -float("inf")
    running_sum: tl.float32 = 0.0
    acc_v = tl.zeros([NOPE_SIZE], dtype=tl.float32)
    token_offs = tl.arange(0, BLOCK_T)

    for tile_start in range(0, TOPK, BLOCK_T):
        topk_pos = tile_start + token_offs
        raw_locs = tl.load(
            indices_ptr + row_id * stride_ir + topk_pos,
            mask=topk_pos < TOPK,
            other=-1,
        ).to(tl.int64)
        valid = (
            (topk_pos < valid_topk)
            & (raw_locs >= 0)
            & (raw_locs < cache_capacity)
        )
        safe_locs = tl.where(valid, raw_locs, 0)
        row_byte_bases = safe_locs * ROW_BYTES
        nope_byte_addrs = row_byte_bases[:, None] + nope_offs[None, :]
        if CACHE_FORMAT == 1:
            quant_nope = tl.load(
                cache_fp8_ptr + nope_byte_addrs,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            quant_nope = tl.load(
                cache_int8_ptr + nope_byte_addrs,
                mask=valid[:, None],
                other=0,
            ).to(tl.float32)
        scale_addrs = (
            (row_byte_bases + NOPE_SIZE)[:, None] // 4 + group_ids[None, :]
        )
        scales = tl.load(
            cache_fp32_ptr + scale_addrs,
            mask=valid[:, None],
            other=1.0,
        )
        kv_nope = tl.where(valid[:, None], quant_nope * scales, 0.0)
        rope_elem_bases = (row_byte_bases + NOPE_SIZE + NUM_GROUPS * 4) // 2
        kv_rope = tl.load(
            cache_bf16_ptr + rope_elem_bases[:, None] + rope_offs[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q_nope[None, :] * kv_nope, axis=1) + tl.sum(
            q_rope[None, :] * kv_rope, axis=1
        )
        scores_log2 = tl.where(valid, scores * LOG2E, -float("inf"))
        tile_max = tl.max(scores_log2)
        new_max = tl.maximum(running_max, tile_max)
        old_scale = tl.where(
            running_max == -float("inf"), 0.0, tl.exp2(running_max - new_max)
        )
        probs = tl.where(valid, tl.exp2(scores_log2 - new_max), 0.0)
        running_sum = running_sum * old_scale + tl.sum(probs)
        acc_v = acc_v * old_scale + tl.sum(probs[:, None] * kv_nope, axis=0)
        running_max = new_max

    has_value = running_sum > 0.0
    safe_sum = tl.where(has_value, running_sum, 1.0)
    output = tl.where(has_value, acc_v / safe_sum, 0.0)
    lse = tl.where(
        has_value, running_max / LOG2E + tl.log(safe_sum), -float("inf")
    )
    out_base = row_id * stride_or + head_id * stride_oh
    tl.store(out_ptr + out_base + nope_offs, output)
    tl.store(lse_ptr + row_id * num_heads + head_id, lse)


def _normalize_topk_length(
    topk_length: Optional[torch.Tensor], batch: int, seq_q: int
) -> Optional[torch.Tensor]:
    if topk_length is None:
        return None
    lengths = topk_length.reshape(-1).to(dtype=torch.int32)
    rows = batch * seq_q
    if lengths.numel() == batch and seq_q != 1:
        lengths = lengths.repeat_interleave(seq_q)
    if lengths.numel() != rows:
        raise ValueError(f"topk_length has {lengths.numel()} entries, expected {rows}")
    return lengths.contiguous()


def _validate_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: Optional[torch.Tensor],
    head_dim_v: int,
    causal: bool,
) -> DSAKVCacheFormat:
    if indices is None:
        raise ValueError("The functional DSA Triton path requires sparse indices")
    if causal:
        raise ValueError("Sparse DSA indices already encode causality")
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"Expected q shape [B, S, H, {HEAD_DIM}], got {q.shape}")
    if head_dim_v != NOPE_DIM:
        raise ValueError(f"Expected head_dim_v={NOPE_DIM}, got {head_dim_v}")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("Expected k_cache [num_pages, page_size, 1, packed_bytes]")
    if k_cache.shape[-1] != PACKED_BYTES_PER_TOKEN:
        raise ValueError(
            f"Expected {PACKED_BYTES_PER_TOKEN} bytes per row, got {k_cache.shape[-1]}"
        )
    if not k_cache.is_contiguous():
        raise ValueError("The functional Triton kernel requires contiguous KV pages")
    if indices.ndim != 3 or indices.shape[:2] != q.shape[:2]:
        raise ValueError("indices batch/query dimensions must match q")
    cache_format = get_kv_cache_format(k_cache.dtype)
    if cache_format not in (DSAKVCacheFormat.FP8, DSAKVCacheFormat.INT8):
        raise ValueError("The functional cache reader only supports FP8 and INT8")
    return cache_format


def flash_mla_with_kvcache_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata=None,
    num_splits=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    indices_all_valid_per_q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA over packed FP8/INT8 rows, matching the relevant vendor API."""
    del block_table, cache_seqlens, tile_scheduler_metadata, num_splits
    del indices_all_valid_per_q, is_fp8_kvcache
    if descale_q is not None or descale_k is not None:
        raise NotImplementedError("Q/K descale is outside the basic path")
    if attn_sink is not None:
        raise NotImplementedError("attn_sink is outside the basic path")
    if (
        extra_k_cache is not None
        or extra_indices_in_kvcache is not None
        or extra_topk_length is not None
    ):
        raise NotImplementedError("extra KV cache is outside the basic path")

    cache_format = _validate_inputs(q, k_cache, indices, head_dim_v, causal)
    assert indices is not None
    if softmax_scale is None:
        softmax_scale = HEAD_DIM**-0.5
    batch, seq_q, num_heads, _ = q.shape
    rows = batch * seq_q
    q_rows = q.contiguous().reshape(rows, num_heads, HEAD_DIM)
    index_rows = indices.to(dtype=torch.int32).contiguous().reshape(rows, -1)
    valid_lengths = _normalize_topk_length(topk_length, batch, seq_q)
    raw_u8 = k_cache.view(torch.uint8).reshape(-1)
    cache_fp8 = raw_u8.view(torch.float8_e4m3fn)
    cache_int8 = raw_u8.view(torch.int8)
    cache_fp32 = raw_u8.view(torch.float32)
    cache_bf16 = raw_u8.view(torch.bfloat16)
    output = torch.zeros((rows, num_heads, NOPE_DIM), dtype=q.dtype, device=q.device)
    lse = torch.full(
        (rows, num_heads), float("-inf"), dtype=torch.float32, device=q.device
    )
    topk = index_rows.shape[-1]
    _dsa_quantized_sparse_decode_kernel[(rows, num_heads)](
        q_rows,
        cache_fp8,
        cache_int8,
        cache_fp32,
        cache_bf16,
        index_rows,
        valid_lengths if valid_lengths is not None else index_rows,
        output,
        lse,
        float(softmax_scale),
        k_cache.shape[0] * k_cache.shape[1],
        q_rows.stride(0),
        q_rows.stride(1),
        index_rows.stride(0),
        output.stride(0),
        output.stride(1),
        num_heads,
        TOPK=topk,
        BLOCK_T=16,
        CACHE_FORMAT=int(cache_format),
        HAS_TOPK_LENGTH=valid_lengths is not None,
        NOPE_SIZE=NOPE_DIM,
        ROPE_SIZE=ROPE_DIM,
        GROUP_SIZE=QUANT_GROUP_SIZE,
        NUM_GROUPS=NUM_QUANT_GROUPS,
        ROW_BYTES=PACKED_BYTES_PER_TOKEN,
        num_warps=8,
        num_stages=2,
    )
    return (
        output.reshape(batch, seq_q, num_heads, NOPE_DIM),
        lse.reshape(batch, seq_q, num_heads).permute(0, 2, 1),
    )


def flash_mla_with_kvcache_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    head_dim_v: int = NOPE_DIM,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent Torch oracle for precision validation."""
    _validate_inputs(q, k_cache, indices, head_dim_v, causal=False)
    if softmax_scale is None:
        softmax_scale = HEAD_DIM**-0.5
    batch, seq_q, num_heads, _ = q.shape
    rows = batch * seq_q
    index_rows = indices.reshape(rows, -1)
    topk = index_rows.shape[-1]
    capacity = k_cache.shape[0] * k_cache.shape[1]
    invalid = (index_rows < 0) | (index_rows >= capacity)
    lengths = _normalize_topk_length(topk_length, batch, seq_q)
    if lengths is not None:
        positions = torch.arange(topk, device=q.device).view(1, -1)
        invalid |= positions >= lengths.view(-1, 1)
    safe_indices = index_rows.clamp(min=0, max=capacity - 1)
    kv = dequantize_k_cache_paged_ref(k_cache, safe_indices).reshape(
        rows, topk, HEAD_DIM
    )
    q_rows = q.reshape(rows, num_heads, HEAD_DIM).float()
    scores = torch.einsum("rhd,rtd->rht", q_rows, kv.float()) * softmax_scale
    scores.masked_fill_(invalid[:, None, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - lse.unsqueeze(-1))
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    output = torch.einsum("rht,rtv->rhv", weights, kv[..., :head_dim_v].float()).to(
        q.dtype
    )
    return (
        output.reshape(batch, seq_q, num_heads, head_dim_v),
        lse.reshape(batch, seq_q, num_heads).permute(0, 2, 1),
    )
