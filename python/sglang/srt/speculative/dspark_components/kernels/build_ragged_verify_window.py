from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.dspark_info import RaggedVerifyWindow
from sglang.srt.speculative.dspark_components.kernels.compact_layout import (
    COMPACT_ROW_SEARCH_NBITS,
    compact_row_from_qo_indptr,
    compact_row_index,
    compact_row_index_triton,
    compact_verify_ids,
    compact_verify_ids_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_RAGGED_WINDOW.get()


class BuildRaggedVerifyWindow:
    @classmethod
    def execute(cls, *args, **kwargs) -> RaggedVerifyWindow:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def execute_qwen3_5(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        semantic_layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        if _KERNEL_IMPL == "torch":
            return build_ragged_verify_window_with_spares(
                batch=batch,
                runtime_layout=layout,
                semantic_layout=semantic_layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=bs,
                device=device,
                verify_num_draft_tokens=verify_num_draft_tokens,
                model_runner=model_runner,
            )
        return build_ragged_verify_window_with_spares_triton(
            batch=batch,
            runtime_layout=layout,
            semantic_layout=semantic_layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )

    @classmethod
    def torch(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )

    @classmethod
    def triton(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window_triton(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )


def build_ragged_verify_window(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, valid = compact_row_index(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    safe_req = req_id.clamp(max=bs - 1)
    positions = torch.where(
        valid,
        prefix_lens.to(torch.int64)[safe_req] + within,
        torch.zeros_like(within),
    )
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    verify_cache_loc = torch.nn.functional.pad(
        real_cache_loc, (0, padded_total - real_cache_loc.shape[0])
    )
    verify_cache_loc = torch.where(
        valid, verify_cache_loc, torch.zeros_like(verify_cache_loc)
    )

    verify_ids = compact_verify_ids(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )

    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )


def build_ragged_verify_window_with_spares(
    *,
    batch: ScheduleBatch,
    runtime_layout: RaggedVerifyLayout,
    semantic_layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    stride = verify_num_draft_tokens
    prefix_lens = batch.seq_lens
    runtime_lens = runtime_layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = runtime_layout.graph_num_tokens

    if int(semantic_layout.verify_lens.shape[0]) != bs:
        raise ValueError(
            "DSpark compact graph semantic layout must match the real batch size: "
            f"semantic_bs={int(semantic_layout.verify_lens.shape[0])}, bs={bs}."
        )
    if padded_total > bs * stride:
        raise ValueError(
            "DSpark compact graph cannot materialize more verify tokens than the "
            "real requests reserved: "
            f"graph_num_tokens={padded_total}, bs={bs}, stride={stride}."
        )

    req_id, within, valid = compact_row_index(
        verify_lens=runtime_lens, padded_total=padded_total, device=device
    )
    mapped_req = req_id.clamp(max=bs - 1)
    mapped_within = within
    mapped_valid = valid & (req_id < bs) & (mapped_within < stride)

    full_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + stride,
        batch_size=bs,
        draft_token_num=stride,
        device=device,
    )
    dense_pos = mapped_req * stride + mapped_within
    verify_cache_loc = torch.where(
        mapped_valid,
        full_cache_loc[dense_pos],
        torch.zeros(padded_total, dtype=full_cache_loc.dtype, device=device),
    )
    positions = torch.where(
        mapped_valid,
        prefix_lens.to(torch.int64)[mapped_req] + mapped_within,
        torch.zeros(padded_total, dtype=torch.int64, device=device),
    )

    draft_block_ids = draft_block_ids.to(device=device, dtype=torch.int64)
    draft_tokens = draft_tokens.to(device=device, dtype=torch.int64)
    anchors = draft_block_ids[:, 0]
    drafts = draft_tokens[mapped_req, (mapped_within - 1).clamp_min(0)]
    verify_ids = torch.where(mapped_within == 0, anchors[mapped_req], drafts)
    verify_ids = torch.where(mapped_valid, verify_ids, torch.zeros_like(verify_ids)).to(
        torch.int64
    )

    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )


@triton.jit
def _ragged_verify_window_with_spares_kernel(
    qo_indptr_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    draft_block_ids_ptr,
    draft_tokens_ptr,
    positions_out_ptr,
    cache_locs_out_ptr,
    verify_ids_out_ptr,
    req_to_token_stride,
    draft_block_stride,
    draft_tokens_stride,
    real_bs,
    num_slots,
    stride,
    n,
    BLOCK: tl.constexpr,
    NBITS: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    row = offs.to(tl.int64)

    req, within = compact_row_from_qo_indptr(row, qo_indptr_ptr, num_slots, mask, NBITS)
    safe_req = tl.minimum(req, real_bs - 1)
    valid = mask & (req < real_bs) & (within < stride)

    prefix = tl.load(prefix_lens_ptr + safe_req, mask=valid, other=0).to(tl.int64)
    req_pool = tl.load(req_pool_indices_ptr + safe_req, mask=valid, other=0).to(
        tl.int64
    )
    full_loc = tl.load(
        req_to_token_ptr + req_pool * req_to_token_stride + prefix + within,
        mask=valid,
        other=0,
    ).to(tl.int64)

    anchor = tl.load(
        draft_block_ids_ptr + safe_req * draft_block_stride,
        mask=valid,
        other=0,
    ).to(tl.int64)
    draft_col = tl.maximum(within - 1, 0)
    draft = tl.load(
        draft_tokens_ptr + safe_req * draft_tokens_stride + draft_col,
        mask=valid,
        other=0,
    ).to(tl.int64)
    verify_id = tl.where(within == 0, anchor, draft)

    tl.store(positions_out_ptr + offs, tl.where(valid, prefix + within, 0), mask=mask)
    tl.store(cache_locs_out_ptr + offs, tl.where(valid, full_loc, 0), mask=mask)
    tl.store(verify_ids_out_ptr + offs, tl.where(valid, verify_id, 0), mask=mask)


def build_ragged_verify_window_with_spares_triton(
    *,
    batch: ScheduleBatch,
    runtime_layout: RaggedVerifyLayout,
    semantic_layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    stride = verify_num_draft_tokens
    padded_total = runtime_layout.graph_num_tokens
    num_slots = runtime_layout.runtime_num_slots
    if bs < 1:
        raise ValueError(f"DSpark compact verify requires bs >= 1, got {bs}.")
    if int(semantic_layout.verify_lens.shape[0]) != bs:
        raise ValueError(
            "DSpark compact graph semantic layout must match the real batch size: "
            f"semantic_bs={int(semantic_layout.verify_lens.shape[0])}, bs={bs}."
        )
    if num_slots < bs:
        raise ValueError(
            "DSpark compact graph runtime slots cannot be smaller than the real "
            f"batch: runtime_slots={num_slots}, bs={bs}."
        )
    if num_slots > 1 << COMPACT_ROW_SEARCH_NBITS:
        raise ValueError(
            "DSpark fused ragged verify supports at most 2048 runtime slots, got "
            f"runtime_slots={num_slots}."
        )
    if padded_total > bs * stride:
        raise ValueError(
            "DSpark compact graph cannot materialize more verify tokens than the "
            "real requests reserved: "
            f"graph_num_tokens={padded_total}, bs={bs}, stride={stride}."
        )
    if draft_block_ids.ndim != 2 or draft_block_ids.shape[0] != bs:
        raise ValueError(
            "DSpark draft block ids must have one row per request, got "
            f"shape={tuple(draft_block_ids.shape)}, bs={bs}."
        )
    if (
        draft_tokens.ndim != 2
        or draft_tokens.shape[0] != bs
        or draft_tokens.shape[1] < stride - 1
    ):
        raise ValueError(
            "DSpark draft tokens must cover every fused verify position: "
            f"shape={tuple(draft_tokens.shape)}, bs={bs}, stride={stride}."
        )

    prefix_lens = batch.seq_lens.to(device=device, dtype=torch.int64).contiguous()
    req_pool_indices = batch.req_pool_indices.to(
        device=device, dtype=torch.int64
    ).contiguous()
    req_to_token = model_runner.req_to_token_pool.req_to_token
    draft_block_ids = draft_block_ids.to(device=device, dtype=torch.int64).contiguous()
    draft_tokens = draft_tokens.to(device=device, dtype=torch.int64).contiguous()
    qo_indptr = runtime_layout.qo_indptr_device.to(
        device=device, dtype=torch.int32
    ).contiguous()
    if int(qo_indptr.numel()) != num_slots + 1:
        raise ValueError(
            "DSpark runtime qo_indptr does not match runtime slots: "
            f"qo_indptr={int(qo_indptr.numel())}, runtime_slots={num_slots}."
        )

    positions = torch.empty(padded_total, dtype=torch.int64, device=device)
    verify_cache_loc = torch.empty(padded_total, dtype=torch.int64, device=device)
    verify_ids = torch.empty(padded_total, dtype=torch.int64, device=device)
    block = 256
    _ragged_verify_window_with_spares_kernel[(triton.cdiv(padded_total, block),)](
        qo_indptr,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        draft_block_ids,
        draft_tokens,
        positions,
        verify_cache_loc,
        verify_ids,
        req_to_token.stride(0),
        draft_block_ids.stride(0),
        draft_tokens.stride(0),
        bs,
        num_slots,
        stride,
        padded_total,
        BLOCK=block,
        NBITS=COMPACT_ROW_SEARCH_NBITS,
    )
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )


@triton.jit
def _ragged_finalize_kernel(
    req_ptr,
    within_ptr,
    prefix_ptr,
    cache_ptr,
    pos_out_ptr,
    cache_out_ptr,
    bs,
    n,
    real_len,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    req = tl.load(req_ptr + offs, mask=mask, other=0)
    within = tl.load(within_ptr + offs, mask=mask, other=0)
    valid = req < bs
    safe_req = tl.minimum(req, bs - 1)
    prefix = tl.load(prefix_ptr + safe_req, mask=mask, other=0)
    pos = tl.where(valid, prefix + within, 0)
    lmask = mask & (offs < real_len)
    cl = tl.load(cache_ptr + offs, mask=lmask, other=0)
    cl = tl.where(valid, cl, 0)
    tl.store(pos_out_ptr + offs, pos, mask=mask)
    tl.store(cache_out_ptr + offs, cl, mask=mask)


def build_ragged_verify_window_triton(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, _valid = compact_row_index_triton(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    prefix_i64 = prefix_lens.to(device=device, dtype=torch.int64).contiguous()
    positions = torch.empty(padded_total, dtype=torch.int64, device=device)
    verify_cache_loc = torch.empty(
        padded_total, dtype=real_cache_loc.dtype, device=device
    )
    BLOCK = 256
    grid = (triton.cdiv(padded_total, BLOCK),)
    _ragged_finalize_kernel[grid](
        req_id,
        within,
        prefix_i64,
        real_cache_loc,
        positions,
        verify_cache_loc,
        bs,
        padded_total,
        real_cache_loc.shape[0],
        BLOCK=BLOCK,
    )

    verify_ids = compact_verify_ids_triton(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )
