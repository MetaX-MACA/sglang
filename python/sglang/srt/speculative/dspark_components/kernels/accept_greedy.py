from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    CapCorrectLen,
)

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_ACCEPT_GREEDY.get()


class AcceptGreedy:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy_triton(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )


def accept_greedy(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        row_ids = torch.arange(bs, device=target_predict.device)
        bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


@triton.jit
def _gather_row_bonus_kernel(
    table_ptr,
    idx_ptr,
    out_ptr,
    cols,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int64)
    val = tl.load(table_ptr + offs * cols + idx, mask=mask, other=0)
    tl.store(out_ptr + offs, val.to(tl.int64), mask=mask)


def gather_row_bonus_triton(*, table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bs, cols = table.shape
    table = table.contiguous()
    idx = idx.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=table.device)
    BLOCK = 256
    grid = (triton.cdiv(bs, BLOCK),)
    _gather_row_bonus_kernel[grid](table, idx, out, cols, bs, BLOCK=BLOCK)
    return out


@triton.jit
def _compact_predictions_to_strided_kernel(
    compact_predict_ptr,
    runtime_qo_indptr_ptr,
    semantic_verify_lens_ptr,
    out_ptr,
    stride,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    req = offs // stride
    col = offs % stride
    verify_len = tl.load(semantic_verify_lens_ptr + req, mask=mask, other=0)
    compact_start = tl.load(runtime_qo_indptr_ptr + req, mask=mask, other=0)
    valid = mask & (col < verify_len)
    prediction = tl.load(compact_predict_ptr + compact_start + col, mask=valid, other=0)
    tl.store(out_ptr + offs, prediction.to(tl.int64), mask=mask)


def compact_predictions_to_strided_triton(
    *,
    compact_predict: torch.Tensor,
    runtime_qo_indptr: torch.Tensor,
    semantic_verify_lens: torch.Tensor,
    bs: int,
    stride: int,
) -> torch.Tensor:
    if compact_predict.ndim != 1:
        raise ValueError(
            "DSpark compact target predictions must be one-dimensional, got "
            f"shape={tuple(compact_predict.shape)}."
        )
    if int(semantic_verify_lens.numel()) != bs:
        raise ValueError(
            "DSpark compact prediction layout must match the real batch size: "
            f"verify_lens={int(semantic_verify_lens.numel())}, bs={bs}."
        )
    if int(runtime_qo_indptr.numel()) < bs + 1:
        raise ValueError(
            "DSpark compact prediction layout has too few runtime offsets: "
            f"qo_indptr={int(runtime_qo_indptr.numel())}, required={bs + 1}."
        )

    n = bs * stride
    out = torch.empty(n, dtype=torch.int64, device=compact_predict.device)
    block = 256
    _compact_predictions_to_strided_kernel[(triton.cdiv(n, block),)](
        compact_predict,
        runtime_qo_indptr,
        semantic_verify_lens,
        out,
        stride,
        n,
        BLOCK=block,
    )
    return out.view(bs, stride)


def accept_greedy_from_predictions_triton(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidates.shape != target_predict.shape:
        raise ValueError(
            "DSpark greedy candidates and predictions must have the same shape: "
            f"candidates={tuple(candidates.shape)}, "
            f"target_predict={tuple(target_predict.shape)}."
        )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        bonus = gather_row_bonus_triton(table=target_predict, idx=correct_len)
    return correct_len, bonus, cap_trim_lens


def accept_greedy_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        bonus = gather_row_bonus_triton(table=target_predict, idx=correct_len)
    return correct_len, bonus, cap_trim_lens
