from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.dspark_components.kernels.accept_greedy import (
    AcceptGreedy,
    accept_greedy_from_predictions_triton,
    compact_predictions_to_strided_triton,
)
from sglang.srt.speculative.dspark_components.kernels.accept_sampling import (
    AcceptSampling,
)
from sglang.srt.speculative.dspark_components.kernels.mixed_accept_select import (
    SelectMixedAccept,
)
from sglang.srt.speculative.dspark_components.kernels.softmax_temp import SoftmaxTemp
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


def accept_draft_tokens(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_block: DraftBlockResult,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    greedy_mask = draft_block.greedy_mask
    cutoff_verify_lens = None if cutoff_layout is None else cutoff_layout.verify_lens
    all_greedy = sampling_info is None or sampling_info.is_all_greedy
    if all_greedy:
        return AcceptGreedy.execute(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    bs, gamma_rows, vocab = draft_block.corrected_logits.shape
    draft_probs = SoftmaxTemp.execute(
        logits=draft_block.corrected_logits.reshape(bs * gamma_rows, vocab),
        temperatures=draft_block.temperatures,
        rows_per_request=gamma_rows,
    ).view(bs, gamma_rows, vocab)
    if not sampling_info.is_any_greedy:
        return AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    greedy_len, greedy_bonus, greedy_trim = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    sampling_len, sampling_bonus, sampling_trim = AcceptSampling.execute(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    selected = SelectMixedAccept.execute(
        greedy_mask=greedy_mask,
        greedy_len=greedy_len,
        greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim,
        sampling_len=sampling_len,
        sampling_bonus=sampling_bonus,
        sampling_trim=sampling_trim,
    )
    return selected.correct_len, selected.bonus, selected.cap_trim_lens


def accept_compact_greedy(
    *,
    candidates: torch.Tensor,
    compact_target_logits: torch.Tensor,
    runtime_layout: RaggedVerifyLayout,
    semantic_layout: RaggedVerifyLayout,
    verify_num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = int(candidates.shape[0])
    if int(semantic_layout.verify_lens.numel()) != bs:
        raise ValueError(
            "DSpark compact accept semantic layout must match candidates: "
            f"semantic_bs={int(semantic_layout.verify_lens.numel())}, bs={bs}."
        )
    if compact_target_logits.ndim != 2:
        raise ValueError(
            "DSpark compact accept requires [tokens, vocab] target logits, got "
            f"shape={tuple(compact_target_logits.shape)}."
        )
    if compact_target_logits.shape[0] != runtime_layout.graph_num_tokens:
        raise ValueError(
            "DSpark compact target logits do not match the runtime graph tier: "
            f"rows={compact_target_logits.shape[0]}, "
            f"graph_num_tokens={runtime_layout.graph_num_tokens}."
        )

    compact_predict = torch.argmax(compact_target_logits, dim=-1)
    target_predict = compact_predictions_to_strided_triton(
        compact_predict=compact_predict,
        runtime_qo_indptr=runtime_layout.qo_indptr_device,
        semantic_verify_lens=semantic_layout.verify_lens,
        bs=bs,
        stride=verify_num_draft_tokens,
    )
    return accept_greedy_from_predictions_triton(
        candidates=candidates,
        target_predict=target_predict,
        cutoff_verify_lens=semantic_layout.verify_lens,
    )
