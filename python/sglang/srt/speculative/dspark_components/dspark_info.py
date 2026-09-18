from typing import Optional

import msgspec
import torch


class VerifyWindow(msgspec.Struct, frozen=True):
    positions_2d: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_cache_loc_2d: torch.Tensor


class RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor


class TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool
    verify_forward_batch: object


class CompactVerifyOutput(msgspec.Struct, frozen=True):
    runtime_layout: object
    semantic_layout: object
    ragged_window: RaggedVerifyWindow
    compact_logits: torch.Tensor
    compact_hidden: torch.Tensor


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    confidence_tap: Optional[torch.Tensor] = None
    folded: bool = False
