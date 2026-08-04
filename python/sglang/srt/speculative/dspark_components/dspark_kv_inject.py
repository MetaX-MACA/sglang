from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.dspark_info import CompactVerifyOutput
from sglang.srt.speculative.dspark_components.kernels.commit_inject_layout import (
    BuildCommitInjectLayout,
    build_compact_commit_cache_locs_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func


class TargetHiddenKvInjector:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        model_runner,
        device,
        verify_num_draft_tokens: int,
        block_pos_offsets: torch.Tensor,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.model_runner = model_runner
        self.device = device
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self._block_pos_offsets = block_pos_offsets

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.cuda.Event]:
        if target_hidden is None or target_hidden.numel() == 0:
            return None
        device = self.model_runner.device
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        target_hidden = target_hidden.to(device=device, non_blocking=True)
        n_real = positions.shape[0]
        if target_hidden.shape[0] > n_real:
            target_hidden = target_hidden[:n_real]
        if cache_loc_2d is not None:
            cache_loc_2d = cache_loc_2d.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if commit_lens is not None:
            commit_lens = commit_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            )

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
            return self._record_cuda_event(device)

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                target_hidden=target_hidden,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
        return self._record_cuda_event(device)

    @staticmethod
    def _record_cuda_event(device) -> Optional[torch.cuda.Event]:
        if not torch.cuda.is_available():
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return event

    def _inject_mla(
        self,
        *,
        pool,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
    ) -> None:
        swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
            committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            swa_loc = torch.where(committed_mask, swa_loc, torch.full_like(swa_loc, -1))

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def inject_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        hidden_strided: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> None:
        stride = self.verify_num_draft_tokens
        prefix_lens = batch.seq_lens
        hidden = hidden_strided.view(bs, stride, -1)

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if hidden_strided.numel() == 0:
                return
            inject_layout = BuildCommitInjectLayout.execute(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                prefix_lens=prefix_lens,
                block_pos_offsets=self._block_pos_offsets[:stride],
                full_to_swa_mapping=pool.full_to_swa_index_mapping,
                commit_lens=commit_lens,
                stride=stride,
            )
            with torch.inference_mode():
                self.draft_model.write_target_hidden_kv(
                    main_hidden=hidden.reshape(-1, hidden.shape[-1]),
                    swa_loc=inject_layout.swa_loc,
                    positions=inject_layout.positions,
                    pool=pool,
                )
            return

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + stride,
            batch_size=bs,
            draft_token_num=stride,
            device=self.device,
        )
        verify_cache_loc_2d = verify_cache_loc.view(bs, stride)
        self.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )

    def inject_qwen3_5_ragged_compact(
        self,
        *,
        compact_output: CompactVerifyOutput,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> None:
        hidden = compact_output.compact_hidden
        if hidden.numel() == 0:
            return
        runtime_layout = compact_output.runtime_layout
        if hidden.shape[0] != runtime_layout.graph_num_tokens:
            raise ValueError(
                "DSpark compact hidden rows must match the runtime graph tier: "
                f"hidden_rows={hidden.shape[0]}, "
                f"graph_num_tokens={runtime_layout.graph_num_tokens}."
            )

        if self.model_runner.mambaish_config is None:
            raise RuntimeError(
                "Qwen3.5/Qwen3.6 compact DSpark hidden commit requires a "
                "recurrent target model."
            )

        # Generic KV pools reserve slot 0 for padded writes. Masking uncommitted
        # compact rows to that slot lets the existing model writer consume M rows
        # directly without exposing a new memory-pool API.
        cache_loc = build_compact_commit_cache_locs_triton(
            verify_cache_loc=compact_output.ragged_window.verify_cache_loc,
            runtime_qo_indptr=runtime_layout.qo_indptr_device,
            commit_lens=commit_lens,
            bs=bs,
            graph_num_tokens=runtime_layout.graph_num_tokens,
        )
        self.inject_target_hidden(
            target_hidden=hidden,
            cache_loc=cache_loc,
            positions=compact_output.ragged_window.positions,
        )
