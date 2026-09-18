import logging
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components.dspark_info import (
    CompactVerifyOutput,
    RaggedVerifyWindow,
    TargetVerifyResult,
    VerifyWindow,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    apply_logits_adjustments_strided,
)
from sglang.srt.speculative.dspark_components.kernels.build_ragged_verify_window import (
    BuildRaggedVerifyWindow,
)
from sglang.srt.speculative.dspark_components.kernels.scatter_compact_to_strided import (
    ScatterCompactToStrided,
    scatter_runtime_compact_to_strided_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

from sglang.srt.speculative.spec_utils import (
    record_stream_each,
    record_stream_for_v2_verify,
)

logger = logging.getLogger(__name__)


class TargetVerifyExecutor:
    def __init__(
        self,
        *,
        target_worker,
        verify_num_draft_tokens: int,
        model_runner,
        kv_injector: TargetHiddenKvInjector,
        verify_epilogue=None,
        qwen3_5_family_enabled: bool = False,
    ) -> None:
        self.target_worker = target_worker
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self.model_runner = model_runner
        self.kv_injector = kv_injector
        self.verify_epilogue = verify_epilogue
        self.qwen3_5_family_enabled = qwen3_5_family_enabled
        self._verify_backend_self_adds_seq_lens_cache: Optional[bool] = None
        self._logged_static_fallbacks: set[tuple] = set()

    def _warn_static_fallback_once(self, key: tuple, message: str, *args) -> None:
        if self.model_runner.tp_rank != 0:
            return
        if key in self._logged_static_fallbacks:
            return
        self._logged_static_fallbacks.add(key)
        logger.warning(message, *args)

    def run_non_compact(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_ids_2d: torch.Tensor,
        verify_window: VerifyWindow,
        sampling_info,
    ) -> TargetVerifyResult:
        verify_w = self.verify_num_draft_tokens
        positions_2d = verify_window.positions_2d
        verify_cache_loc = verify_window.verify_cache_loc

        verify_input = DFlashVerifyInput(
            draft_token=verify_ids_2d.reshape(-1),
            positions=positions_2d.reshape(-1),
            draft_token_num=verify_w,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch.out_cache_loc = verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if not self._verify_backend_self_adds_seq_lens():
            base_seq_lens_cpu = (
                seq_lens_cpu_backup
                if seq_lens_cpu_backup is not None
                else batch.seq_lens.cpu()
            )
            batch.seq_lens_cpu = base_seq_lens_cpu
            batch.seq_lens_sum = int(base_seq_lens_cpu.sum())

        fwd_stream = torch.get_device_module(self.target_worker.device).current_stream()
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=verify_w,
            )

        return TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            verify_forward_batch=verify_forward_batch,
        )

    def commit_hidden(
        self,
        *,
        batch: ScheduleBatch,
        layout: Optional[RaggedVerifyLayout],
        hidden_strided: Optional[torch.Tensor],
        compact_output: Optional[CompactVerifyOutput],
        verify_window: VerifyWindow,
        logits_output,
        commit_lens: torch.Tensor,
        bs: int,
        run_compact: bool,
    ) -> None:
        if run_compact and self.qwen3_5_family_enabled:
            if compact_output is None:
                raise RuntimeError("DSpark compact verify output is missing.")
            self.kv_injector.inject_qwen3_5_ragged_compact(
                compact_output=compact_output,
                commit_lens=commit_lens,
                bs=bs,
            )
            return
        if run_compact:
            self.kv_injector.inject_ragged(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                commit_lens=commit_lens,
                bs=bs,
            )
            return
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
        self.kv_injector.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
            positions=verify_window.positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )

    def commit_recurrent_states(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit the selected contiguous verify state for recurrent targets."""
        model_runner = self.target_worker.model_runner
        if model_runner.mambaish_config is None:
            return
        attn_backend = model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return
        if batch.forward_mode.is_idle() or commit_lens.numel() == 0:
            return

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        if batch.mamba_track_indices is not None:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            ).to(torch.int64)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith,
                torch.full_like(to_track_ith, -1),
            )
        else:
            mamba_steps_to_track = None

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=model_runner.model,
        )

    def _run_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        ragged_window: RaggedVerifyWindow,
        sampling_info,
    ) -> TargetVerifyResult:
        verify_input = DFlashVerifyInput(
            draft_token=ragged_window.verify_ids,
            positions=ragged_window.positions,
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
        )
        batch.out_cache_loc = ragged_window.verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if not self._verify_backend_self_adds_seq_lens():
            base_seq_lens_cpu = (
                seq_lens_cpu_backup
                if seq_lens_cpu_backup is not None
                else batch.seq_lens.cpu()
            )
            batch.seq_lens_cpu = base_seq_lens_cpu
            batch.seq_lens_sum = int(base_seq_lens_cpu.sum())

        fwd_stream = torch.get_device_module(self.target_worker.device).current_stream()
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        return TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            verify_forward_batch=verify_forward_batch,
        )

    def _prepare_ragged_forward_batch(
        self,
        *,
        batch: ScheduleBatch,
        semantic_layout: RaggedVerifyLayout,
        runtime_layout: RaggedVerifyLayout,
        ragged_window: RaggedVerifyWindow,
    ):
        verify_input = DFlashVerifyInput(
            draft_token=ragged_window.verify_ids,
            positions=ragged_window.positions,
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=runtime_layout,
        )
        batch.out_cache_loc = ragged_window.verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            if runtime_layout.verify_lens_cpu is not None:
                batch.seq_lens_cpu = seq_lens_cpu_backup + torch.tensor(
                    runtime_layout.verify_lens_cpu[: semantic_layout.bs],
                    dtype=seq_lens_cpu_backup.dtype,
                )
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                # Hybrid FlashInfer/GDN metadata consumes device seq_lens and
                # the known aggregate. Avoid synchronizing dynamic verify_lens
                # back to the host solely to populate an unused per-row copy.
                batch.seq_lens_cpu = None
                batch.seq_lens_sum = (
                    int(seq_lens_cpu_backup.sum()) + runtime_layout.graph_num_tokens
                )

        try:
            fwd_stream = torch.get_device_module(self.target_worker.device).current_stream()
            record_stream_for_v2_verify(batch, verify_input, fwd_stream)
            verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
                batch,
                self.target_worker,
                init_eager_on_graph_miss=False,
            )
            record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)
        finally:
            batch.seq_lens_cpu = seq_lens_cpu_backup
            batch.seq_lens_sum = seq_lens_sum_backup

        return verify_forward_batch, can_run_cuda_graph

    def _run_prepared_ragged(self, *, verify_forward_batch) -> TargetVerifyResult:
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        return TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            verify_forward_batch=verify_forward_batch,
        )

    def _runtime_layout_for_graph(
        self, layout: RaggedVerifyLayout
    ) -> tuple[Optional[RaggedVerifyLayout], Optional[str]]:
        runner = self.target_worker.model_runner.decode_cuda_graph_runner
        if runner is None or not runner.ragged_verify_mode:
            return None, "ragged_cuda_graph_runner_unavailable"
        if not runner.qwen3_5_ragged_graph_grid_enabled:
            return layout, None
        semantic_tokens = layout.total_verify_tokens
        if semantic_tokens is None:
            return None, "semantic_verify_token_count_unavailable"
        graph_spec = runner.select_ragged_verify_graph(
            runtime_bs=layout.bs, semantic_tokens=semantic_tokens
        )
        if graph_spec is None:
            return None, "no_safe_ragged_graph_tier"

        # Graph-only tokens use real requests' already-reserved verify cache
        # locations. Virtual slots remain zero length until a dedicated scratch
        # KV pool exists; this avoids graph padding writes corrupting live KV.
        runtime_layout = layout.padded_for_graph(
            spec=graph_spec,
            max_verify_len=self.verify_num_draft_tokens,
            align_real_tokens=True,
        )
        return runtime_layout, None

    def run_compact(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        sampling_info,
        inject_gate: bool = False,
    ) -> tuple[TargetVerifyResult, torch.Tensor]:
        ragged_window = BuildRaggedVerifyWindow.execute(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )
        if self.verify_epilogue is not None:
            self.verify_epilogue.begin_step(layout.verify_lens, armed=inject_gate)
        target_verify = self._run_ragged(
            batch=batch,
            layout=layout,
            ragged_window=ragged_window,
            sampling_info=sampling_info,
        )
        logits_output = target_verify.logits_output

        stride = self.verify_num_draft_tokens
        if self.verify_epilogue is not None and target_verify.can_run_cuda_graph:
            strided_logits = self.verify_epilogue.strided_logits
            hidden_strided = self.verify_epilogue.strided_hidden
            assert strided_logits is not None and hidden_strided is not None, (
                "verify epilogue buffers unwritten after a graph replay -- the "
                "replayed graph was captured without the epilogue"
            )
            strided_logits = strided_logits[: bs * stride]
            hidden_strided = hidden_strided[: bs * stride]
        else:
            compact_logits = logits_output.next_token_logits
            strided_logits = ScatterCompactToStrided.execute(
                compact=compact_logits,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=stride,
            )
            compact_hidden = logits_output.hidden_states
            if compact_hidden is None:
                raise RuntimeError(
                    "DSpark verify requires target hidden states, got None."
                )
            hidden_strided = ScatterCompactToStrided.execute(
                compact=compact_hidden,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=stride,
            )
        apply_logits_adjustments_strided(
            next_token_logits=strided_logits,
            sampling_info=sampling_info,
            verify_num_draft_tokens=stride,
        )
        logits_output.next_token_logits = strided_logits
        logits_output.hidden_states = hidden_strided
        return target_verify, hidden_strided

    def run_qwen3_5_compact(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        sampling_info,
        materialize_logits: bool,
    ) -> Optional[tuple[TargetVerifyResult, CompactVerifyOutput]]:
        runtime_layout, fallback_reason = self._runtime_layout_for_graph(layout)
        if runtime_layout is None:
            self._warn_static_fallback_once(
                (fallback_reason,),
                "DSPARK_RAGGED_FALLBACK destination=static reason=%s "
                "semantic_bs=%d semantic_tokens=%s",
                fallback_reason,
                layout.bs,
                layout.total_verify_tokens,
            )
            return None

        ragged_window = BuildRaggedVerifyWindow.execute_qwen3_5(
            batch=batch,
            layout=runtime_layout,
            semantic_layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )
        verify_forward_batch, can_run_cuda_graph = self._prepare_ragged_forward_batch(
            batch=batch,
            semantic_layout=layout,
            runtime_layout=runtime_layout,
            ragged_window=ragged_window,
        )
        if not can_run_cuda_graph:
            self._warn_static_fallback_once(
                ("cuda_graph_admission_failed",),
                "DSPARK_RAGGED_FALLBACK destination=static "
                "reason=cuda_graph_admission_failed semantic_bs=%d "
                "graph_slots=%d graph_tokens=%d",
                layout.bs,
                runtime_layout.runtime_num_slots,
                runtime_layout.graph_num_tokens,
            )
            return None

        target_verify = self._run_prepared_ragged(
            verify_forward_batch=verify_forward_batch,
        )

        logits_output = target_verify.logits_output
        compact_hidden = logits_output.hidden_states
        if compact_hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        compact_output = CompactVerifyOutput(
            runtime_layout=runtime_layout,
            semantic_layout=layout,
            ragged_window=ragged_window,
            compact_logits=logits_output.next_token_logits,
            compact_hidden=compact_hidden,
        )
        if materialize_logits:
            logits_output.next_token_logits = self.materialize_compact_logits(
                compact_output=compact_output,
                sampling_info=sampling_info,
            )
        return target_verify, compact_output

    def materialize_compact_logits(
        self,
        *,
        compact_output: CompactVerifyOutput,
        sampling_info,
    ) -> torch.Tensor:
        strided_logits = scatter_runtime_compact_to_strided_triton(
            compact=compact_output.compact_logits,
            runtime_layout=compact_output.runtime_layout,
            semantic_layout=compact_output.semantic_layout,
            fill_value=0.0,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        apply_logits_adjustments_strided(
            next_token_logits=strided_logits,
            sampling_info=sampling_info,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        return strided_logits

    def _verify_backend_self_adds_seq_lens(self) -> bool:
        if self._verify_backend_self_adds_seq_lens_cache is None:
            backend = self.target_worker.model_runner.attn_backend
            self._verify_backend_self_adds_seq_lens_cache = hasattr(
                backend, "make_forward_metadata_from_raw_verify"
            )
        return self._verify_backend_self_adds_seq_lens_cache
