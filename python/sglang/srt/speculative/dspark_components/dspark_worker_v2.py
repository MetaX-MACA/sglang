import logging
from contextlib import ExitStack, contextmanager
from typing import Optional

import torch

from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
    compute_position,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    draft_is_deepseek_v4,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_accept import (
    accept_compact_greedy,
    accept_draft_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator import (
    BlockAcceptEstimateRecorder,
    create_block_accept_estimate_recorder,
)
from sglang.srt.speculative.dspark_components.dspark_confidence_metrics import (
    ConfidenceMetricsProbe,
)
from sglang.srt.speculative.dspark_components.dspark_decision_dump import (
    DsparkDecisionDumper,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DsparkDraftSampler,
    make_next_draft_input,
    uses_anchor_plus_gamma_layout,
)
from sglang.srt.speculative.dspark_components.dspark_draft_proposer import (
    DraftBlockProposer,
)
from sglang.srt.speculative.dspark_components.dspark_info_dumper import (
    DecodeStepObservation,
    DsparkInfoDumper,
    InfoSegment,
    resolve_components,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_sps_recorder import (
    SpsDataRecorder,
)
from sglang.srt.speculative.dspark_components.dspark_sts_recorder import (
    StsDataRecorder,
)
from sglang.srt.speculative.dspark_components.dspark_target_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_utils import (
    dspark_gamma_from_num_draft_tokens,
    is_qwen3_5_family_dspark_enabled,
    is_qwen3_5_family_target,
    parse_dspark_draft_config,
    qwen3_5_family_draft_config_context,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify_epilogue import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
)
from sglang.srt.speculative.dspark_components.dspark_verify_planner import (
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.kernels.build_out_tokens import (
    BuildOutTokens,
)
from sglang.srt.speculative.dspark_components.kernels.finalize_accept_lens import (
    FinalizeAcceptLens,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_tp_context,
)
from sglang.srt.speculative.spec_utils import draft_tp_context, record_stream_each
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)

_STS_COLLECT_FLUSH_EVERY: int = 256


class DSparkWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device
        # Only the tail PP stage owns the complete DSPARK draft model and its
        # private KV state. Earlier stages execute target forward only.
        self._draft_worker = None
        self._hosts_draft = get_pp_group().is_last_rank
        if not self._hosts_draft:
            return
        self._qwen3_5_family_enabled = is_qwen3_5_family_dspark_enabled(
            self.model_runner
        )
        if (
            is_qwen3_5_family_target(self.model_runner)
            and not self._qwen3_5_family_enabled
        ):
            raise RuntimeError(
                "Qwen3.5/Qwen3.6 DSpark support is experimental. Set "
                "SGLANG_ENABLE_DSPARK_QWEN3_5_FAMILY=1 to enable it."
            )

        self._draft_is_moe = draft_is_deepseek_v4(server_args=server_args)
        self._draft_dp_context_enabled = (
            server_args.enable_dp_attention and not self._draft_is_moe
        )
        attn_tp_size = server_args.tp_size // max(server_args.dp_size, 1)
        # if server_args.enable_dp_attention and self._draft_is_moe and attn_tp_size > 1:
        #     raise ValueError(
        #         "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
        #         "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
        #         "MoE-under-DP all-reduce."
        #     )

        with (
            self._draft_context(),
            qwen3_5_family_draft_config_context(self._qwen3_5_family_enabled),
        ):
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DSPARK",
                random_seed=target_worker.random_seed,
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None

        dspark_config = parse_dspark_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config,
            enable_qwen3_5_family_aliases=self._qwen3_5_family_enabled,
        )
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0; got "
                f"markov_rank={dspark_config.markov_rank}."
            )
        if server_args.speculative_num_draft_tokens is None:
            gamma = int(dspark_config.resolve_gamma(default=None) or 0)
            if gamma < 1:
                raise ValueError(
                    "DSpark could not resolve gamma from the draft config and "
                    "speculative_num_draft_tokens is unset."
                )
            self.gamma = gamma
        else:
            self.gamma = dspark_gamma_from_num_draft_tokens(
                int(server_args.speculative_num_draft_tokens)
            )
            config_gamma = dspark_config.resolve_gamma(default=None)
            if config_gamma is not None and int(config_gamma) != self.gamma:
                logger.warning(
                    "DSpark gamma mismatch: using gamma=%s (from "
                    "speculative_num_draft_tokens=%s) but draft config block_size=%s.",
                    self.gamma,
                    server_args.speculative_num_draft_tokens,
                    config_gamma,
                )
        self.verify_num_draft_tokens = self.gamma + 1
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens

        if dspark_config.mask_token_id is None:
            raise ValueError(
                "DSpark requires mask_token_id to be set in the draft model config."
            )
        self._mask_token_id = int(dspark_config.mask_token_id)
        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if self._mask_token_id >= vocab_size:
            raise ValueError(
                f"DSpark mask_token_id={self._mask_token_id} is outside the target "
                f"vocab size {vocab_size}."
            )

        if self.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, mask_token_id=%s, "
                "markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(
                self.verify_num_draft_tokens
                if uses_anchor_plus_gamma_layout(self.draft_model)
                else self.gamma
            ),
            device=self.device,
        )

        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )
        self.draft_model.attach_shared_modules(
            embed_tokens=self._resolve_target_embed_tokens(target_model),
            lm_head=lm_head,
        )

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=self.tp_rank,
            server_args=self.server_args,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        if (
            server_args.enable_dp_attention
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under --enable-dp-attention does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        self._kv_injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
        )
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            dp_moe_sync=self._draft_is_moe and server_args.enable_dp_attention,
        )
        self._verify_epilogue = None
        if (
            not self._qwen3_5_family_enabled
            and self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
            and is_cuda()
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(server_args.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )
        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            verify_epilogue=self._verify_epilogue,
            qwen3_5_family_enabled=self._qwen3_5_family_enabled,
        )

        self._sts_recorder: Optional[StsDataRecorder] = None

        self._block_accept_recorder: Optional[BlockAcceptEstimateRecorder] = (
            create_block_accept_estimate_recorder(
                gamma=self.gamma, device=self.device, tp_rank=self.tp_rank
            )
        )

        self._forced_budget_frac: Optional[float] = None

        self._sps_recorder: Optional[SpsDataRecorder] = None
        if envs.SGLANG_DSPARK_ENABLE_SPS_RECORD.get():
            self._sps_recorder = SpsDataRecorder()

        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        self._simulated_correct_drafts_buf: Optional[torch.Tensor] = None
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )
        if self._simulate_acc_len > 0 and self._block_accept_recorder is not None:
            raise ValueError(
                "SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH cannot be combined with "
                "SGLANG_SIMULATE_ACC_LEN (simulated correct_len breaks the "
                "accept-probability bookkeeping of the estimator)."
            )

        self._confidence_probe = ConfidenceMetricsProbe(
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.tp_rank,
        )

        self._decision_dumper = DsparkDecisionDumper(
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.tp_rank,
        )

        self._info_dumper = DsparkInfoDumper(
            components=resolve_components(envs.SGLANG_DSPARK_DEBUG_DUMP.get()),
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.tp_rank,
            device=self.device,
            mode_value=self._verify_planner.mode_value,
            sps_report_interval=envs.SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL.get(),
        )

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._hosts_draft and self._verify_planner.carries_confidence

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if not self._hosts_draft:
            return super().spec_v2_attn_backends
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    @contextmanager
    def _draft_context(self):
        with ExitStack() as stack:
            if self._draft_dp_context_enabled:
                stack.enter_context(draft_tp_context(get_attention_tp_group()))
            stack.enter_context(speculative_moe_backend_context())
            stack.enter_context(speculative_moe_a2a_backend_context())
            yield

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if not self._hosts_draft:
            return
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        if not self._hosts_draft:
            return
        with self._draft_context():
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        if not self._hosts_draft:
            return
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context():
            if capture_decode_cuda_graph:
                self._draft_sampler = self._maybe_build_draft_sampler()
                if self._draft_sampler is not None:
                    self.draft_model_runner.capture_tail_hooks.append(
                        make_draft_sampler_capture_hook(self._draft_sampler)
                    )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _maybe_build_draft_sampler(self):
        def _eager(reason):
            if self.tp_rank == 0:
                logger.info(
                    "DSpark draft greedy proposal kept eager (reason=%s).", reason
                )
            return None

        if self.gamma <= 0:
            return _eager("gamma<=0")
        if not hasattr(self.draft_model, "compute_base_logits"):
            return _eager("no compute_base_logits")
        if getattr(self.draft_model, "markov_head", None) is None:
            return _eager("no markov head")
        if self.tp_rank == 0:
            logger.info(
                "DSpark draft greedy proposal folded into the draft cuda graph."
            )
        return DsparkDraftSampler(
            model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
            device=self.device,
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        pass

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        if not self._hosts_draft:
            return
        self._forced_budget_frac = frac
        self._verify_planner.set_forced_budget_frac(frac)

    def _recorder_verify_tokens(self, *, bs: int, verify_ids_2d) -> int:
        forced_frac = self._forced_budget_frac
        if (
            forced_frac is None
            or self._verify_planner.mode_value == RaggedVerifyMode.STATIC.value
        ):
            return int(verify_ids_2d.numel())
        max_len = int(verify_ids_2d.shape[1])
        budget = int(float(forced_frac) * bs * max_len)
        return bs + min(budget, bs * (max_len - 1))

    def dump_sps_records(self) -> Optional[dict]:
        if not self._hosts_draft or self._sps_recorder is None:
            return None
        dumped = {
            "mode": self._verify_planner.mode_value,
            "verify_num_draft_tokens": int(self.verify_num_draft_tokens),
            "simulate_acc_len": (
                self._simulate_acc_len if self._simulate_acc_len > 0 else None
            ),
            "records": self._sps_recorder.dump_records(),
        }
        if self._qwen3_5_family_enabled:
            dumped["qwen3_5_family_enabled"] = True
        return dumped

    def dump_info_records(self) -> Optional[dict]:
        if not self._hosts_draft:
            return None
        dumped = self._info_dumper.dump()
        if dumped is None:
            return None
        if self._qwen3_5_family_enabled:
            dumped["qwen3_5_family_enabled"] = True
        dumped["simulate_acc_len"] = (
            self._simulate_acc_len if self._simulate_acc_len > 0 else None
        )
        return dumped

    def clear_info_records(self) -> None:
        if not self._hosts_draft:
            return
        self._info_dumper.clear()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        if not self._hosts_draft or self._block_accept_recorder is None:
            return None
        return self._block_accept_recorder.estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        if not self._hosts_draft or self._block_accept_recorder is None:
            return
        self._block_accept_recorder.note_request_finished(
            rid=rid, natural_stop=natural_stop
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> GenerationBatchResult:
        if not self._hosts_draft:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                batch, pp_proxy_tensors=pp_proxy_tensors
            )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._verify_planner.note_non_decode_step()
            if self._sps_recorder is not None:
                self._sps_recorder.note_non_decode_step()
            self._info_dumper.note_non_decode_step()
            if self._block_accept_recorder is not None:
                self._block_accept_recorder.flush()
            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

        return self._forward_decode(batch, on_publish)

    def _forward_prefill(
        self,
        batch: ScheduleBatch,
        on_publish,
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if self.server_args.enable_dp_attention:
                batch.capture_hidden_mode = CaptureHiddenMode.FULL
                self.target_worker.forward_batch_generation(
                    batch, pp_proxy_tensors=pp_proxy_tensors
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(
            batch, pp_proxy_tensors=pp_proxy_tensors
        )
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = next_token_ids.device
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _run_idle_verify_participation(self, batch: ScheduleBatch) -> None:
        if self._verify_epilogue is not None:
            self._verify_epilogue.begin_step(None, armed=False)
        idle_layout = self._idle_verify_ragged_layout(batch)
        num_dummy_tokens = (
            idle_layout.graph_num_tokens if idle_layout is not None else 0
        )
        verify_input = DFlashVerifyInput(
            draft_token=torch.zeros(
                (num_dummy_tokens,), dtype=torch.int64, device=self.device
            ),
            positions=torch.zeros(
                (num_dummy_tokens,), dtype=torch.int64, device=self.device
            ),
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=idle_layout,
        )
        batch.out_cache_loc = torch.zeros(
            (num_dummy_tokens,), dtype=torch.int64, device=self.device
        )
        if idle_layout is not None:
            num_dummy_slots = int(idle_layout.verify_lens.numel())
            batch.seq_lens = torch.ones(
                (num_dummy_slots,), dtype=torch.int64, device=self.device
            )
            batch.req_pool_indices = torch.zeros(
                (num_dummy_slots,), dtype=torch.int64, device=self.device
            )
            batch.seq_lens_cpu = torch.ones((num_dummy_slots,), dtype=torch.int64)
            batch.seq_lens_sum = num_dummy_slots
            batch.forward_mode = ForwardMode.TARGET_VERIFY
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
        )

    def inject_pd_hidden_chunk(
        self,
        req,
        hidden: torch.Tensor,
        hidden_start: int,
    ) -> Optional[torch.cuda.Event]:
        if hidden is None or hidden.numel() == 0:
            return None
        row_len = int(hidden.shape[0])
        pos = torch.arange(
            int(hidden_start),
            int(hidden_start) + row_len,
            dtype=torch.int64,
            device=self.device,
        )
        req_pool_idx = int(req.req_pool_idx)
        cache_loc = self.model_runner.req_to_token_pool.req_to_token[req_pool_idx, pos]
        return self._kv_injector.inject_target_hidden(
            target_hidden=hidden,
            cache_loc=cache_loc,
            positions=pos,
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_decode(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            if self._sps_recorder is not None:
                self._sps_recorder.note_non_decode_step()
            self._info_dumper.note_non_decode_step()
            if self.server_args.enable_dp_attention:
                if self._draft_is_moe:
                    with self._draft_context():
                        self._proposer.run_idle_participation(batch)
                self._run_idle_verify_participation(batch)
            return self._decode_idle_result(on_publish=on_publish)

        # batch.seq_lens.record_stream(
        #     torch.get_device_module(self.device).current_stream()
        # )
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_each(
            (batch.seq_lens, batch.req_pool_indices),
            fwd_stream,
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        self._info_dumper.begin_step()

        target_model = self.target_worker.model_runner.model

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        sampling_info = batch.sampling_info
        if (
            self._qwen3_5_family_enabled
            and self._simulate_acc_len > 0
            and sampling_info is not None
            and not sampling_info.is_all_greedy
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN for DSpark is a greedy-only smoke-test "
                "mode. Set request temperature=0 (or top_k=1) when using it; "
                "sampling requests would require target sampling from the simulated "
                "bonus position and are intentionally rejected to avoid misleading "
                "outputs."
            )
        with self._draft_context(), self._info_dumper.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens

        confidence = proposal.confidence
        if confidence is None:
            confidence = self._verify_planner.compute_confidence_tensor(
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
                confidence_tap=proposal.confidence_tap,
            )

        verify_token_budget = self._resolve_verify_token_budget(
            batch=batch,
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
        )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            else None
        )
        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        # Grammar-constrained decoding: the verify chain is linear, so the mask is
        # built over it directly. Build the tree before the target verify launch so
        # its async D2H overlaps that forward.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        if self._sps_recorder is not None:
            sps_dp_tier = self._dp_verify_tier_num_tokens(batch)
            self._sps_recorder.observe_decode_step(
                forward_ct=int(batch.forward_iter),
                num_running_reqs=bs,
                num_verify_tokens=self._recorder_verify_tokens(
                    bs=bs, verify_ids_2d=verify_ids_2d
                ),
                verify_tokens_local=int(batch.spec_verify_tier_num_tokens),
                verify_tokens_dp_synced=(
                    -1 if sps_dp_tier is None else int(sps_dp_tier)
                ),
                verify_tokens_graph_key=(
                    layout.graph_num_tokens
                    if layout is not None
                    else int(verify_ids_2d.numel())
                ),
            )

        compact_native_accept = (
            self._qwen3_5_family_enabled
            and run_compact
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
            and (self._block_accept_recorder is None or proposal.folded)
            # A live grammar is masked on the verify logits below, which the
            # native compact accept path does not read.
            and not batch.has_grammar
        )
        fold_eligible = (
            not self._qwen3_5_family_enabled
            and self._verify_executor.verify_epilogue is not None
            and proposal.folded
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
            # The folded epilogue accepts inside the cuda graph off its own
            # buffers, where the mask below never lands.
            and not batch.has_grammar
        )
        compact_output = None
        hidden_strided = None
        with self._info_dumper.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                if self._qwen3_5_family_enabled:
                    compact_result = self._verify_executor.run_qwen3_5_compact(
                        batch=batch,
                        layout=layout,
                        draft_block_ids=draft_block_ids,
                        draft_tokens=draft_tokens,
                        bs=bs,
                        device=device,
                        sampling_info=sampling_info,
                        materialize_logits=not compact_native_accept,
                    )
                    if compact_result is None:
                        run_compact = False
                        layout = None
                    else:
                        target_verify, compact_output = compact_result
                else:
                    target_verify, hidden_strided = self._verify_executor.run_compact(
                        batch=batch,
                        layout=layout,
                        draft_block_ids=draft_block_ids,
                        draft_tokens=draft_tokens,
                        bs=bs,
                        device=device,
                        sampling_info=sampling_info,
                        inject_gate=fold_eligible,
                    )
            if not run_compact:
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                )
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        if batch.has_grammar:
            # Host-side tree traversal; overlaps the target verify forward above.
            vocab_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                verify_input=draft_input,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
            )
            if vocab_mask is not None:
                assert draft_input.grammar is not None
                draft_input.grammar.apply_vocab_mask(
                    logits=logits_output.next_token_logits, vocab_mask=vocab_mask
                )

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = fold_eligible and run_compact and can_run_cuda_graph
        if folded_accept:
            accept = epilogue.read_accept(bs)
            correct_len = accept.correct_len
            bonus = accept.bonus
            cap_trim_lens = accept.cap_trim_lens
            commit_lens = accept.commit_lens
            new_seq_lens = accept.new_seq_lens
            out_tokens = accept.out_tokens
            observed_correct_len = correct_len
        elif compact_native_accept and run_compact:
            if compact_output is None:
                raise RuntimeError("DSpark compact accept output is missing.")
            correct_len, bonus, cap_trim_lens = accept_compact_greedy(
                candidates=verify_ids_2d,
                compact_target_logits=compact_output.compact_logits,
                runtime_layout=compact_output.runtime_layout,
                semantic_layout=compact_output.semantic_layout,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
            )
            observed_correct_len = correct_len
        else:
            correct_len, bonus, cap_trim_lens = accept_draft_tokens(
                candidates=verify_ids_2d,
                target_logits=logits_output.next_token_logits,
                draft_block=draft_block,
                sampling_info=sampling_info,
                draft_input=draft_input,
                gamma=self.gamma,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                cutoff_layout=layout,
            )
            observed_correct_len = correct_len

        if not folded_accept:
            if self._simulate_acc_len > 0 and self._qwen3_5_family_enabled:
                correct_len, bonus, cap_trim_lens = self._apply_simulated_accept(
                    logits_output=logits_output,
                    bs=bs,
                    correct_len=correct_len,
                    cap_trim_lens=cap_trim_lens,
                )
            elif self._simulate_acc_len > 0:
                correct_len = self._simulated_correct_len(
                    bs=bs, dtype=correct_len.dtype, device=correct_len.device
                )

            finalized = FinalizeAcceptLens.execute(
                correct_len=correct_len,
                cap_trim_lens=cap_trim_lens,
                prefix_lens=prefix_lens,
            )
            commit_lens = finalized.commit_lens
            new_seq_lens = finalized.new_seq_lens
            cap_trim_lens = finalized.cap_trim_lens
            out_tokens = BuildOutTokens.execute(
                draft_tokens=draft_tokens,
                correct_len=correct_len,
                bonus=bonus,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                gamma=self.gamma,
            )
        if self._qwen3_5_family_enabled:
            self._verify_executor.commit_recurrent_states(
                batch=batch,
                seq_lens_pre_verify=prefix_lens,
                seq_lens_post_verify=new_seq_lens,
                commit_lens=commit_lens,
            )
        online_c128 = getattr(self.model_runner.attn_backend, "online_c128_mtp", None)
        if online_c128 is not None and online_c128.enabled():
            online_c128.commit_pending(
                req_pool_indices=batch.req_pool_indices,
                seq_lens=new_seq_lens,
            )
        if on_publish is not None:
            if confidence is not None:
                on_publish(
                    new_seq_lens,
                    confidence=confidence,
                    confidence_seq_lens=prefix_lens,
                )
            else:
                on_publish(new_seq_lens)

        folded_commit = folded_accept and epilogue.folds_commit
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                compact_output=compact_output,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        if not proposal.folded:
            self._maybe_record_sts_collect(
                num_correct_drafts=observed_correct_len,
                verify_ids_2d=verify_ids_2d,
                target_logits=logits_output.next_token_logits,
                bs=bs,
            )
            self._confidence_probe.maybe_observe(
                carries_confidence=self._verify_planner.carries_confidence,
                is_compact_mode=self._verify_planner.is_compact_mode,
                confidence_raw=self._verify_planner.last_confidence_raw,
                verify_ids_2d=verify_ids_2d,
                target_logits=logits_output.next_token_logits,
                bs=bs,
            )
        self._decision_dumper.maybe_dump(
            forward_ct=batch.forward_iter,
            bs=bs,
            mode=self._verify_planner.mode_value,
            budget=verify_token_budget,
            lag_steps=self._verify_planner.lag_steps,
            verify_lens=layout.verify_lens if layout is not None else None,
            confidence=confidence,
            req_pool_indices=batch.req_pool_indices,
            rids=[req.rid for req in batch.reqs],
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            bonus_tokens=bonus,
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            commit_lens=commit_lens,
        )
        if self._block_accept_recorder is not None and not proposal.folded:
            self._block_accept_recorder.observe_verify_step(
                forward_ct=int(batch.forward_iter),
                rids=[req.rid for req in batch.reqs],
                draft_tokens=draft_tokens,
                corrected_logits=draft_block.corrected_logits,
                draft_temperatures=draft_block.temperatures,
                greedy_mask=draft_block.greedy_mask,
                target_logits=logits_output.next_token_logits,
                target_temperatures=(
                    sampling_info.temperatures
                    if sampling_info is not None
                    else draft_block.temperatures
                ),
                truncated_sampling_mask=(
                    (sampling_info.top_ks != TOP_K_ALL)
                    | (sampling_info.top_ps != 1.0)
                    | (sampling_info.min_ps > 0)
                    if sampling_info is not None
                    else None
                ),
                logits_adjustments_are_noop=verify_logits_adjustments_are_noop(
                    sampling_info
                ),
                correct_len=correct_len,
                cap_trim_lens=cap_trim_lens,
                bonus=bonus,
                prefix_lens=prefix_lens,
                layout=layout,
            )
        if self._info_dumper.enabled:
            info_dp_tier = self._dp_verify_tier_num_tokens(batch)
            budget_decision = self._verify_planner.take_budget_decision()
            predicted_step_ms = (
                None
                if budget_decision is None
                or budget_decision.predicted_step_seconds is None
                else budget_decision.predicted_step_seconds * 1e3
            )
            predicted_theta = (
                None if budget_decision is None else budget_decision.predicted_theta
            )
            self._info_dumper.observe_decode_step(
                DecodeStepObservation(
                    forward_ct=int(batch.forward_iter),
                    bs=bs,
                    mode=self._verify_planner.mode_value,
                    budget=verify_token_budget,
                    lag_steps=self._verify_planner.lag_steps,
                    num_verify_tokens=(
                        layout.graph_num_tokens
                        if layout is not None
                        else int(verify_ids_2d.numel())
                    ),
                    verify_tokens_local=int(batch.spec_verify_tier_num_tokens),
                    verify_tokens_dp_synced=(
                        -1 if info_dp_tier is None else int(info_dp_tier)
                    ),
                    verify_tokens_graph_key=(
                        layout.graph_num_tokens
                        if layout is not None
                        else int(verify_ids_2d.numel())
                    ),
                    predicted_step_ms=predicted_step_ms,
                    predicted_theta=predicted_theta,
                    verify_lens=layout.verify_lens if layout is not None else None,
                    confidence=confidence,
                    req_pool_indices=batch.req_pool_indices,
                    prefix_lens=prefix_lens,
                    draft_tokens=draft_tokens,
                    bonus_tokens=bonus,
                    correct_len=correct_len,
                    cap_trim_lens=cap_trim_lens,
                    commit_lens=commit_lens,
                    rids=[req.rid for req in batch.reqs],
                )
            )

        next_draft_input = make_next_draft_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            block_accept_lens=commit_lens + cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=new_seq_lens,
            extra_keep_alive_refs=[target_verify.verify_forward_batch],
        )

    def _simulated_correct_len(
        self, *, bs: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        buf = self._simulated_correct_drafts_buf
        if buf is None or buf.numel() < bs or buf.dtype != dtype:
            correct_target = int(
                round(min(max(self._simulate_acc_len - 1.0, 0.0), float(self.gamma)))
            )
            buf = torch.full(
                (max(bs, 512),), correct_target, dtype=dtype, device=device
            )
            self._simulated_correct_drafts_buf = buf
        return buf[:bs]

    def _apply_simulated_accept(
        self,
        *,
        logits_output,
        bs: int,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        simulated_correct_len = self._simulated_correct_len(
            bs=bs, dtype=correct_len.dtype, device=correct_len.device
        )
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )
        bonus = target_predict[
            torch.arange(bs, device=target_predict.device),
            simulated_correct_len.to(torch.long),
        ].to(torch.int64)
        return simulated_correct_len, bonus, torch.zeros_like(cap_trim_lens)

    def _maybe_record_sts_collect(
        self,
        *,
        num_correct_drafts: torch.Tensor,
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor,
        bs: int,
    ) -> None:
        collect_path = envs.SGLANG_DSPARK_STS_COLLECT_PATH.get()
        if not collect_path:
            return
        if not self._verify_planner.carries_confidence:
            return
        confidence_raw = self._verify_planner.last_confidence_raw
        if confidence_raw is None:
            return
        if self._sts_recorder is None:
            self._sts_recorder = StsDataRecorder(
                path_stem=collect_path,
                gamma=self.gamma,
                flush_every=_STS_COLLECT_FLUSH_EVERY,
            )
        if not self._qwen3_5_family_enabled:
            target_predict = torch.argmax(target_logits, dim=-1).view(
                bs, self.verify_num_draft_tokens
            )
            num_correct_drafts, _ = compute_dflash_correct_drafts_and_bonus(
                candidates=verify_ids_2d,
                target_predict=target_predict,
            )
        self._sts_recorder.record(
            confidence_raw=confidence_raw,
            num_correct_drafts=num_correct_drafts,
        )

    def _resolve_verify_token_budget(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        confidence: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
    ) -> Optional[int]:
        if not self._verify_planner.schedules_verify_budget or confidence is None:
            return None
        if not self.server_args.disable_overlap_schedule:
            return draft_input.verify_token_budget
        return self._verify_planner.compute_budget_sync(
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        if not self._hosts_draft or not self._verify_planner.schedules_verify_budget:
            return None
        return self._verify_planner.prepare_verify_budget
