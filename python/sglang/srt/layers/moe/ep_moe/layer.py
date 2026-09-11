from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, List

import torch
from mcoplib.op import  fused_silu_mul_dq_reorder_quant
from sglang.kernels.ops.moe.ep_moe_kernels import (
    grouped_gemm_triton,
    m_grouped_gemm_nt_masked,
    silu_and_mul_masked_fwd_no_pack,
)

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import (
    get_is_extend_in_batch,
    set_is_extend_in_batch,
)
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    moe_forward_piecewise_cuda_graph_impl,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    NPUCompressedTensorsW4A16Int4DynamicMoE,
)
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


logger = logging.getLogger(__name__)

class GroupedGemmRunner(torch.nn.Module):
    flashinfer_gemm_warpper = None

    def __init__(
        self,
        device,
        use_flashinfer: bool = False,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        self.use_per_token_if_dynamic = use_per_token_if_dynamic
        if self.use_flashinfer and GroupedGemmRunner.flashinfer_gemm_warpper is None:
            GroupedGemmRunner._init_flashinfer_wrapper(device)

    @classmethod
    def _init_flashinfer_wrapper(cls, device):
        from flashinfer import SegmentGEMMWrapper

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        cls.flashinfer_gemm_warpper = SegmentGEMMWrapper(workspace_buffer)

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
        c_dtype=None,
    ):
        if self.use_flashinfer:
            # TODO: flashinfer
            assert False
            assert GroupedGemmRunner.flashinfer_gemm_warpper is not None
            c = GroupedGemmRunner.flashinfer_gemm_warpper.run(
                x=a,
                weights=b,
                batch_size=batch_size,
                weight_column_major=weight_column_major,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
            )
        else:
            assert weight_column_major == True
            c = grouped_gemm_triton(
                a,
                b,
                c,
                batch_size,
                weight_column_major,
                seg_indptr,
                weight_indices,
                use_fp8_w8a8,
                scale_a,
                scale_b,
                block_shape=block_shape,
                c_dtype=c_dtype,
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )
        return c

class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        is_humming = (
            get_moe_runner_backend().is_humming()
            or get_moe_runner_backend().is_auto()
            and quant_config is not None
            and quant_config.get_name() == "humming"
        )
        if get_moe_a2a_backend().is_deepep_v2():
            self.deprecate_flag = True
        elif is_humming:
            self.deprecate_flag = True
        elif _use_aiter:
            self.deprecate_flag = True
        elif _is_npu:
            self.deprecate_flag = True
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and \
            (isinstance(quant_config, Fp8Config) or \
            isinstance(quant_config, CompressedTensorsConfig) or \
            isinstance(quant_config, W8A8Int8Config)):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ):
            self.deprecate_flag = True
        elif (
            get_moe_runner_backend().is_flashinfer_cutedsl()
            and quant_config is not None
            and quant_config.get_name() in ("modelopt_fp4", "modelopt_mixed")
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and get_moe_runner_backend().is_deep_gemm()
            and quant_config is not None
            and quant_config.get_name() == "mxfp4"
        ):
            # MXFP4 experts (e.g. Kimi K3) on the DeepGEMM fp8_fp4 W4A8 path:
            # route through the modern FusedMoE runner (Mxfp4MoEMethod.apply).
            self.deprecate_flag = True
        elif (
            quant_config is None
            and self.w13_weight.dtype == torch.bfloat16
            and get_moe_runner_backend().is_deep_gemm()
            and (get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_pplx())
            and not _is_npu
            and not _is_hip
        ):
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), "Unquantized DeepEP MoE requires DeepGEMM BF16"
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False

        self.deepep_mode = get_deepep_mode()
        self.activation = activation
        self.grouped_gemm_runner = None
        # if (
        #     self.deepep_mode.enable_low_latency()
        #     and not _is_npu
        #     and not _is_hip
        #     and quant_config is not None
        # ):
        #     # AMD HIP and NPU support low_latency DeepEP without DeepGEMM.
        #     assert (
        #         deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        #     ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"

    def _a2a_forward_with_output_impl(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # eager run under breakable cuda graph
        saved_is_extend_in_batch = get_is_extend_in_batch()
        set_is_extend_in_batch(True)
        try:
            output.copy_(
                self.forward_impl(
                    hidden_states,
                    StandardTopKOutput(topk_weights, topk_ids, router_logits),
                )
            )
        finally:
            set_is_extend_in_batch(saved_is_extend_in_batch)

    def _a2a_forward_capture_stub(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Capture pass only: record the buffer address, skip the
        # rank-coupled a2a. Warmup and replay run the real body.
        output.zero_()

    a2a_forward_with_output = eager_on_graph(
        True, capture_stub=_a2a_forward_capture_stub
    )(_a2a_forward_with_output_impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        # DeepEP NORMAL mode is not capturable; run it as an eager node.
        if is_in_breakable_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for breakable cuda graph"
            output = torch.empty_like(hidden_states)
            self.a2a_forward_with_output(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                output,
            )
            return output
        if is_in_tc_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        return self.dispatcher.combine(combine_input=combine_input)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):

        if self.deprecate_flag:
            return super().run_moe_core(dispatch_output)

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            assert False, "_use_aiter not support deepep"
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            output = self.forward_normal(dispatch_output)
            #if self.quant_config is None:
            #    raise NotImplementedError(
            #        "Unquantized DeepEP MoE currently supports low_latency mode only"
            #    )
            #elif self.use_w4afp8:
            #    output = self.forward_cutlass_w4afp8(dispatch_output)
            #else:
            #    assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            output = self.forward_deepgemm_masked_v1(dispatch_output)
            #if self.use_w4afp8:
            #    output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            #else:
            #    assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )

        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_normal(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        # TODO  this function need  mcops ready
        
        hidden_states, _, _,_, _, reorder_topk_ids, seg_indptr = dispatch_output
        assert self.quant_method is not None
        assert self.activation == "silu"
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device, use_flashinfer=False  # TODO: use flashinfer
            )

        # if self.activation_scheme == "dynamic" and not self.use_block_quant:
        #     max_value = (
        #         torch.max(hidden_states)
        #         .repeat(self.num_local_experts)
        #         .to(torch.float32)
        #     )
        #     self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_local_experts,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        gateup_output = torch.empty(
            hidden_states.shape[0],
            self.w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if hidden_states.shape[0] > 0:
            # if hidden_states.shape[1] == 3840 and self.fused_quant_packed:
            #     # scale_a_ptr = hidden_states[:, 3584:3586].view(torch.float32).contiguous()
            #     # hidden_states_int8 = hidden_states[:, :3584].view(torch.int8).contiguous()
            #     hidden_states_int8, scale_a_ptr = triton_unpack(hidden_states, 3584, 3586)
            # else:
            hidden_states_int8 = hidden_states
            scale_a_ptr = None
            gateup_output = self.grouped_gemm_runner(
                a=hidden_states_int8,
                b=self.w13_weight,
                c=gateup_output,
                batch_size=self.num_local_experts,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a = scale_a_ptr,
                scale_b=(
                    self.w13_weight_scale_inv
                    if self.use_block_quant
                    else self.w13_weight_scale
                ),
                block_shape=None,
            )

        # Act
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.nn.Parameter(torch.ones(
                self.num_local_experts,
                dtype=torch.float32,
                device=hidden_states.device,
            ))

        if self.activation == "silu":

            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=torch.int8
            )
            down_input_scale = torch.empty(
                gateup_output.shape[0],
                1,
                device=gateup_output.device,
                dtype=torch.float32
            )
            if gateup_output.shape[0] > 0:
                fused_silu_mul_dq_reorder_quant(
                    down_input,
                    down_input_scale,
                    gateup_output,
                    reorder_topk_ids,
                    self.w2_input_scale,
                    0,
                    self.num_local_experts - 1,
                )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if down_input.shape[0] > 0:
            down_output = self.grouped_gemm_runner(
                a=down_input,
                b=self.w2_weight,
                c=down_output,
                batch_size=self.num_local_experts,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=down_input_scale,
                scale_b=(
                    self.w2_weight_scale_inv
                    if self.use_block_quant
                    else self.w2_weight_scale
                ),
                block_shape=None,
            )
        return down_output
    
    def forward_deepgemm_masked_v1(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, _,_,_, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.activation == "silu"

        # GroupGemm-0
        num_groups, m, k = hidden_states.size()
        n = self.w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
        )

        m_grouped_gemm_nt_masked(
            hidden_states,
            self.w13_weight,
            gateup_output,
            masked_m,
            expected_m,
            self.w13_weight_scale,
            use_triton_kernel = False,
            unpack_tensor=True
        )
        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=gateup_output.dtype,
        )

        # if self.use_fused_quant:
        #     down_input = silu_and_mul_masked_fwd(gateup_output, masked_m)
        # else:
        silu_and_mul_masked_fwd_no_pack(gateup_output, down_input, masked_m)

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )
        m_grouped_gemm_nt_masked(
            down_input,
            self.w2_weight,
            down_output,
            masked_m,
            expected_m,
            self.w2_weight_scale,
            use_triton_kernel = False,
            unpack_tensor=False
        )
        return down_output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    # [TODO] kk, temporary solution
    if (
        get_moe_a2a_backend().is_mori()
        or get_moe_a2a_backend().is_deepep()
        or get_moe_a2a_backend().is_deepep_v2()
        or get_moe_a2a_backend().is_mooncake()
        or get_moe_a2a_backend().is_nixl()
        or get_moe_a2a_backend().is_pplx()
    ):
        return DeepEPMoE
    return FusedMoE
