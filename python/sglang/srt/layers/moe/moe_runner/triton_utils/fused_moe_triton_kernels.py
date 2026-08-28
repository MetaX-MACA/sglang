from __future__ import annotations

import functools
from collections import OrderedDict
import os
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.moe.utils import get_moe_padding_size
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.int8_kernel import (
    per_token_group_quant_int8,
    per_token_quant_int8,
    sglang_per_token_group_quant_int8,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_sm90_supported,
)

from sgl_kernel import cutlass_moe_mm_w8a8
try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    _support_tensor_descriptor = True
except:
    _support_tensor_descriptor = False

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
from sglang.srt.environ import envs

if _is_cuda:
    from mcoplib.triton_fused_moe import sgl_fused_moe_kernel
    from mcoplib.triton_fused_moe import sgl_fused_moe_kernel_gptq_awq
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    pass

padding_size = get_moe_padding_size(_use_aiter)
enable_mctlass_fused_moe = (os.getenv("ENABLE_MCTLASS_FUSED_MOE", "1") == "1")
enable_mctlass_fused_moe_python_api = (os.getenv("ENABLE_MCTLASS_FUSED_MOE_PYTHON_API", "1") == "1")
enable_maca_sglang_fused_moe_mctlass_w4a16 = envs.ENABLE_MACA_SGLANG_FUSED_MOE_MCTLASS_W4A16.get()

if enable_mctlass_fused_moe_python_api:
    import mctlassEx
    from mctlassEx import FusedMoeGEMM
    gemm = FusedMoeGEMM()

def support_tensor_descriptor():
    return _support_tensor_descriptor


# swap_ab benefits SM90 GPUs (H20, H100, H200, etc.) for certain block shapes.
@functools.lru_cache(maxsize=8)
def should_enable_swap_ab(
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
) -> bool:
    if not _is_cuda or is_batch_invariant_mode_enabled():
        return False

    return is_sm90_supported() and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def _set_triton_tma_allocator():
    """TMA descriptors require a global allocator; set it once to avoid per-call overhead."""
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        # NOTE: keep this allocation on CUDA device
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    _TMA_ALLOCATOR_SET = True


# --- B TensorDescriptor cache (LRU) ---
_B_DESC_CACHE_MAX = 64
_B_DESC_CACHE: OrderedDict[tuple, TensorDescriptor] = OrderedDict()


def _get_b_tma_desc_cached(B: torch.Tensor, block_n: int, block_k: int):
    """
    Cache TensorDescriptor for constant weight B.
    Keyed by storage ptr + shape/stride/dtype + tile shape.
    """
    key = (
        int(B.data_ptr()),
        tuple(B.shape),
        tuple(B.stride()),
        str(B.dtype),
        int(block_n),
        int(block_k),
    )

    desc = _B_DESC_CACHE.get(key, None)
    if desc is not None:
        _B_DESC_CACHE.move_to_end(key)
        return desc

    # Create outside lock to reduce lock hold time (ok if duplicated rarely)
    desc = TensorDescriptor(
        B,
        B.shape,
        B.stride(),
        [1, block_n, block_k],
    )

    _B_DESC_CACHE[key] = desc
    _B_DESC_CACHE.move_to_end(key)
    if len(_B_DESC_CACHE) > _B_DESC_CACHE_MAX:
        _B_DESC_CACHE.popitem(last=False)

    return desc


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a8: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
    fuse_sum_all_reduce: bool = False,
    router_topk: int = 1,
    fuse_add_to_output: bool = False,
    add_output_mask: Optional[torch.Tensor] = None,
    mask_output: bool = False,
    lora_preserve_base: bool = False,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        swap_ab = should_enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    else:
        swap_ab = False
    group_size = 0
    padded_size = 0
    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation tensor-wise fp8 quantization, dynamic or static
            padded_size = padding_size
            # activations apply per-token quantization when weights apply per-channel quantization by default
            A, A_scale = scaled_fp8_quant(
                A, A_scale, use_per_token_if_dynamic=per_channel_quant
            )
        else:
            # activation block-wise fp8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            group_size = block_n
            if _is_cuda:
                A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
            else:
                A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a8 or use_int4_w4a8:
        assert B_scale is not None
        if block_shape is None:
            # activation channel-wise int8 quantization
            assert (
                per_channel_quant
            ), "int8 quantization only supports channel-wise quantization except for block-wise quantization"
            A, A_scale = per_token_quant_int8(A)
        else:
            # activation block-wise int8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            # if _is_cuda:
            #     A, A_scale = sglang_per_token_group_quant_int8(A, block_k)
            # else:
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False

    if fuse_sum_all_reduce:
        assert not c_sorted, "fuse_sum_all_reduce only supports c_sorted=False"
    if fuse_add_to_output:
        assert (
            not fuse_sum_all_reduce
        ), "fuse_add_to_output and fuse_sum_all_reduce are mutually exclusive"
        assert (
            add_output_mask is not None
        ), "add_output_mask required when fuse_add_to_output=True"
    # ===== TO BE REFACTORED ====
    if mask_output:
        assert (
            not fuse_add_to_output
        ), "mask_output and fuse_add_to_output are mutually exclusive"
        assert (
            not fuse_sum_all_reduce
        ), "mask_output and fuse_sum_all_reduce are mutually exclusive"
        assert (
            add_output_mask is not None
        ), "add_output_mask required when mask_output=True"
    # ===== END TO BE REFACTORED ====

    if (
        enable_maca_sglang_fused_moe_mctlass_w4a16 and use_int4_w4a16
        and block_shape is not None
        and block_shape[1] > 0
        and enable_mctlass_fused_moe_python_api
        and B_zp is None
    ):
        # if B_zp is not None:
        #     group_size = 64
        # else:
        #     group_size = 32
        C1 = C.view(-1, C.size(-1)).contiguous()
        gemm(
            A.shape[0], B.shape[1], A.shape[1], B.shape[0], sorted_token_ids.shape[0], top_k, A, B.view(dtype=torch.quint4x2), C1, A_scale, B_scale, None,
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, filter_expert = filter_expert,  is_blockwise=True, group_size=32, zp_b=B_zp)
    elif (
        (use_int8_w8a16 or use_int4_w4a16)
        and block_shape is not None
        and block_shape[1] > 0
    ):
        assert (
            not fuse_sum_all_reduce
        ), "fuse_sum_all_reduce is not supported for GPTQ/AWQ kernels"
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3
        assert bias is None
        sgl_fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            even_Ks=even_Ks,
            filter_expert=filter_expert,
            **config,
        )
    elif use_int8_w8a8 and enable_mctlass_fused_moe:
        if enable_mctlass_fused_moe_python_api:
            C1 = C.view(-1, C.size(-1)).contiguous()
            gemm(A.shape[0], B.shape[1], A.shape[1], B.shape[0], sorted_token_ids.shape[0], top_k, A, B, C1, A_scale, B_scale, None, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, filter_expert = filter_expert)
        else:
            cutlass_moe_mm_w8a8(A, B, C,
                                A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids,
                                num_tokens_post_padded,
                                B.shape[1], # N
                                A.shape[1], # K
                                sorted_token_ids.shape[0],
                                topk_ids.numel(), # num_valid_tokens
                                top_k,
                                mul_routed_weight)
    elif use_int4_w4a8:
        assert enable_mctlass_fused_moe_python_api, ("int4_w4a8 only support enable_mctlass_fused_moe_python_api now")
        EM = sorted_token_ids.shape[0]
        B = B.view(dtype = torch.quint4x2)
        gemm(A.size(0),
            B.size(1),
            A.size(1),
            B.size(0),
            EM,
            top_k,
            A,
            B,
            C,
            A_scale,
            B_scale,
            None,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight)
    elif use_fp8_w8a8 and enable_mctlass_fused_moe_python_api:
        C1 = C.view(-1, C.size(-1)).contiguous()
        fp8_kwargs = {
            "filter_expert": filter_expert,
            "use_fp8": True,
        }
        if block_shape is not None:
            A_scale = A_scale.T.contiguous()
            B_scale = B_scale.transpose(1, 2).contiguous()
            fp8_kwargs.update(
                is_blockwise=True,
                group_size=group_size,
                is_scale_a_1d=True,
                is_scale_b_1d=False,
                scale_a_layout="m-major",
                scale_b_layout="n-major",
            )
        gemm(
            A.shape[0],
            B.shape[1],
            A.shape[1],
            B.shape[0],
            sorted_token_ids.shape[0],
            top_k,
            A,
            B,
            C1,
            A_scale,
            B_scale,
            None,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            **fp8_kwargs,
        )
    else:
        if a_use_tma or b_use_tma:
            _set_triton_tma_allocator()

        if a_use_tma:
            a_desc = TensorDescriptor(
                A, A.shape, A.stride(), [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]]
            )
        else:
            a_desc = None
        if b_use_tma:
            # B is constant weights -> cache descriptor
            b_desc = _get_b_tma_desc_cached(
                B,
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_K"],
            )
        else:
            b_desc = None
            
        if enable_mctlass_fused_moe_python_api:
            C1 = C.view(-1, C.size(-1)).contiguous()
            gemm(A.shape[0], B.shape[1], A.shape[1], B.shape[0], sorted_token_ids.shape[0], top_k, A, B, C1, A_scale, B_scale, None, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, filter_expert = filter_expert)
        else:
            sgl_fused_moe_kernel[grid](
                A,
                a_desc,
                B,
                b_desc,
                bias,
                C,
                A_scale,
                B_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                add_output_mask,
                B.shape[1],
                B.shape[2] - padded_size,
                sorted_token_ids.shape[0],
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                bias.stride(0) if bias is not None else 0,
                bias.stride(1) if bias is not None else 0,
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                per_channel_quant=per_channel_quant,
                even_Ks=even_Ks,
                c_sorted=c_sorted,
                filter_expert=filter_expert,
                swap_ab=swap_ab,
                FUSE_ADD_TO_OUTPUT=fuse_add_to_output,
                MASK_OUTPUT=mask_output,
                LORA_PRESERVE_BASE=lora_preserve_base,
                FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,
                ROUTER_TOPK=router_topk,
                **config,
            )


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    """
    Apply activation function based on compile-time constant.

    Args:
        x: Input tensor (converted to float32 inside)
        ACTIVATION_TYPE: Compile-time constant string ("silu" or "gelu")

    Returns:
        Activated output in the same dtype as input
    """
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        raise ValueError(f"Unsupported activation: {ACTIVATION_TYPE}")


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    expert_ids_ptr,
    expert_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr = 0.0,
    HAS_SWIGLU_LIMIT: tl.constexpr = False,
):
    """
    Unified activation and multiply kernel that handles both sorted and unsorted routing,
    and both SiLU and GELU activations using compile-time constants.
    """
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    expert_id = tl.load(expert_ids_ptr + pid // expert_step)

    if expert_id == -1:
        return

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    gate_output_ptr = gateup_output_ptr
    up_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)

        if HAS_SWIGLU_LIMIT:
            gate_output = tl.minimum(gate_output, SWIGLU_LIMIT)
            up_output = tl.maximum(tl.minimum(up_output, SWIGLU_LIMIT), -SWIGLU_LIMIT)

        gate_output_activated = _apply_activation(gate_output, ACTIVATION_TYPE)
        gate_output_activated = gate_output_activated.to(InDtype)

        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


def act_and_mul_triton(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    config: Dict[str, Any],
    topk_ids: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    down_moe_use_tma: bool = False,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
) -> None:
    """
    Args:
        gateup_output: Input tensor containing gate and up outputs concatenated
        down_input: Output tensor for the result
        config: Configuration dictionary with BLOCK_SIZE_M and BLOCK_SIZE_N
        topk_ids: Expert IDs for unsorted routing (used when down_moe_use_tma=False)
        expert_ids: Expert IDs for sorted routing (used when down_moe_use_tma=True)
        down_moe_use_tma: Whether to use sorted routing layout
        activation: Activation type ("silu" or "gelu")
        swiglu_limit: if not None, clamp gate to [-inf, L] and up to [-L, L] before activation
                      (compiles a separate kernel variant via tl.constexpr).
    """
    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    expert_ids_row = topk_ids.view(-1) if not down_moe_use_tma else expert_ids
    expert_step = 1 if not down_moe_use_tma else config["BLOCK_SIZE_M"]
    has_swiglu_limit = swiglu_limit is not None
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        expert_ids_row,
        expert_step,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
        SWIGLU_LIMIT=float(swiglu_limit) if has_swiglu_limit else 0.0,
        HAS_SWIGLU_LIMIT=has_swiglu_limit,
    )


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


@triton.jit
def _fused_append_shared_experts_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    N_BASE,  # runtime scalar
    scale_factor,  # runtime scalar
    K: tl.constexpr,
    S: tl.constexpr,
):
    """
    for m in range(M):
        for n in range(K):
            fused_ids[m, n] = topk_ids[m, n]
            fused_weights[m, n] = topk_weights[m, n]
        for s in range(S):
            fused_ids[m, K + s] = N + s
            fused_weights[m, K + s] = scale_factor
    """
    pid = tl.program_id(0)

    ids_row_ptr = pid * K
    w_row_ptr = pid * K
    out_ids_row_ptr = pid * (K + S)
    out_w_row_ptr = pid * (K + S)

    offs_k = tl.arange(0, K)
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k)
    ws = tl.load(topk_weights_ptr + w_row_ptr + offs_k)

    tl.store(out_ids_ptr + out_ids_row_ptr + offs_k, ids)
    tl.store(out_weights_ptr + out_w_row_ptr + offs_k, ws)

    offs_s = tl.arange(0, S)

    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)
    shared_ws = tl.full([S], scale_factor, dtype=ws.dtype)

    tl.store(out_ids_ptr + out_ids_row_ptr + K + offs_s, shared_ids)
    tl.store(out_weights_ptr + out_w_row_ptr + K + offs_s, shared_ws)


def fused_append_shared_experts(
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N=None
):
    assert N is not None, "N (shared expert base id) must be provided"
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_weights = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )

    _fused_append_shared_experts_kernel[(m,)](
        topk_ids,
        topk_weights,
        out_ids,
        out_weights,
        N_BASE=N,
        scale_factor=scale_factor,
        K=k,
        S=s,
        num_warps=1,
    )
    return out_ids, out_weights


@triton.jit
def _fused_append_remap_shared_experts_deepep_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    shared_id_base,  # runtime scalar: ep_rank * num_local_experts + num_local_routed
    num_local_routed,  # runtime scalar: routed experts per rank (for gap-insertion)
    scale_factor,  # runtime scalar: shared-expert weight
    K: tl.constexpr,
    S: tl.constexpr,
):
    """Append shared experts AND apply the DeepEP interleaved remap in one pass.

    Equivalent to fused_append_shared_experts() immediately followed by
    topk._remap_topk_for_deepep(), but the remap math runs on the rows already
    loaded into registers, so it costs a few ALU ops instead of ~6 extra eager
    kernel launches (div_floor / add / arange / fill / copy) per MoE layer.

    Routed IDs:   e -> e + e // num_local_routed   (insert gaps for shared slots)
    Shared IDs:   shared_id_base + arange(S)        (one id per shared slot)
    Shared wgt:   scale_factor                     (1.0 on aiter; 1/rsf otherwise)
    """
    pid = tl.program_id(0)

    ids_row_ptr = pid * K
    out_ids_row_ptr = pid * (K + S)

    offs_k = tl.arange(0, K)
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k)
    ws = tl.load(topk_weights_ptr + ids_row_ptr + offs_k)

    # DeepEP interleaved layout: shift each routed id past the shared slots that
    # precede it. Matches `routed + routed // num_local_routed` exactly.
    ids = ids + ids // num_local_routed

    tl.store(out_ids_ptr + out_ids_row_ptr + offs_k, ids)
    tl.store(out_weights_ptr + out_ids_row_ptr + offs_k, ws)

    offs_s = tl.arange(0, S)
    shared_ids = tl.cast(shared_id_base + offs_s, ids.dtype)
    shared_ws = tl.full([S], scale_factor, dtype=ws.dtype)

    tl.store(out_ids_ptr + out_ids_row_ptr + K + offs_s, shared_ids)
    tl.store(out_weights_ptr + out_ids_row_ptr + K + offs_s, shared_ws)


def fused_append_remap_shared_experts_deepep(
    topk_ids,
    topk_weights,
    num_fused_shared_experts,
    scale_factor,
    shared_id_base,
    num_local_routed,
):
    """Fused append + DeepEP remap (see kernel docstring).

    Replaces the fused_append_shared_experts() + _remap_topk_for_deepep() pair on
    the aiter/DeepEP-class path. Host computes the scalar remap params so the
    kernel stays branch-free.
    """
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_weights = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )

    _fused_append_remap_shared_experts_deepep_kernel[(m,)](
        topk_ids,
        topk_weights,
        out_ids,
        out_weights,
        shared_id_base,
        num_local_routed,
        scale_factor,
        K=k,
        S=s,
        num_warps=1,
    )
    return out_ids, out_weights


@triton.jit
def _fused_append_shared_experts_with_weights_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    shared_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    N_BASE,
    K: tl.constexpr,
    S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)

    ids_row_ptr = pid * K
    out_row_ptr = pid * (K + S)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k, mask=mask_k)
    ws = tl.load(topk_weights_ptr + ids_row_ptr + offs_k, mask=mask_k)

    tl.store(out_ids_ptr + out_row_ptr + offs_k, ids, mask=mask_k)
    tl.store(out_weights_ptr + out_row_ptr + offs_k, ws, mask=mask_k)

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)
    shared_ws = tl.load(shared_weights_ptr + pid * S + offs_s, mask=mask_s)

    tl.store(out_ids_ptr + out_row_ptr + K + offs_s, shared_ids, mask=mask_s)
    tl.store(out_weights_ptr + out_row_ptr + K + offs_s, shared_ws, mask=mask_s)


def fused_append_shared_experts_with_weights(
    topk_ids, topk_weights, shared_weights, num_fused_shared_experts, N=None
):
    """Like fused_append_shared_experts but accepts per-token shared weights tensor."""
    assert N is not None, "N (shared expert base id) must be provided"
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    shared_weights_2d = shared_weights.to(topk_weights.dtype)
    if shared_weights_2d.ndim == 1:
        shared_weights_2d = shared_weights_2d.unsqueeze(-1)
    if shared_weights_2d.shape[1] < s:
        shared_weights_2d = shared_weights_2d.expand(m, s)
    shared_weights_2d = shared_weights_2d.contiguous()

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_weights = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )

    block_k = triton.next_power_of_2(k)
    block_s = triton.next_power_of_2(s)

    _fused_append_shared_experts_with_weights_kernel[(m,)](
        topk_ids,
        topk_weights,
        shared_weights_2d,
        out_ids,
        out_weights,
        N_BASE=N,
        K=k,
        S=s,
        BLOCK_K=block_k,
        BLOCK_S=block_s,
        num_warps=1,
    )
    return out_ids, out_weights
