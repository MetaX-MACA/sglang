# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import logging
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_NEED_TMA_ALIGNED_SCALES,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

#if ENABLE_JIT_DEEPGEMM:
#    import deep_gemm
#    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401
import deep_gemm

_SANITY_CHECK = envs.SGLANG_DEEPGEMM_SANITY_CHECK.get()

import importlib
def try_import_with_message(name: str):
    try:
        module = importlib.import_module(name)
        return module
    except ImportError:
        print(f"can't import {name}, please use 'pip install {name}'")
        return None

if try_import_with_message("mctlassEx") != None :
    backend_name = "mctlassEx" 
    from deep_gemm import get_mctlass_op
    mctlass_op = get_mctlass_op()

else:
    backend_name = "torch_bf16"

# if try_import_with_message("mctlassEx") != None :
#     backends.append("mctlassEx")


# TODO maybe rename these functions
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    # kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    # lhs = _ensure_cuda(lhs)
    # rhs = _ensure_cuda(rhs)

    # with compile_utils.deep_gemm_execution_hook(
    #     expected_m, n, k, num_groups, kernel_type
    # ):
    #     with configure_deep_gemm_num_sms(
    #         overlap_args.num_sms if overlap_args is not None else None
    #     ):

    #         fp4_kwargs = {}
    #         if recipe_a is not None:
    #             fp4_kwargs["recipe_a"] = recipe_a
    #         if recipe_b is not None:
    #             fp4_kwargs["recipe_b"] = recipe_b

    return deep_gemm.fp8_m_grouped_gemm_nt_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
        backend=backend_name,
        # **fp4_kwargs,
        # **(
        #     dict(
        #         enable_overlap=True,
        #         max_block_n=max_block_n,
        #         signal=overlap_args.signal,
        #     )
        #     if overlap_args is not None
        #     else {}
        # ),
    )

def append_scale_batch_cuda(input_tensor: torch.Tensor, scale_tensor: torch.Tensor) -> torch.Tensor:
    """
    pack scale in float32 to input in int8

    args:
        input_tensor (torch.Tensor): BxMxK,int8
        scale_tensor (torch.Tensor): BxMx1,float32

    return:
        torch.Tensor: BxMx(K+4), int8
    """
    assert input_tensor.dtype == torch.int8, "input_tensor must be int8"
    assert scale_tensor.dtype == torch.float32, "scale_tensor must be float32"
    assert input_tensor.shape[:2] == scale_tensor.shape[:2], "Shape mismatch on B and M"
    assert scale_tensor.shape[2] == 1, "scale_tensor must be of shape BxMx1"

    B, M, K = input_tensor.shape

    # float32 -> 4*int8
    # scale_flat = scale_tensor.contiguous().view(-1)  # (B*M,)
    # scale_bytes = scale_flat.cpu().numpy().tobytes()  # 4×B×M bytes
    # scale_bytes_tensor = torch.tensor(list(scale_bytes), dtype=torch.uint8, device=input_tensor.device)
    # scale_bytes_tensor = scale_bytes_tensor.view(B, M, 4).to(torch.int8)
    scale_bytes_tensor = scale_tensor.view(torch.int8)

    # pack scale to input
    output = torch.cat([input_tensor, scale_bytes_tensor], dim=-1)  # B×M×(K+4)
    return output

def grouped_gemm_nt_int8int8bf16_masked(
    lhs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 128,
    clamp_limit: Optional[float] = None,
):  
    if not isinstance(lhs, tuple):
        lhs = (lhs.view(torch.int8), None)
    if clamp_limit is not None:
        import mctlassEx
        lhs_data, lhs_scales = lhs
        rhs_data, rhs_scales = rhs

        @lru_cache(maxsize=1)
        def get_mctlass_op():
            """Get mctlassEx operation handle (singleton)"""
            return mctlassEx.mctlassExHandleWrapper()

        mctlass_op = get_mctlass_op()
        stream = torch.cuda.current_stream().cuda_stream

        mctlass_op.mctlass_masked_group_gemm(
            lhs_data,
            rhs_data,
            out,
            lhs_scales,
            rhs_scales,
            masked_m,
            stream
        )
        return None

    if overlap_args is None:
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
        )
        return None
    else:
        import mctlassEx
        from mctlassEx import MaskedGroupedGEMM
        assert isinstance(lhs, tuple), "lhs must be a tuple"
        assert isinstance(rhs, tuple), "rhs must be a tuple"
        num_groups, m, _ = lhs[0].shape
        _, n, k = rhs[0].shape
        MaskedGroupedGEMM()([m, n, k],
                lhs[0],
                rhs[0],
                out,
                masked_m,
                A_packed = True,
                B_packed = False,
                scale_b = rhs[1],
                signal = overlap_args.signal,
                single_batch_overlap = True,
                num_blocks=overlap_args.num_sms)
        kernel_n = 128
        kernel_m = 128
        threshold = (n + kernel_n - 1) // kernel_n
        # logger.info(f"kernel_m: {kernel_m}, threshold: {threshold}")
        return kernel_m, threshold

def _ensure_cuda(
    pair: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        pair[0].cuda() if not pair[0].is_cuda else pair[0],
        pair[1].cuda() if not pair[1].is_cuda else pair[1],
    )


def grouped_gemm_nt_bf16_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = a.shape
    _, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return deep_gemm.m_grouped_bf16_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            expected_m,
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    if m == 0:
        return

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    fp4_kwargs = {}
    if recipe_a is not None:
        fp4_kwargs["recipe_a"] = recipe_a
    if recipe_b is not None:
        fp4_kwargs["recipe_b"] = recipe_b

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            lhs, rhs, out, m_indices, backend=backend_name, **fp4_kwargs,
        )


def grouped_gemm_nt_bf16_contig(
    a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, m_indices: torch.Tensor
):
    m, k = a.shape
    num_groups, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
            backend=backend_name,
        )


def gemm_nt_bf16bf16f32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
):
    m, k = lhs.shape
    n, _ = rhs.shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16F32

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.bf16_gemm_nt(lhs, rhs, out)


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    compile_utils.update_deep_gemm_config(gpu_id, server_args)


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        #deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            #deep_gemm.set_num_sms(original_num_sms)
            a = 3


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
