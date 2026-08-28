from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _make_name(*args):
    return "kimi_k3_" + "_".join(str(a) for a in args)


def _fast_math_flags() -> list[str]:
    # Mirrors sgl-kernel's CMake policy: fast-math on SM90, precise on
    # SM100+ (Blackwell needs bit-exact expf), off on HIP (clang rejects).
    if is_hip_runtime():
        return []
    if get_jit_cuda_arch().major >= 10:
        return []
    return ["--use_fast_math"]


@cache_once
def _jit_situ_and_mul_module(in_dtype: torch.dtype, out_dtype: torch.dtype) -> Module:
    """Compile and cache the JIT SiTU-and-mul module for an (in, out) dtype pair."""
    args = make_cpp_args(in_dtype, out_dtype, is_arch_support_pdl())
    return load_jit(
        _make_name("situ_and_mul"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulKernel<{args}>::run")],
        extra_cuda_cflags=_fast_math_flags(),
    )


def situ_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    """Fused SiTU (SoftCap-GLU) activation.

    gate_out = beta * tanh(gate / beta) * sigmoid(gate)
    up_out   = linear_beta * tanh(up / linear_beta)  [if linear_beta is not None]
    output   = gate_out * up_out

    Parameters
    ----------
    input       : bf16 or fp32 CUDA tensor [*, 2*D]
    out         : optional pre-allocated CUDA tensor [*, D]; its dtype selects
                  the output dtype (bf16 for an fp32 input)
    beta        : gate softcap scalar (e.g. 4.0)
    linear_beta : up softcap scalar (e.g. 25.0), or None to skip
    """
    hidden_size = input.shape[-1] // 2
    if out is None:
        out_dtype = torch.bfloat16 if input.dtype == torch.float32 else input.dtype
        out = input.new_empty(*input.shape[:-1], hidden_size, dtype=out_dtype)

    # 2D inputs may be row-strided (e.g. a slice of a fused-GEMM output);
    # higher-rank inputs keep the dense-view path.
    if input.dim() == 2 and input.stride(1) == 1:
        input_2d = input
    else:
        input_2d = input.contiguous().view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)

    has_linear_beta = linear_beta is not None
    module = _jit_situ_and_mul_module(input_2d.dtype, out_2d.dtype)
    module.run(
        input_2d,
        out_2d,
        float(beta),
        float(linear_beta) if has_linear_beta else 0.0,
        has_linear_beta,
    )
    return out


def masked_situ_quant_int8_pack(
    input: torch.Tensor,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    """Fuse masked SiTU, per-token INT8 quantization, and aligned packing.

    ``input`` has shape ``[E, M, 2D]``. The BF16 output is a byte carrier
    whose rows contain D INT8 values, one raw FP32 scale, then padding.
    """
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be BF16, got {input.dtype}")
    if input.dim() != 3 or input.shape[-1] % 2 != 0:
        raise ValueError(
            f"input must have shape [E, M, 2D], got {tuple(input.shape)}"
        )
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if masked_m.dtype != torch.int32:
        raise TypeError(f"masked_m must be INT32, got {masked_m.dtype}")
    if masked_m.shape != (input.shape[0],):
        raise ValueError(
            f"masked_m must have shape {(input.shape[0],)}, "
            f"got {tuple(masked_m.shape)}"
        )
    if not masked_m.is_contiguous():
        raise ValueError("masked_m must be contiguous")
    if masked_m.device != input.device:
        raise ValueError(
            "input and masked_m must be on one device, got "
            f"input={input.device}, masked_m={masked_m.device}"
        )

    hidden_size = input.shape[-1] // 2
    packed_size = (hidden_size // 2 + 257) // 256 * 256
    output = torch.empty(
        (*input.shape[:-1], packed_size),
        dtype=torch.bfloat16,
        device=input.device,
    )

    from mcoplib import op as mcoplib_op

    has_linear_beta = linear_beta is not None
    mcoplib_op.fused_situ_mul_dq_mask_quant_pack(
        output,
        input,
        masked_m,
        float(beta),
        float(linear_beta) if has_linear_beta else 0.0,
        int(has_linear_beta),
    )
    return output
