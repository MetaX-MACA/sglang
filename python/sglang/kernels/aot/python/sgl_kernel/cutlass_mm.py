from typing import Optional

import torch
from sgl_kernel.platform import is_maca


def _scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    maca = is_maca()
    if maca and out_dtype != torch.bfloat16:
        raise NotImplementedError(
            "MACA W8A8 GEMM requires BF16 output: mctlassEx does not support FP16 output"
        )

    out = torch.empty((*a.shape[:-1], b.shape[-1]), device=a.device, dtype=out_dtype)
    if not maca:
        torch.ops.sgl_kernel.cutlass_scaled_mm.default(
            out, a, b, scale_a, scale_b, bias
        )
        return out

    scale_a_shape = (*a.shape[:-1], 1)
    scale_b_shape = (*b.shape[:-2], 1, b.shape[-1])
    scale_a = (
        scale_a.reshape(()).expand(scale_a_shape)
        if scale_a.numel() == 1
        else scale_a.reshape(scale_a_shape)
    ).contiguous()
    scale_b = (
        scale_b.reshape(()).expand(scale_b_shape)
        if scale_b.numel() == 1
        else scale_b.reshape(scale_b_shape)
    ).contiguous()
    if bias is not None:
        bias = bias.reshape(scale_b_shape).contiguous()

    if out.numel() == 0:
        return out
    if a.ndim == 2:
        from mctlassEx import mctlassExHandleWrapper

        mctlassExHandleWrapper().mctlass_w8a8_scaled_mm_azp(
            a,
            b,
            out,
            scale_a,
            scale_b,
            bias,
            None,
            None,
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    from mctlassEx import ScaledGEMM

    gemm = ScaledGEMM()
    shape = [a.shape[-2], b.shape[-1], a.shape[-1]]
    if a.ndim == 3 and bias is not None:
        # Current mctlassEx batch GEMM shares bias across groups.
        for i in range(a.shape[0]):
            gemm(shape, a[i], b[i], out[i], scale_a[i], scale_b[i], bias[i])
    else:
        gemm(shape, a, b, out, scale_a, scale_b, bias)
    return out


def cutlass_scaled_batch_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched INT8 GEMM with activation and weight scales."""
    assert a.ndim == b.ndim == 3
    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]
    assert out_dtype in (torch.bfloat16, torch.float16)
    return _scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """INT8 GEMM with per-token activation and per-channel weight scales."""
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
    assert out_dtype in (torch.bfloat16, torch.float16)
    assert bias is None or (bias.numel() == b.shape[1] and bias.dtype == out_dtype)
    return _scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
