from typing import TYPE_CHECKING, Optional, Union
import torch
import os
enable_mctlass_fused_moe_python_api = (os.getenv("ENABLE_MCTLASS_FUSED_MOE_PYTHON_API", "0") == "1")

# Batch gemm in vllm, support w8a8 int8 quantization
def cutlass_scaled_batch_mm(a: torch.Tensor, b: torch.Tensor,
                            scale_a: torch.Tensor, scale_b: torch.Tensor,
                            out_dtype: torch.dtype, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert (a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1])
    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), device = a.device, dtype = out_dtype)
    torch.ops.sgl_kernel_custom.cutlass_scaled_mm.default(out, a, b, scale_a, scale_b, bias)
    return out

def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]

    cutlass_compatible_b = (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    # if current_platform.is_rocm() or not cutlass_compatible_b:
    #     from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
    #         triton_scaled_mm)
    #     return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    if enable_mctlass_fused_moe_python_api:
        import mctlassEx
        stream_ptr = torch.cuda.current_stream().cuda_stream
        mctlass_op = mctlassEx.mctlassExHandleWrapper()
        if bias is not None and bias.ndim == 1:
            bias = bias.unsqueeze(0)
        mctlass_op.mctlass_w8a8_scaled_mm_azp(a, b, out, scale_a, scale_b.T, bias, None,None, stream_ptr)
    else:
        torch.ops.sgl_kernel_custom.cutlass_scaled_mm.default(out, a, b, scale_a, scale_b, bias)

    return out


def cutlass_scaled_mm_azp(a: torch.Tensor,
                          b: torch.Tensor,
                          scale_a: torch.Tensor,
                          scale_b: torch.Tensor,
                          out_dtype: torch.dtype,
                          azp_adj: torch.Tensor,
                          azp: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.numel(
    ) == b.shape[1] and bias.dtype == out_dtype
    assert azp is None or azp.numel() == a.shape[0]

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops.sgl_kernel_custom.cutlass_scaled_mm_azp.default(out, a, b, scale_a, scale_b, azp_adj,
                                                        azp, bias)
    return out