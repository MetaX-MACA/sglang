import sys

import pytest
import torch

if "metax" in torch.__version__:
    from sgl_kernel import cutlass_scaled_mm as int8_scaled_mm
else:
    from sgl_kernel import int8_scaled_mm
from utils import is_sm10x


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if bias is not None:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1) + bias
    else:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1)
    return o.to(out_dtype)


def _test_accuracy_once(M, N, K, with_bias, out_dtype, device):
    a = to_int8(torch.randn((M, K), device=device) * 5)
    b = to_int8(torch.randn((N, K), device=device).t() * 5)
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    if with_bias:
        bias = torch.randn((N,), device="cuda", dtype=out_dtype) * 10
    else:
        bias = None
    o = int8_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    o1 = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    torch.testing.assert_close(o, o1)


@pytest.mark.skipif(
    is_sm10x(),
    reason="int8_scaled_mm is only supported on sm90 and lower",
)
@pytest.mark.parametrize("M", [1, 16, 32, 64, 128, 512, 1024, 4096, 8192])
@pytest.mark.parametrize("N", [16, 128, 512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_accuracy(M, N, K, with_bias, out_dtype):
    if "metax" in torch.__version__ and out_dtype == torch.float16:
        pytest.skip("mctlassEx W8A8 GEMM supports BF16 output only")
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "cuda")


@pytest.mark.skipif("metax" not in torch.__version__, reason="MACA GEMM paths")
@pytest.mark.parametrize("shape", [(7, 128, 256), (7, 256, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_maca_rectangular_gemm_paths(shape, with_bias):
    from sgl_kernel import cutlass_mm

    out_dtype = torch.bfloat16
    m, n, k = shape
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda").t()
    scale_a = torch.rand((m, 1), device="cuda") / 127
    scale_b = torch.rand((n, 1), device="cuda") / 127
    bias = torch.randn(n, dtype=out_dtype, device="cuda") if with_bias else None
    out = cutlass_mm.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    expected = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif("metax" not in torch.__version__, reason="MACA GEMM paths")
@pytest.mark.parametrize("static_scales", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_maca_batched_gemm(static_scales, with_bias):
    from sgl_kernel import cutlass_scaled_batch_mm

    out_dtype = torch.bfloat16
    torch.manual_seed(0)
    g, m, n, k = 2, 7, 128, 256
    a = torch.randint(-127, 128, (g, m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(
        -127, 128, (g, n, k), dtype=torch.int8, device="cuda"
    ).transpose(1, 2)
    scale_a = torch.rand(1 if static_scales else g * m, device="cuda") / 127
    scale_b = torch.rand(1 if static_scales else g * n, device="cuda") / 127
    bias = torch.randn(g, n, dtype=out_dtype, device="cuda") if with_bias else None
    out = cutlass_scaled_batch_mm(a, b, scale_a, scale_b, out_dtype, bias)
    expected = torch.matmul(a.float(), b.float())
    if static_scales:
        expected = expected * scale_a * scale_b
    else:
        expected = expected * scale_a.reshape(g, m, 1) * scale_b.reshape(g, 1, n)
    if bias is not None:
        expected = expected + bias.unsqueeze(1)
    torch.testing.assert_close(out, expected.to(out_dtype), rtol=0.02, atol=0.02)


@pytest.mark.skipif("metax" not in torch.__version__, reason="MACA GEMM paths")
@pytest.mark.parametrize("batched", [False, True])
def test_maca_fp16_gemm_rejected(batched):
    from sgl_kernel import cutlass_scaled_batch_mm, cutlass_scaled_mm

    shape_a = (2, 7, 256) if batched else (7, 256)
    shape_b = (2, 256, 128) if batched else (256, 128)
    a = torch.empty(shape_a, dtype=torch.int8)
    b = torch.empty(shape_b, dtype=torch.int8)
    scale = torch.ones(1, dtype=torch.float32)
    gemm = cutlass_scaled_batch_mm if batched else cutlass_scaled_mm
    with pytest.raises(NotImplementedError, match="BF16 output"):
        gemm(a, b, scale, scale, torch.float16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
