"""Exercise AOT operators without FlashInfer's private PyTorch helper header."""

import pytest
import torch
from sgl_kernel import merge_state_v2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [32, 64, 128])
def test_merge_state_matches_torch(dtype, head_size):
    torch.manual_seed(0)
    shape = (3, 2, head_size)
    a = torch.randn(shape, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    sa = torch.randn(shape[:2], device="cuda")
    sb = torch.randn_like(sa)

    output, lse = merge_state_v2(a, sa, b, sb)
    expected_lse = torch.logaddexp(sa, sb)
    expected = (
        a.float() * torch.exp(sa - expected_lse).unsqueeze(-1)
        + b.float() * torch.exp(sb - expected_lse).unsqueeze(-1)
    ).to(dtype)
    tolerance = {
        torch.float32: (1e-5, 1e-6),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]
    torch.testing.assert_close(output, expected, rtol=tolerance[0], atol=tolerance[1])
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)


def test_merge_state_rejects_mismatched_shapes():
    a = torch.ones((3, 2, 32), device="cuda")
    b = torch.ones((3, 2, 64), device="cuda")
    lse = torch.zeros((3, 2), device="cuda")
    with pytest.raises(RuntimeError, match="same shape"):
        merge_state_v2(a, lse, b, lse)


def test_merge_state_rejects_noncontiguous_input():
    a = torch.ones((3, 2, 64), device="cuda")[:, :, ::2]
    b = torch.ones_like(a)
    lse = torch.zeros((3, 2), device="cuda")
    with pytest.raises(RuntimeError, match="contiguous"):
        merge_state_v2(a, lse, b, lse)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_norm_exports_work_with_maca_subset(monkeypatch, dtype, gemma, force_fallback):
    """The serving layer imports all norm APIs, even with the MACA subset build."""
    import sgl_kernel
    from sgl_kernel import elementwise

    if force_fallback:
        monkeypatch.setattr(elementwise, "_has_flashinfer", False)
    x = torch.randn((3, 128), device="cuda", dtype=dtype)
    weight = torch.randn((128,), device="cuda", dtype=dtype)
    scale = weight.float() + 1 if gemma else weight.float()
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * scale
    ).to(dtype)
    out = torch.empty_like(x)
    op = sgl_kernel.gemma_rmsnorm if gemma else sgl_kernel.rmsnorm
    assert op(x, weight, out=out) is out
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_gemma_fused_norm_updates_both_outputs(monkeypatch, dtype, force_fallback):
    import sgl_kernel
    from sgl_kernel import elementwise

    if force_fallback:
        monkeypatch.setattr(elementwise, "_has_flashinfer", False)
    x = torch.randn((3, 128), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn((128,), device="cuda", dtype=dtype)
    acc = x.float() + residual.float()
    expected = (
        acc
        * torch.rsqrt(acc.square().mean(-1, keepdim=True) + 1e-6)
        * (weight.float() + 1)
    ).to(dtype)
    sgl_kernel.gemma_fused_add_rmsnorm(x, residual, weight)
    torch.testing.assert_close(residual, acc.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(x, expected, rtol=1e-2, atol=1e-2)
