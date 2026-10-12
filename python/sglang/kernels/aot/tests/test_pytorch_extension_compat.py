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
