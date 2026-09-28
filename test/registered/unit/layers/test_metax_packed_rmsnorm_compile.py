"""Regression tests for the MCOP packed RMSNorm torch.compile boundary."""

import pytest
import torch

if "+metax" not in torch.__version__ or not torch.cuda.is_available():
    pytest.skip("Requires the MetaX MCOP CUDA implementation", allow_module_level=True)

pytest.importorskip("mcoplib")

from sglang.srt.layers.layernorm import (
    fused_add_rms_norm_dynamic_per_token_quant_padding_output,
)
from sglang.srt.utils.common import align_packed_tensor_size
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("shape", [(1, 256), (7, 512)])
@pytest.mark.parametrize("has_residual", [False, True])
def test_packed_rmsnorm_operator_contract(shape, has_residual):
    torch.manual_seed(519)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if has_residual else None
    weight = torch.ones(shape[-1], device="cuda", dtype=x.dtype)
    pad_size = align_packed_tensor_size(shape[-1])
    torch.library.opcheck(
        torch.ops.sglang.fused_add_rms_norm_dynamic_per_token_quant_padding_output.default,
        (x, residual, weight, pad_size, 1e-6),
    )


@pytest.mark.parametrize("has_residual", [False, True])
def test_packed_rmsnorm_aot_preserves_values_and_residual_mutation(has_residual):
    torch.manual_seed(517)
    x = torch.randn((3, 512), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if has_residual else None
    weight = torch.ones(512, device="cuda", dtype=x.dtype)
    pad_size = align_packed_tensor_size(x.shape[-1])

    def forward(x, residual, weight):
        packed, residual_out, norm, quant, scales = (
            fused_add_rms_norm_dynamic_per_token_quant_padding_output(
                x, residual, weight, pad_size, 1e-6
            )
        )
        # Packed bytes can encode BF16 NaNs; compare their actual byte payload.
        return packed.view(torch.uint8), residual_out, norm, quant, scales

    eager_residual = residual.clone() if has_residual else None
    compiled_residual = residual.clone() if has_residual else None
    expected = forward(x.clone(), eager_residual, weight)
    actual = torch.compile(forward, backend="aot_eager", fullgraph=True)(
        x.clone(), compiled_residual, weight
    )
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    if has_residual:
        assert expected[1] is eager_residual
        assert actual[1] is compiled_residual
        torch.testing.assert_close(compiled_residual, eager_residual, rtol=0, atol=0)
    packed, _, norm, quant, scales = actual
    torch.testing.assert_close(packed[..., :512].view(torch.int8), quant, rtol=0, atol=0)
    torch.testing.assert_close(
        packed[..., 512:516].contiguous().view(torch.float32), scales, rtol=0, atol=0
    )
    assert torch.count_nonzero(packed[..., 516:]) == 0
