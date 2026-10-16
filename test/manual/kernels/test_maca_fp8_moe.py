"""Compare MACA FP8 expert GEMM with Triton, including output-view fallbacks.

Run in the MACA container with ENABLE_MCTLASS_FUSED_MOE_PYTHON_API=1.
A noncontiguous output caught a silent write-to-copy regression in the backport.
"""

import unittest
from unittest.mock import patch

import torch
import triton.language as tl

from sglang.kernels.ops.moe import fused_moe_triton_kernels as kernels
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.utils.common import is_maca
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(is_maca(), "Requires MACA and mctlassEx")
class TestMacaFp8Moe(CustomTestCase):
    def test_fp8_output_matches_triton(self):
        torch.manual_seed(27)
        m, experts, n, k, topk = 5, 4, 256, 256, 2
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(experts, n, k, device="cuda").to(torch.float8_e4m3fn)
        routes = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], device="cuda")
        weights = torch.full((m, topk), 0.5, device="cuda")
        for mode in ("block", "tensor", "strided", "bias", "rectangular"):
            with self.subTest(mode=mode):
                block = (
                    None
                    if mode == "tensor"
                    else [64 if mode == "rectangular" else 128, 128]
                )
                scales = torch.full(
                    (experts, n // block[0], k // block[1])
                    if block
                    else (experts, n, 1),
                    0.1,
                    device="cuda",
                )
                bias = (
                    torch.randn(experts, n, device="cuda", dtype=torch.bfloat16)
                    if mode == "bias"
                    else None
                )
                tile = 32
                if block is not None:
                    tile = kernels.gemm.get_kernel_m(
                        torch.empty_like(a, dtype=torch.float8_e4m3fn),
                        b,
                        torch.empty(m * topk, n, device="cuda", dtype=torch.bfloat16),
                        experts,
                        m,
                        n,
                        k,
                        topk,
                        use_fp8=True,
                        is_blockwise=True,
                        group_size=128,
                    )
                self.assertGreater(tile, 0)
                sorted_ids, expert_ids, padded = moe_align_block_size(
                    routes, tile, experts
                )
                outputs = []
                for vendor in (False, True):
                    c = torch.zeros(
                        m,
                        topk,
                        n * (2 if mode == "strided" else 1),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    if mode == "strided":
                        c = c[..., ::2]
                    with patch.object(
                        kernels, "enable_mctlass_fused_moe_python_api", vendor
                    ):
                        kernels.invoke_fused_moe_kernel(
                            a,
                            b,
                            bias,
                            c,
                            None,
                            scales,
                            None,
                            weights,
                            routes,
                            sorted_ids,
                            expert_ids,
                            padded,
                            True,
                            topk,
                            {
                                "BLOCK_SIZE_M": tile,
                                "BLOCK_SIZE_N": 64,
                                "BLOCK_SIZE_K": 128,
                                "GROUP_SIZE_M": 1,
                            },
                            tl.bfloat16,
                            True,
                            False,
                            False,
                            False,
                            False,
                            True,
                            block_shape=block,
                        )
                    outputs.append(c.clone())
                torch.cuda.synchronize()
                self.assertTrue(torch.isfinite(outputs[1]).all().item())
                torch.testing.assert_close(outputs[1], outputs[0], atol=0.01, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
