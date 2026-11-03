"""MACA W8A8 low-latency expert runner against a BF16 mathematical reference."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import is_maca
from sglang.test.test_utils import CustomTestCase


class TestMacaW8A8DeepEP(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not is_maca():
            self.skipTest("Requires the MACA vendor INT8 kernels")
        self.config_scope = get_context().override_server_args(model_path="dummy")
        self.config_scope.install()
        self.addCleanup(self.config_scope.restore)
        parallel = get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0)
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)
        torch.manual_seed(27)

    def test_w8a8_masked_experts(self):
        experts, rows, hidden, intermediate = 8, 64, 256, 256
        x = (
            torch.randn(experts, rows, hidden, device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        original = x.clone()
        w13 = torch.randint(
            -32,
            32,
            (experts, 2 * intermediate, hidden),
            device="cuda",
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -32, 32, (experts, hidden, intermediate), device="cuda", dtype=torch.int8
        )
        s13 = torch.full((experts, 2 * intermediate, 1), 0.002, device="cuda")
        s2 = torch.full((experts, hidden, 1), 0.002, device="cuda")
        counts = torch.tensor(
            [3, 0, 7, 1, 5, 2, 8, 4], device="cuda", dtype=torch.int32
        )
        runner = DeepGemmRunnerCore(MoeRunnerConfig(activation="silu"))
        quant = DeepGemmMoeQuantInfo(
            w13, w2, use_fp8=False, use_int8_w8a8=True, w13_scale=s13, w2_scale=s2
        )
        result = runner.run(
            DeepGemmRunnerInput(x, None, True, masked_m=counts, expected_m=8), quant, {}
        ).hidden_states
        for expert, count in enumerate(counts.tolist()):
            if not count:
                continue
            gateup = F.linear(
                original[expert, :count].float(), w13[expert].float() * s13[expert]
            )
            gate, up = gateup.chunk(2, -1)
            expected = F.linear(F.silu(gate) * up, w2[expert].float() * s2[expert])
            torch.testing.assert_close(
                result[expert, :count].float(), expected, rtol=0.06, atol=0.002
            )


if __name__ == "__main__":
    unittest.main()
