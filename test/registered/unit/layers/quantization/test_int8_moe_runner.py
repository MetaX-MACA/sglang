import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.int8_utils import get_int8_moe_runner_backend
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestInt8MoeRunner(CustomTestCase):
    def test_w8a8_checkpoint_formats_select_modern_deepep(self):
        """The compressed-tensors CLI spelling differs from its config name."""

        def init_without_weights(layer, **kwargs):
            torch.nn.Module.__init__(layer)

        for config_type in (W8A8Int8Config, CompressedTensorsConfig):
            with self.subTest(config=config_type.__name__):
                # Only the public config identity is needed by this routing gate;
                # weight allocation and kernel loading are outside its contract.
                config = object.__new__(config_type)
                with (
                    get_flags().moe.override(
                        runner_backend=MoeRunnerBackend.DEEP_GEMM,
                        a2a_backend=MoeA2ABackend.DEEPEP,
                        deepep_mode=DeepEPMode.LOW_LATENCY,
                    ),
                    patch.object(FusedMoE, "__init__", init_without_weights),
                    patch(
                        "sglang.srt.layers.moe.ep_moe.layer.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM",
                        True,
                    ),
                ):
                    layer = DeepEPMoE(
                        num_experts=2,
                        top_k=1,
                        hidden_size=8,
                        intermediate_size=8,
                        layer_id=0,
                        quant_config=config,
                    )
                self.assertTrue(layer.deprecate_flag)

    def test_default_and_triton_keep_the_existing_path(self):
        for backend in (MoeRunnerBackend.AUTO, MoeRunnerBackend.TRITON):
            with self.subTest(backend=backend):
                with get_flags().moe.override(runner_backend=backend):
                    self.assertEqual(
                        get_int8_moe_runner_backend(), MoeRunnerBackend.TRITON
                    )

    def test_deepgemm_requires_maca_low_latency(self):
        for maca, a2a, mode in (
            (True, MoeA2ABackend.DEEPEP, DeepEPMode.LOW_LATENCY),
            (False, MoeA2ABackend.DEEPEP, DeepEPMode.LOW_LATENCY),
            (True, MoeA2ABackend.NONE, DeepEPMode.LOW_LATENCY),
            (True, MoeA2ABackend.DEEPEP, DeepEPMode.NORMAL),
        ):
            with self.subTest(maca=maca, a2a=a2a, mode=mode):
                with (
                    get_flags().moe.override(
                        runner_backend=MoeRunnerBackend.DEEP_GEMM,
                        a2a_backend=a2a,
                        deepep_mode=mode,
                    ),
                    patch("sglang.srt.utils.is_maca", return_value=maca),
                ):
                    if maca and a2a.is_deepep() and mode.is_low_latency():
                        self.assertEqual(
                            get_int8_moe_runner_backend(), MoeRunnerBackend.DEEP_GEMM
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, "low_latency"):
                            get_int8_moe_runner_backend()


if __name__ == "__main__":
    unittest.main()
