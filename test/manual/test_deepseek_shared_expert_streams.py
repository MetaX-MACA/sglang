"""MACA numerical regression for shared/routed expert input aliasing.

Run with ENABLE_MCTLASS_FUSED_MOE=0 ENABLE_MCTLASS_FUSED_MOE_PYTHON_API=0
and CUDA_VISIBLE_DEVICES/MACA_VISIBLE_DEVICES restricted to one idle die.
The Triton runner exercises the inplace output contract of routed experts.
"""

import socket
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_maca
from sglang.test.test_utils import CustomTestCase, publish_build_topology


@unittest.skipUnless(is_maca() and torch.cuda.is_available(), "Requires a MACA GPU")
class TestDeepseekSharedExpertStreams(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.layers.moe.utils import initialize_moe_config

        cls.original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        publish_build_topology(
            tp_size=1,
            disable_shared_experts_fusion=True,
            moe_runner_backend="triton",
        )
        initialize_model_parallel()
        initialize_moe_config()

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from sglang.srt.runtime_context import reset_context

        try:
            destroy_model_parallel()
            destroy_distributed_environment()
            reset_context()
        finally:
            if hasattr(cls, "original_dtype"):
                torch.set_default_dtype(cls.original_dtype)
            super().tearDownClass()

    def _build(self, enabled):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        config = SimpleNamespace(
            routed_scaling_factor=1.0,
            n_shared_experts=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            hidden_size=128,
            moe_intermediate_size=128,
            hidden_act="silu",
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            scoring_func="softmax",
            topk_method="greedy",
        )
        with (
            torch.device("cuda"),
            envs.SGLANG_ENABLE_SHARED_EXPERTS_MULTI_STREAM.override(enabled),
        ):
            torch.manual_seed(17)
            # The draft path also lets the direct DeepEP scheduling test below
            # run without constructing an unrelated EPLB metadata registry.
            model = DeepseekV2MoE(
                config,
                layer_id=0,
                alt_stream=torch.cuda.Stream(),
                is_nextn=True,
                reduce_results=False,
            )
            for parameter in model.parameters():
                parameter.data.normal_(0, 0.03)
        return model

    @torch.inference_mode()
    def _check_forward(self, enabled, capture):
        from sglang.srt.model_executor.runner import model_capture_mode

        model = self._build(enabled)
        x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
        original = x.clone()
        expected = model.forward_normal(x.clone())
        main_stream = [torch.cuda.current_stream()]

        def shared_after_routed(_module, _args):
            # Force the dangerous ordering deterministically: shared GEMMs read
            # only after routed output has been written. Both kernels are real.
            model.alt_stream.wait_stream(main_stream[0])

        handle = model.shared_experts.register_forward_pre_hook(shared_after_routed)

        def run():
            main_stream[0] = torch.cuda.current_stream()
            return model(x)

        try:
            # A runtime env change must not desync the constructor's inplace
            # decision from the forward's stream decision.
            with (
                envs.SGLANG_ENABLE_SHARED_EXPERTS_MULTI_STREAM.override(not enabled),
                model_capture_mode(),
            ):
                actual = run()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
            if enabled:
                torch.testing.assert_close(x, original, rtol=0, atol=0)
            if capture:
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup), model_capture_mode():
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), model_capture_mode():
                    captured = run()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, expected, rtol=0.02, atol=0.002)
                torch.testing.assert_close(x, original, rtol=0, atol=0)
        finally:
            handle.remove()

    def test_disabled_uses_serial_shared_input(self):
        """Capture mode must not silently opt into the unsafe stream path."""
        self._check_forward(enabled=False, capture=False)

    def test_enabled_preserves_shared_input_and_graph_replay(self):
        """Routed inplace output corrupted the later shared expert read."""
        self._check_forward(enabled=True, capture=True)

    @torch.inference_mode()
    def test_deepep_event_lifecycle(self):
        """The shared completion event exists only when its stream was launched."""
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: None)
        for enabled in (False, True):
            model = self._build(enabled)
            for rows in (0, 7):
                with self.subTest(enabled=enabled, rows=rows):
                    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
                    expected = model.forward_normal(x.clone())
                    # Run the real single-rank dispatcher and kernels through
                    # DeepEP's shared-expert scheduling, without network traffic.
                    actual = model.forward_deepep(x.clone(), batch)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


if __name__ == "__main__":
    unittest.main()
