import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _IdleBackend:
    def __init__(self, mtp_enabled, metadata):
        self.mtp_enabled = mtp_enabled
        self.forward_metadata = metadata

    @property
    def token_to_kv_pool(self):
        raise AssertionError("Idle attention must not access the KV pool")


class TestDeepseekV4Idle(unittest.TestCase):
    _FORWARDS = (
        DeepseekV4AttnBackend._forward_attention,
        DeepseekV4HipRadixBackend.forward,
    )

    def test_idle_skips_metadata_and_cache_with_or_without_mtp(self):
        for forward, mtp_enabled, num_tokens, metadata in product(
            self._FORWARDS, (False, True), (0, 8), (None, object())
        ):
            with self.subTest(
                backend=forward.__qualname__,
                mtp_enabled=mtp_enabled,
                num_tokens=num_tokens,
                metadata=metadata,
            ):
                q = torch.empty((num_tokens, 16, 64), dtype=torch.bfloat16)
                result = forward(
                    _IdleBackend(mtp_enabled, metadata),
                    q=q,
                    k=None,
                    v=None,
                    layer=SimpleNamespace(v_head_dim=32),
                    forward_batch=SimpleNamespace(forward_mode=ForwardMode.IDLE),
                    compress_ratio=4,
                )

                self.assertEqual(result.shape, (num_tokens, 16, 32))
                self.assertEqual(result.dtype, q.dtype)
                self.assertEqual(result.device, q.device)

    def test_active_modes_still_validate_shared_kv(self):
        for forward, mode in product(
            self._FORWARDS,
            (ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY),
        ):
            with self.subTest(backend=forward.__qualname__, mode=mode):
                with self.assertRaisesRegex(AssertionError, "shares k and v"):
                    forward(
                        _IdleBackend(False, None),
                        q=torch.empty((1, 16, 64)),
                        k=object(),
                        v=object(),
                        layer=SimpleNamespace(v_head_dim=32),
                        forward_batch=SimpleNamespace(forward_mode=mode),
                        compress_ratio=4,
                    )


if __name__ == "__main__":
    unittest.main()
