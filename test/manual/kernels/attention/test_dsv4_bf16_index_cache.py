"""MACA V4 BF16 index cache: compressed writes, Q transform and paged logits."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    compress_norm_rope_store,
)
from sglang.kernels.ops.attention.dsv4.elementwise import (
    fused_q_indexer_rope_hadamard_quant,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool
from sglang.srt.utils.common import is_maca
from sglang.test.test_utils import CustomTestCase


def hadamard(n, device):
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / n**0.5


def rope_hadamard(x, freqs):
    pairs = torch.view_as_complex(
        x[..., 64:].reshape(*x.shape[:-1], 32, 2).contiguous()
    )
    tail = torch.view_as_real(pairs * freqs).flatten(-2)
    return torch.cat((x[..., :64], tail), -1) @ hadamard(128, x.device)


@unittest.skipUnless(is_maca(), "BF16 V4 index-cache integration is MACA-specific")
class TestDSV4BF16IndexCache(CustomTestCase):
    def setUp(self):
        self.override = envs.SGLANG_DSV4_INDEX_CACHE_TYPE.override("bf16")
        self.override.__enter__()
        self.addCleanup(self.override.__exit__, None, None, None)
        torch.manual_seed(73)

    def make_pool(self, size):
        return DeepSeekV4IndexerPool(
            size, 64, torch.bfloat16, 128, 1, "cuda", False, use_fp4_indexer=False
        )

    def test_fused_compressed_k_write_and_batched_accessor(self):
        n = 67
        pool = self.make_pool(128)
        kv = torch.randn((n, 128), dtype=torch.bfloat16, device="cuda")
        weight = torch.randn((128,), dtype=torch.bfloat16, device="cuda")
        angles = torch.randn((n * 4, 32), device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        plan = torch.zeros((n, 4), dtype=torch.int32, device="cuda")
        plan[:, 0] = torch.arange(n, device="cuda") * 4 + 4
        loc = torch.arange(n, dtype=torch.int64, device="cuda").flip(0)
        compress_norm_rope_store(
            kv,
            CompressorDecodePlan(4, plan.view(torch.uint8)),
            norm_weight=weight,
            norm_eps=1e-6,
            freq_cis=freqs,
            out_loc=loc,
            kvcache=pool.get_index_k_with_scale_buffer(0),
            page_size=64,
        )
        normalized = (
            kv.float()
            * (kv.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * weight.float()
        )
        expected = rope_hadamard(
            normalized, freqs[torch.arange(n, device="cuda") * 4]
        ).bfloat16()
        actual = pool.get_index_k_with_scale_buffer(0).view(-1, 128)[loc]
        # Independent GEMM Hadamard sums in a different order; allow BF16 rounding.
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=0.016, rtol=0.004
        )
        standalone = torch.randn((n, 128), dtype=torch.bfloat16, device="cuda")
        pool.set_index_fused(
            0, torch.arange(n, device="cuda", dtype=torch.int32), standalone
        )
        gathered, scales = pool.get_index_k_scale_buffer(
            0,
            torch.tensor([n], device="cuda", dtype=torch.int32),
            torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
            n,
            n,
        )
        self.assertIsNone(scales)
        torch.testing.assert_close(gathered, standalone, atol=0, rtol=0)

    def test_bf16_q_and_production_paged_logits(self):
        import deep_gemm

        b, h, n = 2, 64, 2112
        pool = self.make_pool(n)
        keys = torch.randn((n, 128), dtype=torch.bfloat16, device="cuda")
        pool.set_index_fused(0, torch.arange(n, device="cuda", dtype=torch.int32), keys)
        query = torch.randn((b, h, 128), dtype=torch.bfloat16, device="cuda")
        gates = torch.rand((b, h), device="cuda").bfloat16()
        angles = torch.randn((b, 32), device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        weight_scale = (128 * h) ** -0.5
        q, w = fused_q_indexer_rope_hadamard_quant(
            query,
            gates,
            weight_scale,
            freqs,
            torch.arange(b, device="cuda", dtype=torch.int64),
            torch.bfloat16,
        )
        ref_q = rope_hadamard(query.float(), freqs[:, None, :]).bfloat16()
        torch.testing.assert_close(q.float(), ref_q.float(), atol=0.016, rtol=0.004)
        torch.testing.assert_close(
            w.squeeze(-1), gates.float() * weight_scale, atol=1e-8, rtol=1e-7
        )
        lengths = torch.tensor([[n], [2053]], device="cuda", dtype=torch.int32)
        table = torch.arange(n // 64, device="cuda", dtype=torch.int32).repeat(b, 1)
        meta = deep_gemm.get_paged_mqa_logits_metadata(
            lengths, 64, torch.cuda.get_device_properties(0).multi_processor_count
        )
        actual = deep_gemm.bf16_paged_mqa_logits(
            q.unsqueeze(1),
            pool.get_index_k_with_scale_buffer(0).view(-1, 64, 1, 128),
            w.squeeze(-1),
            lengths,
            table,
            meta,
            n,
            False,
        )
        reference = (
            torch.einsum("bhd,nd->bhn", q.cpu().double(), keys.cpu().double())
            .clamp_min(0)
            .mul(w.squeeze(-1).cpu().double()[:, :, None])
            .sum(1)
        )
        for row, length in enumerate([n, 2053]):
            torch.testing.assert_close(
                actual[row, :length].cpu().double(),
                reference[row, :length],
                atol=2e-6,
                rtol=2e-6,
            )
            a = set(actual[row, :length].topk(2048).indices.cpu().tolist())
            r = set(reference[row, :length].topk(2048).indices.tolist())
            self.assertEqual(a, r)


if __name__ == "__main__":
    unittest.main()
