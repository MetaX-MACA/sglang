"""Validate the sgl_kernel FlashMLA adapter against dense PyTorch attention."""

import math
import unittest

import torch
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
from sgl_kernel.platform import is_maca

from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(is_maca(), "Requires the MACA FlashMLA package")
class TestMacaFlashMla(CustomTestCase):
    def test_dense_attention_and_graph(self):
        torch.manual_seed(27)
        batch, query_len, heads, dim, value_dim, page = 2, 3, 16, 576, 512, 64
        lengths = [259, 137]
        seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        table = torch.randperm(batch * 5, dtype=torch.int32, device="cuda").view(
            batch, 5
        )
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                q = (torch.randn(batch, query_len, heads, dim, device="cuda") * 0.2).to(
                    dtype
                )
                cache = (torch.randn(batch * 5, page, 1, dim, device="cuda") * 0.2).to(
                    dtype
                )
                fp8 = dtype == torch.float8_e4m3fn
                q_scale, k_scale = (0.25, 0.75) if fp8 else (1.0, 1.0)
                descale_q = torch.tensor([q_scale], device="cuda") if fp8 else None
                descale_k = torch.tensor([k_scale], device="cuda") if fp8 else None
                metadata, splits = get_mla_metadata(
                    seq_lens, query_len * heads, 1, is_fp8_kvcache=fp8
                )

                def run():
                    return flash_mla_with_kvcache(
                        q,
                        cache,
                        table,
                        seq_lens,
                        value_dim,
                        metadata,
                        splits,
                        causal=True,
                        descale_q=descale_q,
                        descale_k=descale_k,
                        is_fp8_kvcache=fp8,
                    )

                def reference():
                    outputs, lses = [], []
                    for b, length in enumerate(lengths):
                        kv = (
                            cache[table[b].long()].float().reshape(-1, dim)[:length]
                            * k_scale
                        )
                        logits = torch.einsum(
                            "qhd,kd->hqk", q[b].float() * q_scale, kv
                        ) / math.sqrt(dim)
                        mask = torch.arange(length, device="cuda")[None, :] > (
                            length
                            - query_len
                            + torch.arange(query_len, device="cuda")[:, None]
                        )
                        logits.masked_fill_(mask, float("-inf"))
                        outputs.append(
                            torch.einsum(
                                "hqk,kd->qhd", logits.softmax(-1), kv[:, :value_dim]
                            )
                        )
                        lses.append(logits.logsumexp(-1))
                    return torch.stack(outputs), torch.stack(lses)

                for _ in range(3):
                    out, lse = run()
                expected, expected_lse = reference()
                torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_out, graph_lse = run()
                q.copy_((q.float() * 0.5).to(dtype))
                graph.replay()
                torch.cuda.synchronize()
                expected, expected_lse = reference()
                torch.testing.assert_close(
                    graph_out.float(), expected, atol=2e-3, rtol=2e-2
                )
                torch.testing.assert_close(
                    graph_lse, expected_lse, atol=2e-4, rtol=2e-4
                )


if __name__ == "__main__":
    unittest.main()
