"""Numerical regression for V4 INT8 packed stores, gathers and sparse decode."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)
from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla
from sglang.kernels.ops.attention.dsv4.kv_layout import kv_cache_page_bytes
from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
    flash_mla_sparse_decode_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDSV4Int8Cache(CustomTestCase):
    def test_fused_norm_rope_int8_store(self):
        torch.manual_seed(7)
        device, n, page_size = "cuda", 5, 2
        keys = torch.randn((n, 512), device=device, dtype=torch.bfloat16)
        weight = torch.randn((512,), device=device, dtype=torch.bfloat16)
        positions = torch.arange(n, device=device, dtype=torch.int64)
        angles = torch.randn((n, 32), device=device)
        freqs = torch.polar(torch.ones_like(angles), angles)
        locations = torch.arange(n, device=device, dtype=torch.int32)
        cache = torch.zeros(
            (3, kv_cache_page_bytes("v4", torch.int8, page_size)),
            device=device,
            dtype=torch.int8,
        )
        fused_k_norm_rope_flashmla(
            keys, weight, 1e-6, freqs, positions, locations, cache, page_size
        )
        actual = (
            dequantize_k_cache_paged(cache, locations, page_size).squeeze(1).float()
        )
        normalized = (
            keys.float()
            * (keys.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * weight.float()
        )
        rope = (
            torch.view_as_complex(normalized[:, 448:].reshape(n, 32, 2).contiguous())
            * freqs
        )
        normalized[:, 448:] = torch.view_as_real(rope).flatten(1)
        # The quantization error is bounded by half a group scale plus BF16 rounding.
        error_bound = (
            normalized[:, :448].reshape(n, 7, 64).abs().amax(-1).clamp_min(1e-4)
            / 127
            / 2
        )
        error = (actual[:, :448] - normalized[:, :448]).reshape(n, 7, 64).abs()
        self.assertTrue(
            torch.all(
                error
                <= error_bound[..., None]
                + normalized[:, :448].reshape(n, 7, 64).abs() / 256
                + 2e-5
            ).item()
        )
        torch.testing.assert_close(
            actual[:, 448:],
            normalized[:, 448:].bfloat16().float(),
            rtol=0.004,
            atol=0.008,
        )

    def test_store_gather_and_decode(self):
        torch.manual_seed(42)
        device = "cuda"
        for page_size in (2, 64):
            with self.subTest(page_size=page_size):
                n = page_size + 3
                cache = torch.zeros(
                    (3, kv_cache_page_bytes("v4", torch.int8, page_size)),
                    device=device,
                    dtype=torch.int8,
                )
                locations = torch.arange(n, device=device, dtype=torch.int32).flip(0)
                keys = torch.randn((n, 512), device=device, dtype=torch.bfloat16)
                # Include the zero-group scale floor and values near the INT8 limit.
                keys[0, :64] = 0
                fused_store_cache(
                    keys, cache, locations, page_size=page_size, type="flashmla"
                )
                gathered = dequantize_k_cache_paged(
                    cache, locations, page_size
                ).squeeze(1)
                nope = keys[:, :448].float().reshape(n, 7, 64)
                scale = nope.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 127
                # Dequantization must match the byte-level reference exactly.
                reference = dequantize_k_cache_paged_ref(
                    cache, locations, page_size
                ).squeeze(1)
                torch.testing.assert_close(gathered, reference, rtol=0, atol=0)
                torch.testing.assert_close(
                    gathered[:, 448:], keys[:, 448:], rtol=0, atol=0
                )
                pages = locations.long() // page_size
                slots = locations.long() % page_size
                quantized = cache[
                    pages[:, None],
                    slots[:, None] * 576 + torch.arange(448, device=device),
                ].float()
                scales = cache.view(torch.float32)[
                    pages[:, None],
                    (page_size * 576 + slots[:, None] * 32) // 4
                    + torch.arange(7, device=device),
                ]
                torch.testing.assert_close(
                    scales, scale.squeeze(-1), rtol=2e-7, atol=1e-8
                )
                full_nope = quantized.reshape(n, 7, 64) * scales[..., None]
                # roundf uses ties away from zero; torch.round uses ties to even.
                # Both tie choices obey this half-step bound. Evaluate it before
                # BF16 output rounding so the allowance is exactly one half scale.
                self.assertTrue(
                    torch.all(
                        (full_nope - nope).abs() <= scales[..., None] / 2 + 1e-6
                    ).item()
                )
                full_keys = torch.cat(
                    (full_nope.reshape(n, 448), keys[:, 448:].float()), -1
                )

                query = torch.randn(
                    (2, 1, 32, 512), device=device, dtype=torch.bfloat16
                )
                # Different valid lengths, masked indices, and a second compressed pool.
                indices = torch.stack((locations, locations)).unsqueeze(1)
                lengths = torch.tensor([n, n - 2], device=device, dtype=torch.int32)
                sink = torch.randn((32,), device=device)
                cache_view = cache.view(-1, 1, 1).as_strided(
                    (3, page_size, 1, 576), (cache.stride(0), 576, 576, 1)
                )
                out, lse = flash_mla_sparse_decode_triton(
                    query,
                    cache_view,
                    indices,
                    lengths,
                    sink,
                    512,
                    512**-0.5,
                    cache_view,
                    indices,
                    lengths,
                )
                for batch in range(2):
                    k = full_keys[: int(lengths[batch].item())]
                    k = torch.cat((k, k))
                    logits = query[batch, 0].float() @ k.T / (512**0.5)
                    logits_with_sink = torch.cat((logits, sink[:, None]), -1)
                    expected_out = logits_with_sink.softmax(-1)[:, :-1] @ k
                    print(
                        f"page_size={page_size} batch={batch} sparse max_abs={(out[batch, 0].float() - expected_out).abs().max().item():.8g}"
                    )
                    torch.testing.assert_close(
                        out[batch, 0].float(), expected_out, atol=0.002, rtol=0.004
                    )
                    torch.testing.assert_close(
                        lse[batch, :, 0],
                        logits_with_sink.logsumexp(-1),
                        atol=0.001,
                        rtol=0.0003,
                    )


if __name__ == "__main__":
    unittest.main()
