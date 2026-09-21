"""Stepwise correctness tests for the GLM/DSA FP8 and INT8 main KV cache."""

import unittest

import torch

from sglang.srt.layers.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)
from sglang.srt.layers.attention.dsa.kv_cache_layout import (
    HEAD_DIM,
    NOPE_DIM,
    PACKED_BYTES_PER_TOKEN,
    ROPE_DIM,
)
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache_int8_ref,
    quantize_k_cache_int8_separate,
    quantize_k_cache_separate,
)
from sglang.srt.layers.attention.dsa.triton_flashmla_kv import (
    flash_mla_with_kvcache_torch,
    flash_mla_with_kvcache_triton,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.triton_ops.mla_buffer import set_mla_kv_buffer_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


def _source_kv(num_tokens: int, device: str):
    generator = torch.Generator(device="cpu").manual_seed(20260909)
    k_nope = (
        torch.randn(num_tokens, 1, NOPE_DIM, generator=generator)
        .clamp(-3, 3)
        .to(torch.bfloat16)
        .to(device)
    )
    k_rope = (
        torch.randn(num_tokens, 1, ROPE_DIM, generator=generator)
        .clamp(-2, 2)
        .to(torch.bfloat16)
        .to(device)
    )
    return k_nope, k_rope


def _join_packed(nope_part, rope_part, dtype, num_pages=2, page_size=64):
    packed_u8 = torch.cat([nope_part, rope_part], dim=-1)
    assert packed_u8.shape == (num_pages * page_size, 1, PACKED_BYTES_PER_TOKEN)
    return packed_u8.view(dtype).reshape(
        num_pages, page_size, 1, PACKED_BYTES_PER_TOKEN
    )


class TestDSAInt8ReferenceLayout(unittest.TestCase):
    def test_reference_layout_and_error_bound_on_cpu(self):
        k_nope, k_rope = _source_kv(7, "cpu")
        nope_part, rope_part = quantize_k_cache_int8_ref(k_nope, k_rope)
        self.assertEqual(nope_part.shape, (7, 1, 528))
        self.assertEqual(rope_part.shape, (7, 1, 128))
        packed = torch.cat([nope_part, rope_part], dim=-1).view(torch.int8)
        reconstructed = dequantize_k_cache_paged_ref(
            packed, torch.arange(7, dtype=torch.int32)
        )
        self.assertTrue(torch.equal(reconstructed[..., NOPE_DIM:], k_rope))
        torch.testing.assert_close(
            reconstructed[..., :NOPE_DIM].float(),
            k_nope.float(),
            atol=3.0e-2,
            rtol=0,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton tests")
class TestDSAQuantizedKVCacheCUDA(unittest.TestCase):
    device = "cuda"

    def test_int8_pool_allocates_packed_rows(self):
        pool = DSATokenToKVPool(
            size=128,
            page_size=64,
            kv_lora_rank=NOPE_DIM,
            dtype=torch.int8,
            qk_rope_head_dim=ROPE_DIM,
            layer_num=1,
            device=self.device,
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=PACKED_BYTES_PER_TOKEN,
        )
        self.assertTrue(pool.dsa_kv_cache_store_int8)
        self.assertTrue(pool.dsa_kv_cache_is_quantized)
        self.assertIs(pool.store_dtype, torch.uint8)
        self.assertEqual(pool.kv_cache_dim, PACKED_BYTES_PER_TOKEN)
        self.assertIs(pool.get_key_buffer(0).dtype, torch.int8)

    def test_int8_triton_quant_matches_reference(self):
        k_nope, k_rope = _source_kv(129, self.device)
        actual_nope, actual_rope = quantize_k_cache_int8_separate(k_nope, k_rope)
        ref_nope, ref_rope = quantize_k_cache_int8_ref(k_nope, k_rope)
        self.assertTrue(torch.equal(actual_nope[..., :NOPE_DIM], ref_nope[..., :NOPE_DIM]))
        torch.testing.assert_close(
            actual_nope[..., NOPE_DIM:].contiguous().view(torch.float32),
            ref_nope[..., NOPE_DIM:].contiguous().view(torch.float32),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        self.assertTrue(torch.equal(actual_rope, ref_rope))

    def test_int8_scatter_then_paged_dequant(self):
        k_nope, k_rope = _source_kv(5, self.device)
        nope_part, rope_part = quantize_k_cache_int8_separate(k_nope, k_rope)
        raw_pool = torch.zeros(
            128, 1, PACKED_BYTES_PER_TOKEN, dtype=torch.uint8, device=self.device
        )
        loc = torch.tensor([65, 2, 97, 5, 66], dtype=torch.int64, device=self.device)
        set_mla_kv_buffer_triton(raw_pool, loc, nope_part, rope_part)
        actual = dequantize_k_cache_paged(raw_pool.view(torch.int8), loc.to(torch.int32))
        sequential = torch.cat([nope_part, rope_part], dim=-1).view(torch.int8)
        expected = dequantize_k_cache_paged_ref(
            sequential, torch.arange(5, dtype=torch.int32, device=self.device)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def _run_decode(self, cache_dtype):
        num_pages, page_size = 2, 64
        k_nope, k_rope = _source_kv(num_pages * page_size, self.device)
        if cache_dtype is torch.int8:
            nope_part, rope_part = quantize_k_cache_int8_separate(k_nope, k_rope)
        else:
            nope_part, rope_part = quantize_k_cache_separate(k_nope, k_rope)
        cache = _join_packed(nope_part, rope_part, cache_dtype)
        generator = torch.Generator(device="cpu").manual_seed(17)
        q = (
            torch.randn(3, 1, 4, HEAD_DIM, generator=generator)
            .clamp(-2, 2)
            .to(torch.bfloat16)
            .to(self.device)
        )
        indices = torch.stack(
            [
                torch.randperm(128, generator=generator)[:37],
                torch.randperm(128, generator=generator)[:37],
                torch.full((37,), -1),
            ]
        ).to(device=self.device, dtype=torch.int32)
        indices[1, 31:] = -1
        indices = indices.unsqueeze(1)
        expected, expected_lse = flash_mla_with_kvcache_torch(
            q=q, k_cache=cache, indices=indices, softmax_scale=HEAD_DIM**-0.5
        )
        actual, actual_lse = flash_mla_with_kvcache_triton(
            q=q,
            k_cache=cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=NOPE_DIM,
            indices=indices,
            softmax_scale=HEAD_DIM**-0.5,
        )
        torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(actual_lse, expected_lse, atol=5e-3, rtol=5e-3)

    def test_fp8_triton_decode_matches_torch(self):
        self._run_decode(torch.float8_e4m3fn)

    def test_int8_triton_decode_matches_torch(self):
        self._run_decode(torch.int8)


if __name__ == "__main__":
    unittest.main()
