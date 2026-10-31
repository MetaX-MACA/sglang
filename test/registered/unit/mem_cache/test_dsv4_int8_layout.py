"""Packed V4 dtype geometry and INT8 scale addressing across page boundaries."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged_ref,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import (
    KVLayout,
    kv_cache_bytes_per_token,
    kv_cache_page_bytes,
    kv_cache_scale_bytes_per_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSV4Int8Layout(CustomTestCase):
    def test_dtype_page_geometry(self):
        for dtype, width, scales in (
            (torch.int8, 608, 32),
            (torch.uint8, 584, 8),
            (torch.float8_e4m3fn, 584, 8),
            (torch.bfloat16, 1024, 0),
        ):
            with self.subTest(dtype=dtype):
                self.assertEqual(kv_cache_bytes_per_token(KVLayout.V4, dtype), width)
                self.assertEqual(
                    kv_cache_scale_bytes_per_token(KVLayout.V4, dtype), scales
                )
                for page_size in (2, 8, 64, 256):
                    expected = page_size * width
                    if scales:
                        expected = (expected + 575) // 576 * 576
                    self.assertEqual(
                        kv_cache_page_bytes("v4", dtype, page_size), expected
                    )
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            with self.assertRaisesRegex(ValueError, "V4 layout"):
                kv_cache_page_bytes(layout, torch.int8, 64)
            self.assertEqual(
                kv_cache_page_bytes(layout, torch.uint8, 64), layout.page_bytes(64)
            )

    def test_int8_reference_reads_fp32_scales_and_bf16_rope(self):
        page_size = 2
        cache = torch.zeros(
            (3, kv_cache_page_bytes("v4", torch.int8, page_size)), dtype=torch.int8
        )
        expected = []
        for token in range(6):
            page, slot = divmod(token, page_size)
            row = cache[page]
            values = (
                torch.arange(448, dtype=torch.int32)
                .remainder(255)
                .sub(127)
                .to(torch.int8)
            )
            row[slot * 576 : slot * 576 + 448] = values
            rope = torch.full((64,), token + 0.25, dtype=torch.bfloat16)
            row[slot * 576 + 448 : (slot + 1) * 576].view(torch.bfloat16).copy_(rope)
            scales = torch.arange(1, 8, dtype=torch.float32) * (token + 1) / 32
            scale_base = page_size * 576 + slot * 32
            row[scale_base : scale_base + 28].view(torch.float32).copy_(scales)
            expected.append(
                torch.cat((values.float() * scales.repeat_interleave(64), rope.float()))
            )
        locations = torch.tensor([5, 0, 2, 1, 4, 3], dtype=torch.int32)
        result = dequantize_k_cache_paged_ref(cache, locations, page_size).squeeze(1)
        torch.testing.assert_close(
            result, torch.stack(expected)[locations.long()].bfloat16(), atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
