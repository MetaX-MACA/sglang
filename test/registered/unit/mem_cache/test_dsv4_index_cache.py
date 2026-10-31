"""V4 index-cache selection and allocation agree in elements and bytes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    get_dsv4_indexer_bytes_per_token,
)
from sglang.srt.mem_cache.dsv4_index_cache import (
    get_dsv4_index_cache_storage_dtype,
    get_dsv4_index_cache_type,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _dsv4_indexer_regions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDSV4IndexCache(CustomTestCase):
    @patch("sglang.srt.mem_cache.dsv4_index_cache.is_maca", return_value=True)
    def test_cache_selection_allocation_and_hicache_bytes(self, _platform):
        for mode, dtype, width in (
            ("int8", torch.int8, 132),
            ("fp8", torch.uint8, 132),
            ("bf16", torch.bfloat16, 256),
            ("bfloat16", torch.bfloat16, 256),
        ):
            with (
                self.subTest(mode=mode),
                envs.SGLANG_DSV4_INDEX_CACHE_TYPE.override(mode),
            ):
                pool = DeepSeekV4IndexerPool(
                    128, 64, torch.bfloat16, 128, 1, "cpu", False, use_fp4_indexer=False
                )
                buf = pool.get_index_k_with_scale_buffer(0)
                self.assertIs(buf.dtype, dtype)
                self.assertEqual(buf.shape, (3, 64 * width // dtype.itemsize))
                self.assertEqual(buf.nbytes, 3 * 64 * width)
                self.assertEqual(get_dsv4_indexer_bytes_per_token(128, False), width)
                self.assertIs(get_dsv4_index_cache_storage_dtype(), dtype)
                (region,) = _dsv4_indexer_regions(
                    SimpleNamespace(c4_indexer_kv_pool=pool), 256
                )
                self.assertEqual(region.item_bytes, 64 * width)
                self.assertEqual(region.page_aligned_only, dtype is torch.bfloat16)
        with envs.SGLANG_DSV4_INDEX_CACHE_TYPE.override("bogus"):
            with self.assertRaisesRegex(ValueError, "must be int8, fp8, or bf16"):
                get_dsv4_index_cache_type()
        with envs.SGLANG_DSV4_INDEX_CACHE_TYPE.override("bf16"):
            with self.assertRaisesRegex(ValueError, "incompatible with FP4"):
                DeepSeekV4IndexerPool(
                    128, 64, torch.bfloat16, 128, 1, "cpu", False, use_fp4_indexer=True
                )

    @patch("sglang.srt.mem_cache.dsv4_index_cache.is_maca", return_value=False)
    def test_other_platform_default_remains_fp8(self, _platform):
        with envs.SGLANG_DSV4_INDEX_CACHE_TYPE.override("bf16"):
            self.assertEqual(get_dsv4_index_cache_type(), "fp8")
            self.assertEqual(get_dsv4_indexer_bytes_per_token(128, False), 132)


if __name__ == "__main__":
    unittest.main()
