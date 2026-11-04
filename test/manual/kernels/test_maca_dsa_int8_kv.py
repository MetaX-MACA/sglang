"""MACA regression for packed DSA INT8 KV, sparse decode and HiCache."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)
from sglang.kernels.ops.attention.dsa.triton_flashmla_kv import (
    flash_mla_with_kvcache_torch,
    flash_mla_with_kvcache_triton,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import is_maca
from sglang.test.test_utils import CustomTestCase


class TestMacaDsaInt8KV(CustomTestCase):
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

    def make_pool(self):
        return MLATokenToKVPool(
            size=192,
            page_size=64,
            dtype=torch.int8,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
            use_dsa=True,
            override_kv_cache_dim=656,
        )

    def test_store_gather_decode_and_graph(self):
        pool = self.make_pool()
        keys = torch.randn(4, 1, 512, device="cuda", dtype=torch.bfloat16)
        rope = torch.randn(4, 1, 64, device="cuda", dtype=torch.bfloat16)
        keys[0, :, :128] = 0
        loc = torch.tensor([130, 2, 65, 0], device="cuda", dtype=torch.int64)
        pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), loc, keys, rope)
        cache = pool.get_key_buffer(0)
        self.assertEqual(cache.dtype, torch.int8)
        self.assertEqual(cache.shape[-1], 656)
        self.assertTrue(torch.all(cache[0] == 0).item())
        actual = dequantize_k_cache_paged(cache, loc[:3])
        reference = dequantize_k_cache_paged_ref(cache, loc[:3])
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual[..., 512:], rope[:3], rtol=0, atol=0)
        torch.testing.assert_close(actual[..., :512], keys[:3], rtol=0.01, atol=0.02)

        query = torch.randn(2, 1, 16, 576, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor(
            [[[130, 2, -1, 65]], [[-1, -1, -1, -1]]], device="cuda", dtype=torch.int32
        )
        # Match the production sparse width, including fully invalid tiles.
        indices = F.pad(indices, (0, 2044), value=-1)
        paged = cache.view(-1, 64, 1, 656)
        expected, _ = flash_mla_with_kvcache_torch(query, paged, indices)
        kwargs = dict(
            q=query,
            k_cache=paged,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            indices=indices,
        )
        output, _ = flash_mla_with_kvcache_triton(**kwargs)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue((output[1] == 0).all().item())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, _ = flash_mla_with_kvcache_triton(**kwargs)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0.02, atol=0.02)

    def test_hicache_preserves_packed_bytes(self):
        pool = self.make_pool()
        pool.kv_buffer[0].random_(0, 256)
        host = MLATokenToKVPoolHost(
            device_pool=pool,
            host_to_device_ratio=1,
            host_size=0,
            page_size=64,
            layout="layer_first",
            override_kv_cache_dim=656,
        )
        # Use the production HiCache per-layer interface in both directions.
        loc = torch.cat(
            [torch.arange(p * 64, (p + 1) * 64, device="cuda") for p in (1, 2)]
        )
        host_loc = torch.arange(128, device="cuda", dtype=torch.int64)
        expected = pool.kv_buffer[0][loc].clone()
        host.backup_from_device_all_layer(pool, host_loc, loc, io_backend="kernel")
        torch.cuda.synchronize()
        pool.kv_buffer[0][loc] = 0
        host.load_to_device_per_layer(pool, host_loc, loc, 0, io_backend="kernel")
        torch.cuda.synchronize()
        torch.testing.assert_close(pool.kv_buffer[0][loc], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
