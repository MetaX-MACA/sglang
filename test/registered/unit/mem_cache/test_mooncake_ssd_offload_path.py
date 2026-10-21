import os
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fake_store_class():
    class FakeMooncakeDistributedStore:
        instances = []

        def __init__(self):
            self.setup_kwargs = None
            self.objects = {}
            type(self).instances.append(self)

        def setup(self, *args, **kwargs):
            self.setup_kwargs = kwargs
            return 0

        def register_buffer(self, *args, **kwargs):
            return 0

        def put(self, key, value, *args):
            self.objects[key] = value
            return 0

        def get(self, key):
            return self.objects.get(key)

        def is_exist(self, key):
            return 1 if key in self.objects else 0

        def batch_is_exist(self, keys):
            return [1 if key in self.objects else 0 for key in keys]

        def remove(self, key):
            self.objects.pop(key, None)
            return 0

    return FakeMooncakeDistributedStore


def _fake_mooncake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
    }


def _fake_host_pool_modules():
    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")

    class HostKVCache:
        pass

    class HostTensorAllocator:
        pass

    pool_host.HostKVCache = HostKVCache
    pool_host.HostTensorAllocator = HostTensorAllocator

    memory_pool_host = types.ModuleType("sglang.srt.mem_cache.memory_pool_host")

    class MLATokenToKVPoolHost:
        pass

    memory_pool_host.MLATokenToKVPoolHost = MLATokenToKVPoolHost
    return {
        "sglang.srt.mem_cache.pool_host": pool_host,
        "sglang.srt.mem_cache.memory_pool_host": memory_pool_host,
    }


def _make_config(
    *,
    tp_rank,
    pp_rank,
    ssd_offload_path,
    dp_rank=0,
    tp_size=8,
    pp_size=1,
    attn_cp_rank=0,
    attn_cp_size=1,
):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_rank=dp_rank,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "global_segment_size": 1024 * 1024,
            "enable_ssd_offload": True,
            "standalone_storage": False,
            "ssd_offload_path": ssd_offload_path,
        },
    )


def _make_store(
    *,
    tp_rank,
    pp_rank,
    ssd_offload_path,
    dp_rank=0,
    tp_size=8,
    pp_size=1,
    attn_cp_rank=0,
    attn_cp_size=1,
):
    cfg = _make_config(
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        ssd_offload_path=ssd_offload_path,
        dp_rank=dp_rank,
        tp_size=tp_size,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
    )
    return _make_store_from_config(cfg)


def _make_store_from_config(cfg):
    fake_store_cls = _fake_store_class()
    with patch.dict(
        "sys.modules",
        {
            **_fake_mooncake_modules(fake_store_cls),
            **_fake_host_pool_modules(),
        },
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        MooncakeStore(cfg)
    return fake_store_cls.instances[-1]


class TestMooncakeSsdOffloadPath(CustomTestCase):
    def test_disabled_ssd_preserves_path_without_filesystem_access(self):
        with tempfile.TemporaryDirectory() as base:
            path = os.path.join(base, "unused_ssd")
            for explicit_disable in (False, True):
                with self.subTest(explicit_disable=explicit_disable):
                    cfg = _make_config(
                        tp_rank=0, pp_rank=0, ssd_offload_path=path
                    )
                    if explicit_disable:
                        cfg.extra_config["enable_ssd_offload"] = False
                    else:
                        del cfg.extra_config["enable_ssd_offload"]
                    with patch.object(
                        os, "makedirs", side_effect=PermissionError("read-only path")
                    ) as mkdir:
                        client = _make_store_from_config(cfg)
                    mkdir.assert_not_called()
                    self.assertEqual(client.setup_kwargs["ssd_offload_path"], path)
                    self.assertNotIn("enable_ssd_offload", client.setup_kwargs)
                    self.assertFalse(os.path.exists(path))

    def test_disabled_ssd_does_not_validate_unused_path(self):
        with tempfile.TemporaryDirectory() as base:
            path = os.path.join(base, "ordinary_file")
            with open(path, "w") as output:
                output.write("not a directory")
            cfg = _make_config(tp_rank=0, pp_rank=0, ssd_offload_path=path)
            cfg.extra_config["enable_ssd_offload"] = False
            client = _make_store_from_config(cfg)
            self.assertEqual(client.setup_kwargs["ssd_offload_path"], path)

    def test_no_explicit_ssd_path_does_not_create_directory(self):
        for enabled in (False, True):
            with self.subTest(enable_ssd_offload=enabled):
                cfg = _make_config(tp_rank=0, pp_rank=0, ssd_offload_path=None)
                cfg.extra_config["enable_ssd_offload"] = enabled
                with patch.object(os, "makedirs") as mkdir:
                    client = _make_store_from_config(cfg)
                mkdir.assert_not_called()
                self.assertNotIn("ssd_offload_path", client.setup_kwargs)
                self.assertEqual(
                    client.setup_kwargs.get("enable_ssd_offload", False), enabled
                )

    def test_decode_offload_manager_forwards_replica_dp_rank(self):
        from sglang.srt.disaggregation import (
            decode_kvcache_offload_manager as manager_module,
        )

        class FakeMLAPool:
            pass

        args = types.SimpleNamespace(
            page_size=256,
            hicache_ratio=2,
            hicache_size=0,
            hicache_mem_layout="page_first",
            hicache_io_backend="direct",
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config='{"master_server_address":"test"}',
            served_model_name="test",
        )
        allocator = types.SimpleNamespace(get_kvcache=FakeMLAPool)
        # Omitting the new argument must preserve the existing DP0 default.
        for dp_rank in (None, 0, 1, 2, 3):
            with self.subTest(dp_rank=dp_rank):
                with (
                    patch.multiple(
                        manager_module,
                        MLATokenToKVPool=FakeMLAPool,
                        MLATokenToKVPoolHost=Mock(),
                        HiCacheController=Mock(),
                    ),
                    patch.object(
                        manager_module.envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE,
                        "get",
                        return_value=None,
                    ),
                    patch.object(
                        manager_module.torch.distributed,
                        "get_world_size",
                        return_value=1,
                    ),
                ):
                    manager_module.DecodeKVCacheOffloadManager(
                        req_to_token_pool=object(),
                        token_to_kv_pool_allocator=allocator,
                        tp_group=object(),
                        tree_cache=object(),
                        server_args=args,
                        **({} if dp_rank is None else {"dp_rank": dp_rank}),
                    )
                    self.assertEqual(
                        manager_module.HiCacheController.call_args.kwargs["dp_rank"],
                        0 if dp_rank is None else dp_rank,
                    )

    def test_replica_dp_rank_reaches_storage_without_dp_attention(self):
        from sglang.srt.managers import cache_controller as controller_module

        with tempfile.TemporaryDirectory() as base:
            paths = set()
            for dp_rank in range(4):
                for pp_rank in range(4):
                    controller = controller_module.HiCacheController.__new__(
                        controller_module.HiCacheController
                    )
                    controller._replica_dp_rank = dp_rank
                    controller.mem_pool_device = object()
                    controller.mem_pool_host = types.SimpleNamespace(layout="page_first")
                    controller.enable_storage_metrics = False
                    controller.attn_cp_group = None
                    extra_config = _make_config(
                        tp_rank=0, pp_rank=pp_rank, ssd_offload_path=base
                    ).extra_config
                    with patch.multiple(
                        controller_module,
                        is_dp_attention_enabled=Mock(return_value=False),
                        get_tensor_model_parallel_rank=Mock(return_value=0),
                        get_tensor_model_parallel_world_size=Mock(return_value=1),
                        get_pipeline_model_parallel_rank=Mock(return_value=pp_rank),
                        get_pipeline_model_parallel_world_size=Mock(return_value=4),
                    ):
                        cfg = controller._generate_storage_config("test", extra_config)
                    self.assertEqual(cfg.dp_rank, dp_rank)
                    client = _make_store_from_config(cfg)
                    path = client.setup_kwargs["ssd_offload_path"]
                    self.assertEqual(
                        path, os.path.join(base, f"rank_{dp_rank}_0_{pp_rank}_0")
                    )
                    self.assertNotIn(path, paths)
                    paths.add(path)
            self.assertEqual(len(paths), 16)

    def test_dp_attention_uses_attention_dp_rank(self):
        from sglang.srt.managers import cache_controller as controller_module

        controller = controller_module.HiCacheController.__new__(
            controller_module.HiCacheController
        )
        controller._replica_dp_rank = 0
        controller.mem_pool_device = object()
        controller.mem_pool_host = types.SimpleNamespace(layout="page_first")
        controller.enable_storage_metrics = False
        controller.attn_cp_group = None
        with patch.multiple(
            controller_module,
            is_dp_attention_enabled=Mock(return_value=True),
            get_attention_dp_rank=Mock(return_value=3),
            get_attention_tp_rank=Mock(return_value=0),
            get_attention_tp_size=Mock(return_value=1),
            get_pipeline_model_parallel_rank=Mock(return_value=0),
            get_pipeline_model_parallel_world_size=Mock(return_value=1),
        ):
            cfg = controller._generate_storage_config("test", {})
        self.assertEqual(cfg.dp_rank, 3)

    def test_prefill_parallel_topologies(self):
        # TP is the full tensor group; attention TP = TP / DP / CP.
        for tp_size, dp_size, cp_size, pp_size in (
            (8, 1, 8, 1),
            (8, 1, 1, 2),
            (8, 8, 1, 1),
            (8, 2, 2, 2),
        ):
            with self.subTest(tp=tp_size, dp=dp_size, cp=cp_size, pp=pp_size):
                with tempfile.TemporaryDirectory() as base:
                    attn_tp_size = tp_size // dp_size // cp_size
                    paths = set()
                    for pp_rank in range(pp_size):
                        for rank in range(tp_size):
                            dp_rank = rank // (attn_tp_size * cp_size)
                            cp_rank = (rank // attn_tp_size) % cp_size
                            tp_rank = rank % attn_tp_size
                            client = _make_store(
                                tp_rank=tp_rank,
                                pp_rank=pp_rank,
                                dp_rank=dp_rank,
                                tp_size=attn_tp_size,
                                pp_size=pp_size,
                                attn_cp_rank=cp_rank,
                                attn_cp_size=cp_size,
                                ssd_offload_path=base,
                            )
                            path = client.setup_kwargs["ssd_offload_path"]
                            self.assertEqual(
                                path,
                                os.path.join(
                                    base, f"rank_{dp_rank}_{tp_rank}_{pp_rank}_{cp_rank}"
                                ),
                            )
                            self.assertTrue(os.path.isdir(path))
                            self.assertNotIn(path, paths)
                            paths.add(path)
                    self.assertEqual(len(paths), tp_size * pp_size)

    def test_each_rank_gets_private_subdirectory(self):
        with tempfile.TemporaryDirectory() as base:
            for tp_rank in (0, 3):
                client = _make_store(tp_rank=tp_rank, pp_rank=0, ssd_offload_path=base)
                expected = os.path.join(base, f"rank_0_{tp_rank}_0_0")
                self.assertEqual(client.setup_kwargs.get("ssd_offload_path"), expected)
                self.assertTrue(os.path.isdir(expected))

    def test_rank_directories_are_distinct(self):
        with tempfile.TemporaryDirectory() as base:
            paths = {
                _make_store(
                    tp_rank=tp_rank, pp_rank=0, ssd_offload_path=base
                ).setup_kwargs["ssd_offload_path"]
                for tp_rank in range(4)
            }
            self.assertEqual(len(paths), 4)

    def test_dp_ranks_are_distinct_when_attn_tp_rank_is_zero(self):
        # dp-attention with attn_tp_size 1: every DP rank reports tp_rank 0
        with tempfile.TemporaryDirectory() as base:
            paths = {
                _make_store(
                    tp_rank=0, pp_rank=0, ssd_offload_path=base, dp_rank=dp_rank
                ).setup_kwargs["ssd_offload_path"]
                for dp_rank in range(8)
            }
            self.assertEqual(len(paths), 8)
            self.assertIn(os.path.join(base, "rank_5_0_0_0"), paths)

    def test_cp_ranks_are_distinct_with_dp_attention(self):
        """CP peers with identical DP/TP/PP ranks must not share SSD files."""
        with tempfile.TemporaryDirectory() as base:
            paths = {
                _make_store(
                    tp_rank=0,
                    pp_rank=0,
                    ssd_offload_path=base,
                    dp_rank=dp_rank,
                    tp_size=1,
                    attn_cp_rank=cp_rank,
                    attn_cp_size=4,
                ).setup_kwargs["ssd_offload_path"]
                for dp_rank in range(2)
                for cp_rank in range(4)
            }
            self.assertEqual(len(paths), 8)


if __name__ == "__main__":
    unittest.main()
