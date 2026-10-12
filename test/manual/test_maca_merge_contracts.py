"""CPU contract checks for the MACA 0.5.21 ports, using stubbed GPU operators.

Run directly with Python. These tests execute the production control flow via
AST extraction so GPU libraries are not needed; they do not check GPU numerics.
"""

import ast
import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[2]


class DType(Enum):
    BF16 = 2
    FP8 = 1
    UINT8 = 0
    INT8 = -1

    @property
    def itemsize(self):
        return 2 if self is DType.BF16 else 1


class TorchStub:
    bfloat16 = DType.BF16
    uint8 = DType.UINT8
    int8 = DType.INT8
    float8_e4m3fn = DType.FP8
    allocations = []

    @classmethod
    def empty(cls, shape, dtype, device):
        result = NS(shape=tuple(shape), dtype=dtype, device=device)
        cls.allocations.append(result)
        return result

    @classmethod
    def zeros(cls, *shape, dtype, device):
        result = cls.empty(shape, dtype, device)
        result.nbytes = shape[0] * shape[1] * dtype.itemsize
        return result


def load(relative, name, namespace, *, owner=None):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    body = tree.body
    if owner:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    node = next(
        n
        for n in body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
    )
    code = "from __future__ import annotations\n" + ast.unparse(node)
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[name]


class TestMacaMergeContracts(unittest.TestCase):
    def setUp(self):
        TorchStub.allocations = []

    def test_maca_compiler_fallback_does_not_modify_toolkit(self):
        ns = {
            "cache_once": lambda fn: fn,
            "os": os,
            "is_hip_runtime": lambda: False,
            "cuda_home": lambda: "/opt/maca/tools/cu-bridge",
        }
        fn = load(
            "python/sglang/kernels/jit/utils/compile/toolchain.py",
            "device_compiler_path",
            ns,
        )
        for maca, nvcc_exists, compiler in (
            (False, False, "nvcc"),
            (True, True, "nvcc"),
            (True, False, "cucc"),
        ):
            ns["is_maca_runtime"] = lambda: maca
            with patch.object(os.path, "isfile", return_value=nvcc_exists):
                self.assertEqual(fn(), f"/opt/maca/tools/cu-bridge/bin/{compiler}")


    def test_memcpy_masks_offsets_past_either_buffer(self):
        class Scalar(int):
            def to(self, dtype):
                return self

        store = Mock()
        tl = NS(
            int64=int, load=lambda p, **kwargs: p, store=store,
            arange=lambda a, b: 0, minimum=min,
        )
        fn = load(
            "python/sglang/kernels/ops/memory/memcpy_triton.py",
            "memcpy_triton_kernel",
            {"triton": NS(jit=lambda fn: fn), "tl": tl},
        )
        for source_offset, offset, row, expected in (
            (True, 7, 0, True), (True, 7, 1, False),
            (False, 4, 0, True), (False, 4, 1, False),
            (True, 8, 0, False), (False, 5, 0, False),
            (True, -1, 0, False),
        ):
            tl.program_id = lambda axis: Scalar(row)
            fn(2000, 1000, Scalar(offset), Scalar(8), 8, 5, source_offset, 1, 1)
            self.assertEqual(bool(store.call_args.kwargs["mask"]), expected)


    def test_collective_library_binds_native_and_maca_symbols(self):
        descriptor = NS(name="ncclAllReduce", restype=int, argtypes=[])
        config = NS(name="ncclCommInitRankConfig", restype=int, argtypes=[])
        for maca, prefix in ((False, "nccl"), (True, "mccl")):
            funcs = {f"{prefix}AllReduce": Mock()}
            if not maca:
                funcs[f"{prefix}CommInitRankConfig"] = Mock()
            registry = NS(
                exported_functions=[descriptor, config],
                exported_functions_symm_mem=[],
                path_to_dict_mapping={},
                path_to_library_cache={},
            )
            ns = {
                "NCCLLibrary": registry,
                "ctypes": NS(CDLL=lambda _: NS(**funcs)),
                "is_maca": lambda: maca,
            }
            init = load(
                "python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py",
                "__init__",
                ns,
                owner="NCCLLibrary",
            )
            instance = NS()
            init(instance, "test.so")
            self.assertIs(instance._funcs["ncclAllReduce"], funcs[f"{prefix}AllReduce"])
            self.assertEqual("ncclCommInitRankConfig" in instance._funcs, not maca)
            self.assertEqual(len(registry.exported_functions), 2)
            if maca:
                instance.ncclCommInitRank = Mock(return_value="communicator")
                init_config = load(
                    "python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py",
                    "ncclCommInitRankConfig",
                    ns,
                    owner="NCCLLibrary",
                )
                self.assertEqual(init_config(instance, 2, "id", 1), "communicator")
                instance.ncclCommInitRank.assert_called_once_with(2, "id", 1)


    def test_portable_capture_status_queries_requested_stream(self):
        original_stream = NS(cuda_stream=1234)
        requested_stream = NS(cuda_stream=5678)
        current_stream = original_stream

        @contextmanager
        def use_stream(stream):
            nonlocal current_stream
            previous = current_stream
            current_stream = stream
            try:
                yield
            finally:
                current_stream = previous

        device = NS(stream=Mock(side_effect=use_stream))
        for platform in ("maca", "hip", "xpu"):
            for active in (False, True):
                with self.subTest(platform=platform, active=active):

                    def capture_status():
                        self.assertIs(current_stream, requested_stream)
                        return active

                    device.is_current_stream_capturing = Mock(
                        side_effect=capture_status
                    )
                    native_query = Mock(
                        side_effect=AssertionError("NVIDIA runtime used")
                    )
                    ns = {
                        "is_maca": lambda: platform == "maca",
                        "is_hip": lambda: platform == "hip",
                        "_is_xpu": platform == "xpu",
                        "get_device_module": lambda: device,
                        "_capture_status": native_query,
                        "rt": None,
                    }
                    fn = load(
                        "python/sglang/srt/model_executor/runner_backend_utils/"
                        "breakable_cuda_graph/breakable_cuda_graph.py",
                        "_is_stream_capturing",
                        ns,
                    )
                    self.assertEqual(fn(requested_stream), active)
                    self.assertIs(current_stream, original_stream)
                    device.stream.assert_called_with(requested_stream)
                    device.is_current_stream_capturing.assert_called_once_with()
                    native_query.assert_not_called()


    def test_nvidia_capture_status_keeps_native_runtime_query(self):
        stream = NS(cuda_stream=1234)
        query = Mock()
        portable_device = Mock(side_effect=AssertionError("portable query used"))
        ns = {
            "is_maca": lambda: False,
            "is_hip": lambda: False,
            "_is_xpu": False,
            "get_device_module": portable_device,
            "_capture_status": query,
            "rt": NS(cudaStreamCaptureStatus=NS(cudaStreamCaptureStatusActive=1)),
        }
        fn = load(
            "python/sglang/srt/model_executor/runner_backend_utils/"
            "breakable_cuda_graph/breakable_cuda_graph.py",
            "_is_stream_capturing",
            ns,
        )
        for status in (0, 1, 2):
            query.return_value = status
            self.assertEqual(fn(stream), status == 1)
            query.assert_called_with(stream.cuda_stream)
        portable_device.assert_not_called()


    def test_maca_graph_dedup_uses_plain_torch_graphs(self):
        native_graph = Mock(return_value=NS(raw_cuda_graph=Mock()))
        registry = Mock()
        ns = {
            "is_maca": lambda: True,
            "envs": NS(SGLANG_ENABLE_CUDA_GRAPH_DEDUP=NS(get=lambda: True)),
            "cuda_drv": object(),
            "cuda_rt": object(),
            "torch": NS(cuda=NS(CUDAGraph=native_graph)),
            "DedupedCudaGraphRegistry": registry,
        }
        fn = load(
            "python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py",
            "build_deduped_cuda_graph",
            ns,
            owner="DedupedCudaGraphMixin",
        )
        self.assertIsNone(fn(NS()))
        native_graph.assert_not_called()
        registry.assert_not_called()
        ns["is_maca"] = lambda: False
        self.assertIs(fn(NS()), registry.return_value)
        native_graph.assert_called_once_with(keep_graph=True)
        registry.assert_called_once_with()


    def test_dp_local_cache_reuses_shape_and_preserves_hidden_size(self):
        dp = NS()
        ns = {
            "torch": TorchStub,
            "get_flags": lambda: NS(dp=dp),
            "use_symmetric_memory": lambda *a, **kw: nullcontext(),
        }
        wrapper = load(
            "python/sglang/srt/layers/dp_attention.py", "_DpGatheredBufferWrapper", ns
        )
        wrapper.set_metadata(16, DType.BF16, "cuda:3")
        wrapper.set_dp_buffer_len(8, 4, False)
        first = wrapper.get_local_dp_buffer(None, hidden_size=8)
        self.assertIs(first, wrapper.get_local_dp_buffer(None, hidden_size=8))
        narrower = wrapper.get_local_dp_buffer(None, hidden_size=4)
        self.assertEqual(narrower.shape, (4, 4))
        self.assertIsNot(first, narrower)
        wrapper.set_dp_buffer_len(8, 6, False)
        self.assertEqual(wrapper.get_local_dp_buffer(None, hidden_size=4).shape, (6, 4))


    def test_dp_metadata_change_invalidates_both_buffers(self):
        dp = NS()
        ns = {
            "torch": TorchStub,
            "get_flags": lambda: NS(dp=dp),
            "use_symmetric_memory": lambda *a, **kw: nullcontext(),
        }
        wrapper = load(
            "python/sglang/srt/layers/dp_attention.py", "_DpGatheredBufferWrapper", ns
        )
        wrapper.set_metadata(16, DType.BF16, "cuda:3")
        wrapper.set_dp_buffer_len(8, 4, False)
        old = wrapper.get_global_dp_buffer(None)
        wrapper.get_local_dp_buffer(None)
        wrapper.set_metadata(32, DType.BF16, "cuda:1")
        new = wrapper.get_global_dp_buffer(None)
        self.assertIsNot(old, new)
        self.assertEqual((new.shape, new.device), ((8, 32), "cuda:1"))
        self.assertIsNone(wrapper._local_dp_buffer)


    def test_dense_reduce_scatter_switch_preserves_sparse_routing(self):
        variant = NS(
            SEQUENCE_PARALLEL=object(),
            INPUT_SCATTERED=object(),
            CONTEXT_PARALLEL=object(),
        )
        parallel = NS(tp_size=2, attn_cp_size=1, attn_dp_size=1)
        strategy = NS(boundary_reduction="rs+rsv")
        layout = NS(sharded_over=lambda *a, **kw: "rows")
        output = lambda rows, **kw: NS(
            rs=kw.get("leaves_for_reduce_scatter", False),
            rsv=kw.get("leaves_for_reduce_scatterv", False),
        )
        ns = {
            "os": os,
            "get_parallel": lambda: parallel,
            "get_exec": lambda: NS(comm=strategy),
            "NORM_READ": object(),
            "ADD": NS(at_producer=False, can_defer_across_layers=True),
            "ProducerReduction": NS(SCOPED=object()),
            "BatchVariant": variant,
            "Layout": layout,
            "SumGroup": NS(MOE_OUTPUT="moe", TP="tp"),
            "_generic_prefill_cp_shards_tokens": lambda: False,
            "_rows": lambda v: ((), "attn", "local", "full"),
            "is_moe_input_scattered_across_dp_ranks": lambda: False,
            "enable_moe_dense_fully_dp": lambda: False,
            "StageInput": lambda *a, **kw: None,
            "StageOutput": output,
            "StageDecl": lambda i, o: o,
        }
        fn = load(
            "python/sglang/srt/layers/layer_boundary/factories.py", "_ffn_decl", ns
        )
        for disabled in (False, True):
            with patch.dict(
                os.environ, {"MX_DISABLE_DENSE_REDUCE_SCATTER": "1" if disabled else ""}
            ):
                for sparse in (False, True):
                    result, _, _ = fn(object(), sparse=sparse)
                    self.assertEqual(
                        (result.rs, result.rsv), (not disabled or sparse,) * 2
                    )


    def test_padding_only_idle_detection_excludes_real_and_speculative_batches(self):
        ns = {}
        fn = load(
            "python/sglang/srt/model_executor/model_runner.py",
            "_should_reroute_padding_only_extend_to_idle",
            ns,
            owner="ModelRunner",
        )
        fb = NS(
            global_num_tokens_cpu=[4, 0],
            dp_padding_mode=NS(is_max_len=lambda: True),
            is_extend_in_batch=True,
            forward_mode=NS(is_extend=lambda **kw: True),
            _original_forward_mode=NS(is_idle=lambda: True),
            batch_size=0,
            spec_info=None,
            seq_lens=NS(numel=lambda: 0),
            req_pool_indices=NS(numel=lambda: 0),
            input_ids=NS(numel=lambda: 4),
        )
        self.assertTrue(fn(None, fb))
        for name, value in (
            ("batch_size", 1),
            ("spec_info", object()),
            ("input_ids", NS(numel=lambda: 0)),
        ):
            with patch.object(fb, name, value):
                self.assertFalse(fn(None, fb))


if __name__ == "__main__":
    unittest.main()
