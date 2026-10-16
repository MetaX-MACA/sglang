"""CPU contract checks for the MACA 0.5.21 ports, using stubbed GPU operators.

Run directly with Python. These tests execute the production control flow via
AST extraction so GPU libraries are not needed; they do not check GPU numerics.
"""

import ast
import inspect
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

    def test_dsv4_query_dtype_preserves_native_fp8(self):
        path = ROOT / "python/sglang/srt/layers/attention/dsv4/indexer.py"
        tree = ast.parse(path.read_text())
        owner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "C4Indexer"
        )
        fn = next(
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name == "compute_q"
        )
        start = next(
            i
            for i, node in enumerate(fn.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "index_cache_type"
                for t in node.targets
            )
        )
        # Execute the selection and quantizer call, omitting the projection setup.
        body = ast.FunctionDef(
            name="select_query",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=fn.body[start:],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[]))
        for maca, expected in ((False, DType.FP8), (True, DType.INT8)):
            quantize = Mock()
            ns = {
                "is_maca": lambda: maca,
                "envs": NS(SGLANG_DSV4_INDEX_CACHE_TYPE=NS(get=lambda: "int8")),
                "torch": TorchStub,
                "q": object(),
                "weight": object(),
                "positions": object(),
                "self": NS(weight_scale=1.0, freqs_cis=object()),
                "fused_q_indexer_rope_hadamard_quant": quantize,
            }
            exec(compile(module, str(path), "exec"), ns)
            ns["select_query"]()
            self.assertIs(quantize.call_args.args[-1], expected)

    def test_w8a8_preserves_batched_prequantized_input_shape(self):
        q = Mock(shape=(2, 3, 128))
        scale = Mock(shape=(2, 3, 1))
        output = Mock()
        gemm = Mock(return_value=output)
        fn = load(
            "python/sglang/srt/layers/quantization/w8a8_int8.py",
            "apply",
            {
                "use_intel_amx_backend": lambda _: False,
                "_is_cpu_arm64": False,
                "int8_scaled_mm": gemm,
            },
            owner="W8A8Int8LinearMethod",
        )
        layer = NS(weight=NS(shape=(128, 256)), weight_scale=object())
        fn(None, layer, (DType.BF16, q, scale))
        q.view.assert_called_once_with(-1, 128)
        scale.view.assert_called_once_with(-1, 1)
        self.assertIs(gemm.call_args.args[0], q.view.return_value)
        output.view.assert_called_once_with([2, 3, 256])

    def test_moe_runner_conversion_matches_fused_execution_signature(self):
        import inspect
        import copy

        runner = ast.parse((ROOT / "python/sglang/srt/layers/moe/moe_runner/triton.py").read_text())
        source = ast.parse((ROOT / "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py").read_text())
        sequence = copy.deepcopy(next(n for n in source.body if isinstance(n, ast.FunctionDef)
                                     and n.name == "_fused_moe_kernel_sequence"))
        sequence.body = [ast.Pass()]
        namespace = {}
        exec("from __future__ import annotations\n" + ast.unparse(sequence), namespace)
        call = next(n for n in ast.walk(runner) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == sequence.name)
        inspect.signature(namespace[sequence.name]).bind(
            *[None] * len(call.args), **{k.arg: None for k in call.keywords})
        converter = next(n for n in runner.body if isinstance(n, ast.FunctionDef)
                         and n.name == "pre_permute_standard_to_triton")
        unpack = next(n for n in converter.body if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == "_prepare_fused_moe_run")
        prepare = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_prepare_fused_moe_run")
        returned = next(n for n in prepare.body if isinstance(n, ast.Return))
        self.assertEqual(len(unpack.targets[0].elts), len(returned.value.elts))

    def test_moe_activation_reaches_kernel_sequence(self):
        for name, inplace in (
            ("inplace_fused_experts", True),
            ("outplace_fused_experts", False),
        ):
            impl = Mock()
            fn = load(
                "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py",
                name,
                {
                    "register_custom_op": lambda *a, **kw: lambda fn: fn,
                    "fused_experts_impl": impl,
                },
            )
            fn(*[object() for _ in range(5)], activation="gelu")
            self.assertEqual(impl.call_args.args[7:9], (inplace, "gelu"))

    def test_dsa_index_cache_dtype_selection_and_storage(self):
        source = "python/sglang/srt/layers/attention/dsa/utils.py"
        for maca, configured, expected in (
            (True, None, "int8"),
            (True, "fp8_e4m3fn", "fp8"),
            (True, "bf16", "bf16"),
            (False, None, "fp8"),
            (False, "int8", "int8"),
        ):
            with self.subTest(maca=maca, configured=configured):
                ns = {
                    "os": os,
                    "is_maca": lambda: maca,
                    "envs": NS(
                        MX_NSA_INDEXER_CACHE_DTYPE=NS(get=lambda: configured or "int8")
                    ),
                }
                selector = load(source, "get_nsa_index_cache_dtype", ns)
                env = (
                    {}
                    if configured is None
                    else {"MX_NSA_INDEXER_CACHE_DTYPE": configured}
                )
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(selector(), expected)

        storage = load(
            source, "get_nsa_index_cache_storage_dtype", {"torch": TorchStub}
        )
        self.assertIs(storage("int8"), DType.INT8)
        self.assertIs(storage("fp8"), DType.UINT8)
        self.assertIs(storage("bf16"), DType.BF16)

    def test_precomputed_flashmla_contract_in_fused_and_fallback_replay(self):
        precomputed_type = load(
            "python/sglang/srt/layers/attention/dsa/dsa_backend_mtp_precompute.py",
            "PrecomputedMetadata",
            {"dataclass": dataclass, "__name__": __name__},
        )
        copy_buffers = load(
            "python/sglang/srt/layers/attention/dsa/dsa_metadata_manager.py",
            "_copy_base_replay_buffers",
            {"_is_hip": False},
            owner="DSAMetadataManagementMixin",
        )
        for has_flashmla in (False, True):
            for fused in (False, True):
                with self.subTest(has_flashmla=has_flashmla, fused=fused):
                    values = {
                        field.name: MagicMock()
                        for field in fields(precomputed_type)
                        if field.name != "flashmla_metadata"
                    }
                    values.update(
                        max_len=4,
                        max_seqlen_k=4,
                        seqlens_expanded_size=2,
                        real_page_table=None,
                    )
                    if has_flashmla:
                        values["flashmla_metadata"] = Mock()
                    precomputed = precomputed_type(**values)
                    if not has_flashmla:
                        self.assertIsNone(precomputed.flashmla_metadata)

                    metadata = MagicMock()
                    kernel = Mock(
                        side_effect=None if fused else RuntimeError("force fallback")
                    )
                    module = "sglang.kernels.ops.attention.fused_metadata_copy"
                    mode = NS(is_decode_or_idle=lambda: True)
                    with (
                        patch.dict(
                            sys.modules, {module: NS(fused_metadata_copy_cuda=kernel)}
                        ),
                        patch("builtins.print") as warning,
                    ):
                        copy_buffers(NS(), 2, metadata, precomputed, mode)
                    kernel.assert_called_once()
                    self.assertEqual(warning.call_count, 0 if fused else 1)
                    if fused:
                        metadata.cache_seqlens_int32.copy_.assert_not_called()
                    else:
                        metadata.cache_seqlens_int32.copy_.assert_called_once_with(
                            precomputed.cache_seqlens
                        )
                        flashmla_copy = (
                            metadata.flashmla_metadata.slice.return_value.copy_
                        )
                        if has_flashmla:
                            flashmla_copy.assert_called_once_with(
                                precomputed.flashmla_metadata
                            )
                        else:
                            flashmla_copy.assert_not_called()

    def test_maca_mqa_schedule_does_not_read_device_scalar(self):
        context_lens = Mock()
        context_lens.item.side_effect = RuntimeError("host sync during capture")
        schedule = object()

        def backend(lengths, block_kv, num_sms, *, workers_per_sm=None):
            if workers_per_sm is None:
                lengths.item()
            self.assertEqual((block_kv, num_sms, workers_per_sm), (64, 80, 8))
            return schedule

        ns = {"is_maca": lambda: True, "inspect": inspect}
        fn = load(
            "python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py",
            "get_paged_mqa_logits_metadata",
            ns,
        )
        with patch.dict(
            sys.modules, {"deep_gemm": NS(get_paged_mqa_logits_metadata=backend)}
        ):
            self.assertIs(fn(context_lens, 64, 80), schedule)
        context_lens.item.assert_not_called()

    def test_nvidia_mqa_schedule_preserves_three_argument_api(self):
        schedule = object()
        lengths = object()

        def backend(context_lens, block_kv, num_sms):
            self.assertIs(context_lens, lengths)
            self.assertEqual((block_kv, num_sms), (64, 80))
            return schedule

        signature = Mock(side_effect=AssertionError("CUDA must not inspect MACA API"))
        ns = {"is_maca": lambda: False, "inspect": NS(signature=signature)}
        fn = load(
            "python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py",
            "get_paged_mqa_logits_metadata",
            ns,
        )
        with patch.dict(
            sys.modules, {"deep_gemm": NS(get_paged_mqa_logits_metadata=backend)}
        ):
            self.assertIs(fn(lengths, 64, 80), schedule)
        signature.assert_not_called()

    def test_legacy_maca_mqa_schedule_preserves_backend_defaults(self):
        schedule = object()
        lengths = object()

        def backend(context_lens, block_kv, num_sms, blocks_per_split=4):
            self.assertIs(context_lens, lengths)
            self.assertEqual((block_kv, num_sms, blocks_per_split), (64, 80, 4))
            return schedule

        fn = load(
            "python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py",
            "get_paged_mqa_logits_metadata",
            {"is_maca": lambda: True, "inspect": inspect},
        )
        with patch.dict(
            sys.modules, {"deep_gemm": NS(get_paged_mqa_logits_metadata=backend)}
        ):
            self.assertIs(fn(lengths, 64, 80), schedule)

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

    def test_bf16_paged_kv_has_correct_physical_size(self):
        layout = NS(V4=NS(page_bytes=lambda size: ((size * 584 + 575) // 576) * 576))
        ns = {
            "torch": TorchStub,
            "KVLayout": layout,
            "KVCache": object,
            "ceil_div": lambda a, b: (a + b - 1) // b,
        }
        cls = load(
            "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
            "DeepSeekV4SingleKVPool",
            ns,
        )
        for dtype in (DType.BF16, DType.FP8):
            pool = cls.__new__(cls)
            pool.dtype = dtype
            pool.store_dtype = DType.BF16 if dtype is DType.BF16 else DType.UINT8
            pool.qk_nope_head_dim, pool.qk_rope_head_dim = 448, 64
            pool.quantize_block_size, pool.scale_pad = 64, 1
            pool.rope_storage_dtype = DType.BF16
            pool.kv_layout, pool.page_size, pool.device = layout.V4, 64, "cuda:3"
            buf = pool.create_buffer(num_pages=3)
            expected = (
                64 * 1024 if dtype is DType.BF16 else ((64 * 584 + 575) // 576) * 576
            )
            self.assertEqual(buf.nbytes, 3 * expected)
            self.assertEqual(pool.bytes_per_page_padded, expected)
            self.assertEqual(buf.dtype, pool.store_dtype)

    def test_dsv4_bf16_cache_view_uses_elements_not_fp8_bytes(self):
        path = "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
        ns = {"torch": TorchStub}
        swa_bytes = load(
            path, "get_swa_key_bytes_per_token", ns, owner="DeepSeekV4TokenToKVPool"
        )
        extra_bytes = load(
            path, "get_extra_key_bytes_per_token", ns, owner="DeepSeekV4TokenToKVPool"
        )
        for dtype, expected_bytes in ((DType.BF16, 1024), (DType.FP8, 584)):
            with self.subTest(dtype=dtype):
                pool = NS(
                    dtype=dtype,
                    uniform_fp8=False,
                    qk_nope_head_dim=448,
                    qk_rope_head_dim=64,
                    kv_layout=NS(bytes_per_token=584),
                    layer_mapping={
                        0: (None, None, NS(kv_cache_total_dim=expected_bytes))
                    },
                )
                view_dim = expected_bytes // dtype.itemsize
                self.assertEqual(swa_bytes(pool) // dtype.itemsize, view_dim)
                self.assertEqual(extra_bytes(pool, 0) // dtype.itemsize, view_dim)
                self.assertEqual(view_dim, 512 if dtype is DType.BF16 else 584)

    def test_unified_fp8_accounting_includes_rope_pool(self):
        ns = {"get_tensor_size_bytes": lambda b: b.nbytes}
        fn = load(
            "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
            "get_kv_size_bytes",
            ns,
            owner="DeepSeekV4UnifiedKVPool",
        )
        pool = NS(fp8=True, kv_buffer=[NS(nbytes=512)], kv_buffer_rope=[NS(nbytes=128)])
        self.assertEqual(fn(pool), 640)

    def test_stage_accounting_handles_missing_c4_and_low_ratio_pools(self):
        ns = {}
        fn = load(
            "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
            "get_kv_size_bytes",
            ns,
            owner="DeepSeekV4TokenToKVPool",
        )
        pool = NS(
            _unified_kv=False,
            swa_kv_pool=NS(get_kv_size_bytes=lambda: 10),
            kv_pools={1: NS(get_kv_size_bytes=lambda: 20), 4: None},
            index_pools={1: NS(get_kv_size_bytes=lambda: 3)},
        )
        self.assertEqual(fn(pool), 33)

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

    def test_mctlass_rejects_unsupported_output_semantics(self):
        path = ROOT / "python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py"
        tree = ast.parse(path.read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "invoke_fused_moe_kernel"
        )
        assignment = next(
            n
            for n in fn.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "mctlass_output_supported"
                for t in n.targets
            )
        )
        code = compile(
            ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"
        )
        ns = dict(
            bias=None,
            block_shape=None,
            C=NS(is_contiguous=lambda: True, dtype=TorchStub.bfloat16),
            torch=TorchStub,
            c_sorted=False,
            fuse_add_to_output=False,
            mask_output=False,
            lora_preserve_base=False,
            fuse_sum_all_reduce=False,
            fuse_swiglu=False,
        )
        exec(code, ns)
        self.assertTrue(ns["mctlass_output_supported"])
        for name, value in (
            ("bias", object()),
            ("block_shape", [128, 128]),
            ("C", NS(is_contiguous=lambda: False)),
            ("c_sorted", True),
            ("mask_output", True),
            ("fuse_add_to_output", True),
            ("lora_preserve_base", True),
            ("fuse_sum_all_reduce", True),
            ("fuse_swiglu", True),
        ):
            with patch.dict(ns, {name: value}):
                exec(code, ns)
                self.assertFalse(ns["mctlass_output_supported"], name)

    def test_mctlass_int8_dispatch_preserves_scales_and_route_weights(self):
        class Tensor:
            dtype = DType.BF16
            shape = (5, 2, 8)

            @property
            def ndim(self):
                return len(self.shape)

            def stride(self, dim):
                return 1

            def is_contiguous(self):
                return True

            def size(self, dim):
                return self.shape[dim]

            def view(self, *shape):
                return self

            def contiguous(self):
                return self

            def numel(self):
                return 10

        a, b, c = Tensor(), Tensor(), Tensor()
        a.shape = (5, 8)
        b.shape = (3, 8, 8)
        weights, ids = Tensor(), Tensor()
        scale_a, scale_b = Tensor(), Tensor()
        scale_a.shape = (5, 1)
        scale_b.shape = (3, 8)
        gemm = Mock()
        triton_kernel = MagicMock()
        ns = {
            "torch": NS(bfloat16=DType.BF16, float16=object()),
            "tl": NS(bfloat16=object()),
            "triton": NS(cdiv=lambda a, b: (a + b - 1) // b),
            "_is_maca": True,
            "enable_mctlass_fused_moe": True,
            "enable_mctlass_fused_moe_python_api": True,
            "per_token_quant_int8": lambda x: (a, scale_a),
            "envs": NS(
                ENABLE_MACA_SGLANG_FUSED_MOE_MCTLASS_W4A16=NS(get=lambda: False)
            ),
            "gemm": gemm,
            "fused_moe_kernel": triton_kernel,
            "is_arch_support_pdl": lambda: False,
        }
        fn = load(
            "python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py",
            "invoke_fused_moe_kernel",
            ns,
        )
        for python_api in (True, False):
            with self.subTest(python_api=python_api):
                ns["enable_mctlass_fused_moe_python_api"] = python_api
                gemm.reset_mock()
                triton_kernel.reset_mock()
                fn(
                    a, b, None, c, None, scale_b, None, weights, ids,
                    Tensor(), Tensor(), Tensor(), True, 2,
                    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 8},
                    object(),
                    use_fp8_w8a8=False,
                    use_int8_w8a8=True,
                    use_int8_w8a16=False,
                    use_int4_w4a8=False,
                    use_int4_w4a16=False,
                    per_channel_quant=True,
                    filter_expert=True,
                )
                if python_api:
                    args = gemm.call_args.args
                    self.assertIs(args[9], scale_a)
                    self.assertIs(args[10], scale_b)
                    self.assertIs(args[12], weights)
                    self.assertTrue(args[-1])
                    self.assertTrue(gemm.call_args.kwargs["filter_expert"])
                    triton_kernel.__getitem__.assert_not_called()
                else:
                    gemm.assert_not_called()
                    launch = triton_kernel.__getitem__.return_value
                    launch.assert_called_once()
                    self.assertIs(launch.call_args.args[6], scale_a)
                    self.assertIs(launch.call_args.args[7], scale_b)
                    self.assertIs(launch.call_args.args[8], weights)

    def test_mctlass_mxfp4_queries_each_stage_and_aligns_down_tile(self):
        path = (
            ROOT / "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
        )
        tree = ast.parse(path.read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_fused_moe_kernel_sequence"
        )
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get_kernel_m"
        ]
        packed = [
            n for n in calls if any(k.arg == "is_b_e2m1fp4x2" for k in n.keywords)
        ]
        self.assertEqual(len(packed), 2)
        self.assertTrue(
            all(any(k.arg == "mul_weight" for k in n.keywords) for n in packed)
        )
        # Execute the production re-alignment branch with intentionally different tiles.
        down_branch = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "down_m != stage1_config['BLOCK_SIZE_M']"
        )
        align = Mock(return_value=("tokens", "experts", "padded"))
        ns = dict(
            down_m=64,
            stage1_config={"BLOCK_SIZE_M": 16},
            moe_align_block_size=align,
            topk_ids=object(),
            E=3,
        )
        exec(
            compile(ast.Module(body=[down_branch], type_ignores=[]), str(path), "exec"),
            ns,
        )
        self.assertEqual(ns["sorted_token_ids"], "tokens")
        align.assert_called_once_with(ns["topk_ids"], 64, 3)

    def test_triton_kernels_do_not_capture_host_platform_flags(self):
        path = ROOT / "python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py"
        tree = ast.parse(path.read_text())
        platform_probes = {"is_maca", "is_cuda", "is_hip", "is_cpu"}
        host_flags = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in platform_probes
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or not any(
                isinstance(decorator, ast.Attribute)
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "jit"
                for decorator in fn.decorator_list
            ):
                continue
            referenced_flags = {
                node.id
                for node in ast.walk(fn)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            } & host_flags
            self.assertFalse(
                referenced_flags,
                f"{fn.name} captures non-constexpr host flags: {referenced_flags}",
            )

    def test_quantizers_keep_empty_groups_on_original_device(self):
        ns = {
            "torch": TorchStub,
            "fp8_dtype": DType.FP8,
            "_mcoplib_per_token_cast_to_fp8": lambda *a: None,
            "per_token_cast_to_fp8": lambda x: (None, None),
        }
        fn = load(
            "python/sglang/kernels/ops/quantization/fp8_kernel.py",
            "per_token_cast_to_fp8_3d",
            ns,
        )
        x = NS(shape=(0, 2, 256), device="cuda:3", dim=lambda: 3)
        with (
            patch.object(
                TorchStub,
                "empty_like",
                create=True,
                side_effect=lambda x, dtype: TorchStub.empty(x.shape, dtype, x.device),
            ),
            patch.object(TorchStub, "float32", DType.BF16, create=True),
        ):
            q, scales = fn(x)
        self.assertEqual((q.shape, scales.shape), ((0, 2, 256), (0, 2, 2)))
        self.assertEqual((q.device, scales.device), ("cuda:3", "cuda:3"))


    def test_minimax_composite_pool_quantization_policy(self):
        identity = NS(name="unquantized")
        fn = load(
            "python/sglang/srt/mem_cache/memory_pool.py",
            "get_kv_cache_quant_method",
            {
                "is_float4_e2m1fn_x2": lambda dtype: dtype == "fp4",
                "UnquantizedKVCacheMethod": lambda: identity,
            },
            owner="KVCache",
        )
        quantized = NS(name="fp8")
        self.assertIs(fn(NS(dtype="bf16", main_pool=NS(quant_method=quantized))), quantized)
        self.assertIs(fn(NS(dtype="bf16")), identity)
        self.assertIs(fn(NS(dtype="bf16", quant_method=identity)), identity)
        with self.assertRaisesRegex(ValueError, "does not expose"):
            fn(NS(dtype="fp4"))

    def test_minimax_block_tuning_is_maca_only(self):
        source = ROOT / "python/sglang/srt/layers/attention/minimax_sparse_backend.py"
        tree = ast.parse(source.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "MiniMaxSparseAttnBackend")
        init = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                    and n.name == "__init__")
        body = [n for n in init.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute)
                        and t.attr in ("block_size_q", "block_size_k")
                        for t in n.targets)]
        self.assertEqual(len(body), 2)
        code = compile(ast.Module(body=body, type_ignores=[]), str(source), "exec")
        for maca, expected in ((False, (1, 128)), (True, (2, 32))):
            instance = NS()
            exec(code, {"self": instance, "is_maca": lambda: maca,
                        "sparse_cfg": {"sparse_block_size": 128}})
            self.assertEqual((instance.block_size_q, instance.block_size_k), expected)


    def test_kernel_version_gate_preserves_platform_minimums(self):
        source = ROOT / "python/sglang/srt/entrypoints/engine.py"
        tree = ast.parse(source.read_text())
        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "_is_cuda"
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "assert_pkg_version"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "sglang-kernel"
                for call in ast.walk(node)
            )
        ]
        self.assertEqual(len(guards), 1)
        code = compile(ast.Module(body=guards, type_ignores=[]), str(source), "exec")
        for cuda, maca, expected in (
            (True, True, "0.4.6.post1"),
            (True, False, "0.4.7"),
            (False, False, None),
            (False, True, None),
        ):
            check = Mock()
            exec(code, {"_is_cuda": cuda, "is_maca": lambda: maca,
                        "assert_pkg_version": check})
            if expected is None:
                check.assert_not_called()
            else:
                check.assert_called_once()
                self.assertEqual(check.call_args.args[:2], ("sglang-kernel", expected))


if __name__ == "__main__":
    unittest.main()
