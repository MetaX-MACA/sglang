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


if __name__ == "__main__":
    unittest.main()
