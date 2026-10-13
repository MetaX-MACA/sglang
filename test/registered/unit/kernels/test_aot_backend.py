"""Exercise AOT backend selection without loading GPU libraries."""

import ast
import importlib.util
import os
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
AOT = ROOT / "python/sglang/kernels/aot"
INIT = AOT / "python/sgl_kernel/__init__.py"
VERSION = runpy.run_path(str(INIT.parent / "version.py"))["__version__"]
spec = importlib.util.spec_from_file_location(
    "ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
ci_register.register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def torch_stub(version):
    torch = ModuleType("torch")
    torch.__version__ = version
    torch.version = SimpleNamespace(cuda=None, hip=None)
    torch.Tensor = object
    torch.cuda = Mock()
    torch.cuda.is_available.return_value = False
    torch.ops = Mock()
    return torch


class TestAotBackend(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("SGLANG_USE_MCOPLIB", None)

    def load_kernel(self, torch, *, missing_mcoplib=False, missing_native=False):
        # Stub unrelated exports, but execute __init__.py and moe.py in full.
        modules = {"torch": torch}
        for node in ast.walk(ast.parse(INIT.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module.startswith(
                "sgl_kernel."
            ):
                module = modules.setdefault(node.module, ModuleType(node.module))
                module.__all__ = []
                for alias in node.names:
                    if alias.name != "*":
                        setattr(module, alias.name, Mock())
        modules.pop("sgl_kernel.moe")
        modules.pop("sgl_kernel.platform")
        modules["sgl_kernel.version"].__version__ = VERSION
        modules["sgl_kernel.debug_utils"].maybe_wrap_debug_kernel = lambda fn, _: fn
        native = modules["sgl_kernel.load_utils"]._load_architecture_specific_ops
        native.return_value = ModuleType("common_ops")
        if missing_native:
            native.side_effect = ImportError("common_ops unavailable")
        modules["mcoplib"] = None if missing_mcoplib else ModuleType("mcoplib")
        for name in ("_C", "sgl_kernel", "sgl_grouped_gemm_cuda"):
            modules[f"mcoplib.{name}"] = None if missing_mcoplib else ModuleType(name)
        spec = importlib.util.spec_from_file_location("sgl_kernel", INIT)
        kernel = importlib.util.module_from_spec(spec)
        modules["sgl_kernel"] = kernel
        with patch.dict(sys.modules, modules):
            platform_spec = importlib.util.spec_from_file_location(
                "sgl_kernel.platform", INIT.parent / "platform.py"
            )
            platform = importlib.util.module_from_spec(platform_spec)
            sys.modules["sgl_kernel.platform"] = platform
            platform_spec.loader.exec_module(platform)
            moe_spec = importlib.util.spec_from_file_location(
                "sgl_kernel.moe", INIT.parent / "moe.py"
            )
            moe = importlib.util.module_from_spec(moe_spec)
            sys.modules["sgl_kernel.moe"] = moe
            moe_spec.loader.exec_module(moe)
            # This MACA source tree comments out the non-MACA FP8 export.
            moe.fp8_blockwise_scaled_grouped_mm = Mock()
            spec.loader.exec_module(kernel)
            kernel.moe_sum("input", "output")
        torch.ops.sgl_kernel.moe_sum.default.assert_called_once_with("input", "output")
        return kernel, native, modules

    def test_runtime_defaults_and_overrides(self):
        for version, flag, mcoplib in (
            ("2.10.0+metax", None, True),
            ("2.13.0+cu130", None, False),
            ("2.10.0+metax", "0", False),
            ("2.10.0+metax", "false", False),
            ("2.13.0+cu130", "1", True),
            ("2.13.0+cu130", "true", True),
        ):
            with self.subTest(version=version, flag=flag):
                if flag is None:
                    os.environ.pop("SGLANG_USE_MCOPLIB", None)
                else:
                    os.environ["SGLANG_USE_MCOPLIB"] = flag
                kernel, native, modules = self.load_kernel(torch_stub(version))
                if mcoplib:
                    native.assert_not_called()
                    self.assertIs(kernel.common_ops, modules["mcoplib.sgl_kernel"])
                else:
                    native.assert_called_once_with()
                    self.assertIs(kernel.common_ops, native.return_value)

    def test_native_backend_import_without_mcoplib(self):
        os.environ["SGLANG_USE_MCOPLIB"] = "0"
        torch = torch_stub("2.10.0+metax")
        kernel, _, _ = self.load_kernel(torch, missing_mcoplib=True)
        with patch.dict(sys.modules, {"torch": torch, "mcoplib": None}):
            with self.assertRaises(ImportError):
                kernel.fused_moe_gate_opt(*([None] * 10))

    def test_maca_fused_gate_uses_mcoplib_op(self):
        torch = torch_stub("2.10.0+metax")
        kernel, _, modules = self.load_kernel(torch)
        op = ModuleType("mcoplib.op")
        op.fused_moe_gate_opt = Mock(return_value=7)
        with patch.dict(
            sys.modules,
            {"torch": torch, "mcoplib": modules["mcoplib"], "mcoplib.op": op},
        ):
            self.assertEqual(kernel.fused_moe_gate_opt(*range(10)), 7)
        op.fused_moe_gate_opt.assert_called_once_with(*range(10))

    def test_missing_selected_mcoplib_raises(self):
        os.environ["SGLANG_USE_MCOPLIB"] = "1"
        with self.assertRaises(ImportError):
            self.load_kernel(torch_stub("2.10.0+metax"), missing_mcoplib=True)

    def test_missing_selected_native_backend_raises(self):
        os.environ["SGLANG_USE_MCOPLIB"] = "0"
        with self.assertRaisesRegex(ImportError, "common_ops unavailable"):
            self.load_kernel(torch_stub("2.10.0+metax"), missing_native=True)

    def test_wrappers_build_without_torch(self):
        os.environ["SGLANG_USE_MCOPLIB"] = "1"
        setuptools = Mock()
        with patch.dict(
            sys.modules,
            {"torch": None, "flashinfer": None, "setuptools": setuptools},
        ):
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path(str(AOT / "setup.py"), run_name="__main__")
        self.assertEqual(exc.exception.code, 0)
        self.assertTrue(
            setuptools.setup.call_args.kwargs["version"].endswith(".torchunknown")
        )
        self.assertNotIn("ext_modules", setuptools.setup.call_args.kwargs)

    def test_build_defaults_and_overrides(self):
        for version, flag, mcoplib in (
            ("2.10.0+metax", None, True),
            ("2.13.0+cu130", None, False),
            ("2.10.0+metax", "0", False),
            ("2.10.0+metax", "false", False),
            ("2.13.0+cu130", "1", True),
            ("2.13.0+cu130", "true", True),
        ):
            with self.subTest(version=version, flag=flag):
                if flag is None:
                    os.environ.pop("SGLANG_USE_MCOPLIB", None)
                else:
                    os.environ["SGLANG_USE_MCOPLIB"] = flag
                setuptools = Mock()
                cpp = Mock(COMMON_NVCC_FLAGS=[])
                torch = torch_stub(version)
                torch.utils = SimpleNamespace(cpp_extension=cpp)
                modules = {
                    "torch": torch,
                    "torch.utils.cpp_extension": cpp,
                    "setuptools": setuptools,
                    "flashinfer": (
                        None
                        if mcoplib
                        else SimpleNamespace(__file__=str(AOT / "flashinfer.py"))
                    ),
                }
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(sys, "argv", ["setup.py"]),
                    patch.dict(os.environ, {"MACA_AI_VERSION": "0.0.0.0"}),
                ):
                    try:
                        runpy.run_path(str(AOT / "setup.py"), run_name="__main__")
                    except SystemExit as exc:
                        self.assertEqual(exc.code, 0)
                kwargs = setuptools.setup.call_args.kwargs
                self.assertEqual(
                    kwargs["version"],
                    VERSION + "+maca0.0.0.0.torch" + version.split("+")[0],
                )
                if mcoplib:
                    self.assertNotIn("ext_modules", kwargs)
                    cpp.CUDAExtension.assert_not_called()
                else:
                    self.assertEqual(len(kwargs["ext_modules"]), 1)
                    extension = cpp.CUDAExtension.call_args.kwargs
                    self.assertEqual(extension["name"], "sgl_kernel.common_ops")
                    self.assertTrue(
                        all((AOT / src).is_file() for src in extension["sources"])
                    )


if __name__ == "__main__":
    unittest.main()
