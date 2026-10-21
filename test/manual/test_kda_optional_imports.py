"""Check optional kernel switches without requiring Triton, mcoplib, or GPUs."""

import ast
import builtins
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "python/sglang/kernels/ops"
CONFIG = OPS / "attention/fla/kda_opt_config.py"
FILES = (
    "attention/fla/chunk_delta_h.py",
    "attention/fla/chunk_intra.py",
    "attention/fla/fused_sigmoid_gating_recurrent.py",
    "attention/fla/kda.py",
    "mamba/causal_conv1d_triton.py",
)


class OptionalImportTests(unittest.TestCase):
    def load_switch(self):
        spec = importlib.util.spec_from_file_location("kda_switch_under_test", CONFIG)
        module = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "triton", "mcoplib", "sglang"):
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            spec.loader.exec_module(module)
        return module._c500_kda_opt_enabled

    def test_disabled_needs_no_kernel_dependencies(self):
        enabled = self.load_switch()
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(enabled())
            for value in ("0", "false", "off", "no"):
                os.environ["MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS"] = value
                os.environ["MX_USE_MCOPLIB_KDA"] = "1"
                self.assertFalse(enabled())

    def test_enabled_and_mutual_exclusion(self):
        enabled = self.load_switch()
        with patch.dict(os.environ, {}, clear=True):
            for value in ("1", "true", "ON", "Yes"):
                os.environ["MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS"] = value
                self.assertTrue(enabled())
            os.environ["MX_USE_MCOPLIB_KDA"] = "1"
            with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
                enabled()

    def test_wrappers_only_import_heavy_module_inside_enabled_guard(self):
        for relative in FILES:
            with self.subTest(file=relative):
                tree = ast.parse((OPS / relative).read_text())
                self.assertTrue(any(
                    isinstance(node, ast.ImportFrom)
                    and node.module and node.module.endswith("kda_opt_config")
                    and any(alias.name == "_c500_kda_opt_enabled" for alias in node.names)
                    for node in tree.body
                ))
                heavy_imports = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and node.module and node.module.endswith("kda_opt_c500")
                ]
                self.assertTrue(heavy_imports)
                guarded_imports = []
                for node in ast.walk(tree):
                    if (isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                        and isinstance(node.test.func, ast.Name)
                        and node.test.func.id == "_c500_kda_opt_enabled"):
                        guarded_imports.extend(
                            child for body in node.body for child in ast.walk(body)
                            if isinstance(child, ast.ImportFrom)
                            and child.module and child.module.endswith("kda_opt_c500")
                        )
                self.assertEqual({id(n) for n in heavy_imports},
                                 {id(n) for n in guarded_imports})


if __name__ == "__main__":
    unittest.main()
