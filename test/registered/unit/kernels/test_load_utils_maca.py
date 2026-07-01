import os
import sys
import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[4]
LOAD_UTILS_PATH = REPO_ROOT / "sgl-kernel" / "python" / "sgl_kernel" / "load_utils.py"

spec = importlib.util.spec_from_file_location("sgl_kernel_load_utils", LOAD_UTILS_PATH)
load_utils = importlib.util.module_from_spec(spec)
sys.modules["sgl_kernel_load_utils"] = load_utils
assert spec.loader is not None
spec.loader.exec_module(load_utils)


class LoadUtilsMacaTest(unittest.TestCase):
    def test_find_cuda_home_uses_maca_cu_bridge(self):
        with TemporaryDirectory() as tmp_dir:
            maca_path = Path(tmp_dir) / "maca"
            test_env = os.environ.copy()
            test_env.pop("CUDA_HOME", None)
            test_env.pop("CUDA_PATH", None)
            test_env["MACA_PATH"] = str(maca_path)
            with patch.dict(os.environ, test_env, clear=True):
                self.assertEqual(
                    Path(load_utils._find_cuda_home()),
                    maca_path / "tools" / "cu-bridge",
                )

    def test_candidate_runtime_library_dirs_include_maca_paths(self):
        with TemporaryDirectory() as tmp_dir:
            maca_path = Path(tmp_dir) / "maca"
            cuda_home = maca_path / "tools" / "cu-bridge"
            test_env = os.environ.copy()
            test_env.pop("CUDA_HOME", None)
            test_env.pop("CUDA_PATH", None)
            test_env["MACA_PATH"] = str(maca_path)
            with patch.dict(os.environ, test_env, clear=True):
                dirs = load_utils._candidate_runtime_library_dirs(cuda_home)

            self.assertIn(maca_path / "lib", dirs)
            self.assertIn(maca_path / "mxgpu_llvm" / "lib", dirs)
            self.assertIn(cuda_home / "lib", dirs)
            self.assertIn(cuda_home / "lib64", dirs)


if __name__ == "__main__":
    unittest.main()
