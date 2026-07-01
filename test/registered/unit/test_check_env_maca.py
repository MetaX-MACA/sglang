import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_check_env_module():
    resource = types.ModuleType("resource")
    resource.RLIMIT_NOFILE = 7
    resource.getrlimit = lambda _limit: (1024, 4096)

    torch = types.ModuleType("torch")
    torch.__version__ = "2.8.0"
    torch.version = types.SimpleNamespace(cuda=None, hip=None)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_name=lambda _index: "MetaX GPU",
        get_device_capability=lambda _index: (0, 0),
    )

    utils = types.ModuleType("sglang.srt.utils")
    utils.is_hip = lambda: False
    utils.is_mps = lambda: False
    utils.is_musa = lambda: False
    utils.is_npu = lambda: False

    with patch.dict(
        sys.modules,
        {
            "sglang": types.ModuleType("sglang"),
            "sglang.srt": types.ModuleType("sglang.srt"),
            "sglang.srt.utils": utils,
            "resource": resource,
            "torch": torch,
        },
    ):
        module_path = Path(__file__).parents[3] / "python" / "sglang" / "check_env.py"
        spec = importlib.util.spec_from_file_location("test_check_env_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestMACAEnv(unittest.TestCase):
    def setUp(self):
        self.check_env = load_check_env_module()

    def test_is_maca_v2_detects_maca_env(self):
        with patch.dict(os.environ, {"MACA_PATH": "/opt/maca"}, clear=True):
            with patch.object(self.check_env.shutil, "which", return_value=None):
                self.assertTrue(self.check_env.is_maca_v2())

    def test_is_maca_v2_detects_cucc_on_path(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(self.check_env.shutil, "which") as mock_which:
                mock_which.side_effect = lambda name: "/usr/bin/cucc" if name == "cucc" else None
                self.assertTrue(self.check_env.is_maca_v2())

    def test_cucc_info_uses_cucc_path_first(self):
        env = self.check_env.MACAEnv()

        with patch.dict(os.environ, {"CUCC_PATH": "/opt/cucc"}, clear=True):
            with patch.object(self.check_env.subprocess, "check_output") as mock_check_output:
                mock_check_output.return_value = b"cucc version 3.0\nBuild test"
                self.assertEqual(env._get_cucc_info(), {"CUCC": "cucc version 3.0"})
                mock_check_output.assert_called_once_with(
                    [os.path.join("/opt/cucc", "bin", "cucc"), "--version"],
                    stderr=self.check_env.subprocess.STDOUT,
                    timeout=10,
                )

    def test_cucc_info_is_not_available_without_working_candidates(self):
        env = self.check_env.MACAEnv()

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(self.check_env.shutil, "which", return_value=None):
                self.assertEqual(env._get_cucc_info(), {"CUCC": "Not Available"})

    def test_mx_smi_info_is_not_available_when_missing(self):
        env = self.check_env.MACAEnv()

        with patch.object(self.check_env.shutil, "which", return_value=None):
            self.assertEqual(env._get_mx_smi_info(), {"MX-SMI": "Not Available"})

    def test_base_package_list_is_not_mutated_by_subclasses(self):
        base_env = self.check_env.BaseEnv()
        npu_env = self.check_env.NPUEnv()

        self.assertEqual(base_env.package_list.count("torch_npu"), 0)
        self.assertEqual(npu_env.package_list.count("torch_npu"), 1)
        self.assertEqual(self.check_env.PACKAGE_LIST.count("torch_npu"), 0)


if __name__ == "__main__":
    unittest.main()
