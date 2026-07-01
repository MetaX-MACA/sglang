import subprocess
import unittest
from unittest.mock import patch

from sglang.check_env import GPUEnv, HIPEnv, MUSAEnv, NPUEnv


class TestCheckEnv(unittest.TestCase):
    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_gpu_topology_missing_command_returns_empty_info(self, mock_run):
        self.assertEqual(GPUEnv().get_topology(), {})
        mock_run.assert_called_once_with(
            ["nvidia-smi", "topo", "-m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_hip_topology_missing_command_returns_empty_info(self, mock_run):
        self.assertEqual(HIPEnv().get_topology(), {})
        mock_run.assert_called_once_with(
            ["rocm-smi", "--showtopotype"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_npu_topology_missing_command_returns_empty_info(self, mock_run):
        self.assertEqual(NPUEnv().get_topology(), {})
        mock_run.assert_called_once_with(
            ["npu-smi", "info", "-t", "topo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("sglang.check_env.subprocess.run", side_effect=FileNotFoundError)
    def test_musa_topology_missing_command_returns_empty_info(self, mock_run):
        self.assertEqual(MUSAEnv().get_topology(), {})
        mock_run.assert_called_once_with(
            ["mthreads-gmi", "topo", "-m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
