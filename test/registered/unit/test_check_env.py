import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sglang.check_env import GPUEnv, HIPEnv


class TestCheckEnv(unittest.TestCase):
    @patch("sglang.check_env.subprocess.check_output", side_effect=FileNotFoundError)
    def test_nvcc_missing_binary_returns_not_available(self, mock_check_output):
        with TemporaryDirectory() as cuda_home:
            with patch("torch.utils.cpp_extension.CUDA_HOME", cuda_home):
                self.assertEqual(GPUEnv()._get_nvcc_info(), {"NVCC": "Not Available"})
        mock_check_output.assert_called_once()

    @patch("sglang.check_env.subprocess.check_output", side_effect=FileNotFoundError)
    def test_hipcc_missing_binary_returns_not_available(self, mock_check_output):
        with TemporaryDirectory() as rocm_home:
            with patch("torch.utils.cpp_extension.ROCM_HOME", rocm_home):
                self.assertEqual(HIPEnv()._get_hipcc_info(), {"HIPCC": "Not Available"})
        mock_check_output.assert_called_once()

    @patch("torch.utils.cpp_extension.ROCM_HOME", None)
    def test_hipcc_missing_rocm_home_returns_not_available(self):
        self.assertEqual(HIPEnv()._get_hipcc_info(), {"HIPCC": "Not Available"})


if __name__ == "__main__":
    unittest.main()
