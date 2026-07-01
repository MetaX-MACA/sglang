import unittest
from unittest.mock import patch

from sglang.check_env import CHECK_ENV_COMMAND_TIMEOUT, GPUEnv, HIPEnv, MUSAEnv, NPUEnv


class TestCheckEnv(unittest.TestCase):
    def _assert_topology_uses_timeout(self, env):
        with patch("sglang.check_env.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "topology"
            env.get_topology()

        self.assertEqual(run.call_args.kwargs["timeout"], CHECK_ENV_COMMAND_TIMEOUT)

    def test_gpu_topology_uses_timeout(self):
        self._assert_topology_uses_timeout(GPUEnv())

    def test_hip_topology_uses_timeout(self):
        self._assert_topology_uses_timeout(HIPEnv())

    def test_npu_topology_uses_timeout(self):
        self._assert_topology_uses_timeout(NPUEnv())

    def test_musa_topology_uses_timeout(self):
        self._assert_topology_uses_timeout(MUSAEnv())


if __name__ == "__main__":
    unittest.main()
