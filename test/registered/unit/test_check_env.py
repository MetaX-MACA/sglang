import unittest

from sglang.check_env import CPUEnv


class TestCheckEnv(unittest.TestCase):
    def test_cpu_env_reports_no_accelerator(self):
        self.assertEqual(CPUEnv().get_info(), {"Accelerator": "Not Available"})
        self.assertEqual(CPUEnv().get_topology(), {})


if __name__ == "__main__":
    unittest.main()
