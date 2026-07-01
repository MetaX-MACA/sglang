import importlib.util
import os
import sys
import types
from pathlib import Path


def _load_check_env(monkeypatch):
    utils_module = types.ModuleType("sglang.srt.utils")
    utils_module.is_hip = lambda: False
    utils_module.is_mps = lambda: False
    utils_module.is_musa = lambda: False
    utils_module.is_npu = lambda: False

    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_module)
    resource_module = types.ModuleType("resource")
    resource_module.RLIMIT_NOFILE = 0
    resource_module.getrlimit = lambda _limit: (1024, 1024)
    monkeypatch.setitem(sys.modules, "resource", resource_module)

    module_path = Path(__file__).resolve().parents[3] / "python" / "sglang" / "check_env.py"
    spec = importlib.util.spec_from_file_location("unit_check_env", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nvcc_info_uses_argument_list(monkeypatch, tmp_path):
    check_env = _load_check_env(monkeypatch)
    from torch.utils import cpp_extension

    captured = {}

    def _fake_check_output(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return "Cuda compilation tools, release 12.4, V12.4.99 Build test"

    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(check_env.subprocess, "check_output", _fake_check_output)

    assert check_env.GPUEnv()._get_nvcc_info() == {"NVCC": "Cuda compilation tools, release 12.4, V12.4.99"}
    assert captured["command"] == [os.path.join(str(tmp_path), "bin/nvcc"), "-V"]
    assert captured["kwargs"] == {"text": True}


def test_hipcc_info_uses_argument_list(monkeypatch, tmp_path):
    check_env = _load_check_env(monkeypatch)
    from torch.utils import cpp_extension

    captured = {}

    def _fake_check_output(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return "HIP version: 6.2.0\nAMD clang version test"

    monkeypatch.setattr(cpp_extension, "ROCM_HOME", str(tmp_path))
    monkeypatch.setattr(check_env.subprocess, "check_output", _fake_check_output)

    assert check_env.HIPEnv()._get_hipcc_info() == {"HIPCC": "HIP version: 6.2.0"}
    assert captured["command"] == [os.path.join(str(tmp_path), "bin/hipcc"), "--version"]
    assert captured["kwargs"] == {"text": True}
