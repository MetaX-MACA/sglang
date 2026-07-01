import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path


def _load_check_env(monkeypatch):
    sglang_spec = importlib.util.find_spec("sglang")
    if sglang_spec and sglang_spec.submodule_search_locations:
        module_path = (
            Path(next(iter(sglang_spec.submodule_search_locations))) / "check_env.py"
        )
    else:
        module_path = (
            Path(__file__).resolve().parents[3] / "python" / "sglang" / "check_env.py"
        )

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

    spec = importlib.util.spec_from_file_location("unit_check_env", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_package_versions_handles_missing_packages(monkeypatch):
    check_env = _load_check_env(monkeypatch)

    class UnitEnv(check_env.BaseEnv):
        def get_info(self):
            return {}

        def get_topology(self):
            return {}

    env = UnitEnv()
    env.package_list = ["sglang", "missing-package-for-unit-test"]

    def _fake_version(package_name):
        if package_name == "sglang":
            return "1.2.3"
        raise importlib.metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(importlib.metadata, "version", _fake_version)

    assert env.get_package_versions() == {
        "sglang": "1.2.3",
        "missing-package-for-unit-test": "Module Not Found",
    }
