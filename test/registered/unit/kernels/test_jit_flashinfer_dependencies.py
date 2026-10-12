from pathlib import Path

import pytest

from sglang.kernels.jit.utils import deps
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_flashinfer_tree(tmp_path, monkeypatch, *, maca, spdlog):
    root = tmp_path / "flashinfer"
    paths = ["include", "csrc", "cutlass/include", "cutlass/tools/util/include"]
    if spdlog:
        paths.append("spdlog/include")
    for path in paths:
        (root / "data" / path).mkdir(parents=True)
    monkeypatch.setattr(deps, "_find_package_root", lambda package: root)
    monkeypatch.setattr(deps, "is_maca_runtime", lambda: maca)
    return root


def test_maca_flashinfer_without_spdlog(tmp_path, monkeypatch):
    make_flashinfer_tree(tmp_path, monkeypatch, maca=True, spdlog=False)
    paths = deps.get_flashinfer_include_paths()
    assert len(paths) == 4
    assert all(Path(path).is_dir() for path in paths)


@pytest.mark.parametrize("maca", [False, True])
def test_flashinfer_with_spdlog(tmp_path, monkeypatch, maca):
    root = make_flashinfer_tree(tmp_path, monkeypatch, maca=maca, spdlog=True)
    assert str(root / "data/spdlog/include") in deps.get_flashinfer_include_paths()


def test_non_maca_still_requires_spdlog(tmp_path, monkeypatch):
    make_flashinfer_tree(tmp_path, monkeypatch, maca=False, spdlog=False)
    with pytest.raises(RuntimeError, match="spdlog"):
        deps.get_flashinfer_include_paths()


def test_maca_still_requires_core_headers(tmp_path, monkeypatch):
    root = make_flashinfer_tree(tmp_path, monkeypatch, maca=True, spdlog=False)
    (root / "data/include").rmdir()
    with pytest.raises(RuntimeError, match="Required header path"):
        deps.get_flashinfer_include_paths()
