import os
from pathlib import Path
from typing import Mapping

import torch


def get_torch_maca_version(torch_module=torch) -> str | None:
    return getattr(torch_module.version, "maca", None)


def has_maca_toolkit_env(env: Mapping[str, str] | None = None) -> bool:
    env = env or os.environ
    maca_path = env.get("MACA_PATH")
    if not maca_path:
        return False
    maca_root = Path(maca_path)
    return (
        (maca_root / "tools" / "cu-bridge").exists()
        or (maca_root / "mxgpu_llvm").exists()
        or (maca_root / "lib").exists()
    )


def is_maca_available(torch_module=torch, env: Mapping[str, str] | None = None) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return False
    return (
        get_torch_maca_version(torch_module) is not None
        or has_maca_toolkit_env(env)
    )
