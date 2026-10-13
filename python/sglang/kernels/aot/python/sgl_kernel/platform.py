"""Platform checks shared by the standalone kernel build and runtime."""

import os


def is_maca(torch_version: str | None = None) -> bool:
    if torch_version is None:
        import torch

        torch_version = torch.__version__
    return "metax" in torch_version


def use_mcoplib(default: bool) -> bool:
    """Read the switch without importing SGLang from the standalone wheel."""
    return os.getenv("SGLANG_USE_MCOPLIB", str(default)).lower() in ("true", "1")
