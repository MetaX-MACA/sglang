from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    import torch


@cache_once
def _jit_host_pointer_module():
    return load_jit(
        "host_pointer",
        cuda_files=["memory/host_pointer.cuh"],
        cuda_wrappers=[("get_device_accessible_ptr", "get_device_accessible_ptr")],
    )


def get_device_accessible_ptr(tensor: torch.Tensor, device_index: int) -> int:
    """Return the runtime mapping of registered host memory on one CUDA device."""
    return int(
        _jit_host_pointer_module().get_device_accessible_ptr(tensor, device_index)
    )
