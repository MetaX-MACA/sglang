# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Config-driven online rotation modules (QuaRot-style Hadamard rotation).

Mounted config-driven at pre-norm locations: right before the final RMSNorm of
the residual stream and before the RMSNorm of the MTP shared head.  They
implement the normalized composite Hadamard transform

    y = D (H_K (x) H_P) / sqrt(N)     with   N = K * P,  P a power of two

matching the vLLM-metax ``CompositeHadamardRotation`` used by QuaRot W4A8
quantized GLM/DeepSeek models.  ``sign`` (int8 [N]) and ``hk`` (float32 [K, K])
come from the checkpoint.  The math is pixel-identical to the reference
TileLang kernel (``sglang.jit_kernel.hadamard_tilelang``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from sglang.jit_kernel.hadamard_tilelang import hadamard
from sglang.jit_kernel.hadamard_tilelang.utils import (
    make_h16,
    resolve_hadamard_dim,
)


def get_online_rotation_module_config(
    config: Any,
    module_name: str,
) -> Optional[Mapping[str, Any]]:
    """Return the online-rotation spec mounted at ``module_name`` (or None)."""
    rotation_config = getattr(config, "online_rotation", None)
    if rotation_config is None:
        return None
    if not isinstance(rotation_config, Mapping):
        raise TypeError("online_rotation must be a mapping")
    modules = rotation_config.get("modules")
    if modules is None:
        raise ValueError("online_rotation.modules is required")
    if not isinstance(modules, Mapping):
        raise TypeError("online_rotation.modules must be a mapping")
    module_config = modules.get(module_name)
    if module_config is None:
        return None
    if not isinstance(module_config, Mapping):
        raise TypeError(
            f"online_rotation.modules[{module_name!r}] must be a mapping"
        )
    return module_config


def build_rotation(config: Mapping[str, Any]) -> nn.Module:
    """Build an online rotation from one module-local rotation spec."""
    rotation_type = config.get("type")
    if rotation_type == "composite_hadamard":
        return CompositeHadamardRotation.from_rotation_config(config)
    raise ValueError(f"unsupported online rotation type: {rotation_type!r}")


class CompositeHadamardRotation(nn.Module):
    r"""Normalized composite Hadamard rotation with checkpoint ``sign``/``hk``.

    Delegates to the TileLang composite Hadamard kernel.  For ``inverse=True``
    the sign is applied on the output; for ``inverse=False`` it is applied on
    the input, exactly mirroring the reference vLLM-metax kernel.
    """

    def __init__(
        self,
        hidden_size: int,
        sign: Optional[torch.Tensor] = None,
        hk: Optional[torch.Tensor] = None,
        *,
        k: Optional[int] = None,
        p: Optional[int] = None,
        inverse: bool = False,
        num_threads: int = 512,
        mma_rows: int = 128,
    ) -> None:
        super().__init__()
        k, power_of_two_part = resolve_hadamard_dim(hidden_size, k=k, p=p)
        if num_threads <= 0 or num_threads % 64:
            raise ValueError("num_threads must be a positive multiple of 64")
        if mma_rows <= 0 or mma_rows % 16:
            raise ValueError("mma_rows must be a positive multiple of 16")
        if (sign is None) != (hk is None):
            raise ValueError("sign and hk must either both be provided or both be None")

        if sign is None:
            sign = torch.empty(hidden_size, dtype=torch.int8)
            hk = torch.empty((k, k), dtype=torch.float32)
        else:
            if sign.shape != (hidden_size,) or sign.dtype != torch.int8:
                raise ValueError("sign must be a [hidden_size] int8 tensor")
            if hk is None or hk.shape != (k, k):
                raise ValueError("hk must be a [K, K] tensor")

        sign_param = sign.detach().to(dtype=torch.int8).contiguous()
        hk_param = hk.detach().to(dtype=torch.float32).contiguous()
        self.register_parameter(
            "sign", nn.Parameter(sign_param, requires_grad=False)
        )
        self.register_parameter("hk", nn.Parameter(hk_param, requires_grad=False))
        self.register_buffer(
            "h16",
            make_h16(device=sign_param.device, dtype=torch.bfloat16),
            persistent=False,
        )

        self.hidden_size = hidden_size
        self.k = k
        self.p = power_of_two_part
        self.inverse = inverse
        self.num_threads = num_threads
        self.mma_rows = mma_rows

    def forward(
        self,
        x: torch.Tensor,
        inverse: Optional[bool] = None,
    ) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden_size={self.hidden_size}, got {x.shape[-1]}"
            )
        if inverse is None:
            inverse = self.inverse
        sign = self.sign.to(device=x.device)
        hk = self.hk.to(device=x.device, dtype=torch.float32)
        h16 = self.h16.to(device=x.device, dtype=x.dtype)
        return hadamard(
            x,
            sign,
            hk,
            h16,
            inverse=inverse,
            k=self.k,
            p=self.p,
            num_threads=self.num_threads,
            mma_rows=self.mma_rows,
        )

    def rotation_config(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "K": self.k,
            "P": self.p,
            "inverse": self.inverse,
        }

    @classmethod
    def from_rotation_config(
        cls,
        config: Mapping[str, Any],
        sign: Optional[torch.Tensor] = None,
        hk: Optional[torch.Tensor] = None,
    ) -> "CompositeHadamardRotation":
        # ``hk`` is kept in the on-disk orientation; the kernel indexes it as
        # ``hk[out_block, in_block]`` (inverse) / ``hk[in_block, out_block]``
        # (forward), exactly like the reference vLLM-metax module.
        return cls(
            hidden_size=int(config["hidden_size"]),
            sign=sign,
            hk=hk,
            k=int(config["K"]),
            p=None if config.get("P") is None else int(config["P"]),
            inverse=bool(config.get("inverse", True)),
            num_threads=int(config.get("num_threads", 512)),
            mma_rows=int(config.get("mma_rows", 128)),
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, K={self.k}, P={self.p}, "
            f"inverse={self.inverse}, num_threads={self.num_threads}, "
            f"mma_rows={self.mma_rows}"
        )


__all__ = [
    "CompositeHadamardRotation",
    "build_rotation",
    "get_online_rotation_module_config",
]