# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight switch for optional C500 kernels; no kernel imports here."""

import os

_C500_KDA_OPTIMIZED_KERNELS_ENV = "MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS"
_MCOPLIB_KDA_ENV = "MX_USE_MCOPLIB_KDA"
_C500_KDA_OPT_TRUTHY = ("1", "true", "on", "yes")


def _env_truthy(name: str) -> bool:
    # Unset (or any value outside the truthy set) counts as off, matching the
    # EnvBool(False) default of MX_USE_MCOPLIB_KDA in sglang.srt.environ.
    return os.environ.get(name, "").lower() in _C500_KDA_OPT_TRUTHY


def _c500_kda_opt_enabled() -> bool:
    """Master switch for the C500-optimized KDA kernels (default: off).

    Read from the environment on every call so the flag can be flipped for a
    new process without any caching surprises; the read is a dict lookup.

    Mutually exclusive with the mcoplib KDA kernels (MX_USE_MCOPLIB_KDA):
    enabling both raises on the first guarded dispatch (fail-fast during
    warmup) instead of silently resolving the conflict. MX_USE_MCOPLIB_KDA
    unset or =0 is not a conflict; with the master switch unset nothing here
    raises and behavior is unchanged.
    """
    enabled = _env_truthy(_C500_KDA_OPTIMIZED_KERNELS_ENV)
    if enabled and _env_truthy(_MCOPLIB_KDA_ENV):
        raise RuntimeError(
            f"{_C500_KDA_OPTIMIZED_KERNELS_ENV} and {_MCOPLIB_KDA_ENV} are "
            "mutually exclusive; enable exactly one of them:\n"
            "  - optimized C500 Triton kernels: "
            "MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS=1 with MX_USE_MCOPLIB_KDA "
            "unset or =0\n"
            "  - mcoplib KDA kernels: MX_USE_MCOPLIB_KDA=1 with "
            "MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS unset or =0"
        )
    return enabled
