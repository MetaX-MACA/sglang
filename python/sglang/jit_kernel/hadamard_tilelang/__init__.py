# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#       Ported into sglang.

from .hadamard import hadamard
from .utils import factor_hadamard_dim, resolve_hadamard_dim

__all__ = ["factor_hadamard_dim", "hadamard", "resolve_hadamard_dim"]