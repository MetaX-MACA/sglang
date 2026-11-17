
from __future__ import annotations

import functools
from enum import Enum

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_maca


class DSV4IndexCacheKind(Enum):
    FP4 = "fp4"
    FP8 = "fp8"
    INT8 = "int8"

    def slot_bytes(self, index_head_dim: int) -> int:
        """Bytes of one index-K slot, payload plus its per-row scale."""
        if self is DSV4IndexCacheKind.FP4:
            # index_head_dim // 2 payload bytes + 4B ue8m0 scale per 32 values
            return index_head_dim // 2 + index_head_dim // 32
        # One byte per value (fp8 or int8) + a 4B float32 scale per 128 values
        return index_head_dim + index_head_dim // 128 * 4

    @property
    def buffer_dtype(self) -> torch.dtype:
        """The dtype the index-K pool exposes the slot bytes as."""
        return torch.int8 if self.is_int8 else torch.uint8

    @property
    def is_fp4(self) -> bool:
        return self is DSV4IndexCacheKind.FP4

    @property
    def is_fp8(self) -> bool:
        return self is DSV4IndexCacheKind.FP8

    @property
    def is_int8(self) -> bool:
        return self is DSV4IndexCacheKind.INT8


@functools.cache
def dsv4_index_cache_kind() -> DSV4IndexCacheKind:
    """Resolve this process's index cache kind from ``SGLANG_DSV4_INDEX_CACHE_TYPE``.

    Must be called only after ``_deepseek_v4_index_cache_dtype`` has been
    initialized; calling earlier may resolve against a stale/default value.
    The result is cached, so the first call fixes the kind for the process.
    """
    cache_type = envs.SGLANG_DSV4_INDEX_CACHE_TYPE.get()
    if cache_type == "fp8":
        return DSV4IndexCacheKind.FP8
    if is_maca() and cache_type == "int8":
        return DSV4IndexCacheKind.INT8
    return DSV4IndexCacheKind.FP4
