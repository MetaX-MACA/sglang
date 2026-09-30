"""Shared layout contract for the DeepSeek-V4 main KV caches.

The packed formats keep every token's 448-D NOPE payload and 64-D BF16
RoPE payload in the page value section.  Per-token scales live in a second,
page-wide section so attention can issue contiguous vector loads over values.
"""

from enum import IntEnum

import torch


class DSV4KVCacheFormat(IntEnum):
    BF16 = 0
    FP8 = 1
    INT8 = 2


NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
QUANT_GROUP_SIZE = 64
NUM_QUANT_GROUPS = NOPE_DIM // QUANT_GROUP_SIZE

# Per-token bytes in the page-wide value section.
VALUE_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2  # 448 quantized + 64 BF16

# FP8 stores 7 UE8M0 bytes plus one padding byte. INT8 stores 7 FP32 scales
# plus four padding bytes, preserving a 32-byte scale record per token.
FP8_SCALE_BYTES_PER_TOKEN = 8
INT8_SCALE_BYTES_PER_TOKEN = 32
FP8_BYTES_PER_TOKEN = VALUE_BYTES_PER_TOKEN + FP8_SCALE_BYTES_PER_TOKEN  # 584
INT8_BYTES_PER_TOKEN = VALUE_BYTES_PER_TOKEN + INT8_SCALE_BYTES_PER_TOKEN  # 608
BF16_BYTES_PER_TOKEN = HEAD_DIM * 2  # 1024

INT8_SCALE_EPS = 1.0e-4
INT8_QMAX = 127.0


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    )


def get_kv_cache_format(dtype: torch.dtype) -> DSV4KVCacheFormat:
    if dtype is torch.int8:
        return DSV4KVCacheFormat.INT8
    if is_fp8_dtype(dtype) or dtype is torch.uint8:
        # The production FP8 pool exposes a float8 view, while a few kernel
        # call sites intentionally pass its raw uint8 storage.
        return DSV4KVCacheFormat.FP8
    if dtype is torch.bfloat16:
        return DSV4KVCacheFormat.BF16
    raise ValueError(f"Unsupported DSV4 main KV cache dtype: {dtype}")


def bytes_per_token(dtype: torch.dtype) -> int:
    cache_format = get_kv_cache_format(dtype)
    if cache_format == DSV4KVCacheFormat.FP8:
        return FP8_BYTES_PER_TOKEN
    if cache_format == DSV4KVCacheFormat.INT8:
        return INT8_BYTES_PER_TOKEN
    return BF16_BYTES_PER_TOKEN


def scale_bytes_per_token(dtype: torch.dtype) -> int:
    cache_format = get_kv_cache_format(dtype)
    if cache_format == DSV4KVCacheFormat.FP8:
        return FP8_SCALE_BYTES_PER_TOKEN
    if cache_format == DSV4KVCacheFormat.INT8:
        return INT8_SCALE_BYTES_PER_TOKEN
    return 0


def packed_page_bytes(page_size: int, dtype: torch.dtype) -> int:
    """Return the physical packed-page stride, aligned to 576 bytes."""
    logical_bytes = page_size * bytes_per_token(dtype)
    return (
        (logical_bytes + VALUE_BYTES_PER_TOKEN - 1) // VALUE_BYTES_PER_TOKEN
    ) * VALUE_BYTES_PER_TOKEN
