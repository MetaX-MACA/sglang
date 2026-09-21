"""Layout contract for the quantized GLM/DeepSeek-DSA MLA KV cache.

FP8 and INT8 share the same 656-byte row ABI:

    [0:512]   quantized c_kv (four groups of 128 values)
    [512:528] four FP32 dequantization scales
    [528:656] 64 BF16 RoPE values

Only the interpretation of the first 512 bytes differs between FP8 and INT8.
"""

from enum import IntEnum

import torch


class DSAKVCacheFormat(IntEnum):
    BF16 = 0
    FP8 = 1
    INT8 = 2


NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
QUANT_GROUP_SIZE = 128
NUM_QUANT_GROUPS = NOPE_DIM // QUANT_GROUP_SIZE

QUANT_VALUE_BYTES = NOPE_DIM
SCALE_BYTES_PER_TOKEN = NUM_QUANT_GROUPS * torch.float32.itemsize
ROPE_BYTES_PER_TOKEN = ROPE_DIM * torch.bfloat16.itemsize
PACKED_BYTES_PER_TOKEN = (
    QUANT_VALUE_BYTES + SCALE_BYTES_PER_TOKEN + ROPE_BYTES_PER_TOKEN
)
BF16_BYTES_PER_TOKEN = HEAD_DIM * torch.bfloat16.itemsize

INT8_SCALE_EPS = 1.0e-4
INT8_QMAX = 127.0


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype is torch.float8_e4m3fn


def get_kv_cache_format(dtype: torch.dtype) -> DSAKVCacheFormat:
    if dtype is torch.int8:
        return DSAKVCacheFormat.INT8
    if is_fp8_dtype(dtype):
        return DSAKVCacheFormat.FP8
    if dtype is torch.bfloat16:
        return DSAKVCacheFormat.BF16
    raise ValueError(f"Unsupported DSA main KV cache dtype: {dtype}")


def is_quantized_kv_cache(dtype: torch.dtype) -> bool:
    return get_kv_cache_format(dtype) in (
        DSAKVCacheFormat.FP8,
        DSAKVCacheFormat.INT8,
    )
