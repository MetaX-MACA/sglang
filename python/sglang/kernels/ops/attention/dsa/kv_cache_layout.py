"""Packed GLM/DeepSeek DSA cache rows: quantized NoPE, FP32 scales, BF16 RoPE.

The packed row has 512 quantized NoPE bytes followed by four FP32 scales.
Legacy DSA models append 64 BF16 RoPE values (656 bytes in total), while
GLM-5.3-Flash has no RoPE tail (528 bytes in total).

FP8 readers also support other dimensions; these constants must not restrict
their existing dynamic-dimension paths.
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
SCALE_BYTES_PER_TOKEN = NUM_QUANT_GROUPS * torch.float32.itemsize
GLM53_FLASH_PACKED_BYTES_PER_TOKEN = NOPE_DIM + SCALE_BYTES_PER_TOKEN
PACKED_BYTES_PER_TOKEN = (
    GLM53_FLASH_PACKED_BYTES_PER_TOKEN + ROPE_DIM * torch.bfloat16.itemsize
)


def packed_kv_row_bytes(rope_dim: int) -> int:
    if rope_dim not in (0, ROPE_DIM):
        raise ValueError(f"Unsupported DSA RoPE dimension: {rope_dim}")
    return GLM53_FLASH_PACKED_BYTES_PER_TOKEN + rope_dim * torch.bfloat16.itemsize


def rope_dim_from_packed_row(row_bytes: int) -> int:
    for rope_dim in (0, ROPE_DIM):
        if row_bytes == packed_kv_row_bytes(rope_dim):
            return rope_dim
    raise ValueError(f"Unsupported DSA packed row width: {row_bytes}")


def get_kv_cache_format(dtype: torch.dtype) -> DSAKVCacheFormat:
    if dtype is torch.int8:
        return DSAKVCacheFormat.INT8
    if dtype is torch.float8_e4m3fn:
        return DSAKVCacheFormat.FP8
    if dtype is torch.bfloat16:
        return DSAKVCacheFormat.BF16
    raise ValueError(f"Unsupported packed DSA KV cache dtype: {dtype}")


def validate_int8_kv_cache(dim_nope: int, dim_rope: int) -> None:
    if dim_nope != NOPE_DIM or dim_rope not in (0, ROPE_DIM):
        raise ValueError(
            "DSA INT8 KV cache requires kv_lora_rank=512 and "
            "qk_rope_head_dim in {0, 64}; "
            f"got {dim_nope}+{dim_rope}"
        )
