"""Packed GLM/DeepSeek DSA cache rows: quantized NoPE, FP32 scales, BF16 RoPE.

The initial INT8 ABI matches the 512+64 FP8 layout (656 bytes per token).
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
PACKED_BYTES_PER_TOKEN = NOPE_DIM + NUM_QUANT_GROUPS * 4 + ROPE_DIM * 2


def get_kv_cache_format(dtype: torch.dtype) -> DSAKVCacheFormat:
    if dtype is torch.int8:
        return DSAKVCacheFormat.INT8
    if dtype is torch.float8_e4m3fn:
        return DSAKVCacheFormat.FP8
    if dtype is torch.bfloat16:
        return DSAKVCacheFormat.BF16
    raise ValueError(f"Unsupported packed DSA KV cache dtype: {dtype}")


def validate_int8_kv_cache(dim_nope: int, dim_rope: int) -> None:
    if (dim_nope, dim_rope) != (NOPE_DIM, ROPE_DIM):
        raise ValueError(
            "DSA INT8 KV cache requires kv_lora_rank=512 and qk_rope_head_dim=64; "
            f"got {dim_nope}+{dim_rope}"
        )
