"""Storage selection shared by V4 indexer allocation, budgeting and scoring."""

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_maca


def get_dsv4_index_cache_type() -> str:
    if not is_maca():
        return "fp8"
    value = envs.SGLANG_DSV4_INDEX_CACHE_TYPE.get().strip().lower()
    value = "bf16" if value == "bfloat16" else value
    if value not in ("int8", "fp8", "bf16"):
        raise ValueError(
            f"SGLANG_DSV4_INDEX_CACHE_TYPE must be int8, fp8, or bf16; got {value!r}"
        )
    return value


def get_dsv4_index_cache_storage_dtype() -> torch.dtype:
    return {"int8": torch.int8, "fp8": torch.uint8, "bf16": torch.bfloat16}[
        get_dsv4_index_cache_type()
    ]
