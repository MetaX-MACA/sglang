"""Paged KV cache layouts of the DeepSeek-V4 family sparse MLA decode kernels.

A page block stores ``page_size`` data rows followed by ``page_size`` scale rows.
The reader selects the format from the bytes per token (the last dim of the
``(num_pages, page_size, 1, bytes_per_token)`` view) and requires the page
stride to be a multiple of its TMA row stride, which :meth:`KVLayout.page_bytes`
pads to. Mirrors ``sgl_kernel/deepseek_v4/kv_layout.cuh``.
"""

from __future__ import annotations

import enum
from typing import Union

import torch


class KVLayout(str, enum.Enum):
    # 448 fp8 nope + 64 bf16 rope, 7 ue8m0 scales (+1 pad) per 64 values.
    V4 = "v4"
    # 512 fp8 (rope quantized too), 16 ue8m0 scales per 32 values.
    V41 = "v41"
    # 512 e2m1 packed two per byte (even index low nibble), 32 e4m3 scales per 16 values.
    V41_FP4 = "v41_fp4"

    @property
    def data_bytes(self) -> int:
        return {KVLayout.V4: 576, KVLayout.V41: 512, KVLayout.V41_FP4: 256}[self]

    @property
    def scale_bytes(self) -> int:
        return {KVLayout.V4: 8, KVLayout.V41: 16, KVLayout.V41_FP4: 32}[self]

    @property
    def tile_size(self) -> int:
        """Values sharing one scale."""
        return {KVLayout.V4: 64, KVLayout.V41: 32, KVLayout.V41_FP4: 16}[self]

    @property
    def bytes_per_token(self) -> int:
        return self.data_bytes + self.scale_bytes

    @property
    def page_align(self) -> int:
        """Unit the page stride is padded to: the reader's TMA row stride."""
        return {KVLayout.V4: 576, KVLayout.V41: 512, KVLayout.V41_FP4: 256}[self]

    @property
    def is_fp4(self) -> bool:
        return self is KVLayout.V41_FP4

    def page_bytes(self, page_size: int) -> int:
        raw = page_size * self.bytes_per_token
        return -(-raw // self.page_align) * self.page_align

    def scale_offset(self, page_size: int) -> int:
        """Byte offset of the scale rows inside a page."""
        return page_size * self.data_bytes

    @property
    def cpp_name(self) -> str:
        """The C++ enumerator, for JIT template arguments. Bare, because it is
        also part of the JIT module name; the headers `using enum` it in."""
        return self.name

    @classmethod
    def parse(cls, value: Union[str, KVLayout]) -> KVLayout:
        if isinstance(value, KVLayout):
            return value
        return cls(str(value).lower())


def is_valid_kv_layout_pair(kv: KVLayout, extra_kv: KVLayout) -> bool:
    """The (main, extra) cache pairs the decode kernel accepts: identical layouts,
    or the fp4 extra cache next to a V4.1 fp8 main cache."""
    return extra_kv is kv or (kv is KVLayout.V41 and extra_kv is KVLayout.V41_FP4)


# V4 dtype variants. KVLayout properties retain their original FP8/FP4 semantics.
class DSV4KVCacheFormat(enum.IntEnum):
    BF16 = 0
    FP8 = 1
    INT8 = 2


NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
QUANT_GROUP_SIZE = KVLayout.V4.tile_size
NUM_QUANT_GROUPS = NOPE_DIM // QUANT_GROUP_SIZE

# Per-token bytes in the page-wide value section.
VALUE_BYTES_PER_TOKEN = KVLayout.V4.data_bytes  # 448 quantized + 64 BF16

# FP8 stores 7 UE8M0 bytes plus one padding byte. INT8 stores 7 FP32 scales
# plus four padding bytes, preserving a 32-byte scale record per token.
FP8_SCALE_BYTES_PER_TOKEN = KVLayout.V4.scale_bytes
INT8_SCALE_BYTES_PER_TOKEN = 32
FP8_BYTES_PER_TOKEN = KVLayout.V4.bytes_per_token  # 584
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



def kv_cache_bytes_per_token(layout: Union[str, KVLayout], dtype: torch.dtype) -> int:
    layout = KVLayout.parse(layout)
    if dtype is torch.int8 or dtype is torch.bfloat16:
        if layout is not KVLayout.V4:
            raise ValueError("INT8/BF16 main KV requires the V4 layout")
        return INT8_BYTES_PER_TOKEN if dtype is torch.int8 else BF16_BYTES_PER_TOKEN
    get_kv_cache_format(dtype)
    return layout.bytes_per_token


def kv_cache_scale_bytes_per_token(layout: Union[str, KVLayout], dtype: torch.dtype) -> int:
    layout = KVLayout.parse(layout)
    kv_cache_bytes_per_token(layout, dtype)
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.int8:
        return INT8_SCALE_BYTES_PER_TOKEN
    return layout.scale_bytes


def kv_cache_page_bytes(layout: Union[str, KVLayout], dtype: torch.dtype, page_size: int) -> int:
    layout = KVLayout.parse(layout)
    raw = page_size * kv_cache_bytes_per_token(layout, dtype)
    if dtype is torch.bfloat16:
        return raw
    return -(-raw // layout.page_align) * layout.page_align
