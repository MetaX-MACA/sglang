from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsv4.kv_cache_layout import (
    DSV4KVCacheFormat,
    FP8_SCALE_BYTES_PER_TOKEN,
    INT8_SCALE_BYTES_PER_TOKEN,
    NOPE_DIM,
    NUM_QUANT_GROUPS,
    QUANT_GROUP_SIZE,
    ROPE_DIM,
    VALUE_BYTES_PER_TOKEN,
    get_kv_cache_format,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# V4 packed KV cache layout. Both formats use a P*576-byte value section:
#   FP8: 448 FP8 NOPE + 64 BF16 RoPE + an 8-byte UE8M0 scale record.
#   INT8: 448 INT8 NOPE + 64 BF16 RoPE + a 32-byte FP32 scale record.
# All value records precede all scale records; each physical page is padded to
# a multiple of 576 bytes.
DIM_NOPE = NOPE_DIM
DIM_ROPE = ROPE_DIM
TILE_SIZE = QUANT_GROUP_SIZE
NUM_SCALE_TILES = NUM_QUANT_GROUPS
NOPE_ROPE_BYTES = VALUE_BYTES_PER_TOKEN


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize the DeepSeek v4 paged KV cache for a list of token IDs.

    Args:
        quant_k_cache: FP8/uint8 or INT8 view of
            (num_pages, bytes_per_page_padded).
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_NOPE + DIM_ROPE) bf16 destination.
            May be a slice of a larger workspace; the kernel uses out.stride(0)
            so contiguous-along-dim-0 slices work.

    Returns:
        (num_tokens, 1, DIM_NOPE + DIM_ROPE) bfloat16.
    """
    assert quant_k_cache.ndim == 2
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    cache_format = get_kv_cache_format(quant_k_cache.dtype)
    assert cache_format in (DSV4KVCacheFormat.FP8, DSV4KVCacheFormat.INT8)

    # The buffer's dtype is whatever the pool exposes (often bf16); the
    # underlying storage is uint8. Reinterpret to byte-space first.
    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    # Three typed views over the same underlying bytes.
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_int8 = quant_k_cache_u8.view(torch.int8).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)
    buf_fp32 = quant_k_cache_u8.view(torch.float32).reshape(-1)

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16

    _dequantize_k_cache_paged_kernel[(num_tokens,)](
        out,
        buf_fp8,
        buf_int8,
        buf_bf16,
        buf_uint8,
        buf_fp32,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        SCALE_BYTES_PER_TOKEN=(
            FP8_SCALE_BYTES_PER_TOKEN
            if cache_format == DSV4KVCacheFormat.FP8
            else INT8_SCALE_BYTES_PER_TOKEN
        ),
        S_OFFSET_BYTES=s_offset_bytes,
        CACHE_FORMAT=int(cache_format),
    )
    return out


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_int8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    buf_fp32_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    SCALE_BYTES_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * SCALE_BYTES_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        if CACHE_FORMAT == 1:
            quant_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)
            scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(
                tl.int32
            )
            scale = tl.exp2((scale_u8 - 127).to(tl.float32))
        else:
            quant_vals = tl.load(buf_int8_ptr + fp8_off).to(tl.float32)
            scale = tl.load(buf_fp32_ptr + token_scale_base // 4 + tile_id)

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (quant_vals * scale).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)


def dequantize_k_cache_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Pure-torch reference for :func:`dequantize_k_cache_paged`.

    Decodes the same v4 paged layout with vectorized torch indexing instead of
    a Triton kernel. Used to validate the kernel (see the ``__main__`` block
    below); not on any hot path.
    """
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    assert quant_k_cache.ndim == 2
    cache_format = get_kv_cache_format(quant_k_cache.dtype)
    assert cache_format in (DSV4KVCacheFormat.FP8, DSV4KVCacheFormat.INT8)
    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    flat_u8 = u8.reshape(-1)
    flat_fp8 = u8.view(fp8_dtype).reshape(-1)
    flat_int8 = u8.view(torch.int8).reshape(-1)
    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)
    flat_fp32 = u8.view(torch.float32).reshape(-1)

    loc = page_table_1_flattened.to(torch.int64)
    page_idx = loc // page_size
    in_page = loc % page_size
    page_byte_base = page_idx * bytes_per_page
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    scale_stride = (
        FP8_SCALE_BYTES_PER_TOKEN
        if cache_format == DSV4KVCacheFormat.FP8
        else INT8_SCALE_BYTES_PER_TOKEN
    )
    token_scale_base = page_byte_base + s_offset_bytes + in_page * scale_stride

    device = quant_k_cache.device
    nope_byte = (
        token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
    )
    if cache_format == DSV4KVCacheFormat.FP8:
        nope_quant = flat_fp8[nope_byte].to(torch.float32)
        scale_byte = (
            token_scale_base[:, None]
            + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
        )
        scale_u8 = flat_u8[scale_byte].to(torch.int32)
        scales = torch.exp2((scale_u8 - 127).to(torch.float32))
        scales = torch.where(
            scales < (2.0**-126), torch.zeros_like(scales), scales
        )
    else:
        nope_quant = flat_int8[nope_byte].to(torch.float32)
        scale_idx = token_scale_base[:, None] // 4 + torch.arange(
            NUM_SCALE_TILES, device=device
        )[None, :]
        scales = flat_fp32[scale_idx]
    nope = nope_quant * scales.repeat_interleave(TILE_SIZE, dim=1)

    rope_bf16_base = (token_data_base + DIM_NOPE) // 2
    rope_idx = rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
    rope = flat_bf16[rope_idx]

    out = torch.empty(
        (loc.shape[0], 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=device,
    )
    out[:, 0, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, 0, DIM_NOPE:] = rope
    return out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "this self-test needs a CUDA device"
    torch.manual_seed(0)
    device = "cuda"

    page_size = 64
    num_pages = 8
    num_tokens = 333
    raw_bytes = page_size * (NOPE_ROPE_BYTES + FP8_SCALE_BYTES_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES

    quant_k_cache = torch.randint(
        0, 256, (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )
    page_table = torch.randint(
        0, num_pages * page_size, (num_tokens,), dtype=torch.int32, device=device
    )

    out_kernel = dequantize_k_cache_paged(quant_k_cache, page_table, page_size)
    out_ref = dequantize_k_cache_paged_ref(quant_k_cache, page_table, page_size)

    torch.testing.assert_close(out_kernel, out_ref, atol=0, rtol=0, equal_nan=True)
    print(
        f"OK: kernel matches torch ref for {num_tokens} tokens "
        f"(page_size={page_size}, bytes_per_page={bytes_per_page})"
    )
