"""Offset/size-driven device memcpy kernel, migrated from
``sglang.srt.layers.dp_attention`` (RFC #29630, Phase 2.5).
"""

import functools

import triton
import triton.language as tl


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    src_numel,
    dst_numel,
    offset_src: tl.constexpr,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size
    if offset_src:
        max_sz_src = src_numel - offset
        max_sz_dst = dst_numel
    else:
        max_sz_src = src_numel
        max_sz_dst = dst_numel - offset

    real_sz = tl.minimum(sz, tl.minimum(max_sz_src, max_sz_dst))

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = (start_index + offs < real_sz) & (offset >= 0)

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    offset_val = offset.clone().detach()
    sz_val = sz.clone().detach()
    memcpy_triton_kernel[grid](
        dst,
        src,
        offset_val,
        sz_val,
        src.numel(),
        dst.numel(),
        offset_src,
        chunk_size,
        BLOCK_SIZE,
    )
