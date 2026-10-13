"""GLM-5.2 FP8 KV-cache quantization and paged dequantization operators."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def get_available_fp8_dtype() -> torch.dtype:
    return torch.float8_e4m3fn


DIM_NOPE = 512
DIM_ROPE = 64
GROUP_SIZE = 128
NUM_NOPE_BLOCKS = DIM_NOPE // GROUP_SIZE
PACKED_BYTES = DIM_NOPE + NUM_NOPE_BLOCKS * 4 + DIM_ROPE * 2


def quantize_k_cache_separate(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    tile_size: int = 128,
):
    """Quantize separate NoPE/RoPE inputs into packed byte tensors.

    Returns (nope_part, rope_part) with shapes [N, 1, 528] and [N, 1, 128].
    NoPE contains 512 FP8 values followed by four FP32 scales; RoPE stays BF16.
    Both outputs can be passed directly to set_mla_kv_buffer_triton.
    """
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    num_tokens = k_nope_2d.shape[0]
    dim_nope = k_nope_2d.shape[1]
    dim_rope = k_rope_2d.shape[1]

    if dim_nope != 512:
        raise ValueError(f"Expected dim_nope=512 for FP8 MLA, got {dim_nope}")
    if dim_rope != 64:
        raise ValueError(f"Expected dim_rope=64 for FP8 MLA, got {dim_rope}")
    if k_rope_2d.shape[0] != num_tokens:
        raise ValueError(
            f"k_nope and k_rope must have same num_tokens, got {num_tokens} vs {k_rope_2d.shape[0]}"
        )

    return _quantize_k_cache_fast_separate(
        k_nope=k_nope_2d, k_rope=k_rope_2d, group_size=tile_size
    )


@triton.jit
def _quantize_k_cache_fast_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        effective_block_id = raw_block_id
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs

        y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.max(tl.abs(y)) / FP8_MAX
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, FP8_MIN, FP8_MAX).to(
            output_nope_q_ptr.dtype.element_ty
        )
        dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
        dst_s_ptr = (
            output_nope_s_ptr + token_id * output_nope_s_stride_0 + effective_block_id
        )

        tl.store(dst_q_ptr, y_q, mask=mask)
        tl.store(dst_s_ptr, y_s)
    else:
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs
        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)


def _quantize_k_cache_fast_separate(k_nope, k_rope, group_size: int = 128):
    """Reference small-batch kernel with separate packed NoPE and RoPE outputs."""
    num_tokens, dim_nope = k_nope.shape
    num_tokens_, dim_rope = k_rope.shape
    assert num_tokens == num_tokens_, "k_nope and k_rope must have same num_tokens"

    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()
    num_tiles = dim_nope // group_size

    nope_part_bytes = dim_nope + num_tiles * 4
    rope_part_bytes = dim_rope * k_rope.element_size()
    nope_part_u8 = torch.empty(
        (num_tokens, nope_part_bytes), dtype=torch.uint8, device=k_nope.device
    )
    rope_part_u8 = torch.empty(
        (num_tokens, rope_part_bytes), dtype=torch.uint8, device=k_rope.device
    )

    nope_q_view = nope_part_u8[:, :dim_nope].view(get_available_fp8_dtype())
    nope_s_view = nope_part_u8[:, dim_nope:].view(torch.float32)
    if rope_part_bytes > 0:
        rope_view = rope_part_u8.view(torch.bfloat16)
    else:
        # NoPE: a fresh (N, 0) uint8 tensor has stride(0)=1, which torch
        # refuses to view as bf16; hand the kernel an empty bf16 tensor
        # instead (never dereferenced when DIM_ROPE == 0).
        rope_view = torch.empty(
            (num_tokens, 0), dtype=torch.bfloat16, device=k_rope.device
        )

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    NUM_NOPE_BLOCKS = dim_nope // group_size
    _quantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        nope_q_view,
        nope_s_view,
        rope_view,
        k_nope,
        k_rope,
        nope_q_view.stride(0),
        nope_s_view.stride(0),
        rope_view.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        FP8_MIN=torch.finfo(get_available_fp8_dtype()).min,
        FP8_MAX=torch.finfo(get_available_fp8_dtype()).max,
    )
    return nope_part_u8.unsqueeze(1), rope_part_u8.unsqueeze(1)


def quantize_k_cache_separate_optimized(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    tile_size: int = 128,
    block_m: int | None = None,
    num_warps: int | None = None,
):
    """Quantize FP8 MLA K cache with multiple tokens handled per program."""
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    if k_rope is None or k_rope.numel() == 0:
        # No-PE MLA (e.g. GLM-5.3: rope dim 0, 528-byte rows). The multi-token
        # optimized kernels below assume the 656-byte rope layout, so route to
        # the generic per-token kernel, which handles DIM_ROPE=0 natively.
        k_rope_2d = torch.empty(
            (k_nope_2d.shape[0], 0), dtype=k_nope_2d.dtype, device=k_nope_2d.device
        )
        return _quantize_k_cache_fast_separate(k_nope_2d, k_rope_2d, group_size=tile_size)
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    if k_nope_2d.ndim != 2 or k_nope_2d.shape[1] != DIM_NOPE:
        raise ValueError(
            f"Expected k_nope shape [N, {DIM_NOPE}], got {tuple(k_nope_2d.shape)}"
        )
    if k_rope_2d.ndim != 2 or k_rope_2d.shape[1] != DIM_ROPE:
        raise ValueError(
            f"Expected k_rope shape [N, {DIM_ROPE}], got {tuple(k_rope_2d.shape)}"
        )
    if k_nope_2d.shape[0] != k_rope_2d.shape[0]:
        raise ValueError(
            "k_nope and k_rope must have the same number of tokens, got "
            f"{k_nope_2d.shape[0]} and {k_rope_2d.shape[0]}"
        )
    if tile_size != GROUP_SIZE:
        raise ValueError(
            f"The optimized kernel currently requires tile_size={GROUP_SIZE}, "
            f"got {tile_size}"
        )
    if block_m is not None and block_m not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError(f"Unsupported block_m={block_m}")
    if num_warps is not None and num_warps not in (1, 2, 4):
        raise ValueError(f"Unsupported num_warps={num_warps}")

    # Decode shapes are dominated by fixed allocation/launch overhead on C500.
    # Use the original kernel unless a configuration is explicitly requested.
    if block_m is None and num_warps is None and k_nope_2d.shape[0] <= 64:
        return _quantize_k_cache_fast_separate(
            k_nope_2d, k_rope_2d, group_size=tile_size
        )

    # On C500, processing all four groups in one program makes the four FP32
    # scale stores contiguous and is faster for large prefills.
    if block_m is None and num_warps is None and k_nope_2d.shape[0] >= 8192:
        return _quantize_k_cache_separate_all_groups(
            k_nope_2d, k_rope_2d, block_m=4, num_warps=1
        )

    return _quantize_k_cache_fast_separate_optimized(
        k_nope_2d,
        k_rope_2d,
        block_m=block_m,
        num_warps=num_warps,
    )


@triton.jit
def _quantize_k_cache_fast_kernel_optimized(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    num_tokens,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_block_id = tl.program_id(0)
    group_id = tl.program_id(1)

    token_offsets = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    group_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    token_mask = token_offsets < num_tokens

    input_offsets = token_offsets[:, None] * k_nope_stride_0 + group_offsets[None, :]
    values = tl.load(
        k_nope_ptr + input_offsets,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    max_abs = tl.max(tl.abs(values), axis=1)
    scales = max_abs / FP8_MAX
    safe_scales = tl.where(max_abs > 0.0, scales, 1.0)
    quantized = tl.clamp(values / safe_scales[:, None], FP8_MIN, FP8_MAX).to(
        output_nope_q_ptr.dtype.element_ty
    )

    output_q_offsets = (
        token_offsets[:, None] * output_nope_q_stride_0 + group_offsets[None, :]
    )
    output_s_offsets = token_offsets * output_nope_s_stride_0 + group_id
    tl.store(
        output_nope_q_ptr + output_q_offsets,
        quantized,
        mask=token_mask[:, None],
    )
    tl.store(
        output_nope_s_ptr + output_s_offsets,
        safe_scales,
        mask=token_mask,
    )

    if group_id == 0:
        rope_offsets = tl.arange(0, DIM_ROPE)
        rope_input_offsets = (
            token_offsets[:, None] * k_rope_stride_0 + rope_offsets[None, :]
        )
        rope_output_offsets = (
            token_offsets[:, None] * output_rope_stride_0 + rope_offsets[None, :]
        )
        rope = tl.load(
            k_rope_ptr + rope_input_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )
        tl.store(
            output_rope_ptr + rope_output_offsets,
            rope,
            mask=token_mask[:, None],
        )


@triton.jit
def _quantize_k_cache_all_groups_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    num_tokens,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    BLOCK_M: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_block_id = tl.program_id(0)
    token_offsets = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    nope_offsets = tl.arange(0, DIM_NOPE)
    token_mask = token_offsets < num_tokens

    input_offsets = token_offsets[:, None] * k_nope_stride_0 + nope_offsets[None, :]
    values = tl.load(
        k_nope_ptr + input_offsets,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    values_grouped = tl.reshape(values, (BLOCK_M, NUM_GROUPS, GROUP_SIZE))

    max_abs = tl.max(tl.abs(values_grouped), axis=2)
    scales = max_abs / FP8_MAX
    safe_scales = tl.where(max_abs > 0.0, scales, 1.0)
    quantized_grouped = tl.clamp(
        values_grouped / safe_scales[:, :, None], FP8_MIN, FP8_MAX
    ).to(output_nope_q_ptr.dtype.element_ty)
    quantized = tl.reshape(quantized_grouped, (BLOCK_M, DIM_NOPE))

    output_q_offsets = (
        token_offsets[:, None] * output_nope_q_stride_0 + nope_offsets[None, :]
    )
    group_offsets = tl.arange(0, NUM_GROUPS)
    output_s_offsets = (
        token_offsets[:, None] * output_nope_s_stride_0 + group_offsets[None, :]
    )
    tl.store(
        output_nope_q_ptr + output_q_offsets,
        quantized,
        mask=token_mask[:, None],
    )
    tl.store(
        output_nope_s_ptr + output_s_offsets,
        safe_scales,
        mask=token_mask[:, None],
    )

    rope_offsets = tl.arange(0, DIM_ROPE)
    rope_input_offsets = (
        token_offsets[:, None] * k_rope_stride_0 + rope_offsets[None, :]
    )
    rope_output_offsets = (
        token_offsets[:, None] * output_rope_stride_0 + rope_offsets[None, :]
    )
    rope = tl.load(
        k_rope_ptr + rope_input_offsets,
        mask=token_mask[:, None],
        other=0.0,
    )
    tl.store(
        output_rope_ptr + rope_output_offsets,
        rope,
        mask=token_mask[:, None],
    )


def _quantize_k_cache_separate_all_groups(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    block_m: int = 8,
    num_warps: int = 1,
):
    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()
    num_tokens = k_nope.shape[0]
    nope_part_u8 = torch.empty(
        (num_tokens, DIM_NOPE + NUM_NOPE_BLOCKS * 4),
        dtype=torch.uint8,
        device=k_nope.device,
    )
    rope_part_u8 = torch.empty(
        (num_tokens, DIM_ROPE * k_rope.element_size()),
        dtype=torch.uint8,
        device=k_rope.device,
    )
    fp8_dtype = get_available_fp8_dtype()
    nope_q_view = nope_part_u8[:, :DIM_NOPE].view(fp8_dtype)
    nope_s_view = nope_part_u8[:, DIM_NOPE:].view(torch.float32)
    rope_view = rope_part_u8.view(torch.bfloat16)

    _quantize_k_cache_all_groups_kernel[(triton.cdiv(num_tokens, block_m),)](
        nope_q_view,
        nope_s_view,
        rope_view,
        k_nope,
        k_rope,
        num_tokens,
        nope_q_view.stride(0),
        nope_s_view.stride(0),
        rope_view.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        BLOCK_M=block_m,
        NUM_GROUPS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=GROUP_SIZE,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        FP8_MIN=torch.finfo(fp8_dtype).min,
        FP8_MAX=torch.finfo(fp8_dtype).max,
        num_warps=num_warps,
    )
    return nope_part_u8.unsqueeze(1), rope_part_u8.unsqueeze(1)


def _quantize_k_cache_fast_separate_optimized(
    k_nope,
    k_rope,
    block_m: int | None = None,
    num_warps: int | None = None,
):
    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()
    num_tokens = k_nope.shape[0]

    nope_part_u8 = torch.empty(
        (num_tokens, DIM_NOPE + NUM_NOPE_BLOCKS * 4),
        dtype=torch.uint8,
        device=k_nope.device,
    )
    rope_part_u8 = torch.empty(
        (num_tokens, DIM_ROPE * k_rope.element_size()),
        dtype=torch.uint8,
        device=k_rope.device,
    )

    fp8_dtype = get_available_fp8_dtype()
    nope_q_view = nope_part_u8[:, :DIM_NOPE].view(fp8_dtype)
    nope_s_view = nope_part_u8[:, DIM_NOPE:].view(torch.float32)
    rope_view = rope_part_u8.view(torch.bfloat16)

    if block_m is None:
        if num_tokens <= 4096:
            block_m = 4
        else:
            block_m = 32
    if num_warps is None:
        num_warps = 1

    grid = (triton.cdiv(num_tokens, block_m), NUM_NOPE_BLOCKS)
    _quantize_k_cache_fast_kernel_optimized[grid](
        nope_q_view,
        nope_s_view,
        rope_view,
        k_nope,
        k_rope,
        num_tokens,
        nope_q_view.stride(0),
        nope_s_view.stride(0),
        rope_view.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        BLOCK_M=block_m,
        GROUP_SIZE=GROUP_SIZE,
        DIM_ROPE=DIM_ROPE,
        FP8_MIN=torch.finfo(fp8_dtype).min,
        FP8_MAX=torch.finfo(fp8_dtype).max,
        num_warps=num_warps,
    )
    return nope_part_u8.unsqueeze(1), rope_part_u8.unsqueeze(1)


def pack_kv_cache(
    nope_part_u8: torch.Tensor, rope_part_u8: torch.Tensor
) -> torch.Tensor:
    """Pack separate operator outputs into the runtime [N, 1, 656] FP8 view."""
    assert nope_part_u8.shape[:-1] == rope_part_u8.shape[:-1]
    assert nope_part_u8.shape[-1] == 528
    assert rope_part_u8.shape[-1] == 128
    packed_u8 = torch.cat((nope_part_u8, rope_part_u8), dim=-1)
    return packed_u8.view(get_available_fp8_dtype())


def dequantize_k_cache_paged_optimized(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Gather and dequantize a paged FP8 K cache with one program per token."""
    dim_quant = quant_k_cache.shape[-1]
    # 528-byte rows are NoPE MLA (rope dim 0, e.g. GLM-5.3); 656-byte rows
    # carry 64 bf16 rope values. Derive the rope width from the row size.
    _nope_scale_bytes = DIM_NOPE + NUM_NOPE_BLOCKS * 4
    dim_rope = (dim_quant - _nope_scale_bytes) // 2
    assert dim_rope in (0, DIM_ROPE) and dim_quant == _nope_scale_bytes + dim_rope * 2, (
        f"dim_quant: {dim_quant} (derived rope {dim_rope}) != {PACKED_BYTES} or "
        "the NoPE 528-byte row width"
    )
    assert quant_k_cache.dtype == get_available_fp8_dtype()
    assert page_table_1_flattened.ndim == 1
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    assert page_table_1_flattened.is_contiguous()
    assert group_size == GROUP_SIZE

    quant_k_cache = quant_k_cache.view((-1, dim_quant))
    num_tokens = page_table_1_flattened.shape[0]
    output = torch.empty(
        (num_tokens, 1, DIM_NOPE + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if num_tokens == 0:
        return output

    input_nope_q = quant_k_cache[:, :DIM_NOPE]
    input_nope_s = quant_k_cache[:, DIM_NOPE : DIM_NOPE + NUM_NOPE_BLOCKS * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, DIM_NOPE + NUM_NOPE_BLOCKS * 4 :].view(torch.bfloat16)

    # Keep 32-bit address arithmetic until the largest FP8 row can overflow it.
    input_nope_q_stride = input_nope_q.stride(0)
    use_int64_input_index = (
        input_nope_q_stride != 0
        and quant_k_cache.shape[0] > (2**31 - 1) // input_nope_q_stride
    )
    # BF16 output offsets exceed int32 once num_tokens reaches 3,728,271.
    use_int64_output_index = num_tokens > (2**31 - 1) // (DIM_NOPE + dim_rope)
    _dequantize_k_cache_paged_optimized_kernel[(num_tokens,)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        OUTPUT_STRIDE=output.stride(0),
        INPUT_NOPE_Q_STRIDE=input_nope_q_stride,
        INPUT_NOPE_S_STRIDE=input_nope_s.stride(0),
        INPUT_ROPE_STRIDE=input_rope.stride(0),
        GROUP_SIZE=group_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=dim_rope,
        USE_INT64_INPUT_INDEX=use_int64_input_index,
        USE_INT64_OUTPUT_INDEX=use_int64_output_index,
        num_warps=1,
        num_stages=1,
    )
    return output


@triton.jit
def _dequantize_k_cache_paged_optimized_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    OUTPUT_STRIDE: tl.constexpr,
    INPUT_NOPE_Q_STRIDE: tl.constexpr,
    INPUT_NOPE_S_STRIDE: tl.constexpr,
    INPUT_ROPE_STRIDE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    USE_INT64_INPUT_INDEX: tl.constexpr,
    USE_INT64_OUTPUT_INDEX: tl.constexpr,
):
    token_id = tl.program_id(0)
    paged_token_id = tl.load(page_table_1_ptr + token_id)
    paged_token_id = (
        paged_token_id.to(tl.int64)
        if USE_INT64_INPUT_INDEX
        else paged_token_id.to(tl.int32)
    )
    output_token_id = token_id.to(tl.int64) if USE_INT64_OUTPUT_INDEX else token_id

    nope_offsets = tl.arange(0, DIM_NOPE)
    quantized = tl.load(
        input_nope_q_ptr + paged_token_id * INPUT_NOPE_Q_STRIDE + nope_offsets
    ).to(tl.float32)
    scales = tl.load(
        input_nope_s_ptr
        + paged_token_id * INPUT_NOPE_S_STRIDE
        + nope_offsets // GROUP_SIZE
    )
    tl.store(
        output_ptr + output_token_id * OUTPUT_STRIDE + nope_offsets,
        quantized * scales,
    )

    if DIM_ROPE > 0:
        rope_offsets = tl.arange(0, DIM_ROPE)
        rope = tl.load(
            input_rope_ptr + paged_token_id * INPUT_ROPE_STRIDE + rope_offsets
        )
        tl.store(
            output_ptr + output_token_id * OUTPUT_STRIDE + DIM_NOPE + rope_offsets,
            rope,
        )


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference paged dequantization for comparison with the optimized kernel."""
    dim_quant = quant_k_cache.shape[-1]
    assert (
        dim_quant == 656
    ), f"dim_quant: {dim_quant} != 656 detected in dequantize_k_cache_paged"
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    # Prefix sharing can make num_tokens exceed the physical cache capacity.
    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == get_available_fp8_dtype()
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size

    output = torch.empty(
        (num_tokens, 1, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5
    assert dim_nope % group_size == 0

    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    _dequantize_k_cache_paged_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )
    return output


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int64)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        effective_block_id = raw_block_id
        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = (
            input_nope_s_ptr
            + token_id_paged * input_nope_s_stride_0
            + effective_block_id
        )

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)
        y = (y_q * y_s).to(output_ptr.dtype.element_ty)
        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs
        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)
