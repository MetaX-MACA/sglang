import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.srt.utils import is_hip

_is_hip = is_hip()


def quantize_k_cache_int8_separate(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    group_size: int = 128,
):
    k_nope = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope
    if k_nope.ndim != 2 or k_nope.shape[1] != 512:
        raise ValueError(f"Expected k_nope shape (num_tokens, 512), got {k_nope.shape}")
    if k_rope.ndim != 2 or k_rope.shape[1] != 64:
        raise ValueError(f"Expected k_rope shape (num_tokens, 64), got {k_rope.shape}")
    if k_nope.shape[0] != k_rope.shape[0]:
        raise ValueError("k_nope and k_rope must have the same number of tokens")
    if k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16:
        raise TypeError("INT8 MLA KV cache expects BF16 inputs")
    if group_size != 128:
        raise ValueError(f"INT8 MLA KV cache requires group_size=128, got {group_size}")

    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()
    num_tokens = k_nope.shape[0]
    nope_part = torch.empty((num_tokens, 528), dtype=torch.uint8, device=k_nope.device)
    rope_part = torch.empty((num_tokens, 128), dtype=torch.uint8, device=k_rope.device)
    if num_tokens == 0:
        return nope_part.unsqueeze(1), rope_part.unsqueeze(1)

    nope_q = nope_part[:, :512].view(torch.int8)
    nope_s = nope_part[:, 512:].view(torch.float32)
    rope_bf16 = rope_part.view(torch.bfloat16)
    _quantize_k_cache_int8_kernel[(num_tokens, 5)](
        nope_q,
        nope_s,
        rope_bf16,
        k_nope,
        k_rope,
        nope_q.stride(0),
        nope_s.stride(0),
        rope_bf16.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        IS_HIP=_is_hip,
        GROUP_SIZE=128,
    )
    return nope_part.unsqueeze(1), rope_part.unsqueeze(1)


@triton.jit
def _quantize_k_cache_int8_kernel(
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
    IS_HIP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    group_id = tl.program_id(1)
    offs = tl.arange(0, GROUP_SIZE)
    if group_id < 4:
        dim_offs = group_id * GROUP_SIZE + offs
        values = tl.load(k_nope_ptr + token_id * k_nope_stride_0 + dim_offs).to(
            tl.float32
        )
        abs_max = tl.max(tl.abs(values))
        scale = tl.maximum(abs_max, 1.0e-4) / 127.0
        quant = values / scale
        if IS_HIP:
            quant = libdevice.round(quant)
        else:
            quant = tl.extra.cuda.libdevice.round(quant)
        quant = tl.maximum(tl.minimum(quant, 127.0), -127.0).to(tl.int8)
        tl.store(
            output_nope_q_ptr + token_id * output_nope_q_stride_0 + dim_offs,
            quant,
        )
        tl.store(
            output_nope_s_ptr + token_id * output_nope_s_stride_0 + group_id,
            scale,
        )
    else:
        rope = tl.load(k_rope_ptr + token_id * k_rope_stride_0 + offs, mask=offs < 64)
        tl.store(
            output_rope_ptr + token_id * output_rope_stride_0 + offs,
            rope,
            mask=offs < 64,
        )


def dequantize_k_cache_paged_int8(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
    num_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    dim_quant = quant_k_cache.shape[-1]
    assert dim_quant == 656
    assert quant_k_cache.dtype == torch.int8
    assert group_size == 128
    quant_k_cache = quant_k_cache.view((-1, dim_quant))
    valid_num_tokens = num_tokens
    num_tokens = page_table_1_flattened.shape[0]
    output = torch.empty(
        (num_tokens, 1, 576),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    if num_tokens == 0:
        return output

    input_nope_q = quant_k_cache[:, :512]
    input_nope_s = quant_k_cache[:, 512:528].view(torch.float32)
    input_rope = quant_k_cache[:, 528:].view(torch.bfloat16)
    _dequantize_k_cache_paged_int8_kernel[(num_tokens, 5)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        valid_num_tokens if valid_num_tokens is not None else page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        GROUP_SIZE=group_size,
        USE_NUM_TOKENS_PTR=valid_num_tokens is not None,
    )
    return output


def dequantize_k_cache_paged_int8_to_workspace(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    output: torch.Tensor,
    num_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    assert quant_k_cache.shape[-1] == 656
    assert quant_k_cache.dtype == torch.int8
    assert output.shape[-1] == 576
    assert output.dtype == torch.bfloat16
    quant_k_cache = quant_k_cache.view((-1, 656))
    output = output.view((-1, 576))
    if page_table_1_flattened.numel() == 0:
        return output

    input_nope_q = quant_k_cache[:, :512]
    input_nope_s = quant_k_cache[:, 512:528].view(torch.float32)
    input_rope = quant_k_cache[:, 528:].view(torch.bfloat16)
    num_programs = min(32, page_table_1_flattened.numel())
    num_tokens_ptr = (
        num_tokens.reshape(-1) if num_tokens is not None else page_table_1_flattened
    )
    _dequantize_k_cache_paged_int8_to_workspace_kernel[(num_programs, 5)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        num_tokens_ptr,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        STATIC_NUM_TOKENS=(
            0 if num_tokens is not None else page_table_1_flattened.numel()
        ),
        USE_NUM_TOKENS_PTR=num_tokens is not None,
        GROUP_SIZE=128,
    )
    return output


@triton.jit
def _dequantize_k_cache_paged_int8_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    num_tokens_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    GROUP_SIZE: tl.constexpr,
    USE_NUM_TOKENS_PTR: tl.constexpr,
):
    token_id = tl.program_id(0)
    valid = True
    if USE_NUM_TOKENS_PTR:
        valid = token_id < tl.load(num_tokens_ptr)
    token_id_paged = tl.load(page_table_1_ptr + token_id, mask=valid, other=0).to(
        tl.int32
    )
    group_id = tl.program_id(1)
    offs = tl.arange(0, GROUP_SIZE)

    if group_id < 4:
        dim_offs = group_id * GROUP_SIZE + offs
        quant = tl.load(
            input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + dim_offs,
            mask=valid,
            other=0,
        ).to(tl.float32)
        scale = tl.load(
            input_nope_s_ptr + token_id_paged * input_nope_s_stride_0 + group_id,
            mask=valid,
            other=0,
        )
        tl.store(
            output_ptr + token_id * output_stride_0 + dim_offs,
            (quant * scale).to(output_ptr.dtype.element_ty),
        )
    else:
        rope = tl.load(
            input_rope_ptr + token_id_paged * input_rope_stride_0 + offs,
            mask=(offs < 64) & valid,
            other=0,
        )
        tl.store(
            output_ptr + token_id * output_stride_0 + 512 + offs,
            rope,
            mask=offs < 64,
        )


@triton.jit
def _dequantize_k_cache_paged_int8_to_workspace_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    num_tokens_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    STATIC_NUM_TOKENS: tl.constexpr,
    USE_NUM_TOKENS_PTR: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    token_offset = tl.program_id(0)
    group_id = tl.program_id(1)
    if USE_NUM_TOKENS_PTR:
        num_tokens = tl.load(num_tokens_ptr).to(tl.int64)
    else:
        num_tokens = STATIC_NUM_TOKENS
    offs = tl.arange(0, GROUP_SIZE)

    while token_offset < num_tokens:
        token_id = tl.load(page_table_1_ptr + token_offset).to(tl.int64)
        if group_id < 4:
            dim_offs = group_id * GROUP_SIZE + offs
            quant = tl.load(
                input_nope_q_ptr + token_id * input_nope_q_stride_0 + dim_offs
            ).to(tl.float32)
            scale = tl.load(
                input_nope_s_ptr + token_id * input_nope_s_stride_0 + group_id
            )
            tl.store(
                output_ptr + token_id * output_stride_0 + dim_offs,
                (quant * scale).to(output_ptr.dtype.element_ty),
            )
        else:
            rope = tl.load(
                input_rope_ptr + token_id * input_rope_stride_0 + offs,
                mask=offs < 64,
            )
            tl.store(
                output_ptr + token_id * output_stride_0 + 512 + offs,
                rope,
                mask=offs < 64,
            )
        token_offset += tl.num_programs(0)
