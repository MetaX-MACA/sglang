# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#       Ported into sglang (math kept pixel-identical to the vLLM-metax kernel).

import math
from typing import Any

import torch

from .utils import is_pow2

_TILELANG_ERROR: str | None = None
try:
    import tilelang
    import tilelang.language as T
except Exception as exc:  # pragma: no cover - depends on host install
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _TILELANG_ERROR = str(exc)


def _cuda_like() -> bool:
    if _TILELANG_ERROR is not None:
        return False
    if torch.cuda.is_available():
        return True
    # MACA / other torch backends with a CUDA-alike stream
    backend = getattr(torch, "backend", None)
    return hasattr(backend, "maca") if backend is not None else False


pass_configs: dict[Any, Any] = {}
if tilelang is not None:
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if _cuda_like() and torch.version.cuda:
        pass_configs[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] = 10


@tilelang.jit(pass_configs=pass_configs)
def make_composite_hadamard_kernel(
    hidden_size: int,
    base_order: int,
    inverse: bool,
    dtype: str = "bfloat16",
    num_threads: int = 512,
    mma_rows: int = 128,
):
    """Specialize a K x P transform with H16 MMA and wave64 H32/H64."""
    if num_threads <= 0 or num_threads % 64:
        raise ValueError("num_threads must be a positive multiple of 64")
    if mma_rows <= 0 or mma_rows % 16:
        raise ValueError("mma_rows must be a positive multiple of 16")
    if hidden_size % base_order:
        raise ValueError("hidden_size must be divisible by base_order")
    width = hidden_size // base_order
    if not is_pow2(width):
        raise ValueError("hidden_size / base_order must be a power of two")

    log_width = int(math.log2(width))
    load_iterations = (hidden_size + num_threads - 1) // num_threads
    pair_count = hidden_size // 2
    pair_iterations = (pair_count + num_threads - 1) // num_threads
    pairs_per_block = width // 2
    mma_batches = (hidden_size + mma_rows * 16 - 1) // (mma_rows * 16)
    wave_count = num_threads // 64
    wave_groups = hidden_size // 64
    wave_rounds = (wave_groups + wave_count - 1) // wave_count
    scale = 1.0 / math.sqrt(hidden_size)
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, hidden_size), dtype],
        sign: T.Tensor[(hidden_size,), T.int8],
        hk: T.Tensor[(base_order, base_order), T.float32],
        h16: T.Tensor[(16, 16), dtype],
        out: T.Tensor[(num_tokens, hidden_size), dtype],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as token:
            thread = T.get_thread_binding()
            shared = T.alloc_shared((hidden_size,), T.float32)

            for iteration in T.serial(load_iterations):
                index = thread + iteration * num_threads
                if index < hidden_size:
                    value = T.cast(x[token, index], T.float32)
                    if not inverse:
                        value *= T.cast(sign[index], T.float32)
                    shared[index] = value
            T.sync_threads()

            if width >= 16:
                h16_shared = T.alloc_shared((16, 16), dtype)
                mma_input = T.alloc_shared((mma_rows, 16), dtype)
                mma_output = T.alloc_fragment((mma_rows, 16), T.float32)
                for row, column in T.Parallel(16, 16):
                    h16_shared[row, column] = h16[row, column]
                T.sync_threads()

                for batch in T.serial(mma_batches):
                    for row, column in T.Parallel(mma_rows, 16):
                        index = (batch * mma_rows + row) * 16 + column
                        if index < hidden_size:
                            mma_input[row, column] = T.cast(shared[index], dtype)
                        else:
                            mma_input[row, column] = T.cast(0.0, dtype)
                    T.sync_threads()
                    T.clear(mma_output)
                    T.gemm(mma_input, h16_shared, mma_output, transpose_B=True)
                    for row, column in T.Parallel(mma_rows, 16):
                        index = (batch * mma_rows + row) * 16 + column
                        if index < hidden_size:
                            shared[index] = mma_output[row, column]
                    T.sync_threads()
                first_shared_stage = 4
            else:
                first_shared_stage = 0

            if width >= 64:
                wave = thread // 64
                lane = thread - wave * 64
                for round_index in T.serial(wave_rounds):
                    group = round_index * wave_count + wave
                    if group < wave_groups:
                        value = shared[group * 64 + lane]
                        partner = T.shfl_xor(value, 16, 64)
                        stage32 = T.if_then_else(
                            (lane & 16) == 0, value + partner, partner - value
                        )
                        partner = T.shfl_xor(stage32, 32, 64)
                        stage64 = T.if_then_else(
                            (lane & 32) == 0,
                            stage32 + partner,
                            partner - stage32,
                        )
                        shared[group * 64 + lane] = stage64
                T.sync_threads()
                first_shared_stage = 6

            for stage_offset in T.unroll(log_width - first_shared_stage):
                stage = first_shared_stage + stage_offset
                stride = 1 << stage
                for iteration in T.serial(pair_iterations):
                    pair_id = thread + iteration * num_threads
                    if pair_id < pair_count:
                        block = pair_id // pairs_per_block
                        local_pair = pair_id - block * pairs_per_block
                        group = local_pair // stride
                        offset = local_pair - group * stride
                        left_index = block * width + group * (2 * stride) + offset
                        right_index = left_index + stride
                        left, right = shared[left_index], shared[right_index]
                        shared[left_index] = left + right
                        shared[right_index] = left - right
                T.sync_threads()

            if base_order == 1:
                for iteration in T.serial(load_iterations):
                    index = thread + iteration * num_threads
                    if index < hidden_size:
                        value = shared[index] * scale
                        if inverse:
                            value *= T.cast(sign[index], T.float32)
                        out[token, index] = T.cast(value, dtype)
            else:
                mixed = T.alloc_shared((hidden_size,), T.float32)
                for iteration in T.serial(load_iterations):
                    index = thread + iteration * num_threads
                    if index < hidden_size:
                        output_block = index // width
                        position = index - output_block * width
                        accumulator = T.alloc_var(T.float32)
                        accumulator = 0.0
                        for input_block in T.serial(base_order):
                            if inverse:
                                coefficient = hk[output_block, input_block]
                            else:
                                coefficient = hk[input_block, output_block]
                            accumulator += (
                                shared[input_block * width + position] * coefficient
                            )
                        mixed[index] = accumulator
                for iteration in T.serial(load_iterations):
                    index = thread + iteration * num_threads
                    if index < hidden_size:
                        value = mixed[index] * scale
                        if inverse:
                            value *= T.cast(sign[index], T.float32)
                        out[token, index] = T.cast(value, dtype)

    return kernel


__all__ = ["make_composite_hadamard_kernel", "tilelang", "T", "_TILELANG_ERROR"]