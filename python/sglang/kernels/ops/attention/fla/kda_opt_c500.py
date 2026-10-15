# SPDX-License-Identifier: Apache-2.0
"""Self-contained C500/MetaX-C550-optimized KDA Triton kernels (fast path).

This module is a standalone copy of the six optimized KDA/GDN operators from the
C500_triton_kernels_opt workspace (commit 40fa642b52, "perf(kda): integrate 6
optimized Triton kernels"). It intentionally does NOT import the five
production modules it replaces fast paths for (chunk_delta_h / chunk_intra /
fused_sigmoid_gating_recurrent / kda / causal_conv1d_triton), so those modules
can import this file's switch helper without an import cycle.

Everything here is dead code unless the master switch is on:

    MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS=1|true|on|yes   (case-insensitive)

The switch is re-read from os.environ on every call (default: off). With the
switch off, the production modules never call into this file and behavior is
byte-identical to the unoptimized kernels.

The switch is mutually exclusive with MX_USE_MCOPLIB_KDA (the mcoplib KDA
kernels): setting both truthy raises RuntimeError on the first guarded
dispatch (fail-fast at warmup) rather than silently picking a winner.

Public entry points (same names and signatures as the production wrappers they
replace; called from the guards at the top of those wrappers):

  [1/6] chunk_gated_delta_rule_fwd_h          (fla/chunk_delta_h.py)      1.79x
  [2/6] chunk_kda_fwd_intra                   (fla/chunk_intra.py)        2.69x
  [3/6] fused_sigmoid_gating_delta_rule_update (fla/fused_sigmoid_gating_recurrent.py) 4.27x
  [4/6] chunk_gla_fwd_o_gk                    (fla/kda.py)                1.215x
  [5/6] kda_gate_chunk_cumsum                 (fla/kda.py)                1.485x
  [6/6] causal_conv1d_fn                      (mamba/causal_conv1d_triton.py) 1.334x

Measured speedups are per-operator kernel times on MetaX C550 (E2E +13.9%);
see the per-kernel REPORT.md / ITERATIONS.md in the source workspace.

Fine-grained fallbacks kept from the optimized versions (all under the master
switch above):
  * SGLANG_GDN_CHUNK_H_FAST=0 -> chunk_h falls back to the generic kernel
    (copied below) instead of the specialized K=V=128 fast kernels.
  * The fused_sigmoid fast path only covers retrieve_parent_token is None and
    not cache_ring and split_n_hv_grid; everything else uses the copied
    generic kernel with the same math.
  * The conv1d register-sliding path only engages for width==4 and
    max_seq_len >= 128.

NOTE: chunk_kda_fwd_intra keeps the production lazy import of
recompute_w_u_fwd from fla/kda.py at call time (function-level, no import
cycle; unchanged from the original module).
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as _drv
from triton.runtime.driver import driver as _triton_driver

from mcoplib.triton_sglang_chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.op import exp, exp2, gather, log, safe_exp
from sglang.kernels.ops.attention.fla.utils import (
    autotune_cache_kwargs,
    is_gather_supported,
    is_nvidia_hopper,
    is_tf32_supported,
)

_C500_KDA_OPTIMIZED_KERNELS_ENV = "MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS"
_MCOPLIB_KDA_ENV = "MX_USE_MCOPLIB_KDA"
_C500_KDA_OPT_TRUTHY = ("1", "true", "on", "yes")


def _env_truthy(name: str) -> bool:
    # Unset (or any value outside the truthy set) counts as off, matching the
    # EnvBool(False) default of MX_USE_MCOPLIB_KDA in sglang.srt.environ.
    return os.environ.get(name, "").lower() in _C500_KDA_OPT_TRUTHY


def _c500_kda_opt_enabled() -> bool:
    """Master switch for the C500-optimized KDA kernels (default: off).

    Read from the environment on every call so the flag can be flipped for a
    new process without any caching surprises; the read is a dict lookup.

    Mutually exclusive with the mcoplib KDA kernels (MX_USE_MCOPLIB_KDA):
    enabling both raises on the first guarded dispatch (fail-fast during
    warmup) instead of silently resolving the conflict. MX_USE_MCOPLIB_KDA
    unset or =0 is not a conflict; with the master switch unset nothing here
    raises and behavior is unchanged.
    """
    enabled = _env_truthy(_C500_KDA_OPTIMIZED_KERNELS_ENV)
    if enabled and _env_truthy(_MCOPLIB_KDA_ENV):
        raise RuntimeError(
            f"{_C500_KDA_OPTIMIZED_KERNELS_ENV} and {_MCOPLIB_KDA_ENV} are "
            "mutually exclusive; enable exactly one of them:\n"
            "  - optimized C500 Triton kernels: "
            "MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS=1 with MX_USE_MCOPLIB_KDA "
            "unset or =0\n"
            "  - mcoplib KDA kernels: MX_USE_MCOPLIB_KDA=1 with "
            "MX_ENABLE_C500_KDA_OPTIMIZED_KERNELS unset or =0"
        )
    return enabled


# ============================================================================
# [1/6] chunk_gated_delta_rule_fwd_h -- source: fla/chunk_delta_h.py
# ============================================================================
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
CHUNK_SIZE = 64
GDN_CHUNK_H_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_BV", "32"))
GDN_CHUNK_H_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", "4"))
GDN_CHUNK_H_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "1"))
# C550 fast-path knobs (see chunk_gated_delta_rule_fwd_kernel_h_blockdim64_c550):
# K=V=128 extend configuration takes a specialized transposed-state kernel with a
# slim signature and direct launch (no autotuner host overhead; the small-T
# launches are host-bound). SGLANG_GDN_CHUNK_H_FAST=0 falls back to the generic
# kernel unconditionally.
GDN_CHUNK_H_FAST = os.getenv("SGLANG_GDN_CHUNK_H_FAST", "1") == "1"
GDN_CHUNK_H_FAST_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_FAST_BV", "16"))
GDN_CHUNK_H_FAST_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_FAST_NUM_WARPS", "4"))
GDN_CHUNK_H_FAST_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_FAST_NUM_STAGES", "1"))
GDN_CHUNK_H_FAST_PIPELINE = os.getenv("SGLANG_GDN_CHUNK_H_FAST_PIPELINE", "cpasync-mixed")
# Direct-launch cache: after the first JIT launch of a given specialization,
# keep the CompiledKernel's own runner and bypass JITFunction.run entirely
# (binder + cache-key computation + launch metadata are pure per-call host
# overhead; small-T prefill launches are host-bound). Keyed on everything the
# compiled binary depends on: device, NB variant, grid, and the specialization
# class of the runtime int arg. All tensor arguments are top-level torch
# allocations in this path (>=256B aligned), so pointer-alignment
# specialization is stable.
_DIRECT_LAUNCH_CACHE = {}
_FAST_GX = triton.cdiv(128, GDN_CHUNK_H_FAST_BV)


@triton.autotune(
    # Single hardcoded config. The kernel writes ht (final state) back into
    # initial_state in-place; with multiple configs, triton's autotune benchmark
    # phase invokes the kernel many times for timing and corrupts the cache pool,
    # producing silently wrong output on the first user request. Restoring via
    # `restore_value=["initial_state"]` works for unit tests but OOMs on
    # production-scale models (e.g. Kimi-Linear-48B at default mem_fraction)
    # because cloning the cache pool for each benchmark exceeds available memory.
    # NT_BUCKET is kept in the autotune key for forward-compatibility (allows
    # future per-bucket configs once the kernel is refactored to write final
    # state to a separate output buffer). The env knobs keep this single-config
    # property while allowing model/hardware-local validation of the selected
    # tile without corrupting the state pool through multi-config autotune.
    configs=[
        triton.Config(
            {"BV": GDN_CHUNK_H_BV},
            num_warps=GDN_CHUNK_H_NUM_WARPS,
            num_stages=GDN_CHUNK_H_NUM_STAGES,
        )
    ],
    key=["H", "K", "V", "BT", "USE_GK", "NT_BUCKET"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    # Slot stride comes from the caller (initial_state.stride(0)): the state pool
    # may be an envelope-strided view (page-major / unified memory), where the
    # per-slot pitch spans ALL layers' state, not H*V*K. int64: envelope pitches
    # overflow an int32 index product.
    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    # Padded rows carry the -1 sentinel; the decode kernel guards on it
    # (fused_recurrent.py), the chunked extend path did not.
    valid_state = index >= 0
    h0 = initial_state + index * stride_init_state
    ht = initial_state + index * stride_init_state
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    # load initial state
    if USE_INITIAL_STATE and valid_state:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue
    if INPLACE_UPDATE and valid_state:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _store_h_opt(h, i_t, stride_h, b_h, i_v, BV: tl.constexpr):
    # Per (chunk, head) h is (V, K) row-major; a [K, BV] tile at column i_v*BV
    # is one fully contiguous BV*K*2-byte slab (4 KiB for BV=16).
    p = tl.make_block_ptr(
        h + i_t * stride_h, (128, 128), (1, 128), (0, i_v * BV), (128, BV), (0, 1)
    )
    tl.store(p, b_h.to(tl.bfloat16))


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_c550(
    k,
    v,
    w,
    v_new,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    BV: tl.constexpr,
    NB: tl.constexpr,
):
    # K=V=128, BT=64, USE_GK+exp2, initial state + in-place update + v_new:
    # all guaranteed by the wrapper dispatch; kept out of the signature on
    # purpose (host launch cost scales with argument count).
    K: tl.constexpr = 128
    V: tl.constexpr = 128
    BT: tl.constexpr = 64
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)
    gk += ((bos * H + i_h) * K).to(tl.int64)

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    state = initial_state + tl.maximum(index, 0) * stride_init_state + i_h * V * K
    # [K, BV] slab of the fp32 state pool, same transposed layout as b_h
    p_s = tl.make_block_ptr(state, (128, 128), (1, 128), (0, i_v * BV), (128, BV), (0, 1))
    o_k = tl.arange(0, 128)
    keep = tl.where(valid_state, 1.0, 0.0)
    b_h = tl.load(p_s).to(tl.float32) * keep

    # v tile of chunk 0, kept one chunk ahead in registers (PV=1)
    n_v = tl.load(
        tl.make_block_ptr(v, (T, V), (stride_v, 1), (0, i_v * BV), (BT, BV), (1, 0)),
        boundary_check=(0,),
    )

    # NB=1: main loop runs only over full chunks with no boundary checks
    # (masks are provably all-true there); the partial tail chunk is peeled.
    nmain = NT if NB == 0 else T // BT
    for i_t in range(nmain):
        if NB:
            b_w = tl.load(
                tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, K), (1, 0))
            )
            b_k = tl.load(
                tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (K, BT), (0, 1))
            )
        else:
            b_w = tl.load(
                tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, K), (1, 0)),
                boundary_check=(0,),
            )
            b_k = tl.load(
                tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (K, BT), (0, 1)),
                boundary_check=(1,),
            )
        b_vv = n_v
        nt = (i_t + 1) * BT
        n_v = tl.load(
            tl.make_block_ptr(v, (T, V), (stride_v, 1), (nt, i_v * BV), (BT, BV), (1, 0)),
            boundary_check=(0,),
        )
        last = tl.minimum(nt, T) - 1
        b_g = tl.load(gk + last * H * K + o_k)
        # store chunk state (HP=1: after the loads, before the dots)
        _store_h_opt(h, i_t, stride_h, b_h, i_v, BV)

        b_v = b_vv - tl.dot(b_w, b_h.to(tl.bfloat16))
        p_vn = tl.make_block_ptr(
            v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        if NB:
            tl.store(p_vn, b_v.to(tl.bfloat16))
        else:
            tl.store(p_vn, b_v.to(tl.bfloat16), boundary_check=(0,))

        b_h = b_h * exp2(b_g)[:, None] + tl.dot(b_k, b_v.to(tl.bfloat16))

    if NB and (T % BT) != 0:
        # peeled partial tail chunk (with boundary checks)
        i_t = nmain
        b_w = tl.load(
            tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, K), (1, 0)),
            boundary_check=(0,),
        )
        b_k = tl.load(
            tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (K, BT), (0, 1)),
            boundary_check=(1,),
        )
        b_vv = n_v
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_g = tl.load(gk + last * H * K + o_k)
        _store_h_opt(h, i_t, stride_h, b_h, i_v, BV)
        b_v = b_vv - tl.dot(b_w, b_h.to(tl.bfloat16))
        tl.store(
            tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            ),
            b_v.to(tl.bfloat16),
            boundary_check=(0,),
        )
        b_h = b_h * exp2(b_g)[:, None] + tl.dot(b_k, b_v.to(tl.bfloat16))

    if valid_state:
        tl.store(p_s, b_h.to(p_s.dtype.element_ty))


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_c550_nt1(
    k,
    v,
    w,
    v_new,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
):
    # Single-chunk (NT == 1) specialization: BV=16 / NB=0 baked in, no loop,
    # two fewer binder arguments -- the T<=64 prefill cases are host-launch
    # bound and every argument costs C-launcher parsing time.
    K: tl.constexpr = 128
    V: tl.constexpr = 128
    BT: tl.constexpr = 64
    BV: tl.constexpr = 16
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)
    gk += ((bos * H + i_h) * K).to(tl.int64)

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    state = initial_state + tl.maximum(index, 0) * stride_init_state + i_h * V * K
    p_s = tl.make_block_ptr(state, (128, 128), (1, 128), (0, i_v * BV), (128, BV), (0, 1))
    o_k = tl.arange(0, 128)
    keep = tl.where(valid_state, 1.0, 0.0)
    b_h = tl.load(p_s).to(tl.float32) * keep

    b_w = tl.load(
        tl.make_block_ptr(w, (T, K), (stride_w, 1), (0, 0), (BT, K), (1, 0)),
        boundary_check=(0,),
    )
    b_k = tl.load(
        tl.make_block_ptr(k, (K, T), (1, stride_k), (0, 0), (K, BT), (0, 1)),
        boundary_check=(1,),
    )
    b_vv = tl.load(
        tl.make_block_ptr(v, (T, V), (stride_v, 1), (0, i_v * BV), (BT, BV), (1, 0)),
        boundary_check=(0,),
    )
    last = tl.minimum(BT, T) - 1
    b_g = tl.load(gk + last * H * K + o_k)
    _store_h_opt(h, 0, stride_h, b_h, i_v, BV)
    b_v = b_vv - tl.dot(b_w, b_h.to(tl.bfloat16))
    tl.store(
        tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (0, i_v * BV), (BT, BV), (1, 0)),
        b_v.to(tl.bfloat16),
        boundary_check=(0,),
    )
    b_h = b_h * exp2(b_g)[:, None] + tl.dot(b_k, b_v.to(tl.bfloat16))
    if valid_state:
        tl.store(p_s, b_h.to(p_s.dtype.element_ty))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    use_exp2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert not (
        use_exp2 and g is not None
    ), "use_exp2 covers only the per-channel gk path; scalar g stays natural-exp"
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    if (
        GDN_CHUNK_H_FAST
        and K == 128
        and V == 128
        and BT == 64
        and cu_seqlens is not None
        and gk is not None
        and g is None
        and use_exp2
        and initial_state is not None
        and initial_state_indices is not None
        and v_new is not None
    ):
        # C550 fast path: transposed [K, BV] state slabs + fused 128-wide dot
        # operands + cpasync-mixed pipeline. Direct positional launch: the
        # small-T cases are host-launch-bound, so no autotuner wrapper, no
        # intermediate dict and no dispatchable-but-unused flag arguments.
        # NB=1 (full chunks without boundary masks + peeled tail chunk) only
        # for long inputs; the peeled body doubles the kernel code, which
        # costs more than the masks save when NT is small.
        stride_init_state = initial_state.stride(0)
        gy = N * H
        nt1 = T <= 64
        dl_key = (
            torch.cuda.current_device(),
            2 if nt1 else (1 if T >= 2048 else 0),
            gy,
            stride_init_state == 1,
            stride_init_state % 16 == 0,
        )
        runner = _DIRECT_LAUNCH_CACHE.get(dl_key)
        if runner is None:
            if nt1:
                compiled = chunk_gated_delta_rule_fwd_kernel_h_blockdim64_c550_nt1[
                    (_FAST_GX, gy)
                ](
                    k,
                    u,
                    w,
                    v_new,
                    gk,
                    h,
                    initial_state,
                    initial_state_indices,
                    stride_init_state,
                    cu_seqlens,
                    chunk_offsets,
                    T,
                    H,
                    Hg,
                    num_warps=GDN_CHUNK_H_FAST_NUM_WARPS,
                    num_stages=GDN_CHUNK_H_FAST_NUM_STAGES,
                    pipeline=GDN_CHUNK_H_FAST_PIPELINE,
                )
            else:
                compiled = chunk_gated_delta_rule_fwd_kernel_h_blockdim64_c550[
                    (_FAST_GX, gy)
                ](
                    k,
                    u,
                    w,
                    v_new,
                    gk,
                    h,
                    initial_state,
                    initial_state_indices,
                    stride_init_state,
                    cu_seqlens,
                    chunk_offsets,
                    T,
                    H,
                    Hg,
                    GDN_CHUNK_H_FAST_BV,
                    1 if T >= 2048 else 0,
                    num_warps=GDN_CHUNK_H_FAST_NUM_WARPS,
                    num_stages=GDN_CHUNK_H_FAST_NUM_STAGES,
                    pipeline=GDN_CHUNK_H_FAST_PIPELINE,
                )
            _DIRECT_LAUNCH_CACHE[dl_key] = compiled[(_FAST_GX, gy, 1)]
        elif nt1:
            runner(
                k,
                u,
                w,
                v_new,
                gk,
                h,
                initial_state,
                initial_state_indices,
                stride_init_state,
                cu_seqlens,
                chunk_offsets,
                T,
                H,
                Hg,
            )
        else:
            runner(
                k,
                u,
                w,
                v_new,
                gk,
                h,
                initial_state,
                initial_state_indices,
                stride_init_state,
                cu_seqlens,
                chunk_offsets,
                T,
                H,
                Hg,
                GDN_CHUNK_H_FAST_BV,
                1 if T >= 2048 else 0,
            )
        return h, v_new

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != H*V*K; contiguous pools pass exactly H*V*K.
        stride_init_state=(initial_state.stride(0) if initial_state is not None else 0),
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_BUCKET=(0 if NT <= 32 else (1 if NT <= 128 else 2)),
        USE_EXP2=use_exp2,
    )
    return h, v_new


# ============================================================================
# [2/6] chunk_kda_fwd_intra -- source: fla/chunk_intra.py
# ============================================================================
if is_tf32_supported:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################
# Optimized variant (MetaX C550 tuning, see kda_triton_opt workspace):
#  * Aqk/Akk written as one full-row [BT, BT] tile (16-wide bf16 block stores are
#    partial-sector writes, ~6x slower on C550); the non-fused path keeps the
#    diagonal/upper Aqk blocks written by the previous kernel via a bitwise RMW select.
#  * non-fused (large-T) launches use a dedicated kernel whose inter products are
#    [BC, BT] row-block dots (6 large MMAs per K-tile instead of 20 [16,16] ones).
#  * fused (small-T) recompute is split into a second per-sub-chunk kernel when
#    T > BC (register pressure / latency), passing the merged inverse through an
#    internal bf16 workspace (bit-identical to the previous in-register cast).
#  * launches are pinned to the configs the production autotuner selected on C550
#    (recorded in the captures); @triton.autotune overhead is also removed.
#  * NSC constexpr specialization: sub-chunks beyond cdiv(min(T, BT), BC) are
#    compiled out (removes the worst-case register allocation; 0 spills).
#  * _fl_launch: cached CompiledKernel direct submission (skips the JIT binder
#    per-launch Python arg processing, ~20us/launch on this host), with alignment
#    revalidation and automatic fallback to the normal JIT path.
_INTER_SOLVE_WKS = {}

# ---------------------------------------------------------------------------
# Fast-launch path: after the first (normal) JIT launch of a given argument
# structure, reuse the compiled-kernel handle and submit directly with
# kernel.run(...), skipping the JITFunction binder/cache-key work (~20us per
# launch on this host). Falls back to the normal path whenever anything about
# the capture is unexpected, and validates pointer alignment per call (the only
# specialization dimension that can drift under a fixed key).
# ---------------------------------------------------------------------------
from triton.runtime import driver as _drv

_FL_CACHE = {}  # (kernel-name, key) -> (ck, fn, packed_metadata) | False

_SIG_FUSED = ("q", "k", "g", "beta", "Aqk", "Akkd", "Akk", "scale", "v_in", "w_out",
              "u_out", "kg_out", "wks", "cu_seqlens", "chunk_indices", "T", "H", "K", "V",
              "BT", "BC", "BK", "BV", "IS_VARLEN", "USE_SAFE_GATE", "FUSE_RECOMPUTE",
              "FUSE_DIAGONAL", "SPLIT_RECOMPUTE", "NSC")
_SIG_PHASEB = ("wks", "k", "g", "beta", "v_in", "w_out", "u_out", "kg_out", "cu_seqlens",
               "chunk_indices", "T", "H", "K", "V", "BT", "BC", "BK", "BV", "IS_VARLEN", "NSC")
_SIG_LARGE = ("q", "k", "g", "beta", "Aqk", "Akkd", "Akk", "scale", "v_in", "w_out",
              "u_out", "kg_out", "cu_seqlens", "chunk_indices", "T", "H", "K", "V",
              "BT", "BC", "BK", "BV", "IS_VARLEN", "USE_SAFE_GATE")


_FL_TENSOR_SLOTS = {}


def _fl_args_aligned(vals, slots):
    for i in slots:
        if vals[i].data_ptr() % 16:
            return False
    return True


def _fl_launch(jf, name, key, grid, vals, lopts):
    """vals: signature-ordered value list; lopts: launch options (warps/stages only)."""
    ent = _FL_CACHE.get((name, key))
    if ent is not None and ent is not False:
        slots = _FL_TENSOR_SLOTS.get(name)
        if slots is None:
            slots = _FL_TENSOR_SLOTS[name] = [i for i, v in enumerate(vals)
                                              if isinstance(v, torch.Tensor)]
        if not _fl_args_aligned(vals, slots):
            jf[grid](*vals, **lopts)
            return
        ck, fn, pmeta, dev = ent
        try:
            stream = _drv.active.get_current_stream(dev)
            ck.run(grid[0], grid[1], 1, stream, fn, pmeta, None, None, None, *vals)
            return
        except Exception:
            _FL_CACHE[(name, key)] = False
    elif ent is None:
        try:
            dev = _drv.active.get_current_device()
            caches = jf.device_caches[dev][0]
            before = set(caches.keys())
            jf[grid](*vals, **lopts)
            if len(caches) == 1:
                ck = next(iter(caches.values()))
            else:
                new = caches.keys() - before
                ck = caches[next(iter(new))] if len(new) == 1 else None
            if ck is None:
                _FL_CACHE[(name, key)] = False
                return
            ck._init_handles()
            _FL_CACHE[(name, key)] = (ck, ck.function, ck.packed_metadata, dev)
        except Exception:
            _FL_CACHE[(name, key)] = False
        return
    jf[grid](*vals, **lopts)

INTER_SOLVE_CONFIGS = {
    False: dict(BK=64, BV=64, num_warps=2, num_stages=3),
    True: dict(BK=64, BV=64, num_warps=4, num_stages=3),
}


@triton.jit
def _hcat(a, b):
    """[R, C] x 2 -> [R, 2C] (column concat)."""
    R: tl.constexpr = a.shape[0]
    C: tl.constexpr = a.shape[1]
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (R, 2 * C))


@triton.jit
def _vcat(a, b):
    """[R, C] x 2 -> [2R, C] (row concat)."""
    R: tl.constexpr = a.shape[0]
    C: tl.constexpr = a.shape[1]
    return tl.reshape(tl.permute(tl.join(a, b), (2, 0, 1)), (2 * R, C))


@triton.jit
def _hsplit(a):
    """[R, 2C] -> 2 x [R, C] (column split)."""
    R: tl.constexpr = a.shape[0]
    C2: tl.constexpr = a.shape[1]
    return tl.split(tl.permute(tl.reshape(a, (R, 2, C2 // 2)), (0, 2, 1)))


@triton.jit
def _tile4x4(b00, b10, b11, b20, b21, b22, b30, b31, b32, b33, z):
    """Assemble a lower-block-triangular [4*BC, 4*BC] tile from 16x16 blocks (upper blocks = z)."""
    r0 = _hcat(_hcat(b00, z), _hcat(z, z))
    r1 = _hcat(_hcat(b10, b11), _hcat(z, z))
    r2 = _hcat(_hcat(b20, b21), _hcat(b22, z))
    r3 = _hcat(_hcat(b30, b31), _hcat(b32, b33))
    return _vcat(_vcat(r0, r1), _vcat(r2, r3))


@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    v_in,
    w_out,
    u_out,
    kg_out,
    wks,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
    FUSE_RECOMPUTE: tl.constexpr,
    FUSE_DIAGONAL: tl.constexpr,
    SPLIT_RECOMPUTE: tl.constexpr,
    NSC: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    if FUSE_DIAGONAL:
        b_Aqk_d0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d3 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d3 = tl.zeros([BC, BC], dtype=tl.float32)
        m_tc0 = (i_tc0 + o_i) < T

    ################################################################################
    # off-diagonal blocks (+ optional diagonal blocks)
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        p_g0 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if FUSE_DIAGONAL:
            p_q0 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            b_q0 = tl.load(p_q0, boundary_check=(0, 1)).to(tl.float32)
            b_gn0 = tl.load(g + i_tc0 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gm0 = tl.clamp(b_g0 - b_gn0[None, :], -126.0, 126.0)
            b_gq0 = tl.where(m_tc0[:, None], exp2(b_gm0), 0.0)
            b_gk0 = tl.where(m_tc0[:, None], exp2(-b_gm0), 0.0)
            b_kgt_d0 = tl.trans(b_k0 * b_gk0)
            b_Aqk_d0 += tl.dot(b_q0 * b_gq0, b_kgt_d0)
            b_Akk_d0 += tl.dot(b_k0 * b_gq0, b_kgt_d0)

        if NSC >= 2:
          if i_tc1 < T:
            p_q1 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_g1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0)).to(tl.bfloat16)
            # [BC, BC]
            b_qg1 = (b_q1 * b_gqn).to(tl.bfloat16)
            b_kg1 = (b_k1 * b_gqn).to(tl.bfloat16)
            b_Aqk10 += tl.dot(b_qg1, b_kgt)
            b_Akk10 += tl.dot(b_kg1, b_kgt)

            if FUSE_DIAGONAL:
                b_gm1_d = tl.clamp(b_gn1[None, :] - b_g1, -126.0, 126.0)
                b_gk1_d = tl.where(m_tc1[:, None], exp2(b_gm1_d), 0.0)
                b_kgt_d1 = tl.trans(b_k1 * b_gk1_d)
                b_Aqk_d1 += tl.dot(b_q1 * b_gqn, b_kgt_d1)
                b_Akk_d1 += tl.dot(b_k1 * b_gqn, b_kgt_d1)

            if NSC >= 3:
              if i_tc2 < T:
                p_q2 = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                # [BC, BK]
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = (b_q2 * b_gqn2).to(tl.bfloat16)
                b_kg2 = (b_k2 * b_gqn2).to(tl.bfloat16)
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0)).to(tl.bfloat16)
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1)).to(tl.bfloat16)
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if FUSE_DIAGONAL:
                    b_gm2_d = tl.clamp(b_gn2[None, :] - b_g2, -126.0, 126.0)
                    b_gk2_d = tl.where(m_tc2[:, None], exp2(b_gm2_d), 0.0)
                    b_kgt_d2 = tl.trans(b_k2 * b_gk2_d)
                    b_Aqk_d2 += tl.dot(b_q2 * b_gqn2, b_kgt_d2)
                    b_Akk_d2 += tl.dot(b_k2 * b_gqn2, b_kgt_d2)

                if NSC >= 4:
                  if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(
                        q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = (b_q3 * b_gqn3).to(tl.bfloat16)
                    b_kg3 = (b_k3 * b_gqn3).to(tl.bfloat16)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

                    if FUSE_DIAGONAL:
                        b_gm3_d = tl.clamp(b_gn3[None, :] - b_g3, -126.0, 126.0)
                        b_gk3_d = tl.where(m_tc3[:, None], exp2(b_gm3_d), 0.0)
                        b_kgt_d3 = tl.trans(b_k3 * b_gk3_d)
                        b_Aqk_d3 += tl.dot(b_q3 * b_gqn3, b_kgt_d3)
                        b_Akk_d3 += tl.dot(b_k3 * b_gqn3, b_kgt_d3)

    ################################################################################
    # prepare Akk (beta row scaling); Aqk is stored later as one full [BT, BT] tile
    ################################################################################
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_Akk10 = b_Akk10 * b_b1[:, None]
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_Akk20 = b_Akk20 * b_b2[:, None]
    b_Akk21 = b_Akk21 * b_b2[:, None]
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    b_Akk30 = b_Akk30 * b_b3[:, None]
    b_Akk31 = b_Akk31 * b_b3[:, None]
    b_Akk32 = b_Akk32 * b_b3[:, None]

    if FUSE_DIAGONAL:
        m_Aqk_diag = o_i[:, None] >= o_i[None, :]
        m_Akk_diag = o_i[:, None] > o_i[None, :]

        b_Aqk_d0 = tl.where(m_Aqk_diag, b_Aqk_d0, 0.0)
        b_Akk_d0 = tl.where(m_Akk_diag, b_Akk_d0, 0.0)
        b_Aqk_d1 = tl.where(m_Aqk_diag, b_Aqk_d1, 0.0)
        b_Akk_d1 = tl.where(m_Akk_diag, b_Akk_d1, 0.0)
        b_Aqk_d2 = tl.where(m_Aqk_diag, b_Aqk_d2, 0.0)
        b_Akk_d2 = tl.where(m_Akk_diag, b_Akk_d2, 0.0)
        b_Aqk_d3 = tl.where(m_Aqk_diag, b_Aqk_d3, 0.0)
        b_Akk_d3 = tl.where(m_Akk_diag, b_Akk_d3, 0.0)

        p_bd0 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,)
        )
        p_bd1 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        p_bd2 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        p_bd3 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_bd0 = tl.load(p_bd0, boundary_check=(0,)).to(tl.float32)
        b_bd1 = tl.load(p_bd1, boundary_check=(0,)).to(tl.float32)
        b_bd2 = tl.load(p_bd2, boundary_check=(0,)).to(tl.float32)
        b_bd3 = tl.load(p_bd3, boundary_check=(0,)).to(tl.float32)
        b_Akk_d0 = b_Akk_d0 * b_bd0[:, None]
        b_Akk_d1 = b_Akk_d1 * b_bd1[:, None]
        b_Akk_d2 = b_Akk_d2 * b_bd2[:, None]
        b_Akk_d3 = b_Akk_d3 * b_bd3[:, None]

        p_Akkd00 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akkd11 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akkd22 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akkd33 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        tl.store(p_Akkd00, b_Akk_d0.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd11, b_Akk_d1.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd22, b_Akk_d2.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd33, b_Akk_d3.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        # Forward substitution reloads these global tiles across warps below.
        tl.debug_barrier()

        b_Ai00 = b_Akk_d0
        b_Ai11 = b_Akk_d1
        b_Ai22 = b_Akk_d2
        b_Ai33 = b_Akk_d3
    else:
        p_Akk00 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akk11 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akk22 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akk33 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
        b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    # ---- Aqk store.
    # Non-fused (large-T) mode: one full-row [BT, BT] tile store; 16-wide bf16 block stores are
    # partial-sector writes and very slow on C550. The diagonal/upper blocks written by the
    # previous kernel are preserved bitwise via the read-modify-write select below.
    # Fused-diagonal (small-T) mode: the [BT, BT] concat raised register pressure on an already
    # 105-spill compile and made small T *slower*; keep the production per-block stores instead.
    z_blk = tl.zeros([BC, BC], dtype=tl.float32)
    if FUSE_DIAGONAL:
        # one full-row [BT, BT] tile store (128B rows). Upper blocks are exact zeros,
        # matching the caller's zero prealloc; blocks beyond NSC only cover rows >= T
        # (masked by boundary_check), so zero substitution is value-identical.
        if NSC >= 4:
            b_Aqk_full = _tile4x4(b_Aqk_d0, b_Aqk10, b_Aqk_d1, b_Aqk20, b_Aqk21, b_Aqk_d2,
                                  b_Aqk30, b_Aqk31, b_Aqk32, b_Aqk_d3, z_blk)
        elif NSC == 3:
            b_Aqk_full = _tile4x4(b_Aqk_d0, b_Aqk10, b_Aqk_d1, b_Aqk20, b_Aqk21, b_Aqk_d2,
                                  z_blk, z_blk, z_blk, z_blk, z_blk)
        elif NSC == 2:
            b_Aqk_full = _tile4x4(b_Aqk_d0, b_Aqk10, b_Aqk_d1, z_blk, z_blk, z_blk,
                                  z_blk, z_blk, z_blk, z_blk, z_blk)
        else:
            b_Aqk_full = _tile4x4(b_Aqk_d0, z_blk, z_blk, z_blk, z_blk, z_blk,
                                  z_blk, z_blk, z_blk, z_blk, z_blk)
        p_Aqk_f = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc0, 0), (BT, BT), (1, 0))
        b_Aqk_full = (b_Aqk_full * scale).to(Aqk.dtype.element_ty)
        # RMW select: keep the (block-wise) upper-triangle pre-state bitwise intact, so the
        # store is value-identical to the production per-block stores under ANY pre-state
        # (production preallocs zeros, but preserve arbitrary values like the ori kernel does).
        o_t_f = tl.arange(0, BT)
        b_Aqk_old = tl.load(p_Aqk_f, boundary_check=(0, 1))
        b_Aqk_full = tl.where((o_t_f[:, None] // BC) >= (o_t_f[None, :] // BC),
                              b_Aqk_full, b_Aqk_old)
        tl.store(p_Aqk_f, b_Aqk_full, boundary_check=(0, 1))
    else:
        b_Aqk_full = _tile4x4(z_blk, b_Aqk10, z_blk, b_Aqk20, b_Aqk21, z_blk,
                              b_Aqk30, b_Aqk31, b_Aqk32, z_blk, z_blk)
        b_Aqk_full = (b_Aqk_full * scale).to(Aqk.dtype.element_ty)
        p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc0, 0), (BT, BT), (1, 0))
        o_t = tl.arange(0, BT)
        b_Aqk_old = tl.load(p_Aqk, boundary_check=(0, 1))
        b_Aqk_full = tl.where((o_t[:, None] // BC) > (o_t[None, :] // BC), b_Aqk_full, b_Aqk_old)
        tl.store(p_Aqk, b_Aqk_full, boundary_check=(0, 1))

    ################################################################################
    # forward substitution on diagonals
    # Diagonal blocks are RAW (need substitution) when:
    #   - FUSE_DIAGONAL=True: blocks were computed fresh above as gated k.k.
    #   - FUSE_DIAGONAL=False with USE_SAFE_GATE=False: token_parallel wrote raw.
    # They are pre-inverted only by the safe_gate diagonal kernel
    # (USE_SAFE_GATE=True, FUSE_DIAGONAL=False).
    ################################################################################

    if FUSE_DIAGONAL or not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        if NSC >= 2:
          for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        if NSC >= 3:
          for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        if NSC >= 4:
          for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    if NSC >= 2:
        b_Ai10 = -tl.dot(
            tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai00,
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
    if NSC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai11,
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai20 = -tl.dot(
            b_Ai22,
            tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
    if NSC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai22,
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai31 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai30 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )

    ################################################################################
    # Output: store Akk_inv OR compute w, u, kg from registers
    ################################################################################

    if FUSE_RECOMPUTE and SPLIT_RECOMPUTE:
        # Store the merged inverse (bf16, exactly the cast the in-kernel recompute used)
        # into the internal workspace; the per-sub-chunk follow-up kernel reads it back.
        wks += (bos * H + i_h) * BT
        wty = wks.dtype.element_ty
        z_h = tl.zeros([BC, BC], dtype=wty)
        if NSC >= 2:
            tl.store(tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BT), (1, 0)),
                     _hcat(_hcat(b_Ai00.to(wty), z_h), _hcat(z_h, z_h)),
                     boundary_check=(0, 1))
            tl.store(tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BT), (1, 0)),
                     _hcat(_hcat(b_Ai10.to(wty), b_Ai11.to(wty)), _hcat(z_h, z_h)),
                     boundary_check=(0, 1))
        else:
            tl.store(tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BT), (1, 0)),
                     _hcat(_hcat(b_Ai00.to(wty), z_h), _hcat(z_h, z_h)), boundary_check=(0, 1))
        if NSC >= 3:
            tl.store(tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BT), (1, 0)),
                     _hcat(_hcat(b_Ai20.to(wty), b_Ai21.to(wty)), _hcat(b_Ai22.to(wty), z_h)),
                     boundary_check=(0, 1))
        if NSC >= 4:
            tl.store(tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BT), (1, 0)),
                     _hcat(_hcat(b_Ai30.to(wty), b_Ai31.to(wty)), _hcat(b_Ai32.to(wty), b_Ai33.to(wty))),
                     boundary_check=(0, 1))
    elif FUSE_RECOMPUTE:
        if FUSE_RECOMPUTE:
            # Cast A-inverse sub-blocks to input dtype for dot products
            b_Ai00_h = b_Ai00.to(k.dtype.element_ty)
            if NSC >= 2:
                b_Ai10_h = b_Ai10.to(k.dtype.element_ty)
                b_Ai11_h = b_Ai11.to(k.dtype.element_ty)
            if NSC >= 3:
                b_Ai20_h = b_Ai20.to(k.dtype.element_ty)
                b_Ai21_h = b_Ai21.to(k.dtype.element_ty)
                b_Ai22_h = b_Ai22.to(k.dtype.element_ty)
            if NSC >= 4:
                b_Ai30_h = b_Ai30.to(k.dtype.element_ty)
                b_Ai31_h = b_Ai31.to(k.dtype.element_ty)
                b_Ai32_h = b_Ai32.to(k.dtype.element_ty)
                b_Ai33_h = b_Ai33.to(k.dtype.element_ty)

            # Load beta for all 4 sub-chunks
            p_b0 = tl.make_block_ptr(
                beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,)
            )
            b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
            p_b1r = tl.make_block_ptr(
                beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
            )
            b_b1r = tl.load(p_b1r, boundary_check=(0,)).to(tl.float32)
            p_b2r = tl.make_block_ptr(
                beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
            )
            b_b2r = tl.load(p_b2r, boundary_check=(0,)).to(tl.float32)
            p_b3r = tl.make_block_ptr(
                beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
            )
            b_b3r = tl.load(p_b3r, boundary_check=(0,)).to(tl.float32)

            # ---- u = A_inv @ (v * beta) ----
            v_base = v_in + (bos * H + i_h) * V
            u_base = u_out + (bos * H + i_h) * V
            for i_v in range(tl.cdiv(V, BV)):
                p_v0 = tl.make_block_ptr(
                    v_base, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0)
                )
                b_v0 = tl.load(p_v0, boundary_check=(0, 1))
                b_vb0 = (b_v0 * b_b0[:, None]).to(b_v0.dtype)
                b_u0 = tl.dot(b_Ai00_h, b_vb0)
                p_u0 = tl.make_block_ptr(
                    u_base, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0)
                )
                tl.store(p_u0, b_u0.to(p_u0.dtype.element_ty), boundary_check=(0, 1))

                # Sub-chunks beyond T only produce fully-masked stores; skip their
                # loads/dots/stores entirely (identical output, much less tail work).
                if NSC >= 2:
                  if i_tc1 < T:
                    p_v1 = tl.make_block_ptr(
                        v_base, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0)
                    )
                    b_v1 = tl.load(p_v1, boundary_check=(0, 1))
                    b_vb1 = (b_v1 * b_b1r[:, None]).to(b_v1.dtype)
                    b_u1 = tl.dot(b_Ai10_h, b_vb0) + tl.dot(b_Ai11_h, b_vb1)
                    p_u1 = tl.make_block_ptr(
                        u_base, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0)
                    )
                    tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))

                    if NSC >= 3:
                      if i_tc2 < T:
                        p_v2 = tl.make_block_ptr(
                            v_base, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0)
                        )
                        b_v2 = tl.load(p_v2, boundary_check=(0, 1))
                        b_vb2 = (b_v2 * b_b2r[:, None]).to(b_v2.dtype)
                        b_u2 = (
                            tl.dot(b_Ai20_h, b_vb0)
                            + tl.dot(b_Ai21_h, b_vb1)
                            + tl.dot(b_Ai22_h, b_vb2)
                        )
                        p_u2 = tl.make_block_ptr(
                            u_base, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0)
                        )
                        tl.store(
                            p_u2, b_u2.to(p_u2.dtype.element_ty), boundary_check=(0, 1)
                        )

                        if NSC >= 4:
                          if i_tc3 < T:
                            p_v3 = tl.make_block_ptr(
                                v_base, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0)
                            )
                            b_v3 = tl.load(p_v3, boundary_check=(0, 1))
                            b_vb3 = (b_v3 * b_b3r[:, None]).to(b_v3.dtype)
                            b_u3 = (
                                tl.dot(b_Ai30_h, b_vb0)
                                + tl.dot(b_Ai31_h, b_vb1)
                                + tl.dot(b_Ai32_h, b_vb2)
                                + tl.dot(b_Ai33_h, b_vb3)
                            )
                            p_u3 = tl.make_block_ptr(
                                u_base, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0)
                            )
                            tl.store(
                                p_u3, b_u3.to(p_u3.dtype.element_ty), boundary_check=(0, 1)
                            )

            # ---- w = A_inv @ (k * beta * exp2(gk)), kg = k * exp2(gn - gk) ----
            w_base = w_out + (bos * H + i_h) * K
            kg_base = kg_out + (bos * H + i_h) * K
            last_idx = min(i_t * BT + BT, T) - 1

            for i_k in range(tl.cdiv(K, BK)):
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                b_gn = tl.load(g + last_idx * H * K + o_k, mask=m_k, other=0.0).to(
                    tl.float32
                )

                p_k0 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
                )
                p_k1 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k3 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                )

                p_gk0 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
                )
                p_gk1 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
                )
                p_gk2 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_gk3 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                )

                b_k0r = tl.load(p_k0, boundary_check=(0, 1))
                b_gk0r = tl.load(p_gk0, boundary_check=(0, 1)).to(tl.float32)
                b_kb0 = (b_k0r * b_b0[:, None] * exp2(b_gk0r)).to(b_k0r.dtype)

                b_w0 = tl.dot(b_Ai00_h, b_kb0)
                p_w0 = tl.make_block_ptr(
                    w_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
                )
                tl.store(p_w0, b_w0.to(p_w0.dtype.element_ty), boundary_check=(0, 1))
                b_kg0 = b_k0r * exp2(b_gn[None, :] - b_gk0r)
                p_kg0 = tl.make_block_ptr(
                    kg_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
                )
                tl.store(p_kg0, b_kg0.to(p_kg0.dtype.element_ty), boundary_check=(0, 1))

                # Sub-chunks beyond T only produce fully-masked stores; skip their
                # loads/dots/stores entirely (identical output, much less tail work).
                if NSC >= 2:
                  if i_tc1 < T:
                    b_k1r = tl.load(p_k1, boundary_check=(0, 1))
                    b_gk1r = tl.load(p_gk1, boundary_check=(0, 1)).to(tl.float32)
                    b_kb1 = (b_k1r * b_b1r[:, None] * exp2(b_gk1r)).to(b_k1r.dtype)
                    b_w1 = tl.dot(b_Ai10_h, b_kb0) + tl.dot(b_Ai11_h, b_kb1)
                    p_w1 = tl.make_block_ptr(
                        w_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
                    )
                    tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))
                    b_kg1 = b_k1r * exp2(b_gn[None, :] - b_gk1r)
                    p_kg1 = tl.make_block_ptr(
                        kg_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
                    )
                    tl.store(
                        p_kg1, b_kg1.to(p_kg1.dtype.element_ty), boundary_check=(0, 1)
                    )

                    if NSC >= 3:
                      if i_tc2 < T:
                        b_k2r = tl.load(p_k2, boundary_check=(0, 1))
                        b_gk2r = tl.load(p_gk2, boundary_check=(0, 1)).to(tl.float32)
                        b_kb2 = (b_k2r * b_b2r[:, None] * exp2(b_gk2r)).to(b_k2r.dtype)
                        b_w2 = (
                            tl.dot(b_Ai20_h, b_kb0)
                            + tl.dot(b_Ai21_h, b_kb1)
                            + tl.dot(b_Ai22_h, b_kb2)
                        )
                        p_w2 = tl.make_block_ptr(
                            w_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                        )
                        tl.store(
                            p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1)
                        )
                        b_kg2 = b_k2r * exp2(b_gn[None, :] - b_gk2r)
                        p_kg2 = tl.make_block_ptr(
                            kg_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                        )
                        tl.store(
                            p_kg2, b_kg2.to(p_kg2.dtype.element_ty), boundary_check=(0, 1)
                        )

                        if NSC >= 4:
                          if i_tc3 < T:
                            b_k3r = tl.load(p_k3, boundary_check=(0, 1))
                            b_gk3r = tl.load(p_gk3, boundary_check=(0, 1)).to(tl.float32)
                            b_kb3 = (
                                b_k3r * b_b3r[:, None] * exp2(b_gk3r)
                            ).to(b_k3r.dtype)
                            b_w3 = (
                                tl.dot(b_Ai30_h, b_kb0)
                                + tl.dot(b_Ai31_h, b_kb1)
                                + tl.dot(b_Ai32_h, b_kb2)
                                + tl.dot(b_Ai33_h, b_kb3)
                            )
                            p_w3 = tl.make_block_ptr(
                                w_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                            )
                            tl.store(
                                p_w3, b_w3.to(p_w3.dtype.element_ty), boundary_check=(0, 1)
                            )
                            b_kg3 = b_k3r * exp2(b_gn[None, :] - b_gk3r)
                            p_kg3 = tl.make_block_ptr(
                                kg_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                            )
                            tl.store(
                                p_kg3, b_kg3.to(p_kg3.dtype.element_ty), boundary_check=(0, 1)
                            )
    else:
        # one full-row [BT, BT] tile store; upper blocks are written as exact zeros
        # (Akk is zero-initialized by the caller, so this is value-identical).
        # Blocks beyond the chunk's active sub-chunks only cover rows >= T (masked
        # stores), so substituting exact zeros there is value-identical.
        if NSC >= 4:
            b_Akk_full = _tile4x4(b_Ai00, b_Ai10, b_Ai11, b_Ai20, b_Ai21, b_Ai22,
                                  b_Ai30, b_Ai31, b_Ai32, b_Ai33, z_blk)
        elif NSC == 3:
            b_Akk_full = _tile4x4(b_Ai00, b_Ai10, b_Ai11, b_Ai20, b_Ai21, b_Ai22,
                                  z_blk, z_blk, z_blk, z_blk, z_blk)
        elif NSC == 2:
            b_Akk_full = _tile4x4(b_Ai00, b_Ai10, b_Ai11, z_blk, z_blk, z_blk,
                                  z_blk, z_blk, z_blk, z_blk, z_blk)
        else:
            b_Akk_full = _tile4x4(b_Ai00, z_blk, z_blk, z_blk, z_blk, z_blk,
                                  z_blk, z_blk, z_blk, z_blk, z_blk)
        p_Akk = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BT, BT), (1, 0))
        tl.store(p_Akk, b_Akk_full.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_large(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    v_in,
    w_out,
    u_out,
    kg_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    """Non-fused path (FUSE_RECOMPUTE=False, FUSE_DIAGONAL=False): off-diagonal inter products
    as [BC, BT] row-block dots + solve + Akk/Aqk full-tile stores.

    Same interface/semantics as the production kernel restricted to FUSE_*=False: writes the
    strictly-lower Aqk blocks (diagonal/upper blocks written by the previous kernel are
    preserved bitwise by the RMW select) and the full [BT, BT] Akk_inv tile.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_tc0 + BC
    i_tc2 = i_tc0 + 2 * BC
    i_tc3 = i_tc0 + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    o_t = tl.arange(0, BT)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T
    # reference rows for the per-row-block gate split, clamped to a valid row for the tail
    # chunk: g is a non-increasing cumsum, so the clamped exponents stay <= 1 (no overflow)
    # and fully-masked rows are zeroed below anyway.
    r1 = tl.minimum(i_tc1, T - 1)
    r2 = tl.minimum(i_tc2, T - 1)
    r3 = tl.minimum(i_tc3, T - 1)

    ################################################################################
    # off-diagonal blocks as [BC, BT] row-block dots:
    #   A_X[c] = (q_X * exp2(g_X - gn_X)) @ (k_all * exp2(gn_X - g_all))^T
    # columns at/after the row block are zeroed by the causal row mask, so each big dot
    # replaces up to 3 of the production [16,16] dots (6 MMAs instead of 20 per K-tile).
    ################################################################################
    b_A1 = tl.zeros([BC, BT], dtype=tl.float32)
    b_B1 = tl.zeros([BC, BT], dtype=tl.float32)
    b_A2 = tl.zeros([BC, BT], dtype=tl.float32)
    b_B2 = tl.zeros([BC, BT], dtype=tl.float32)
    b_A3 = tl.zeros([BC, BT], dtype=tl.float32)
    b_B3 = tl.zeros([BC, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_kall = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BT, BK), (1, 0)
        )
        p_gall = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BT, BK), (1, 0)
        )
        b_kall = tl.load(p_kall, boundary_check=(0, 1)).to(tl.float32)
        b_gall = tl.load(p_gall, boundary_check=(0, 1)).to(tl.float32)
        b_gn1 = tl.load(g + r1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
        b_gn2 = tl.load(g + r2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
        b_gn3 = tl.load(g + r3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)

        # row block 1
        b_r = tl.where((o_t < BC)[:, None], b_kall * exp2(b_gn1[None, :] - b_gall), 0.0)
        b_rt = tl.trans(b_r).to(tl.bfloat16)
        p_q1 = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
        )
        p_k1 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
        )
        p_g1 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
        )
        b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
        b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
        b_e1 = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
        b_A1 += tl.dot((b_q1 * b_e1).to(tl.bfloat16), b_rt)
        b_B1 += tl.dot((b_k1 * b_e1).to(tl.bfloat16), b_rt)

        # row block 2
        b_r = tl.where((o_t < 2 * BC)[:, None], b_kall * exp2(b_gn2[None, :] - b_gall), 0.0)
        b_rt = tl.trans(b_r).to(tl.bfloat16)
        p_q2 = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
        )
        p_k2 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
        )
        p_g2 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
        )
        b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
        b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
        b_e2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
        b_A2 += tl.dot((b_q2 * b_e2).to(tl.bfloat16), b_rt)
        b_B2 += tl.dot((b_k2 * b_e2).to(tl.bfloat16), b_rt)

        # row block 3
        b_r = tl.where((o_t < 3 * BC)[:, None], b_kall * exp2(b_gn3[None, :] - b_gall), 0.0)
        b_rt = tl.trans(b_r).to(tl.bfloat16)
        p_q3 = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
        )
        p_k3 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
        )
        p_g3 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
        )
        b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
        b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
        b_e3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
        b_A3 += tl.dot((b_q3 * b_e3).to(tl.bfloat16), b_rt)
        b_B3 += tl.dot((b_k3 * b_e3).to(tl.bfloat16), b_rt)

    # extract the 6 off-diagonal Akk blocks (cols beyond the causal part are exact zeros)
    b_h0, b_h1 = _hsplit(b_B1)
    b_Akk10, _unused10 = _hsplit(b_h0)
    b_h0, b_h1 = _hsplit(b_B2)
    b_Akk20, b_Akk21 = _hsplit(b_h0)
    b_h0, b_h1 = _hsplit(b_B3)
    b_Akk30, b_Akk31 = _hsplit(b_h0)
    b_Akk32, _unused32 = _hsplit(b_h1)

    ################################################################################
    # Aqk: one full-row [BT, BT] tile store (RMW keeps the diagonal/upper blocks written by
    # the previous kernel bitwise intact)
    ################################################################################
    b_Aqk_full = _vcat(
        _vcat(tl.zeros([BC, BT], dtype=tl.float32), b_A1),
        _vcat(b_A2, b_A3),
    )
    b_Aqk_full = (b_Aqk_full * scale).to(Aqk.dtype.element_ty)
    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc0, 0), (BT, BT), (1, 0))
    b_Aqk_old = tl.load(p_Aqk, boundary_check=(0, 1))
    b_Aqk_full = tl.where(
        (o_t[:, None] // BC) > (o_t[None, :] // BC), b_Aqk_full, b_Aqk_old
    )
    tl.store(p_Aqk, b_Aqk_full, boundary_check=(0, 1))

    ################################################################################
    # prepare Akk (beta row scaling)
    ################################################################################
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_Akk10 = b_Akk10 * b_b1[:, None]
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_Akk20 = b_Akk20 * b_b2[:, None]
    b_Akk21 = b_Akk21 * b_b2[:, None]
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    b_Akk30 = b_Akk30 * b_b3[:, None]
    b_Akk31 = b_Akk31 * b_b3[:, None]
    b_Akk32 = b_Akk32 * b_b3[:, None]

    # load (pre-inverted when USE_SAFE_GATE, else raw) diagonal blocks
    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals (only when they were written raw, i.e. the
    # non-safe-gate token-parallel kernel)
    ################################################################################
    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ################################################################################
    # Output: full [BT, BT] Akk_inv tile store (upper blocks are exact zeros, matching the
    # caller's zero-initialization)
    ################################################################################
    z_blk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk_full = _tile4x4(b_Ai00, b_Ai10, b_Ai11, b_Ai20, b_Ai21, b_Ai22,
                          b_Ai30, b_Ai31, b_Ai32, b_Ai33, z_blk)
    p_Akk = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BT, BT), (1, 0))
    tl.store(p_Akk, b_Akk_full.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused_recompute(
    wks,
    k,
    g,
    beta,
    v_in,
    w_out,
    u_out,
    kg_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NSC: tl.constexpr,
    NTILE: tl.constexpr,
):
    """Per-sub-chunk recompute (phase B of the fused path): one CTA per (chunk, sub-chunk,
    b*h). Reads the merged inverse blocks written by phase A from wks (bf16, the exact cast
    the monolithic kernel used internally) and produces w/u/kg with identical dot shapes.
    Static if-chains instead of a runtime j-loop (runtime-bound loops around tl.dot
    miscompile/serialize on this backend)."""
    pid = tl.program_id(0)
    i_t = pid // (4 * NTILE)
    rem = pid % (4 * NTILE)
    i_sc = rem // NTILE
    i_tl = rem % NTILE
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
    else:
        bos = i_b * T
    i_tc0 = i_t * BT
    i_tcs = i_tc0 + i_sc * BC
    i_tc1 = i_tc0 + BC
    i_tc2 = i_tc0 + 2 * BC
    i_tc3 = i_tc0 + 3 * BC
    if i_tcs >= T:
        return

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    v_in += (bos * H + i_h) * V
    w_out += (bos * H + i_h) * K
    u_out += (bos * H + i_h) * V
    kg_out += (bos * H + i_h) * K
    wks += (bos * H + i_h) * BT
    beta += bos * H + i_h

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T
    last_idx = min(i_t * BT + BT, T) - 1

    b_b0 = tl.load(beta + (i_tc0 + o_i) * H, mask=m_tc0, other=0).to(tl.float32)
    # loop-invariant beta rows / Ai blocks: loaded once instead of per tile iteration.
    # Loads are boundary-masked (rows past T read as 0) and their values are only
    # consumed under the matching i_sc runtime guards, so unconditional loads are safe.
    p_ai0 = tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tcs, 0), (BC, BC), (1, 0))
    b_ai0 = tl.load(p_ai0, boundary_check=(0, 1))
    if NSC >= 2:
        b_b1 = tl.load(beta + (i_tc1 + o_i) * H, mask=m_tc1, other=0).to(tl.float32)
        p_ai1 = tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tcs, BC), (BC, BC), (1, 0))
        b_ai1 = tl.load(p_ai1, boundary_check=(0, 1))
        if NSC >= 3:
            b_b2 = tl.load(beta + (i_tc2 + o_i) * H, mask=m_tc2, other=0).to(tl.float32)
            p_ai2 = tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tcs, 2 * BC), (BC, BC), (1, 0))
            b_ai2 = tl.load(p_ai2, boundary_check=(0, 1))
            if NSC >= 4:
                b_b3 = tl.load(beta + (i_tc3 + o_i) * H, mask=m_tc3, other=0).to(tl.float32)
                p_ai3 = tl.make_block_ptr(wks, (T, BT), (H * BT, 1), (i_tcs, 3 * BC), (BC, BC), (1, 0))
                b_ai3 = tl.load(p_ai3, boundary_check=(0, 1))

    # ---- u_s = sum_{j<=s} Ai[s,j] @ (v_j * beta_j) ----
    # tile-split: this CTA handles only V-tile i_tl of u and K-tile i_tl of w/kg
    if i_tl < tl.cdiv(V, BV):
        i_v = i_tl
        p_v0 = tl.make_block_ptr(
            v_in, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0)
        )
        b_v0 = tl.load(p_v0, boundary_check=(0, 1))
        b_vb0 = (b_v0 * b_b0[:, None]).to(b_v0.dtype)
        b_u = tl.dot(b_ai0, b_vb0)
        if NSC >= 2:
          if i_sc >= 1:
            p_v1 = tl.make_block_ptr(
                v_in, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0)
            )
            b_v1 = tl.load(p_v1, boundary_check=(0, 1))
            b_vb1 = (b_v1 * b_b1[:, None]).to(b_v1.dtype)
            b_u += tl.dot(b_ai1, b_vb1)
            if NSC >= 3:
              if i_sc >= 2:
                p_v2 = tl.make_block_ptr(
                    v_in, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0)
                )
                b_v2 = tl.load(p_v2, boundary_check=(0, 1))
                b_vb2 = (b_v2 * b_b2[:, None]).to(b_v2.dtype)
                b_u += tl.dot(b_ai2, b_vb2)
                if NSC >= 4:
                  if i_sc >= 3:
                    p_v3 = tl.make_block_ptr(
                        v_in, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0)
                    )
                    b_v3 = tl.load(p_v3, boundary_check=(0, 1))
                    b_vb3 = (b_v3 * b_b3[:, None]).to(b_v3.dtype)
                    b_u += tl.dot(b_ai3, b_vb3)
        p_us = tl.make_block_ptr(
            u_out, (T, V), (H * V, 1), (i_tcs, i_v * BV), (BC, BV), (1, 0)
        )
        tl.store(p_us, b_u.to(p_us.dtype.element_ty), boundary_check=(0, 1))

    # ---- w_s = sum_{j<=s} Ai[s,j] @ (k_j * beta_j * exp2(g_j)); kg_s = k_s * exp2(gn - g_s)
    if i_tl < tl.cdiv(K, BK):
        i_k = i_tl
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(g + last_idx * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)

        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        p_g0 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)
        b_kb0 = (b_k0 * b_b0[:, None] * exp2(b_g0)).to(b_k0.dtype)
        b_w = tl.dot(b_ai0, b_kb0)
        if NSC >= 2:
          if i_sc >= 1:
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            p_g1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_kb1 = (b_k1 * b_b1[:, None] * exp2(b_g1)).to(b_k1.dtype)
            b_w += tl.dot(b_ai1, b_kb1)
            if NSC >= 3:
              if i_sc >= 2:
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                b_kb2 = (b_k2 * b_b2[:, None] * exp2(b_g2)).to(b_k2.dtype)
                b_w += tl.dot(b_ai2, b_kb2)
                if NSC >= 4:
                  if i_sc >= 3:
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    b_kb3 = (b_k3 * b_b3[:, None] * exp2(b_g3)).to(b_k3.dtype)
                    b_w += tl.dot(b_ai3, b_kb3)
        p_ws = tl.make_block_ptr(
            w_out, (T, K), (H * K, 1), (i_tcs, i_k * BK), (BC, BK), (1, 0)
        )
        tl.store(p_ws, b_w.to(p_ws.dtype.element_ty), boundary_check=(0, 1))

        p_ks = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tcs, i_k * BK), (BC, BK), (1, 0)
        )
        p_gs = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tcs, i_k * BK), (BC, BK), (1, 0)
        )
        b_ks = tl.load(p_ks, boundary_check=(0, 1))
        b_gs = tl.load(p_gs, boundary_check=(0, 1)).to(tl.float32)
        b_kg = b_ks * exp2(b_gn[None, :] - b_gs)
        p_kg = tl.make_block_ptr(
            kg_out, (T, K), (H * K, 1), (i_tcs, i_k * BK), (BC, BK), (1, 0)
        )
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    beta = beta + bos * H + i_h
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    if USE_GATHER:
        b_gn = gather(
            b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0
        )
    else:
        # calculate offset
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

    b_kgt = tl.trans(b_k * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    fuse_recompute: bool = False,
    fuse_diagonal: bool = False,
):
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    if fuse_diagonal:
        Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    else:
        Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    # Step 1: compute diagonal blocks into Akkd (fp32)
    # When fuse_diagonal=True, diagonal blocks are computed inside inter_solve
    if not fuse_diagonal:
        if safe_gate:
            grid = (NT, NC, B * H)
            BK = triton.next_power_of_2(K)
            chunk_kda_fwd_kernel_intra_sub_chunk[grid](
                q=q,
                k=k,
                g=gk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                H=H,
                K=K,
                BT=BT,
                BC=BC,
                BK=BK,
                USE_GATHER=is_gather_supported,
            )
        else:
            Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
                q=q,
                k=k,
                gk=gk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=BT,
                sub_chunk_size=BC,
            )

    # Step 2: inter_solve (+ optional fused recompute)
    grid = (NT, B * H)


    if fuse_recompute:
        w = torch.empty_like(k)
        u = torch.empty_like(v)
        kg = torch.empty_like(k)
        # Split the fused recompute into a second per-sub-chunk kernel only when the
        # chunk has more than one sub-chunk of work (a second launch's host cost
        # otherwise dominates T <= BC single-sub-chunk cases).
        split = T > BC
        if split:
            _key = (B, k.shape[1], H, BT, str(k.device))
            wks = _INTER_SOLVE_WKS.get(_key)
            if wks is None:
                wks = torch.empty(B, k.shape[1], H, BT, device=k.device, dtype=k.dtype)
                _INTER_SOLVE_WKS[_key] = wks
        else:
            wks = k
        _cfg = INTER_SOLVE_CONFIGS[True]
        _lo = dict(num_warps=_cfg["num_warps"], num_stages=_cfg["num_stages"])
        _isv = cu_seqlens is not None
        _nsc = triton.cdiv(min(T, BT), BC)
        _fl_launch(
            chunk_kda_fwd_kernel_inter_solve_fused, "kda_is_fused",
            (B, T, H, K, V, BT, BC, _isv, safe_gate, True, fuse_diagonal, split, _nsc,
             scale == 1, cu_seqlens is None, wks is k, str(q.dtype), str(k.dtype),
             str(Aqk.dtype), str(Akkd.dtype)),
            grid,
            [q, k, gk, beta, Aqk, Akkd, k, scale, v, w, u, kg, wks,
             cu_seqlens, chunk_indices, T, H, K, V, BT, BC,
             _cfg["BK"], _cfg["BV"], _isv, safe_gate, True, fuse_diagonal, split, _nsc],
            _lo,
        )
        if split:
            _ntile = -(-max(K, V) // max(_cfg["BK"], _cfg["BV"]))
            _fl_launch(
                chunk_kda_fwd_kernel_inter_solve_fused_recompute, "kda_is_phaseb",
                (B, T, H, K, V, BT, BC, _isv, _nsc, _ntile,
                 str(k.dtype), str(v.dtype), str(wks.dtype)),
                (NT * 4 * _ntile, B * H),
                [wks, k, gk, beta, v, w, u, kg,
                 cu_seqlens, chunk_indices, T, H, K, V, BT, BC,
                 _cfg["BK"], _cfg["BV"], _isv, _nsc, _ntile],
                _lo,
            )
        return w, u, None, kg, Aqk, None

    # Non-fused path: inter_solve stores Akk, then separate recompute. Uses the
    # dedicated row-block-dot kernel (identical interface and semantics); the
    # fuse_diagonal=True combo (not produced by the current caller, which always pairs
    # it with fuse_recompute) falls back to the general fused kernel.
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    if fuse_diagonal:
        chunk_kda_fwd_kernel_inter_solve_fused[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akkd=Akkd,
            Akk=Akk,
            scale=scale,
            v_in=k,
            w_out=k,
            u_out=k,
            kg_out=k,
            wks=k,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            V=0,
            BT=BT,
            BC=BC,
            USE_SAFE_GATE=safe_gate,
            IS_VARLEN=cu_seqlens is not None,
            FUSE_RECOMPUTE=False,
            FUSE_DIAGONAL=True,
            SPLIT_RECOMPUTE=False,
            **INTER_SOLVE_CONFIGS[False],
        )
    else:
        _cfg = INTER_SOLVE_CONFIGS[False]
        _isv = cu_seqlens is not None
        _fl_launch(
            chunk_kda_fwd_kernel_inter_solve_large, "kda_is_large",
            (B, T, H, K, 0, BT, BC, _isv, safe_gate, scale == 1, cu_seqlens is None,
             str(q.dtype), str(k.dtype), str(Aqk.dtype)),
            grid,
            [q, k, gk, beta, Aqk, Akkd, Akk, scale,
             k, k, k, k,  # v_in/w_out/u_out/kg_out unused when FUSE_RECOMPUTE=False
             cu_seqlens, chunk_indices, T, H, K, 0, BT, BC,
             _cfg["BK"], _cfg["BV"], _isv, safe_gate],
            dict(num_warps=_cfg["num_warps"], num_stages=_cfg["num_stages"]),
        )

    from sglang.kernels.ops.attention.fla.kda import (
        recompute_w_u_fwd as kda_recompute_w_u_fwd,
    )

    w, u, kg = kda_recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, None, kg, Aqk, Akk


# ============================================================================
# [3/6] fused_sigmoid_gating_delta_rule_update -- source: fla/fused_sigmoid_gating_recurrent.py
# ============================================================================

# launch config of the fast path (tuned on C550 for the GLM-5.3-Flash KDA verify shapes)
_CFG = {
    "BV": 8,
    "num_warps": 0,  # 0 = auto by (CTA count, steps): >=2048 CTA and T>=5 -> 2 else 1
    "num_stages": 1,
    # drop masks on always-in-bounds accesses when BK==K and V%BV==0 (C500 disasm:
    # masked stores cost predicate/select work in the hot loop; T12 -0.8us)
    "unmasked_even": 1,
    # tl.range(loop_unroll_factor) on the recurrence loop: 2 beats plain+ns2
    # (T12 60.5 vs 61.3us, T6 34.7 vs 35.0us, 3x repeats); >=3 regresses.
    "unroll": 2,
}


@triton.jit(do_not_specialize=["T"])
def _fused_sigmoid_gating_delta_rule_update_kernel_generic(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    # ReplaySSM fused ring-write. Pointers stay None and CACHE_RING False for
    # decode / flag-off -> byte-identical. The gate ring layout follows IS_KDA
    # (see the store below).
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
    stride_rawv_slot: tl.constexpr = 0,
    stride_rawk_slot: tl.constexpr = 0,
    stride_g_slot: tl.constexpr = 0,
    stride_beta_slot: tl.constexpr = 0,
    MAX_CACHE_LEN: tl.constexpr = 0,
    CACHE_RING: tl.constexpr = False,
    SPLIT_N_HV_GRID: tl.constexpr = False,
    USE_GDC: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    (Unchanged production kernel, used as the fallback for paths the fast kernel
    does not cover: EAGLE tree mask, ReplaySSM ring-write, non-split grid.)
    """
    if SPLIT_N_HV_GRID:
        i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        # The GPU wrapper asserts NK == 1. Keep N and HV on independent grid
        # axes so large GLM5 decode batches do not exceed CUDA's grid limit.
        i_k = 0
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
    # PDL: overlap this kernel's prologue with the producer (the KDA/GDN
    # conv1d_update). All global loads below happen after the wait, so
    # numerics are unchanged. The immediate trigger releases the LAUNCH of
    # the next PDL kernel so its prologue overlaps this whole body;
    # consumers' own gdc_wait still fences on full completion.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # Slot stride comes from the caller (h0_source.stride(0)): the state pool
        # may be an envelope-strided view (page-major / unified memory), where the
        # per-slot pitch spans ALL layers' state, not HV*K*V. int64: envelope
        # pitches overflow an int32 index product.
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * stride_h0_source
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled. int64: the buffer is
    # contiguous but `cache_idx * cache_steps * HV * K * V` can exceed int32 for
    # large slot counts.
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        if USE_LOWER_BOUND:
            # KDA safe gate: lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
            b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)
        else:
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            beta_x = softplus_beta * x
            # Apply softplus with numerical stability
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # fused ring-write: stash this step's raw inputs + in-kernel gate/beta
        # into the per-slot ring for the commit fold to replay. Must sit here --
        # b_k is still pre-l2norm, b_v still pre-delta, b_g/b_beta are formed,
        # so the fold's replay is bit-identical to the update below. rawk uses
        # the k-head i_h (shared across a GQA group); rawv/g/beta use the v-head
        # i_hv. step_idx < MAX_CACHE_LEN: absorb-inflated rows can exceed the
        # ring; the overflow steps are past the committable prefix, so drop them
        # (writing them would smash the next slot's ring).
        if CACHE_RING:
            ring_slot = tl.load(h0_indices + i_n).to(tl.int64)
            if ring_slot >= 0 and step_idx < MAX_CACHE_LEN:
                tl.store(
                    replayssm_rawv
                    + ring_slot * stride_rawv_slot
                    + i_hv * MAX_CACHE_LEN * V
                    + step_idx * V
                    + o_v,
                    b_v.to(replayssm_rawv.dtype.element_ty),
                    mask=mask_v,
                )
                if i_v == 0:
                    tl.store(
                        replayssm_rawk
                        + ring_slot * stride_rawk_slot
                        + i_h * MAX_CACHE_LEN * K
                        + step_idx * K
                        + o_k,
                        b_k.to(replayssm_rawk.dtype.element_ty),
                        mask=mask_k,
                    )
                    # b_g follows IS_KDA: KDA loads a/dt_bias with mask_k, so the
                    # gate is a per-K vector and the ring row is K wide; GDN's is
                    # a scalar per (head, step). The two layouts are not
                    # interchangeable -- storing one into the other's stride is a
                    # shape error, not a slow path -- and memory_pool.py sizes
                    # replayssm_g off the same is_kda test.
                    if IS_KDA:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN * K
                            + step_idx * K
                            + o_k,
                            b_g,
                            mask=mask_k,
                        )
                    else:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_g,
                        )
                    if i_k == 0:
                        tl.store(
                            replayssm_beta
                            + ring_slot * stride_beta_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_beta,
                        )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * stride_h0_source
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)



@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_precompute_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    b,
    ws,
    qn_off,
    kn_off,
    decay_off,
    beta_off,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_b,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    """Step 1 of the 2-kernel fast path: precompute the per-token, per-head
    vectors the recurrence needs (l2norm+scale of q/k, decay exp(g), beta) once
    per (t, head). The fused recurrence kernel reads these instead of
    recomputing them in every i_v-slice CTA (C550: that replication was
    ~20us of vector ALU for the decode shapes). Identical formula order as the
    production kernel, so values are bit-identical."""
    pid = tl.program_id(0)
    i_hv = pid % HV
    t = pid // HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    b_A = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    if IS_KDA:
        p_a = a + t * stride_a + i_hv * K + o_k
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0).to(tl.float32)
    else:
        p_a = a + t * stride_a + i_hv
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    x = b_a + b_dt_bias
    if USE_LOWER_BOUND:
        b_g = lower_bound * tl.sigmoid(b_A * x)
    else:
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -b_A * softplus_x
    b_decay = tl.exp(b_g)

    b_b = tl.load(b + t * stride_b + i_hv).to(tl.float32)
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    b_q = tl.load(q + t * stride_q + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(k + t * stride_k + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM:
        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
    b_q = b_q * scale

    p_qn = ws + qn_off + (t * H + i_h) * K + o_k
    p_kn = ws + kn_off + (t * H + i_h) * K + o_k
    tl.store(p_qn, b_q, mask=mask_k)
    tl.store(p_kn, b_k, mask=mask_k)
    if IS_KDA:
        p_decay = ws + decay_off + (t * HV + i_hv) * K + o_k
        tl.store(p_decay, b_decay, mask=mask_k)
    else:
        p_decay = ws + decay_off + (t * HV + i_hv)
        tl.store(p_decay, b_decay)  # scalar gate
    tl.store(ws + beta_off + (t * HV + i_hv), b_beta)


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel_opt(
    ws,
    qn_off,
    kn_off,
    decay_off,
    beta_off,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    T,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr,
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
    USE_GDC: tl.constexpr,
    UNMASKED_EVEN: tl.constexpr = False,
    UNROLL: tl.constexpr = 0,
):
    """Fast path (no EAGLE tree mask, no ReplaySSM ring, SPLIT_N_HV_GRID) of
    fused_sigmoid_gating_delta_rule_update_kernel; identical math per token step.

    C550 profiling (ablation): the production kernel spends ~75% of its time on
    the serial per-step chain load(q/k/v/b/a) -> gate/l2norm -> state update ->
    16KB state store; the raw memory pattern alone (micro benchmark) runs 5-8x
    faster. Changes that decouple the chain without changing the math:
      1. the gate (A_log/dt_bias/a/b -> decay, beta) and the qk l2norm+scale are
         precomputed once per (t, head) by fused_sigmoid_gating_precompute_kernel
         (the fused kernel previously recomputed them in every i_v-slice CTA:
         16x replicated vector ALU, ~20us on the decode shapes);
      2. the next step's preprocessed vectors are prefetched at the TOP of the
         current iteration (double buffering), so the load latency overlaps the
         state update and the state store of the current step.
    The final iteration's prefetch is masked off.
    """
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k = 0
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # VEC2D experiment (R2, rejected): loading q/k/decay as [BK,1] tiles to
    # anchor their layout to the state's blocked layout (and thus kill the 4
    # per-step cross-warp layout conversions of the NW=2 build) made the
    # compiler pick a pathological lowering: T12 160us vs 61us. Negative result
    # recorded in ITERATIONS.md; conversion tax stays until the layout policy
    # improves.
    p_qn = ws + qn_off + (bos * H + i_h) * K + o_k
    p_kn = ws + kn_off + (bos * H + i_h) * K + o_k
    if IS_KDA:
        p_decay = ws + decay_off + (bos * HV + i_hv) * K + o_k
    else:
        p_decay = ws + decay_off + (bos * HV + i_hv)
    p_beta = ws + beta_off + bos * HV + i_hv
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * stride_h0_source
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            if UNMASKED_EVEN:
                b_h += tl.load(p_h0).to(tl.float32)
            else:
                b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    cache_idx = -1
    p_cache = intermediate_states_buffer
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)
        p_cache = (
            intermediate_states_buffer
            + cache_idx * cache_steps * HV * K * V
            + i_hv * K * V
            + o_v[None, :] * K
            + o_k[:, None]
        )

    # ---- prologue: load step-0 preprocessed inputs ----
    if UNMASKED_EVEN:
        b_q = tl.load(p_qn)
        b_k = tl.load(p_kn)
        b_v = tl.load(p_v).to(tl.float32)
    else:
        b_q = tl.load(p_qn, mask=mask_k, other=0)
        b_k = tl.load(p_kn, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    if IS_KDA:
        b_decay = tl.load(p_decay, mask=mask_k, other=0) if not UNMASKED_EVEN else tl.load(p_decay)
    else:
        b_decay = tl.load(p_decay)
    b_beta = tl.load(p_beta)

    step_idx = 0
    for _ in tl.range(0, T, loop_unroll_factor=UNROLL if UNROLL > 1 else 1):
        # ---- prefetch step+1 preprocessed inputs BEFORE the state update /
        # stores of this step, so the load latency hides under the compute +
        # state store. The last iteration's prefetch is masked off (results
        # never consumed).
        has_next = step_idx + 1 < T
        m_kn = mask_k & has_next
        m_vn = mask_v & has_next
        n_q = tl.load(p_qn + H * K, mask=m_kn, other=0)
        n_k = tl.load(p_kn + H * K, mask=m_kn, other=0)
        if IS_KDA:
            n_decay = tl.load(p_decay + HV * K, mask=m_kn, other=0)
        else:
            n_decay = tl.load(p_decay + HV, mask=has_next, other=0)
        n_beta = tl.load(p_beta + HV, mask=has_next, other=0)
        n_v = tl.load(p_v + stride_v, mask=m_vn, other=0).to(tl.float32)

        # Apply gating to hidden state: h *= exp(g) (decay precomputed)
        if IS_KDA:
            b_h *= b_decay[:, None]
        else:
            b_h *= b_decay

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        if UNMASKED_EVEN:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty))
        else:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                if UNMASKED_EVEN:
                    tl.store(
                        p_cache + step_idx * HV * K * V,
                        b_h.to(p_cache.dtype.element_ty),
                    )
                else:
                    tl.store(
                        p_cache + step_idx * HV * K * V,
                        b_h.to(p_cache.dtype.element_ty),
                        mask=mask_h,
                    )

        step_idx += 1

        # ---- rotate prefetch buffers into the current regs, advance pointers ----
        b_q = n_q
        b_k = n_k
        b_decay = n_decay
        b_beta = n_beta
        b_v = n_v
        p_qn += H * K
        p_kn += H * K
        if IS_KDA:
            p_decay += HV * K
        else:
            p_decay += HV
        p_beta += HV
        p_v += stride_v
        p_o += HV * V

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * stride_h0_source
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                if UNMASKED_EVEN:
                    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty))
                else:
                    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    lower_bound: Optional[float] = None,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    # fused ReplaySSM ring-write (spec verify). When cache_ring, each draft step
    # stores pre-norm k / raw v / gate / beta into these per-slot rings,
    # replacing the eager ring-write. Off by default -> decode unchanged.
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    # KDA decode also passes 4-D [B, T, H, K], where [-2] is the head stride, not the
    # token stride; take dim 1 explicitly for that layout.
    stride_a = a.stride()[1] if a.ndim == 4 else a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    split_n_hv_grid = q.device.type == "cuda"
    grid = (NV, N, HV) if split_n_hv_grid else (NK, NV, N * HV)

    # Adaptive spec changes the runtime draft count without changing the
    # allocated per-request pitch, which is preserved in stride(0).
    if intermediate_states_buffer is not None:
        cache_stride_steps = intermediate_states_buffer.stride(0) // (HV * K * V)
    elif cache_steps is not None and cache_steps > 0:
        cache_stride_steps = cache_steps
    else:
        cache_stride_steps = 0

    # ring strides (per-slot rings are contiguous [num_slots, heads, L, dim];
    # the kernel offsets within a slot with MAX_CACHE_LEN and the dim extents).
    if cache_ring:
        # stride(0) is used as the slot pitch, so a tensor still carrying the
        # layer dim would scribble outside its slot. The gate ring is the one
        # whose rank depends on the model: per-K vector for KDA, per-head scalar
        # for GDN, matching g_shape in memory_pool.py and the IS_KDA branch in
        # the store above.
        assert (
            replayssm_rawv.dim() == 4
            and replayssm_rawk.dim() == 4
            and replayssm_g.dim() == (4 if is_kda else 3)
            and replayssm_beta.dim() == 3
        ), "cache_ring expects per-layer ring views"
        max_cache_len = replayssm_rawv.shape[-2]
        stride_rawv_slot = replayssm_rawv.stride(0)
        stride_rawk_slot = replayssm_rawk.stride(0)
        stride_g_slot = replayssm_g.stride(0)
        stride_beta_slot = replayssm_beta.stride(0)
    else:
        max_cache_len = 0
        stride_rawv_slot = stride_rawk_slot = stride_g_slot = stride_beta_slot = 0

    # PDL (sm90+): chain this kernel behind its producer conv1d_update, which
    # already launches dependents. Bit-exact (scheduling only) — benefits both
    # KDA and GDN recurrent paths.
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    if retrieve_parent_token is None and not cache_ring and split_n_hv_grid:
        # fast path: every GLM-5.3-Flash KDA decode / target_verify launch
        # (no EAGLE tree mask, no ReplaySSM ring). Identical math; see the
        # kernel docstrings for the schedule.
        BVf = min(triton.next_power_of_2(V), _CFG["BV"])
        # C550 tuning (R2 N/T sweep, graph timing): 2 warps win only when there
        # is enough per-CTA serial work to hide the extra sync AND the grid is
        # large: T>=5 and >=2048 CTAs (T12 61.3 vs 64.0us; N=8/T6 199.5 vs
        # 219.8us). Decode (T=1) picks 1 warp at every N (N=128: 779.5 vs
        # 885.1us); short verifies (T<=4) tie or prefer 1 warp.
        n_warps = _CFG["num_warps"] if _CFG["num_warps"] else (
            2
            if (T >= 5 and triton.cdiv(V, BVf) * N * HV >= 2048)
            else 1
        )
        even_tile = BK == K and V % BVf == 0
        # single fp32 workspace: [qn T*H*K][kn T*H*K][decay T*HV*K|T*HV][beta T*HV]
        qn_off = 0
        kn_off = qn_off + T * H * K
        decay_off = kn_off + T * H * K
        decay_el = T * HV * (K if is_kda else 1)
        beta_off = decay_off + decay_el
        ws = torch.empty(beta_off + T * HV, dtype=torch.float32, device=q.device)
        fused_sigmoid_gating_precompute_kernel[(T * HV,)](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            b=b,
            ws=ws,
            qn_off=qn_off,
            kn_off=kn_off,
            decay_off=decay_off,
            beta_off=beta_off,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            lower_bound=lower_bound if lower_bound is not None else 0.0,
            scale=scale,
            T=T,
            stride_a=stride_a,
            stride_q=stride_q,
            stride_k=stride_k,
            stride_b=stride_b,
            H=H,
            HV=HV,
            K=K,
            BK=BK,
            IS_KDA=is_kda,
            USE_LOWER_BOUND=lower_bound is not None,
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            num_warps=1,
            num_stages=1,
        )
        fused_sigmoid_gating_delta_rule_update_kernel_opt[(triton.cdiv(V, BVf), N, HV)](
            ws=ws,
            qn_off=qn_off,
            kn_off=kn_off,
            decay_off=decay_off,
            beta_off=beta_off,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            stride_h0_source=(
                initial_state_source.stride(0) if initial_state_source is not None else 0
            ),
            cu_seqlens=cu_seqlens,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_stride_steps,
            T=T,
            stride_q=stride_q,
            stride_k=stride_k,
            stride_v=stride_v,
            stride_b=stride_b,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BVf,
            USE_INITIAL_STATE=initial_state_source is not None,
            IS_VARLEN=cu_seqlens is not None,
            IS_KDA=is_kda,
            DISABLE_STATE_UPDATE=disable_state_update,
            CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
            USE_GDC=is_arch_support_pdl(),
            UNMASKED_EVEN=bool(_CFG["unmasked_even"]) and even_tile,
            UNROLL=_CFG.get("unroll", 0),
            num_warps=n_warps,
            num_stages=_CFG["num_stages"],
        )
        return o.squeeze(0)

    _fused_sigmoid_gating_delta_rule_update_kernel_generic[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        lower_bound=lower_bound if lower_bound is not None else 0.0,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != HV*K*V; contiguous pools pass exactly HV*K*V.
        stride_h0_source=(
            initial_state_source.stride(0) if initial_state_source is not None else 0
        ),
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        USE_LOWER_BOUND=lower_bound is not None,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
        stride_rawv_slot=stride_rawv_slot,
        stride_rawk_slot=stride_rawk_slot,
        stride_g_slot=stride_g_slot,
        stride_beta_slot=stride_beta_slot,
        MAX_CACHE_LEN=max_cache_len,
        CACHE_RING=cache_ring,
        SPLIT_N_HV_GRID=split_n_hv_grid,
        num_warps=1,
        num_stages=3,
        **pdl_kwargs,
    )
    o = o.squeeze(0)
    return o



# ============================================================================
# [4/6] chunk_gla_fwd_o_gk -- source: fla/kda.py
# ============================================================================

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

# ---- chunk_gla_fwd_kernel_o (optimized) ------------------------------------
# Changes vs the previous autotuned version (see chunk_gla_fwd_kernel_o_run/
# ITERATIONS.md, iterations 1-5):
#   1. no @triton.autotune: fixed config per T bucket (the autotuner wrapper
#      costs ~15 us host time per launch; small-T launches are host-bound in
#      eager production).
#   2. large T (>=2048): BK=64, BV=128, warps=4, stages=4, grid (NV, B*H, NT)
#      (consecutive CTAs read adjacent h tiles); small T: BK=128, BV=32,
#      warps=4, stages=1, grid (NV, NT, B*H) for more CTAs when NT is tiny.
#   3. UNIFY (large T, BK == BT): the intra-chunk A@v dot runs as iteration 0
#      of the k loop so the whole accumulation uses a single tl.dot site.
#      Round-1 TTGIR showed the two dots picked different MMA warp layouts
#      ([4,1] vs [2,2]) forcing fp32 accumulator layout converts; a single
#      dot site keeps one accumulator layout (T=8192 graph 100.7 -> 98.5 us,
#      T=3072 44.7 -> 41.8; machine-code stream shrinks ~40%).
#   4. EVEN_T / FULL_VK constexpr paths drop boundary_check masks when every
#      chunk has full rows (T % BT == 0 and single packed sequence) and when
#      the h tile is always in range (V % BV == 0 and K % BK == 0).
#   5. the wrapper caches the CompiledKernel per bucket and dispatches via
#      CompiledKernel.run directly (~11 us host instead of ~35 us). Pointer
#      16B-alignment and Triton launch-hook guards fall back to the normal
#      Triton dispatch path.
# Numerics: identical element-wise math; the tl.dot tiling/order changes can
# shift bf16 results by 1-2 ulp (max_abs 1.5e-05 on captures, tolerance 1e-2).
_CFG_LARGE = (64, 128, 4, 4, 1)   # (BK, BV, num_warps, num_stages, ORDER)
_CFG_SMALL = (128, 32, 4, 1, 0)
_LARGE_T = 2048
_FAST_LAUNCH = {}
_get_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_DEV = None


def _hooks_active():
    """True when any Triton launch hook would fire (proton etc. attached)."""
    for hk in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
        if hk is None:
            continue
        calls = getattr(hk, "calls", None)
        if calls is None or calls:
            return True
    return False


@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ORDER: tl.constexpr,
    EVEN_T: tl.constexpr,
    FULL_VK: tl.constexpr,
    UNIFY: tl.constexpr,
):
    if ORDER == 0:
        i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    elif ORDER == 1:
        i_v, i_bh, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
        i_bh, i_v, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV] -- v and A are loaded first and the intra-chunk dot runs
    # while the q/g/h streams are still warming up (reorders the fp32
    # accumulation: A@v before qg@h; <=1-2 bf16 ulp vs baseline)
    if EVEN_T:
        if FULL_VK:
            b_v = tl.load(p_v)
        else:
            b_v = tl.load(p_v, boundary_check=(0, 1))
    else:
        if FULL_VK:
            b_v = tl.load(p_v, boundary_check=(0,))
        else:
            b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT] -- column dim is exactly BT, only rows can be partial
    if EVEN_T:
        b_A = tl.load(p_A)
    else:
        b_A = tl.load(p_A, boundary_check=(0,))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)

    if UNIFY:
        # single textual tl.dot: iteration 0 is the intra-chunk A@v, the
        # rest are (q*2^g)@h^T. One dot site -> one MMA warp layout -> no
        # fp32 accumulator layout convert (round-1 TTGIR: [4,1] vs [2,2])
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK) + 1):
            if i_k == 0:
                b_lhs = b_A
                b_rhs = b_v
            else:
                kk = i_k - 1
                p_q = tl.make_block_ptr(
                    q + (bos * H + i_h) * K,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, kk * BK),
                    (BT, BK),
                    (1, 0),
                )
                p_g = tl.make_block_ptr(
                    g + (bos * H + i_h) * K,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, kk * BK),
                    (BT, BK),
                    (1, 0),
                )
                p_h = tl.make_block_ptr(
                    h + (i_tg * H + i_h) * V * K,
                    (V, K),
                    (K, 1),
                    (i_v * BV, kk * BK),
                    (BV, BK),
                    (1, 0),
                )
                if EVEN_T:
                    if FULL_VK:
                        b_q = tl.load(p_q)
                        b_g = tl.load(p_g)
                    else:
                        b_q = tl.load(p_q, boundary_check=(0, 1))
                        b_g = tl.load(p_g, boundary_check=(0, 1))
                else:
                    if FULL_VK:
                        b_q = tl.load(p_q, boundary_check=(0,))
                        b_g = tl.load(p_g, boundary_check=(0,))
                    else:
                        b_q = tl.load(p_q, boundary_check=(0, 1))
                        b_g = tl.load(p_g, boundary_check=(0, 1))
                b_q = (b_q * scale).to(b_q.dtype)
                b_lhs = (b_q * exp2(b_g)).to(b_q.dtype)
                if FULL_VK:
                    b_h = tl.load(p_h)
                else:
                    b_h = tl.load(p_h, boundary_check=(0, 1))
                b_rhs = tl.trans(b_h).to(b_q.dtype)
            b_o += tl.dot(b_lhs, b_rhs, allow_tf32=False)
    else:
        b_o = tl.dot(b_A, b_v, allow_tf32=False)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_g = tl.make_block_ptr(
                g + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_h = tl.make_block_ptr(
                h + (i_tg * H + i_h) * V * K,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            # [BT, BK]
            if EVEN_T:
                if FULL_VK:
                    b_q = tl.load(p_q)
                    b_g = tl.load(p_g)
                else:
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_g = tl.load(p_g, boundary_check=(0, 1))
            else:
                if FULL_VK:
                    b_q = tl.load(p_q, boundary_check=(0,))
                    b_g = tl.load(p_g, boundary_check=(0,))
                else:
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_g = tl.load(p_g, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            # [BT, BK]
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            # [BV, BK]
            if FULL_VK:
                b_h = tl.load(p_h)
            else:
                b_h = tl.load(p_h, boundary_check=(0, 1))
            # [BT, BV]
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    if EVEN_T:
        if FULL_VK:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty))
        else:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    else:
        if FULL_VK:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
        else:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    is_varlen = cu_seqlens is not None
    NT = cdiv(T, BT) if not is_varlen else len(chunk_indices)

    BK, BV, warps, stages, order = _CFG_LARGE if T >= _LARGE_T else _CFG_SMALL
    # the unified-dot restructure needs the lhs shapes to match: [BT, BT]
    # (masked A) vs [BT, BK] (qg), i.e. BK == BT (large-T bucket only)
    unify = BK == BT
    NV = -(V // -BV)
    BH = B * H
    grid = (NV, BH, NT) if order == 1 else (NV, NT, BH)
    # every chunk has full rows iff the single packed sequence divides BT;
    # multi-sequence varlen batches conservatively keep the row masks
    if not is_varlen:
        even_t = T % BT == 0
    else:
        even_t = T % BT == 0 and cu_seqlens.numel() == 2
    full_vk = V % BV == 0 and K % BK == 0

    kern = _FAST_LAUNCH.get((T >= _LARGE_T, even_t, full_vk, is_varlen))
    if kern is not None and is_varlen and not _hooks_active():
        # direct CompiledKernel.run; the binary was compiled with the
        # 16-byte-divisibility specialization of the pointer args it first
        # saw, so guard against differently-aligned views and fall back to
        # the normal Triton dispatch if any pointer is unaligned.
        if not ((q.data_ptr() | v.data_ptr() | g.data_ptr() | h.data_ptr()
                 | o.data_ptr() | A.data_ptr() | cu_seqlens.data_ptr()
                 | chunk_indices.data_ptr()) & 15):
            global _DEV
            if _get_raw_stream is not None:
                if _DEV is None:
                    _DEV = torch.cuda.current_device()
                stream = _get_raw_stream(_DEV)
            else:
                stream = torch.cuda.current_stream().cuda_stream
            kern.run(grid[0], grid[1], grid[2],
                     stream, kern.function, kern.packed_metadata, None,
                     knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
                     q, v, g, h, o, A, cu_seqlens, chunk_indices, scale, T,
                     H, K, V, BT, BK, BV, is_varlen, order, even_t, full_vk, unify)
            return o
    kern = chunk_gla_fwd_kernel_o[grid](
        q, v, g, h, o, A, cu_seqlens, chunk_indices, scale, T,
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, IS_VARLEN=is_varlen,
        ORDER=order, EVEN_T=even_t, FULL_VK=full_vk, UNIFY=unify,
        num_warps=warps, num_stages=stages,
    )
    if is_varlen:
        _FAST_LAUNCH[(T >= _LARGE_T, even_t, full_vk, is_varlen)] = kern
    return o


# ============================================================================
# [5/6] kda_gate_chunk_cumsum -- source: fla/kda.py
# ============================================================================
@triton.jit
def softplus_fwd(x):
    """Standard softplus: log(1 + exp(x)), with linear approx for large x."""
    return tl.where(x < 20.0, log(1.0 + exp(x)), x)


@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SMASK: tl.constexpr,
    P: tl.constexpr,
    USE3D: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    # [BT, BS]
    if SMASK:
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)

    if HAS_BIAS:
        p_b = tl.make_block_ptr(
            dt_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    # scale folded into the scalar gate multiplier (cumsum(m*g) == m*cumsum(g) up to
    # fp rounding); saves one full-tile fp32 multiply after the scan.
    m = scale if HAS_SCALE else 1.0
    if not USE_LOWER_BOUND:
        # Standard gate: -exp(A_log) * softplus(g + bias)
        b_gate = -(m * exp(b_A)) * softplus_fwd(b_s)
    else:
        # Safe gate: lower_bound * sigmoid(exp(A_log) * (g + bias))
        b_gate = (m * lower_bound) * tl.sigmoid(exp(b_A) * b_s)

    # Chunk-local cumulative sum
    if USE3D:
        # 3D decomposition: view [BT, BS] as (P, Q, BS), Q = BT // P. The in-panel
        # scan (len Q) is short and the panel-prefix scan (len P) runs intra-thread
        # in the reshape layout, shrinking cross-wave communication ~8x vs a flat
        # 64-row tl.cumsum on MACA/C550 (see workspace ITERATIONS.md, iteration 5).
        Q: tl.constexpr = BT // P
        x = tl.reshape(b_gate, (P, Q, BS))
        y = tl.cumsum(x, axis=1)
        a = tl.sum(x, axis=1)
        ap = tl.cumsum(a, axis=0) - a
        b_o = tl.reshape(y + ap[:, None, :], (BT, BS))
    else:
        b_o = tl.cumsum(b_gate, axis=0)
    if SMASK:
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    else:
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


# ---- pinned-config fast launch (tuned on MetaX C550; see ops_ticket workspace
# kda_gate_chunk_cumsum_vector_kernel_run/ITERATIONS.md, iterations 1-5) ----
from triton.runtime.driver import driver as _triton_driver

_GATE_CFG_SMALL = (16, 4, 0, False)   # (BS, num_warps, P, USE3D) for NT <= 4
_GATE_CFG_LARGE = (64, 4, 16, True)    # for NT > 4: 3D scan (P=16, Q=4)
_GATE_NUM_STAGES = 3
_GATE_LAUNCH_CACHE = {}


def _gate_align(t):
    return (t.data_ptr() % 16) if t is not None else 0


def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused KDA gate activation + chunk-local cumulative sum.

    Combines two memory-bound kernels into one:
      1. Gate activation: g = -exp(A_log) * softplus(raw_g + dt_bias)
      2. Chunk-local cumsum along the time axis

    Args:
        g: Raw gate tensor of shape [B, T, H, K] (before activation).
        A_log: Per-head log-scale parameter, [H] elements (any shape, numel=H).
        chunk_size: Chunk size for cumsum (must be power of 2).
        scale: Optional scale factor applied to output.
        dt_bias: Optional per-head bias, flat [H*K] elements.
        cu_seqlens: Cumulative sequence lengths for variable-length input.
        output_dtype: Output dtype (default float32).
        chunk_indices: Pre-computed chunk indices for varlen mode.
        lower_bound: If set, use safe gate: lower_bound * sigmoid(exp(A_log) * g).

    Returns:
        Cumulative-summed gated tensor of shape [B, T, H, K].
    """
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    assert len(g.shape) == 4
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    has_bias = dt_bias is not None
    has_scale = scale is not None
    is_varlen = cu_seqlens is not None
    has_lb = lower_bound is not None
    bs, num_warps, p, use3d = _GATE_CFG_SMALL if NT <= 4 else _GATE_CFG_LARGE
    smask = (S % bs) != 0

    grid = (cdiv(S, bs), NT, B * H)

    key = (
        H, S, BT, bs, num_warps, p, use3d, has_bias, has_scale, is_varlen, has_lb, smask,
        g.dtype, g_org.dtype, A_log.dtype,
        dt_bias.dtype if has_bias else None,
        cu_seqlens.dtype if is_varlen else None,
        chunk_indices.dtype if is_varlen else None,
        _gate_align(g), _gate_align(g_org), _gate_align(A_log),
        _gate_align(dt_bias) if has_bias else 0,
        _gate_align(cu_seqlens) if is_varlen else 0,
        _gate_align(chunk_indices) if is_varlen else 0,
    )
    entry = _GATE_LAUNCH_CACHE.get(key)

    if entry is None:
        kern = kda_gate_chunk_cumsum_vector_kernel[grid](
            s=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            o=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
            T=T,
            H=H,
            S=S,
            BT=BT,
            BS=bs,
            HAS_BIAS=has_bias,
            HAS_SCALE=has_scale,
            IS_VARLEN=is_varlen,
            USE_LOWER_BOUND=has_lb,
            SMASK=smask,
            P=p,
            USE3D=use3d,
            num_warps=num_warps,
            num_stages=_GATE_NUM_STAGES,
        )
        try:
            kern._init_handles()
            _GATE_LAUNCH_CACHE[key] = (kern.run, kern.function, kern.packed_metadata)
        except Exception:
            _GATE_LAUNCH_CACHE[key] = None
        return g

    if entry is not None:
        krun, kfun, kmeta = entry
        device = _triton_driver.active.get_current_device()
        stream = _triton_driver.active.get_current_stream(device)
        krun(grid[0], grid[1], grid[2], stream, kfun, kmeta, None, None, None,
             g_org, A_log, dt_bias, g, scale, cu_seqlens, chunk_indices, lower_bound, T,
             H, S, BT, bs, has_bias, has_scale, is_varlen, has_lb, smask, p, use3d)
    return g


# ============================================================================
# [6/6] causal_conv1d_fn -- source: mamba/causal_conv1d_triton.py
# ============================================================================
PAD_SLOT_ID = -1


@triton.jit
def _conv4(acc0, a0, a1, a2, a3, w0, w1, w2, w3):
    # same accumulation order as the baseline: acc = bias; acc += bf16(x[t-3]*w0); ...; acc += bf16(x[t]*w3).
    # Each tap product is rounded to bf16 (RNE) before the fp32 accumulate -> bitwise identical to
    # the production kernel. Writing the rounding explicitly ((a*W).to(bf16).to(f32)) is measurably
    # faster on MACA than the implicit bf16 `a * w` lowering (~3% kernel time at T=8192).
    acc = acc0
    acc += (a0.to(tl.float32) * w0.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    acc += (a1.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    acc += (a2.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    acc += (a3.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    return acc


@triton.jit
def _silu(acc):
    return acc / (1 + tl.exp(-acc))

@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FAST: tl.constexpr,  # width==4 register-sliding fast path (needs BLOCK_M % 4 == 0)
    GRID_FEATS_FAST: tl.constexpr,  # rasterize feature-slices fastest (contiguous DRAM waves)
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    if GRID_FEATS_FAST:
        idx_feats = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        chunk_offset = tl.program_id(1)
        idx_seq = tl.program_id(2)
    else:
        idx_seq = tl.program_id(0)
        chunk_offset = tl.program_id(1)
        idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr
        + sequence_start_index.to(tl.int64) * stride_x_token
        + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + (
                    (sequence_start_index + idx_tokens_last).to(tl.int64)
                    * stride_x_token
                )[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if KERNEL_WIDTH == 4 and FAST:
        # ---- fast path (max_seq_len >= 128): 4-row groups, each row = one vectorized
        # [BLOCK_N] load; the 3-token sliding window (col0, col1, col2) stays in registers
        # -> x is read exactly once. Tail rows of the last chunk are masked off (loads and
        # stores), so short sequences inside a long-seq batch stay correct.
        mask_f = idx_feats < dim
        if HAS_BIAS:
            acc_preload = tl.load(bias_ptr + idx_feats, mask=mask_f, other=0.0).to(tl.float32)
        else:
            acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
        w0 = tl.load(w_base + 0 * stride_w_width, mask_f, other=0.0)
        w1 = tl.load(w_base + 1 * stride_w_width, mask_f, other=0.0)
        w2 = tl.load(w_base + 2 * stride_w_width, mask_f, other=0.0)
        w3 = tl.load(w_base + 3 * stride_w_width, mask_f, other=0.0)
        x_row = x_base + token_offset.to(tl.int64) * stride_x_token
        o_row = (
            o_ptr
            + (sequence_start_index + token_offset).to(tl.int64) * stride_o_token
            + idx_feats * stride_o_dim
        )
        # manual double-buffering: issue the NEXT 4-row group's loads before this
        # group's silu chain so memory and ALU overlap inside the unrolled loop.
        # manual double-buffer: issue next group's loads before this group's silu chain
        N_GROUPS: tl.constexpr = BLOCK_M // 4
        n0 = tl.load(x_row + 0 * stride_x_token, mask=mask_f & (0 < segment_len), other=0.0)
        n1 = tl.load(x_row + 1 * stride_x_token, mask=mask_f & (1 < segment_len), other=0.0)
        n2 = tl.load(x_row + 2 * stride_x_token, mask=mask_f & (2 < segment_len), other=0.0)
        n3 = tl.load(x_row + 3 * stride_x_token, mask=mask_f & (3 < segment_len), other=0.0)
        for g in tl.static_range(N_GROUPS):
            r = g * 4
            x0 = n0
            x1 = n1
            x2 = n2
            x3 = n3
            if g + 1 < N_GROUPS:
                r2 = (g + 1) * 4
                n0 = tl.load(x_row + (r2 + 0) * stride_x_token, mask=mask_f & (r2 + 0 < segment_len), other=0.0)
                n1 = tl.load(x_row + (r2 + 1) * stride_x_token, mask=mask_f & (r2 + 1 < segment_len), other=0.0)
                n2 = tl.load(x_row + (r2 + 2) * stride_x_token, mask=mask_f & (r2 + 2 < segment_len), other=0.0)
                n3 = tl.load(x_row + (r2 + 3) * stride_x_token, mask=mask_f & (r2 + 3 < segment_len), other=0.0)
            y0 = _conv4(acc_preload, col0, col1, col2, x0, w0, w1, w2, w3)
            y1 = _conv4(acc_preload, col1, col2, x0, x1, w0, w1, w2, w3)
            y2 = _conv4(acc_preload, col2, x0, x1, x2, w0, w1, w2, w3)
            y3 = _conv4(acc_preload, x0, x1, x2, x3, w0, w1, w2, w3)
            if SILU_ACTIVATION:
                y0 = _silu(y0)
                y1 = _silu(y1)
                y2 = _silu(y2)
                y3 = _silu(y3)
            tl.store(o_row + (r + 0) * stride_o_token, y0, mask=mask_f & (r + 0 < segment_len))
            tl.store(o_row + (r + 1) * stride_o_token, y1, mask=mask_f & (r + 1 < segment_len))
            tl.store(o_row + (r + 2) * stride_o_token, y2, mask=mask_f & (r + 2 < segment_len))
            tl.store(o_row + (r + 3) * stride_o_token, y3, mask=mask_f & (r + 3 < segment_len))
            col0 = x1
            col1 = x2
            col2 = x3
        return
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):

            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token).to(tl.int64)
            * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    **kwargs,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    seq_lens_cpu: (batch) int32
        The sequence lengths of the sequences in the batch
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    out = torch.empty_like(x)

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        # assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        padded_batch = query_start_loc.size(0) - 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert (
                conv_states is not None
            ), "ERROR: `has_initial_state` is used, which needs also `conv_states`"
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    max_seq_len = max(seq_lens_cpu)
    # Fast register-sliding path only pays off for long sequences: below the threshold we
    # compile the generic loop (identical to the previous production kernel), so small-T
    # latency (launch-bound; graph time ~3-6 us) is unchanged.
    FAST = width == 4 and max_seq_len >= 128
    if FAST:
        BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES = 32, 512, 4, 1
    else:
        BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES = 8, 256, 4, 2

    def grid(META):
        if FAST:
            # feature slices rasterized fastest -> resident CTAs cover contiguous DRAM
            return (
                triton.cdiv(dim, META["BLOCK_N"]),
                (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
                len(seq_lens_cpu),
            )
        return (
            len(seq_lens_cpu),  # batch_size
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        FAST=FAST,
        GRID_FEATS_FAST=FAST,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out
