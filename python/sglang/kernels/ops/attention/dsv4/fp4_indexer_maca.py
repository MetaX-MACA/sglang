from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


INDEX_HEAD_DIM = 128
PAYLOAD_BYTES = tl.constexpr(64)
SCALE_BYTES = tl.constexpr(4)

HAS_FP4_TO_FP_SCALED = hasattr(tl, "fp4_to_fp_scaled")
_AUTOTUNE_ENABLED = os.environ.get(
    "SGLANG_FP4_INDEXER_LOGIT_AUTOTUNE", "0"
).lower() in {"1", "true", "yes", "on"}


def _device_arch(device: torch.device | int | None = None) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return int(major) * 10 + int(minor)


def _arch_group(arch: int) -> int | None:
    """Group CUDA-compatible architecture values for configuration selection."""
    if arch == 80:
        return 80
    if arch == 86:
        return 86
    if 87 <= arch < 90:
        return 87
    return None


def supports_fp4_to_fp_scaled(
    device: torch.device | int | None = None, *, arch: int | None = None
) -> bool:
    if os.environ.get("SGLANG_FP4_INDEXER_KERNEL") == "manual":
        return False
    if not HAS_FP4_TO_FP_SCALED:
        return False
    if arch is None:
        arch = _device_arch(device)
    return _arch_group(arch) == 87


# Normal execution uses exactly one entry from these tables.  The names make
# the checked-in winner visible without reconstructing it from helper logic.
# H32 means H < 64; H64 means H >= 64.
# MACA minreg scheduling avoids spills introduced by the uniform skip branch.
_MANUAL_MINREG_OPTIONS = " -mllvm -metaxgpu-sched-select=metaxgpu-minreg"

MANUAL_BEST_CONFIGS = {
    "arch80_H32": triton.Config(
        {"BLOCK_L": 32, "scenario": "fullstage"},
        num_warps=2,
        num_stages=3,
    ),
    "arch80_H64": triton.Config(
        {"BLOCK_L": 64, "scenario": "fullstage"},
        num_warps=4,
        num_stages=3,
    ),
    "arch86_H32": triton.Config(
        {
            "BLOCK_L": 128,
            "pipeline": "cpasync",
            "scenario": "storeCoalesce",
        },
        num_warps=2,
        num_stages=2,
    ),
    "arch86_H64": triton.Config(
        {
            "BLOCK_L": 64,
            "pipeline": "basic",
            "scenario": "storeCoalesce",
        },
        num_warps=2,
        num_stages=2,
    ),
    "arch87plus_H32": triton.Config(
        {
            "BLOCK_L": 128,
            "pipeline": "cpasync",
            "scenario": "loadCoalesce",
            "extra_options": _MANUAL_MINREG_OPTIONS,
        },
        num_warps=2,
        num_stages=1,
    ),
    "arch87plus_H64": triton.Config(
        {
            "BLOCK_L": 64,
            "pipeline": "basic",
            "scenario": "storeCoalesce",
            "extra_options": _MANUAL_MINREG_OPTIONS,
        },
        num_warps=2,
        num_stages=1,
    ),
}

CVT_BEST_CONFIGS = {
    "H32_SHORT": triton.Config(
        {
            "BLOCK_L": 128,
            "pipeline": "cpasync",
            "scenario": "loadCoalesce",
        },
        num_warps=2,
        num_stages=2,
    ),
    "H32_LONG": triton.Config(
        {
            "BLOCK_L": 128,
            "pipeline": "cpasync",
            "scenario": "storeCoalesce",
        },
        num_warps=2,
        num_stages=2,
    ),
    "H64": triton.Config(
        {
            "BLOCK_L": 64,
            "pipeline": "cpasync",
            "scenario": "storeCoalesce",
        },
        num_warps=2,
        num_stages=2,
    ),
}


def _make_arch87plus_autotune_configs() -> list[triton.Config]:
    """Shared 72-config search space validated for both arch87plus kernels."""
    return [
        triton.Config(
            {
                "BLOCK_L": block_l,
                "pipeline": pipeline,
                "scenario": scenario,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_l in (32, 64, 128)
        for num_warps in (2, 4, 8)
        for num_stages in (1, 2)
        for pipeline in ("cpasync", "basic")
        for scenario in ("storeCoalesce", "loadCoalesce")
    ]


def _make_arch80_autotune_configs() -> list[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_L": block_l, "scenario": "fullstage"},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_l in (32, 64, 128)
        for num_warps in (2, 4)
        for num_stages in (1, 2, 3)
    ]


def _make_arch86_autotune_configs() -> list[triton.Config]:
    """Search legacy fullstage and the 72 pipeline/scenario combinations."""
    return _make_arch80_autotune_configs() + _make_arch87plus_autotune_configs()


def _make_manual_autotune_configs() -> list[triton.Config]:
    family = _arch_group(_device_arch())
    if family == 80:
        return _make_arch80_autotune_configs()
    if family == 86:
        return _make_arch86_autotune_configs()
    configs = _make_arch87plus_autotune_configs()
    if family == 87:
        configs += [
            triton.Config(
                {**config.kwargs, "extra_options": _MANUAL_MINREG_OPTIONS},
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            for config in configs
        ]
    return configs


def _best_manual_config(
    configs: list[triton.Config], named_args: dict, **kwargs
) -> list[triton.Config]:
    heads = int(kwargs.get("H", named_args.get("H", 0)))
    family = _arch_group(_device_arch())
    config_group = (
        {80: "arch80", 86: "arch86", 87: "arch87plus"}.get(family, "arch80")
    )
    head_group = "H64" if heads >= 64 else "H32"
    return [MANUAL_BEST_CONFIGS[f"{config_group}_{head_group}"]]


def _best_cvt_config(
    configs: list[triton.Config], named_args: dict, **kwargs
) -> list[triton.Config]:
    heads = int(kwargs.get("H", named_args.get("H", 0)))
    length = int(kwargs.get("L", named_args.get("L", 0)))
    if heads >= 64:
        name = "H64"
    elif length >= 32768:
        name = "H32_LONG"
    else:
        name = "H32_SHORT"
    return [CVT_BEST_CONFIGS[name]]


@triton.jit
def _e2m1_decode(code):
    """Decode E2M1 through exact float32 exponent-field construction."""
    e = ((code >> 1) & 3).to(tl.int32)
    m = (code & 1).to(tl.int32)
    mf = m.to(tl.float32)
    mantissa = tl.where(e == 0, mf, 1.0 + mf * 0.5)
    exponent = tl.cast((e + 126) << 23, tl.float32, bitcast=True)
    value = mantissa * exponent
    return tl.where((code >> 3) == 1, -value, value)


@triton.jit
def _fp4_index_logits_kernel_manual(
    q_ptr,
    w_ptr,
    slots_ptr,
    lens_ptr,
    table_ptr,
    out_ptr,
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)

    n_vis = tl.load(lens_ptr + b)
    # Read the live device length on every graph replay. Entirely invisible
    # tiles need only initialize output; partial tiles keep all BF16 rounds.
    if lb * BLOCK_L >= n_vis:
        tl.store(out_ptr + b * L + offs_l, float("-inf"), mask=offs_l < L)
    else:
        valid = offs_l < tl.minimum(n_vis, L)
        slot = tl.load(
            slots_ptr + b * L + offs_l, mask=offs_l < L, other=0
        ).to(tl.int64)
        page = slot // page_size
        off = slot % page_size
        row_base = page * row_stride

        pay = tl.load(
            table_ptr
            + row_base[:, None]
            + off[:, None] * PAYLOAD_BYTES
            + offs_i[None, :],
            mask=valid[:, None],
            other=0,
        )
        low = _e2m1_decode(pay & 0x0F)
        high = _e2m1_decode((pay >> 4) & 0x0F)
        sc_idx = offs_i // 16
        exps = tl.load(
            table_ptr
            + row_base[:, None]
            + page_size * PAYLOAD_BYTES
            + off[:, None] * SCALE_BYTES
            + sc_idx[None, :],
            mask=valid[:, None],
            other=127,
        )
        scale = tl.cast(exps.to(tl.int32) << 23, tl.float32, bitcast=True)
        k_low = (low * scale).to(tl.bfloat16)
        k_high = (high * scale).to(tl.bfloat16)

        keys = tl.interleave(k_low, k_high)
        q = tl.load(
            q_ptr
            + b * stride_qb
            + offs_h[None, :] * stride_qh
            + tl.arange(0, 2 * HALF_D)[:, None]
        )

        acc = tl.dot(keys, q)
        s = acc.to(tl.bfloat16).to(tl.float32)
        s = tl.maximum(s, 0.0)
        w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
        s = (s * w[None, :]).to(tl.bfloat16).to(tl.float32)
        logit = tl.sum(s, axis=1).to(tl.bfloat16).to(tl.float32)
        logit = tl.where(valid, logit, float("-inf"))
        tl.store(out_ptr + b * L + offs_l, logit, mask=offs_l < L)


_fp4_index_logits_kernel_manual = triton.autotune(
    configs=(
        _make_manual_autotune_configs()
        if _AUTOTUNE_ENABLED
        else list(MANUAL_BEST_CONFIGS.values())
    ),
    key=["H", "L"],
    prune_configs_by=(
        None
        if _AUTOTUNE_ENABLED
        else {"early_config_prune": _best_manual_config}
    ),
)(_fp4_index_logits_kernel_manual)


if HAS_FP4_TO_FP_SCALED:

    @triton.jit
    def _fp4_index_logits_kernel_cvt(
        q_ptr,
        w_ptr,
        slots_ptr,
        lens_ptr,
        table_ptr,
        out_ptr,
        L: tl.constexpr,
        page_size,
        row_stride,
        stride_qb,
        stride_qh,
        stride_wb,
        H: tl.constexpr,
        HALF_D: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        b = tl.program_id(0)
        lb = tl.program_id(1)
        offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_h = tl.arange(0, H)
        offs_i = tl.arange(0, HALF_D)

        n_vis = tl.load(lens_ptr + b)
        # Read the live device length on every graph replay. Entirely invisible
        # tiles need only initialize output; partial tiles keep all BF16 rounds.
        if lb * BLOCK_L >= n_vis:
            tl.store(out_ptr + b * L + offs_l, float("-inf"), mask=offs_l < L)
        else:
            valid = offs_l < tl.minimum(n_vis, L)
            slot = tl.load(
                slots_ptr + b * L + offs_l, mask=offs_l < L, other=0
            ).to(tl.int64)
            page = slot // page_size
            off = slot % page_size
            row_base = page * row_stride

            pay = tl.load(
                table_ptr
                + row_base[:, None]
                + off[:, None] * PAYLOAD_BYTES
                + offs_i[None, :],
                mask=valid[:, None],
                other=0,
            )
            sc_idx = offs_i // 16
            exps = tl.load(
                table_ptr
                + row_base[:, None]
                + page_size * PAYLOAD_BYTES
                + off[:, None] * SCALE_BYTES
                + sc_idx[None, :],
                mask=valid[:, None],
                other=127,
            )
            keys = tl.fp4_to_fp_scaled(pay, exps, tl.bfloat16, axis=1)
            q = tl.load(
                q_ptr
                + b * stride_qb
                + offs_h[:, None] * stride_qh
                + tl.arange(0, 2 * HALF_D)[None, :]
            )

            acc = tl.dot(q, tl.trans(keys))
            s = acc.to(tl.bfloat16).to(tl.float32)
            s = tl.maximum(s, 0.0)
            w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
            s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
            logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
            logit = tl.where(valid, logit, float("-inf"))
            tl.store(out_ptr + b * L + offs_l, logit, mask=offs_l < L)

    _fp4_index_logits_kernel_cvt = triton.autotune(
        configs=(
            _make_arch87plus_autotune_configs()
            if _AUTOTUNE_ENABLED
            else list(CVT_BEST_CONFIGS.values())
        ),
        key=["H", "L"],
        prune_configs_by=(
            None
            if _AUTOTUNE_ENABLED
            else {"early_config_prune": _best_cvt_config}
        ),
    )(_fp4_index_logits_kernel_cvt)
else:
    _fp4_index_logits_kernel_cvt = None


def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Return ``[B, L]`` FP4 index logits using the supported internal kernel.

    The six-argument public contract intentionally matches
    ``sm90_fp4_indexer.fp4_index_logits_decode``.  Benchmark/autotune tools may
    set ``SGLANG_FP4_INDEXER_KERNEL`` to exercise one internal kernel.
    """
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    batch, heads, _ = q.shape
    length = slots.shape[1]
    assert table.dtype == torch.uint8 and table.dim() == 2
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    lens = lens.to(torch.int64).contiguous()
    out = torch.empty((batch, length), dtype=torch.float32, device=q.device)
    if length == 0:
        return out

    kernel = _fp4_index_logits_kernel_manual
    if supports_fp4_to_fp_scaled(q.device):
        assert _fp4_index_logits_kernel_cvt is not None
        kernel = _fp4_index_logits_kernel_cvt

    grid = lambda meta: (batch, triton.cdiv(length, meta["BLOCK_L"]))
    kernel[grid](
        q,
        weights,
        slots,
        lens,
        table,
        out,
        length,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        H=heads,
        HALF_D=INDEX_HEAD_DIM // 2,
    )
    return out
