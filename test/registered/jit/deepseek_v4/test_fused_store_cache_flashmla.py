"""
Test for the fused_store_cache flashmla kernel (SWA KV cache store).

Covers both packed layouts the kernel supports:
  - fp8 : 584B/token (576B value + 8B UE8M0 scales), uint8 storage
  - int8: 608B/token (576B value + 32B scale record: 7 fp32 scales + 4B pad),
          int8 storage — per-64-group scale = max(1e-4, abs_max) / 127,
          round-to-nearest int8 (matches main_norm_rope.cuh convention)

Design Notes:
  1. torch.cuda.synchronize() needed after TVM FFI kernel call.
  2. The int8 reference replicates the kernel formula in fp32 and should be
     bit-exact (all ops are deterministic IEEE fp32 + round-half-away).
  3. The fp8 path uses UE8M0 scales; validated via dequantized roundtrip
     instead of byte comparison (cast tie-breaks differ between impls).
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.jit_kernel.dsv4 import fused_store_cache
from sglang.test.ci.ci_register import register_cuda_ci

try:
    from sglang.jit_kernel.utils import is_hip_runtime

    _is_hip = is_hip_runtime()
except Exception:
    _is_hip = False

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
VALUE_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2  # 576
FP8_SCALE_BYTES_PER_TOKEN = 8
INT8_SCALE_BYTES_PER_TOKEN = 32
FP8_BYTES_PER_TOKEN = VALUE_BYTES_PER_TOKEN + FP8_SCALE_BYTES_PER_TOKEN  # 584
INT8_BYTES_PER_TOKEN = VALUE_BYTES_PER_TOKEN + INT8_SCALE_BYTES_PER_TOKEN  # 608
FP8_E4M3_MAX = 448.0
INT8_QMAX = 127.0


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if _is_hip:
        pytest.skip("CUDA JIT fused_store_cache is CUDA-specific")


def _bytes_per_token(store_dtype: torch.dtype) -> int:
    return FP8_BYTES_PER_TOKEN if store_dtype == torch.uint8 else INT8_BYTES_PER_TOKEN


def _padded_page_bytes(page_size: int, store_dtype: torch.dtype) -> int:
    # Matches kv_cache_layout.packed_page_bytes: align to 576B pages.
    logical = page_size * _bytes_per_token(store_dtype)
    return (logical + VALUE_BYTES_PER_TOKEN - 1) // VALUE_BYTES_PER_TOKEN * VALUE_BYTES_PER_TOKEN


def _make_buffer(num_pages: int, page_size: int, store_dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(
        (num_pages, _padded_page_bytes(page_size, store_dtype)),
        dtype=store_dtype,
        device="cuda",
    )


def _round_half_away(v: torch.Tensor) -> torch.Tensor:
    return torch.sign(v) * torch.floor(v.abs() + 0.5)


def _reference_int8_store(
    key_bf16: torch.Tensor, loc: torch.Tensor, num_pages: int, page_size: int
) -> torch.Tensor:
    """Replicate the int8 kernel formula in fp32 (bit-exact)."""
    buf = _make_buffer(num_pages, page_size, torch.int8)
    buf_flat = buf.reshape(-1)
    page_bytes = _padded_page_bytes(page_size, torch.int8)
    scale_section = VALUE_BYTES_PER_TOKEN * page_size
    key_f32 = key_bf16.float()

    for i in range(key_bf16.shape[0]):
        idx = int(loc[i].item())
        page, off = idx // page_size, idx % page_size
        base = page * page_bytes + off * VALUE_BYTES_PER_TOKEN

        row = key_f32[i]
        for g in range(NOPE_DIM // 64):
            grp = row[g * 64 : (g + 1) * 64]
            abs_max = grp.abs().max()
            scale = torch.clamp(abs_max, min=1e-4) / INT8_QMAX
            q = _round_half_away(grp / scale).clamp(-128, 127).to(torch.int8)
            buf_flat[base + g * 64 : base + (g + 1) * 64] = q
            s_off = page * page_bytes + scale_section + off * INT8_SCALE_BYTES_PER_TOKEN + g * 4
            buf_flat[s_off : s_off + 4] = (
                scale.reshape(1).to(torch.float32).view(torch.int8)
            )

        rope_u8 = key_bf16[i, NOPE_DIM:].reshape(-1).view(torch.uint8).to(torch.int8)
        buf_flat[base + NOPE_DIM : base + VALUE_BYTES_PER_TOKEN] = rope_u8

    return buf


def _gather_int8(
    buf: torch.Tensor, loc: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (nope int8 [N,448], scales f32 [N,7], rope bf16 [N,64])."""
    buf_flat = buf.reshape(-1)
    page_bytes = buf.shape[1]
    scale_section = VALUE_BYTES_PER_TOKEN * page_size
    N = loc.shape[0]
    nope = torch.empty((N, NOPE_DIM), dtype=torch.int8, device=buf.device)
    scales = torch.empty((N, NOPE_DIM // 64), dtype=torch.float32, device=buf.device)
    rope = torch.empty((N, ROPE_DIM), dtype=torch.bfloat16, device=buf.device)
    for i in range(N):
        idx = int(loc[i].item())
        page, off = idx // page_size, idx % page_size
        base = page * page_bytes + off * VALUE_BYTES_PER_TOKEN
        nope[i] = buf_flat[base : base + NOPE_DIM]
        s_off = page * page_bytes + scale_section + off * INT8_SCALE_BYTES_PER_TOKEN
        scales[i] = buf_flat[s_off : s_off + 28].view(torch.float32)
        rope[i] = buf_flat[base + NOPE_DIM : base + VALUE_BYTES_PER_TOKEN].view(
            torch.bfloat16
        )
    return nope, scales, rope


# ---------------------------------------------------------------------------
# int8 layout: bit-exact against the replicated formula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page_size", [64, 256])
@pytest.mark.parametrize("num_tokens,base_index", [(1, 0), (64, 5), (257, 65)])
def test_int8_matches_reference_formula(num_tokens, base_index, page_size):
    _skip_if_unavailable()
    key = torch.randn((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    loc = (
        base_index + torch.randperm(num_tokens, device="cuda", dtype=torch.int64)
    ).contiguous()
    num_pages = int(loc.max().item()) // page_size + 2

    ref_buf = _reference_int8_store(key, loc, num_pages, page_size)

    out_buf = _make_buffer(num_pages, page_size, torch.int8)
    fused_store_cache(
        input=key, cache=out_buf, indices=loc, page_size=page_size, type="flashmla"
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_buf, ref_buf)


def test_int8_zero_input():
    _skip_if_unavailable()
    key = torch.zeros((4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    loc = torch.arange(4, device="cuda", dtype=torch.int64)
    page_size = 64

    out_buf = _make_buffer(1, page_size, torch.int8)
    fused_store_cache(
        input=key, cache=out_buf, indices=loc, page_size=page_size, type="flashmla"
    )
    torch.cuda.synchronize()

    nope, scales, rope = _gather_int8(out_buf, loc, page_size)
    assert (nope == 0).all()
    assert (rope == 0).all()
    torch.testing.assert_close(
        scales, torch.full_like(scales, 1e-4 / INT8_QMAX), rtol=1e-5, atol=1e-10
    )


def test_int8_int32_indices():
    _skip_if_unavailable()
    key = torch.randn((33, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    loc = torch.randperm(300, device="cuda", dtype=torch.int64)[:33].to(torch.int32)
    page_size = 64
    num_pages = int(loc.max().item()) // page_size + 2

    ref_buf = _reference_int8_store(key, loc, num_pages, page_size)
    out_buf = _make_buffer(num_pages, page_size, torch.int8)
    fused_store_cache(
        input=key, cache=out_buf, indices=loc, page_size=page_size, type="flashmla"
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out_buf, ref_buf)


def test_int8_roundtrip_reconstruction():
    _skip_if_unavailable()
    key = torch.randn((128, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    loc = torch.randperm(512, device="cuda", dtype=torch.int64)[:128]
    page_size = 64
    num_pages = int(loc.max().item()) // page_size + 2

    out_buf = _make_buffer(num_pages, page_size, torch.int8)
    fused_store_cache(
        input=key, cache=out_buf, indices=loc, page_size=page_size, type="flashmla"
    )
    torch.cuda.synchronize()

    nope, scales, rope = _gather_int8(out_buf, loc, page_size)
    reconstructed = torch.cat(
        [nope.float() * scales, rope.float()], dim=-1
    )
    # int8 quantization: 1/127 relative error per group + exact bf16 rope
    torch.testing.assert_close(
        reconstructed, key.float(), rtol=1 / INT8_QMAX + 1e-3, atol=1e-2
    )


def test_int8_padding_bytes_untouched():
    """Tokens must only write their own 608B slot; page padding stays zero."""
    _skip_if_unavailable()
    key = torch.randn((8, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    loc = torch.arange(8, device="cuda", dtype=torch.int64)
    page_size = 64
    num_pages = 1

    out_buf = _make_buffer(num_pages, page_size, torch.int8)
    fused_store_cache(
        input=key, cache=out_buf, indices=loc, page_size=page_size, type="flashmla"
    )
    torch.cuda.synchronize()

    page_bytes = out_buf.shape[1]
    assert page_bytes > page_size * INT8_BYTES_PER_TOKEN  # padding exists
    # scale pad bytes (28..31 of each scale record) stay zero
    for off in range(8):
        s_off = VALUE_BYTES_PER_TOKEN * page_size + off * INT8_SCALE_BYTES_PER_TOKEN
        assert (out_buf.reshape(-1)[s_off + 28 : s_off + 32] == 0).all()
    # trailing page padding stays zero
    used = page_size * INT8_BYTES_PER_TOKEN
    assert (out_buf.reshape(-1)[used:] == 0).all()


# ---------------------------------------------------------------------------
# fp8 layout: regression (unchanged behavior)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_tokens", [1, 64, 257])
def test_fp8_roundtrip_regression(num_tokens):
    _skip_if_unavailable()
    key = torch.randn((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    loc = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    page_size = 64
    num_pages = num_tokens // page_size + 1

    out_buf = _make_buffer(num_pages, page_size, torch.uint8)
    fused_store_cache(
        input=key, cache=out_buf, indices=loc, page_size=page_size, type="flashmla"
    )
    torch.cuda.synchronize()

    buf_flat = out_buf.reshape(-1)
    page_bytes = out_buf.shape[1]
    for i in range(num_tokens):
        idx = int(loc[i].item())
        page, off = idx // page_size, idx % page_size
        base = page * page_bytes + off * VALUE_BYTES_PER_TOKEN
        nope = buf_flat[base : base + NOPE_DIM].view(torch.float8_e4m3fn).float()
        scale_u8 = buf_flat[
            page * page_bytes
            + VALUE_BYTES_PER_TOKEN * page_size
            + off * FP8_SCALE_BYTES_PER_TOKEN
        ].item()
        assert scale_u8 >= 127 or scale_u8 == 0, "UE8M0 scale byte must be exponent-only"
        scale = 2.0 ** (scale_u8 - 127) if scale_u8 > 0 else 0.0
        rope = buf_flat[base + NOPE_DIM : base + VALUE_BYTES_PER_TOKEN].view(
            torch.bfloat16
        )
        torch.testing.assert_close(rope, key[i, NOPE_DIM:])
        torch.testing.assert_close(
            nope * scale, key[i, :NOPE_DIM].float(), rtol=0.15, atol=5e-2
        )


# ---------------------------------------------------------------------------
# Layout contract enforcement
# ---------------------------------------------------------------------------
def test_wrong_stride_rejected():
    _skip_if_unavailable()
    key = torch.randn((4, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    loc = torch.arange(4, device="cuda", dtype=torch.int64)
    # non-padded stride -> TensorMatcher must reject
    bad = torch.zeros((1, 64 * INT8_BYTES_PER_TOKEN), dtype=torch.int8, device="cuda")
    with pytest.raises(Exception):
        fused_store_cache(
            input=key, cache=bad, indices=loc, page_size=64, type="flashmla"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
