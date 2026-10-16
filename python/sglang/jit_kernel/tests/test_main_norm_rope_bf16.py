import math
import sys
import types

import pytest
import torch
import triton


def _install_deep_gemm_stub() -> None:
    # deepseek_v4.py imports deep_gemm_wrapper for unrelated GEMM helpers. Some
    # platform builds initialize heavy backend modules during that import, so keep
    # this focused kernel test isolated from those dependencies.
    name = "sglang.srt.layers.deep_gemm_wrapper"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.ENABLE_JIT_DEEPGEMM = False

    def _unavailable(*args, **kwargs):
        raise RuntimeError("deep_gemm_wrapper is stubbed in this unit test")

    stub.gemm_nt_bf16bf16f32 = _unavailable
    sys.modules[name] = stub


_install_deep_gemm_stub()

from sglang.jit_kernel.dsv4 import (  # noqa: E402
    fused_k_norm_rope_flashmla,
    fused_q_norm_rope,
)
from sglang.jit_kernel.utils import (  # noqa: E402
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.layers.attention.dsv4.dequant_k_cache import (  # noqa: E402
    dequantize_k_cache_paged,
)
from sglang.srt.layers.fused_qk_norm_rope_store import (  # noqa: E402
    fused_qk_norm_rope_swa_store,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-6

HEAD_DIM_Q = 128
HEAD_DIM_K = 512
HEAD_DIM_INDEXER = 128
ROPE_DIM = 64
MAX_POS = 128
INDEXER_WEIGHT_SCALE = 0.625


def _make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


@cache_once
def _jit_indexer_module(input_dtype: torch.dtype, output_dtype: torch.dtype):
    # The normal Python API still defaults to fp8 output, so this test directly
    # instantiates the third template argument to cover the new bf16 output path:
    #   FusedQIndexerRopeHadamardQuantKernel<DType, kUsePDL, OutDType>
    args = make_cpp_args(input_dtype, is_arch_support_pdl(), output_dtype)
    return load_jit(
        _make_name("main_q_indexer_rope_hadamard_quant_bf16_test"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def _make_freqs_cis(max_pos: int = MAX_POS, rope_dim: int = ROPE_DIM) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rope_dim, 2, dtype=torch.float32, device=DEVICE)
            / rope_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def _freqs_real(freqs_cis: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(freqs_cis).flatten(-2).contiguous()


def _rmsnorm_self(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)


def _rmsnorm_weighted(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return _rmsnorm_self(x, eps) * w.float()


def _apply_tail_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    out = x.float().clone()
    tail = out[..., -ROPE_DIM:].reshape(*out.shape[:-1], ROPE_DIM // 2, 2)
    freq = torch.view_as_real(freqs_cis[positions.long()]).to(torch.float32)
    while freq.ndim < tail.ndim:
        freq = freq.unsqueeze(1)
    real = tail[..., 0]
    imag = tail[..., 1]
    freq_real = freq[..., 0]
    freq_imag = freq[..., 1]
    rotated = torch.stack(
        (real * freq_real - imag * freq_imag, real * freq_imag + imag * freq_real),
        dim=-1,
    ).flatten(-2)
    out[..., -ROPE_DIM:] = rotated
    return out


def _fwht_128(x: torch.Tensor) -> torch.Tensor:
    out = x.float().clone()
    h = 1
    while h < HEAD_DIM_INDEXER:
        y = out.reshape(*out.shape[:-1], -1, h * 2)
        a = y[..., :h].clone()
        b = y[..., h:].clone()
        y[..., :h] = a + b
        y[..., h:] = a - b
        h *= 2
    return out * (1.0 / math.sqrt(HEAD_DIM_INDEXER))


def _indexer_reference(
    q_input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    return _fwht_128(_apply_tail_rope(q_input, freqs_cis, positions))


def _run_indexer(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    weight: torch.Tensor,
    weights_out: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    module = _jit_indexer_module(q_input.dtype, q_output.dtype)
    module.forward(
        q_input,
        q_output,
        weight,
        weights_out,
        INDEXER_WEIGHT_SCALE,
        _freqs_real(freqs_cis),
        positions,
    )
    torch.cuda.synchronize()


# ============================================================================
# (1) FusedQNormRopeKernel: bf16 input/output
# ============================================================================


@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
def test_fused_q_norm_rope_bf16(positions_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    batch_size = 7
    num_heads = 3
    q_input = torch.randn(
        batch_size, num_heads, HEAD_DIM_Q, device=DEVICE, dtype=DTYPE
    )
    q_output = torch.empty_like(q_input)
    positions = torch.randint(0, MAX_POS, (batch_size,), device=DEVICE).to(
        positions_dtype
    )
    freqs_cis = _make_freqs_cis()

    fused_q_norm_rope(q_input, q_output, EPS, freqs_cis, positions)
    torch.cuda.synchronize()

    ref = _apply_tail_rope(
        _rmsnorm_self(q_input, EPS).to(DTYPE), freqs_cis, positions
    ).to(DTYPE)
    triton.testing.assert_close(q_output.float(), ref.float(), atol=2e-2, rtol=2e-2)


# ============================================================================
# (2) FusedKNormRopeFlashMLAKernel: bf16 cache + original fp8 cache regression
# ============================================================================


@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
def test_fused_k_norm_rope_flashmla_bf16_cache(positions_dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    batch_size = 5
    page_size = 8
    num_pages = 3
    kv_storage = torch.randn(
        batch_size, HEAD_DIM_K + 16, device=DEVICE, dtype=DTYPE
    )
    kv = kv_storage[:, 16:]
    kv_weight = torch.randn(HEAD_DIM_K, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, MAX_POS, (batch_size,), device=DEVICE).to(
        positions_dtype
    )
    out_loc = torch.tensor([0, 3, 7, 9, 17], device=DEVICE, dtype=torch.int32)
    freqs_cis = _make_freqs_cis()
    kvcache = torch.full(
        (num_pages, page_size * HEAD_DIM_K),
        float("nan"),
        device=DEVICE,
        dtype=DTYPE,
    )

    fused_k_norm_rope_flashmla(
        kv, kv_weight, EPS, freqs_cis, positions, out_loc, kvcache, page_size
    )
    torch.cuda.synchronize()

    ref = _apply_tail_rope(
        _rmsnorm_weighted(kv, kv_weight, EPS).to(DTYPE), freqs_cis, positions
    ).to(DTYPE)
    flat_cache = kvcache.view(-1, HEAD_DIM_K)
    triton.testing.assert_close(
        flat_cache[out_loc.long()].float(), ref.float(), atol=2e-2, rtol=2e-2
    )


def _flashmla_fp8_page_bytes(page_size: int) -> int:
    return math.ceil(584 * page_size / 576) * 576


def _flashmla_int8_page_bytes(page_size: int) -> int:
    return math.ceil(608 * page_size / 576) * 576


def test_fused_k_norm_rope_flashmla_fp8_cache_interface_regression() -> None:
    torch.manual_seed(2)
    batch_size = 4
    page_size = 8
    num_pages = 2
    page_bytes = _flashmla_fp8_page_bytes(page_size)
    kv = torch.randn(batch_size, HEAD_DIM_K, device=DEVICE, dtype=DTYPE)
    kv_weight = torch.randn(HEAD_DIM_K, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, MAX_POS, (batch_size,), device=DEVICE, dtype=torch.int32)
    out_loc = torch.tensor([0, 2, 7, 9], device=DEVICE, dtype=torch.int32)
    freqs_cis = _make_freqs_cis()
    kvcache = torch.zeros((num_pages, page_bytes), device=DEVICE, dtype=torch.uint8)

    fused_k_norm_rope_flashmla(
        kv, kv_weight, EPS, freqs_cis, positions, out_loc, kvcache, page_size
    )
    torch.cuda.synchronize()

    ref = _apply_tail_rope(
        _rmsnorm_weighted(kv, kv_weight, EPS).to(DTYPE), freqs_cis, positions
    ).to(DTYPE)
    host_cache = kvcache.cpu()
    for i, loc in enumerate(out_loc.cpu().tolist()):
        page = loc // page_size
        offset = loc % page_size
        value_start = offset * 576
        rope_start = value_start + 448
        scale_start = 576 * page_size + offset * 8
        assert torch.count_nonzero(host_cache[page, value_start : value_start + 448]) > 0
        assert torch.count_nonzero(host_cache[page, scale_start : scale_start + 7]) > 0
        rope = host_cache[page, rope_start : rope_start + ROPE_DIM * 2].view(torch.bfloat16)
        triton.testing.assert_close(
            rope.float(), ref[i, -ROPE_DIM:].cpu().float(), atol=2e-2, rtol=2e-2
        )


def test_fused_k_norm_rope_flashmla_int8_cache() -> None:
    torch.manual_seed(3)
    batch_size = 5
    page_size = 8
    num_pages = 3
    page_bytes = _flashmla_int8_page_bytes(page_size)
    kv = torch.randn(batch_size, HEAD_DIM_K, device=DEVICE, dtype=DTYPE)
    kv_weight = torch.randn(HEAD_DIM_K, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(
        0, MAX_POS, (batch_size,), device=DEVICE, dtype=torch.int32
    )
    out_loc = torch.tensor([0, 3, 7, 9, 17], device=DEVICE, dtype=torch.int32)
    freqs_cis = _make_freqs_cis()
    kvcache = torch.zeros(
        (num_pages, page_bytes), device=DEVICE, dtype=torch.int8
    )

    fused_k_norm_rope_flashmla(
        kv, kv_weight, EPS, freqs_cis, positions, out_loc, kvcache, page_size
    )
    torch.cuda.synchronize()

    out = dequantize_k_cache_paged(kvcache, out_loc, page_size)[:, 0]
    ref = _apply_tail_rope(
        _rmsnorm_weighted(kv, kv_weight, EPS), freqs_cis, positions
    )
    groups = ref[:, :448].view(batch_size, 7, 64)
    scales = groups.abs().amax(dim=-1).clamp(min=1.0e-4) / 127.0
    quant = torch.round(groups / scales.unsqueeze(-1)).clamp(-127, 127)
    expected = ref.to(torch.bfloat16)
    expected[:, :448] = (quant * scales.unsqueeze(-1)).reshape(
        batch_size, 448
    ).to(torch.bfloat16)

    triton.testing.assert_close(
        out.float(), expected.float(), atol=4e-2, rtol=4e-2
    )

    # Verify the physical scale record directly: 7 FP32 values + 4-byte pad.
    raw = kvcache.view(torch.uint8)
    for i, loc in enumerate(out_loc.tolist()):
        page, offset = divmod(loc, page_size)
        scale_byte = page_size * 576 + offset * 32
        stored_scales = raw[page, scale_byte : scale_byte + 28].view(torch.float32)
        triton.testing.assert_close(
            stored_scales, scales[i], atol=1e-6, rtol=1e-5
        )


def test_fused_qk_norm_rope_swa_store_int8_cache() -> None:
    """Cover the primary Triton SWA writer used by deepseek_v4._forward_prepare."""
    torch.manual_seed(4)
    batch_size = 5
    num_heads = 2
    page_size = 8
    num_pages = 3
    q = torch.randn(
        batch_size, num_heads * HEAD_DIM_K, device=DEVICE, dtype=DTYPE
    )
    kv = torch.randn(batch_size, HEAD_DIM_K, device=DEVICE, dtype=DTYPE)
    kv_input = kv.clone()
    kv_weight = torch.randn(HEAD_DIM_K, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(
        0, MAX_POS, (batch_size,), device=DEVICE, dtype=torch.int32
    )
    out_loc = torch.tensor([0, 3, 7, 9, 17], device=DEVICE, dtype=torch.int32)
    freqs_cis = _make_freqs_cis()
    angles = torch.angle(freqs_cis)
    cos_cache = torch.cos(angles).to(DTYPE)
    sin_cache = torch.sin(angles).to(DTYPE)
    cache = torch.zeros(
        (num_pages, _flashmla_int8_page_bytes(page_size)),
        device=DEVICE,
        dtype=torch.int8,
    )

    fused_qk_norm_rope_swa_store(
        q=q,
        kv=kv,
        q_norm_weight=None,
        kv_norm_weight=kv_weight,
        q_rms_eps=EPS,
        kv_rms_eps=EPS,
        rope_head_dim=ROPE_DIM,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        positions=positions,
        swa_cache=cache,
        swa_loc=out_loc,
        swa_page_size=page_size,
        dtype=DTYPE,
    )
    torch.cuda.synchronize()

    out = dequantize_k_cache_paged(cache, out_loc, page_size)[:, 0]
    ref = _apply_tail_rope(
        _rmsnorm_weighted(kv_input, kv_weight, EPS), freqs_cis, positions
    )
    # This fused Triton writer first publishes normalized NOPE back to the
    # BF16 kv tensor, then reloads that BF16 value for cache quantization.
    groups = ref[:, :448].to(torch.bfloat16).float().view(batch_size, 7, 64)
    scales = groups.abs().amax(dim=-1).clamp(min=1.0e-4) / 127.0
    quant = torch.round(groups / scales.unsqueeze(-1)).clamp(-127, 127)
    expected = ref.to(torch.bfloat16)
    expected[:, :448] = (quant * scales.unsqueeze(-1)).reshape(
        batch_size, 448
    ).to(torch.bfloat16)
    triton.testing.assert_close(
        out.float(), expected.float(), atol=4e-2, rtol=4e-2
    )


# ============================================================================
# (3) FusedQIndexerRopeHadamardQuantKernel: bf16 output + fp8 regression
# ============================================================================


@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
def test_fused_q_indexer_rope_hadamard_quant_bf16_output(
    positions_dtype: torch.dtype,
) -> None:
    torch.manual_seed(30)
    batch_size = 5
    num_heads = 3

    q_input = torch.randn(
        batch_size, num_heads, HEAD_DIM_INDEXER, device=DEVICE, dtype=DTYPE
    )
    q_output = torch.empty_like(q_input)
    weight = torch.randn(batch_size, num_heads, device=DEVICE, dtype=DTYPE)
    weights_out = torch.empty(
        batch_size, num_heads, 1, device=DEVICE, dtype=torch.float32
    )
    positions = torch.tensor([0, 1, 7, 31, 63], device=DEVICE, dtype=positions_dtype)
    freqs_cis = _make_freqs_cis()

    _run_indexer(q_input, q_output, weight, weights_out, freqs_cis, positions)

    ref = _indexer_reference(q_input, freqs_cis, positions).to(DTYPE)
    triton.testing.assert_close(q_output.float(), ref.float(), atol=2e-2, rtol=2e-2)

    ref_weight = (weight.float() * INDEXER_WEIGHT_SCALE).unsqueeze(-1)
    triton.testing.assert_close(weights_out, ref_weight, atol=1e-5, rtol=1e-5)


def test_fused_q_indexer_rope_hadamard_quant_fp8_regression() -> None:
    torch.manual_seed(31)
    batch_size = 4
    num_heads = 2

    q_input = torch.randn(
        batch_size, num_heads, HEAD_DIM_INDEXER, device=DEVICE, dtype=DTYPE
    )
    q_fp8 = torch.empty(
        batch_size,
        num_heads,
        HEAD_DIM_INDEXER,
        device=DEVICE,
        dtype=torch.float8_e4m3fn,
    )
    # Avoid zeros to make q_scale recovery from weights_out stable.
    weight = (
        torch.rand(batch_size, num_heads, device=DEVICE, dtype=DTYPE) + 0.5
    ).contiguous()
    weights_out = torch.empty(
        batch_size, num_heads, 1, device=DEVICE, dtype=torch.float32
    )
    positions = torch.tensor([0, 2, 9, 17], device=DEVICE, dtype=torch.int32)
    freqs_cis = _make_freqs_cis()

    _run_indexer(q_input, q_fp8, weight, weights_out, freqs_cis, positions)

    ref = _indexer_reference(q_input, freqs_cis, positions)
    q_scale = weights_out.squeeze(-1) / (weight.float() * INDEXER_WEIGHT_SCALE)
    dequant = q_fp8.float() * q_scale.unsqueeze(-1)
    triton.testing.assert_close(dequant, ref, atol=8e-2, rtol=8e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

