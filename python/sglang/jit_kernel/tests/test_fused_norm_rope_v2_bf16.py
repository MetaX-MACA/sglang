import math

import pytest
import torch
import triton

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-6
ROPE_DIM = 64
MAX_POS = 64


@cache_once
def _jit_fused_norm_rope_v2_module(
    input_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    page_size: int,
):
    args = make_cpp_args(
        input_dtype,
        cache_dtype,
        head_dim,
        rope_dim,
        page_size,
        is_arch_support_pdl(),
    )
    return load_jit(
        "dpsk_v4_fused_norm_rope_v2_bf16_test",
        *args,
        cuda_files=["deepseek_v4/fused_norm_rope_v2.cuh"],
        cuda_wrappers=[("forward", f"FusedNormRopeKernel<{args}>::forward")],
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


def _rmsnorm_weighted(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * w.float()


def _apply_tail_rope(x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    tail = out[..., -ROPE_DIM:].float().reshape(*out.shape[:-1], ROPE_DIM // 2, 2)
    freq = torch.view_as_real(freqs_cis[positions]).to(torch.float32)
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
    out[..., -ROPE_DIM:] = rotated.to(out.dtype)
    return out


def _indexer_hadamard_like_kernel(x: torch.Tensor) -> torch.Tensor:
    # Match the CUDA kernel exactly: each lane owns 4 contiguous elements,
    # first does 4-point butterflies inside a lane, then cross-lane xor butterflies.
    assert x.shape[-1] == 128
    ys = []
    for row in x.float():
        data = row.view(32, 4).clone()
        a0 = data[:, 0].clone()
        a1 = data[:, 1].clone()
        a2 = data[:, 2].clone()
        a3 = data[:, 3].clone()
        data[:, 0] = a0 + a1
        data[:, 1] = a0 - a1
        data[:, 2] = a2 + a3
        data[:, 3] = a2 - a3

        a0 = data[:, 0].clone()
        a1 = data[:, 1].clone()
        a2 = data[:, 2].clone()
        a3 = data[:, 3].clone()
        data[:, 0] = a0 + a2
        data[:, 1] = a1 + a3
        data[:, 2] = a0 - a2
        data[:, 3] = a1 - a3

        for mask in [1, 2, 4, 8, 16]:
            old = data.clone()
            for lane in range(32):
                other = old[lane ^ mask]
                data[lane] = other - old[lane] if (lane & mask) else old[lane] + other
        ys.append(data.reshape(128) * (1.0 / math.sqrt(128.0)))
    return torch.stack(ys, dim=0)



_PLAN_BYTES_CANDIDATES = (16, 8, 4)


def _make_decode_plan(seq_lens: torch.Tensor, plan_bytes: int) -> torch.Tensor:
    # fused_norm_rope_v2.cuh expects compress_v2 raw DecodePlan, not a 1-D seq_lens tensor.
    # The kernel only reads plan.seq_len, stored as the first int32 field.
    assert plan_bytes % 4 == 0
    plan = torch.zeros(
        (seq_lens.numel(), plan_bytes),
        device=seq_lens.device,
        dtype=torch.uint8,
    )
    plan.view(torch.int32).view(seq_lens.numel(), plan_bytes // 4)[:, 0] = seq_lens.to(torch.int32)
    return plan


def _run_kernel_decode(
    *,
    head_dim: int,
    page_size: int,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    freqs_real: torch.Tensor,
    seq_lens: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    compress_ratio: int,
) -> None:
    module = _jit_fused_norm_rope_v2_module(
        input_tensor.dtype,
        kvcache.dtype,
        head_dim,
        freqs_real.shape[-1],
        page_size,
    )
    last_error = None
    for plan_bytes in _PLAN_BYTES_CANDIDATES:
        plan = _make_decode_plan(seq_lens, plan_bytes)
        try:
            module.forward(
                input_tensor,
                plan,
                weight,
                EPS,
                freqs_real,
                out_loc,
                kvcache,
                True,  # is_decode
                compress_ratio,
            )
            return
        except RuntimeError as exc:
            msg = str(exc)
            if "compress_v2.cuh" in msg and (
                "Tensor match failed" in msg
                or "Tensor dimension mismatch" in msg
                or "Tensor shape mismatch" in msg
                or "Tensor dtype mismatch" in msg
            ):
                last_error = exc
                continue
            raise
    assert last_error is not None
    raise last_error


def test_fused_norm_rope_v2_indexer_bf16_cache_decode() -> None:
    torch.manual_seed(10)
    batch_size = 6
    head_dim = 128
    page_size = 8
    compress_ratio = 4
    out_loc = torch.tensor([0, 3, 7, 8, 13, 19], device=DEVICE, dtype=torch.int32)
    num_pages = int(out_loc.max().item()) // page_size + 1
    seq_lens = torch.tensor([4, 8, 12, 16, 20, 24], device=DEVICE, dtype=torch.int32)
    positions = seq_lens.long() - compress_ratio

    input_tensor = torch.randn(batch_size, head_dim, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    freqs_cis = _make_freqs_cis()
    kvcache = torch.full(
        (num_pages, page_size * head_dim),
        float("nan"),
        device=DEVICE,
        dtype=DTYPE,
    )

    _run_kernel_decode(
        head_dim=head_dim,
        page_size=page_size,
        input_tensor=input_tensor,
        weight=weight,
        freqs_real=_freqs_real(freqs_cis),
        seq_lens=seq_lens,
        out_loc=out_loc,
        kvcache=kvcache,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    ref = _apply_tail_rope(_rmsnorm_weighted(input_tensor, weight), freqs_cis, positions)
    ref = _indexer_hadamard_like_kernel(ref).to(DTYPE)
    flat_cache = kvcache.view(-1, head_dim)
    triton.testing.assert_close(
        flat_cache[out_loc.long()].float(), ref.float(), atol=3e-2, rtol=3e-2
    )


def test_fused_norm_rope_v2_flashmla_bf16_cache_decode() -> None:
    torch.manual_seed(11)
    batch_size = 5
    head_dim = 512
    page_size = 8
    compress_ratio = 4
    out_loc = torch.tensor([0, 1, 7, 9, 17], device=DEVICE, dtype=torch.int32)
    num_pages = int(out_loc.max().item()) // page_size + 1
    seq_lens = torch.tensor([4, 8, 12, 16, 20], device=DEVICE, dtype=torch.int32)
    positions = seq_lens.long() - compress_ratio

    input_tensor = torch.randn(batch_size, head_dim, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    freqs_cis = _make_freqs_cis()
    kvcache = torch.full(
        (num_pages, page_size * head_dim),
        float("nan"),
        device=DEVICE,
        dtype=DTYPE,
    )

    _run_kernel_decode(
        head_dim=head_dim,
        page_size=page_size,
        input_tensor=input_tensor,
        weight=weight,
        freqs_real=_freqs_real(freqs_cis),
        seq_lens=seq_lens,
        out_loc=out_loc,
        kvcache=kvcache,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    ref = _apply_tail_rope(_rmsnorm_weighted(input_tensor, weight), freqs_cis, positions).to(DTYPE)
    flat_cache = kvcache.view(-1, head_dim)
    triton.testing.assert_close(
        flat_cache[out_loc.long()].float(), ref.float(), atol=2e-2, rtol=2e-2
    )


def _flashmla_fp8_page_bytes(page_size: int) -> int:
    return math.ceil(584 * page_size / 576) * 576


def _flashmla_int8_page_bytes(page_size: int) -> int:
    return math.ceil(608 * page_size / 576) * 576


def test_fused_norm_rope_v2_int8_flashmla_cache() -> None:
    torch.manual_seed(13)
    batch_size = 5
    head_dim = 512
    page_size = 8
    compress_ratio = 4
    out_loc = torch.tensor([0, 1, 7, 9, 17], device=DEVICE, dtype=torch.int64)
    num_pages = int(out_loc.max().item()) // page_size + 1
    seq_lens = torch.tensor([4, 8, 12, 16, 20], device=DEVICE, dtype=torch.int32)
    positions = seq_lens.long() - compress_ratio
    x = torch.randn(batch_size, head_dim, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    freqs_cis = _make_freqs_cis()
    cache = torch.zeros(
        (num_pages, _flashmla_int8_page_bytes(page_size)),
        device=DEVICE,
        dtype=torch.int8,
    )

    _run_kernel_decode(
        head_dim=head_dim,
        page_size=page_size,
        input_tensor=x,
        weight=weight,
        freqs_real=_freqs_real(freqs_cis),
        seq_lens=seq_lens,
        out_loc=out_loc,
        kvcache=cache,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    out = dequantize_k_cache_paged(cache, out_loc, page_size)[:, 0]
    ref = _apply_tail_rope(_rmsnorm_weighted(x, weight), freqs_cis, positions)
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


def test_fused_norm_rope_v2_existing_fp8_cache_interfaces_still_run() -> None:
    torch.manual_seed(12)
    page_size = 8
    compress_ratio = 4
    freqs_cis = _make_freqs_cis()
    freqs_real = _freqs_real(freqs_cis)

    # Indexer old path: uint8 cache layout = 132 bytes per token before page rounding.
    batch_i = 4
    head_i = 128
    out_i = torch.tensor([0, 3, 7, 9], device=DEVICE, dtype=torch.int32)
    pages_i = int(out_i.max().item()) // page_size + 1
    seq_i = torch.tensor([4, 8, 12, 16], device=DEVICE, dtype=torch.int32)
    x_i = torch.randn(batch_i, head_i, device=DEVICE, dtype=DTYPE)
    w_i = torch.randn(head_i, device=DEVICE, dtype=DTYPE)
    cache_i = torch.zeros((pages_i, 132 * page_size), device=DEVICE, dtype=torch.uint8)
    _run_kernel_decode(
        head_dim=head_i,
        page_size=page_size,
        input_tensor=x_i,
        weight=w_i,
        freqs_real=freqs_real,
        seq_lens=seq_i,
        out_loc=out_i,
        kvcache=cache_i,
        compress_ratio=compress_ratio,
    )

    # FlashMLA old path: uint8 cache layout = fp8 nope + bf16 rope + scale bytes.
    batch_f = 4
    head_f = 512
    out_f = torch.tensor([0, 2, 7, 9], device=DEVICE, dtype=torch.int32)
    pages_f = int(out_f.max().item()) // page_size + 1
    seq_f = torch.tensor([4, 8, 12, 16], device=DEVICE, dtype=torch.int32)
    x_f = torch.randn(batch_f, head_f, device=DEVICE, dtype=DTYPE)
    w_f = torch.randn(head_f, device=DEVICE, dtype=DTYPE)
    page_bytes_f = _flashmla_fp8_page_bytes(page_size)
    cache_f = torch.zeros((pages_f, page_bytes_f), device=DEVICE, dtype=torch.uint8)
    _run_kernel_decode(
        head_dim=head_f,
        page_size=page_size,
        input_tensor=x_f,
        weight=w_f,
        freqs_real=freqs_real,
        seq_lens=seq_f,
        out_loc=out_f,
        kvcache=cache_f,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    host_i = cache_i.cpu()
    for loc in out_i.cpu().tolist():
        page = loc // page_size
        offset = loc % page_size
        assert torch.count_nonzero(host_i[page, offset * 128 : offset * 128 + 128]) > 0
        assert torch.count_nonzero(
            host_i[page, 128 * page_size + offset * 4 : 128 * page_size + offset * 4 + 4]
        ) > 0

    host_f = cache_f.cpu()
    for loc in out_f.cpu().tolist():
        page = loc // page_size
        offset = loc % page_size
        value_start = offset * 576
        scale_start = 576 * page_size + offset * 8
        assert torch.count_nonzero(host_f[page, value_start : value_start + 448]) > 0
        assert torch.count_nonzero(host_f[page, scale_start : scale_start + 7]) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

