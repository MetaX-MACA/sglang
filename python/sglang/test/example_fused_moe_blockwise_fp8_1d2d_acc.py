# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from mctlassEx import FusedMoeGEMM


@dataclass
class LayoutCase:
    name: str
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    kwargs: Dict[str, object]


def tensor_info(name: str, x: torch.Tensor) -> str:
    return (
        f"{name}: shape={tuple(x.shape)}, stride={x.stride()}, "
        f"dtype={x.dtype}, contiguous={x.is_contiguous()}"
    )


def build_moe_token_and_expert_ids(
    batch_size: int,
    num_experts: int,
    topk: int,
    kernel_m: int,
    num_valid_tokens: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, List[int]]:
    topk_ids = []
    experts = torch.arange(num_experts)
    for _ in range(batch_size):
        topk_ids.extend(experts[torch.randperm(num_experts)[:topk]].tolist())

    token_ids_per_expert = [[] for _ in range(num_experts)]
    for token_id, expert_id in enumerate(topk_ids):
        token_ids_per_expert[int(expert_id)].append(token_id)

    token_ids_host = []
    expert_ids_host = []
    padded_counts = []
    total_padded = 0

    for expert_id, ids in enumerate(token_ids_per_expert):
        padded_count = int(math.ceil(len(ids) / kernel_m) * kernel_m) if ids else 0
        padded_counts.append(padded_count)
        if padded_count == 0:
            continue
        ids = ids + [num_valid_tokens] * (padded_count - len(ids))
        token_ids_host.extend(ids)
        expert_ids_host.extend([expert_id] * (padded_count // kernel_m))
        total_padded += padded_count

    token_ids = torch.tensor(token_ids_host, dtype=torch.int32, device=device)
    expert_ids = torch.tensor(expert_ids_host, dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor(total_padded, dtype=torch.int32, device=device)
    return token_ids, expert_ids, num_tokens_post_padded, total_padded, padded_counts


def expand_scale_a(scale_a: torch.Tensor, k: int, group_size: int) -> torch.Tensor:
    return scale_a.repeat_interleave(group_size, dim=1)[:, :k]


def expand_scale_b(
    scale_b: torch.Tensor, n: int, k: int, block_n: int, group_size: int
) -> torch.Tensor:
    return scale_b.repeat_interleave(block_n, dim=1).repeat_interleave(
        group_size, dim=2
    )[:, :n, :k]


def cpu_reference(
    num_experts: int,
    n: int,
    k: int,
    topk: int,
    num_valid_tokens: int,
    kernel_m: int,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    topk_weights: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    block_n: int,
    group_size: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    del num_experts
    a_cpu = a.float().cpu()
    b_cpu = b.float().cpu()
    scale_a_cpu = scale_a.float().cpu()
    scale_b_cpu = scale_b.float().cpu()
    topk_weights_cpu = topk_weights.float().cpu()
    token_ids_cpu = token_ids.cpu()
    expert_ids_cpu = expert_ids.cpu()

    a_deq_scales = expand_scale_a(scale_a_cpu, k, group_size)
    b_deq_scales = expand_scale_b(scale_b_cpu, n, k, block_n, group_size)
    out = torch.zeros((num_valid_tokens, n), dtype=torch.float32)

    for block_idx, expert_id_tensor in enumerate(expert_ids_cpu):
        expert_id = int(expert_id_tensor)
        start = block_idx * kernel_m
        rows = token_ids_cpu[start : start + kernel_m].long()
        rows = rows[rows < num_valid_tokens]
        if rows.numel() == 0:
            continue
        input_rows = rows // topk
        lhs = a_cpu[input_rows] * a_deq_scales[input_rows]
        rhs = b_cpu[expert_id] * b_deq_scales[expert_id]
        block_out = lhs @ rhs.t()
        if mul_routed_weight:
            block_out *= topk_weights_cpu[rows, None]
        out[rows] = block_out

    return out.to(torch.bfloat16)


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1e-6)
    metrics = {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "mean_rel": (diff / denom).mean().item(),
        "allclose_1e-2_1e-3": float(
            torch.allclose(actual, expected, rtol=1e-2, atol=1e-3)
        ),
    }
    print(
        f"[{name}] max_abs={metrics['max_abs']:.6g} "
        f"mean_abs={metrics['mean_abs']:.6g} max_rel={metrics['max_rel']:.6g} "
        f"mean_rel={metrics['mean_rel']:.6g} "
        f"allclose={bool(metrics['allclose_1e-2_1e-3'])}"
    )
    return metrics


def make_fp8(shape: Tuple[int, ...], device: str, pattern: str) -> torch.Tensor:
    if pattern == "ones":
        return torch.ones(shape, dtype=torch.float8_e4m3fn, device=device)
    if pattern == "arange":
        x = torch.arange(math.prod(shape), dtype=torch.float32, device=device)
        x = ((x % 17) - 8).reshape(shape) / 8.0
        return x.to(torch.float8_e4m3fn)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    return x.to(torch.float8_e4m3fn)


def run_case(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = args.device
    num_valid_tokens = args.batch_size * args.topk
    block_n = math.ceil(args.n / args.group_size)
    block_k = math.ceil(args.k / args.group_size)

    a = make_fp8((args.batch_size, args.k), device, args.pattern)
    b = make_fp8((args.num_experts, args.n, args.k), device, args.pattern)
    c_for_kernel_m = torch.zeros(
        (num_valid_tokens, args.n), dtype=torch.bfloat16, device=device
    )
    scale_a = (
        torch.rand((args.batch_size, block_k), dtype=torch.float32, device=device)
        * args.scale_range
    )
    scale_b = (
        torch.rand(
            (args.num_experts, block_n, block_k), dtype=torch.float32, device=device
        )
        * args.scale_range
    )
    topk_weights = torch.rand(
        (num_valid_tokens,), dtype=torch.float32, device=device
    )
    if not args.mul_routed_weight:
        topk_weights.fill_(1.0)

    fused_moe = FusedMoeGEMM()
    kernel_kwargs = {
        "use_fp8": True,
        "is_blockwise": True,
        "group_size": args.group_size,
    }
    kernel_m = fused_moe.get_kernel_m(
        a,
        b,
        c_for_kernel_m,
        args.num_experts,
        args.batch_size,
        args.n,
        args.k,
        args.topk,
        **kernel_kwargs,
    )
    print(f"kernel_m={kernel_m}")

    token_ids, expert_ids, num_tokens_post_padded, em, padded_counts = (
        build_moe_token_and_expert_ids(
            args.batch_size,
            args.num_experts,
            args.topk,
            kernel_m,
            num_valid_tokens,
            device,
        )
    )
    print(f"num_valid_tokens={num_valid_tokens}, EM={em}")
    print(tensor_info("A", a))
    print(tensor_info("B", b))
    print(tensor_info("scale_a_canonical", scale_a))
    print(tensor_info("scale_b_canonical", scale_b))
    print(f"non_empty_experts={sum(int(x > 0) for x in padded_counts)}")

    expected = cpu_reference(
        args.num_experts,
        args.n,
        args.k,
        args.topk,
        num_valid_tokens,
        kernel_m,
        token_ids,
        expert_ids,
        a,
        b,
        topk_weights,
        scale_a,
        scale_b,
        args.group_size,
        args.group_size,
        args.mul_routed_weight,
    ).to(device=device)

    base_kwargs = {
        "filter_expert": False,
        "use_fp8": True,
        "is_blockwise": True,
        "group_size": args.group_size,
    }
    cases = [
        LayoutCase(
            "raw_k_major_1d2d",
            scale_a,
            scale_b,
            {
                **base_kwargs,
                "is_scale_a_1d": True,
                "is_scale_b_1d": False,
                "scale_a_layout": "k-major",
                "scale_b_layout": "k-major",
            },
        ),
        LayoutCase(
            "transposed_mn_major_1d2d",
            scale_a.t().contiguous(),
            scale_b.transpose(1, 2).contiguous(),
            {
                **base_kwargs,
                "is_scale_a_1d": True,
                "is_scale_b_1d": False,
                "scale_a_layout": "m-major",
                "scale_b_layout": "n-major",
            },
        ),
        LayoutCase(
            "raw_mn_major_flags",
            scale_a,
            scale_b,
            {
                **base_kwargs,
                "is_scale_a_1d": True,
                "is_scale_b_1d": False,
                "scale_a_layout": "m-major",
                "scale_b_layout": "n-major",
            },
        ),
        LayoutCase(
            "raw_k_major_a_2d_flag",
            scale_a,
            scale_b,
            {
                **base_kwargs,
                "is_scale_a_1d": False,
                "is_scale_b_1d": False,
                "scale_a_layout": "k-major",
                "scale_b_layout": "k-major",
            },
        ),
    ]

    metrics = {}
    for case in cases:
        out = torch.zeros((num_valid_tokens, args.n), dtype=torch.bfloat16, device=device)
        print(tensor_info(f"{case.name}.scale_a", case.scale_a))
        print(tensor_info(f"{case.name}.scale_b", case.scale_b))
        try:
            fused_moe(
                args.batch_size,
                args.n,
                args.k,
                args.num_experts,
                em,
                args.topk,
                a,
                b,
                out,
                case.scale_a,
                case.scale_b,
                None,
                topk_weights,
                token_ids,
                expert_ids,
                num_tokens_post_padded,
                args.mul_routed_weight,
                **case.kwargs,
            )
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception as exc:
            print(f"[{case.name}] failed: {type(exc).__name__}: {exc}")
            continue
        metrics[case.name] = compare(case.name, out, expected)

    if not metrics:
        raise SystemExit("All layout cases failed before producing output.")
    best_name = min(metrics, key=lambda key: metrics[key]["max_abs"])
    print(f"best_case={best_name}")
    if metrics[best_name]["allclose_1e-2_1e-3"] != 1.0:
        raise SystemExit(
            "No layout case matched the CPU reference. This points away from "
            "Python layout flags and toward kernel math/routing/dtype handling."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=257)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-range", type=float, default=1.0)
    parser.add_argument("--pattern", choices=["ones", "random", "arange"], default="ones")
    parser.add_argument("--mul-routed-weight", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("ENABLE_MCTLASS_FUSED_MOE_PYTHON_API", "1")
    run_case(parse_args())
