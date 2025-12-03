import time
import torch

from vllm.model_executor.layers.fused_moe import fused_experts as vllm_fused_experts
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as sglang_fused_moe
from sglang.srt.layers.moe.topk import select_experts as sglang_select_experts
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List
import csv
import re

def convert_time_to_us(time_str):
    pattern = r'^(\d+\.?\d*)\s*([smu]?s)$'
    match = re.match(pattern, time_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
    if unit == 's':
        return value * 1000000
    elif unit == 'ms':
        return value * 1000
    elif unit == 'us':
        return value
    return None

def extract_cuda_total_time(data):
    lines = data.strip().split('\n')
    total_time_us = 0.0

    for line in lines:
        if 'fused_moe_kernel' in line or 'fusedMoe' in line:
            # Split the Line by spaces and filter out empty strings
            parts = [part.strip() for part in line.split() if part.strip()]
            # The 'CUDA total' column is the 8th column
            # The 'CUDA time avg' column is the 9th column
            cuda_total_str = parts[9]
            cuda_total_us = convert_time_to_us(cuda_total_str)
            total_time_us += cuda_total_us

    return total_time_us

def create_random_cuda_tensor(shape, dtype, mean=0.25, std=0.12):
    """Create a random CUDA tensor

    Args:
        shape: Tensor shape
        dtype: Data type
        mean: Mean value
        std: Standard deviation

    Returns:
        torch.Tensor: Randomly initialized CUDA tensor
    """
    return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

def test_fused_moe_once(M: int, N: int, K: int, Experts: int, topk: int, use_vllm: bool = False):
    data_type = torch.bfloat16

    hidden_states = create_random_cuda_tensor((M, K), data_type)
    w1 = create_random_cuda_tensor((Experts, 2 * N, K), data_type)
    w2 = create_random_cuda_tensor((Experts, K, N), data_type)
    score = create_random_cuda_tensor((M, Experts), data_type)

    if use_vllm:
        assert score.shape[1] == w1.shape[0], "Number of experts mismatch"

        topk_weights, topk_ids = sglang_select_experts(
            hidden_states = hidden_states,
            router_logits = score,
            use_grouped_topk = False,
            top_k = topk,
            renormalize = False,
            topk_group = None,
            num_expert_group = None,
            custom_routing_function = None,
        )
        vllm_output = vllm_fused_experts(
            hidden_states = hidden_states,
            w1 = w1,
            w2 = w2,
            topk_weights = topk_weights,
            topk_ids = topk_ids,
        )
    else:
        sglang_output = sglang_fused_moe(
            hidden_states = hidden_states,
            w1 = w1,
            w2 = w2,
            gating_output = score,
            topk = topk,
            renormalize = False,
            use_fp8_w8a8 = False,
            use_int8_w8a16 = False,
            w1_scale = None,
            w2_scale = None,
            a1_scale = None,
            a2_scale = None,
        )

    torch.cuda.synchronize()

def main():
    m_values: List[int] = [1, 2, 4, 7, 8, 16, 24, 28, 32, 48, 56, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6140, 7170, 8192]
    n_values: int = 64
    k_values: int = 7168
    experts:  int = 256
    topk: int = 8
    repeat_time: int = 100

    print(f"Warm-up start")
    for idx in range(len(m_values)):
        test_fused_moe_once(M = m_values[idx], N = n_values, K = k_values, Experts = experts, topk = topk)
    print(f"Warm-up done")

    

    # for idx in range(len(m_values)):
    #     with profile(activities = [ProfilerActivity.CUDA],
    #              with_stack = True,
    #              profile_memory = True) as prof:
    #         for _ in range(repeat_time):
    #             test_fused_moe_once(M = m_values[idx], N = n_values, K = k_values, Experts = experts, topk = topk)
    #             prof.step()
    #     print(f"======================== Profile sglang ========================")
    #     print(f"M->{m_values[idx]}, N->{n_values}, K->{k_values}")
    #     print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 16))

    cvs_filepath = 'sgl045_fused_moe_time.csv'
    with open(cvs_filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        for idx in range(len(m_values)):
            with profile(activities = [ProfilerActivity.CUDA],) as prof:
                for _ in range(repeat_time):
                    test_fused_moe_once(M = m_values[idx], N = n_values, K = k_values, Experts = experts, topk = topk, use_vllm = False)
                    prof.step()
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit = 16)
            elapsed_time_torch = extract_cuda_total_time(table)
            print(f"======================== Profile sglang ========================")
            print(f"M->{m_values[idx]}, N->{n_values}, K->{k_values}, fused_moe_kernel elapsed time->{elapsed_time_torch} us")
            print(table)
            row = [m_values[idx], n_values, k_values, elapsed_time_torch]
            if f.tell() == 0:
                writer.writerow(['M', 'N', 'K', 'elapsed_time / us'])
            writer.writerow(row)

if __name__ == '__main__':
    main()