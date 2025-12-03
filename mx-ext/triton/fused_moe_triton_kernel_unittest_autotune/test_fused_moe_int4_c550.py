import time
import torch

from vllm.model_executor.layers.fused_moe import fused_experts as vllm_fused_experts
from vllm.model_executor.layers.fused_moe import fused_moe as vllm_fused_moe
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as sglang_fused_moe
from sglang.srt.layers.moe.topk import select_experts as sglang_select_experts
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List
from vllm.scalar_type import scalar_types
import csv
import re

def convert_time_to_us(time_str):
    pattern = r'^(\d+\.?\d*)\s*([smu]?s)$'
    match = re.match(pattern, time_str)#使用match函数代替search
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
        # if 'fused_moe_kernel' in line or 'fusedMoe' in line:
        if 'MacaGemm' in line:
            # Split the Line by spaces and filter out empty strings
            print(line)
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
    group_size = 64
    pack_factor = 2
    group_size_div_factor = 1
    intermediate_size_per_partition = N
    while intermediate_size_per_partition % group_size or K % group_size:
        group_size = group_size // 2
        group_size_div_factor *= 2
        assert group_size >= 32

    hidden_states = create_random_cuda_tensor((M, K), data_type)
    score = create_random_cuda_tensor((M, Experts), data_type)

    w1 = torch.randint(size=(Experts, 2 * N, K // pack_factor),  low=scalar_types.uint4.min(), 
                            high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(size=(Experts, K, N // pack_factor),  low=scalar_types.uint4.min(), 
                            high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")

    w1_scale = torch.rand((Experts, 2 * N, K // group_size), dtype=hidden_states.dtype, device="cuda") / 100
    w2_scale = torch.rand((Experts, K, N // group_size), dtype=hidden_states.dtype, device="cuda") / 100

    w1_zp = torch.randint(size=(Experts, 2 * N // pack_factor, K // group_size),  low=scalar_types.uint4.min(), 
                        high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")
    w2_zp = torch.randint(size=(Experts, K // pack_factor, N // group_size),  low=scalar_types.uint4.min(), 
                        high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")
    block_shape = [0, group_size]

    if use_vllm:
        vllm_fused_moe(hidden_states,
            w1,
            w2,
            score,
            topk,
            renormalize=False,
            use_int4_w4a16=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            block_shape=block_shape
        )
        # assert score.shape[1] == w1.shape[0], "Number of experts mismatch"

        # topk_weights, topk_ids = sglang_select_experts(
        #     hidden_states = hidden_states,
        #     router_logits = score,
        #     use_grouped_topk = False,
        #     top_k = topk,
        #     renormalize = False,
        #     topk_group = None,
        #     num_expert_group = None,
        #     custom_routing_function = None,
        # )
        # vllm_output = vllm_fused_experts(
        #     hidden_states = hidden_states,
        #     w1 = w1,
        #     w2 = w2,
        #     topk_weights = topk_weights,
        #     topk_ids = topk_ids,
        #     inplace = True,
        #     use_int4_w4a16 = True,
        #     w1_scale = w1_scale,
        #     w2_scale = w2_scale,
        #     w1_zp = w1_zp,
        #     w2_zp = w2_zp,
        #     block_shape = block_shape
        # )
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
            use_int4_w4a16 = True,
            w1_scale = w1_scale,
            w2_scale = w2_scale,
            w1_zp = w1_zp,
            w2_zp = w2_zp,
            a1_scale = None,
            a2_scale = None,
            block_shape = block_shape
        )

    torch.cuda.synchronize()

def main():
    # m_values: List[int] = [1, 2, 4, 7, 8, 16, 24, 28, 32, 48, 56, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6140, 7170, 8192]
    m_values: List[int] = [1]
    n_values: int = 256 # 2048 / 8 tp8
    k_values: int = 7168
    experts:  int = 256
    topk: int = 8
    repeat_time: int = 100

    print(f"Warm-up start")
    for idx in range(len(m_values)):
        test_fused_moe_once(M = m_values[idx], N = n_values, K = k_values, Experts = experts, topk = topk, use_vllm = False)
    print(f"Warm-up done")

    # cvs_filepath = 'sgl045_fused_moe_time.csv'
    cvs_filepath = 'sgl045_fused_moe_w4a16_time.csv'
    with open(cvs_filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        for idx in range(len(m_values)):
            with profile(activities = [ProfilerActivity.CUDA],) as prof:
                for _ in range(repeat_time):
                    test_fused_moe_once(M = m_values[idx], N = n_values, K = k_values, Experts = experts, topk = topk, use_vllm = False)
                    prof.step()
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit = 100)
            prof.export_chrome_trace("fused_moe_w4a16_profile.json")
            elapsed_time_torch = extract_cuda_total_time(table)
            print(f"======================== Profile sglang ========================")
            print(f"M->{m_values[idx]}, N->{n_values}, K->{k_values}, mctlass_fused_moe_kernel_w4a16 elapsed time->{elapsed_time_torch} us")
            os.environ[""COLUMNS] = "2000"
            print(table)
            row = [m_values[idx], n_values, k_values, elapsed_time_torch]
            if f.tell() == 0:
                writer.writerow(['M', 'N', 'K', 'elapsed_time / us'])
            writer.writerow(row)

if __name__ == '__main__':
    main()