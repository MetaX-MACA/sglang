# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import torch
import math
from torch.profiler import profile, record_function, ProfilerActivity
from mctlassEx import FusedMoeGEMM

import sys
if len(sys.argv) > 1:
     chip = sys.argv[1]
else:
     chip = None

def cpu_fp8_blockwise_moe_gemm(num_experts, N, K, topk, num_valid_tokens, group_size, mul_routed_weight, gemm_m, token_ids, A, B, C, topk_weights, scale_a, scale_b):
    start_m = 0
    for g in range(num_experts):
        M = gemm_m[g]
        for m in range(start_m, start_m + M):
            row = token_ids[m]
            if row >= num_valid_tokens:
                continue
            rowa = row // topk
            for n in range(N):
                accum = 0.0
                col = g * N + n
                for k in range(K):
                    sa = scale_a[rowa, k // group_size]
                    sb = scale_b[g, n // 128, k // group_size]
                    accum += sa * sb * A[rowa, k] * B[g, n, k]
                if mul_routed_weight:
                    accum *= topk_weights[row]
                C[row, n] = accum
        start_m += M
    return

def build_moe_token_and_expert_ids(batch_size, num_experts, topk, kernel_m, num_valid_tokens, device="cuda"):
    # Step 1: Prepare topk_ids
    topk_ids = []
    num_vec = torch.arange(num_experts)
    for _ in range(batch_size):
        perm = num_vec[torch.randperm(num_experts)]
        topk_ids.extend(perm[:topk])
    topk_ids = torch.tensor(topk_ids, dtype=torch.int32)
    print("topk ids: ", topk_ids.size())
    # Step 2: init token_id before padding
    token_ids_vec = [[] for _ in range(num_experts)]
    gemm_m_host = [0 for _ in range(num_experts)]
    for j, expert_id in enumerate(topk_ids):
        expert_id = int(expert_id)
        token_ids_vec[expert_id].append(j)
        gemm_m_host[expert_id] += 1

    # Step 3: padding
    token_ids_host = []
    expert_ids_host = []
    grid_m = 0

    for i in range(num_experts):
        m = gemm_m_host[i]
        grid_m_i = (m + kernel_m - 1) // kernel_m
        padded_m = grid_m_i * kernel_m
        gemm_m_host[i] = padded_m

        while len(token_ids_vec[i]) < padded_m:
            token_ids_vec[i].append(num_valid_tokens)

        if len(token_ids_vec[i]) > 0:
            token_ids_host.extend(token_ids_vec[i])
            expert_ids_host.extend([i] * grid_m_i)

        grid_m += grid_m_i

    # Step 4: get tensor
    token_ids = torch.tensor(token_ids_host, dtype=torch.int32, device=device)
    expert_ids = torch.tensor(expert_ids_host, dtype=torch.int32, device=device)
    gemm_m_host = torch.tensor(gemm_m_host, dtype=torch.int32)
    EM = token_ids.shape[0]  # num_tokens_post_padded

    return token_ids, expert_ids, EM, grid_m, gemm_m_host

verify = 1
num_experts = 257
batch_size = 6
N = 256
K = 6144
block_n = (N + 127) // 128
block_k = (K + 127) // 128
topk = 9
group_size = 128
num_valid_tokens = batch_size * topk
mul_routed_weight = 0
A = torch.ones((batch_size, K), dtype = torch.float8_e4m3fn, device='cuda:0')
B = torch.ones((num_experts, N, K), dtype = torch.float8_e4m3fn, device='cuda:0')
C = torch.zeros((num_valid_tokens, N), dtype = torch.bfloat16, device='cuda:0')
scale_a = torch.randn((batch_size, block_k), dtype = torch.float32, device='cuda:0')
scale_b = torch.randn((num_experts, block_n, block_k), dtype = torch.float32, device='cuda:0')

topk_weights = torch.randn((num_valid_tokens), dtype = torch.float32, device='cuda:0')
bias = None
if chip == "C600U":
    # scale mn-major
    scale_a_trans = scale_a.T.contiguous()
    scale_b_trans = scale_b.transpose(1, 2).contiguous()

    fused_moe = FusedMoeGEMM()
    # kernel_m = gemm.get_kernel_m(hidden_states, w1, intermediate_cache1, w1.shape[0], hidden_states.shape[0], w1.shape[1], hidden_states.shape[1], topk_ids.shape[1])
    kernel_m = fused_moe.get_kernel_m(A, B, C, num_experts, batch_size, N, K, topk)
    print(f"kernel_m {kernel_m}")
    token_ids, expert_ids, EM, grid_m, gemm_m_host = build_moe_token_and_expert_ids(batch_size, num_experts, topk, kernel_m, num_valid_tokens, device='cuda:0')
    num_tokens_post_padded = torch.tensor(EM, dtype=torch.int32, device='cuda:0')
    print("num valid tokens: ", num_valid_tokens)
    print("num tokens post padded: ", num_tokens_post_padded)
    print("EM: ", EM)
    fused_moe(batch_size, N, K, num_experts, EM, topk, A, B, C, scale_a_trans, scale_b_trans, 
              bias, topk_weights, token_ids, expert_ids, num_tokens_post_padded, 
              mul_routed_weight, is_blockwise=True, group_size=group_size,
              use_fp8 = True,
              is_scale_a_1d = True,
              is_scale_b_1d = False,
              scale_a_layout = "m-major",
              scale_b_layout = "n-major")

if verify:
    print("start verify to CPU:")
    A32 = A.to(dtype=torch.float32, device='cpu')
    B32 = B.to(dtype=torch.float32, device='cpu')
    C32 = torch.zeros((num_valid_tokens, N), dtype = torch.float32, device='cpu')
    sa = scale_a.to(device='cpu')
    sb = scale_b.to(device='cpu')
    topk_weights_cpu = topk_weights.to(device='cpu')
    token_ids_cpu = token_ids.to(device='cpu')
    expert_ids_cpu = expert_ids.to(device='cpu')
    cpu_fp8_blockwise_moe_gemm(num_experts, N, K, topk, num_valid_tokens, group_size, mul_routed_weight, gemm_m_host, token_ids_cpu, A32, B32, C32, topk_weights_cpu, sa, sb)
    ref_C = C32.to(dtype=torch.bfloat16, device='cuda:0')
    print("verify to cpu result: ", torch.allclose(C, ref_C, rtol=1e-2, atol=1e-3))
