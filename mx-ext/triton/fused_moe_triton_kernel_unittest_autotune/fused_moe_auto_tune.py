# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict

import ray
import torch
import triton
import numpy as np
import random
import re
import os
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig
import threading
import triton.language as tl
from functools import wraps
import torch.nn.functional as F

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe,
    get_config_dtype_str,
    get_config_file_name,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.utils import is_hip
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import invoke_fused_moe_kernel
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_align_block_size, select_experts

_is_hip = False

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def extract_cuda_avg_time_normal(data):
    lines = data.strip().split('\n')
    total_time_us = 0.0

    for line in lines:
        if 'fused_moe_kernel' in line or 'fusedMoe' in line:
            # Split the Line by spaces and filter out empty strings
            parts = [part.strip() for part in line.split() if part.strip()]
            # The 'CUDA avg' column is the 8th column (index 9)
            cuda_total_str = parts[9]
            cuda_total_us = convert_time_to_us(cuda_total_str)
            total_time_us += cuda_total_us

    return total_time_us

def extract_cuda_avg_time_awq(data):
    lines = data.strip().split('\n')
    total_time_us = 0.0

    for line in lines:
        if 'fused_moe_kernel_gptq_awq' in line:
            # Split the Line by spaces and filter out empty strings
            parts = [part.strip() for part in line.split() if part.strip()]
            # The 'CUDA avg' column is the 8th column (index 9)
            cuda_total_str = parts[9]
            cuda_total_us = convert_time_to_us(cuda_total_str)
            total_time_us += cuda_total_us

    return total_time_us

def performance_watch_dog(target_func, timeout = 18):
    def wrapper(*args, **kwargs):
        result_container = {}
        exception_container = {}
        
        def worker():
            try:
                result_container['result'] = target_func(*args, **kwargs)
            except triton.runtime.autotuner.OutOfResources as e:
                # Some configurations may be invalid and fail to compile.
                exception_container['exception'] = e
            except Exception as e:
                exception_container['exception'] = e

        thread = threading.Thread(target=worker)
        thread.start()

        thread.join(timeout)

        if thread.is_alive():
            torch.cuda.empty_cache()
            return False
        elif 'exception' in exception_container:
            torch.cuda.empty_cache()
            return False
        else:
            return True
        
    return wrapper

@performance_watch_dog
def run_test1(x, intermediate_cache1, w1, input_gating, topk, use_fp8_w8a8, use_int8_w8a8,
            use_int8_w8a16, use_int4_w4a16, w1_scale, w1_zp,
            block_shape, num_experts, compute_type, config):
    gating_output_softmax = F.softmax(input_gating[1], dim=-1)
    topk_weights, topk_ids = select_experts(x, gating_output_softmax, topk, False, False)
    sorted_token_ids, expert_ids, num_tokens_post_padded = (moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], num_experts))
    invoke_fused_moe_kernel(x,
                        w1,
                        intermediate_cache1,
                        None,
                        w1_scale,
                        w1_zp, 
                        topk_weights,
                        topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        False,
                        topk_ids.shape[1],
                        config,
                        compute_type=compute_type,
                        use_fp8_w8a8=use_fp8_w8a8,
                        use_int8_w8a8=use_int8_w8a8,
                        use_int8_w8a16=use_int8_w8a16,
                        use_int4_w4a16=use_int4_w4a16,
                        block_shape=block_shape,
                        )
    torch.cuda.synchronize()

@performance_watch_dog
def run_test2(x, intermediate_cache2, intermediate_cache3, w2, input_gating, topk, use_fp8_w8a8, use_int8_w8a8,
            use_int8_w8a16, use_int4_w4a16, w2_scale, w2_zp,
            block_shape, num_experts, compute_type, config):
    gating_output_softmax = F.softmax(input_gating[1], dim=-1)
    topk_weights, topk_ids = select_experts(x, gating_output_softmax, topk, False, False)
    sorted_token_ids, expert_ids, num_tokens_post_padded = (moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], num_experts))
    invoke_fused_moe_kernel(
                            intermediate_cache2,
                            w2,
                            intermediate_cache3,
                            None,
                            w2_scale,
                            w2_zp, 
                            topk_weights,
                            topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            True,
                            1,
                            config,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            block_shape=block_shape,
                            )
    torch.cuda.synchronize()

def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
    stage: str = "stage1"
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    elif use_int4_w4a16:
        from vllm.scalar_type import scalar_types
        pack_factor = 2
        w1 = torch.randint(size=(num_experts, shard_intermediate_size, hidden_size // pack_factor),  low=scalar_types.uint4.min(), 
                            high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")
        w2 = torch.randint(size=(num_experts, hidden_size, shard_intermediate_size // 2 // pack_factor),  low=scalar_types.uint4.min(), 
                            high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")
        
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    w1_zp = None
    w2_zp = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )
    if use_int4_w4a16:
        group_size = 64
        group_size_div_factor = 1
        pack_factor = 2
        intermediate_size_per_partition = shard_intermediate_size // 2
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        w1_scale = torch.rand((num_experts, shard_intermediate_size, hidden_size // group_size), dtype=x.dtype, device="cuda") / 100
        w2_scale = torch.rand((num_experts, hidden_size, shard_intermediate_size // 2 // group_size), dtype=x.dtype, device="cuda") / 100

        w1_zp = torch.randint(size=(num_experts, shard_intermediate_size // pack_factor, hidden_size // group_size),  low=scalar_types.uint4.min(), 
                            high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")
        w2_zp = torch.randint(size=(num_experts, hidden_size // pack_factor, shard_intermediate_size // 2 // group_size),  low=scalar_types.uint4.min(), 
                            high=scalar_types.uint4.max(), dtype=torch.uint8, device="cuda")
        block_shape = [0, group_size]

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_iters, num_tokens, num_experts, dtype=x.dtype)
    compute_type = (tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16)
    intermediate_cache1 = None
    intermediate_cache2 = None
    intermediate_cache3 = None
    if stage == "stage1":
        intermediate_cache1 = torch.randn((num_tokens, topk, shard_intermediate_size),device="cuda", dtype=dtype)
    else:
        intermediate_cache2 = torch.randn((num_tokens * topk, shard_intermediate_size//2), device="cuda", dtype=dtype)
        intermediate_cache3 = torch.randn((num_tokens, topk, w2.shape[1]), device="cuda", dtype=dtype)

    def run(idx: int, stage: str):
        gating_output_softmax = F.softmax(input_gating[idx], dim=-1)
        topk_weights, topk_ids = select_experts(x, gating_output_softmax, topk, False, False)
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], num_experts))
        if stage == "stage1":
            # print(f"==> Dive into stage1")
            invoke_fused_moe_kernel(x,
                                w1,
                                intermediate_cache1,
                                None,
                                w1_scale,
                                w1_zp, 
                                topk_weights,
                                topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                topk_ids.shape[1],
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                block_shape=block_shape,
                                )
        else:
            print(f"==> Dive into stage2")
            invoke_fused_moe_kernel(
                            intermediate_cache2,
                            w2,
                            intermediate_cache3,
                            None,
                            w2_scale,
                            w2_zp, 
                            topk_weights,
                            topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            True,
                            1,
                            config,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            block_shape=block_shape,
                            )
        torch.cuda.synchronize()

    # JIT compilation & warmup
    test_result = True
    if stage == "stage1":
        test_result = run_test1(x, intermediate_cache1, w1, input_gating, topk, 
                use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                w1_scale, w1_zp,
                block_shape, num_experts, compute_type, config)
    else:
        test_result = run_test2(x, intermediate_cache2, intermediate_cache3, w2, input_gating, topk,
                                use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                                w2_scale, w2_zp, 
                                block_shape, num_experts, compute_type, config)
    if test_result is False:
        torch.cuda.synchronize()
        del x, w1, w2, input_gating
        if intermediate_cache1 is not None:
            del intermediate_cache1
        if (intermediate_cache2 is not None) and (intermediate_cache3 is not None):
            del intermediate_cache2, intermediate_cache3
        if (w1_scale is not None) and (w2_scale is not None):
            del w1_scale, w2_scale
        if (w1_zp is not None) and (w2_zp is not None):
            del w1_zp, w2_zp
        torch.cuda.empty_cache()
        print(f"===> Time out: tokens->{num_tokens}, config->{config}")
        return float("inf")

    # Warmup
    for _ in range(1):
        run(0, stage)

    # Profiler
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i in range(num_iters):
            run(i, stage)
            prof.step()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    prof_avg_time = 0.0
    if use_int4_w4a16 or use_int8_w8a16:
        prof_avg_time = extract_cuda_avg_time_awq(table)
    else:
        prof_avg_time = extract_cuda_avg_time_normal(table)
    return prof_avg_time


def get_rocm_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    waves_per_eu_range = 0
    for num_stages in [2]:
        for block_m in [32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [16, 32, 64, 128, 256]:
                    for num_warps in [1, 2, 4, 8]:
                        for group_size in [1, 4, 8, 16, 32]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu_range,
                                }
                            )
    return configs


def get_configs_compute_bound() -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs: List[BenchmarkConfig] = []
    if _is_hip:
        configs = get_rocm_configs_compute_bound()
    else:
        for num_stages in [3, 4]:
            for block_m in [16, 32, 64, 128, 256]:
                for block_k in [32, 64, 128, 256]:
                    for block_n in [32, 64, 128, 256]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                for pipeline in ["basic", "cpasync"]:
                                    for scenario in ["", "unroll"]:
                                        if pipeline == "basic" and scenario == "unroll":
                                            continue
                                        if block_m == block_n and block_m >= 128 and pipeline == "cpasync":
                                            if (block_m + block_n) * block_k > 65536:
                                                continue
                                        elif pipeline == "cpasync":
                                            if (block_m + block_n) * block_k * num_stages > 65536:
                                                continue
                                        configs.append(
                                            {
                                                "BLOCK_SIZE_M": block_m,
                                                "BLOCK_SIZE_N": block_n,
                                                "BLOCK_SIZE_K": block_k,
                                                "GROUP_SIZE_M": group_size,
                                                "num_warps": num_warps,
                                                "num_stages": num_stages,
                                                "pipeline":pipeline,
                                                "scenario": scenario,
                                            }
                                        )
    return configs

sce1 = ["", "roll", "unroll"]
sce2 = ["", "unprefetch"]
sce3 = ["", "fullstage"]
sce4 = ["", "storeCoalesce"]
def get_configs_compute_bound_w4a16() -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs: List[BenchmarkConfig] = []
    if _is_hip:
        configs = get_rocm_configs_compute_bound()
    else:
        for num_stages in [1, 2, 3, 4]:
            for block_m in [16, 32, 64]:
                for block_k in [32, 64, 128, 256]:
                    for block_n in [32, 64, 128, 256]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                for pipeline in ["basic", "cpasync"]:
                                    for sce1_ in sce1:
                                        for sce2_ in sce2:
                                            for sce3_ in sce3:
                                                for sce4_ in sce4:
                                                    # if (block_k == 256):
                                                    #     continue

                                                    if pipeline == "basic" and sce1_ == "unroll":
                                                        continue
                                                    # if sce2_ == "unprefetch" and (pipeline != "basic" or sce3_ != "fullstage"):
                                                    #     continue
                                                    # if pipeline == "cpasync" and sce1_ == "unroll":
                                                    #     continue
                                                    # if pipeline == "cpasync" and sce2_ == "unprefetch":
                                                    #     continue
                                                    # if pipeline != "cpasync" and sce4_ == "storeCoalesce":
                                                    #     continue
                                                    # if num_stages >= 2 and sce4_ == "storeCoalesce":
                                                    #     continue

                                                    scenario = ""
                                                    if len(sce1_) != 0:
                                                        scenario = sce1_
                                                    if len(scenario) != 0 and len(sce2_) != 0:
                                                        scenario = scenario + ";" + sce2_
                                                    if len(scenario) != 0 and len(sce3_) != 0:
                                                        scenario = scenario + ";" + sce3_
                                                    if len(scenario) != 0 and len(sce4_) != 0:
                                                        scenario = scenario + ";" + sce4_

                                                    if pipeline == "cpasync" and scenario == "roll;fullstage":
                                                        continue
                                                    if pipeline == "cpasync" and scenario == "unroll;fullstage":
                                                        continue

                                                    if block_m == block_n and block_m >= 128 and pipeline == "cpasync":
                                                        if (block_m + block_n) * block_k > 65536:
                                                            continue
                                                    elif pipeline == "cpasync" and (num_stages == 3 or num_stages == 2 or num_stages == 1):
                                                        if (block_m + block_n) * block_k * num_stages > 65536:
                                                            continue
                                                    if (block_n ==256 and block_k ==256):
                                                        continue
                                                    if (block_m ==16 and block_k ==32) and num_warps == 8 and pipeline == "cpasync":
                                                        continue
                                                    if (block_m ==256 and block_k ==256 and (block_n ==128 or block_n ==256)):
                                                        continue
                                                    if (block_m ==256 and block_k ==128 and block_n ==256):
                                                        continue
                                                    if (block_m == 128 and block_n == 128 and block_k == 256) and num_stages != 4:
                                                        continue
                                                    # if (block_m >= 64 and block_n >= 64 and block_k == 128) and pipeline == "cpasync":
                                                    #     continue

                                                    configs.append(
                                                        {
                                                            "BLOCK_SIZE_M": block_m,
                                                            "BLOCK_SIZE_N": block_n,
                                                            "BLOCK_SIZE_K": block_k,
                                                            "GROUP_SIZE_M": group_size,
                                                            "num_warps": num_warps,
                                                            "num_stages": num_stages,
                                                            "pipeline": pipeline,
                                                            "scenario": scenario,
                                                            "SPLIT_K": 1,
                                                        }
                                                    )

    return configs

class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed

    @staticmethod
    def benchmark(
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        block_shape: List[int],
    ) -> Tuple[Dict[str, int], float]:
        dtype_str = get_config_dtype_str(
            dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        op_config = get_moe_configs(
            num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                False,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        kernel_time = benchmark_config(
            config,
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            block_shape,
        )
        return config, kernel_time

    @staticmethod
    def tune(
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
        best_config,
        best_time,
    ) -> Dict[str, int]:
        for config in tqdm(search_space):
            if num_tokens <= 32 and config["BLOCK_SIZE_M"] >= 64:
                continue
            if num_tokens <= 64 and config["BLOCK_SIZE_M"] >= 128:
                continue
            if num_tokens < 256 and config["BLOCK_SIZE_M"] == 256:
                continue
            try:
                kernel_time = benchmark_config(
                    config,
                    num_tokens,
                    num_experts,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    block_shape,
                    num_iters=10,
                )
            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid and fail to compile.
                continue
            except RuntimeError:
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}, best_time={best_time}, best_config={best_config}")
        assert best_config is not None
        return best_config

def tune(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: List[int],
    search_space: List[Dict[str, int]],
    stage: str,
    best_config,
    best_time,
):
    save_dir = f"bf16_{num_experts}"
    if use_int4_w4a16:
        save_dir = f"w4a16_{num_experts}"
    elif use_int8_w8a8:
        save_dir = f"w8a8_int8_{num_experts}"
    
    os.makedirs(save_dir, exist_ok=True)
    save_csv_path = f"{save_dir}/{stage}-{num_tokens}.txt"
    with open(save_csv_path, "w", buffering=1) as f:
        for config in tqdm(search_space):
            if num_tokens <= 32 and config["BLOCK_SIZE_M"] >= 64:
                continue
            if num_tokens <= 64 and config["BLOCK_SIZE_M"] >= 128:
                continue
            if num_tokens < 256 and config["BLOCK_SIZE_M"] == 256:
                continue
            if num_tokens >= 1024 and config["BLOCK_SIZE_M"] == 16:
                continue
            
            kernel_time = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                block_shape,
                num_iters=10,
                stage=stage,
            )

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config
                item = f"{config}, {kernel_time}"
                f.write(item + "\n")

    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}, best_time={best_time}, best_config={best_config}")

def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"pipeline": config["pipeline"]} if "pipeline" in config else {}
        ),
        **(
            {"scenario": config["scenario"]} if "scenario" in config else {}
        ),
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: List[int] = None,
) -> None:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int4_w4a16=use_int4_w4a16
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        block_shape,
    )

    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"]:
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        n_share_fusion_experts = args.n_share_experts_fusion
        E = (
            config.n_routed_experts + n_share_fusion_experts
            if config.architectures[0] in ["DeepseekV3ForCausalLM"]
            else config.n_routed_experts
        )
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1AForCausalLM",
    ]:
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size

    hidden_size = config.hidden_size
    dtype = config.torch_dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_w4a16 = args.dtype == "int4_w4a16"
    stage = args.stage
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            7,
            8,
            16,
            24,
            28,
            32,
            48,
            56,
            64,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            6140, 
            7170, 
            8192
        ]
    else:
        batch_sizes = [args.batch_size]

    torch.set_default_device("cuda")
    if args.tune:
        search_space: List[Dict[str, int]] = []
        if use_int4_w4a16 or use_int8_w8a16 or use_int8_w8a8:
            search_space = get_configs_compute_bound_w4a16()
        else:
            search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_n, block_k = block_shape[0], block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]
        print(f"Batch sizes: {batch_sizes}")
        print(f"Start tuning {args.dtype} over {len(search_space)} configurations...")

        start = time.time()
        configs = [None for _ in range(len(batch_sizes))]
        best_times = [float("inf") for _ in range(len(batch_sizes))]
        for idx in range(len(batch_sizes)):
            tune(batch_sizes[idx],
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                block_shape,
                search_space,
                stage,
                configs[idx],
                best_times[idx])
            time.sleep(1)

        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = []
        for batch_size in batch_sizes:
            output = BenchmarkWorker.benchmark(
                                        batch_size,
                                        E,
                                        shard_intermediate_size,
                                        hidden_size,
                                        topk,
                                        dtype,
                                        use_fp8_w8a8,
                                        use_int8_w8a8,
                                        use_int8_w8a16,
                                        block_shape,
                                    )
            outputs.append(output)
        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "-tp", type=int, default=2)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument(
        "--n-share-experts-fusion",
        type=int,
        default=0,
        help="The number of shared_experts need to be replica to fuse with normal experts in deepseek v3/r1",
    )
    args = parser.parse_args()

    main(args)