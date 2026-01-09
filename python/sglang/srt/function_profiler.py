import os
import time
import logging
import json
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.io_struct import ProfileReq
from sglang.srt.distributed import get_tensor_model_parallel_rank, get_pp_group

logger = logging.getLogger(__name__)

import triton
import triton.language as tl

@triton.jit
def attn_start_marker():
    pass

@triton.jit
def attn_end_marker():
    pass   

@triton.jit
def moe_start_marker():
    pass

@triton.jit
def moe_end_marker():
    pass    

class FunctionProfiler:
    """
    Store MX global profiling settings.
    """

    def __init__(self):
        self.profile_funcs = None
        self.profiler_activities = [ProfilerActivity.CUDA]        
        self.record_shapes = False
        self.profile_memory = False
        self.with_stack = True
        self.trace_dir = None       
        self.profile_steps = []
        self.step_counters = {}
        self.func_stats = []
        self.tp_ranks = []
        self.profile_id = None

    def init_profiler_config(self, tp_rank, profiler_config):
        self.tp_ranks = profiler_config.get("tp_ranks")
        self.profile_funcs = profiler_config.get("profile_funcs")
        self.activities = profiler_config.get("activities")       
        self.record_shapes = profiler_config.get("record_shapes")
        self.profile_memory = profiler_config.get("profile_memory")
        self.with_stack = profiler_config.get("with_stack")    
        self.profile_steps = profiler_config.get("profile_steps")
        self.profile_id = profiler_config.get("profile_id")
        self.export_trace = profiler_config.get("export_trace", True)

        self.profiler_activities = [ProfilerActivity.CUDA]
        if(self.activities is not None):
            activity_map = {
                "CPU": ProfilerActivity.CPU,
                "GPU": ProfilerActivity.CUDA,
            }
            self.profiler_activities = [
                activity_map[a] for a in self.activities if a in activity_map
            ] 
        self.trace_dir = profiler_config.get("output_dir")
        if self.trace_dir is None:
            self.trace_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp") 
        if(self.profile_id is None):
            self.profile_id = ""
    
        current_time = time.time()
        timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime(current_time)) 
        self.profile_id += f'-{timestamp}'
        self.trace_dir = os.path.join(self.trace_dir, self.profile_id)
        os.makedirs(self.trace_dir, exist_ok=True)

        logger.info(f"MX profiler starts: trace_dir={self.trace_dir},\n \
            tp_ranks={self.tp_ranks}, activities={self.activities}, \
            record_shapes={self.record_shapes}, profile_memory={self.profile_memory}, \
            with_stack={self.with_stack}, profile_funcs={self.profile_funcs}, \
            profile_steps={self.profile_steps}, export_trace={self.export_trace}") 

    def start_profile(self, profiler_config):
        tp_rank = get_tensor_model_parallel_rank()
        self.init_profiler_config(tp_rank, profiler_config) 

    def stop_profile(self):
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pp_group().rank_in_group
        output_file = os.path.join(self.trace_dir, f"PP{pp_rank}-TP{tp_rank}-func_stats.csv")
        logger.info(f"Outputting func stats: {output_file}")

        # output function stats
        df = pd.DataFrame(columns=[
            'profile_id','pp_rank', 'tp_rank', 'func', 'step', 'latency(ms)', 'batch_size', 'seq_lens_sum', 'bid' 
        ]) 
        # Write in chunks
        chunk_size = 1000
        for i in range(0, len(self.func_stats), chunk_size):
            end = min(i + chunk_size, len(self.func_stats))
            # Create a DataFrame for the current chunk
            chunk_data = []
            for j in range(i, end):
                stat = self.func_stats[j]
                chunk_data.append([
                    self.profile_id,
                    pp_rank,
                    tp_rank,
                    stat[0],
                    stat[1],
                    stat[2] * 1000,  # convert second to millisecond
                    stat[3],
                    stat[4],
                    stat[5],
                ])
            
            # Create DataFrame with proper column names
            chunk = pd.DataFrame(chunk_data, columns=[
                'profile_id','pp_rank', 'tp_rank', 'func', 'step', 'latency(ms)', 'batch_size', 'seq_lens_sum', 'bid' 
            ])
            
            mode = 'w' if i == 0 else 'a'  # Write header only for first chunk
            header = i == 0
            chunk.to_csv(output_file, mode=mode, header=header, index=False)                 

        # reset configs
        self.profile_funcs = None
        self.tp_ranks = []
        self.profile_steps = []            
        self.step_counters = {} 
        self.func_stats = []

        logger.info("Function profiler stops")

        return
        
    def should_profile(self, func_name):
        return self.profile_funcs is not None \
                and func_name in self.profile_funcs

    def export_profiler_to_csv(self, profiler, output_file, record_shapes, batch_info=None):    
        df = pd.DataFrame(columns=[
            'func', 
            'self_cpu(ms)', 
            'self_cpu_avg(ms)',
            'cpu_total(ms)', 
            'cpu_total_avg(ms)',
            'self_gpu(ms)', 
            'self_gpu_avg(ms)', 
            'gpu_total(ms)',
            'gpu_total_avg(ms)',
            'cpu_mem', 
            'self_cpu_mem', 
            'num_calls', 
            'input_shapes',
            'batch_size',
            'seq_lens_sum',
            'bid'
        ])

        for event in profiler.key_averages(record_shapes):
            df.loc[len(df)] = [
                event.key,
                f"{event.self_cpu_time_total / 1000:.4f}",
                f"{event.self_cpu_time_total / event.count / 1000:.4f}",
                f"{event.cpu_time_total / 1000:.4f}",
                f"{event.cpu_time_total / event.count / 1000:.4f}",
                f"{event.self_device_time_total / 1000:.4f}",
                f"{event.self_device_time_total / event.count / 1000:.4f}",                
                f"{event.device_time_total / 1000:.4f}",
                f"{event.device_time_total / event.count / 1000:.4f}",
                f"{event.cpu_memory_usage / 1024 / 1024:.4f}",
                f"{event.self_cpu_memory_usage / 1024 / 1024:.4f}",
                event.count,
                event.input_shapes,
                batch_info[0] if batch_info is not None else None,
                batch_info[1] if batch_info is not None else None,
                batch_info[2] if batch_info is not None else None
            ]
        
        df.to_csv(output_file, index=False)

    def run(self, name, func, *args):
        if(not self.should_profile(name)):
            return func(*args)

        tp_rank = get_tensor_model_parallel_rank()
        if(tp_rank not in self.tp_ranks):
            return func(*args)

        if name in self.step_counters:
            self.step_counters[name] += 1
        else:
            self.step_counters[name] = 1

        batch_info = [None, None, None]
        for i, arg in enumerate(args):
            if isinstance(arg, ForwardBatch):
                batch_info[0] = arg.batch_size
                batch_info[1] = arg.seq_lens_sum
                break
            if isinstance(arg, ScheduleBatch):
                batch_info[0] = arg.batch_size()
                batch_info[1] = arg.seq_lens_sum
                break    
            if isinstance(arg, ModelWorkerBatch):
                batch_info[0] = len(arg.seq_lens)
                batch_info[1] = arg.seq_lens_sum
                batch_info[2] = arg.bid
                break                              

        if(self.step_counters[name] not in self.profile_steps):
            logger.debug(f"Skip profiling function={name}, step={self.step_counters[name]}, profile_steps={self.profile_steps}")
            start = time.monotonic()
            result = func(*args)
            latency = time.monotonic() - start
            self.func_stats.append((name, self.step_counters[name], latency, batch_info[0], batch_info[1], batch_info[2]))
            return result
 
        logger.info(f"Profiling function={name}, activities={self.activities}, step={self.step_counters[name]}") 

        pp_rank = get_pp_group().rank_in_group

        #cuda profiling
        if(self.activities is not None):
            if("MEM" in self.activities):
                torch.cuda.memory._record_memory_history(max_entries=100000)
                start = time.monotonic()
                result = func(*args)
                latency = time.monotonic() - start
                self.func_stats.append((name, self.step_counters[name], latency, batch_info[0], batch_info[1], batch_info[2]))

                memory_profile_path = os.path.join(
                    self.trace_dir,
                    f"{name}-PP{pp_rank}-TP{tp_rank}-memory" + str(time.time()) + ".pickle",
                )
                logger.info(f"Outputting memory snapshot {memory_profile_path}")
                torch.cuda.memory._dump_snapshot(memory_profile_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                return result
            elif("CUDA_PROFILER" in self.activities):
                torch.cuda.cudart().cudaProfilerStart()
                start = time.monotonic()
                result = func(*args)
                latency = time.monotonic() - start
                self.func_stats.append((name, self.step_counters[name], latency, batch_info[0], batch_info[1], batch_info[2]))
                torch.cuda.cudart().cudaProfilerStop()
                return result

        def trace_handler(p):
            file_path = f"{self.trace_dir}/PP{pp_rank}-TP{tp_rank}-{name}-S{self.step_counters[name]}"
            trace_csv = f"{file_path}.csv"
            logger.info(f"Outputting kernel stats: {trace_csv}")
            #Outputting csv file
            self.export_profiler_to_csv(p, trace_csv, self.record_shapes, batch_info)
            if self.export_trace:
                trace_file = f"{file_path}.trace.json.gz"
                logger.info(f"Outputting trace: {trace_file}")
                #Outputting trace file
                p.export_chrome_trace(trace_file)

        with profile(
            activities=self.profiler_activities,
            on_trace_ready=trace_handler,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        ) as prof:
            with record_function(name):
                start = time.monotonic()
                result = func(*args)
                latency = time.monotonic() - start
                self.func_stats.append((name, self.step_counters[name], latency, batch_info[0], batch_info[1], batch_info[2]))

        return result  

profiler = FunctionProfiler()