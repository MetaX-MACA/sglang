import os
import csv
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

default_columns = ['tp_rank', 'time', 'forwardmode', 'allocated', 'reserved', 'reservedadd','free', 'freelost', 'mem_summary', 'total']
default_columns = ['tp_rank', 'layerid', 'old', 'new' ]

default_file_path="/workspace/sglang/c.csv"
class AsyncCSVLogger:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, file_path=default_file_path, columns=default_columns):
        with cls._lock:
            if file_path not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[file_path] = instance
                instance._init_instance(file_path, columns)
            return cls._instances[file_path]

    def _init_instance(self, file_path, columns):
        self.file_path = file_path
        self.columns = columns
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._file_lock = threading.Lock()

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log(self, **kwargs):
        def write_row():
            try:
                with self._file_lock:
                    with open(self.file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row = [kwargs.get(col, "") for col in self.columns]
                        writer.writerow(row)
            except Exception as e:
                print(f"[AsyncCSVLogger] 写入失败: {e}")

        self.executor.submit(write_row)
        
# Case1 记录Torch显存池变化情况
# from datetime import datetime
# from sglang.srt.utils.csv_logger import AsyncCSVLogger as csv_logger

# # 构造文件路径
# tp_rank = get_tensor_model_parallel_rank()
# file_name = f"TP{tp_rank}-mem.csv"
# file_path = f"/mnt/shared_data/ljy/container/sglang048install051/{file_name}"
# columns = ['tp_rank', 'time', 'allocated', 'reserved', 'reservedadd','free', 'freelost', 'total']
# csv = csv_logger(file_path, columns)

# allocated = torch.cuda.memory_allocated() / (1 << 30)
# reserved = torch.cuda.memory_reserved() / (1 << 30)
# max_alloc = torch.cuda.max_memory_allocated() / (1 << 30)

# # free, total = torch.cuda.cudart().cudaMemGetInfo()
# free,  total = torch.cuda.mem_get_info(tp_rank %  8)
# free = free / (1 << 30)
# total= total/(1 << 30)

# # print(f"Free: {free/1024**2:.1f} MB, Total: {total/1024**2:.1f} MB")
# # async_write_to_csv(allocated, reserved, (reserved -self.last_reserved), free, (free - self.last_free))
# csv.log(
#     tp_rank=tp_rank,
#     time=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
#     allocated=allocated,
#     reserved=reserved,
#     reservedadd=(reserved -self.last_reserved),
#     free=free,
#     freelost=(free - self.last_free),
#     total=total,
# )


# self.last_reserved = reserved
# self.last_free = free


# Case2 记录专家分配情况
# from sglang.srt.utils.csv_logger import AsyncCSVLogger as csv_logger
# from sglang.srt.distributed import (
#     divide,
#     get_tensor_model_parallel_rank,
#     get_tensor_model_parallel_world_size,
# )    
# tp_rank = get_tensor_model_parallel_rank()    
# file_name = f"weight_loader_{tp_rank}.csv"
# file_path = f"/workspace/data/ljy/H20/ktransformer-shell/stats/{file_name}"
# columns = ['tp_rank', 'layerid', "orgi", 'logid', 'phyid' ]
# csv = csv_logger(file_path, columns)
# csv.log(
#     tp_rank=tp_rank,
#     layerid=self.layer_id,
#     orgi=orgi,
#     logid=expert_id,
#     new=physical_expert_ids,
# )