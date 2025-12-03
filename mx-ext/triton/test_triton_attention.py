import unittest
import torch
#import triton.testing as tt
import triton
import triton.language as tl

from torch.profiler import profile, record_function, ProfilerActivity

from decode_attention_C500 import decode_attention_fwd
#from decode_attention_A100 import decode_attention_fwd

def set_seed(seed):
    torch.manual_seed(seed)             # 设置CPU生成随机数的种子
    torch.cuda.manual_seed(seed)        # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 如果有多个GPU，为所有的GPU设置随机种子
    #np.random.seed(seed)                # 设置numpy生成随机数的种子
    #random.seed(seed)                   # 设置Python生成随机数的种子
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False     # 如果确定输入数据的大小或每次的输入数据变化不大，设置为False


class TestDecodeAttention(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda'
        self.dtype = torch.bfloat16 #float16
        self.H_Q =  8 # Number of heads of query
        self.H_KV = 8  # Number of heads of key & value
        self.D_QK = 128  # Dimension per head of query & key
        self.D_V = 128 # Dimension per head of value

        # model
        self.max_batch_size = 4097
        self.N_CTX = 8196  # context length base on server.model
        # request
        self.total_tokens = 323596 

    def test_decode_attention_fwd(self):
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
        seq_lens = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        # if deepseek R1
        self.H_Q = 32
        self.H_KV = 8 
        self.D_QK = 128
        self.D_V = 128
        self.total_tokens = 322540
        self.max_batch_size = 2049
        self.N_CTX = 128*1024
        self.attn_logits_batch_size = 160 
        self.num_kv_splits = 8
        # if Llama 3
        # self.H_Q = 8 
        # self.H_KV = 8 
        # self.D_QK = 128
        # self.D_V = 128
        # self.total_tokens = 323596 
        # self.max_batch_size =  4097
        # self.N_CTX = 8196 
        # self.attn_logits_batch_size = 128
        # self.num_kv_splits = 8

        print(f"warm up")
        for batch_size in batch_sizes:
            if (batch_size <= self.max_batch_size and batch_size <= self.attn_logits_batch_size):
                o = self.__test_decode_attention_fwd(batch_size, seq_lens[0])

        print(f"warm up done")
        
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./', 'profiling-C500-kernel-maca2.29.0.10.json'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True) as prof:
            with record_function('decode_attention_fwd'):
                for seq_len in seq_lens:
                    for batch_size in batch_sizes:
                        if (batch_size <= self.max_batch_size and batch_size <= self.attn_logits_batch_size and seq_len <= self.N_CTX):
                            o = self.__test_decode_attention_fwd(batch_size, seq_len)
                            prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16))
            

    def __test_decode_attention_fwd(self, batch_size, max_seq_len):
        num_kv_splits = 8
        sm_scale = 1.0 / (self.D_QK**0.5)
        # dynamic update total tokens
        self.total_tokens = batch_size * max_seq_len
        if (max_seq_len > self.N_CTX):
            self.N_CTX = max_seq_len

        # Create random input tensors
        Q = torch.randn(batch_size, self.H_Q, self.D_QK, dtype=self.dtype, device=self.device)
        K = torch.randn(self.total_tokens, self.H_KV, self.D_QK, dtype=self.dtype, device=self.device)
        V = torch.randn(self.total_tokens, self.H_KV, self.D_V, dtype=self.dtype, device=self.device)
        Out = torch.empty(batch_size, self.H_Q, self.D_QK, dtype=self.dtype, device=self.device)

        req_to_token = torch.randint(0,
                                self.total_tokens,
                                (self.max_batch_size, self.N_CTX),
                                device=self.device)
        b_req_idx = torch.arange(batch_size, device=self.device)
        b_seq_len = torch.full((batch_size,), max_seq_len, device=self.device)
         
        attn_logits = torch.empty(
            (self.attn_logits_batch_size, self.H_Q, self.num_kv_splits, self.D_V + 1),
            dtype=self.dtype,
            device=self.device,
        )

        # Compute reference output using PyTorch
        # ref_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

        # Compute output using Triton kernel
        # 定义CUDA事件
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 记录开始时间
        start_event.record()
        decode_attention_fwd(
            Q,
            K,
            V,
            Out,
            req_to_token,
            b_req_idx,
            b_seq_len,
            attn_logits,
            self.num_kv_splits,
            sm_scale
        )
        # 记录结束时间
        end_event.record()

        # 等待kernel执行完毕
        torch.cuda.synchronize()

        # 计算耗时
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"bs: {batch_size}, l: {max_seq_len}, kernel execution time: {elapsed_time} ms")

        return Out
        # Check if the outputs are close enough
        # self.assertTrue(torch.allclose(ref_out, Out, atol=1e-2, rtol=1e-3))

if __name__ == '__main__':
    set_seed(42)
    unittest.main()

