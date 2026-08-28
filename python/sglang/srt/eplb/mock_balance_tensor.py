# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import os
import numpy as np
import torch
from collections import Counter

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# EP  参数
num_experts = int(os.environ.get("mx_num_experts", 256))
num_cards = int(os.environ.get("mx_num_cards", 64))
micro_batch = int(os.environ.get("mx_micro_batch", 32))
seq_len = int(os.environ.get("mx_seq_len", 1)) # decode set 1; prefill set actual seq_len

topk_per_batch = 8
batches_per_card = micro_batch * seq_len # 32 * 1
topk_per_card = batches_per_card * topk_per_batch  # 32 * 8 = 256
total_topk = num_cards * topk_per_card  # 144 * 256
topk_per_expert = total_topk // num_experts  # 128

ids = np.repeat(np.arange(num_experts), topk_per_expert) # 288, 128
np.random.shuffle(ids)
np_tensor = ids.reshape(num_cards, batches_per_card, topk_per_batch) # 144 32 8
EP_BALANCED_TENSOR = torch.tensor(np_tensor, dtype=torch.int64)

# verify
assert EP_BALANCED_TENSOR.shape == (
    num_cards,
    batches_per_card,
    topk_per_batch,
)  # [128, 16,8]
flat_ids = EP_BALANCED_TENSOR.flatten().tolist()
counter = Counter(flat_ids)
assert all(v == topk_per_expert for v in counter.values()), "专家分布不均衡"



# AFD pd mixed
num_experts = 256
num_attn_cards = 32 # 需要动态调整
micro_batch = 32 
seq_len = 1

topk_per_batch = 8 
batches_per_card = micro_batch * seq_len
topk_per_card = batches_per_card * topk_per_batch
total_topk = num_attn_cards * topk_per_card # 32* 16 *8
topk_per_expert = total_topk // num_experts # 16

ids = np.tile(np.arange(num_experts), topk_per_expert)
# np.random.shuffle(ids)
np_tensor = ids.reshape(num_attn_cards, -1, topk_per_batch)
AFD_BALANCED_TENSOR_ORI = torch.tensor(np_tensor, dtype=torch.int32)
# verify
## micro_bs: 16 * 8 * 32 // 16 = 16 * 16
## shape ==> 16 * 32 * 8
assert AFD_BALANCED_TENSOR_ORI.shape == (num_attn_cards, batches_per_card, topk_per_batch)
flat_ids = AFD_BALANCED_TENSOR_ORI.flatten().tolist()
counter = Counter(flat_ids)
assert all(v == topk_per_expert for v in counter.values()), "专家分布不均衡"

# AFD_BALANCED_TENSOR = dedup_tensor(AFD_BALANCED_TENSOR_ORI)
AFD_BALANCED_TENSOR = AFD_BALANCED_TENSOR_ORI