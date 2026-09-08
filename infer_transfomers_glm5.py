# Load model directly

import psutil
import torch
from typing import Union
from easydict import EasyDict

from transformers import AutoModelForCausalLM
from transformers import AutoProcessor


def get_device_max_memory() -> dict[Union[int, str], Union[int, str]]:
    memory_rate = EasyDict({"gpu0_ratio": 0.5, "other_ratio": 0.875})
    # memory_rate = EasyDict({"gpu0_ratio": 0.85, "other_ratio": 0.95})

    for i in range(torch.cuda.device_count()):
        _ = torch.tensor([0], device=i)
        cuda_avail_memory = {
            i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())
        }
        cpu_avail_memory = psutil.virtual_memory().available
        max_memory = {}
        for cuda_num, cuda_memory in cuda_avail_memory.items():
            cuda_memory_gb = cuda_memory / (10**9)
            # print(f"GPU{cuda_num} cuda_avail_memory: {cuda_memory_gb:.1f}GB")
            if cuda_num == 0:
                # The ratio is an experience value that you can manually adjust yourself.
                gpu0_ratio = memory_rate.gpu0_ratio if cuda_memory_gb > 30 else 0.3
                max_memory[cuda_num] = f"{cuda_memory_gb * gpu0_ratio:.1f}GB"
            else:
                other_ratio = memory_rate.other_ratio if cuda_memory_gb > 30 else 0.7
                max_memory[cuda_num] = f"{cuda_memory_gb * other_ratio:.1f}GB"
        print(f"cpu_avail_memory: {cpu_avail_memory / (10**9):.1f}GB")
        cpu_ratio = 0.875
        max_memory["cpu"] = f"{cpu_avail_memory / (10**9) * cpu_ratio:.1f}GB"
        print("final_use_model_kwargs: ", max_memory)
        # max_memory =  {0: '0.1GB', 'cpu': '100GB'}

    return max_memory


def infer():
    MODEL_PATH = "/mxstorage/pde_ai/models/llm/ChatGLM/GLM-5-W8A8"

    print(f"MODEL_PATH: {MODEL_PATH}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    max_memory = get_device_max_memory()

    print(f"max_memory: {max_memory}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="flash_attention_2",  # 或 "sdpa"
    )
    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image", "url": "/sw_home/diwang/workspace/inference_engine/LightCompress/examples/GLM-4.5V/candy.JPG"},
                # {"type": "text", "text": "What animal is on the candy?"}
                # {"type": "text", "text": "What animal is the king of forest?"}
                {"type": "text", "text": "评价中国经济现状"}
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cpu")

    outputs = model.generate(**inputs, max_new_tokens=512)
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))


if __name__ == "__main__":
    import debugpy

    debugpy.connect(5678)
    debugpy.wait_for_client()
    infer()
