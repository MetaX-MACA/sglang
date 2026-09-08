from typing import Union

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = (
    "/mxstorage/pde_ai/models/model_quant_opt/DeepSeek/"
    "DeepSeek-V4-Flash_FlexSMQ-AWQ_W8A8/vllm_quant_model/"
)


def get_device_max_memory() -> dict[Union[int, str], str]:
    max_memory: dict[Union[int, str], str] = {}
    for device_id in range(torch.cuda.device_count()):
        torch.empty(1, device=f"cuda:{device_id}")
        available_bytes = torch.cuda.mem_get_info(device_id)[0]
        available_gb = available_bytes / 10**9
        if device_id == 0:
            ratio = 0.5 if available_gb > 30 else 0.3
        else:
            ratio = 0.875 if available_gb > 30 else 0.7
        max_memory[device_id] = f"{available_gb * ratio:.1f}GB"

    cpu_available_gb = psutil.virtual_memory().available / 10**9
    max_memory["cpu"] = f"{cpu_available_gb * 0.875:.1f}GB"
    return max_memory


def infer() -> None:
    print(f"MODEL_PATH: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    max_memory = get_device_max_memory()
    print(f"max_memory: {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        max_memory=max_memory,
    )
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain the current state of China's economy."}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=512)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    infer()
