import sglang as sgl
import os
# import sglang.srt.models.deepseek_v2

# os.environ["DISABLE_FLASHINFER"] = "1"

def main():
    # Sample prompts.
    prompts = [
        # "would you like",
        # "The president of the United States is",
        "1+1 =",
        "中国的首都？",
        "人体最大器官？",
        "The capital of France is",
        "介绍下朱元璋",
    ]
    # Create a sampling params object.
    sampling_params = {"temperature": 0.5, "top_p": 0.68}

    # #Test 1
    # model_path = "/workspace/mnt/storage/admin@metax.com/pde-ai-models/llm/Llama/Meta-Llama-3-8B-Instruct/"
    # llm = sgl.Engine(model_path=model_path, tp_size=1, disable_cuda_graph=True)
    
    # #Test 2
    # model_path ="/mnt/shared_data/Qwen3-8B"
    # model_path ="/mnt/shared_data/meta-llama/Llama-2-7b-hf/"
    model_path = "/mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-V2-Lite/"
    llm = sgl.Engine(model_path=model_path, tp_size=8, disable_cuda_graph=False, trust_remote_code=True, attention_backend="flashinfer", dtype="bfloat16") 

    #Test 3
    # model_path = "/workspace/mnt/storage/admin@metax.com/pde-ai-models/llm/DeepSeek/DeepSeek-R1-BF16-tp32"
    # llm = sgl.Engine(model_path=model_path, tp_size=1, disable_cuda_graph=True, trust_remote_code=True, dtype="bfloat16") 

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()

