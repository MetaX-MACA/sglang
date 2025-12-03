import argparse
import os
import re
import json
import time
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
import concurrent.futures

def extract_after_think(output):
    # 使用 </think> 作为分隔符分割字符串
    parts = output.split('</think>')

    # 如果分割后的部分大于1，说明找到了 </think>
    if len(parts) > 1:
        # 返回 </think> 后面的部分
        return parts[1].strip()
    else:
        # 如果没有找到 </think>，返回原始字符串或空字符串
        return output.strip()

def build_prompt(text):
    return "[Round {}]\n\n问：{}\n\n答：".format(1, text)

def build_baichuan_prompt(text):
    return "<reserved_106>{}<reserved_107>".format(text)

def first_option_postprocess(text: str, options: str = "ABCD", cushion: bool = True) -> str:
    # 预处理流水线 ----------------------------------------------------------
    text = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', text)  # 合并被空格分隔的字母
    clean_text = re.sub(r'[^\u4e00-\u9fa5A-Da-d，。！？：:（）()【】\[\]“”\*★▲◆]', ' ', text)  # 保留关键符号
    
    # 语义锚点检测规则 (高精度) ---------------------------------------------
    anchor_rules = [
        # 锚点类型 1：明确答案声明
        (r'(?<![a-zA-Z])(?:答案|正确选项)[为是：:]\s*([%s])(?![a-zA-Z])' % options, 20),
        (r'(?:故|因此|综上)[^。]{0,15}?选?[择]?\s*([%s])\b' % options, 18),
        
        # 锚点类型 2：特殊格式标识
        (r'\\boxed{([%s])}' % options, 25),  # LaTeX公式最高优先级
        (r'(?:\*\*|▲{2}|★{2})([%s])(?:\.|选项)?(?:\*\*|▲{2}|★{2})' % options, 20),
        (r'[(\[〔【《〈「]([%s])[)\]〕】》〉」]' % options, 15),
        
        # 锚点类型 3：结构化答案位置
        (r'[。！？；;]\s*([%s])\s*(?:选项)?$' % options, 12)  # 严格句尾检测
    ]

    # 抗干扰过滤规则 ------------------------------------------------------
    filter_rules = [
        (r'(?:参数|变量|示例)[A-D]', -100),  # 排除技术术语中的字母
        (r'\b([%s])[a-zA-Z]+' % options, -50)  # 排除连续字母组合
    ]

    # 多阶段检测系统 ------------------------------------------------------
    candidates = {}
    
    # 阶段 1：高置信锚点检测
    for pattern, weight in anchor_rules:
        for match in re.findall(pattern, clean_text):
            if match in options:
                candidates[match] = candidates.get(match, 0) + weight

    # 阶段 2：干扰项过滤
    for pattern, penalty in filter_rules:
        for match in re.findall(pattern, text):
            if match in options:
                candidates[match] = candidates.get(match, 0) + penalty

    # 阶段 3：可信结果裁决
    if candidates:
        valid = [k for k, v in candidates.items() if v > 0]
        if valid:
            return max(valid, key=lambda x: candidates[x])
    
    # 阶段 4：安全兜底策略
    # 策略 1：逆向精准扫描
    reverse_text = clean_text[::-1]
    for char in reverse_text:  
        if char in options:
            return char
    
    # 策略 2：首字母聚焦检测
    first_alpha = re.search(r'(?<![a-zA-Z])([%s])(?![a-zA-Z])' % options, clean_text)
    if first_alpha:
        return first_alpha.group(1)
        
    return ''  # 安全容错
    
def query_llm(prompt):
    input_ids = tokenizer.encode(prompt)
    tries = 0

    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=15000,
                top_p=0.7,
            )
            content = completion.choices[0].message.content
            result = extract_after_think(content)
            return result
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def main(args):
    global model, client, tokenizer
    model = args.model
    URL = "http://localhost:8000/v1"
    API_KEY = "EMPTY"
    client = OpenAI(base_url=URL, api_key=API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # 一次性读取所有任务
    tasks = []
    with open(args.test_jsonl, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            prompt = json_obj["origin_prompt"]
            if args.model.find('Baichuan') > 0:
                prompt = build_baichuan_prompt(prompt)
            tasks.append({
                "origin_prompt": prompt,
                "gold": json_obj["gold"]
            })

    # 初始化结果容器
    results = [''] * len(tasks)
    total_correct = 0

    # 创建线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        # 提交所有任务并保留索引
        future_to_index = {
            executor.submit(query_llm, task["origin_prompt"]): idx
            for idx, task in enumerate(tasks)
        }
        
        # 按完成顺序收集结果
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result
                # 实时统计正确率
                choice = first_option_postprocess(result, options="ABCD")
                if choice == tasks[idx]["gold"]:
                    total_correct += 1
                    print(f"Task {idx + 1}/{len(tasks)}: Correct! Predicted: {choice}, Gold: {tasks[idx]['gold']}")
                else:
                    print(f"Task {idx + 1}/{len(tasks)}: Incorrect. Predicted: {choice}, Gold: {tasks[idx]['gold']}")
            except Exception as e:
                print(f"Error in task {idx}: {str(e)}")
                results[idx] = ''

    # 构建输出结构
    output_json = []
    for idx, task in enumerate(tasks):
        choice = first_option_postprocess(results[idx], options="ABCD")
        output_json.append({
            "origin_prompt": task["origin_prompt"],
            "output": results[idx],
            "labels": choice,
            "golden": task["gold"],
            "correct": choice == task["gold"]
        })

    # 保存结果
    model_name = args.model.split('/')[-2]
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = os.path.join(args.save_dir, f"{model_name}_bs{args.batch_size}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)

    # 输出统计信息
    total = len(tasks)
    print(f"Total: {total}, Correct: {total_correct}, Accuracy: {total_correct/total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--test_jsonl", "-t", type=str, default="/pde_ai/datasets/ceval_vllm_client/ceval_val_cmcc.jsonl")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, required=True)
    args = parser.parse_args()
    main(args)