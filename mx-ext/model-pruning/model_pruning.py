from transformers import AutoConfig, AutoModelForCausalLM
import json, os, shutil, argparse
from transformers.modeling_utils import no_init_weights
from transformers.tokenization_utils_fast import TOKENIZER_FILE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
import torch


def main(args: argparse.Namespace):
    os.makedirs(args.output_path, exist_ok=True)
    # model type
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # copy file
    for fn in os.listdir(args.model_name_or_path):
        if not os.path.isfile(os.path.join(args.model_name_or_path, fn)):
            continue
        if fn.endswith('.safetensors'):  # TODO: .bin
            continue
        shutil.copy(os.path.join(args.model_name_or_path, fn), 
                    os.path.join(args.output_path, fn))

    # tokenizer
    if args.tensor_parallel_size > 1:
        with open(os.path.join(args.model_name_or_path, TOKENIZER_FILE), 'r') as fr, \
            open(os.path.join(args.output_path, TOKENIZER_FILE), 'w') as fw:
            tokenizer_json = json.load(fr)
            vocab_size = len(tokenizer_json['model']['vocab'])
            added_tokens_num = sum([1 for token in tokenizer_json['added_tokens'] if token['id'] >= vocab_size])
            new_vocab_size = config.vocab_size // args.tensor_parallel_size - added_tokens_num  # TODO: maybe negative
            new_added_tokens = []
            for token in tokenizer_json['added_tokens']:
                if token['id'] < vocab_size and token['id'] >= new_vocab_size:
                    continue
                if token['id'] >= vocab_size:
                    token['id'] = token['id'] - vocab_size + new_vocab_size
                new_added_tokens.append(token)
                    
            tokenizer_json['model']['added_tokens'] = new_added_tokens
            tokenizer_json['model']['vocab'] = dict(list(tokenizer_json['model']['vocab'].items())[:new_vocab_size])
            new_merges = []
            for merge in tokenizer_json['model']['merges']:
                token = ''.join(merge)
                if token not in tokenizer_json['model']['vocab']:
                    continue
                new_merges.append(merge)
            tokenizer_json['model']['merges'] = new_merges
            json.dump(tokenizer_json, fw, indent=2, ensure_ascii=False)

    # moe
    assert config.n_routed_experts % args.expert_parallel_size == 0, 'n_routed_experts must be divisible by expert_parallel_size'
    config.n_routed_experts = config.n_routed_experts // args.expert_parallel_size
    assert config.intermediate_size % args.tensor_parallel_size == 0, 'intermediate_size must be divisible by tensor_parallel_size'
    config.intermediate_size = config.intermediate_size // args.tensor_parallel_size
    
    # attention
    assert config.num_attention_heads % args.tensor_parallel_size == 0, 'num_attention_heads must be divisible by tensor_parallel_size'
    config.num_attention_heads = config.num_attention_heads // args.tensor_parallel_size
    
    # embedding/output layer
    if args.tensor_parallel_size > 1:
        config.pad_token_id = config.vocab_size // args.tensor_parallel_size - 1
    assert config.vocab_size % args.tensor_parallel_size == 0, 'vocab_size must be divisible by tensor_parallel_size'
    config.vocab_size = config.vocab_size // args.tensor_parallel_size

    #layer
    if args.layer is not None:
        config.num_hidden_layers = args.layer

    with no_init_weights(args.fast_init):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.to(torch.bfloat16)
    model.save_pretrained(args.output_path, safe_serialization=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model pruning')
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--expert_parallel_size', type=int, default=1)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--fast_init', action="store_true")  # pass weight initiation
    args = parser.parse_args()
    main(args)
