################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace
import logging
import os
import random
import json

from data.dataset import Dataset
os.chdir(root_dir)

from models import choose_model_class

qmsum_prompt_format =  "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"


def get_inputs(prompt_len, num_prompts, tokenizer, path, model_type, seed):
	random.seed(seed)
	sys_prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct."
	# datasets = ["qmsum", "musique", ]  
	dataset = "qmsum"  # Change this to the desired dataset
	filename = f'{path}/{dataset}.jsonl'            
	data_all = [json.loads(line) for line in open(filename, encoding='utf-8')]
	tokenized_prompts = []
	for _ in range(num_prompts):
		context = ''
		while True:
			data = random.choice(data_all)
			context += data['context']
			context += '\n\n'
			prompt = qmsum_prompt_format.format(context=context, input=data['input'])
			messages = [
				{"role": "system", "content": sys_prompt},
				{"role": "user", "content": prompt},
			]
			if model_type == "llama3":
				text = tokenizer.apply_chat_template(
					messages, 
					tokenize=False,
					add_generation_prompt=True
				)
			elif model_type == "qwen3":
				text = tokenizer.apply_chat_template(
					messages,
					tokenize=False,
					add_generation_prompt=True,
					enable_thinking=False 
				)
			else:
				raise ValueError(f"Unknown model type: {model_type}")
			input_ids = tokenizer.encode(text)
			if len(input_ids) >= prompt_len:
				input_ids = input_ids[:prompt_len//2] + input_ids[-prompt_len//2:]
				break
		assert len(input_ids) == prompt_len, f"{len(input_ids)} != {prompt_len}"
		tokenized_prompts.append(torch.tensor(input_ids, dtype=torch.long))
	return tokenized_prompts


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model_path", type=str)
    p.add_argument("--min_prompt_len", type=int)
    p.add_argument("--bsz", type=int)
    p.add_argument("--budget", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--genlen", type=int)
    p.add_argument("--chunk_size", type=int)
    p.add_argument("--rank", type=int)
    p.add_argument("--input_path", type=str)
    
    p.add_argument("--cache_dir", type=str)
    p.add_argument("--offload_device", type=str, choices=["cpu", "disk"])
    p.add_argument("--log_file", type=str)

    return p.parse_args()

MAX_LEN = 32 * 1024

if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        filename=args.log_file,    
        filemode='w',        
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running e2e test with args: {args}")

    model_path = args.model_path
    model_name = model_path.split("/")[-1]
    min_prompt_len = args.min_prompt_len
    offload_device = args.offload_device
    cache_dir = args.cache_dir
    shadowkv_bsz = args.bsz
    sparse_budget = args.budget
    gen_len = args.genlen
    chunk_size = args.chunk_size
    rank = args.rank
    
    assert gen_len + min_prompt_len <= MAX_LEN, f"gen_len {gen_len} + min_prompt_len {min_prompt_len} should be less than {MAX_LEN}"
        
    if 'llama-3' in model_name.lower():
        model_type = 'llama3'
    elif 'qwen' in model_name.lower():
        model_type = 'qwen3'
    else:
        raise ValueError(f"Unknown model type for {model_name}.")
    
    LLM = choose_model_class(model_name)
    ##################### ShadowKV #####################
    llm = LLM(model_name=model_path, device='cuda:0',  batch_size=shadowkv_bsz, 
              max_length=min_prompt_len+gen_len, attn_mode='shadowkv_cpu', sparse_budget=sparse_budget,
            rank=rank, chunk_size=chunk_size, cache_dir=cache_dir, offload_device=offload_device, logger=logger)

    input_ids = get_inputs(min_prompt_len, llm.batch_size, llm.tokenizer, args.input_path, model_type, seed=args.seed)
    input_ids = torch.stack(input_ids, dim=0)
    print(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}", flush=True)
    
    output, throughput_shadowkv, prefill_time, decode_time = llm.batch_generate(input_ids.to(llm.device), gen_len=gen_len, benchmark=True, temperature=0.0)
    logger.info(f"Throughput: {throughput_shadowkv} tokens/s")
    logger.info(f"Prefill time: {prefill_time:.2f} ms")
    logger.info(f"Decode time: {decode_time:.2f} ms")
    total_time = prefill_time + decode_time
    logger.info(f"Total time: {total_time:.2f} ms")
    logger.info(f"Output: {output}")
    print(colored(f"[ShadowKV] Throughput: {throughput_shadowkv} tokens/s", 'red'))
    
    # if hasattr(llm.kv_cache, "v_cache_disk"):
    #     llm.kv_cache.v_cache_disk.get_stats()
    #     prefill_io_time = sum(llm.kv_cache.v_cache_disk.prefill_time_list)*1000  # convert to ms
    #     logger.info(f"Prefill IO Time: {prefill_io_time:.2f} ms")
    
    
    # if hasattr(llm.kv_cache, "svd_time_list"):
    #     svd_time_list = llm.kv_cache.svd_time_list
    #     if len(svd_time_list) > 0:
    #         sum_svd_time = sum(svd_time_list)
    #         len_svd_time = len(svd_time_list)
    #         logger.info(f"SVD Time: {sum_svd_time:.2f} ms, len={len_svd_time}")
    #         percent = sum_svd_time / prefill_time * 100
    #         logger.info(f"SVD Time Percent/Prefill: {percent:.2f}%")
    #         total_time_wo_svd = total_time - sum_svd_time
    #         logger.info(f"Total time without SVD: {total_time_wo_svd:.2f} ms")
    #         slowdown = total_time / total_time_wo_svd
    #         logger.info(f"Slowdown due to SVD: {slowdown:.2f}x")
    #     else:
    #         logger.info("No SVD time recorded.")
    
    # logger.info("="*50)
    
    # if hasattr(llm.kv_cache, "svd_time_list2"):
    #     svd_time_list2 = llm.kv_cache.svd_time_list2
    #     if len(svd_time_list2) > 0:
    #         sum_svd_time2 = sum(svd_time_list2)
    #         len_svd_time2 = len(svd_time_list2)
    #         logger.info(f"SVD Time 2: {sum_svd_time2:.2f} ms, len={len_svd_time2}")
    #         sum_svd_time2_wo_io = sum_svd_time2 - prefill_io_time
    #         logger.info(f"SVD Time 2 without IO: {sum_svd_time2_wo_io:.2f} ms")
    #         prefill_time_wo_io = prefill_time - prefill_io_time
    #         prefill_time_wo_io_wo_svd = prefill_time_wo_io - sum_svd_time
    #         prefill_time_wo_io_wo_svd_wo_svd2 = prefill_time_wo_io_wo_svd - sum_svd_time2_wo_io
    #         logger.info(f"Prefill time without IO: {prefill_time_wo_io:.2f} ms")
    #         logger.info(f"Prefill time without IO and SVD: {prefill_time_wo_io_wo_svd:.2f} ms")
    #         logger.info(f"Prefill time without IO and SVD+SVD2: {prefill_time_wo_io_wo_svd_wo_svd2:.2f} ms")
    #         percent2 = (sum_svd_time2_wo_io + sum_svd_time) / prefill_time_wo_io * 100
    #         logger.info(f"SVD+SVD2 Time Percent/Prefill: {percent2:.2f}%")
    #     else:
    #         logger.info("No SVD time 2 recorded.")    
        
    # print(colored(f"Speedup: {throughput_shadowkv / throughput_baseline:.2f}x", 'red'))
    