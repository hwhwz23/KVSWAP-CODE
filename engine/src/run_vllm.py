from vllm import LLM, SamplingParams
import time
import torch
import random
random.seed(12)
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import os
import csv

def load_existing_results(output_csv):
	existing = {}
	if not os.path.exists(output_csv):
		return existing
	with open(output_csv, "r", encoding="utf-8", errors="ignore") as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				seqlen = int(row["seqlen"])
				batch = int(row["batch"])
				throughput = float(row["throughput"])
				existing[(seqlen, batch)] = throughput
			except (KeyError, ValueError, TypeError):
				# Ignore malformed rows and keep valid ones.
				continue
	return existing

def get_promopts(tokenizer, batch, prompt_len, shuffle=False):
	def shuffle_words(text):
		words = text.split()
		random.shuffle(words)
		return ' '.join(words)
	prompts = []
	for i in range(batch):
		input_text = dataset[i]['input'] + 'nice to meet you! ' * 500
		if shuffle:
			# shuffle to avoid any KV prefix caching 
			input_text = shuffle_words(input_text)
		input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
		assert input_ids.shape[1] >= prompt_len, f"Input length {input_ids.shape} is less than prompt length {prompt_len}"
		text = tokenizer.batch_decode(input_ids[0, :prompt_len], skip_special_tokens=False)
		prompts.append("".join(text))
	return prompts

def run(batch, seqlen):
	prompt_len = seqlen-100

	prefill_params = SamplingParams(temperature=0, max_tokens=1)
	decode_params = SamplingParams(temperature=0, max_tokens=101)

	torch.cuda.synchronize()
	start_prefill = time.time()
	_ = llm.generate(get_promopts(tokenizer, batch, prompt_len, shuffle=True), prefill_params)
	torch.cuda.synchronize()
	end_prefill = time.time()
	prefill_time = end_prefill - start_prefill
 
	torch.cuda.synchronize()
	start_decode = time.time()
	outputs = llm.generate(get_promopts(tokenizer, batch, prompt_len, shuffle=True), decode_params)
	torch.cuda.synchronize()
	end_decode = time.time()
	total_time = end_decode - start_decode

	print("Generated text:", outputs[0].outputs[0].text)
	print(f"Prefill time:  {prefill_time:.4f} seconds")
	print(f"Total time (prefill + decode): {total_time:.4f} seconds")
	print(f"Estimated decode time: {total_time - prefill_time:.4f} seconds")
	throughput = 100*batch / (total_time - prefill_time)
	print(f"Throughput: {throughput:.2f} tokens/sec")
	return throughput


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--seqlen-list', type=str, default='16384,32768')
    parser.add_argument('--batch-list', type=str, default='1,2,4,8,16')

    args = parser.parse_args()
    model = args.model_path

    model_name = os.path.basename(model)
    seqlen_list = [int(seqlen) for seqlen in args.seqlen_list.split(',')]
    batch_list = [int(batch) for batch in args.batch_list.split(',')]
    print(f"seqlen_list: {seqlen_list}, batch_list: {batch_list}")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_csv = os.path.join(args.output_path, f'{model_name}_results.csv')
    expected_cases = {(seqlen, batch) for seqlen in seqlen_list for batch in batch_list}
    result_dict = load_existing_results(output_csv)

    if os.path.exists(output_csv):
        print(f"Found existing result file: {output_csv}")
        missing_cases = sorted(expected_cases - set(result_dict.keys()))
        if len(missing_cases) == 0:
            print("All requested (seqlen, batch) cases already exist. Exit.")
            exit(0)
        print(f"Existing complete cases: {len(result_dict)}/{len(expected_cases)}")
        print(f"Will continue running missing cases: {missing_cases}")
    else:
        print(f"No existing csv found. Will run all cases: {sorted(expected_cases)}")

    if 'llama' in model.lower():
        model_type = 'llama3'
    elif 'qwen' in model.lower():
        model_type = 'qwen3'
    else:
        raise NotImplementedError()

    if model_type == 'llama3':
        model_dir = 'Llama'
    elif 'qwen' in model_type:
        model_dir = 'Qwen3'
    else:
        raise NotImplementedError()

    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = load_dataset("json", data_files=f'./data/ruler/{model_dir}/synthetic/32768/data/niah_single_1/validation.jsonl', split='train')

    llm = LLM(
        model=model,
        max_model_len=32768,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        # enforce_eager=True,
        enforce_eager=False,
        swap_space=0
    )

    # warmup 
    _ = run(1, 4096)
    print("Warmup done")

    for seqlen in seqlen_list:
        for batch in batch_list:
            if (seqlen, batch) in result_dict:
                print(f"Skip existing result for seqlen={seqlen}, batch={batch}")
                continue
            print(f"Running with seqlen={seqlen}, batch={batch}")
            throughput = run(batch, seqlen)
            result_dict[(seqlen, batch)] = throughput
            print(f"Throughput: {throughput:.2f} tokens/sec for seqlen={seqlen}, batch={batch}")

    # save results to csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seqlen', 'batch', 'throughput'])
        for seqlen, batch in sorted(result_dict.keys()):
            throughput = result_dict[(seqlen, batch)]
            writer.writerow([seqlen, batch, throughput])
            
    print(f"Results saved to {output_csv}")

