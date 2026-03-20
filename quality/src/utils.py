from datasets import load_dataset
import os
import torch
import json

def get_dataset(dataset, **kwargs):
	if "wikitext" in dataset:
		data_list = load_dataset("wikitext", "wikitext-2-raw-v1", split='test', num_proc=8)['text'] 
	elif 'ptb' in dataset:
		data_list = load_dataset("ptb_text_only", "penn_treebank", split='test', num_proc=8)['sentence'] 
	elif "c4" in dataset:
		data_list = load_dataset('allenai/c4', 
                           	data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
                            split='validation', num_proc=8)['text'] 	
	# elif 'arc' in dataset:
	# 	data_list = load_dataset("allenai/ai2_arc", "ARC-Easy", num_proc=8)['train']['question']
	elif 'math500' in dataset:
		sys_prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct."
		prompt_format = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()
		import pandas
		split = "math_500_test"
		df = pandas.read_csv(f"https://openaipublic.blob.core.windows.net/simple-evals/{split}.csv")
		data_all = [{"_id": str(i), **row.to_dict()} for i, row in df.iterrows()]
		print(f"Loaded {len(data_all)} samples from {split}")
		tokenizer = kwargs['tokenizer']
		data_list = []
		for item in data_all:
			text = tokenizer.apply_chat_template(
				[
					{"role": "system", "content": sys_prompt},
					{"role": "user", "content": prompt_format.format(Question=item['Question'])},
					{"role": "assistant", "content": f"<think>\n\nAnswer:{item['Answer']}</think>\n\n"},
				],
				tokenize=False,
				add_generation_prompt=True,
				enable_thinking=False
			)
			data_list.append(text)
	elif 'livecodebench' in dataset:
		dataset_file = './bench/livecodebench/livecodebenchv5_prompts.json'
		data_all = json.load(open(dataset_file, encoding='utf-8'))
		data_list = []
		for item in data_all:
			data_list.append(item)
	else:
		raise ValueError(f"Dataset {dataset} not recognized")
	return data_list





def get_testenc(tokenizer, dataset, seqlen, model_name, save=True, cant_find_error=False, save_dir='./tmp'):
	testenc_fname = f'{save_dir}/{dataset}-{model_name}-{seqlen}.dataset'
	if os.path.exists(testenc_fname):
		testenc = torch.load(testenc_fname).view(-1, seqlen)
		print(f"Loading testenc from {testenc_fname}, testenc.shape={testenc.shape}")
		return testenc
	if cant_find_error:
		raise ValueError(f"Cannot find {testenc_fname}")
	data_list = get_dataset(dataset, tokenizer=tokenizer)
	data_list = "\n\n".join(data_list)
	tokenizer.model_max_length = int(1e30)
	testenc = tokenizer(data_list, return_tensors='pt')
	sample_len = (testenc.input_ids.size(1) // seqlen) * seqlen
	testenc = testenc.input_ids[:,:sample_len].reshape(-1, seqlen)
	print(f"Loading {dataset} testenc.shape={testenc.shape}")
	# save to file
	if save:
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		torch.save(testenc, testenc_fname)
		print(f"Saved testenc to {testenc_fname}")
	return testenc


def get_mask_rate(decoder_model):
	mask_non_zero_rate = []
	kv_groups = decoder_model.layers[0].self_attn.num_key_value_groups if hasattr(decoder_model.layers[0].self_attn, 'num_key_value_groups') else 1
	head_dim = decoder_model.layers[0].self_attn.head_dim
	print("kv_groups: ", kv_groups)
	for layer_i in range(len(decoder_model.layers)):
		skewing_mask = decoder_model.layers[layer_i].self_attn.skewing_mask
		skewing_mask_group = skewing_mask.view(-1, kv_groups, head_dim).sum(dim=1)
		r = (skewing_mask_group != 0).float().sum().item() / skewing_mask_group.numel()
		mask_non_zero_rate.append(r)
	print("mask_non_zero_rate: ", mask_non_zero_rate)
	avg_mask_non_zero_rate = sum(mask_non_zero_rate) / len(mask_non_zero_rate)
	print("avg_mask_non_zero_rate: ", avg_mask_non_zero_rate)
	print("normed_avg_mask_non_zero_rate: ", avg_mask_non_zero_rate/kv_groups)
	return avg_mask_non_zero_rate, avg_mask_non_zero_rate/kv_groups

