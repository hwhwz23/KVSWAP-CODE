import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import numpy as np
import argparse

def make_np_weights(hf_model_path, save_dir_):
	model_name = hf_model_path.split('/')[-1]
	save_dir = os.path.join(save_dir_, model_name, "weights-np")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	else:
		print(f"Warning: {save_dir} already exists, skipping...")
		return
		
	model = AutoModelForCausalLM.from_pretrained(hf_model_path, 
													attn_implementation='eager',
													torch_dtype=torch.float16, 
													device_map='cpu', 
													).eval()
	for key, param in model.named_parameters():
		np.save(os.path.join(save_dir, key), param.detach().half().numpy())
		# print(key, param.shape)
	# copy configs
	config_file = os.path.join(hf_model_path, '*.json')
	os.system(f'cp {config_file} {os.path.join(save_dir_, model_name)}')

	config_file = os.path.join(hf_model_path, '*.txt')
	os.system(f'cp {config_file} {os.path.join(save_dir_, model_name)}')


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--hf_model_path', type=str)
	parser.add_argument('--save_dir', type=str)
	args = parser.parse_args()

	if not os.path.exists(args.hf_model_path):
		print(f"Error: {args.hf_model_path} does not exist when creating np weights")
		exit(0)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	make_np_weights(args.hf_model_path, args.save_dir)

	print(f"Done: {args.hf_model_path} -> {args.save_dir}")

