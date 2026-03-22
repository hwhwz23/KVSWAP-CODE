
import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig
from models.modeling_llama import eager_attention_forward as llama_eager_attention_forward
from models.modeling_llama import att_forward as llama_att_forward
from models.modeling_llama import model_forward as llama_model_forward

from models.modeling_qwen2 import eager_attention_forward as qwen2_eager_attention_forward
from models.modeling_qwen2 import att_forward as qwen2_att_forward
from models.modeling_qwen2 import model_forward as qwen2_model_forward

from models.modeling_qwen2_5_vl import att_forward as qwen2_5_vl_att_forward
from models.modeling_qwen2_5_vl import model_forward as qwen2_5_vl_model_forward

from models.modeling_qwen3 import eager_attention_forward as qwen3_eager_attention_forward
from models.modeling_qwen3 import att_forward as qwen3_att_forward
from models.modeling_qwen3 import model_forward as qwen3_model_forward

# from models.modeling_opt import att_forward as opt_att_forward
# from models.modeling_opt import decoder_forward as opt_decoder_forward

from utils import *
import gc

from sklearn.decomposition import PCA

import argparse
import transformers

transformers.models.llama.modeling_llama.eager_attention_forward = llama_eager_attention_forward
transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_att_forward
transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward

transformers.models.qwen2.modeling_qwen2.eager_attention_forward = qwen2_eager_attention_forward
transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_att_forward
transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward

transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen2_5_vl_att_forward
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen2_5_vl_model_forward

transformers.models.qwen3.modeling_qwen3.eager_attention_forward = qwen3_eager_attention_forward
transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_att_forward
transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward = qwen3_model_forward

# transformers.models.opt.modeling_opt.OPTAttention.forward = opt_att_forward
# transformers.models.opt.modeling_opt.OPTDecoder.forward = opt_decoder_forward

def merge_heads_gqa(ind, kv_groups):
	# ind: b, h, fn -> b, 
	assert kv_groups >= 1, f"kv_groups={kv_groups}"
	if kv_groups == 1:
		return ind
	else:
		b, h, fn = ind.shape
		ind = ind.view(b, h//kv_groups, kv_groups, fn)
		ind = ind.view(-1, kv_groups*fn)
		max_value = ind.max()
		counts = torch.zeros(ind.shape[0], max_value+1, dtype=torch.long, device=ind.device)
		counts.scatter_add_(1, ind, torch.ones_like(ind))
		_, topk_indices = counts.topk(fn, dim=1) 
		# b*h//kv_groups, fn -> b*h//kv_groups, kv_groups, fn
		ind = topk_indices.unsqueeze(1).expand(-1, kv_groups, -1).reshape(b, h, fn)
		return ind

def get_skewing_mask(query_states, key_states, ratio, skewing_matrix, head_dim, head_num, kv_groups, q_proj_weight, merge_heads):
	weight_mask = torch.zeros_like(q_proj_weight.t())
	n = int(head_dim * ratio)
	# Speculate major columns in Wq and Wk
	for head in range(head_num):
		start = head * head_dim
		end = (head+1) * head_dim
		skew_matrix = skewing_matrix[head]
		if skew_matrix.device != query_states.device:
			skew_matrix = skew_matrix.to(query_states.device)
		# s, d @ d, d -> s, d -> d
		_, ind = torch.topk(torch.sum(torch.abs(query_states[0, head] @ skew_matrix), dim=-2), n)
		ind = ind.repeat(weight_mask.shape[0], 1)
		weight_mask[:, start:end].scatter_add_(-1, ind, torch.ones_like(ind, dtype=weight_mask.dtype).repeat(weight_mask.shape[0], 1))  
		# if not self.skewing_only_query:
		_, ind = torch.topk(torch.sum(torch.abs(key_states[0, head] @ skew_matrix), dim=-2), n)
		ind = ind.repeat(weight_mask.shape[0], 1)
		weight_mask[:, start:end].scatter_add_(-1, ind, torch.ones_like(ind, dtype=weight_mask.dtype).repeat(weight_mask.shape[0], 1))
	weight_mask = weight_mask[0].view(head_num, head_dim) # h, d
	ind = weight_mask.topk(n, dim=-1)[1] # h, n
	if merge_heads:
		ind = merge_heads_gqa(ind.unsqueeze(0), kv_groups)[0] # h, n
	new_weight_mask = torch.zeros_like(weight_mask) # h, d
	new_weight_mask.scatter_(-1, ind, 1)
	return new_weight_mask


def main(args):

	if "deepseek" in args.model_path.lower():
		if "qwen3" in args.model_path.lower():
			args.model_type = 'ds_qwen3'
		elif "llama" in args.model_path.lower():
			args.model_type = 'ds_llama'
		else:
			raise ValueError("DeepSeek model type not supported")
	elif 'videollama3' in args.model_path.lower():
		args.model_type = 'videollama3'
	elif 'internvl3' in args.model_path.lower():
		args.model_type = 'internvl3'
	elif 'opt' in args.model_path.lower():
		args.model_type = 'opt'
	elif 'llama' in args.model_path.lower():
		args.model_type = 'llama'
	elif 'qwen2.5-vl' in args.model_path.lower():
		args.model_type = 'qwen2.5vl'
	elif 'qwen2' in args.model_path.lower():
		args.model_type = 'qwen2'
	elif 'qwen3' in args.model_path.lower():
		args.model_type = 'qwen3'
	elif 'gemma-3' in args.model_path.lower():
		args.model_type = 'gemma3'
	else:
		raise ValueError("Model type not supported")
	
	args.model_name = args.model_path.split("/")[-1]

	torch.manual_seed(1234)
	torch.cuda.manual_seed(1234)
	torch.cuda.manual_seed_all(1234)
	np.random.seed(1234)	
 
	VIDEO_LLMS = ('qwen2.5vl', 'videollama3', 'internvl3')
 
	torch_dtype = torch.bfloat16
	attn_implementation = 'flash_attention_2'
	extra_kwargs = {}
 
	if args.model_type == 'qwen2.5vl':
		from transformers import Qwen2_5_VLForConditionalGeneration
		model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
			args.model_path,
			torch_dtype=torch_dtype,
			attn_implementation=attn_implementation,
			device_map="auto",).eval()
		processor = AutoProcessor.from_pretrained(args.model_path)
		print(f"Model config - Max position embeddings: {model.config.max_position_embeddings}", flush=True)
		print(f"Tokenizer max length: {processor.tokenizer.model_max_length}", flush=True)
	elif args.model_type == 'videollama3':
		model = AutoModelForCausalLM.from_pretrained(
			args.model_path,
			trust_remote_code=True,
			device_map="auto",
			torch_dtype=torch_dtype,
			attn_implementation=attn_implementation,
		).eval()
		processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
	elif args.model_type == 'gemma3':
		from transformers import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
		model = Gemma3ForConditionalGeneration.from_pretrained(
			args.model_path,
			torch_dtype=torch_dtype,
			attn_implementation=attn_implementation,
			device_map="auto",).eval()
		model = model.language_model
		assert isinstance(model, Gemma3ForCausalLM), f"Model {model} is not a Gemma3ForCausalLM"
		processor = AutoProcessor.from_pretrained(args.model_path)
		tokenizer = processor.tokenizer 
	elif args.model_type == 'internvl3':
		from models.internvl3.modeling_internvl_chat import InternVLChatModel
		model = InternVLChatModel.from_pretrained(
			args.model_path,
			torch_dtype=torch_dtype,
			load_in_8bit=False,
			use_flash_attn=attn_implementation=='flash_attention_2',
			device_map='auto').eval()
		tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
	else:
		model = AutoModelForCausalLM.from_pretrained(args.model_path, 
													attn_implementation=attn_implementation,
													torch_dtype=torch_dtype, 
													device_map='auto', 
													**extra_kwargs
													).eval()
		tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False) 
	
	from transformers import GenerationConfig    
	try:
		gen_configs = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=False)
	except Exception as e:
		# we use a temperature of $0.6$, a top-p value of $0.95$,
		assert args.model_type == 'ds_qwen3', "model_type should be ds_qwen3"
		gen_configs = GenerationConfig(
			temperature=0.6,
			top_p=0.95,
			do_sample=True
		)
	print("gen_configs=", gen_configs)
	
	if args.model_type == 'opt':
		decoder_model = model.model.decoder
	elif args.model_type == 'internvl3':
		decoder_model = model.language_model.model
	else:
		decoder_model = model.model
	
	if 'infinigen' in args.modes:	
		assert len(args.modes) == 1, f"args.modes={args.modes} should only contain"
		assert args.record_kv_post_rope and not args.record_kv_pre_rope
		assert args.concat_q, f"args.concat_q={args.concat_q} should be True for infinigen mode"
	else:
		assert args.record_kv_post_rope or args.record_kv_pre_rope, f"args.record_kv_post_rope={args.record_kv_post_rope}, args.record_kv_pre_rope={args.record_kv_pre_rope}"
  
	for layer_i in range(len(decoder_model.layers)):
		decoder_model.layers[layer_i].self_attn.record_kv_post_rope = args.record_kv_post_rope
		decoder_model.layers[layer_i].self_attn.record_kv_pre_rope = args.record_kv_pre_rope
	
	head_dim = decoder_model.layers[0].self_attn.head_dim  
	kv_groups = decoder_model.layers[0].self_attn.num_key_value_groups if hasattr(decoder_model.layers[0].self_attn, 'num_key_value_groups') else 1
	num_heads = decoder_model.layers[0].self_attn.config.num_attention_heads 
	kv_heads = num_heads // kv_groups 
	print(f"head_dim={head_dim}, kv_groups={kv_groups}, num_heads={num_heads}, kv_heads={kv_heads}")

	assert args.task in ("wikitext", "ptb", 'c4', 'mlvu', 'math500', 'livecodebench'), f"args.task={args.task}"

	batchsize = args.bsz
	query_key_dict = {'pre_rope': {}, 'post_rope': {}}

	def collect_kv(decoder_model, query_key_dict):
		for layer_i in range(len(decoder_model.layers)):
			if args.record_kv_pre_rope:
				if layer_i not in query_key_dict['pre_rope']:
					query_key_dict['pre_rope'][layer_i] = {'q': [], 'k': []}
				if args.concat_q:
					query_key_dict['pre_rope'][layer_i]['q'].append(decoder_model.layers[layer_i].self_attn.record_query_pre.cpu())
				query_key_dict['pre_rope'][layer_i]['k'].append(decoder_model.layers[layer_i].self_attn.record_key_pre.cpu())
			if args.record_kv_post_rope:
				if layer_i not in query_key_dict['post_rope']:
					query_key_dict['post_rope'][layer_i] = {'q': [], 'k': []}
				if args.concat_q:
					query_key_dict['post_rope'][layer_i]['q'].append(decoder_model.layers[layer_i].self_attn.record_query_post.cpu())
				query_key_dict['post_rope'][layer_i]['k'].append(decoder_model.layers[layer_i].self_attn.record_key_post.cpu())
		return query_key_dict

	if args.task == 'mlvu':
		assert batchsize == 1
		assert args.model_type in VIDEO_LLMS, f"Model type {args.model_type} not supported for MLVU task"
		MLVU_DATA_DIR = './bench/MLVU/MVLU_DATA/MLVU'
		from dataset import MLVU
		from eval_gen import inference_single_video, inference_single_video_internvl3
		data_list = {
			"summary": ("9_summary.json", f"{MLVU_DATA_DIR}/video/9_summary", "video", False)
		}
		dataset = MLVU(f"{MLVU_DATA_DIR}/json", data_list)
		i = 0
		max_new_tokens = 1
		if args.model_type == 'qwen2.5vl':
			PROCESSED_MLVU_DATA_DIR = f'./bench/MLVU/MVLU_DATA/Qwen2.5-VL_processed'
		elif args.model_type == 'videollama3':
			PROCESSED_MLVU_DATA_DIR = f'./bench/MLVU/MVLU_DATA/VideoLlama3_processed'
		elif args.model_type == 'internvl3':
			PROCESSED_MLVU_DATA_DIR = f'./bench/MLVU/MVLU_DATA/InternVL3_processed'
		else:
			raise ValueError(f"Model type {args.model_type} not supported for MLVU task")
		if not os.path.exists(PROCESSED_MLVU_DATA_DIR):   
			os.makedirs(PROCESSED_MLVU_DATA_DIR)
		DO_REUSE_PROCESSED = True
		DO_SAVE_PROCESSED = True
		for example in tqdm(dataset):
			video_path = example["video"]
			sys_ins = "Carefully watch this video and pay attention to every detail. Based on your observations, answer the given questions."
			if args.model_type == 'internvl3':
				_ = inference_single_video_internvl3(args, video_path, '', sys_ins + '\n\n' + example["question"], model, tokenizer, max_new_tokens, 
								gen_configs, PROCESSED_MLVU_DATA_DIR, DO_REUSE_PROCESSED, DO_SAVE_PROCESSED) 
			else:
				_ = inference_single_video(args, video_path, '', sys_ins + '\n\n' + example["question"], model, processor, max_new_tokens, 
								gen_configs, PROCESSED_MLVU_DATA_DIR, DO_REUSE_PROCESSED, DO_SAVE_PROCESSED,
								total_pixels=24576 * 28 * 28) 
			query_key_dict = collect_kv(decoder_model, query_key_dict)
			i += 1
			if i == args.eval_samples:
				break
		del dataset, example, video_path, sys_ins
	else:
		testenc = get_testenc(tokenizer, args.task, args.seq_len, args.model_name, save=True)      
		assert args.eval_samples >= 0, f"args.eval_samples={args.eval_samples}"
		args.eval_samples = int(args.eval_samples)
		if args.eval_samples > 0:
			testenc = testenc[:args.eval_samples]
		nlls = []
		with torch.no_grad():
			for i in tqdm(range(0, testenc.size(0), batchsize)):
				gc.collect()
				torch.cuda.empty_cache()
				b_start, b_end = i, min(i+batchsize, testenc.size(0))
				input_ids = testenc[b_start:b_end].to('cuda:0')
				pad_token_id = tokenizer.pad_token_id
				if pad_token_id is None:
					pad_token_id = tokenizer.eos_token_id
				attention_mask = (input_ids != pad_token_id)
				generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, min_new_tokens=1)
				query_key_dict = collect_kv(decoder_model, query_key_dict)

		del testenc, input_ids, generated_ids
   
	n_layer = len(decoder_model.layers)
 
	if args.modes[0] == 'infinigen':
		q_proj_weight = decoder_model.layers[0].self_attn.q_proj.weight.data.clone()
 
	model = model.cpu()
	del model, decoder_model
	
	gc.collect()
	torch.cuda.empty_cache()
 
	if args.modes[0] == 'infinigen':
		A = torch.zeros(n_layer, num_heads, head_dim, head_dim, device="cuda:0", dtype=torch.float32)

	for key_mode, query_keys in query_key_dict.items():
		for mode in args.modes:
			for layer_i in query_keys.keys():
				if args.task == 'mlvu' or mode == 'infinigen':
					k = torch.concat(query_keys[layer_i]['k'], dim=-2)
					if args.concat_q:
						q = torch.concat(query_keys[layer_i]['q'], dim=-2)
				else:
					k = torch.stack(query_keys[layer_i]['k'], dim=0)
					if args.concat_q:
						q = torch.stack(query_keys[layer_i]['q'], dim=0)
				if mode == 'infinigen':
					for head in range(num_heads):
						in_q = q[0, head].cuda().float() # s, d
						in_k = k[0, head // kv_groups].cuda().float()
						uq, sq, vq = torch.svd(in_q)
						uk, sk, vk = torch.svd(in_k)
						s = sq.to(A.device) * sk.to(A.device)
						vq = vq.to(A.device)
						a = torch.zeros(head_dim, head_dim, device=A.device)
						_, ind = s.sort()
						r, c = a.shape
						A[layer_i, head] = a.scatter(-1, ind.unsqueeze(0).repeat(r,1), vq)
						
					repeat_k = k.cuda().view(1, num_heads, -1, head_dim).repeat_interleave(dim=1, repeats=kv_groups) # B, H, S, D
					q = q.cuda().view(1, num_heads, -1, head_dim)
					for ratio in args.ratios:
						skewing_mask = get_skewing_mask(
								q, repeat_k, 
								ratio, A[layer_i].to(torch_dtype), head_dim, num_heads, kv_groups, q_proj_weight, 
																							merge_heads=args.merge_heads)
						save_dir = f'{args.save_dir}/skewing_mask_{args.task}_{args.eval_samples}/'+args.model_name+'_'+str(ratio)
						if not os.path.exists(save_dir):
							os.makedirs(save_dir)
						torch.save(skewing_mask.cpu(), f"{save_dir}/{layer_i}.pt")	
						partial_index = skewing_mask.nonzero().view(skewing_mask.shape[0], -1, 2)[:,:, 1]
						save_dir = f'{args.save_dir}/skewing_idx_{args.task}_{args.eval_samples}/'+args.model_name+'_'+str(ratio)
						if not os.path.exists(save_dir):
							os.makedirs(save_dir)
						torch.save(partial_index.cpu(), f"{save_dir}/{layer_i}.pt")	
					if layer_i == n_layer - 1:
						save_dir = f'{args.save_dir}/skewing_matrix_{args.task}_{args.eval_samples}'
						if not os.path.exists(save_dir):
							os.makedirs(save_dir)
						save_dir = save_dir + "/" + args.model_name + ".pt"
						print("Saving skewing matrix to ", save_dir)
						A = A.to(torch_dtype)
						torch.save(A.cpu(), save_dir)
					gc.collect()
					torch.cuda.empty_cache()      
					continue
				elif mode == 'loki':
					pca_components = []
					assert torch.isfinite(k).all(), f"stacked_k has inf or nan, {torch.sum(torch.isnan(k))}, {torch.sum(torch.isinf(k))}"
					for head in range(kv_heads):
						if args.task == 'mlvu':
							in_k = k[:, head].reshape(-1, head_dim).float().cpu().numpy()
						else:	
							in_k = k[:, 0, head].reshape(-1, head_dim).float().cpu().numpy()
						pca = PCA()
						pca.fit(in_k)
						pca_components.append(torch.tensor(pca.components_))
					pca_components = torch.stack(pca_components, dim=0).transpose(1, 2)
					for ratio in args.ratios:
						save_dir = f'{args.save_dir}/loki_proj_{key_mode}_{args.task}_{args.eval_samples}/'+args.model_name+'_'+mode+'_'+str(ratio)
						if not os.path.exists(save_dir):
							os.makedirs(save_dir)
						u_save = pca_components[:, :, :int(head_dim * ratio)]
						torch.save(u_save.cpu().half(), f"{save_dir}/loki_proj_{layer_i}.pt")
					gc.collect()
					torch.cuda.empty_cache()  
					continue
				elif mode == 'sh':
					k = k.cuda().float().mean(dim=-3)
					k = k.view(-1, k.shape[-1])
				elif mode == 'mh':
					k = k.transpose(-2, -3).cuda().float().contiguous()
					k = k.view(-1, k.shape[-2]*k.shape[-1])
				gc.collect()
				torch.cuda.empty_cache()
				u, _, _ = torch.linalg.svd(k.t(), full_matrices=False)
				del k
				for ratio in args.ratios:
					save_dir = f'{args.save_dir}/lowrank_proj_{key_mode}_{args.task}_{args.eval_samples}/'+args.model_name+'_'+mode+'_'+str(ratio)
					if not os.path.exists(save_dir):
						os.makedirs(save_dir)
					u_save = u[:, :int(head_dim * ratio)]
					torch.save(u_save.cpu().half(), f"{save_dir}/lr_kproj_{layer_i}.pt")
				del u, u_save
				gc.collect()
				torch.cuda.empty_cache()
			print(f'{key_mode} {mode} with ratios {args.ratios} finished')          
   
	print("Done!")

if __name__ == "__main__":

	args = argparse.Namespace()
	args.torch_dtype = torch.bfloat16
	args.model_path = './MODELS/Qwen3-1.7B'
	args.seq_len = 32768

	# args.ratios = [0.5/4, 0.5/4/4]
	# args.ratios = [1/8, 0.25/8]
	# args.ratios = [1/2, 0.25/2]

	# args.ratios = [1/8, 0.25/8]
	# args.ratios = [0.25/2, 0.25/4/2]

	# args.ratios = [1/8, 0.25/8]
	# args.ratios = [1/4, 0.25/4]
	# args.ratios = [1/2, 0.25/2]
	# args.ratios = [1/8, 1/32]
	# args.ratios = [1/2]
	# args.ratios = [1, 1/4]
	args.ratios = [1/32, 0.125]
	# args.ratios = [0.125/4]
	# args.ratios = [0.5]
	# args.modes = ['mh', 'sh']
	# args.modes = ['mh']
	# args.modes = ['loki']
	args.modes = ['infinigen']
	#############################################################
	args.merge_heads = True
	args.task = 'c4'
	# args.task = 'mlvu'
	args.eval_samples = 20
	args.bsz = 1
	args.record_kv_post_rope = True
	args.record_kv_pre_rope = False
	# args.record_kv_post_rope = False
	# args.record_kv_pre_rope = True
	args.concat_q = False
	HOSTNAME = os.uname()[1]
	if args.modes[0] == 'infinigen':
		args.record_kv_post_rope = True
		args.record_kv_pre_rope = False
		args.concat_q = True
		if args.merge_heads:
			args.save_dir = f'./exps/{HOSTNAME}/infinigen_skew'
		else:
			args.save_dir = f'./exps/{HOSTNAME}/infinigen2_skew'
	elif args.modes[0] == 'loki':
		args.save_dir = f'./exps/{HOSTNAME}/loki_proj'
	else:
		args.save_dir = f'./exps/{HOSTNAME}/lowrank_proj'
	main(args)

