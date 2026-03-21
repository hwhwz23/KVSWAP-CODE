import argparse
import dataclasses
import os
import time
from typing import Union, List, Optional, Any
from functools import partial
import numpy as np
import torch
import nvtx
import json
import gc
from tqdm import tqdm
# from datasets import load_dataset
import random
torch.cuda.set_per_process_memory_fraction(0.8, device=0)
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import AutoTokenizer
from compression import CompressionConfig
from model_config import get_model_config, cache_bytes, hidden_bytes
from pytorch_backend import (TorchDevice, TorchDisk, DeviceType, general_copy, fix_recursive_import, TorchTensor)
from timer import timers
from utils import (Task, ExecutionEnv, GB, T, ValueHolder,
	array_1d, array_2d, array_3d, str2bool)
from methods import merge_qk_weight, get_partial_q_weight, speculate_attention, lr_kcache_func
from model_utils import rms_norm
import torch.nn.functional as F
from cache_manager import CacheManager

fix_recursive_import()

CONCAT_PROJ_WEIGHT = False

@dataclasses.dataclass(frozen=True)
class Policy:
	gpu_batch_size: int
	num_gpu_batches: int

	w_gpu_percent: float
	w_cpu_percent: float
	cache_gpu_percent: float
	cache_cpu_percent: float
	act_gpu_percent: float
	act_cpu_percent: float

	# Whether to use pinned memory for weights on CPU
	pin_weight: bool

	# Compress KV cache with group-wise quantization
	compress_cache: bool
	comp_cache_config: CompressionConfig
	
	max_num_kv: int
	sort_idx: bool

	en_copy_key: bool
	en_finer_prefetch_sync: bool
	en_ahead_prefetch: bool
	start_layer: str
	flash_att: bool
	paged_att: bool
	reuse_budget: int

	att_score_mode: str
	lr_proj_mode: str
	use_token_cache: bool
	batch_split: Any
	alpha: float 
	use_mmap: bool
	token_group: int
	att_comp_mode: str
	
	@property
	def w_disk_percent(self):
		return 100 - self.w_gpu_percent - self.w_cpu_percent

	@property
	def cache_disk_percent(self):
		return 100 - self.cache_gpu_percent - self.cache_cpu_percent

	@property
	def act_disk_percent(self):
		return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
	percents = np.cumsum(percents)
	assert np.abs(percents[-1] - 100) < 1e-5

	for i in range(len(percents)):
		if cur_percent < percents[i]:
			return choices[i]
	return choices[-1]


def init_weight_list(weight_specs, policy, env, config):
	dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
	dev_choices = [env.disk, env.cpu, env.gpu]
	sizes = [np.prod(spec[0]) for spec in weight_specs]
	sizes_cumsum = np.cumsum(sizes)
	ret = []
	for i in range(len(weight_specs)):
		mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
		home = get_choice(mid_percent * 100, dev_percents, dev_choices)
		# print(weight_specs[i], home)
		shape, dtype, filename = weight_specs[i]
		compress = False
		if len(shape) < 2:
			pin_memory = True
		else:
			pin_memory = policy.pin_weight
		if not compress:
			weight = home.allocate(shape, dtype, pin_memory=pin_memory)
			weight.load_from_np_file(weight_specs[i][2])
		else:
			weight = home.compressed_device.allocate(
				shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)
			weight.load_from_np_file(weight_specs[i][2])
		if config.model_type in ('qwen2', 'qwen3') and config.dtype == np.float16:
			weight.data = weight.data.to(torch.bfloat16)
		ret.append(weight)
	return ret


class InputEmbed:
	def __init__(self, config, env, policy):
		self.config = config
		self.env = env
		self.policy = policy
		self.compute = self.env.gpu if self.env.gpu is not None else self.env.cpu
		self.weight_load_dst = self.compute
		self.task = None

	def set_task(self, task):
		self.task = task

	def init_weight(self, weight_home, path):
		v, h, dtype = self.config.vocab_size, self.config.hidden_size, self.config.dtype
		path = os.path.join(path, "")
		if self.config.model_type == 'opt':
			s = self.config.max_position_embeddings
			weight_specs = [
				((v, h), dtype, path + "model.decoder.embed_tokens.weight"),
				((s + 2, h), dtype, path + "model.decoder.embed_positions.weight"),
			]
		else:
			weight_specs = [
				((v, h), dtype, path + "model.embed_tokens.weight"),
			]           
		weights = init_weight_list(weight_specs, self.policy, self.env, self.config)
		weight_home.store(weights)

	def load_weight(self, weight_home, weight_read_buf, k):
		if k == 0:
			dst = self.weight_load_dst
			if self.config.model_type == 'opt':
				w_token, w_pos = weight_home.val
				w_store = (w_token.smart_copy(dst), w_pos.smart_copy(dst))
			else:
				w_token = weight_home.val[0]
				w_store = (w_token.smart_copy(dst), (None, None))
			weight_read_buf.store(w_store)
							
	def init_cache_one_gpu_batch(self, cache_home):
		if self.policy.use_token_cache:
			shape = (self.policy.gpu_batch_size, self.task.prompt_len + self.task.gen_len - 1)
			self.token_cache = self.compute.allocate(shape, np.int64, pin_memory=True)

	def load_cache(self, cache_home, cache_read_buf, i):
		pass

	def store_cache(self, cache_home, cache_write_buf, i):
		pass

	@torch.inference_mode()
	def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
				cache_write_buf, i, k):
		# Compute input embedding
		donate = [False] * 4
		h, donate[0] = hidden.val, True
		mask, donate[1] = attention_mask.val.smart_copy(self.compute)

		if k == self.policy.num_gpu_batches - 1:
			(w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
		else:
			(w_token, _), (w_pos, _) = weight_read_buf.val

		# if self.en_trans_hidden:  
		if hasattr(self, 'token_cache'):  
			# store
			if i == 0: # prefill
				indices = (slice(None), slice(0, h.shape[1])) # b, s
				self.token_cache.data[indices].copy_(h.data)
			# elif i < self.task.gen_len - 1:
			else: # record the last one for consecutive loading
				pos = self.task.prompt_len + i
				indices = (slice(None), slice(pos-h.shape[1], pos)) # b, s+i-1, s+i
				self.token_cache.data[indices].copy_(h.data)
			# load     
			if i > 0:
				indices = (slice(None), slice(0, self.task.prompt_len + i))
				h.data = self.token_cache.data[indices]
		h = self.compute.input_embed(h, mask, w_token, w_pos, donate)
		if self.env.gpu is None:
			h.data = h.data.to(torch.float32)
		hidden.val = h
		

class OutputEmbed:
	def __init__(self, config, env, policy):
		self.config = config
		self.env = env
		self.policy = policy
		self.compute = self.env.gpu if self.env.gpu is not None else self.env.cpu
		self.weight_load_dst = self.compute
		self.task = None

	def set_task(self, task):
		self.task = task

	def init_weight(self, weight_home, path):
		v, h, dtype = (self.config.vocab_size, self.config.hidden_size,
			self.config.dtype)
		path = os.path.join(path, "")
		if self.config.model_type == 'opt':
			weight_specs = [
				((h,), dtype, path + "model.decoder.final_layer_norm.weight"),
				((h,), dtype, path + "model.decoder.final_layer_norm.bias"),
				((v, h), dtype, path + "model.decoder.embed_tokens.weight"),
			]
		else:
			weight_specs = [
				((h,), dtype, path + "model.norm.weight"),
				((v, h), dtype, path + "model.embed_tokens.weight" if self.config.tie_word_embeddings else path + "lm_head.weight")
			]
		weights = init_weight_list(weight_specs, self.policy, self.env, self.config)
		weight_home.store(weights)

	def load_weight(self, weight_home, weight_read_buf, k):
		if k == 0:
			dst1 = self.weight_load_dst
			dst2 = self.compute
			if self.config.model_type == 'opt':
				w_ln, b_ln, w_token = weight_home.val
				w_store = (w_ln.smart_copy(dst2), b_ln.smart_copy(dst2), w_token.smart_copy(dst1))
			else:
				w_norm, w_token = weight_home.val
				w_store = (w_norm.smart_copy(dst2), (None, None), w_token.smart_copy(dst1))
			weight_read_buf.store(w_store)            

	def init_cache_one_gpu_batch(self, cache_home):
		pass 

	def load_cache(self, cache_home, cache_read_buf, i):
		pass 

	def store_cache(self, cache_home, cache_write_buf, i):
		pass 
	
	@torch.inference_mode()
	def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
				cache_write_buf, i, k):
		donate = [False] * 4
		h, donate[0] = hidden.val, True
		
		if k == self.policy.num_gpu_batches - 1:
			# Clear the weight_read_buf if it is the last gpu batch
			(w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
		else:
			(w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val
	
		h = self.compute.output_embed(h, w_ln, b_ln, w_token, donate, self.task.do_sample, self.task.temperature)        
		hidden.val = h


prefetch_kv_buf = None
cache_manager_inst = None

class SelfAttention:
	def __init__(self, config, env, policy, layer_id, 
				 skew_paths, lr_proj_path, enable_pred):
		self.config = config
		self.env = env
		self.layer_id = layer_id
		self.policy = policy
		self.compute = self.env.gpu if self.env.gpu is not None else self.env.cpu
		self.weight_load_dst = self.compute
		self.attention_compute = self.compute
		self.task = None
		self.prefetch_idx = None
		self.prefetch_kv = None
		self.cache_manager = None
		self.lr_k = None
		self.skew_matrix = None
		self.partial_wq = None
		self.partial_index = None
		self.lr_kproj = None
		self.qproj = None
		self.qnorm = None
		self.lr_proj_mode = self.policy.lr_proj_mode
		self.enable_pred = enable_pred
		self.kv_window_index = None
		self.window_kv_buf = None
		# if en_pred_emb, we need the first layer's proj weight
		en_pred_emb = '0' in self.policy.start_layer and 'emb' in self.policy.start_layer
		self.en_pred_emb = en_pred_emb
		if layer_id > 0 or en_pred_emb:
			if self.lr_proj_mode == 'base':
				skew_partial_idx_path, self.skew_matirx_path = skew_paths
				kv_rep = self.config.num_kv_groups
				self.partial_index = torch.load(skew_partial_idx_path+f'/{layer_id}.pt', 
												map_location=self.compute.dev).to(torch.long)
			elif self.lr_proj_mode.startswith('lr_proj'):
				dtype = torch.bfloat16 if self.config.model_type in ('qwen2', 'qwen3') and self.config.dtype == np.float16 else torch.float16
				self.lr_kproj = torch.load(lr_proj_path+f'/lr_kproj_{layer_id}.pt', map_location=self.compute.dev).to(dtype)

	def set_task(self, task):
		self.task = task
		self.chunk_size = 128*1024
  
		# if task.gen_len > 1:
		# base = 1 * 4096 * 4096 // 2
		# base = 1 * 4096 * 4096 // 8
		# MULTIPLE = 128
		# raw_chunk = base // (len(task.inputs) * task.prompt_len)
		# assert raw_chunk > 0, f"raw_chunk={raw_chunk}, bsz={len(task.inputs)}, prompt_len={task.prompt_len}"
		# aligned = ((raw_chunk + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
		# self.chunk_size = max(MULTIPLE, min(aligned, task.prompt_len))
		# if self.layer_id == 0:
		# 	num_chucks = (task.prompt_len + self.chunk_size - 1) // self.chunk_size
		# 	print(f"setting chuck size={self.chunk_size}, num_chucks={num_chucks}, raw_chunk={raw_chunk}, aligned={aligned}")


	def init_weight(self, weight_home, path):
		h, dtype = (self.config.hidden_size, self.config.dtype)
		if self.config.model_type == 'opt':
			path = os.path.join(os.path.join(path, f"model.decoder.layers.{self.layer_id}.self_attn"))
			weight_specs = [
				((h, h), dtype, path + ".q_proj.weight"),
				((h,), dtype, path + ".q_proj.bias"),
				((h, h), dtype, path + ".k_proj.weight"),
				((h,), dtype, path + ".k_proj.bias"),
				((h, h), dtype, path + ".v_proj.weight"),
				((h,), dtype, path + ".v_proj.bias"),
				((h, h), dtype, path + ".out_proj.weight"),
				((h,), dtype, path + ".out_proj.bias"),
				((h,), dtype, path + "_layer_norm.weight"),
				((h,), dtype, path + "_layer_norm.bias"),
			]
			weights = init_weight_list(weight_specs, self.policy, self.env, self.config)
			weights[0].data = torch.cat((weights[0].data, weights[1].data.unsqueeze(1).to(weights[0].data.device)), dim=1)
			weights[0].shape = (h, h+1)
			weights[2].data = torch.cat((weights[2].data, weights[3].data.unsqueeze(1).to(weights[2].data.device)), dim=1)
			weights[2].shape = (h, h+1)
		else:
			path = os.path.join(os.path.join(path, f"model.layers.{self.layer_id}"))
			h_ = self.config.head_dim * self.config.num_kv_heads
			hq = self.config.head_dim * self.config.num_attention_heads
			if self.config.attention_bias:
				weight_specs = [
					((hq, h), dtype, path + ".self_attn.q_proj.weight"),
					((h,), dtype, path + ".self_attn.q_proj.bias"),
					((h_, h), dtype, path + ".self_attn.k_proj.weight"),
					((h_,), dtype, path + ".self_attn.k_proj.bias"),
					((h_, h), dtype, path + ".self_attn.v_proj.weight"),
					((h_,), dtype, path + ".self_attn.v_proj.bias"),
					((h, hq), dtype, path + ".self_attn.o_proj.weight"),
					# ((h,), dtype, path + ".self_attn.o_proj.bias"), # qwen doesn't have o_proj.bias
					((h,), dtype, path + ".input_layernorm.weight"),
				]
				weights = init_weight_list(weight_specs, self.policy, self.env, self.config)
				weights[0].data = torch.cat((weights[0].data, weights[1].data.unsqueeze(1).to(weights[0].data.device)), dim=1)
				weights[0].shape = (h, h+1)
				weights[2].data = torch.cat((weights[2].data, weights[3].data.unsqueeze(1).to(weights[2].data.device)), dim=1)
				weights[2].shape = (h_, h+1)
			else:
				weight_specs = [
					((hq, h), dtype, path + ".self_attn.q_proj.weight"),
					((h_, h), dtype, path + ".self_attn.k_proj.weight"),
					((h_, h), dtype, path + ".self_attn.v_proj.weight"),
					((h, hq), dtype, path + ".self_attn.o_proj.weight"),
					((h,), dtype, path + ".input_layernorm.weight"),
				]     
				weights = init_weight_list(weight_specs, self.policy, self.env, self.config)
	
			if self.config.model_type == 'qwen3':
				weight_specs = [
					((self.config.head_dim,), dtype, path + ".self_attn.q_norm.weight"),
					((self.config.head_dim,), dtype, path + ".self_attn.k_norm.weight"),
				]
				weights += init_weight_list(weight_specs, self.policy, self.env, self.config)
	
		if self.layer_id > 0 or self.en_pred_emb:
			if self.lr_proj_mode != 'none' and self.config.model_type == 'qwen3':
				self.qnorm = weights[-2].data
	  
			if self.lr_proj_mode == 'base':
				if '.pt' in self.skew_matirx_path:
					skew_matrix = torch.load(self.skew_matirx_path, map_location=self.compute.dev)[self.layer_id].clone().to(torch.float16)    
				else:
					skew_matrix = torch.load(self.skew_matirx_path+f'/{self.layer_id}.pt', map_location=self.compute.dev).to(torch.float16)    
				if self.config.model_type == 'opt':
					weights[0].data = merge_qk_weight(weights[0].data, skew_matrix, 
													self.config.num_attention_heads, self.config.head_dim)
					weights[2].data = merge_qk_weight(weights[2].data, skew_matrix, 
													self.config.num_attention_heads, self.config.head_dim)
					self.skew_matrix = None
					self.partial_wq = get_partial_q_weight(weights[0].data, self.partial_index, 
														self.config.num_attention_heads, self.config.head_dim)
				else: # only fuse q weight  
					if skew_matrix.dtype != weights[0].data.dtype:
						skew_matrix = skew_matrix.to(weights[0].data.dtype)
					# skew after rope, can not merge weight 
					# weights[0].data = merge_qk_weight(weights[0].data, skew_matrix, 
					# 								self.config.num_attention_heads, self.config.head_dim) 
					self.partial_wq = weights[0].data
					self.skew_matrix = skew_matrix
					# print("init self.partial_wq, layer_id", self.partial_wq.shape, self.layer_id, flush=True)
			elif self.lr_proj_mode.startswith('lr_proj'):
				self.qproj = weights[0].data
			  
		weight_home.store(weights)


	def load_weight(self, weight_home, weight_read_buf, k):
		if k == 0:
			dst1 = self.weight_load_dst
			dst2 = self.compute
			if self.config.model_type == 'opt':
				w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
				if CONCAT_PROJ_WEIGHT:
					raise NotImplementedError()
				else:
					w_store = (w_q.smart_copy(dst1), b_q.smart_copy(dst2),
								w_k.smart_copy(dst1), b_k.smart_copy(dst2),
								w_v.smart_copy(dst1), b_v.smart_copy(dst2),
								w_out.smart_copy(dst1), b_out.smart_copy(dst2),
								w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
								(None, None), (None, None))
			elif self.config.attention_bias:
				w_q, b_q, w_k, b_k, w_v, b_v, w_out, w_norm = weight_home.val
				if CONCAT_PROJ_WEIGHT:
					raise NotImplementedError()
					# w_qkv = 
					# b_qkv = 
				else:
					w_store = (w_q.smart_copy(dst1), b_q.smart_copy(dst2),
								w_k.smart_copy(dst1), b_k.smart_copy(dst2),
								w_v.smart_copy(dst1), b_v.smart_copy(dst2),
								w_out.smart_copy(dst1), (None, None), 
								w_norm.smart_copy(dst2), (None, None),
								(None, None), (None, None))
			else:     
				if self.config.model_type == 'qwen3':
					w_q, w_k, w_v, w_out, w_norm, q_norm, k_norm = weight_home.val
					if CONCAT_PROJ_WEIGHT:
						raise NotImplementedError()
					else:
						w_store = (w_q.smart_copy(dst1), (None, None),
									w_k.smart_copy(dst1), (None, None),
									w_v.smart_copy(dst1), (None, None),
									w_out.smart_copy(dst1), (None, None),
									w_norm.smart_copy(dst2), (None, None))    
						w_store += (q_norm.smart_copy(dst2), k_norm.smart_copy(dst2))
				else:
					w_q, w_k, w_v, w_out, w_norm = weight_home.val
					if CONCAT_PROJ_WEIGHT:
						raise NotImplementedError()
					else:
						w_store = (w_q.smart_copy(dst1), (None, None),
									w_k.smart_copy(dst1), (None, None),
									w_v.smart_copy(dst1), (None, None),
									w_out.smart_copy(dst1), (None, None),
									w_norm.smart_copy(dst2), (None, None))    
						w_store += ((None, None), (None, None))
   
			weight_read_buf.store(w_store)

	def init_cache_one_gpu_batch(self, cache_home):
		if self.policy.cache_gpu_percent == 100:
			device = self.env.gpu
		elif self.policy.cache_cpu_percent == 100:
			device = self.env.cpu
		elif self.policy.cache_disk_percent == 100:
			device = self.env.disk
		else:
			raise NotImplementedError
			device = self.env.mixed

		if self.policy.compress_cache:
			assert device.device_type != DeviceType.MIXED
			device = device.compressed_device
		
		if self.layer_id == 0:
			if self.en_pred_emb:
				# if en_pred_emb, use_token_cache should be disabled
				assert not self.policy.use_token_cache, "use_token_cache should be disabled when en_pred_emb is True"
			elif self.policy.use_token_cache:
				return
		
		force_bf16 = self.config.model_type in ('qwen2', 'qwen3') and self.config.dtype == np.float16
		if '0' in self.policy.start_layer:
			if 'emb' in self.policy.start_layer:
				start_layer = -1
			else:
   				start_layer = 0
		else:
			start_layer = 1
			
		if self.policy.lr_proj_mode != 'none' and self.layer_id > start_layer:
			assert device == self.env.disk, f'{device.device_type} {self.policy.lr_proj_mode}'
			device2 = self.env.gpu if self.env.gpu is not None else self.env.cpu
			dtype = torch.bfloat16 if force_bf16 else torch.float16
			# s, b, r, pre-allocate
			if self.policy.lr_proj_mode == 'base': 
				lr_k_shape = self.task.prompt_len + self.task.gen_len - 1, self.policy.gpu_batch_size, self.config.num_kv_heads, self.partial_index.shape[1]
			else:
				lr_k_shape = self.task.prompt_len + self.task.gen_len - 1, self.policy.gpu_batch_size, self.lr_kproj.shape[1]
			self.lr_k = torch.empty(*lr_k_shape, dtype=dtype, device=device2.dev)
   
			if self.policy.paged_att:
				global cache_manager_inst
				if cache_manager_inst is None:
					cache_manager_inst = CacheManager(self.config.num_kv_heads, 
									   			self.config.head_dim, self.policy.token_group, self.policy.gpu_batch_size, 
										  		self.policy.max_num_kv, self.config.num_hidden_layers, reuse_budget=self.policy.reuse_budget, dtype=dtype)
				self.cache_manager = cache_manager_inst
			else:
				hd_len = self.config.head_dim * self.config.num_kv_heads
				global prefetch_kv_buf
				if prefetch_kv_buf is None:
					assert self.policy.max_num_kv % self.policy.token_group == 0, f"{self.policy.max_num_kv} % {self.policy.token_group} != 0"
					kv_len = self.policy.max_num_kv // self.policy.token_group 
					# n, b, 2, ghd
					shape = (kv_len, self.policy.gpu_batch_size, 2, self.policy.token_group * hd_len)
					prefetch_kv_buf = device2.allocate(shape, np.float16, pin_memory=True, force_bf16=force_bf16)   
				self.prefetch_kv = prefetch_kv_buf
				# 2, g, b, hd
				shape = (2, self.policy.token_group, self.policy.gpu_batch_size, hd_len)
				self.window_kv_buf = torch.zeros(*shape, dtype=dtype, device=device2.dev) 
			attr = 'bn2ghd'
		else:
			attr = 'bs2hd'
		cache_name = f'kvcache{self.layer_id}' 
		if self.policy.use_mmap:
			cache_name += '.mmap'
		else:
			cache_name += '.bin'
		cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy, name=cache_name,
												force_bf16=force_bf16, batch_split=self.policy.batch_split, attr=attr)
		cache_home.store(cache)            
			
			
	def load_cache(self, cache_home, cache_read_buf, i): # load whole cache
		if i == 0:  # prefill, no cache
			return

		# should only work when self.prefetch_kv is None
		assert self.prefetch_kv is None and self.cache_manager is None, "Prefetch kv should be None"
		kv_home = cache_home.val

		# Pick code path
		if self.policy.compress_cache:
			dst = self.attention_compute.compressed_device
		else:
			dst = self.attention_compute
		# b, s, 2hd
		indices = (slice(0, kv_home.shape[0]),
				slice(0, self.task.prompt_len + i - 1)) 

		cache_read_buf.store(
			(kv_home.smart_copy(dst, indices, copy_key='load_kv'), 'bs2hd')
		)
				  
	
	def prefetch_cache(self, cache_home, cache_read_buf, i, prefetch_idx, spec_stream=None):
		if i == 0:  # prefill, no cache
			return

		# print(f"prefetch {self.layer_id}, {i}", flush=True)
		kv_home = cache_home.val

		if self.policy.compress_cache:
			dst = self.attention_compute.compressed_device
		else:
			dst = self.attention_compute

		assert kv_home.device.device_type == DeviceType.DISK, "Cache should be on disk"
		# b, n
		n = prefetch_idx.shape[1] 
		if self.policy.paged_att:
			cache_read_buf.store(
				((self.cache_manager, False), "cache_manager")
			)
			with nvtx.annotate("get_load", color='yellow'):
				if spec_stream is not None:
					with torch.cuda.stream(spec_stream):
						prefetch_idx, buffer, skip_load = self.cache_manager.get_load_buffer(self.layer_id, prefetch_idx)
						if not skip_load:
							prefetch_idx_cpu = prefetch_idx.cpu()
					spec_stream.synchronize()
				else:
					prefetch_idx, buffer, skip_load = self.cache_manager.get_load_buffer(self.layer_id, prefetch_idx)
					if not skip_load:
						prefetch_idx_cpu = prefetch_idx.cpu()
			# assert n == self.cache_manager.shared_max_blocks_per_seq, f"n={n}, shared_max_blocks_per_seq={self.cache_manager.shared_max_blocks_per_seq}"
			if not skip_load:
				general_copy(buffer, ('cachemanager', self.cache_manager.scatter_mask), kv_home, prefetch_idx_cpu, key='load_kv')
			else:
				print("skip load kv for layer", self.layer_id, flush=True)
		else:
			# n, b, 2, ghd
			k_c_shape = (n, *self.prefetch_kv.shape[1:]) # n, b, 2, ghd
			k_c = TorchTensor(k_c_shape, kv_home.dtype, 
							self.prefetch_kv.data[:n], dst)
			cache_read_buf.store(
				((k_c, False), "nb2ghd")
			)
			if spec_stream is not None:
				with torch.cuda.stream(spec_stream):
					prefetch_idx_cpu = prefetch_idx.cpu()
				spec_stream.synchronize()
			else:
				prefetch_idx_cpu = prefetch_idx.cpu()
			general_copy(self.prefetch_kv, (slice(0, n), ), kv_home, prefetch_idx_cpu, key='load_kv')


	def store_cache(self, cache_home, cache_write_buf, i, sync_func):
		
		if self.policy.use_token_cache and self.layer_id == 0:
			return
		
		if i == self.task.gen_len - 1:  # last token, no need to store cache
			return

		# if self.env.gpu is None:
		#     assert k_new.dtype == torch.float32, f'k_new.dtype={k_new.dtype}'
		#     k_new = k_new.to(kv_home.dtype)
		#     assert v_new.dtype == torch.float32, f'v_new.dtype={v_new.dtype}'
		#     v_new = v_new.to(kv_home.dtype)

		if self.lr_k is None: # load whole cache

			kv_home = cache_home.val
			k_new, v_new = cache_write_buf.pop()
			sync_func()
			# k_new/v_new: b, s, h*d
			if i == 0:
				indices = (slice(0, k_new.shape[0]),
							slice(0, k_new.shape[1]))    
				# (b, s, hd) (b, s, hd) -> (b, s, 2hd)
				general_copy(kv_home, indices, (k_new, v_new), None, key='store_kv')
			else:
				indices = (slice(0, k_new.shape[0]), 
						   slice(self.task.prompt_len + i - 1, self.task.prompt_len + i))
				# (b, 1, hd) (b, 1, hd) -> (b, 1, 2hd)
				general_copy(kv_home, indices, (k_new.reshape(k_new.shape[0], 1, -1), v_new.reshape(v_new.shape[0], 1, -1)), None, key='store_kv')
			return 


		token_group = self.policy.token_group
   
		if i == 0:  # prefill
			# k_new/v_new: b, s, h*d
			kv_home = cache_home.val
			k_new, v_new = cache_write_buf.pop()
   
			save_idx = k_new.shape[1] - k_new.shape[1] % token_group
			assert save_idx % token_group == 0, f"{save_idx} % {token_group} != 0"
			indices = (slice(0, k_new.shape[0]),
						slice(0, save_idx//self.policy.token_group))    
			lr_kcache_func(self.lr_proj_mode, self.lr_k, k_new[:, :save_idx], (0, save_idx), 
						   self.config.num_kv_heads, self.config.num_kv_groups, self.partial_index, self.skew_matrix, self.lr_kproj)
			#  b, s0, h*d
			k_new_last = k_new[:, save_idx:]
			if k_new_last.numel() > 0:
				s0 = k_new_last.shape[1]
				v_new_last = v_new[:, save_idx:]
				if self.policy.paged_att:
					k_new_last = k_new_last.reshape(-1, self.config.num_kv_heads, self.config.head_dim)
					v_new_last = v_new_last.reshape(-1, self.config.num_kv_heads, self.config.head_dim)
					self.cache_manager.update_rolling_buffer(k_new_last, v_new_last, self.layer_id, True, s0)
				else:
					# b, s0, h*d -> s0, b, hd
					self.window_kv_buf[0, :s0].copy_(k_new_last.transpose(0, 1), non_blocking=True)
					self.window_kv_buf[1, :s0].copy_(v_new_last.transpose(0, 1), non_blocking=True)
			self.kv_window_index = k_new.shape[1] % token_group
			# (b, s, hd) (b, s, hd) -> (b, s//g, 2ghd)
			sync_func()
			if self.policy.paged_att:
				if self.layer_id == 0:
					self.cache_manager.init_remaining(k_new.shape[1])
				k_new, v_new = self.cache_manager.reshape(k_new[:, :save_idx], v_new[:, :save_idx], save_idx)
			else:
				k_new = k_new[:, :save_idx].reshape(k_new.shape[0], save_idx//token_group, -1)
				v_new = v_new[:, :save_idx].reshape(v_new.shape[0], save_idx//token_group, -1)
			general_copy(kv_home, indices, (k_new, v_new), None, key='store_kv')
				   
		else:  # decoding         
			kv_home = cache_home.val   
			self.kv_window_index = self.kv_window_index + 1
			if self.kv_window_index == token_group:
				self.kv_window_index = 0
				pos = self.task.prompt_len + i
				assert pos % token_group == 0, f"{pos} % {token_group} != 0"
				pos_g = pos // token_group
				indices = (slice(0, self.policy.gpu_batch_size), 
						   slice(pos_g - 1, pos_g))
				sync_func()
				# (2, g, b, hd) -> (b, 2, g, hd) -> (b, 1, 2, ghd)
				if self.policy.paged_att:
					recentkv, k = self.cache_manager.get_recentkv(self.layer_id)
					general_copy(kv_home, indices, recentkv, None, key='store_kv')
					lr_kcache_func(self.lr_proj_mode, self.lr_k, 
								################################
								# for no_rb
								# k[:, :token_group],
								k,
								################################
								(pos-token_group, pos), self.config.num_kv_heads, self.config.num_kv_groups,
									self.partial_index, self.skew_matrix, self.lr_kproj)      
				else:
					general_copy(kv_home, indices, 
								self.window_kv_buf.permute(2, 0, 1, 3).reshape(self.window_kv_buf.shape[2], 1, 2, -1),
								None, key='store_kv')
					# g, b, hd -> b, g, hd  
					lr_kcache_func(self.lr_proj_mode, self.lr_k, 
								self.window_kv_buf[0].transpose(0, 1),
								(pos-token_group, pos), self.config.num_kv_heads, self.config.num_kv_groups,
									self.partial_index, self.skew_matrix, self.lr_kproj)      

	@torch.inference_mode()
	def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, pos_emb, past_pos_emb,
				cache_write_buf, i, k, spec_stream, prefetch_event, prefetch_sync,
				next_partial_index, next_partial_wq, next_skew_matrix, next_lr_kproj, next_qproj, next_qnorm, next_lr_k):

		donate = [False] * 15
		h, donate[0] = hidden.val, True

		if k == self.policy.num_gpu_batches - 1:
			# Clear the weight_read_buf if it is the last gpu batch
			((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
					(w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
					(w_ln, donate[10]), (b_ln, donate[11]), (q_norm, donate[12]), (k_norm, donate[13])) = weight_read_buf.pop()
		else:
			((w_q, _), (b_q, _), (w_k, _), (b_k, _),
				(w_v, _), (b_v, _), (w_out, _), (b_out, _), (w_ln, _), 
	 				(b_ln, _), (q_norm, _), (k_norm, _)) = weight_read_buf.val           
				
		w_tuple = (w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, q_norm, k_norm)
		
		if i == 0:  # prefill
			mask, donate[1] = attention_mask.val.smart_copy(self.compute)
			h, new_k_cache, new_v_cache = self.compute.mha(h, mask, *w_tuple, 
															self.config.num_attention_heads, 
															self.config.head_dim, self.config.scaling, 
															self.config.num_kv_groups, pos_emb, 
															self.lr_proj_mode,
															donate, self.policy.compress_cache, self.policy.comp_cache_config, 
															self.policy.flash_att, self.chunk_size)
						
			if not (self.policy.use_token_cache and self.layer_id == 0):
				cache_write_buf.store((new_k_cache, new_v_cache))

		else:  # decoding
			mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
			if self.policy.use_token_cache and self.layer_id == 0:
				# for 1st att, kv is replace by token, this trade comp for io. since 1st att KV loading is always full.
				kv_cache = TorchTensor.create_from_torch(h.data[:, :-1].contiguous(), h.device)
				h = TorchTensor.create_from_torch(h.data[:, -1:].contiguous(), h.device)
				kv_layout = 'bs(nd)'
			else:
				(kv_cache, donate[14]), kv_layout = cache_read_buf.pop()
			if next_lr_k is not None:
				len_lr_k = self.task.prompt_len + i - 1 
				len_lr_k = len_lr_k - len_lr_k % self.policy.token_group
				next_lr_k_input = next_lr_k[:len_lr_k]
			else:
				next_lr_k_input = None
			if kv_layout == 'cache_manager':
				if self.layer_id == 0:
					kv_cache.update_seqlens()
				kv_cache.layer_id = self.layer_id
			h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, *w_tuple, 
																self.config.num_attention_heads, self.config.head_dim, self.config.scaling, 
																self.config.num_kv_groups, pos_emb, past_pos_emb,
																self.enable_pred, self.lr_proj_mode, next_lr_kproj, next_qproj, next_qnorm,
																next_partial_index, next_partial_wq, next_skew_matrix,
																kv_cache, next_lr_k_input, self.window_kv_buf, self.kv_window_index,
																kv_layout, self.policy.token_group, 
																self, donate,
																self.policy.compress_cache, self.policy.comp_cache_config, 
																spec_stream, prefetch_event, prefetch_sync,
																self.policy.alpha, self.policy.max_num_kv, 
																self.policy.att_score_mode, att_comp_mode=self.policy.att_comp_mode)
			
			if self.lr_k is None and not (self.policy.use_token_cache and self.layer_id == 0):
				cache_write_buf.store((new_k_cache, new_v_cache))

		hidden.val = h
	

class MLP:
	def __init__(self, config, env, policy, layer_id):
		self.config = config
		self.env = env
		self.layer_id = layer_id
		self.policy = policy
		self.compute = self.env.gpu if self.env.gpu is not None else self.env.cpu
		self.weight_load_dst = self.compute
		self.task = None

	def set_task(self, task):
		self.task = task
		bsz = len(task.inputs)
		plen = task.prompt_len
		self.chunk_size = 32 * 1024
		if bsz * plen > 8 * 1024:
			prop_iter = bsz * plen // (8*1024)
			self.chunk_size = bsz * plen // prop_iter

		if self.layer_id == 0:
			num_chucks = (plen + self.chunk_size - 1) // self.chunk_size
			print(f"MLP setting chunk size={self.chunk_size}, num_chucks={num_chucks}")


	def init_weight(self, weight_home, path):
		h, dtype = (self.config.hidden_size, self.config.dtype)
		if self.config.model_type == 'opt':
			path = os.path.join(os.path.join(path, f"model.decoder.layers.{self.layer_id}."))
			weight_specs = [
				((4 * h, h), dtype, path + "fc1.weight"),
				((4 * h,), dtype, path + "fc1.bias"),
				((h, 4 * h), dtype, path + "fc2.weight"),
				((h,), dtype, path + "fc2.bias"),
				((h,), dtype, path + "final_layer_norm.weight"),
				((h,), dtype, path + "final_layer_norm.bias"),
			]
		else:
			path = os.path.join(os.path.join(path, f"model.layers.{self.layer_id}."))
			intermediate_size = self.config.intermediate_size
			weight_specs = [
				((intermediate_size, h), dtype, path + "mlp.gate_proj.weight"),
				((intermediate_size, h), dtype, path + "mlp.up_proj.weight"),
				((h, intermediate_size), dtype, path + "mlp.down_proj.weight"),
				((h,), dtype, path + "post_attention_layernorm.weight"),
			]            
		weights = init_weight_list(weight_specs, self.policy, self.env, self.config)
		weight_home.store(weights)

	def load_weight(self, weight_home, weight_read_buf, k):
		if k == 0:
			dst1 = self.weight_load_dst
			dst2 = self.compute
			if self.config.model_type == 'opt':
				wi, bi, wo, bo, w_ln, b_ln = weight_home.val
				w_store = (wi.smart_copy(dst1), bi.smart_copy(dst2),
						wo.smart_copy(dst1), bo.smart_copy(dst2),
						w_ln.smart_copy(dst2), b_ln.smart_copy(dst2))
			else:
				w_gate, w_up, w_down, w_norm = weight_home.val
				w_store = (w_gate.smart_copy(dst1), w_up.smart_copy(dst1), 
							w_down.smart_copy(dst1), w_norm.smart_copy(dst2))
			weight_read_buf.store(w_store)
				

	def init_cache_one_gpu_batch(self, cache_home):
		pass

	def load_cache(self, cache_home, cache_read_buf, i):
		pass  

	def store_cache(self, cache_home, cache_write_buf, i):
		pass

	@torch.inference_mode()
	def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
				cache_write_buf, i, k):
		donate = [False] * 7
		h, donate[0] = hidden.val, True

		if self.config.model_type == 'opt':
			if k == self.policy.num_gpu_batches - 1:
				# Clear the weight_read_buf if it is the last gpu batch
				((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
				(w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
			else:
				((wi, _), (bi, _), (wo, _), (bo, _),
				(w_ln, _), (b_ln, _)) = weight_read_buf.val
			h = self.compute.opt_mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
		else:
			if k == self.policy.num_gpu_batches - 1:
				# Clear the weight_read_buf if it is the last gpu batch
				((w_gate, donate[1]), (w_up, donate[2]), (w_down, donate[3]), (w_norm, donate[4])) = weight_read_buf.pop()
			else:
				((w_gate, _), (w_up, _), (w_down, _), (w_norm, _)) = weight_read_buf.val
			
			if i == 0:
				h = self.compute.llama_mlp(h, w_gate, w_up, w_down, w_norm, donate, 
							   act=self.config.hidden_act, chunk_size=self.chunk_size)      
			else:
				h = self.compute.llama_mlp_gen(h, w_gate, w_up, w_down, w_norm, donate, act=self.config.hidden_act)   
		hidden.val = h
		

class LM:
	def __init__(self,
				 config,
				 env: ExecutionEnv,
				 model_path: str,
				 policy: Policy, 
				 skew_paths, 
				 lr_proj_path
				 ):
		self.config = config
		self.env = env
		self.model_path = model_path
		self.policy = policy
		self.num_gpu_batches = policy.num_gpu_batches
		if '0' in policy.start_layer:
			if 'emb' in policy.start_layer:
				self.start_prefetch_layer = -1 # -1 means prefetch from embedding layer
			else:
				self.start_prefetch_layer = 0
			self.start_prefetch_layer2 = 0
			self.use_cur_hidden_forl0 = 'curr' in policy.start_layer
		else:
			self.start_prefetch_layer = 1
			self.start_prefetch_layer2 = 1
		assert self.start_prefetch_layer in [0, 1, -1]
		if self.config.model_type != 'opt':
			if self.config.model_type == 'llama3':
				from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as RotaryEmbedding
			else:
				from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as RotaryEmbedding
			# self.rotary_emb = RotaryEmbedding(dim=config.head_dim, 
											#   max_position_embeddings=config.max_position_embeddings, 
											#   base=config.rope_base, scaling_factor=1.0, device='cuda:0' if self.env.gpu is not None else 'cpu')
			self.rotary_emb = RotaryEmbedding(config=config, device='cuda:0' if self.env.gpu is not None else 'cpu')
		
		layers = []
		self.attn_layer = []		
  
		layers.append(InputEmbed(self.config, self.env, self.policy))
  
		for i in range(self.config.num_hidden_layers):
			if i == 0:
				enable_pred = self.start_prefetch_layer <= 0 and not self.use_cur_hidden_forl0
			else:
				enable_pred = i < self.config.num_hidden_layers - 1 
			layers.append(SelfAttention(self.config, self.env, self.policy, i, skew_paths, lr_proj_path, enable_pred))
			self.attn_layer.append(len(layers) - 1)
			layers.append(MLP(self.config, self.env, self.policy, i))

		layers.append(OutputEmbed(self.config, self.env, self.policy))
		self.layers = layers
		self.num_layers = len(layers)

		if self.policy.act_gpu_percent == 100:
			self.act_home = self.env.gpu
		elif self.policy.act_cpu_percent == 100:
			self.act_home = self.env.cpu
		elif self.policy.act_disk_percent == 100:
			self.act_home = self.env.disk
		else:
			raise NotImplementedError()

		try:
			self.speculation_stream = torch.cuda.Stream()
		except:
			self.speculation_stream = None
		# Intermediate tensors
		# The following buffers store values used
		# for the i-th token, j-th layer, k-th gpu batch.
		num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
		self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
		self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
		self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
		self.partial_cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
		self.weight_read_buf = array_1d(num_layers, ValueHolder)
		self.partial_weight_read_buf = array_1d(num_layers, ValueHolder)
		self.attention_mask = array_1d(num_gpu_batches, ValueHolder)
		self.pos_emb = None
		self.past_pos_emb = None
		self.task = None
		self.init_all_weights()

	def set_task(self, task):
		self.task = task
		for l in self.layers:
			l.set_task(task)

	def init_weight(self, j):
		expanded_path = os.path.abspath(os.path.expanduser(
			os.path.join(self.model_path, f"weights-np")))
		self.layers[j].init_weight(self.weight_home[j], expanded_path)

	def load_weight(self, i, j, k):
		# Handle corner cases
		if j == self.num_layers:
			j = 0
			i += 1
			if i == self.task.gen_len:
				return

		# Load from weight_home to weight_read_buf
		self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

	def delete_weight(self, j, k):
		if k == 0:
			for x in self.weight_home[j].pop():
				if isinstance(x, ValueHolder):
					for y in x.pop():
						y.delete()
				else:
					x.delete()

	def init_cache(self, j, k):
		self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

	def load_cache(self, i, j, k):
		if i == 0:  # prefill, no cache
			return
		
		if self.policy.use_token_cache and j == 1:
			return
				
		if self.policy.lr_proj_mode == 'none':
			self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
		elif j not in self.attn_layer[self.start_prefetch_layer+1:]:
			self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)


	def process_fetch_idx(self, prefetch_idx):
		# prefetch_idx: b, n
		# if self.policy.sort_idx: 
			# prefetch_idx = prefetch_idx.sort(dim=-1, descending=False, stable=False).values            
		return prefetch_idx
	
	def prefetch_cache(self, i, j, k, spec_stream=None):
		# Handle corner cases
		if i == 0:  # prefill, no cache
			return
		if self.policy.lr_proj_mode == 'none':
			return
		if j == 1 and self.start_prefetch_layer <= 0 and self.use_cur_hidden_forl0:
			return
		next_attn = self.attn_layer[self.attn_layer.index(j) + 1]
		prefetch_idx = self.layers[j].prefetch_idx
		self.layers[j].prefetch_idx = None
		with nvtx.annotate(f"selkvl{next_attn}", color="brown"):
			prefetch_idx = self.process_fetch_idx(prefetch_idx)
			self.layers[next_attn].prefetch_cache(self.cache_home[next_attn][k], self.cache_read_buf[next_attn][k], i, 
													prefetch_idx, spec_stream=spec_stream)

	def store_cache(self, i, j, k):
		if i == self.task.gen_len - 1:  # last token, no need to store cache
			self.cache_write_buf[j][k].pop()
			return

		if j in self.attn_layer: # is att                
			self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i, sync_func=partial(self.sync, 'store_kv', 'prefill' if i == 0 else 'decode'))

		if i == 0 and j == self.num_layers-1 and k == 0:
			# last layer in prefill
			self.sync('store_kv', 'prefill')
			start_time = time.time()
			self.env.disk.fstrim()
			print(f"fstrim time: {(time.time()-start_time)*1000:.1f} ms")
			self.env.disk.register_fd()
			

	def delete_cache(self, j, k):
		v = self.cache_home[j][k].pop()
		if v:
			if type(v) not in (list, tuple):
				v.delete()
			else:
				for x in v:
					if x is not None:
						x.delete()

	def load_hidden(self, i, j, k):
		# Load to hidden states buffers
		dst = self.layers[j].compute
		if j == 0:
			gpu_batch_size = self.policy.gpu_batch_size
			left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
			if i == 0:  # load from the input ids
				val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32, pin_memory=None)
				val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
			else:  # load from the last generated token
				pos = self.task.prompt_len + i
				val = dst.allocate((gpu_batch_size, 1), np.int32, pin_memory=None)
				val.load_from_np(self.output_ids[left:right, pos-1:pos])
		else:  # load from the last layer
			val = self.hidden[i][j-1][k].pop().move(dst)
		self.hidden[i][j][k].store(val)

	def store_hidden(self, i, j, k):
		if j == self.num_layers - 1:  # store to output
			gpu_batch_size = self.policy.gpu_batch_size
			left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
			ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
			pos = self.task.prompt_len + i
			if self.task.stop:
				stopped = self.stopped[left:right]
				self.output_ids[left:right, pos:pos+1] = np.where(
					stopped, self.config.pad_token_id, ids)
				stopped[:] = np.logical_or(stopped, ids == self.task.stop)
			else:
				self.output_ids[left:right, pos:pos+1] = ids
		else:  # move to home
			x = self.hidden[i][j][k]
			if x.val:  # x may already be moved due to overlapping
				x.val = x.val.move(self.act_home)
	
	@torch.inference_mode()
	def rope(self, i, j, k):
		if j != 1: 
			return
		if k != 0:
			return
		if self.config.model_type != 'opt':
			inputs_embeds = self.hidden[i][j][k].val.data
			if not hasattr(self, 'position_ids'):
				self.position_ids = torch.arange(0, self.task.prompt_len+self.task.gen_len, 
													device=inputs_embeds.device).unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
				cos_sin_cache = self.rotary_emb(inputs_embeds, self.position_ids[0:1])
				self.cos_sin_cache_vllm = torch.cat((cos_sin_cache[0][0, :, :64], cos_sin_cache[1][0, :, :64]), dim=-1)
				if self.policy.use_token_cache:
					self.cos_sin_cache = cos_sin_cache
	 
			if self.policy.use_token_cache:
				self.past_pos_emb = (self.cos_sin_cache[0][:, :self.task.prompt_len + i-1], 
									self.cos_sin_cache[1][:, :self.task.prompt_len + i-1])
			if i == 0:
				position_ids = self.position_ids[:, :self.task.prompt_len]
			else:
				position_ids = self.position_ids[:, self.task.prompt_len + i-1 : self.task.prompt_len + i]                
			self.pos_emb = (position_ids, self.cos_sin_cache_vllm)

	@torch.inference_mode()
	def fetch_kv(self, input, next_qproj, next_qnorm, next_lr_kproj, next_lr_kcache, next_partial_wq, next_skew_matrix, next_partial_index,
				 weight_read_buf, j, i, k):
		# fetch kv for att1/emb here.
		if i == 0:
			return

		b, tgt_s, h = input.shape
		dtype = input.dtype
		((w_q, _), (b_q, _), (w_k, _), (b_k, _),
				(w_v, _), (b_v, _), (w_out, _), (b_out, _), (w_ln, _), 
					(b_ln, _), (q_norm, _), (k_norm, _)) = weight_read_buf.val       
		if b_ln is not None:
			hidden = F.layer_norm(input, (h,), weight=w_ln.data.to(dtype), bias=b_ln.data.to(dtype))
		else:
			hidden = rms_norm(input, w_ln.data.to(dtype))
		if w_q.data.shape[1] == w_q.data.shape[0] + 1:
			# print(f"hidden.shape={hidden.shape}")
			hidden = torch.cat((hidden, torch.ones(b, 1, 1, dtype=dtype, device=hidden.device)), dim=-1)        

		len_lr_k = self.task.prompt_len + i - 1 
		len_lr_k = len_lr_k - len_lr_k % self.policy.token_group

		prefetch_idx = speculate_attention(self.policy.lr_proj_mode, hidden, next_partial_wq, next_skew_matrix, 
											next_qproj, next_qnorm, next_lr_kproj, next_lr_kcache[:len_lr_k], self.pos_emb,
											next_partial_index, self.config.scaling,
											self.config.num_attention_heads, self.config.num_kv_groups, 
											self.policy.alpha, self.policy.max_num_kv, self.policy.token_group,
											self.policy.att_score_mode)
		prefetch_idx = self.process_fetch_idx(prefetch_idx)
		self.layers[j].prefetch_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i, 
												prefetch_idx, spec_stream=None)
		self.sync('load_kv', 'fetch') # should sync here 


	def compute_layer(self, i, j, k, prefetch=False, prefetch_sync=False):     
		if isinstance(self.layers[j], SelfAttention):            
			if self.policy.lr_proj_mode != 'none' and \
   								((j == 3 and self.start_prefetch_layer <= 0 and self.use_cur_hidden_forl0) \
								or (j == 1 and self.start_prefetch_layer == -1)):
				self.fetch_kv(self.hidden[i][j][k].val.data, self.layers[j].qproj, self.layers[j].qnorm, 
							self.layers[j].lr_kproj, self.layers[j].lr_k,
							self.layers[j].partial_wq, self.layers[j].skew_matrix, self.layers[j].partial_index, 
							self.weight_read_buf[j], j, i, k)
			next_att_idx = j + 2
			if self.policy.lr_proj_mode != 'none' and next_att_idx in self.attn_layer:
				next_partial_index = self.layers[next_att_idx].partial_index
				next_partial_wq = self.layers[next_att_idx].partial_wq
				next_skew_matrix = self.layers[next_att_idx].skew_matrix
				next_qproj = self.layers[next_att_idx].qproj
				next_lr_kproj = self.layers[next_att_idx].lr_kproj
				next_lr_k = self.layers[next_att_idx].lr_k
				next_qnorm = self.layers[next_att_idx].qnorm
			else:
				next_partial_index = next_partial_wq = next_skew_matrix = next_qproj = next_lr_k = next_lr_kproj = next_qnorm = None                
			self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
				self.weight_read_buf[j], self.attention_mask[k], self.pos_emb, self.past_pos_emb,
				self.cache_write_buf[j][k], i, k, self.speculation_stream,
				partial(self.prefetch_cache, i, j, k, self.speculation_stream) if prefetch else None, 
				partial(self.sync, 'load_kv', 'prefetch') if prefetch_sync else None, 
				next_partial_index, next_partial_wq, next_skew_matrix, next_lr_kproj, next_qproj, next_qnorm, next_lr_k)
		else:
			self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
				self.weight_read_buf[j], self.attention_mask[k],
				self.cache_write_buf[j][k], i, k)

	def sync(self, key='all', key2=''):
		if key == 'all':
			self.env.disk.synchronize('all')
		elif key == 'load_kv':
			self.env.disk.synchronize('load_kv', count_time_key=key+'-'+key2)
		elif key == 'store_kv':
			self.env.disk.synchronize('store_kv', count_time_key=key+'-'+key2)
		elif key == 'spec_att_stream':
			if self.speculation_stream is not None:
				self.speculation_stream.synchronize()
		elif key == 'main_compute':
			if self.speculation_stream is not None:
				self.speculation_stream.synchronize()
				torch.cuda.default_stream().synchronize()
		else:
			raise ValueError(f"Invalid mode: {key}")
		
	def init_all_weights(self):
		self.weight_home = array_1d(self.num_layers, ValueHolder)
		for j in range(self.num_layers):
			self.init_weight(j)

	def delete_all_weights(self):
		for j in range(self.num_layers):
			self.delete_weight(j, 0)

	@torch.inference_mode()
	def update_attention_mask(self, i, k):
		if i > 0:
			mask = self.attention_mask[k]
			assert mask.val is not None
			mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
			return

		gpu_batch_size = self.policy.gpu_batch_size
		left = k * gpu_batch_size
		right = left + gpu_batch_size
		input_ids = self.output_ids[left:right, :self.task.prompt_len]

		attention_compute = self.env.gpu if self.env.gpu is not None else self.env.cpu
		val = attention_compute.allocate((self.policy.gpu_batch_size, self.task.prompt_len), bool, pin_memory=None)
		val.load_from_np((input_ids != self.config.pad_token_id))
		self.attention_mask[k].store(val)

	def print_stats(self, task, run_info):
		# get_swap_num
		num_prompts = len(task.inputs)
		prompt_len = task.prompt_len
		swap_info_list = self.env.disk.get_swap_info()
		if len(swap_info_list) > 0:
			layer_avg_swap_num = []
			layer_avg_swap_time = []
			layer_avg_flush_num = []
			layer_avg_flush_time = []
			layer_prefill_time = []
			for layer_name, swap_num_array, swap_time_array, flush_num_array, flush_time_array, prefill_wr_time_array, swap_indices_list in swap_info_list:
				assert len(prefill_wr_time_array) == 1, f"prefill_wr_time_array={prefill_wr_time_array}"
				layer_prefill_time.append(prefill_wr_time_array[0])
				flush_num_array = flush_num_array[1:]
				flush_time_array = flush_time_array[1:]
				avg_swap_num = np.mean(swap_num_array)
				avg_swap_time = np.mean(swap_time_array)
				avg_flush_num = np.mean(flush_num_array)
				avg_flush_time = np.mean(flush_time_array)
				layer_avg_swap_num.append(avg_swap_num)
				layer_avg_swap_time.append(avg_swap_time)
				layer_avg_flush_num.append(avg_flush_num)
				layer_avg_flush_time.append(avg_flush_time)
				avg_swap_size = cache_bytes(self.config, num_prompts, avg_swap_num, dtype_size=2, num_layers=1)
				avg_swap_size = avg_swap_size / 1024 / 1024 # MB
				avg_bw = avg_swap_size / avg_swap_time
				avg_flush_size = cache_bytes(self.config, num_prompts, avg_flush_num, dtype_size=2, num_layers=1)
				avg_flush_size = avg_flush_size / 1024 / 1024 # MB
				avg_flush_bw = avg_flush_size / avg_flush_time
				layer_name = [layer_name_.split('/')[-1] for layer_name_ in layer_name]
				print(f"{layer_name}:")
				print(f"\tSwap: {swap_num_array.shape}, avg_num: {avg_swap_num:.1f}, avg_time: {avg_swap_time*1000:.1f} ms, avg_size: {avg_swap_size:.1f} MB, avg_bw: {avg_bw:.1f} MB/s")
				print(f"\tFlush: {flush_num_array.shape}, avg_num: {avg_flush_num:.1f}, avg_time: {avg_flush_time*1000:.1f} ms, avg_size: {avg_flush_size:.1f} MB, avg_bw: {avg_flush_bw:.1f} MB/s")
			sum_swap_num = np.sum(layer_avg_swap_num)
			sum_swap_time = np.sum(layer_avg_swap_time)
			sum_swap_size = cache_bytes(self.config, num_prompts, sum_swap_num, dtype_size=2, num_layers=1)
			sum_swap_size = sum_swap_size / 1024 / 1024 # MB
			sum_swap_bw = sum_swap_size / sum_swap_time
			sum_flush_num = np.sum(layer_avg_flush_num)
			sum_flush_time = np.sum(layer_avg_flush_time)
			sum_flush_size = cache_bytes(self.config, num_prompts, sum_flush_num, dtype_size=2, num_layers=1)
			sum_flush_size = sum_flush_size / 1024 / 1024 # MB
			sum_flush_bw = sum_flush_size / sum_flush_time
			sum_prefill_time = np.sum(layer_prefill_time)
			sum_prefill_size = cache_bytes(self.config, num_prompts, prompt_len, dtype_size=2, num_layers=len(layer_prefill_time)) / 1024 / 1024 # MB
			sum_prefill_bw = sum_prefill_size / sum_prefill_time
			print(f"Prefill size: {sum_prefill_size:.1f} MB, prefill time: {sum_prefill_time*1000:.1f} ms, prefill bw: {sum_prefill_bw:.1f} MB/s")
			print(f"Sum swap num: {sum_swap_num:.1f}, sum swap time: {sum_swap_time*1000:.1f} ms, sum_swap_size: {sum_swap_size:.1f} MB, sum_swap_bw: {sum_swap_bw:.1f} MB/s")
			print(f"Sum flush num: {sum_flush_num:.1f}, sum flush time: {sum_flush_time*1000:.1f} ms, sum_flush_size: {sum_flush_size:.1f} MB, sum_flush_bw: {sum_flush_bw:.1f} MB/s")
			if run_info != 'none':
				save_name = f"{run_info}_swap_info.pt"
				torch.save(swap_info_list, save_name)
				print(f"Swap info saved to {save_name}")
		else:
			print("No swap")

		global cache_manager_inst
		if cache_manager_inst is not None:
			reuse_info = cache_manager_inst.reuse_info
			layer_reuse = []
			for layer_name, reuse_list in reuse_info.items(): 
				total_access = len(reuse_list)
				reuse_rate = sum(reuse_list) / total_access if total_access > 0 else 0
				print(f"{layer_name}: reuse_rate: {reuse_rate:.2f}, tot_access: {total_access}")
				layer_reuse.append(reuse_rate)
			avg_reuse_rate = sum(layer_reuse) / len(layer_reuse) 
			print(f"Average reuse rate: {avg_reuse_rate:.2f}")
		else:
			print("No reuse")        

		disk_sync_time_dict = self.env.disk.disk_sync_time_dict
		if len(disk_sync_time_dict) > 0:
			t = 0
			print("="*50)
			for key, value in disk_sync_time_dict.items():
				if key == 'store_kv-prefill':
					value_sum = np.sum(value)
					print(f"{key}: {value_sum*1000:.1f} ms, len={len(value)}")
				else:
					if key == 'load_kv-prefetch': # filter out timeout 
						self.timeout_count = (np.array(value) > 30).sum()
						if self.timeout_count > 0:
							print(f"{key}: {self.timeout_count} timeouts")
					value_sum = np.sum(value)
					print(f"{key}: {value_sum*1000:.1f} ms, len={len(value)}")
					t += value_sum
			print(f"Decoding disk sync time: {t*1000:.1f} ms")
		else:
			print("No disk sync time")

	@torch.inference_mode()
	def warmup(self):
		# global prefetch_kv_buf
		# dtype = prefetch_kv_buf.data.dtype
		# device = prefetch_kv_buf.data.device
		dtype = torch.bfloat16
		device = 'cuda'
		bsz = len(self.task.inputs)
		print("Warmup started", flush=True)
		a = torch.randn(bsz, 1024, 1024).to(dtype).to(device)
		b = torch.randn(bsz, 1024, 1024).to(dtype).to(device)
		for _ in range(100):
			torch.bmm(a, b)
		del a, b
		torch.cuda.synchronize()
		print("Warmup done", flush=True)

	def generate(self,
				 inputs: Union[np.array, List[List[int]]],
				 max_new_tokens: int = 32,
				 do_sample: bool = False,
				 temperature: float = 1.0,
				 stop: Optional[int] = None,
				 run_info = 'none'):
		task = Task(
			inputs=inputs,
			prompt_len=len(inputs[0]),
			gen_len=max_new_tokens,
			do_sample=do_sample,
			temperature=temperature,
			stop=stop,
		)
		num_layers = self.num_layers
		num_gpu_batches = self.num_gpu_batches
		gpu_batch_size = self.policy.gpu_batch_size
		prompt_len, gen_len = task.prompt_len, task.gen_len

		# Output token ids
		self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
			self.config.pad_token_id, dtype=np.int32)
		self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
		self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
		assert gpu_batch_size * num_gpu_batches == len(task.inputs)
		num_prompts = len(task.inputs)
		# Intermediate tensors
		# The following buffers store values used
		# for the i-th token, j-th layer, k-th gpu batch.
		num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
		for j in range(num_layers):
			for k in range(num_gpu_batches):
				self.cache_home[j][k].clear()
				self.cache_read_buf[j][k].clear()
				self.cache_write_buf[j][k].clear()
				self.partial_cache_read_buf[j][k].clear()
		for j in range(num_layers):
			self.weight_read_buf[j].clear()
			self.partial_weight_read_buf[j].clear()
		for k in range(num_gpu_batches):
			self.attention_mask[k].clear()
		self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

		# Init cache
		self.set_task(task)
		if self.task.gen_len > 1:
			for j in range(num_layers):
				for k in range(num_gpu_batches):
					self.init_cache(j, k)

		gc.collect()
		if self.env.gpu is not None:
			torch.cuda.empty_cache()

		self.sync()
		if self.policy.en_copy_key:
			self.generation_loop_normal_new()
		else:
			self.generation_loop_normal()

		if max_new_tokens > 1:
			self.print_stats(task, run_info)
		else:
			assert len(self.env.disk.disk_sync_time_dict) == 0, f"{len(self.env.disk.disk_sync_time_dict)}"
	  
		# Delete cache
		for j in range(num_layers):
			for k in range(num_gpu_batches):
				self.delete_cache(j, k)
				
		if hasattr(self, 'position_ids'):
			del self.position_ids
			
		return self.output_ids

	def layer_type(self, i, j):
		if i == 0:
			pre = 'Pre'
		else:
			pre = ''
		if j == 0:
			return pre+"In"
		elif j == self.num_layers - 1:
			return pre+"Out"
		elif j in self.attn_layer:
			return pre+"Att"
		else:
			return pre+"MLP"

	@torch.inference_mode()
	def generation_loop_normal_new(self):
		for i in tqdm(range(self.task.gen_len)):
			if i == 1:
				self.warmup()
			timers("generate").start()
			for k in range(self.num_gpu_batches):
				self.update_attention_mask(i, k)
			for j in range(self.num_layers):
				with nvtx.annotate(f"layer_{j}", color="blue"):
					for k in range(self.num_gpu_batches):
						self.load_weight(i, j, k)
					for k in range(self.num_gpu_batches):
						self.load_cache(i, j, k)
						self.load_hidden(i, j, k)
						self.rope(i, j, k)
						if i > 0 and j in self.attn_layer:                       
							if self.policy.use_token_cache and j == 1:
								pass
							elif self.policy.lr_proj_mode != 'none' and j == 3 and self.start_prefetch_layer <= 0 and self.use_cur_hidden_forl0:
								pass	
							elif j in self.attn_layer[:self.start_prefetch_layer+1] or not self.policy.en_ahead_prefetch:
								self.sync('load_kv', 'prefetch')
							elif not self.policy.en_finer_prefetch_sync:
								self.sync('load_kv', 'prefetch')
							elif self.policy.en_finer_prefetch_sync and j == self.attn_layer[-1]:
								self.sync('load_kv', 'prefetch')                                                            
						with nvtx.annotate(f"Comp{self.layer_type(i, j)}", color="red"):
							if self.policy.en_ahead_prefetch and (j in self.attn_layer[self.start_prefetch_layer2:-1] and i > 0):
								self.compute_layer(i, j, k, True, self.policy.en_finer_prefetch_sync)
							else:    
								self.compute_layer(i, j, k)
						self.store_hidden(i, j, k)
						self.store_cache(i, j, k)
						if self.policy.en_ahead_prefetch == False and j in self.attn_layer[self.start_prefetch_layer2:-1] and i > 0:
							self.prefetch_cache(i, j, k, self.speculation_stream)
							
			timers("generate").stop()

	@torch.inference_mode()
	def generation_loop_normal(self):
		for i in tqdm(range(self.task.gen_len)):
			if i == 1:
				self.warmup()
			timers("generate").start()
			for k in range(self.num_gpu_batches):
				self.update_attention_mask(i, k)
			for j in range(self.num_layers):
				with nvtx.annotate(f"layer_{j}", color="blue"):
					for k in range(self.num_gpu_batches):
						self.load_weight(i, j, k)
					for k in range(self.num_gpu_batches):
						self.load_cache(i, j, k)
						self.load_hidden(i, j, k)
						self.rope(i, j, k) 
						if j in self.attn_layer and i > 0: 
							self.sync()
						with nvtx.annotate(f"Comp{self.layer_type(i, j)}", color="red"):
							self.compute_layer(i, j, k)         
						self.store_hidden(i, j, k)
						self.store_cache(i, j, k)
						if j in self.attn_layer[self.start_prefetch_layer2:-1] and i > 0:
							self.prefetch_cache(i, j, k, self.speculation_stream)
			timers("generate").stop()

	def __del__(self):
		self.delete_all_weights()


# def get_inputs(prompt_len, num_prompts, tokenizer, path, model_type):
# 	if model_type == 'llama3':
# 		model_dir = 'Llama'
# 	elif 'qwen' in model_type:
# 		model_dir = 'Qwen3'
# 	else:
# 		raise NotImplementedError()
# 	dataset = load_dataset("json", data_files=f'{path}/{model_dir}/synthetic/32768/data/niah_single_1/validation.jsonl', split='train')
# 	tokenized_prompts = []
# 	for i in range(num_prompts):
# 		input_text = dataset[i]['input'] 
# 		input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
# 		if input_ids.shape[1] < prompt_len:
# 			input_text += 'nice to meet you! ' * 500
# 			input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
# 		assert input_ids.shape[1] >= prompt_len, f"Input length {input_ids.shape} is less than prompt length {prompt_len}"
# 		tokenized_prompts.append(input_ids[0, :prompt_len])
# 	return tokenized_prompts

qmsum_prompt_format =  "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"
musique_prompt_format =  "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"


def get_inputs(prompt_len, num_prompts, tokenizer, path, model_type, seed):
	random.seed(seed)
	sys_prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct."
	dataset = "qmsum"  # Change this to the desired dataset
	# dataset = "musique"  # Change this to the desired dataset
	prompt_format = qmsum_prompt_format if dataset == "qmsum" else musique_prompt_format

	filename = f'{path}/{dataset}.jsonl'            
	data_all = [json.loads(line) for line in open(filename, encoding='utf-8')]
	tokenized_prompts = []
	for _ in range(num_prompts):
		context = ''
		while True:
			data = random.choice(data_all)
			context += data['context']
			context += '\n\n'
			prompt = prompt_format.format(context=context, input=data['input'])
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


def run_flexgen(args):

	tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
	if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		
	num_prompts = args.num_gpu_batches * args.gpu_batch_size

	args.en_cpu_relay = False # disable to save memory
	args.en_map_dict = False
	args.en_copy_key = False
	args.en_ahead_prefetch = False    
	args.en_finer_prefetch_sync = False
	run_level = int(args.run_args[1])
	########################################################
	if run_level >= 1:
		args.en_map_dict = True
	if run_level >= 2:
		args.en_copy_key = True
	if run_level >= 3:
		args.en_ahead_prefetch = True
	if run_level >= 4:
		args.en_finer_prefetch_sync = True
	########################################################
	config = get_model_config(args.model_path)
	########################################################
	# Task and policy
	# args.warmup_input_path = args.test_input_path if args.warmup_input_path is None else args.warmup_input_path
	# warmup_inputs = get_inputs(128, num_prompts, tokenizer, args.warmup_input_path, config.model_type)
	inputs = get_inputs(args.prompt_len, num_prompts, tokenizer, args.test_input_path, config.model_type, seed=args.seed)

	if args.cpu_only:
		gpu = None
		assert args.flash_att == 0, "Flash attention is not supported in CPU only mode"
		assert args.nv_profile == 0, "NV Profiler is not supported in CPU only mode"
		args.percent = [0, 100, 0, 0, 0, 100]
	else:
		gpu = TorchDevice("cuda:0")
	cpu = TorchDevice("cpu")
	
	# np.lib.format.ARRAY_ALIGN = 64
	np.lib.format.ARRAY_ALIGN = 4096
	print(f"np.lib.format.ARRAY_ALIGN={np.lib.format.ARRAY_ALIGN}")
	

	disk_dev_name = args.disk_dev_name.split(',')
 
	try:
		batch_split = [float(x) for x in args.batch_split.split(',')]
	except:
		batch_split = []
		
	if len(batch_split) <= 1:
		batch_split = None
		assert len(disk_dev_name) == 1, f"{disk_dev_name}"
	else:
		batch_split = torch.tensor(batch_split)
		batch_split = batch_split / batch_split.sum()
		assert len(batch_split) == len(disk_dev_name), f"{batch_split} {disk_dev_name}"		
  
	print(f"batch_split={batch_split}", flush=True)
	print(f"disk_dev_name={disk_dev_name}", flush=True)

	layer_num = config.num_hidden_layers
	if args.use_token_cache:
		layer_num -= 1
  
	if args.lr_proj_mode == 'none':
		args.token_group = 1
		args.max_num_kv = 0
 
	disk_cfg = {'interface':(args.dk_rd, args.dk_wr), 
				'common_cfg': (args.max_num_kv, args.token_group, layer_num), 
				'disk_dev_name': disk_dev_name}
	
	disk = TorchDisk(args.offload_dir, num_copy_threads=2, 
					 en_cpu_relay=args.en_cpu_relay, en_map_dict=args.en_map_dict, 
					 en_copy_key=args.en_copy_key, cfg=disk_cfg)
	
	env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk)

	policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
					args.percent[0], args.percent[1],
					args.percent[2], args.percent[3],
					args.percent[4], args.percent[5],
					args.pin_weight,
					args.compress_cache,
					CompressionConfig(num_bits=4, group_size=64,
									  group_dim=2, symmetric=False), 
					args.max_num_kv, bool(args.sort_idx), 
					bool(args.en_copy_key), bool(args.en_finer_prefetch_sync), 
					bool(args.en_ahead_prefetch), args.start_layer, 
	 				bool(args.flash_att), bool(args.paged_att),
					args.reuse_budget,
	 				args.att_score_mode, args.lr_proj_mode, 
					bool(args.use_token_cache), batch_split, 
					args.alpha, bool(args.use_mmap), args.token_group, args.att_comp_mode)

	cache_size = cache_bytes(config, num_prompts, args.prompt_len + args.gen_len, dtype_size=2)
	hidden_size = hidden_bytes(config, num_prompts, args.prompt_len + args.gen_len, dtype_size=2)
	print("Cache size: {:.2f} MB".format(cache_size / 1024 / 1024))
	print("Hidden size: {:.2f} MB".format(hidden_size / 1024 / 1024))
	
	if args.lr_proj_mode == 'base':
		assert args.skew_partial_idx_path != 'none', "skew_partial_idx_path must be provided"
		assert args.skew_matrix_path != 'none', "skew_matrix_path must be provided"
		# find skew_partial_idx_path
		skew_partial_idx_path_dirname = os.path.dirname(args.skew_partial_idx_path)
		skew_partial_idx_path_basename = os.path.basename(args.skew_partial_idx_path)
		list_fname = [f for f in os.listdir(skew_partial_idx_path_dirname) if f.startswith(skew_partial_idx_path_basename)]
		assert len(list_fname) == 1, f"list_fname = {list_fname}"
		args.skew_partial_idx_path = os.path.join(skew_partial_idx_path_dirname, list_fname[0])
		print("Setting skew_partial_idx_path to", args.skew_partial_idx_path)
		
	skew_paths = (args.skew_partial_idx_path, args.skew_matrix_path)
	model = LM(config, env, args.model_path, policy, skew_paths, args.lr_proj_path)

	try:        
		# print("Warming up...")
		# output_ids = model.generate(
		# 	warmup_inputs, max_new_tokens=1)
		if gpu is not None:
			torch.cuda.synchronize()
		print("Start testing...")
		timers("generate").reset()
		if args.nv_profile and not args.cpu_only:
			torch.cuda.cudart().cudaProfilerStart()
		output_ids = model.generate(
			inputs, max_new_tokens=args.gen_len, run_info=args.run_info
			# do_sample = True,
			# temperature = 0.6
			)
		if args.nv_profile and not args.cpu_only:
			torch.cuda.cudart().cudaProfilerStop()
			torch.cuda.synchronize()
		costs = timers("generate").costs
	finally:
		env.close_copy_threads()

	outputs = tokenizer.batch_decode(output_ids[:, args.prompt_len:], skip_special_tokens=True)
	show_str = ''
	show_str += "Outputs:\n" + 70 * '-' + "\n"
	for i in [0, len(outputs)-1]:
		show_str += f"{i}: {outputs[i]}\n"
		show_str += "-" * 70 + "\n"
	print(show_str)

	# Log output
	prefill_latency = costs[0]
	prefill_throughput = num_prompts * args.prompt_len / prefill_latency
	decode_latency = sum(costs[1:])
	if hasattr(model, "timeout_count") and model.timeout_count > 0:
		print(f"Warning: {model.timeout_count} timeouts during decoding")
		print(f"Adjusted decode latency: {decode_latency:.2f} -> {decode_latency-model.timeout_count * 30:.2f} s")
		decode_latency -= model.timeout_count * 30	
 
	decode_throughput = num_prompts * (args.gen_len - 1) / max(decode_latency, 1e-10)
	num_generated_tokens = num_prompts * args.gen_len
	total_latency = prefill_latency + decode_latency
	total_throughput = num_generated_tokens / total_latency
	
	
	print("+++++++++++++++++++++++++++++++++++++++++++++++++")
	print("input: " + str(args.prompt_len) + " output: " + str(args.gen_len) + " bsz: " + str(num_prompts))
	print("+++++++++++++++++++++++++++++++++++++++++++++++++")
	print("Latency Total: " + str(total_latency) + " Prefill: " + str(prefill_latency) + " Decode: " + str(decode_latency))
	print(f"Throughput Total: {total_throughput:.2f} Prefill: {prefill_throughput:.2f} Decode: {decode_throughput:.2f}")
	print("=================================================")


def add_parser_arguments(parser):
	parser.add_argument("--model_path", type=str)    
	parser.add_argument("--offload_dir", type=str)
	parser.add_argument("--prompt_len", type=int, default=512)
	parser.add_argument("--gen_len", type=int, default=32)
	parser.add_argument("--gpu_batch_size", type=int, default=4)
	parser.add_argument("--num_gpu_batches", type=int, default=1)
	parser.add_argument("--percent", nargs="+", type=int,
		default=[100, 0, 100, 0, 100, 0],
		help="Six numbers. They are "
		 "the percentage of weight on GPU, "
		 "the percentage of weight on CPU, "
		 "the percentage of attention cache on GPU, "
		 "the percentage of attention cache on CPU, "
		 "the percentage of activations on GPU, "
		 "the percentage of activations on CPU")
	parser.add_argument("--pin_weight", type=str2bool, nargs="?",
		const=True, default=True)
	parser.add_argument("--compress_cache", action="store_true",
		help="Whether to compress cache.")

	parser.add_argument("--max_num_kv", type=int, default=400)
	parser.add_argument("--token_group", type=int, default=1)
	
	parser.add_argument("--warmup_input_path", type=str, default=None)
	parser.add_argument("--test_input_path", type=str)
	parser.add_argument("--nv_profile", type=int)
	parser.add_argument("--run_args", type=str, default="L0")
	parser.add_argument("--cpu_only", type=int, default=0)
	
	parser.add_argument("--sort_idx", type=int, default=0)
	parser.add_argument("--start_layer", type=str, default='0-curr')
	
	parser.add_argument("--att_score_mode", type=str, default='none') # none or max
	parser.add_argument("--lr_proj_path", type=str, default='none')
	parser.add_argument("--lr_proj_mode", type=str, default='none')
	
	parser.add_argument("--alpha", type=float, default=4)
	parser.add_argument("--skew_matrix_path", type=str, default='none')
	parser.add_argument("--skew_partial_idx_path", type=str, default='none')
	
	parser.add_argument("--flash_att", type=int, default=1)
	parser.add_argument("--paged_att", type=int, default=1)
	parser.add_argument("--use_token_cache", type=int, default=0)
	parser.add_argument("--dk_wr", type=str, default="none")
	parser.add_argument("--dk_rd", type=str, default="none")
	parser.add_argument("--run_info", type=str, default="none")
	
	parser.add_argument("--use_mmap", type=int, default=0)
	parser.add_argument("--batch_split", type=str, default='1')
	parser.add_argument("--disk_dev_name", type=str, default='nvme')
	
	parser.add_argument("--reuse_budget", type=int, default=0)
 
	parser.add_argument("--att_comp_mode", type=str, default='concat')
	parser.add_argument("--seed", type=int, default=1234)
	
 
	

	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	add_parser_arguments(parser)
	args = parser.parse_args()
	assert len(args.percent) == 6
	run_flexgen(args)
