"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from itertools import count
import os
import queue
import shutil
import time
import threading
import nvtx
import torch
import torch.nn.functional as F
import numpy as np
from utils import GB, cpu_mem_stats, np_dtype_to_torch_dtype, torch_dtype_to_np_dtype, torch_dtype_to_num_bytes
from model_utils import apply_rotary_pos_emb, rms_norm, repeat_kv, llama_mlp_func
from methods import speculate_attention
from vllm.attention.ops.paged_attn import PagedAttention

try:
	from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except:
	pass
	# flash_attn_func = None

IS_CUDA_AVILABLE = torch.cuda.is_available()
general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None

import ctypes
POSIX_FADV_DONTNEED = 4  
libc = ctypes.CDLL("libc.so.6")  

import time
from diskio import DiskIO, DiskInterface, create_kv_file

def fix_recursive_import():
	global general_copy_compressed, TorchCompressedDevice, global_cpu_device
	import compression
	general_copy_compressed = compression.general_copy_compressed
	TorchCompressedDevice = compression.TorchCompressedDevice

class DeviceType(Enum):
	CPU = auto()
	CUDA = auto()
	DISK = auto()
	MIXED = auto()
	COMPRESSED = auto()

	@staticmethod
	def convert(name):
		if name == "cpu":
			return DeviceType.CPU
		elif name == "cuda":
			return DeviceType.CUDA
		elif name == "disk":
			return DeviceType.DISK
		elif name == "mixed":
			return DeviceType.MIXED
		elif name == "compressed":
			return DeviceType.COMPRESSED
		else:
			raise ValueError(f"Invalid name: {name}")

class TorchTensor:
	"""
	Wrap pytorch tensors to support
	  - Unified representation for normal and compressed tensors on
		GPUs, CPUs, disks and mixed devices.
	  - Asynchronous copy between tensors on any formats and any devices.

	This is achieved by implementing the data movement APIs for primitive cases
	and using recursive structures to handle other combinations.

	Note:
	For a tensor on a TorchDevice, self.data is a primitive tensor.
	  type: torch.Tensor.
	For a tensor on a TorchDisk, self.data is a filename.
	  type: str
	For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
	  type: Tuple[Tuple[TorchTensor], Tuple[int]]
	For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
	  type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
	"""
	name_count = count()

	def __init__(self, shape, dtype, data, device, name=None):
		if isinstance(data, torch.Tensor):
			assert data.device == device.dev

		self.shape = shape
		self.dtype = dtype
		self.data = data
		self.device = device

		# Whether delete the file when the tensor is deleted
		self.delete_file = False

		self.name = name or TorchTensor.next_name()

	@property
	def bytes(self):
		return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

	@classmethod
	def next_name(cls):
		return f"t_{next(cls.name_count)}"

	@classmethod
	def create_from_torch(cls, data, device, name=None):
		return cls(data.shape, data.dtype, data, device, name=name)

	def delete(self):
		assert self.device is not None, "already deleted"
		if self.device.device_type == DeviceType.DISK:
			self.device.delete(self)
		self.device = self.data = None

	def load_from_np(self, np_array):
		if self.device.device_type == DeviceType.DISK:
			with open(self.data, "wb") as fout:
				np.save(fout, np_array)
		else:
			if self.device.device_type == DeviceType.COMPRESSED:
				tmp = torch.from_numpy(np_array)
				tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
				general_copy(self, None, tmp, None)
			else:
				# print(np_array.shape, self.data.shape, flush=True)
				self.data.copy_(torch.from_numpy(np_array))

	def load_from_np_file(self, filename):
		filename += '.npy'
		if self.device.device_type == DeviceType.DISK:
			shutil.copy(filename, self.data)
		else:
			fd = os.open(filename, os.O_RDONLY)
			np_array = np.load(filename)
			# print(f"Loading {filename} to {self.name} on {self.device.name}", flush=True)
			libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
			os.close(fd)
			self.load_from_np(np_array)

	def copy(self, dst, src_indices=None, copy_key=None):
		if src_indices:
			assert all(x.step is None for x in src_indices)
			shape = tuple(x.stop - x.start for x in src_indices
				) + self.shape[len(src_indices):]
		else:
			shape = self.shape

		if dst.device_type == DeviceType.COMPRESSED:
			raise NotImplementedError
			ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
		else:
			if self.dtype == torch.bfloat16:
				ret = dst.allocate(shape, torch_dtype_to_np_dtype[torch.float16], force_bf16=True)
			else:
				ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
		general_copy(ret, None, self, src_indices, copy_key)
		return ret

	def smart_copy(self, dst, src_indices=None, copy_key=None):
		if self.device == dst:
			return self, False
		return self.copy(dst, src_indices=src_indices, copy_key=copy_key), True

	def move(self, dst):
		if self.device == dst:
			return self
		ret = self.copy(dst)
		self.delete()
		return ret

	def __str__(self):
		return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
				f"device={self.device.name if self.device else None})")


class TorchDevice:
	"""Wrap tensor and computation APIs of a single CPU or GPU."""

	def __init__(self, name, mem_capacity=None, flops=None):
		self.name = name
		self.mem_capacity = mem_capacity
		self.flops = flops

		self.dev = torch.device(name)
		self.device_type = DeviceType.convert(self.dev.type)

		if self.device_type == DeviceType.CPU:
			global global_cpu_device
			global_cpu_device = self

	def allocate(self, shape, dtype, pin_memory=None, name=None, force_bf16=False):
		if self.device_type == DeviceType.CPU and IS_CUDA_AVILABLE:
			pin_memory = True if pin_memory is None else pin_memory
		else:
			pin_memory = False
		dtype = np_dtype_to_torch_dtype[dtype]
		if force_bf16:
			dtype = torch.bfloat16
		data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
		return TorchTensor.create_from_torch(data, self, name=name)

	def delete(self, tensor):
		pass
	
	@torch.inference_mode()
	def gen_attention_mask(self, token_ids, pad_token_id, donate):
		data = token_ids.data.ne(pad_token_id)
		if donate[0]: token_ids.delete()
		return TorchTensor.create_from_torch(data, self)
	
	@torch.inference_mode()
	def extend_attention_mask(self, attention_mask, donate):
		bs = attention_mask.shape[0]
		data = torch.concat((attention_mask.data,
			 torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
		if donate[0]: attention_mask.delete()
		return TorchTensor.create_from_torch(data, self)
	
	@torch.inference_mode()
	def input_embed(self, inputs, attention_mask, w_token, w_pos, donate):
		# decompress weights
		if w_token.device.device_type == DeviceType.COMPRESSED:
			w_token = w_token.device.decompress(w_token)
			if w_pos is not None:
				w_pos = w_pos.device.decompress(w_pos)

		token_ids = inputs.data
		mask = attention_mask.data
		if donate[0]: inputs.delete()
		if donate[1]: attention_mask.delete()

		# token embedding
		token_embed = F.embedding(token_ids, w_token.data)
		# pos embedding
		if w_pos is not None:
			positions = torch.cumsum(mask, dim=1).int() * mask + 1
			# cut positions if `past_key_values_length` is > 0
			past_key_values_length = mask.shape[1] - token_ids.shape[1]
			positions = positions[:, past_key_values_length:]
			pos_embed = F.embedding(positions, w_pos.data)
			data = token_embed + pos_embed
		else:
			data = token_embed
		return TorchTensor.create_from_torch(data, self)

	@torch.inference_mode()
	def output_embed(self, inputs, w_ln, b_ln, w_token, donate,
						 do_sample, temperature):
		# decompress weights
		if w_token.device.device_type == DeviceType.COMPRESSED:
			w_token = w_token.device.decompress(w_token)

		b, s, h = inputs.shape
		dtype = inputs.data.dtype
		hidden = inputs.data[:,-1:]
		
		if b_ln is not None:
			hidden = F.layer_norm(hidden, (h,), weight=w_ln.data.to(dtype), bias=b_ln.data.to(dtype))
		else:
			hidden = rms_norm(hidden, w_ln.data.to(dtype))
			
		if donate[0]: inputs.delete()

		# output embedding
		logits = F.linear(hidden, w_token.data.to(dtype))
		last_token_logits = logits[:,-1,:]

		if do_sample and not temperature < 1e-5:
			probs = torch.softmax(last_token_logits / temperature, dim=-1)
			ids = torch.multinomial(probs, num_samples=1)
		else:
			ids = last_token_logits.argmax(dim=1, keepdim=True)
		return TorchTensor.create_from_torch(ids, self)

	
	def init_cache_one_gpu_batch(self, config, task, policy):
		raise NotImplementedError
	
	@staticmethod
	@torch.inference_mode()
	def chuck_attn(q, k, v, scaling, flashatt, lr_proj_mode, 
				   b, s, d, attention_mask, dev, n_head, head_dim, kv_rep, chunk_size):

		eff_chunk = chunk_size if 0 < chunk_size < s else s
		out = torch.empty(b, s, d, device=dev, dtype=q.dtype)

		if not flashatt:
			idx = torch.arange(s, device=dev)
			attn_mask_full = attention_mask.data.view(b, 1, 1, s) & (idx <= idx.view(s, 1)).view(1, 1, s, s)
			neg_inf = torch.finfo(q.dtype).min
			
		for start in range(0, s, eff_chunk):
			end = min(start + eff_chunk, s)
			q_chunk = q[:, start:end]  # [b, chunk, h, head_dim]
			chunk_len = end - start
			
			if flashatt:            
				out[:, start:end] = flash_attn_func(q_chunk, k, v, 
									causal=True, 
									softmax_scale=scaling).reshape(b, chunk_len, d)
			else:
				tmp = torch.matmul(
					q_chunk.transpose(1, 2),        # [b, h, chunk, head_dim]
					key.permute(0, 2, 3, 1),        # [b, h, head_dim, s]
				) * scaling
				tmp.masked_fill_(~attn_mask_full[:, :, start:end], neg_inf)
				tmp = torch.softmax(tmp.float(), dim=-1).to(tmp.dtype)
				tmp = torch.matmul(
					tmp,
					value.transpose(1, 2),          # [b, h, s, head_dim]
				)                                   # [b, h, chunk, head_dim]
				out[:, start:end] = tmp.transpose(1, 2).reshape(b, chunk_len, d)
		return out

	@torch.inference_mode()
	def mha(self, inputs, attention_mask, 
				w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, q_norm, k_norm,
				n_head, head_dim, scaling, kv_rep, pos_emb,
				lr_proj_mode, donate, 
				compress_cache, comp_config, flashatt, chunk_size):

		b, s, d = inputs.shape
		dtype = inputs.data.dtype
		
		if b_ln is not None:
			hidden = F.layer_norm(inputs.data, (d,), weight=w_ln.data.to(dtype), bias=b_ln.data.to(dtype))
		else:
			hidden = rms_norm(inputs.data, w_ln.data.to(dtype))

		# shape: (b, s, h)
		if w_q.data.shape[1] == w_q.data.shape[0] + 1:
			hidden_ = torch.cat((hidden, torch.ones(b, s, 1, dtype=hidden.dtype, device=hidden.device)), dim=-1)
			q = F.linear(hidden_, w_q.data.to(dtype), bias=None) 
			k = F.linear(hidden_, w_k.data.to(dtype), bias=None)
			del hidden_
		else: 
			q = F.linear(hidden, w_q.data.to(dtype), bias=b_q.data.to(dtype) if b_q is not None else None)
			k = F.linear(hidden, w_k.data.to(dtype), bias=b_k.data.to(dtype) if b_k is not None else None)
		
		v = F.linear(hidden, w_v.data.to(dtype), bias=b_v.data.to(dtype) if b_v is not None else None)
		del hidden
		
		# shape: (b, s, n_head, head_dim)
		q = q.view(b, s, n_head, head_dim) # b, s, h, d
		if q_norm is not None:
			q = rms_norm(q, q_norm.data.to(dtype))
		k = k.view(b, s, -1, head_dim) 
		if k_norm is not None:
			k = rms_norm(k, k_norm.data.to(dtype))
		v = v.view(b, s, -1, head_dim)

		if pos_emb is not None:
			q, k = apply_rotary_pos_emb(q, k, *pos_emb, layout='bshd')

		out = TorchDevice.chuck_attn(q, k, v, scaling, flashatt, lr_proj_mode, 
				   b, s, head_dim*n_head, attention_mask, self.dev, n_head, head_dim, kv_rep, chunk_size)
		del q
			   
		out = F.linear(out, w_out.data.to(dtype), bias=b_out.data.to(dtype) if b_out is not None else None)
		out.add_(inputs.data)

		if donate[0]: inputs.delete()
		if donate[1]: attention_mask.delete()

		# b, s, h, d -> b, s, h*d
		k = k.reshape(b, s, -1)
		v = v.reshape(b, s, -1)

		if compress_cache:
			k = self.compressed_device.compress(k, comp_config)
			v = self.compressed_device.compress(v, comp_config)
			
		return TorchTensor.create_from_torch(out, self), k, v

	@staticmethod
	@torch.inference_mode()
	def qkvo(q, k, v, scaling, w_out, b_out, shape, dtype, kv_packed=True, cache_manager=None):
		# q/k/v: b, s, h, d
		if cache_manager is not None:
			out = PagedAttention.forward_decode(
				q,
				cache_manager.key_cache,
				cache_manager.value_cache,
				cache_manager.get_block_tables(cache_manager.layer_id),
				cache_manager.get_seqlens(),
				cache_manager.max_seq_len,
				"auto",
				cache_manager.num_kv_heads,
				scaling,
				alibi_slopes=None,
				k_scale=cache_manager.k_scale,
				v_scale=cache_manager.v_scale,
			).view(*shape)
		elif kv_packed:
			out = flash_attn_kvpacked_func(q, k, causal=True, softmax_scale=scaling).view(*shape)
		else:
			out = flash_attn_func(q, k, v, causal=True, softmax_scale=scaling).view(*shape)
		out = F.linear(out, w_out.data.to(dtype), bias=b_out.data.to(dtype) if b_out is not None else None)
		return out

	@torch.inference_mode()
	def mha_gen(self, inputs, attention_mask, 
				w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, q_norm, k_norm,
				n_head, head_dim, scaling, kv_rep, pos_emb, past_pos_emb,
				enable_pred, lr_proj_mode, next_lr_kproj, next_qproj, next_qnorm,
				next_partial_index, next_partial_wq, skew_matrix,
				kv_cache, next_lr_kcache, window_kv_buf, kv_window_index, 
				kv_layout, token_group,
				att_ins, donate, compress_cache, comp_config, 
				spec_stream, prefetch_event, prefetch_sync, alpha, max_num_kv, score_mode, 
				att_comp_mode='mixed'): # mixed / concat 
		"""Multi-head attention (decoding phase)."""

		dtype = inputs.data.dtype
		if spec_stream is not None:
			event = torch.cuda.Event()
		b, tgt_s, d = inputs.shape

		if b_ln is not None:
			hidden = F.layer_norm(inputs.data, (d,), weight=w_ln.data.to(dtype), bias=b_ln.data.to(dtype))
		else:
			hidden = rms_norm(inputs.data, w_ln.data.to(dtype))

		if w_q.data.shape[1] == w_q.data.shape[0] + 1:
			hidden_ = torch.cat((hidden, torch.ones(b, 1, 1, dtype=hidden.dtype, device=hidden.device)), dim=-1)
			spec_hidden_in = hidden_
		else:
			spec_hidden_in = hidden

		# Speculate attention
		newest_fetch_flag = threading.Event()
		
		if enable_pred and lr_proj_mode != 'none':
			if spec_stream is not None:
				event.record()
				 
			with nvtx.annotate("speculate_attention", color='green'):
				if spec_stream is not None:
					with torch.cuda.stream(spec_stream):
						spec_stream.wait_event(event)
						att_ins.prefetch_idx = speculate_attention(lr_proj_mode, spec_hidden_in, next_partial_wq, skew_matrix, 
																	next_qproj, next_qnorm, next_lr_kproj, next_lr_kcache, pos_emb,
																	next_partial_index, scaling,
																	n_head, kv_rep, alpha, max_num_kv, token_group, 
																	score_mode)
				else:
					att_ins.prefetch_idx = speculate_attention(lr_proj_mode, spec_hidden_in, next_partial_wq, skew_matrix, 
																next_qproj, next_qnorm, next_lr_kproj, next_lr_kcache, pos_emb,
																next_partial_index, scaling,
																n_head, kv_rep, alpha, max_num_kv, token_group,
																score_mode)
			if prefetch_event is not None:
				def func():
					if prefetch_sync is not None:
						prefetch_sync()
						newest_fetch_flag.set()
					prefetch_event()
				t = threading.Thread(target=func)
				t.start()
			
		with nvtx.annotate("att_proj", color='yellow'):   
			# shape: (b, 1, h)
			if w_q.data.shape[1] == w_q.data.shape[0] + 1:
				q = F.linear(hidden_, w_q.data.to(dtype), bias=None)
				k_new = F.linear(hidden_, w_k.data.to(dtype), bias=None)
				del hidden_    
			else:
				q = F.linear(hidden, w_q.data.to(dtype), bias=b_q.data.to(dtype) if b_q is not None else None)
				k_new = F.linear(hidden, w_k.data.to(dtype), bias=b_k.data.to(dtype) if b_k is not None else None)
			v_new = F.linear(hidden, w_v.data.to(dtype), bias=b_v.data.to(dtype) if b_v is not None else None)
			q = q.view(b, tgt_s, n_head, head_dim) # b, s, h, d
			if q_norm is not None:
				q = rms_norm(q, q_norm.data.to(dtype))
			k_new = k_new.view(b, tgt_s, -1, head_dim) # b, s, h, d
			if k_norm is not None:
				k_new = rms_norm(k_new, k_norm.data.to(dtype))
			v_new = v_new.view(b, tgt_s, -1, head_dim) # b, s, h, d
			if pos_emb is not None:
				q, k_new = apply_rotary_pos_emb(q, k_new, *pos_emb, layout='bshd')       
			# b, hd                     
			if window_kv_buf is not None:
				window_kv_buf[0, kv_window_index] = k_new.view(b, -1)
				window_kv_buf[1, kv_window_index] = v_new.view(b, -1)

		with nvtx.annotate("prefetch_sync1", color='green'): 
			if lr_proj_mode != 'none' and prefetch_sync is not None and not newest_fetch_flag.is_set(): # avoid sync the newest prefetch
				prefetch_sync()
	
		with nvtx.annotate("kv_concat", color='red'): 
			if compress_cache:
				# shape: (s, b * n_head, head_dim)
				raise NotImplementedError
				kv = kv_cache.device.decompress(kv_cache)[:src_s]
			else:
				if kv_layout == 'cache_manager':
					kv_cache.update_rolling_buffer(k_new.view(b*tgt_s, -1, head_dim), v_new.view(b*tgt_s, -1, head_dim), kv_cache.layer_id)
					assert window_kv_buf is None
				# shape: (s, b * n_head, head_dim)
				elif kv_layout == 'bs(nd)': # for token cache
					past_hidden = kv_cache.data
					if b_ln is not None:
						past_hidden = F.layer_norm(past_hidden, (d,), weight=w_ln.data.to(dtype), bias=b_ln.data.to(dtype))
					else:
						past_hidden = rms_norm(past_hidden, w_ln.data.to(dtype))
					if w_q.data.shape[1] == w_q.data.shape[0] + 1:
						past_hidden_ = torch.cat((past_hidden, torch.ones(b, past_hidden.shape[1], 1, dtype=hidden.dtype, device=hidden.device)), dim=-1) 
						k = F.linear(past_hidden_, w_k.data.to(dtype), bias=None)
						del past_hidden_
					else:
						k = F.linear(past_hidden, w_k.data.to(dtype), bias=None if b_k is None else b_k.data.to(dtype))
					v = F.linear(past_hidden, w_v.data.to(dtype), bias=b_v.data.to(dtype) if b_v is not None else None)  
					k = k.view(b, past_hidden.shape[1], -1, head_dim) # b, s-1, h, d
					if k_norm is not None:
						k = rms_norm(k, k_norm.data.to(dtype))
					v = v.view(b, past_hidden.shape[1], -1, head_dim) # b, s-1, h, d
					del past_hidden, kv_cache
					if past_pos_emb is not None:
						_, k = apply_rotary_pos_emb(None, k, *past_pos_emb, layout='bshd')          
					kv_packed = False               
				elif kv_layout == 'bs2hd':
					# shape: (b, s, 2, h, d)
					kv_data = kv_cache.data.view(kv_cache.shape[0], kv_cache.shape[1], 2, -1, head_dim)
					kv_packed = True
					del kv_cache
				elif kv_layout == 'nb2ghd':
					kv_data = kv_cache.data.view(kv_cache.data.shape[0], kv_cache.data.shape[1], 
												 2, token_group, -1, head_dim).permute(1, 0, 3, 2, 4, 5).reshape(
													 b, kv_cache.data.shape[0]*token_group, 2, -1, head_dim
												 )
					kv_packed = True
					del kv_cache
				else:
					raise ValueError(f"Unsupported kv_layout: {kv_layout}")
			
			if window_kv_buf is not None:
				# 2, g, b, h, d -> b, g, 2, h, d
				# token cache branch won't meet this condition
				local_kv = window_kv_buf.view(2, window_kv_buf.shape[1], b, -1, head_dim)[:, :kv_window_index+1]
				local_kv = local_kv.permute(2, 1, 0, 3, 4) # b, g, 2, h, d
				if att_comp_mode == 'mixed':
					raise NotImplementedError
				else:
					kv_data = torch.cat((kv_data, local_kv), dim=1) # b, s+g, 2, h, d
					del local_kv
				src_s = kv_data.shape[1]
			elif kv_layout != 'cache_manager':
				if kv_packed:
					kv_new = torch.stack((k_new, v_new), dim=2) # b, s, 2, h, d
					kv_data = torch.cat((kv_data, kv_new), dim=1) # b, s+1, 2, h, d
					src_s = kv_data.shape[1]
				else:
					k = torch.cat((k, k_new), dim=1) # b, s, h, d
					v = torch.cat((v, v_new), dim=1) # b, s, h, d
					src_s = k.shape[1]
		
		with nvtx.annotate("qktvo", color='blue'):  
			if kv_layout == 'cache_manager':
				out = TorchDevice.qkvo(q.view(b*tgt_s, n_head, head_dim), None, None, scaling, w_out, b_out, 
						   				(b, tgt_s, n_head*head_dim), dtype, cache_manager=kv_cache)
			elif kv_packed:
				out = TorchDevice.qkvo(q, kv_data, None, scaling, w_out, b_out, (b, tgt_s, n_head*head_dim), dtype, kv_packed=True)
			else:
				out = TorchDevice.qkvo(q, k, v, scaling, w_out, b_out, (b, tgt_s, n_head*head_dim), dtype, kv_packed=False)
			out.add_(inputs.data)
			if donate[0]: inputs.delete()
			if donate[1]: attention_mask.delete()
			
		with nvtx.annotate("kv_new", color='yellow'):
			if compress_cache:
				raise NotImplementedError
				if comp_config.group_dim == 0:
					s_ = src_s // comp_config.group_size * comp_config.group_size
					k_new = k[:, :, s_:].permute(2, 0, 1)
					v_new = v[:, s_:, :].permute(1, 0, 2)
				k_new = self.compressed_device.compress(k_new, comp_config)
				v_new = self.compressed_device.compress(v_new, comp_config)
					
		with nvtx.annotate("waiting", color='brown'):     
			if enable_pred and lr_proj_mode != 'none':
				if prefetch_event is not None:
					t.join()
		
		return TorchTensor.create_from_torch(out, self), k_new, v_new

	@torch.inference_mode()
	def opt_mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
		# decompress weights
		# if wi.device.device_type == DeviceType.COMPRESSED:
		#     wi = wi.device.decompress(wi)
		#     wo = wo.device.decompress(wo)
		b, s, d = inputs.shape
		dtype = inputs.data.dtype
		out = F.layer_norm(inputs.data, (d,), weight=w_ln.data.to(dtype), bias=b_ln.data.to(dtype))
		out = F.linear(out, wi.data.to(dtype), bias=bi.data.to(dtype))
		F.relu(out, inplace=True)
		out = F.linear(out, wo.data.to(dtype), bias=bo.data.to(dtype))
		out.add_(inputs.data)
		if donate[0]: inputs.delete()
		return TorchTensor.create_from_torch(out, self)
	
	@torch.inference_mode()
	def llama_mlp_gen(self, inputs, w_gate, w_up, w_down, w_norm, donate, act):
		# assert act == 'silu', f"Unsupported activation: {act}"
		dtype = inputs.data.dtype
		out = rms_norm(inputs.data, w_norm.data.to(dtype))
		out = llama_mlp_func(out, w_gate.data, w_up.data, w_down.data)
		out.add_(inputs.data)
		if donate[0]: inputs.delete()
		return TorchTensor.create_from_torch(out, self)
	
	@torch.inference_mode()
	def llama_mlp(self, inputs, w_gate, w_up, w_down, w_norm, donate, act, chunk_size=-1):
		# assert act == 'silu', f"Unsupported activation: {act}"
		dtype = inputs.data.dtype
		out = rms_norm(inputs.data, w_norm.data.to(dtype))
		s = out.shape[1]
		eff_chunk = chunk_size if 0 < chunk_size < s else s
		for start in range(0, s, eff_chunk):
			end = min(start + eff_chunk, s)
			out[:, start:end] = llama_mlp_func(out[:, start:end], w_gate.data, w_up.data, w_down.data)
		out.add_(inputs.data)
		if donate[0]: inputs.delete()
		return TorchTensor.create_from_torch(out, self)

	def synchronize(self):
		torch.cuda.synchronize()

	def mem_stats(self):
		if self.device_type == DeviceType.CUDA:
			cur_mem = torch.cuda.memory_allocated(self.dev)
			peak_mem = torch.cuda.max_memory_allocated(self.dev)
		elif self.device_type == DeviceType.CPU:
			cur_mem = cpu_mem_stats()
			peak_mem = 0
		else:
			raise NotImplementedError()

		return cur_mem, peak_mem

	def print_stats(self, output_file=None):
		torch.cuda.synchronize()
		cur_mem, peak_mem = self.mem_stats()

		if output_file is not None:
			with open(output_file, "w") as f:
				f.write(f"TorchDevice: {self.name}\n")
				f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
						f" peak_mem: {peak_mem/GB:.4f} GB\n")
		else:
			print(f"TorchDevice: {self.name}")
			print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
				  f" peak_mem: {peak_mem/GB:.4f} GB")

		return cur_mem, peak_mem

	def __str__(self):
		return f"TorchDevice(name={self.name})"


class ThreadSafeDict:
	def __init__(self):
		# self.lock = threading.Lock()
		self.data = {}

	def set(self, key, value):
		# with self.lock:
		self.data[key] = value

	def get(self, key):
		# with self.lock:
		return self.data.get(key)

disk_io_list = None

class TorchDisk:
	"""Manage tensors stored on a disk."""

	def __init__(self, path, cuda_id=0, num_copy_threads=2, 
				 en_cpu_relay=False, en_map_dict=True, en_copy_key=False, 
				 cfg=None):
		self.name = path
		self.path = [os.path.abspath(os.path.expanduser(path_)) for path_ in path.split(",")]

		self.device_type = DeviceType.DISK
		# self.compressed_device = TorchCompressedDevice(self)
		self.compressed_device = None

		for path_ in self.path:
			if os.path.exists(path_):
				assert os.path.isdir(path_)
			else:
				os.makedirs(path_)

		self.en_copy_key = en_copy_key
		if en_map_dict:
			self.mmap_dict = ThreadSafeDict()
		else:
			self.mmap_dict = None
		
		self.disk_sync_time_dict = {}
		
		if en_copy_key:
			assert num_copy_threads > 1, f"num_copy_threads must be greater than 1"
			self.copy_queue = {
				'load_kv': queue.Queue(),
				'store_kv': queue.Queue(),
			}
			self.copy_threads = {
				'load_kv': [threading.Thread(
								target=copy_worker_func, 
								args=(self.copy_queue['load_kv'], cuda_id, 
									  en_cpu_relay, self.mmap_dict, cfg)
							) for i in range(num_copy_threads//2)],
				'store_kv': [threading.Thread(
								target=copy_worker_func, 
								args=(self.copy_queue['store_kv'], cuda_id, 
									  en_cpu_relay, self.mmap_dict, cfg)
							) for i in range(num_copy_threads//2)],
			}
		else:
			assert num_copy_threads > 0, f"num_copy_threads must be greater than 0"
			self.copy_queue = {
				'all': queue.Queue(),
			}
			self.copy_threads = {
				'all': [threading.Thread(
							target=copy_worker_func, 
							args=(self.copy_queue['all'], cuda_id, 
								  en_cpu_relay, self.mmap_dict, cfg)
						) for i in range(num_copy_threads)],
			}
			
		for key in self.copy_threads:
			for t in self.copy_threads[key]:
				t.start()

		global global_disk_device
		global_disk_device = self


	def fstrim(self):
		pass
		# for path_ in self.path:
			# print(f"fstrim {path_}")
			# os.system(f'echo PASSWORD | sudo -S sh -c "fstrim -v {path_}"')

	def register_fd(self):
		fd_dict = {}
		if self.mmap_dict:
			for disk_tensor in self.mmap_dict.data.values():
				if disk_tensor.disk_interface.use_mmap:
					return
				for i, fd in enumerate(disk_tensor.disk_interface.fd):
					if i not in fd_dict:
						fd_dict[i] = []
					fd_dict[i].append(fd)
			global disk_io_list
			assert len(disk_io_list) == len(fd_dict.keys()), f"{len(disk_io_list)} != {len(fd_dict.keys())}"
			for i, disk_io in enumerate(disk_io_list):
				disk_io.register_files(fd_dict[i]) 

	def allocate(self, shape, dtype, pin_memory=None, name=None, force_bf16=False, tot_shape=None, attr=None):    
		name = name or TorchTensor.next_name()
		path_list = []
		for path_, shape_ in zip(self.path, shape):
			path = os.path.join(path_, name)
			path_list.append(path)
			if name.endswith("mmap"):
				create_kv_file(path, shape_, dtype, use_mmap=True, attr=attr)
			elif name.endswith("bin"):
				create_kv_file(path, shape_, dtype, use_mmap=False, attr=attr)
			else:
				raise ValueError(f"Unsupported file extension: {name}")
		if tot_shape is not None:
			shape = tot_shape
		path = ','.join(path_list)
		return TorchTensor(shape, np_dtype_to_torch_dtype[dtype] if not force_bf16 else torch.bfloat16,
						   path, self, name=name)

	def get_swap_info(self):
		swap_info_list = []
		if self.mmap_dict:
			for disk_tensor in self.mmap_dict.data.values():
				if len(disk_tensor.swap_num_list) > 0:
					swap_info_list.append((disk_tensor.path, 
										  np.array(disk_tensor.swap_num_list), 
										  np.array(disk_tensor.swap_time_list),
										  np.array(disk_tensor.flush_num_list),
										  np.array(disk_tensor.flush_time_list),
										  np.array(disk_tensor.prefill_wr_time_list),
										  disk_tensor.swap_indices_list))
		return swap_info_list

	def get_cache_info(self):
		cache_info_list = []
		if self.mmap_dict:
			for disk_tensor in self.mmap_dict.data.values():
				if hasattr(disk_tensor, "cache"):
					cache_info_list.append((disk_tensor.path, np.array(disk_tensor.get_cache_info())))
		return cache_info_list
		
	def delete(self, tensor):
		if os.path.exists(tensor.data):
			if self.mmap_dict is not None:
				disk_tensor = self.mmap_dict.get(tensor.data)   
				if disk_tensor is not None:
					disk_tensor.close()
					self.mmap_dict.set(tensor.data, None)  
			if tensor.delete_file: 
				os.remove(tensor.data)

	def init_cache_one_gpu_batch(self, config, task, policy, name=None, force_bf16=False, batch_split=None, attr=None):

		seq_len = (task.prompt_len + task.gen_len - 1 + policy.token_group - 1) // policy.token_group 
		seq_len = seq_len * policy.token_group
		
		def split_(batch_split, n):
			if batch_split is not None:
				assert len(batch_split) == len(self.path), f"{len(batch_split)} != {len(self.path)}"  
				batch_split = n * batch_split
				batch_split = batch_split.round().int()
				batch_split[-1] = n - batch_split[:-1].sum()
				assert batch_split.sum() == n, f"{batch_split.sum()} != {n}"
				assert batch_split.min() > 0, f"{batch_split.min()} <= 0"
				batch_split = batch_split.tolist()
				print(f"batch_split={batch_split}")
				return batch_split             
			else:
				return [n]

		batch_split = split_(batch_split, policy.gpu_batch_size)
		shape = []
		for b in batch_split:
			shape.append((b, seq_len, config.num_kv_heads * config.head_dim * 2))
		tot_shape = (sum(batch_split), seq_len, config.num_kv_heads * config.head_dim * 2)                    
		kv_cache = self.allocate(shape, np.float16, name=name, force_bf16=force_bf16, tot_shape=tot_shape, attr=attr)
		return kv_cache

	def submit_copy(self, key, *args):
		if self.en_copy_key:
			self.copy_queue[key].put_nowait(args)
		else:
			self.copy_queue['all'].put_nowait(args)

	def synchronize(self, key='all', count_time_key=None):
		if count_time_key:
			start = time.perf_counter()
		if key == 'all' or self.en_copy_key == False:
			for k in self.copy_queue:
				self.copy_queue[k].join()
		else:
			self.copy_queue[key].join()
		if count_time_key:
			elapsed = time.perf_counter() - start
			if count_time_key not in self.disk_sync_time_dict:
				self.disk_sync_time_dict[count_time_key] = []
			self.disk_sync_time_dict[count_time_key].append(elapsed)

	def close_copy_threads(self):
		for key in self.copy_threads:
			for _ in range(len(self.copy_threads[key])):
				self.copy_queue[key].put_nowait(None)
			for t in self.copy_threads[key]:
				t.join()
			self.copy_queue[key].join()
			self.copy_queue[key] = None
		self.copy_queue = None
		
	def mem_stats(self):
		raise NotImplementedError()

	def print_stats(self):
		raise NotImplementedError()

	def __del__(self):
		if self.copy_queue:
			self.close_copy_threads()
		if self.mmap_dict:
			# close all fd.
			for disk_tensor in self.mmap_dict.data.values():
				if disk_tensor is not None:
					disk_tensor.close()         
			self.mmap_dict = None
		del self.disk_sync_time_dict

def general_copy(dst, dst_indices, src, src_indices, key=None):
	
	if type(src) == tuple or type(src) == torch.Tensor:
		assert dst.device.device_type == DeviceType.DISK
		dst.device.submit_copy(key, dst, dst_indices, src, src_indices)
	elif src.device.device_type == DeviceType.DISK:
		# The tensor is on the disk, dispatch to copy threads for asynchronous copy
		src.device.submit_copy(key, dst, dst_indices, src, src_indices)
	else:
		# The normal path
		src = src.data[src_indices] if src_indices else src.data
		dst = dst.data[dst_indices] if dst_indices else dst.data
		dst.copy_(src, non_blocking=True)


class DiskTensor():
	def __init__(self, path, cfg):
		path = path.split(',')
		self.path = path
		self.max_kv_num, self.token_group, layer_num = cfg['common_cfg']
		
		if path[0].endswith("mmap"):
			self.disk_interface = DiskInterface(path, cfg['interface'], self.token_group, use_mmap=True)
		elif path[0].endswith("bin"):
			self.disk_interface = DiskInterface(path, cfg['interface'], self.token_group, use_mmap=False) 
		else:
			raise ValueError(f"Unsupported file extension: {path}")
				
		shape_ = self.disk_interface.shape
		# check shape
		assert all(shape_[0][2] == shape_[i][2] for i in range(1, len(shape_)))
		assert all(shape_[0][1] == shape_[i][1] for i in range(1, len(shape_)))
		
		batch_alloc = [s[0] for s in shape_]
		self.seq_len = shape_[0][1]
		
		if path[0].endswith("bin"):
			global disk_io_list
			if disk_io_list is None:
				print(f"batch_alloc={batch_alloc}")
				itemsize = self.disk_interface.dtype.itemsize
				diskio_num = len(cfg['disk_dev_name'])
				assert diskio_num == len(shape_), f'{diskio_num} != {len(shape_)}'
				from diskio import get_file_num
				assert layer_num*diskio_num == get_file_num(), f'{layer_num}*{diskio_num} != {get_file_num()}'                
				hd_size = shape_[0][2]
				disk_io_list = [DiskIO(hd_size, self.max_kv_num, b_size_per_io, self.seq_len, self.token_group, layer_num,
										itemsize, wr_tensor_dev='cuda', name=disk_name) for disk_name, b_size_per_io in zip(cfg['disk_dev_name'], batch_alloc)]
				
			self.disk_interface.disk_io_list = disk_io_list
			
		self.diskio_allo_indices = torch.tensor([0]+batch_alloc).cumsum(dim=0).tolist()
		self.b_size = sum(batch_alloc)
		self.device = torch.device("cuda:0")
		
		self.swap_num_list = []
		self.swap_time_list = []
		self.swap_indices_list = []
		self.flush_time_list = []
		self.flush_num_list = []
		self.prefill_wr_time_list = []

		
	def get_tensor_data(self, ind, verbose=False, en_reuse=False):
		# assert type(ind) == torch.Tensor, f"Indices must be a tensor, got {type(ind)}"
		bsz, sel_group_len = ind.shape
		
		self.swap_num_list.append(sel_group_len*self.token_group)
		# if self.disk_interface.use_mmap:
			# self.swap_indices_list.append(ind)

		if verbose:
			print(f"{self.path}:")

		data_size_list = [[0, 0] for _ in range(len(self.diskio_allo_indices)-1)]
		disk_time_break_down_list = [0] * (len(self.diskio_allo_indices)-1)
		
		read_data_list = [None] * (len(self.diskio_allo_indices)-1)
		
		def disk_read(ind_g, g):
			start_time = time.time()
			tid = threading.get_native_id()
			os.sched_setaffinity(tid, {g})
			data_size_list[g][1] = ind_g.numel()
			read_data_list[g] = self.disk_interface.read(ind_g, fd_id=g, diskio_id=g, en_reuse=en_reuse)
			disk_time_break_down_list[g] = (time.time() - start_time)*1000
			data_size_list[g][0] = ind_g.numel() - data_size_list[g][1]

		start_time = time.time()
		t_list = []      
		for g in range(len(self.diskio_allo_indices)-1):
			bh_start = self.diskio_allo_indices[g]
			bh_end = self.diskio_allo_indices[g+1]
			ind_g = ind[bh_start:bh_end]
			if g == len(self.diskio_allo_indices)-2:
				disk_read(ind_g, g)
			else:
				t = threading.Thread(target=disk_read, args=(ind_g, g))
				t.start()
				t_list.append(t)
		for t in t_list:
			t.join()            
		diskio_time = time.time() - start_time
			
		if verbose:
			print(f"Data size: {data_size_list}")
			print(f"DiskIO Time break down: {disk_time_break_down_list} ms")
			print(f"DiskIO time: {diskio_time*1000} ms")
					
		return read_data_list


	def readout(self, indices, dst_data, copy_stream=None, is_cachemanager=False, mask=None):
		if type(indices) == torch.Tensor:
			start_time = time.time()
			output = self.get_tensor_data(indices, en_reuse = mask is not None)
			self.swap_time_list.append(time.time()-start_time)
		else:
			output = [None for _ in range(len(self.diskio_allo_indices) - 1)]
			def read_(ind_g, g):
				tid = threading.get_native_id()
				os.sched_setaffinity(tid, {g})
				output[g] = self.disk_interface.read(ind_g, fd_id=g, diskio_id=g, en_reuse=False) 
			start_time = time.time()
			t_list = []
			for g in range(len(self.diskio_allo_indices) - 1):
				bh_start = self.diskio_allo_indices[g]
				bh_end = self.diskio_allo_indices[g+1]                
				ind_g = (slice(0, bh_end-bh_start), ) + indices[1:]
				if g == len(self.diskio_allo_indices) - 2:
					read_(ind_g, g)
				else:
					t = threading.Thread(target=read_, args=(ind_g, g))
					t.start()
					t_list.append(t)
			for t in t_list:
				t.join()
			self.swap_time_list.append(time.time()-start_time)
			self.swap_num_list.append(output[0].shape[1])
		# start_time = time.time()
		# b,s,2hd: b0, s, 2hd <- b, s, 2hd
		# n,b,2,gnd: n, b0, 2, gnd <- b, n, 2, ghd
		@torch.inference_mode()
		def copy2dst(non_blocking):
			for g in range(len(self.diskio_allo_indices) - 1):
				bh_start = self.diskio_allo_indices[g]
				bh_end = self.diskio_allo_indices[g+1]    
				if type(indices) == tuple:
					dst_data[bh_start:bh_end].copy_(output[g], non_blocking=non_blocking)
				else:
					if is_cachemanager and mask is not None:
						src = output[g].cuda().transpose(0, 1).contiguous()      # [2, B_sel, H_sel]
						dst_data[:, bh_start:bh_end][mask[:, bh_start:bh_end]] = src.view(-1) 
					elif is_cachemanager:
						dst_data[:, bh_start:bh_end].copy_(output[g].permute(2, 0, 1, 3), non_blocking=non_blocking)
					else:
						dst_data[:, bh_start:bh_end].copy_(output[g].transpose(0, 1), non_blocking=non_blocking)

		with nvtx.annotate("copy2dst", color='red'):
			if copy_stream is not None and len(self.diskio_allo_indices) >= 3:
			# if 1:
				with torch.cuda.stream(copy_stream): 
					copy2dst(non_blocking=True)
				copy_stream.synchronize()
			else:
				copy2dst(non_blocking=False)
		# torch.cuda.synchronize()
		# print(f"readout copy time={(time.time()-start_time)*1000:.1f} ms")
		
	def writein(self, indices, src_data, prefill_mode='all_seq'):        
		
		# prefill_mode = 'real_seq'
		
		assert type(indices) == tuple, f"indices must be a tuple, got {type(indices)}"
		
		write_num = indices[1].stop - indices[1].start
		prefill = write_num > 1
		
		if self.disk_interface.attr == 'bn2ghd':
			write_num = write_num * self.token_group
		
		self.flush_num_list.append(write_num)
		
		# 1.  tuple kv, (b, s, hd) + (b, s, hd) -> (b, s, 2hd)
		# 2.1 tuple kv, (b, s//g, ghd) + (b, s//g, ghd) -> (b, s//g, 2ghd)
		# 2.2 tensor: (b, 1, 2ghd)
		
		def write_(ind_g, data_g, g):
			tid = threading.get_native_id()
			os.sched_setaffinity(tid, {g})
			self.disk_interface.write(ind_g, data_g, fd_id=g, diskio_id=g, 
									  prefill=prefill, prefill_mode=prefill_mode)
		start_time = time.time()
		t_list = []
		for g in range(len(self.diskio_allo_indices) - 1):
			bh_start = self.diskio_allo_indices[g]
			bh_end = self.diskio_allo_indices[g+1] 
			if type(src_data) == tuple:
				data_g = (src_data[0][bh_start:bh_end], src_data[1][bh_start:bh_end])
			else:
				data_g = src_data[bh_start:bh_end]
			ind_g = (slice(0, bh_end-bh_start), ) + indices[1:]
			if g == len(self.diskio_allo_indices) - 2:
				write_(ind_g, data_g, g)
			else:
				t = threading.Thread(target=write_, args=(ind_g, data_g, g))
				t.start()
				t_list.append(t)
		for t in t_list:
			t.join()
		if prefill: 
			self.prefill_wr_time_list.append(time.time()-start_time)
		else:
			self.flush_time_list.append(time.time()-start_time)

	def flush(self):
		# start_time = time.time()
		self.disk_interface.flush()
		# self.flush_time_list.append(time.time()-start_time)

	def clear_cache(self):
		self.disk_interface.clear_cache()

	def close_diskio(self):
		global disk_io_list
		if disk_io_list is not None:
			for diskio in disk_io_list:
				diskio.close()
		disk_io_list = None
		self.disk_interface.disk_io_list = None

	def close(self):            
		self.close_diskio()
		self.disk_interface.close() 
		self.swap_num_list = []
		self.swap_time_list = []
		self.swap_indices_list = []
		self.flush_time_list = []
		self.flush_num_list = []
		self.prefill_wr_time_list = []
		


def map_to_torch_tensor(tensor, mmap_dict, cfg=None):
	assert tensor.device.device_type == DeviceType.DISK
	if mmap_dict is None: # mmap_dict is disabled
		disk_tensor = DiskTensor(tensor.data, cfg)
	else:
		disk_tensor = mmap_dict.get(tensor.data)   
		if disk_tensor is None:
			disk_tensor = DiskTensor(tensor.data, cfg)
			mmap_dict.set(tensor.data, disk_tensor)  
	return disk_tensor


def copy_worker_func(queue, cuda_id, en_cpu_relay, mmap_dict, cfg):
	"""The copy worker thread."""
	copy_stream = torch.cuda.Stream()
	while True:
		item = queue.get()
		if item is None:
			queue.task_done()
			return
		dst, dst_indices, src, src_indices = item
		if type(src) == tuple or type(src) == torch.Tensor:
			name = 'WrDk'
			src_data = src
			dst_data = map_to_torch_tensor(dst, mmap_dict, cfg)
		else:            
			name = 'RdDk'
			src_data = map_to_torch_tensor(src, mmap_dict)
			dst_data = dst.data
			
		with nvtx.annotate(name):
			if type(src_data) == DiskTensor:
				if type(dst_indices) == tuple and dst_indices[0] == 'cachemanager':
					src_data.readout(src_indices, dst_data, copy_stream=copy_stream, is_cachemanager=True, mask=dst_indices[1])
				else:
					src_data.readout(src_indices, dst_data[dst_indices] if dst_indices else dst_data, copy_stream=copy_stream)
			else:
				dst_data.writein(dst_indices, src_data[src_indices] if src_indices else src_data)
		del src_data, dst_data, dst, dst_indices, src, src_indices, item
		queue.task_done()

