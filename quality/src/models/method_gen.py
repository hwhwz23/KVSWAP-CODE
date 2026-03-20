import torch
import torch.nn.functional as F
import math

def lr_kcache_func(self, key, is_prefill):
	# key: b,h,s,d
	bsz = key.size(0)
	kv_num = key.size(1)
	kv_groups = self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
	if not is_prefill:
		key_in = key[:, :, -1:, :]
		seq_len = 1
	else:
		key_in = key
		seq_len = key.size(2)
		self.prompt_len = seq_len
	
	def func(self, key_in):
		if self.lr_att_mode.startswith('infinigen'):
			if self.skewing_matrix.device != key.device or self.skewing_matrix.dtype != key.dtype:
				self.skewing_matrix = self.skewing_matrix.to(key.device).to(key.dtype)
			if self.skewing_idx.device != key_in.device:
				self.skewing_idx = self.skewing_idx.to(key_in.device)
			if 'infinigen2' in self.lr_att_mode:
				lr_k_new = key_in.repeat_interleave(dim=1, repeats=kv_groups) @ self.skewing_matrix.unsqueeze(0)
				skewing_idx = self.skewing_idx.view(1, self.skewing_idx.shape[0], 1, -1).expand(bsz, -1, key_in.shape[2], -1)
				lr_k_new = lr_k_new.gather(dim=-1, index=skewing_idx)
				lr_k_new = lr_k_new.reshape(bsz*kv_num*kv_groups, key_in.shape[2], -1)
			else:
				# b, h, s, d @ 1, h, d, d -> b, h, s, d
				lr_k_new = key_in @ self.skewing_matrix[::kv_groups].unsqueeze(0)
				# b, h, s, d --select-(b,h,s,r)-> b, h, s, r
				skewing_idx = self.skewing_idx[::kv_groups]
				skewing_idx = skewing_idx.view(1, skewing_idx.shape[0], 1, -1).expand(bsz, -1, key_in.shape[2], -1)
				lr_k_new = lr_k_new.gather(dim=-1, index=skewing_idx)
				# b, h, s, r -> b*h, s, r
				lr_k_new = lr_k_new.reshape(bsz*kv_num, key_in.shape[2], -1)
		elif self.lr_att_mode in ('lr_proj_mh', 'lr_proj2_mh'):
			# b,h,s,d -> b,s,h,d -> b,s,hd -> b,s,r
			if self.lr_kproj.device != key.device:
				self.lr_kproj = self.lr_kproj.to(key.device)
			lr_k_new = torch.matmul(key_in.transpose(1, 2).reshape(bsz, key_in.shape[2], -1), self.lr_kproj)
		elif self.lr_att_mode.startswith('loki'):
			if self.loki_proj.device != key.device:
				self.loki_proj = self.loki_proj.to(key.device)
			# b,h,s,d @ 1,h,d,r -> b,h,s,r
			lr_k_new = torch.matmul(key_in, self.loki_proj.unsqueeze(0)).reshape(bsz*kv_num, key_in.shape[2], -1)
		else:
			raise NotImplementedError(f"lr_att_mode {self.lr_att_mode} not implemented")
		return lr_k_new	

 
	if is_prefill:
		if self.token_group == 0:
			save_idx = seq_len
		else:
			save_idx = seq_len - seq_len % self.token_group
		key_save = key_in[:, :, :save_idx, :]
		key_save_lr = func(self, key_save)
		key_tail = key_in[:, :, save_idx:, :]
		if key_tail.numel() > 0:
			self.key_buf = key_tail.clone()
		else:
			self.key_buf = None
		return key_save_lr
	else:
		if self.key_buf is not None:
			self.key_buf = torch.cat([self.key_buf, key_in], dim=2)
		else:
			self.key_buf = key_in[:, :, -1:, :].clone()
		if self.key_buf.shape[2] == self.token_group or self.token_group == 0:
			new_key_lr = func(self, self.key_buf)
			self.key_buf = None
			return torch.cat([self.lr_k_cache, new_key_lr], dim=1)
		else:
			return self.lr_k_cache


def speculate_attention(self, src_len, apply_rotary_pos_emb_partial):
	
	num_heads = self.config.num_attention_heads
	kv_groups = self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
	kv_heads = num_heads // kv_groups
	 
	if hasattr(self, 'use_cur_hidden') and self.use_cur_hidden:
		hidden = self.cur_hidden
	else:
		hidden = self.prev_hidden
  
	bsz = hidden.shape[0]
	assert hidden.shape[1] == 1, f"{hidden.shape[1]} != 1"

	len_lr_k = self.lr_k_cache.shape[1]
	# print(f"src_len={src_len}, token_group={self.token_group}, len_lr_k={len_lr_k}, self.lr_k_cache.shape={self.lr_k_cache.shape}")
	# assert len_lr_k <= self.lr_k_cache.shape[1], f"len_lr_k {len_lr_k} >= {self.lr_k_cache.shape[1]}"
	# b,s,r
	lr_k = self.lr_k_cache[:, :len_lr_k]
	# print(f"spec {len_lr_k} len_lr_k", flush=True)
	query = self.q_proj(hidden).view(bsz, 1, num_heads, -1)
	if hasattr(self, 'q_norm'): # for qwen3
		query = self.q_norm(query)
	query = query.transpose(1, 2) # b, h, 1, d
	if apply_rotary_pos_emb_partial is not None:
		query, _ = apply_rotary_pos_emb_partial(query, None)	

	if self.lr_att_mode.startswith('infinigen'):
		if self.skewing_matrix.device != query.device or self.skewing_matrix.dtype != query.dtype:
			self.skewing_matrix = self.skewing_matrix.to(query.device).to(query.dtype)
		if self.skewing_idx.device != query.device:
			self.skewing_idx = self.skewing_idx.to(query.device)
		# b, h, 1, d @ 1, h, d, d -> b, h, 1, d
		query = query @ self.skewing_matrix.unsqueeze(0)
		# b, h, 1, d --select-(b,h,1,r)-> b, h, 1, r
		skewing_idx = self.skewing_idx.view(1, self.skewing_idx.shape[0], 1, -1).expand(bsz, -1, -1, -1)
		query = query.gather(dim=-1, index=skewing_idx) 
		out_heads = num_heads
		query = query.reshape(bsz*num_heads, 1, -1)
		if not 'infinigen2' in self.lr_att_mode:
			lr_k = lr_k.view(bsz, kv_heads, len_lr_k, -1).repeat_interleave(dim=1, repeats=kv_groups)
		lr_k = lr_k.reshape(bsz*num_heads, len_lr_k, -1)
		scaling = self.scaling if hasattr(self, 'scaling') else math.sqrt(self.head_dim)
		attn = torch.bmm(query, lr_k.transpose(1, 2)) * scaling
	elif self.lr_att_mode.startswith('loki'):
		if self.loki_proj.device != query.device or self.loki_proj.dtype != query.dtype:
			self.loki_proj = self.loki_proj.to(query.device).to(query.dtype)
		loki_proj = self.loki_proj.repeat_interleave(dim=0, repeats=kv_groups)
		# b, h, 1, d @ 1, h, d, r -> b, h, 1, r
		query = torch.matmul(query, loki_proj.unsqueeze(0)).reshape(bsz*num_heads, 1, -1)
		lr_k = lr_k.view(bsz, kv_heads, len_lr_k, -1).repeat_interleave(dim=1, repeats=kv_groups)
		lr_k = lr_k.reshape(bsz*num_heads, len_lr_k, -1)
		scaling = self.scaling if hasattr(self, 'scaling') else math.sqrt(self.head_dim)
		attn = torch.bmm(query, lr_k.transpose(1, 2)) * scaling
		out_heads = num_heads
	elif self.lr_att_mode.startswith('lr_proj'):
		if self.lr_att_mode == 'lr_proj_mh':
			# bhsd -> bhd
			query = query.reshape(bsz, kv_heads, kv_groups, -1).mean(dim=2).unsqueeze(-2)
			query = query @ self.lr_kproj.view(kv_heads, -1, lr_k.shape[-1])
			query = query.view(bsz, kv_heads, lr_k.shape[-1]).mean(dim=1)
			out_heads = 1
		elif self.lr_att_mode == 'lr_proj2_mh':
			# b, hg, hkv, 1, d
			query = query.reshape(bsz, kv_heads, kv_groups, -1).transpose(1, 2).unsqueeze(-2)
			# b, hg, hkv, 1, d @ hkv, d, r -> b, hg, hkv, 1, r
			query = query @ self.lr_kproj.view(kv_heads, -1, lr_k.shape[-1])
			query = query.view(bsz, kv_groups, kv_heads, lr_k.shape[-1]).transpose(1, 2)
			lr_k = lr_k.view(bsz, 1, len_lr_k, -1).expand(-1, num_heads, -1, -1)
			out_heads = num_heads
		else:
			raise NotImplementedError(f"lr_att_mode {self.lr_att_mode} not implemented")
		lr_k = lr_k.reshape(bsz*out_heads, len_lr_k, -1)
		query = query.reshape(bsz*out_heads, 1, -1)
		scaling = self.scaling if hasattr(self, 'scaling') else math.sqrt(self.head_dim)
		attn = torch.bmm(query, lr_k.transpose(1, 2)) * scaling
	else:
		raise NotImplementedError(f"lr_att_mode {self.lr_att_mode} not implemented")

	if self.budget <= 1.0:
		max_num_kv = int(self.prompt_len * self.budget) 
	else:
		max_num_kv = int(self.budget)

	fetch_num = min(max_num_kv, attn.shape[-1])

	if self.token_group == 0:
		tail_len = 0
	else:
		tail_len = attn.shape[-1] % self.token_group
	assert tail_len == 0, f"{self.token_group} {attn.shape} != 0"
	##########################################
	# attn = attn.view(bsz, out_heads, -1).softmax(dim=-1).view(bsz, out_heads, -1, self.token_group)
	# attn = attn.mean(dim=-1).mean(dim=1) # b, n//g
	##########################################
	if self.token_group == 0:
		if out_heads > 1:
			attn = attn.view(bsz, kv_heads, out_heads // kv_heads, -1)
			attn = attn.mean(dim=2) # b, h, n
		ind_values = attn.topk(fetch_num, dim=-1, largest=True, sorted=False) # b, h, n
	else:     
		attn = attn.view(bsz, out_heads, -1, self.token_group)
		if out_heads > 1 and self.lr_att_mode.startswith('lr_proj'):
			attn = attn.mean(dim=1, keepdim=True) # TODO decide: b, -1, token_group OR exp().mean()
		elif out_heads > 1:
			if 'mergeh' in self.lr_att_mode:
				attn = attn.mean(dim=1, keepdim=True) # mean across all kv heads is better than aggregating per kv group
			else:
				attn = attn.view(bsz, kv_heads, out_heads // kv_heads, -1, self.token_group)
				attn = attn.mean(dim=2)
		# b, h, s//g, g -> b, h, s//g
		attn = attn.amax(-1) # OR exp().mean()
		##########################################
		ind_values = attn.topk(fetch_num // self.token_group, dim=-1, largest=True, sorted=False) # b, h, n

	ind = ind_values.indices
	if ind.shape[1] == 1:
		ind = ind.expand(-1, kv_heads, -1) # b, h, n
	return ind
 

def sel_kv(self, idx, key, value, mask):
	# always include the last G KV
	# key_ori = key
	# value_ori = value
 
	# key_new = key[:, :, -1:, :]
	# value_new = value[:, :, -1:, :]
	# key_sel = key[:, :, :-1, :]
	# value_sel = value[:, :, :-1, :]
 
	bsz = key.shape[0]
	kv_num = key.shape[1]
	head_dim = key.shape[3]
	src_len = key.shape[2] - 1
 
	if self.token_group == 0:
		tail_len = 0
	else:
		tail_len = src_len % self.token_group

	# print(f"src_len={src_len}, tail_len={tail_len}", flush=True)
	
	sel_len = src_len - tail_len
	# key_tail = key_sel[:, :, sel_len:, :]
	# value_tail = value_sel[:, :, sel_len:, :]

	# generate mask
	mask_sel = torch.zeros(bsz, kv_num, key.shape[2], dtype=torch.bool, device=key.device)	

	if self.token_group == 0:
		mask_sel[:, :, :sel_len].scatter_(2, idx, True)
	else:
		idx = idx.view(bsz, kv_num, -1, 1).expand(-1, -1, -1, self.token_group) # b, h, n
		mask_sel[:, :, :sel_len].view(bsz, kv_num, -1, self.token_group).scatter_(2, idx, True)
	

	mask_sel[:, :, sel_len:] = True
  
	fetch_num = mask_sel.float().sum(dim=2).mean().round().to(torch.int32).item() - 1
 
	if not hasattr(self, "fetch_num_list"):
		self.fetch_num_list = []
	self.fetch_num_list.append(fetch_num)
	
	if not hasattr(self, "density_list"):
		self.density_list = []
	self.density_list.append(fetch_num / src_len)

	mask_sel = mask_sel.view(bsz, kv_num, -1, 1).expand(-1, -1, -1, head_dim)
 
	key_sel = key[mask_sel].view(bsz, kv_num, -1, head_dim)
	value_sel = value[mask_sel].view(bsz, kv_num, -1, head_dim)
 
	assert mask is None, f"mask {mask} != None"
 
	return key_sel, value_sel, mask
		



def save_hidden(self, tg_len, hidden_states):
	if hasattr(self, 'lr_att_mode') and (self.lr_att_mode.startswith('infinigen') or self.lr_att_mode.startswith('lr_proj') or self.lr_att_mode.startswith('loki')):
		if self.eval_mode == 'eager':
			self.cur_hidden = hidden_states.clone()
		elif tg_len == 1:
			# only need hidden states when decoding
			self.cur_hidden = hidden_states.clone()



def pre_attention(self, tg_len, key_states, value_states, attention_mask, apply_rotary_pos_emb_partial):
	
	if not hasattr(self, 'lr_att_mode'):
		return key_states, value_states, attention_mask

	if self.lr_att_mode == 'kv-quan':
		attention_mask_sel = attention_mask
		key_states_sel, value_states_sel = quan_kv(self, key_states, value_states, is_prefill = tg_len>1)
  
	elif self.lr_att_mode.startswith('infinigen') or self.lr_att_mode.startswith('lr_proj') or self.lr_att_mode.startswith('loki'):
		if tg_len == 1 and (hasattr(self, 'prev_hidden') or (hasattr(self, 'use_emb') and self.use_emb)):
			idx = speculate_attention(self, key_states.shape[2], apply_rotary_pos_emb_partial)
			key_states_sel, value_states_sel, attention_mask_sel = sel_kv(self, idx, key_states, value_states, 
																			attention_mask)
		else:
			key_states_sel, value_states_sel, attention_mask_sel = key_states, value_states, attention_mask
		self.lr_k_cache = lr_kcache_func(self, key_states, is_prefill = tg_len>1)   
	else:
		return key_states, value_states, attention_mask

	return key_states_sel, value_states_sel, attention_mask_sel


def save_prev_hidden(self, idx):
	if hasattr(self.layers[0].self_attn, "lr_att_mode") and (self.layers[0].self_attn.lr_att_mode.startswith('infinigen') \
                                                          	or self.layers[0].self_attn.lr_att_mode.startswith('lr_proj') \
                                                           	or self.layers[0].self_attn.lr_att_mode.startswith('loki')):
		if idx >= self.start_idx and idx < self.config.num_hidden_layers - 1:
			if self.layers[idx].self_attn.eval_mode == 'eager':
				self.layers[idx + 1].self_attn.prev_hidden = self.layers[idx].self_attn.cur_hidden
			else:
				if hasattr(self.layers[idx].self_attn, "cur_hidden"):
					self.layers[idx + 1].self_attn.prev_hidden = self.layers[idx].self_attn.cur_hidden
 

def quan_kv(self, key_states, value_states, is_prefill):

	assert self.lr_att_mode == 'kv-quan', f"lr_att_mode {self.lr_att_mode} != kv-quan"

	qmax = (2 ** self.qbit) - 1
	group_size = self.qgroup
	num_groups = int(self.head_dim / group_size)

	s_end = key_states.shape[2]
 
	if is_prefill:
		s_start = 0
	else:
		s_start = s_end - 1

	for i in range(num_groups):
		start = i * group_size
		end = start + group_size

		key_max = (torch.max(key_states[:, :, s_start:s_end, start:end], dim=-1, keepdim = True))[0].repeat(1, 1, 1, group_size)
		key_min = (torch.min(key_states[:, :, s_start:s_end, start:end], dim=-1, keepdim = True))[0].repeat(1, 1, 1, group_size)
		key_delta = key_max - key_min

		value_max = (torch.max(value_states[:, :, s_start:s_end, start:end], dim=-1, keepdim = True))[0].repeat(1, 1, 1, group_size)
		value_min = (torch.min(value_states[:, :, s_start:s_end, start:end], dim=-1, keepdim = True))[0].repeat(1, 1, 1, group_size)
		value_delta = value_max - value_min

		### Quantize
		key_states[:, :, s_start:s_end, start:end] = torch.round(((key_states[:, :, s_start:s_end, start:end] - key_min) / key_delta) * qmax)
		value_states[:, :, s_start:s_end, start:end] = torch.round(((value_states[:, :, s_start:s_end, start:end] - value_min) / value_delta) * qmax)

		### Dequantize
		key_states[:, :, s_start:s_end, start:end] = (key_delta * key_states[:, :, s_start:s_end, start:end] / qmax) + key_min
		value_states[:, :, s_start:s_end, start:end] = (value_delta * value_states[:, :, s_start:s_end, start:end] / qmax) + value_min

	return key_states, value_states


