import torch
import torch.nn.functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
	"""
	This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
	num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
	"""
	batch, num_key_value_heads, slen, head_dim = hidden_states.shape
	if n_rep == 1:
		return hidden_states
	hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
	return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

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

def ind_freq(ind):
	# ind: b, h, fn -> b, 
	num_heads = ind.shape[1]
	fn = ind.shape[2]
	ind = ind.view(ind.shape[0], -1) # b, h*fn
	max_value = ind.max()
	counts = torch.zeros(ind.shape[0], max_value+1, dtype=torch.long, device=ind.device)
	counts.scatter_add_(1, ind, torch.ones_like(ind))
	_, topk_indices = counts.topk(fn, dim=1) # b, fn
	ind = topk_indices.unsqueeze(1).repeat(1, num_heads, 1)
	return ind

def kv_cache_mask(self, attn, out_heads):
	assert torch.isfinite(attn).all(), f"attn has inf or nan"
	batch_heads, tgt_len, src_len = attn.shape
 
 	# num_heads = self.config.num_attention_heads 
	# kv_groups = self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
	# kv_heads = num_heads // kv_groups
 
	num_heads = self.config.num_attention_heads
	bsz = batch_heads // out_heads
	if tgt_len > 1:
		# need to exclude the last one
		attn = attn + torch.triu(torch.full(attn.shape, -10000, dtype=attn.dtype, device=attn.device), diagonal=0)            
	else:
		attn[:, :, -1] += -10000
	
	fetch_max = int(src_len * self.budget) 
 
	if self.att_score_mode == 'max':
		max = torch.max(attn, dim=-1, keepdim = True)[0]
		threshold = max - self.alpha
		mask = (attn >= threshold) # bh, tgt_len, src_len
		fetch_num = torch.sum(mask, dim = -1) # bh, tgt_len
		fetch_num = fetch_num.view(bsz, out_heads, tgt_len) # b, h, tgt_len
		fetch_num = torch.mean(fetch_num.to(attn.dtype), dim=1).round().to(torch.int32) # need to fetch same amount for each head
		fetch_num = torch.where(fetch_num >= fetch_max, fetch_max, fetch_num) # b, tgt_len
	else:
		fetch_num = torch.tensor(fetch_max, dtype=torch.int32, device=attn.device).view(1, 1).expand(bsz, tgt_len)

	kv_groups = self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1

	def get_fetch_mask(attn_slice, fetch_num_slice):
		max_k = fetch_num_slice.max().item()
		if self.token_group > 1:
			pad_len = attn_slice.shape[-1] % self.token_group
			if pad_len > 0:
				pad_ind = torch.arange(attn_slice.shape[-1]-pad_len, attn_slice.shape[-1], device=attn_slice.device).view(1, 1, -1)
				pad_ind = pad_ind.expand(attn_slice.shape[0], attn_slice.shape[1], -1)
				attn_slice_pad = attn_slice[:, :, :attn_slice.shape[-1]-pad_len]
			else:
				attn_slice_pad = attn_slice
			attn_slice_pad = attn_slice_pad.view(attn_slice.shape[0], attn_slice.shape[1], -1, self.token_group).amax(-1)
			assert torch.isfinite(attn_slice_pad).all(), f"attn_slice_pad={attn_slice_pad}"
			max_k = (max_k + self.token_group - 1) // self.token_group
			max_k = min(max_k, attn_slice_pad.shape[-1])
			_, ind = torch.topk(attn_slice_pad, k=max_k, dim=-1) 
			max_k = max_k * self.token_group
			ind = ind.unsqueeze(-1) * self.token_group + torch.arange(self.token_group, device=ind.device).view(1, 1, 1, -1)
			ind = ind.view(ind.shape[0], ind.shape[1], -1)
		else:	
			pad_len = 0
			_, ind = torch.topk(attn_slice, k=max_k, dim=-1) 
		if pad_len > 0:
			ind = torch.cat((ind, pad_ind), dim=-1)
			max_k += pad_len
		if out_heads > 1: # true_key, lr_proj2_mh, base
			ind = merge_heads_gqa(ind, kv_groups)
		return ind

	attn = attn.view(bsz, out_heads, tgt_len, src_len)
	fetch_mask = torch.zeros_like(attn)

	if tgt_len == 1:
		assert fetch_num.ndim == 2, f'fetch_num.ndim={fetch_num.ndim}'
		assert fetch_num.shape[1] == 1, f'fetch_num.shape[1]={fetch_num.shape[1]}'
		ind = get_fetch_mask(attn[:, :, 0, :-1], fetch_num[:, 0])
		fetch_mask[:, :, 0, :-1].scatter_(-1, ind, True) 
		fetch_mask[:, :, 0, -1] = 1 # always fetch the last one
		if hasattr(self, 'density_gen'):
			# calculate swap ratio instead
			self.density_gen.append(((fetch_mask.float().sum().cpu().item() / bsz / out_heads) - 1) / (src_len - 1))
			self.fetch_num_gen.append(fetch_num.float().mean().cpu().item())
		else:
			self.density = ((fetch_mask.float().sum().cpu().item() / bsz / out_heads) - 1) / (src_len - 1)
			self.fetch_num = fetch_num.float().mean(0).cpu()
	else:
		fetch_mask[:, :, :fetch_max] = torch.tril(torch.ones((fetch_max, src_len), dtype=attn.dtype, device=attn.device)).unsqueeze(0).unsqueeze(0)
		for i in range(fetch_max, src_len):
			ind = get_fetch_mask(attn[:, :, i, :i], fetch_num[:, i])
			fetch_mask[:, :, i, :i].scatter_(-1, ind, True) 
			fetch_mask[:, :, i, i] = 1 # always fetch the last one
		# self.density = fetch_mask.float().sum().item() / bsz / out_heads / (tgt_len * (tgt_len + 1) / 2)
		# calculate swap ratio instead
		assert tgt_len == src_len, f"tgt_len={tgt_len} != src_len={src_len}"
		self.density = fetch_mask.float().sum().item() / bsz / out_heads / (src_len * (src_len - 1) / 2) - 2 / (src_len - 1)
		self.fetch_num = fetch_num[:, fetch_max:src_len].float().mean(0).cpu() 
   
	if hasattr(self, "save_fetch_mask") and self.save_fetch_mask:
		if hasattr(self, "saved_fetch_mask_list"):
			self.saved_fetch_mask_list.append(fetch_mask.bool().cpu())
		else:
			self.saved_fetch_mask_list = [fetch_mask.bool().cpu()]

	if hasattr(self, "save_fetch_attn") and self.save_fetch_attn:
		if hasattr(self, "fetch_attn_weights_list"):
			self.fetch_attn_weights_list.append(attn.cpu())
		else:	
			self.fetch_attn_weights_list = [attn.cpu()]
	
	if out_heads > 1:
		fetch_mask = fetch_mask.view(bsz * out_heads, tgt_len, src_len)
	else:
		fetch_mask = fetch_mask.view(bsz, out_heads, tgt_len, src_len).expand(-1, num_heads, -1, -1).view(bsz * num_heads, tgt_len, src_len)
	m_inf = torch.tensor([[-10000]], dtype=attn.dtype, device=attn.device)
	fetch_mask = torch.where(fetch_mask == 1, 0, m_inf)
	return fetch_mask


def lr_att(self, key, apply_rotary_pos_emb_partial=None, hidden=None):
	num_heads = self.config.num_attention_heads 
	kv_groups = self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
	kv_heads = num_heads // kv_groups
	if hasattr(self, 'prev_hidden') and hasattr(self, 'lr_att_mode'):
		if self.lr_att_mode == 'none':
			return None
		else:
			if hidden is None:
				if hasattr(self, 'use_cur_hidden') and self.use_cur_hidden:
					hidden = self.cur_hidden
				else:
					hidden = self.prev_hidden
			bsz = hidden.shape[0]
			if hasattr(self, 'only_last_q_kv_mask') and self.only_last_q_kv_mask:
				# tg_len = 1
				assert bsz == 1, f'only support bsz=1, bsz={bsz}'
				ori_tg_len = hidden.shape[1]
				hidden = hidden[:, -1:] # b, 1, d
			tg_len = hidden.shape[1]

			lr_q = self.q_proj(hidden).view(bsz, tg_len, num_heads, self.head_dim)
			if hasattr(self, 'q_norm'): # for qwen3
				lr_q = self.q_norm(lr_q)
			lr_q = lr_q.transpose(1, 2)
			if apply_rotary_pos_emb_partial is not None:
				lr_q, _ = apply_rotary_pos_emb_partial(lr_q, None)
				if hasattr(self, 'only_last_q_kv_mask') and self.only_last_q_kv_mask:
					lr_q = lr_q[:, :, -1:]
			if key.ndim == 3:
				src_len = key.shape[1]
			elif key.ndim == 4:
				src_len = key.shape[2]
			else:
				raise ValueError(f"key.shape={key.shape}")   

			if self.lr_att_mode.startswith('lr_proj'):
				lr_k = key.view(bsz, kv_heads, src_len, -1)
				lr_q = lr_q.view(bsz, num_heads, tg_len, -1)
				if self.lr_kproj.device != lr_k.device:
					self.lr_kproj = self.lr_kproj.to(lr_k.device)
				if self.fuse_group != 'none':
					if self.group_fuse_proj.device != lr_k.device:
						self.group_fuse_proj = self.group_fuse_proj.to(lr_k.device)
      
				if self.lr_att_mode == 'lr_proj_sh':
					lr_k = lr_k.mean(dim=1) # b, src_len, d
					lr_q = lr_q.mean(dim=1) # b, tg_len, d
					lr_k = torch.matmul(lr_k, self.lr_kproj) # b, src_len, r
					lr_q = torch.matmul(lr_q, self.lr_kproj) # b, tg_len, r
					if self.fuse_group != 'none':
						# group_fuse_proj: g*r, r
						# bsz, -1, g*r
						lr_k = lr_k.view(bsz, -1, self.group_fuse_proj.shape[0])
						# bsz, -1, g*r @ g*r, r -> bsz, -1, r
						lr_k = lr_k @ self.group_fuse_proj 
						lr_k = lr_k.view(bsz, -1, self.group_fuse_proj.shape[1])
						lr_k = lr_k.repeat_interleave(self.group_fuse_proj.shape[0] // self.group_fuse_proj.shape[1], dim=1) # bsz, src_len, r
						# b, tg_len, r -> b, tg_len, g*r @ g*r, r -> b, tg_len, r
						# lr_q = lr_q.repeat(1, 1, self.group_fuse_proj.shape[0] // lr_q.shape[-1]) @ self.group_fuse_proj
						# lr_q = lr_q.view(bsz, -1, self.group_fuse_proj.shape[1])
					out_heads = 1
				elif self.lr_att_mode == 'lr_proj2_mh': # not fuse heads
					lr_k = lr_k.transpose(1, 2).reshape(bsz, src_len, -1) # b, src_len, h*d
					lr_k = torch.matmul(lr_k, self.lr_kproj) # b, src_len, r
					lr_k = lr_k.view(bsz, 1, src_len, -1).expand(-1, num_heads, -1, -1) # b, h, src_len, r
					lr_q = lr_q.view(bsz, kv_heads, kv_groups, tg_len, -1).transpose(1, 3).unsqueeze(-2) # b, tg_len, hg, hkv, 1, d
					# b, tg_len, hg, hkv, 1, d @ hkv, d, r -> b, tg_len, hg, hkv, 1, r
					lr_q = lr_q @ self.lr_kproj.view(kv_heads, -1, lr_k.shape[-1])
					lr_q = lr_q.view(bsz, tg_len, kv_groups, kv_heads, lr_k.shape[-1]).transpose(1, 3)
					if self.fuse_group != 'none':
						pass
     
					out_heads = num_heads
				else: # lr_proj_mh
					lr_k = lr_k.transpose(1, 2).reshape(bsz, src_len, -1) # b, src_len, h*d
					lr_k = torch.matmul(lr_k, self.lr_kproj) # b, src_len, r
					lr_q = lr_q.view(bsz, kv_heads, kv_groups, tg_len, -1).mean(dim=2).transpose(1, 2).unsqueeze(-2) # b, tg_len, hkv, 1, d
					# b, tg_len, hkv, 1, d @ hkv, d, r -> b, tg_len, hkv, 1, r
					lr_q = lr_q @ self.lr_kproj.view(kv_heads, -1, lr_k.shape[-1])
					lr_q = lr_q.view(bsz, tg_len, kv_heads, lr_k.shape[-1]).mean(dim=2) # b, tg_len, r
					if self.fuse_group != 'none':
						pass
     
					out_heads = 1
			else: # true_key or base(lr_k only)
				lr_k = repeat_kv(key.view(bsz, kv_heads, src_len, -1), kv_groups)
				out_heads = num_heads
			if self.lr_att_mode == 'base':
				if self.skewing_matrix.device != lr_q.device or self.skewing_matrix.dtype != lr_q.dtype:
					self.skewing_matrix = self.skewing_matrix.to(lr_q.device).to(lr_q.dtype)
				lr_q = lr_q @ self.skewing_matrix.unsqueeze(0) # b, h, -1, d @ 1, h, d, d -> b, h, -1, d
				if self.skewing_mask.device != lr_q.device:
					self.skewing_mask = self.skewing_mask.to(lr_q.device)
				# lr_q: b, h, -1, d # mask: h, d -> 1, h, 1, d
				mask = self.skewing_mask.view(1, num_heads, 1, self.head_dim).expand_as(lr_q)
				lr_q = torch.where(mask.to(torch.bool), lr_q, torch.zeros_like(lr_q))
				lr_k = lr_k @ self.skewing_matrix.unsqueeze(0) # b, h, -1, d @ 1, h, d, d -> b, h, -1, d
				# self.skewing_mask = self.skewing_mask.cpu()    
			lr_k = lr_k.reshape(bsz * out_heads, src_len, -1)
			lr_q = lr_q.reshape(bsz * out_heads, tg_len, -1)
			if lr_q.device != lr_k.device:
				lr_q = lr_q.to(lr_k.device)
			lr_att = torch.bmm(lr_q, lr_k.transpose(1, 2)) * self.scaling
			kv_mask = kv_cache_mask(self, lr_att, out_heads)   
			if hasattr(self, 'only_last_q_kv_mask') and self.only_last_q_kv_mask:
				ori_kv_mask = torch.triu(torch.full((ori_tg_len, src_len), -10000, dtype=kv_mask.dtype, device=kv_mask.device), diagonal=1)  
				ori_kv_mask = ori_kv_mask.unsqueeze(0).repeat(kv_mask.shape[0], 1, 1)
				ori_kv_mask[:, -1, :] = kv_mask[:, -1, :]
				return ori_kv_mask
			else:
				return kv_mask    
	else:
		return None


def get_skewing_mask(self, query_states, key_states, ratio):
	weight_mask = torch.zeros_like(self.q_proj.weight.data.t())
	n = int(self.head_dim * ratio)
	head_num = self.config.num_attention_heads
	# Speculate major columns in Wq and Wk
	for head in range(head_num):
		start = head * self.head_dim
		end = (head+1) * self.head_dim
		skew_matrix = self.skewing_matrix[head]
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
	weight_mask = weight_mask[0].view(head_num, self.head_dim) # h, d
	ind = weight_mask.topk(n, dim=-1)[1] # h, n
	kv_groups = self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
	ind = merge_heads_gqa(ind.unsqueeze(0), kv_groups)[0] # h, n
	new_weight_mask = torch.zeros_like(weight_mask) # h, d
	new_weight_mask.scatter_(-1, ind, 1)
	return new_weight_mask

