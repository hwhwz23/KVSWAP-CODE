import torch
import torch.nn.functional as F
from model_utils import apply_rotary_pos_emb, repeat_kv, rms_norm
import time
import nvtx

@torch.inference_mode()
def lr_kcache_func(lr_proj_mode, lr_k, key, pos, kv_heads, kv_groups, partial_index=None, skew_matrix=None, lr_kproj=None):
	b, s, hd = key.shape
	if lr_proj_mode == 'base':
		# lr_k: n, b, h, d'
		# b, s, h, d -> s, b, h, d -> s, b, h, d'
		# print(partial_index.shape, key.shape, flush=True)
						# b, s, h, d @ h, d, d -> b, s, h, d
		key = torch.einsum('bshd,hde->bshe', key.view(b, s, kv_heads, -1), skew_matrix.to(key.dtype)[::kv_groups])
		torch.gather(key.transpose(0, 1), 
					 3, 
					 partial_index[::kv_groups].unsqueeze(0).unsqueeze(0).expand(s, b, -1, -1), 
					out=lr_k[pos[0]:pos[1]])
	elif lr_proj_mode == 'lr_proj_sh':
		# b, s, h, d -> s, b, h, d -> s, b, d @ d, r -> s, b, r
		torch.matmul(key.view(b, s, kv_heads, -1).transpose(0, 1).mean(dim=2), lr_kproj,
					 out=lr_k[pos[0]:pos[1]])
	elif lr_proj_mode in ('lr_proj_mh', 'lr_proj2_mh'):
		# b, s, hd -> s, b, hd @ hd, r -> s, b, r
		torch.matmul(key.view(b, s, -1).transpose(0, 1), lr_kproj, 
					 out=lr_k[pos[0]:pos[1]])
	else:
		raise ValueError(f"Invalid lr_proj_mode: {lr_proj_mode}")

@torch.inference_mode()
def speculate_attention(lr_proj_mode, hidden, next_partial_wq, next_skew_matrix, next_qproj, next_qnorm, next_lr_k_proj, next_lr_kcache, pos_emb, next_partial_idx,
						scaling, n_head, kv_rep, alpha, max_num_kv, token_group, score_mode):
		
	b = hidden.shape[0]
	dtype = hidden.dtype
	###########################################################################
	if lr_proj_mode == 'base':
		query = F.linear(hidden, next_partial_wq.to(dtype), bias=None) # b, 1, h*d'
		query = query.view(b, 1, n_head, -1) # b, 1, h, d'
		if next_qnorm is not None:
			query = rms_norm(query, next_qnorm.data.to(dtype))
		if pos_emb is not None:
			query, _ = apply_rotary_pos_emb(query, None, *pos_emb, layout='bshd') # b, 1, h, d'
			# partial_idx: h, d' -> 1, 1, h, d' -> b, 1, h, d'
			query = torch.einsum('bshd,hde->bshe', query, next_skew_matrix.to(query.dtype))
			query = query.gather(3, next_partial_idx.view(1, 1, n_head, -1).expand(b, -1, -1, -1))
		# key: s, b, h, d -> b, h, s, d 
		key = next_lr_kcache.permute(1, 2, 0, 3)
		src_len = key.shape[2]
		key = repeat_kv(key, kv_rep, h_dim=1) # b, h, s, d
		key = key.view(b*n_head, src_len, -1)
		# b, 1, h, d' -> b*h, 1, d
		query = query.reshape(b*n_head, 1, -1) # bh, 1, d
		# b*h, 1, d @ b*h, d, s -> b*h, 1, s
		out_heads = n_head
		attn = torch.bmm(query, key.transpose(1, 2)) * scaling
	elif lr_proj_mode.startswith("lr_proj"):
		if next_lr_k_proj.dtype != dtype:
			next_lr_k_proj = next_lr_k_proj.to(dtype)
		kv_num = n_head // kv_rep
		# next_lr_kcache: s, b, r
		src_len = next_lr_kcache.shape[0]
		key = next_lr_kcache.transpose(0, 1) # b, s, r
		query = F.linear(hidden, next_qproj.to(dtype), bias=None) # b, 1, h*d
		query = query.view(b, 1, n_head, -1) # b, 1, h, d
		if next_qnorm is not None:
			query = rms_norm(query, next_qnorm.data.to(dtype))
		if pos_emb is not None: # pre processing query
			query, _ = apply_rotary_pos_emb(query, None, *pos_emb, layout='bshd') # b, 1, h, d
		if lr_proj_mode == 'lr_proj_sh':
			raise NotImplementedError
			out_heads = 1
		elif lr_proj_mode == 'lr_proj_mh':
			# b, kv_num, kv_rep, d -> b, kv_num, d -> b, kv_num, 1, d
			query = query.view(b, kv_num, kv_rep, -1).mean(dim=2).unsqueeze(-2)
			# b, kv_num, 1, d @ kv_num, d, r -> b, kv_num, 1, r
			query = query @ next_lr_k_proj.view(kv_num, -1, key.shape[-1])
			# b, kv_num, 1, r -> b, kv_num, r -> b, r
			query = query.view(b, kv_num, key.shape[-1]).mean(dim=1)
			out_heads = 1
		elif lr_proj_mode == 'lr_proj2_mh':
			# b, s, r -> b, 1, s, r -> b, h, s, r
			key = key.unsqueeze(1).expand(-1, n_head, -1, -1) # b, h, s, r
			# b, kv_num, kv_rep, d -> b, kv_rep, kv_num, 1, d
			query = query.view(b, kv_num, kv_rep, -1).transpose(1, 2).unsqueeze(-2)
			# b, kv_rep, kv_num, 1, d @ kv_num, d, r -> b, kv_rep, kv_num, 1, r
			query = query @ next_lr_k_proj.view(kv_num, -1, key.shape[-1])
			query = query.view(b, kv_rep, kv_num, key.shape[-1]).transpose(1, 2)
			out_heads = n_head
		else:
			raise ValueError(f"Invalid lr_proj_mode: {lr_proj_mode}")
		key = key.reshape(b*out_heads, src_len, -1) # bh, s, r
		query = query.reshape(b*out_heads, 1, -1)
		attn = torch.bmm(query, key.transpose(1, 2)) * scaling
	else:
		raise ValueError(f"Invalid lr_proj_mode: {lr_proj_mode}")
	###########################################################################
	
	with nvtx.annotate("get fetch_num"):
		if score_mode == 'max':
			max_ = torch.max(attn, dim=-1, keepdim=True)[0] # bh, 1, 1 
			thr_ = (max_ - alpha) # bh, 1, 1
			mask = attn >= thr_ # bh, 1, s
			# bh, 1, s -> bh, 1 -> 1
			fetch_num = torch.sum(mask, dim=-1)
			fetch_num = fetch_num.view(b, out_heads, 1) # b, h, 1
			fetch_num = torch.mean(fetch_num.to(attn.dtype)).round().to(torch.int32).item()
			fetch_num = min(fetch_num, max_num_kv)
		else:
			fetch_num = max_num_kv
			
	tail_len = attn.shape[-1] % token_group
	assert tail_len == 0, f"{token_group} {attn.shape} != 0"
	attn = attn.view(b, out_heads, -1, token_group)
	if out_heads > 1:
		attn = attn.mean(dim=1) 
	attn = attn.amax(-1) 
	# b, 1, n / b, n
	ind = attn.topk(fetch_num // token_group, dim=-1, largest=True, sorted=False).indices # b, n
	return ind.view(b, -1) # b, n


def merge_qk_weight(weight, skew_matrix, num_heads, head_dim):    
	weight_t = weight.t()
	for hi in range(num_heads):
		start_idx = hi * head_dim
		end_idx = start_idx + head_dim
		weight_t[:, start_idx:end_idx] = weight_t[:, start_idx:end_idx] @ skew_matrix[hi]
	return weight_t.t()

def get_partial_q_weight(weight, partial_idx, num_heads, head_dim): 
	# Do, Di -> h, d, Di
	# h, d' -> h, d', 1
	din = weight.shape[-1]
	weight = weight.view(num_heads, head_dim, din).gather(1, partial_idx.unsqueeze(-1).expand(-1, -1, din))
	return weight.view(-1, din)

