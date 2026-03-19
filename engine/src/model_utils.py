import torch
import vllm
import torch.nn.functional as F
from torch.func import vmap

def batched_isin(a, b, assume_unique=False, invert=False):
    return vmap(lambda x, y: torch.isin(x, y, assume_unique=assume_unique, invert=invert))(a, b)
    # return torch.isin(x, y, assume_unique=assume_unique, invert=invert)

def rms_norm(x, w, eps=1e-5, out=None):
	if out is None:
		out = torch.empty_like(x)
	# print("rms_norm", flush=True)
	vllm._custom_ops.rms_norm(out, x, w, eps)   
	return out

def llama_mlp_func(inputs, wgate, wup, wdown):
	gate = F.linear(inputs, wgate)
	up = F.linear(inputs, wup)
	down = F.linear(torch.nn.functional.silu(gate) * up, wdown)
	return down

def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)

dummy = None

def apply_rotary_pos_emb(q, k, arg0, arg1, layout):
	assert layout == 'bshd'
	# bshd layout
	if q is None:
		return apply_rotary_pos_emb_pytorch(q, k, arg0, arg1, layout)
	else:
		if k is None:
			global dummy
			if dummy is None:
				dummy = torch.empty_like(q)
			k = dummy
		head_dim = q.shape[-1]
		return apply_rotary_pos_emb_vllm(q, k, arg0, arg1, head_dim)

def apply_rotary_pos_emb_vllm(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, cos_sin_cache, head_dim) -> torch.Tensor:
	vllm._custom_ops.rotary_embedding(position_ids, q, k, head_dim, cos_sin_cache, True)
	return q, k

def apply_rotary_pos_emb_pytorch(q, k, cos, sin, layout):
	"""Applies Rotary Position Embedding to the query and key tensors.

	Args:
		q (`torch.Tensor`): The query tensor.
		k (`torch.Tensor`): The key tensor.
		cos (`torch.Tensor`): The cosine part of the rotary embedding.
		sin (`torch.Tensor`): The sine part of the rotary embedding.
		position_ids (`torch.Tensor`, *optional*):
			Deprecated and unused.
		unsqueeze_dim (`int`, *optional*, defaults to 1):
			The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
			sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
			that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
			k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
			cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
			the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
	Returns:
		`tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
	"""
	if layout == 'bhsd':
		unsqueeze_dim = 1
	else: # bshd
		unsqueeze_dim = 2
	cos = cos.unsqueeze(unsqueeze_dim)
	sin = sin.unsqueeze(unsqueeze_dim)
	if q is not None:
		q_embed = (q * cos) + (rotate_half(q) * sin)
	else:
		q_embed = None
	if k is not None:
		k_embed = (k * cos) + (rotate_half(k) * sin)
	else:
		k_embed = None
	return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int, h_dim) -> torch.Tensor:
	"""
	This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
	num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
	"""
	if n_rep == 1:
		return hidden_states
	
	shape_ = hidden_states.shape 
	# bhsd -> bh1sd -> bh + n_rep + sd -> b, -1, sd
	# bshd -> bsh1d -> bsh + n_rep + d -> bs,-1, d
	# sbhd -> sbh1d -> sbh + n_rep + d -> sb, -1, d
	hidden_states = hidden_states.unsqueeze(h_dim+1).expand(*shape_[:h_dim+1], n_rep, *shape_[h_dim+1:]).reshape(*shape_[:h_dim], -1, *shape_[h_dim+1:])
	return hidden_states
	
	if sbhd: # sbhd
		slen, batch, num_key_value_heads, head_dim = hidden_states.shape
		hidden_states = hidden_states[:, :, :, None, :].expand(slen, batch, num_key_value_heads, n_rep, head_dim)
		return hidden_states.reshape(slen, batch, num_key_value_heads * n_rep, head_dim)
	else: # bhsd
		batch, num_key_value_heads, slen, head_dim = hidden_states.shape
		hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
		return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
