################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


import torch
import torch.nn.functional as F
import gc
import time

import transformers
from transformers import Qwen2Config, AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLRotaryEmbedding, Qwen2_5_VLConfig
transformers.logging.set_verbosity_error()

from .tensor_op import layer_norm
# from .prompt_template import Templates, Chat_Templates
from .base import LLM
from typing import Any

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



class Qwen25Layer:
    def __init__(self, layer_idx) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.bq :torch.Tensor = None
        self.bk :torch.Tensor = None
        self.bv :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: Qwen2_5_VLDecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        # bias for qkv
        self.bq = hf_layer.self_attn.q_proj.bias.detach()
        self.bk = hf_layer.self_attn.k_proj.bias.detach()
        self.bv = hf_layer.self_attn.v_proj.bias.detach()

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

        self.bq = self.bq.to(device, non_blocking=True)
        self.bk = self.bk.to(device, non_blocking=True)
        self.bv = self.bv.to(device, non_blocking=True)

class Qwen2_5_VL_LLM(LLM):
    def __init__(self, 
        model_name: str = "NULL",
        hf_model: Any = None,
        batch_size :int = 1,
        max_length :int = 64*1024, 
        # device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        sparse_budget: int = 2048,
        rank=160,
        chunk_size=8,
        minference=False, **kwargs) -> None:
        
        assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        # self.device = device
        self.device = 'cuda:0'
        self.dtype = dtype
        self.config = Qwen2_5_VLConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.init_parameters(hf_model)
        self.attn_mode = attn_mode
        self.minference = minference

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=self.config).cuda()
        self.rope_scaling = self.config.rope_scaling

        self.init_kv_cache(sparse_budget, rank, chunk_size, self.config, **kwargs)


    def init_parameters(self, hf_model):
        # hf_model = Qwen2ForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.layers :list[Qwen25Layer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen25Layer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: Qwen25Layer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, buffer.wq, bias=buffer.bq)
        key_states = F.linear(hidden_states, buffer.wk, bias=buffer.bk)
        value_states = F.linear(hidden_states, buffer.wv, bias=buffer.bv)
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        return query_states, key_states, value_states
    
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: Qwen25Layer
    ):  
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = F.linear(hidden_states, buffer.up_proj)
        gate = F.silu(F.linear(hidden_states, buffer.gate_proj))
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states
    
    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # bh, s
        position_ids = position_ids.view(-1, position_ids.size(-1))
        # 3, 1, s, 128 -> 3, bh, s, 128
        last_dim = self.cached_cos_sin_cache[0].shape[-1]
        cos = self.cached_cos_sin_cache[0].expand(-1, position_ids.size(0), -1, -1).gather(2, position_ids.unsqueeze(0).unsqueeze(-1).expand(3, -1, -1, last_dim))
        sin = self.cached_cos_sin_cache[1].expand(-1, position_ids.size(0), -1, -1).gather(2, position_ids.unsqueeze(0).unsqueeze(-1).expand(3, -1, -1, last_dim))
        # 3, bh, s, 128 

        mrope_section = self.rope_scaling["mrope_section"]
        mrope_section = mrope_section * 2
        # bh, s, 128
        # [bsz, 8, sparse_budget, 128]
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).view(*x.shape)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).view(*x.shape)
        q_embed = (x * cos) + (rotate_half(x) * sin)
        
        return q_embed
 
    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_embeddings) -> torch.Tensor:
        cos, sin = position_embeddings
        
        if self.cached_cos_sin_cache is None:
            self.cached_cos_sin_cache = (cos, sin)
        else:
            self.cached_cos_sin_cache = (
                torch.cat([self.cached_cos_sin_cache[0], cos], dim=-2),
                torch.cat([self.cached_cos_sin_cache[1], sin], dim=-2)
            )
        
        unsqueeze_dim = 1
        mrope_section = self.rope_scaling["mrope_section"]
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
        