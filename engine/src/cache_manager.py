
import torch
from vllm.attention.ops.paged_attn import PagedAttention
# from model_utils import batched_isin

class CacheManager:
    def __init__(self, num_kv_heads, head_dim, block_size, batch_size, token_budget, layer_num, reuse_budget, dtype=torch.float16):
        self.block_size = block_size
        self.batch_size = batch_size
        assert token_budget % block_size == 0, "token_budget must be a multiple of block_size"
        assert reuse_budget % block_size == 0, "reuse_budget must be a multiple of block_size"
        self.reuse_budget = reuse_budget
        assert reuse_budget >= token_budget or reuse_budget == 0
        # shared between all layers
        self.token_budget = token_budget
        self.layer_num = layer_num
        self.max_seq_len = token_budget + block_size
        self.num_kv_heads = num_kv_heads
        self.x_size = 16 // dtype.itemsize
        self.hdg_size = num_kv_heads * head_dim * block_size
        if reuse_budget > 0:
            # reuse meta 
            self.reuse_meta_indices = {i: torch.full((self.batch_size, reuse_budget // block_size), -1, dtype=torch.int32, device='cuda') for i in range(layer_num)}
            self.max_blocks_per_seq = (reuse_budget // block_size) * layer_num
            self.reuse_blk_num = reuse_budget // block_size
        else:
            self.max_blocks_per_seq = token_budget // block_size
        # for each layer, we have an unique rolling bufffer of block_size
        self.max_blocks_per_seq += layer_num
        self.shared_max_blocks_per_seq = self.max_blocks_per_seq - layer_num
        self.num_blocks = self.max_blocks_per_seq * self.batch_size
        self.kv_cache = torch.zeros(2, self.num_blocks, num_kv_heads, head_dim, block_size, dtype=dtype, device='cuda')
        self.key_cache = self.kv_cache[0].view(self.num_blocks, num_kv_heads, head_dim // self.x_size, block_size, self.x_size)
        self.value_cache = self.kv_cache[1]
        self.shared_kv_cache = self.kv_cache.view(2, self.batch_size, -1, self.hdg_size)[:, :, :self.shared_max_blocks_per_seq]
        self.layer_kv_cache = self.kv_cache.view(2, self.batch_size, -1, self.hdg_size)[:, :, self.shared_max_blocks_per_seq:]
        self.slot_mappings = torch.arange(self.shared_max_blocks_per_seq*block_size, 
                                          self.max_blocks_per_seq*block_size, 
                                          dtype=torch.long, device='cuda').view(1, self.layer_num, block_size)
        self.slot_mappings = self.slot_mappings + torch.arange(self.batch_size, dtype=torch.long, device='cuda').view(-1, 1, 1) * self.max_blocks_per_seq * block_size
        self.slot_mappings = self.slot_mappings.permute(1, 2, 0).contiguous()  # [layer_num, block_size, batch_size]
        self.k_scale = torch.ones(self.batch_size, dtype=torch.float32, device='cuda')
        self.v_scale = torch.ones(self.batch_size, dtype=torch.float32, device='cuda')
        self.seq_lens = torch.tensor([self.token_budget]*self.batch_size, dtype=torch.int32, device='cuda')
        # [num_seqs]
        # [batch, max_num_blocks_per_seq]
        full_block_tables = torch.arange(self.batch_size * self.max_blocks_per_seq, 
                                        dtype=torch.int32, device='cuda').view(self.batch_size, self.max_blocks_per_seq)
        if reuse_budget > 0:
            self.reuse_layer_block_tables = full_block_tables[:, :self.shared_max_blocks_per_seq].clone()
            self.reuse_layer_block_tables = self.reuse_layer_block_tables.view(self.batch_size, self.layer_num, -1)
            self.block_tables = torch.cat([self.reuse_layer_block_tables[:, 0], # layer_id=0
                                           full_block_tables[:, self.shared_max_blocks_per_seq:self.shared_max_blocks_per_seq+1]], dim=1)
            
        else:
            self.block_tables = full_block_tables[:, :self.shared_max_blocks_per_seq+1].clone()
            self.scatter_mask = None
        self.layer_block_tables = full_block_tables[:, -self.layer_num:].clone()
        self.reuse_info = {i:[] for i in range(self.layer_num)}
    
    def init_remaining(self, kv_len):
        self.remaining = kv_len % self.block_size
    
    def update_seqlens(self):
        # only call this at the beginging of each generation 
        self.remaining = self.remaining % self.block_size
        self.remaining += 1
        self.cur_seq_lens = self.seq_lens + self.remaining
    
    def get_seqlens(self):
        return self.cur_seq_lens
        
    def update_rolling_buffer(self, k, v, layer_idx, init=False, s=None):
        if init:
            assert k.dim() == 3 and v.dim() == 3, "k and v must be 3D tensors"
            # bs, h, d
            # lens = k.shape[0] // self.batch_size
            PagedAttention.write_to_paged_cache(
                k, v,
                self.key_cache, self.value_cache,
                self.slot_mappings[layer_idx, :s], 
                "auto", self.k_scale, self.v_scale)
        else:        
            PagedAttention.write_to_paged_cache(
                k, v,
                self.key_cache, self.value_cache,
                self.slot_mappings[layer_idx, self.remaining-1], 
                "auto", self.k_scale, self.v_scale)

    def get_block_tables(self, layer_idx):
        if self.reuse_budget > 0:
            self.block_tables[:, :-1] = self.reuse_layer_block_tables[:, layer_idx]
        self.block_tables[:, -1] = self.layer_block_tables[:, layer_idx]
        return self.block_tables

    def get_load_buffer(self, layer_idx, indices):
        
        # assert indices.max() < 50000
        
        if self.reuse_budget == 0:
            return indices.int(), self.shared_kv_cache, False
        else:
            # assume_unique = not is_first
            indices = indices.int()
            # b, n, 1 == b, 1, m
            hit_mask = self.reuse_meta_indices[layer_idx].unsqueeze(2) == indices.unsqueeze(1)
            hit_mask0 = hit_mask.any(dim=1)  # b, m
            # hit_mask0 = batched_isin(indices, self.sreuse_meta_indices[layer_idx], assume_unique=assume_unique, invert=False)
            # if not (hit_mask0 == hit_mask.any(dim=1)).all():
            #     print(f"Layer {layer_idx} hit_mask0 and hit_mask mismatch!")
            #     print(hit_mask0, flush=True)
            #     print(hit_mask.any(dim=1), flush=True)
            
            missed_indices = indices[~hit_mask0]
            
            reuse_rate = 1 - missed_indices.numel() / indices.numel()
            # print(f"Layer {layer_idx} reuse rate: {reuse_rate:.2f}", flush=True)
            self.reuse_info[layer_idx].append(reuse_rate)
            
            if missed_indices.numel() > 0:
                self.scatter_mask = ~hit_mask.any(dim=2) # b, n
                # self.scatter_mask = batched_isin(self.reuse_meta_indices[layer_idx], indices, assume_unique=assume_unique, invert=True)
                
                # if not (self.scatter_mask == ~hit_mask.any(dim=2)).all():
                #     print(f"Layer {layer_idx} scatter_mask and hit_mask mismatch!")
                #     print(self.scatter_mask, flush=True)
                #     print(~hit_mask.any(dim=2), flush=True)
                
                # fill the indices with -1 if hit
                indices[hit_mask0] = -1  # b, m
                # update reuse_meta_indices
                self.reuse_meta_indices[layer_idx][self.scatter_mask] = missed_indices
                self.scatter_mask = self.scatter_mask.unsqueeze(0).unsqueeze(-1).expand(2, -1, -1, self.hdg_size)
            
            return indices, self.shared_kv_cache[:, :, self.reuse_blk_num*layer_idx:self.reuse_blk_num*(layer_idx+1)], missed_indices.numel() == 0

    def set_kv_cache(self, kv_data_disk):
        # kv_data_disk: b, 2, n, h, d, g
        self.key_cache.view(self.batch_size, -1, *self.v_cache_shape[1:])[:, :self.shared_max_blocks_per_seq].copy_(kv_data_disk[:, 0])
        self.value_cache.view(self.batch_size, -1, *self.v_cache_shape[1:])[:, :self.shared_max_blocks_per_seq].copy_(kv_data_disk[:, 1])

    def set_kv_cache2(self, kv_data, bh_start, bh_end, non_blocking, layer_idx):
        # kv_data: b, n, 2, hdg -> 2, b, n, hdg
        # self.shared_key_cache[bh_start:bh_end].copy_(kv_data[:, :, 0], non_blocking=non_blocking)
        # self.shared_value_cache[bh_start:bh_end].copy_(kv_data[:, :, 1], non_blocking=non_blocking)
        if self.reuse_budget == 0:
            self.shared_kv_cache[:, bh_start:bh_end].copy_(kv_data.permute(2, 0, 1, 3), non_blocking=non_blocking)
        else:
            scatter_mask = self.scatter_mask
            # 2, b, n, hdg
            self.shared_kv_cache[:, bh_start:bh_end][scatter_mask[:, bh_start:bh_end]] = kv_data.permute(2, 0, 1, 3).cuda()
    
    def reshape(self, k, v, lens):
        # b, s//g, g, h, d//x, x -> b, s//g, h, d//x, g, x
        k = k.view(k.shape[0], lens//self.block_size, self.block_size, self.num_kv_heads, -1, self.x_size).permute(0, 1, 3, 4, 2, 5).reshape(k.shape[0], lens//self.block_size, -1)
        # b, s//g, g, h, d -> b, s//g, h, d, g
        v = v.view(v.shape[0], lens//self.block_size, self.block_size, self.num_kv_heads, -1).permute(0, 1, 3, 4, 2).reshape(v.shape[0], lens//self.block_size, -1)
        return k, v
    
    def get_recentkv(self, layer_idx):
        # 2, b, hdg
        recentkv = self.layer_kv_cache[:,:,layer_idx]
        # b, h, d//x, g, x -> b, g, hd
        k = recentkv[0].view(self.batch_size, self.num_kv_heads, -1, self.block_size, self.x_size).permute(0, 3, 1, 2, 4).reshape(self.batch_size, self.block_size, -1)
        return recentkv.transpose(0, 1).unsqueeze(1), k
    
    