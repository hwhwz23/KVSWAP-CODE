import os
import torch
import time

def zero_out(path, size):
    # Open unbuffered for direct I/O compatibility
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o777)
    try:
        # ensure i_size == size
        os.ftruncate(fd, size)
        # write in big chunks
        chunk_size = (1 << 20) * 32  # 32MB
        zero_chunk = b'\x00' * chunk_size
        written = 0
        while written < size:
            to_write = min(chunk_size, size - written)
            os.write(fd, zero_chunk[:to_write])
            written += to_write
        os.fsync(fd)
        dfd = os.open(os.path.dirname(path), os.O_DIRECTORY)
        os.fsync(dfd)
        os.close(dfd)
    finally:
        os.close(fd)


MAX_KV_SIZE = os.environ.get('MAX_ALLOC_KV_SIZE')
if MAX_KV_SIZE is None:
    raise ValueError("MAX_ALLOC_KV_SIZE is not set")
MAX_KV_SIZE = int(MAX_KV_SIZE)

# MAX_KV_SIZE = 1024*1024*768
BLOCK_DEV_SIZE = 512

class DiskCache:
    def __init__(self, cache_dir, layer_num, bsz, kv_heads, chunk_num, chunk_size, dtype, budget_chunk, head_dim, logger=None):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # b, h, s//c, (d*c)*itemsize
        itemsize = dtype.itemsize
        total_bytes = bsz * kv_heads * chunk_num * chunk_size * itemsize
        last_dim_bytes = chunk_size * itemsize
        self.last_dim_bytes = last_dim_bytes
        if last_dim_bytes >= BLOCK_DEV_SIZE:
            assert last_dim_bytes % BLOCK_DEV_SIZE == 0, f"Last dim size {last_dim_bytes} not aligned to {BLOCK_DEV_SIZE}"
        else:
            assert BLOCK_DEV_SIZE % last_dim_bytes == 0, f"Block size {BLOCK_DEV_SIZE} not aligned to last dim size {last_dim_bytes}"    
        assert total_bytes <= MAX_KV_SIZE, f"{MAX_KV_SIZE} < {total_bytes}"      
        
        self.logger = logger
        self.bsz = bsz
        self.kv_heads = kv_heads
        self.chunk_num = chunk_num
        self.chunk_size = chunk_size
        self.budget_chunk = budget_chunk
        self.dtype = dtype
        self.layer_num = layer_num
        self.head_dim = head_dim
        self.path_list = []
        self.fd = []
        
        # self.disk_values = torch.empty(bsz, kv_heads, budget_chunk * self.chunk_size // self.head_dim, self.head_dim,
        #                                dtype=self.dtype, device='cpu', pin_memory=True)
        from .uring_io import DiskIO
        self.diskio = DiskIO(
            hd_size=head_dim,
            max_kv_num=chunk_size // head_dim * budget_chunk,
            b_size=bsz*kv_heads,
            seq_len=chunk_num*chunk_size // head_dim,
            token_group=chunk_size // head_dim,
            layer_num=layer_num,
            itemsize=itemsize
        )
        for i in range(layer_num):
            path = os.path.join(self.cache_dir, f"v_cache_{i}.bin")   
            need_alloc = (not os.path.exists(path)) or (os.path.getsize(path) < MAX_KV_SIZE)
            if need_alloc:
                print(f"Allocating {path} to {MAX_KV_SIZE} bytes")
                zero_out(path, MAX_KV_SIZE)       
            self.path_list.append(path)
            flags = os.O_RDWR 
            flags |= os.O_DIRECT
            self.fd.append(os.open(path, flags))
        
        self.diskio.register_files(self.fd)
        
        self.time_list = []
        self.size_list = []
        self.prefill_time_list = []
        
    
    def write(self, layer_i, start_bsz, new_v_cache):
        # bsz, h, chunks, (d*c) -> bsz, h, s, d
        # assert start_bsz == 0, f"start_bsz should be 0, got {start_bsz}"
        t = time.time()
        bsz_cur = new_v_cache.shape[0]
        new_v_cache = new_v_cache.view(bsz_cur, self.kv_heads, -1, self.head_dim)
        self.diskio.write(self.fd[layer_i], new_v_cache, start_bsz)
        if layer_i == (self.layer_num - 1) and start_bsz + bsz_cur == self.bsz:
            self.diskio.free_prefill_buffer()
        dur = time.time() - t
        self.prefill_time_list.append(dur)

    def read(self, layer_i, position_ids, buffer, pos):
        # position_ids: bsz, h, chunks
        # bw = 100
        # size = position_ids.numel() * self.last_dim_bytes / 1024 / 1024
        # t = size / bw
        # time.sleep(t)
        # bh, chunks, d
        t = time.time()
        disk_values = self.diskio.read(self.fd[layer_i], position_ids.view(position_ids.shape[0]*position_ids.shape[1], -1), self.dtype)
        dur = time.time() - t
        size = disk_values.numel() * disk_values.element_size() / (1024**2)
        if self.logger:
            pass
            # self.logger.info(f"Reading {size:.2f} MB ({disk_values.shape}) layer {layer_i}, time: {dur*1000:.2f} ms")
        self.time_list.append(dur)
        self.size_list.append(size)
        buffer[layer_i][:, :, pos].copy_(disk_values.view(self.bsz, self.kv_heads, -1, self.head_dim), non_blocking=True)
        
    def get_stats(self):
        total_time = sum(self.time_list)
        total_size = sum(self.size_list)
        avg_time = total_time / len(self.time_list) if self.time_list else 0
        avg_size = total_size / len(self.size_list) if self.size_list else 0
        max_time = max(self.time_list) if self.time_list else 0
        min_time = min(self.time_list) if self.time_list else 0
        if total_time > 0:
            bw = total_size / total_time
        else:
            bw = 0
        if self.logger:
            self.logger.info(f"DiskCache stats: max time: {max_time*1000:.2f} ms, min time: {min_time*1000:.2f} ms")
            self.logger.info(f"DiskCache total time: {total_time*1000:.2f} ms, total size: {total_size:.2f} MB, bandwidth: {bw:.2f} MB/s")    
            self.logger.info(f"DiskCache avg time: {avg_time*1000:.2f} ms, avg size: {avg_size:.2f} MB, len: {len(self.time_list)}")
        else:
            print(f"DiskCache total time: {total_time*1000:.2f} ms, total size: {total_size:.2f} MB, bandwidth: {bw:.2f} MB/s")    
            print(f"DiskCache avg time: {avg_time*1000:.2f} ms, avg size: {avg_size:.2f} MB, len: {len(self.time_list)}")