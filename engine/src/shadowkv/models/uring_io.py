import time
import ctypes
import torch
import numpy as np
import os
import errno
from .diskio_base import DiskIO_Base
from liburing import (io_uring, io_uring_queue_init, io_uring_queue_exit, io_uring_get_sqe, io_uring_submit,
                    io_uring_prep_write, io_uring_wait_cqe, io_uring_cqe, io_uring_cqe_seen,
                    trap_error, io_uring_register_files, io_uring_unregister_files, 
                    io_uring_register_buffers, io_uring_unregister_buffers, iovec,
                    prepare_sqe_batch_submit_wait_advance, 
                    prepare_sqe_batch_submit_wait_advance_timeout,
                    io_uring_register_iowq_max_workers, set_iowq_affinity, 
                    write_prepare_sqe_batch_submit_wait_advance
                    )

IORING_SETUP_SQPOLL = 2
IOSQE_FIXED_FILE = 1

BLOCK_DEV_SIZE = 512
MAX_ENTRIES = 32768 // 2
libc = ctypes.CDLL("libc.so.6")

class DiskIO(DiskIO_Base):
    def __init__(self, hd_size, max_kv_num, b_size, seq_len, token_group, layer_num, 
                 itemsize=2, bind_workers=False, bind_cpi_id=4):
        pin_memory = True
        super().__init__(hd_size, max_kv_num, b_size, pin_memory, itemsize=itemsize)
        self.rd_ring = io_uring()
        self.wr_ring = io_uring()
        self.seq_len = seq_len
        self.b_size = b_size
        self.token_group = token_group

        assert max_kv_num % token_group == 0, f"{max_kv_num} % {token_group} != 0"
        assert max_kv_num >= token_group, f"{max_kv_num} < {token_group}"
        self.max_group_num = max_kv_num // token_group
        self.max_rd_req = self.max_group_num * b_size

        self.batch_offset = self.seq_len * self.hd_bytes
        # self.max_rd_req = min(self.max_rd_req, MAX_ENTRIES)
        assert self.max_rd_req <= MAX_ENTRIES, f"Number of requests {self.max_rd_req} is larger than {MAX_ENTRIES}" 
        # if need to use timeout read, set entries to twice of max_rd_req
        ret = io_uring_queue_init(self.max_rd_req*2, self.rd_ring, IORING_SETUP_SQPOLL)
        assert ret == 0, f"io_uring_queue_init failed: {ret}"
        self.max_wr_req = b_size
        assert self.max_wr_req <= MAX_ENTRIES, f"Number of requests {self.max_wr_req} is larger than {MAX_ENTRIES}"
        ret = io_uring_queue_init(self.max_wr_req, self.wr_ring, IORING_SETUP_SQPOLL)
        assert ret == 0, f"io_uring_queue_init failed: {ret}"
        self.fd_dict = {}
        self.reg_buffer = False
        # self.read_timeout_ns = 0
        # self.read_timeout_ns = 5 * 1000000 # 5ms
        self.read_timeout_ns = 5 * 1000000 * 100 # 500ms
        self.timeout_array = np.array([0 for _ in range(self.max_rd_req)], dtype=np.uint32)
        self.timeout_num_array = np.array([0], dtype=np.uint32)
        self.timeout_max_retries = 2
        self.do_fsync = False
        self.real_req_num_array = np.array([0], dtype=np.uint32)
        ##########################################################################
        if hasattr(self, "read_tensor"):            
            buffer_size = token_group * self.hd_bytes
            assert buffer_size % BLOCK_DEV_SIZE == 0, f"Buffer size {buffer_size} is not aligned to block size {BLOCK_DEV_SIZE}"
            assert buffer_size >= BLOCK_DEV_SIZE, f"Buffer size {buffer_size} is smaller than block device size {BLOCK_DEV_SIZE}"
            self.buffer_size = buffer_size
            self.rd_mv_list = []
            num_buffers = self.read_tensor.numel() // buffer_size
            self.rd_addr_array = (ctypes.c_uint64 * num_buffers)()
            for i in range(num_buffers):
                mv = memoryview((ctypes.c_char * buffer_size).from_address(self.tensor_buf.value + i * buffer_size))
                self.rd_mv_list.append(mv.obj)
                self.rd_addr_array[i] = ctypes.addressof(mv.obj)
            self.register_buffer(self.rd_mv_list)
        ##########################################################################
        self.wr_addr_array = (ctypes.c_uint64 * self.max_wr_req)()
        self.wr_mv_list = []
        if bind_workers:
            arr = np.array([1, 1], dtype=np.uint32)
            ret = io_uring_register_iowq_max_workers(self.rd_ring, arr)
            if ret < 0:
                raise RuntimeError(f"io_uring_register_iowq_max_workers failed: {ret}")
            else:
                print(f"io_uring_register_iowq_max_workers success, arr={arr}")        
        if bind_cpi_id >= 0:
            arr = np.array([bind_cpi_id], dtype=np.int32)
            ret = set_iowq_affinity(self.rd_ring, arr)
            if ret < 0:
                raise RuntimeError(f"set_iowq_affinity failed: {ret}")
            else:
                print(f"set_iowq_affinity success")            
        
    def register_files(self, fd_list):
        io_uring_register_files(self.rd_ring, fd_list)
        io_uring_register_files(self.wr_ring, fd_list)
        self.fd_dict = {fd: idx for idx, fd in enumerate(fd_list)}

    def register_buffer(self, reg_mv_list):
        if self.reg_buffer:
            return
        try:
            io_uring_unregister_buffers(self.rd_ring)
        except Exception as e:
            print(f"Failed to unregister buffers: {e}", flush=True)
            print(self.rd_ring, flush=True)
        if len(reg_mv_list) > 1024:
            print(f"Skip registering {len(reg_mv_list)} buffers")
            return
        ret = io_uring_register_buffers(self.rd_ring, iovec(reg_mv_list), len(reg_mv_list))
        if ret < 0:
            raise RuntimeError(f"io_uring_register_buffers failed: {ret}")
        else:
            print("io_uring_register_buffers success")
        self.reg_buffer = True
        

    def write(self, fd, values, start_bsz):
        if len(self.fd_dict) > 0:
            fd_ = self.fd_dict[fd]
        else:
            fd_ = fd
        # bsz, h, s, d
        bsz, kv_heads, ind_len, _ = values.shape
        write_size_per = self.seq_len * self.hd_bytes
        write_size = bsz * kv_heads * write_size_per    
        if self.prefill_buffer is None:
            self.prefill_tensor, _, self.prefill_buffer, self.prefill_buf_mv = self.allocate_tensor(write_size)
            # b, s, 2, hd
            self.prefill_tensor = self.prefill_tensor.view(values.dtype)
            self.wr_addr_array[0] = ctypes.addressof(self.prefill_buf_mv.obj)

        real_seq_len = ind_len
        self.prefill_tensor = self.prefill_tensor.view(bsz * kv_heads, self.seq_len, -1)
        assert real_seq_len <= self.seq_len, f"Real sequence length {real_seq_len} is larger than {self.seq_len}"
        
        self.prefill_tensor[:, :ind_len].copy_(values.view(bsz * kv_heads, ind_len, -1))
        
        ret = write_prepare_sqe_batch_submit_wait_advance(self.wr_ring, 1, self.wr_addr_array, fd_, 
                                                            write_size, start_bsz*kv_heads*write_size_per, 0, len(self.fd_dict) > 0, self.do_fsync)
        if ret < 0:
            raise RuntimeError(f"write_prepare_sqe_batch_submit_wait_advance failed: {ret}") 
        
            
    def read(self, fd, indices, dtype):
        
        if len(self.fd_dict) > 0:
            fd_ = self.fd_dict[fd]
        else:
            fd_ = fd
                
        # indices: b*h, n
        read_req_n = indices.numel()
        n_group = indices.shape[1]
        if self.read_tensor.dtype != dtype:
            # bh,n,gd
            # print("self.read_tensor...", flush=True)
            self.read_tensor = self.read_tensor.view(dtype).view(self.b_size*self.max_group_num, -1)
        read_addrs = self.rd_addr_array
        set_fixed_buffer = self.reg_buffer
        # file_offsets = indices.contiguous().view(-1).cpu().numpy().astype(np.uint32)
        file_offsets = indices.cpu().view(-1).numpy().astype(np.int32)
        # file_offsets = np.tile(np.arange(n_group, dtype=np.uint32), indices.shape[0]).reshape(-1)
        group_offset = self.buffer_size
        bytes_per_read = self.buffer_size
        out_tensor = self.read_tensor[:n_group*self.b_size].view(self.b_size, n_group, -1)
        timeout_ns = self.read_timeout_ns

        assert read_req_n <= self.max_rd_req, f"Number of requests {read_req_n} is larger than {self.max_rd_req}"

        # dur0 = time.time() - start_time
        if timeout_ns == 0:
            ret = prepare_sqe_batch_submit_wait_advance(self.rd_ring, read_req_n, read_addrs, fd_, bytes_per_read, 
                                                        file_offsets, n_group, group_offset, self.batch_offset, 
                                                        len(self.fd_dict) > 0, set_fixed_buffer)
            if ret < 0:
                raise RuntimeError(f"prepare_sqe_batch_submit_wait_advance failed: {ret}")  
        else:
            self.timeout_array[:read_req_n] = 1
            timeout_ns_ = timeout_ns
            for retry_i in range(self.timeout_max_retries):
                ret = prepare_sqe_batch_submit_wait_advance_timeout(self.rd_ring, read_req_n, read_addrs, fd_, bytes_per_read, 
                                                            file_offsets, n_group, group_offset, self.batch_offset, 
                                                            len(self.fd_dict) > 0, set_fixed_buffer, timeout_ns_,
                                                            self.timeout_array, self.timeout_num_array, self.real_req_num_array)
                if ret < 0:
                    if ret == -2:
                        pass
                    else:
                        raise RuntimeError(f"prepare_sqe_batch_submit_wait_advance_timeout failed: {ret}")  
                elif self.timeout_num_array[0] == 0:
                    break
                timeout_ns_ = timeout_ns

        # dur1 = time.time() - start_time
        # print(f"fd_={fd_}, {dur0*1000:.1f}-{dur1*1000:.1f} ms")
        
        return out_tensor


    def unregister_files(self):
        io_uring_unregister_files(self.rd_ring)
        io_uring_unregister_files(self.wr_ring)
        self.fd_dict.clear()

    def close(self):
        super().close()
        
        if hasattr(self, "spec_read_buffer") and self.spec_read_buffer is not None:
            libc.free(self.spec_read_buffer)
            self.spec_read_buffer = None
            self.spec_read_tensor = None    
            self.spec_read_buf_mv = None
            self.spec_read_mv_list = None
            self.spec_read_addr_array = None
            
        if len(self.fd_dict) > 0:
            self.unregister_files()
        if self.reg_buffer:
            io_uring_unregister_buffers(self.rd_ring)
        # io_uring_unregister_iowq_aff(self.rd_ring)
        io_uring_queue_exit(self.rd_ring)
        io_uring_queue_exit(self.wr_ring)
        del self.fd_dict
        # del self.rd_cqe, self.wr_cqe
        del self.rd_ring, self.wr_ring

