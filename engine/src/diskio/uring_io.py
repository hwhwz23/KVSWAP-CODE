import time
import ctypes
import torch
import numpy as np
import os
import errno
from diskio.diskio_base import DiskIO_Base
from liburing import (io_uring, io_uring_queue_init, io_uring_queue_exit, io_uring_get_sqe, io_uring_submit,
					io_uring_prep_write, io_uring_wait_cqe, io_uring_cqe, io_uring_cqe_seen,
					trap_error, io_uring_register_files, io_uring_unregister_files, 
					io_uring_register_buffers, io_uring_unregister_buffers, iovec,
					prepare_sqe_batch_submit_wait_advance, 
					prepare_sqe_batch_submit_wait_advance_timeout,
					io_uring_register_iowq_max_workers, set_iowq_affinity, 
					write_prepare_sqe_batch_submit_wait_advance, io_uring_queue_init_sqpoll
					)

IORING_SETUP_SQPOLL = 2
IOSQE_FIXED_FILE = 1

BLOCK_DEV_SIZE = 512
MAX_ENTRIES = 32768 // 2
libc = ctypes.CDLL("libc.so.6")


class DiskIO(DiskIO_Base):
	def __init__(self, hd_size, max_kv_num, b_size, seq_len, token_group, layer_num, 
				 itemsize=2, wr_tensor_dev='cuda', name='default', bind_workers=True, bind_cpi_id=4):
		pin_memory = True
		super().__init__(hd_size, max_kv_num, b_size, pin_memory, itemsize=itemsize)
		self.rd_ring = io_uring()
		self.wr_ring = io_uring()
		self.seq_len = seq_len
		self.b_size = b_size
		self.token_group = token_group
		if max_kv_num > 0:
			assert max_kv_num % token_group == 0, f"{max_kv_num} % {token_group} != 0"
			assert max_kv_num >= token_group, f"{max_kv_num} < {token_group}"
			self.max_group_num = max_kv_num // token_group
			self.max_rd_req = self.max_group_num * b_size
		else:
			self.max_rd_req = b_size
		self.batch_offset = self.seq_len * self.hd_bytes
		# self.max_rd_req = min(self.max_rd_req, MAX_ENTRIES)
		assert self.max_rd_req <= MAX_ENTRIES, f"Number of requests {self.max_rd_req} is larger than {MAX_ENTRIES}" 
		# if need to use timeout read, set entries to twice of max_rd_req
		# ret = io_uring_queue_init(self.max_rd_req*2, self.rd_ring, IORING_SETUP_SQPOLL)
		ret = io_uring_queue_init_sqpoll(self.max_rd_req*2, self.rd_ring, 4)
		# ret = io_uring_queue_init(self.max_rd_req*2, self.rd_ring, 0)
		assert ret == 0, f"io_uring_queue_init failed: {ret}"
		self.max_wr_req = b_size
		assert self.max_wr_req <= MAX_ENTRIES, f"Number of requests {self.max_wr_req} is larger than {MAX_ENTRIES}"
		# ret = io_uring_queue_init(self.max_wr_req, self.wr_ring, IORING_SETUP_SQPOLL)
		ret = io_uring_queue_init_sqpoll(self.max_wr_req, self.wr_ring, 4)
		# ret = io_uring_queue_init(self.max_wr_req, self.wr_ring, 0)
		assert ret == 0, f"io_uring_queue_init failed: {ret}"
		self.fd_dict = {}
		self.reg_buffer = False
		self.read_timeout_ns = 5 * 1000000 * 100 # 500ms
		self.timeout_max_retries = 1
		self.do_fsync = False
		self.real_req_num_array = np.array([0], dtype=np.uint32)
		self.timeout_array = np.array([0 for _ in range(self.max_rd_req)], dtype=np.uint32)
		self.timeout_num_array = np.array([0], dtype=np.uint32)
		self.min_read_timeout_ns = 5 * 1000000 # 5ms
		##########################################################################
		if hasattr(self, "read_tensor"):            
			buffer_size = token_group * self.hd_bytes
			if 'nvme' in name.lower():
				self.read_timeout_ns = round(b_size * max_kv_num * self.hd_bytes / 1024 / 1024 / 400 * 1000 * 1000000)
				self.read_timeout_ns = max(self.read_timeout_ns, self.min_read_timeout_ns)
				print(f"Setting read_timeout_ms={self.read_timeout_ns/1e6:.1f}", flush=True)
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
			print(f"io_uring_unregister_buffers failed: {e}", flush=True)
		if len(reg_mv_list) > 1024:
			print(f"Skip registering {len(reg_mv_list)} buffers")
			return
		ret = io_uring_register_buffers(self.rd_ring, iovec(reg_mv_list), len(reg_mv_list))
		if ret < 0:
			raise RuntimeError(f"io_uring_register_buffers failed: {ret}")
		else:
			print("io_uring_register_buffers success")
		self.reg_buffer = True
		

	def write(self, fd, indices, values, prefill, attr, prefill_mode='all_seq', **kwargs):
		if len(self.fd_dict) > 0:
			fd_ = self.fd_dict[fd]
		else:
			fd_ = fd
		
		if prefill and prefill_mode == 'all_seq':
			write_size_per = self.seq_len * self.hd_bytes
			write_size = write_size_per
			if self.prefill_buffer is None:
				self.prefill_tensor, _, self.prefill_buffer, self.prefill_buf_mv = self.allocate_tensor(write_size)
				dtype = values[0].dtype if type(values) == tuple else values.dtype
				# b, s, 2, hd
				self.prefill_tensor = self.prefill_tensor.view(dtype)
				self.wr_addr_array[0] = ctypes.addressof(self.prefill_buf_mv.obj)
			ind_len = indices[1].stop - indices[1].start
			if attr == 'bn2ghd':
				real_seq_len = ind_len * self.token_group
				self.prefill_tensor = self.prefill_tensor.view(1, self.seq_len//self.token_group, 2, -1)
			else:
				real_seq_len = ind_len
				self.prefill_tensor = self.prefill_tensor.view(1, self.seq_len, 2, -1)
			assert real_seq_len <= self.seq_len, f"Real sequence length {real_seq_len} is larger than {self.seq_len}"
			
			# layout: (b, s, hd) + (b, s, hd) -> (b, s, 2hd)
			# layout: (b, s//g, ghd) + (b, s//g, ghd) -> (b, s//g, 2ghd)
			self.prefill_tensor[:, :ind_len, 0].copy_(values[0].view(1, ind_len, -1))
			self.prefill_tensor[:, :ind_len, 1].copy_(values[1].view(1, ind_len, -1))
			batch_i = kwargs['batch_i']
			batch_offset = batch_i * write_size
			ret = write_prepare_sqe_batch_submit_wait_advance(self.wr_ring, 1, self.wr_addr_array, fd_, 
																write_size, batch_offset, 0, len(self.fd_dict) > 0, self.do_fsync)
			if ret < 0:
				raise RuntimeError(f"write_prepare_sqe_batch_submit_wait_advance failed: {ret}") 
			
		else:
			ind_len = indices[1].stop - indices[1].start
			token_group = self.token_group
			if attr == 'bn2ghd':
				real_seq_len = ind_len * token_group
			else:
				real_seq_len = ind_len
			if prefill: # values is a tuple
				raise NotImplementedError
				if self.prefill_buffer is None:
					write_size = real_seq_len * self.hd_bytes * self.b_size
					self.prefill_tensor, _, self.prefill_buffer, self.prefill_buf_mv = self.allocate_tensor(write_size)   
					dtype = values[0].dtype if type(values) == tuple else values.dtype
					self.prefill_tensor = self.prefill_tensor.view(dtype)
					buffer_size = write_size // self.max_wr_req
					for i in range(self.max_wr_req):
						mv = memoryview((ctypes.c_char * buffer_size).from_address(self.prefill_buffer.value + i * buffer_size))
						self.wr_mv_list.append(mv.obj)
						self.wr_addr_array[i] = ctypes.addressof(mv.obj)

				if attr == 'bn2ghd':
					tensor = self.prefill_tensor.view(self.b_size, ind_len, 2, -1)
				else:
					tensor = self.prefill_tensor.view(self.b_size, real_seq_len, 2, -1)
				
				file_offset_i = 0
			else:
				if self.wr_buf is None:
					# use token_group instead as real_seq_len could be smaller than token group
					write_size = self.token_group * self.hd_bytes * 1
					self.free_prefill_buffer()
					self.wr_blk_size = write_size
					assert self.wr_blk_size % BLOCK_DEV_SIZE == 0, f"Write block size {self.wr_blk_size} not aligned to {BLOCK_DEV_SIZE}"        
					self.wr_tensor, _, self.wr_buf, self.mv_wr = self.allocate_tensor(write_size)
					self.wr_mv_list = []
					dtype = values[0].dtype if type(values) == tuple else values.dtype
					self.wr_tensor = self.wr_tensor.view(dtype)
					buffer_size = write_size
					for i in range(1):
						mv = memoryview((ctypes.c_char * buffer_size).from_address(self.wr_buf.value + i * buffer_size))
						self.wr_mv_list.append(mv.obj)
						self.wr_addr_array[i] = ctypes.addressof(mv.obj)             
				# buf = self.wr_buf
				if type(values) == tuple:
					# (b, 1, hd)
					tensor = self.wr_tensor.view(self.b_size, token_group, 2, -1)[:, :real_seq_len]
				else:
					tensor = self.wr_tensor
			
				file_offset_i = indices[1].start * self.hd_bytes
				if attr == 'bn2ghd':
					file_offset_i = file_offset_i * token_group
			
			# print(f"write: fd={fd}, file_offset_i={file_offset_i}, real_seq_len={real_seq_len})")
			if type(values) == tuple:
				tensor[:, :, 0].copy_(values[0].view(values[0].shape[0], ind_len, -1))
				tensor[:, :, 1].copy_(values[1].view(values[1].shape[0], ind_len, -1))
			else: # (b, 1, 2ghd)
				tensor.copy_(values.reshape(-1))     
			
			total_bytes_per = real_seq_len * self.hd_bytes
			assert total_bytes_per % BLOCK_DEV_SIZE == 0, f"{total_bytes_per} is not aligned to block size {BLOCK_DEV_SIZE}"
			batch_i = kwargs['batch_i']
			file_offset_i += self.batch_offset * batch_i
			ret = write_prepare_sqe_batch_submit_wait_advance(self.wr_ring, 1, self.wr_addr_array, fd_, 
																total_bytes_per, file_offset_i, 0, len(self.fd_dict) > 0,
																self.do_fsync)
			if ret < 0:
				raise RuntimeError(f"write_prepare_sqe_batch_submit_wait_advance failed: {ret}")  


	def read(self, fd, indices, attr, dtype, **kwargs):
		en_reuse = kwargs['en_reuse'] 
		# print(f"fd={fd}, indices.numel()={indices.numel()}, attr={attr}, dtype={dtype}", flush=True)
		# print(f"{indices.device}, {indices.dtype}, {indices.is_contiguous()}", flush=True)
		# start_time = time.time()
		
		# try:
		#     os.fstat(fd)
		# except OSError as e:
		#     if e.errno == errno.EBADF:
		#         print(f"fd {fd} is invalid", flush=True)
		#     else:
		#         assert False, f"e={e}"
		
		if len(self.fd_dict) > 0:
			fd_ = self.fd_dict[fd]
		else:
			fd_ = fd
				
		if type(indices) == torch.Tensor:
			assert attr == 'bn2ghd', f"attr is {attr}"
			# indices: b, n
			# print(f"indices={indices}", flush=True)
			read_req_n = indices.numel()
			n_group = indices.shape[1]
			if self.read_tensor.dtype != dtype:
				# b,n,2,ghd
				# print("self.read_tensor...", flush=True)
				self.read_tensor = self.read_tensor.view(dtype).view(self.b_size*self.max_group_num, 2, -1)
			read_addrs = self.rd_addr_array
			set_fixed_buffer = self.reg_buffer
			file_offsets = indices.view(-1).numpy()
			group_offset = self.buffer_size
			bytes_per_read = self.buffer_size
			
			timeout_ns = self.read_timeout_ns
		elif type(indices) == np.ndarray:
			raise NotImplementedError
		else:
			# indices: b, s
			# this branch reads whole cache
			assert attr == 'bs2hd', f"attr is {attr}"
			# this branch corresponds to load whole cache. a specific read buffer should be initialized.
			if (not hasattr(self, "spec_read_buffer")) or self.spec_read_buffer is None:
				alloc_size = self.b_size * self.seq_len * self.hd_bytes    
				print(f"allocating spec_read_buffer of size {alloc_size}")
				self.spec_read_tensor, _, self.spec_read_buffer, self.spec_read_buf_mv = self.allocate_tensor(alloc_size)
				# b, s, 2hd
				self.spec_read_tensor = self.spec_read_tensor.view(dtype).view(self.b_size, self.seq_len, -1)
				self.spec_read_addr_array = (ctypes.c_uint64 * self.b_size)()
				buffer_size = alloc_size // self.b_size
				self.spec_read_mv_list = []
				for i in range(self.b_size):
					mv = memoryview((ctypes.c_char * buffer_size).from_address(self.spec_read_buffer.value + i * buffer_size))
					self.spec_read_mv_list.append(mv.obj)
					self.spec_read_addr_array[i] = ctypes.addressof(mv.obj)        
	
			set_fixed_buffer = False
			cur_seq_len = indices[1].stop - indices[1].start
			assert cur_seq_len <= self.seq_len, f"cur_seq_len {cur_seq_len} is larger than {self.seq_len}"
			out_tensor = self.spec_read_tensor[:, :cur_seq_len]
			read_addrs = self.spec_read_addr_array
			read_req_n = indices[0].stop - indices[0].start
			file_offsets = np.array([indices[1].start for _ in range(read_req_n)], dtype=np.uint32)
			group_offset = self.hd_bytes
			n_group = 1
			bytes_per_read = cur_seq_len * self.hd_bytes
			timeout_ns = 0
		
		assert read_req_n <= self.max_rd_req, f"Number of requests {read_req_n} is larger than {self.max_rd_req}"
		# print("indices=", indices, flush=True)
		# print("file_offsets=", file_offsets, flush=True)
		# max_len = self.seq_len * self.hd_bytes * self.b_size
		# for i in range(read_req_n):
		#     offset = file_offsets[i] * group_offset + (i//n_group) * self.batch_offset
		#     assert offset + bytes_per_read <= max_len, f"i={i}, file_offsets[i]={file_offsets[i]}, group_offset={group_offset}, \
		#         n_group={n_group}, read_req_n={read_req_n}, batch_offset={self.batch_offset}, bytes_per_read={bytes_per_read}, max_len={max_len}"
		
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
				
				if retry_i == 0:
					real_req_num = self.real_req_num_array[0]
				
				if ret < 0:
					if ret == -2:
						pass
					else:
						print(f"prepare_sqe_batch_submit_wait_advance_timeout failed: {ret}", flush=True)
						os._exit(1)
						# raise RuntimeError(f"prepare_sqe_batch_submit_wait_advance_timeout failed: {ret}")  
				elif self.timeout_num_array[0] == 0:
					break
				timeout_ns_ = timeout_ns					
			if en_reuse:
				out_tensor = self.read_tensor[:real_req_num].view(real_req_num, 2, -1)
				# print(f"Read {real_req_num} requests, out_tensor.shape={out_tensor.shape}", flush=True)
				# print(f"indices={indices}", flush=True)
			else:
				out_tensor = self.read_tensor[:read_req_n].view(self.b_size, n_group, 2, -1)
		
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

