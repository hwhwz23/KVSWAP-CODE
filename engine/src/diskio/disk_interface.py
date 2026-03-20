import torch.nn.functional as F
import numpy as np
import os
import torch 
import sys
sys.path.append('../')
from utils import np_dtype_to_torch_dtype
import time
import mmap
import ctypes

POSIX_FADV_DONTNEED = 4  
MADV_DONTNEED = 4
libc = ctypes.CDLL("libc.so.6", use_errno=True)  
PAGE_SIZE = mmap.ALLOCATIONGRANULARITY
BLOCK_DEV_SIZE = 512

MAX_KV_SIZE = os.environ.get('MAX_ALLOC_KV_SIZE')
if MAX_KV_SIZE is None:
    raise ValueError("MAX_ALLOC_KV_SIZE is not set")
MAX_KV_SIZE = int(MAX_KV_SIZE)


shape_dict = {}
dtype_dict = {}
attr_dict = {}

def get_file_num():
	return len(shape_dict)


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

def create_kv_file(path, shape, dtype, use_mmap, attr=None):
	if use_mmap:
		np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
	else:
		itemsize = np.dtype(dtype).itemsize
		total_bytes = np.prod(shape) * itemsize
		last_dim_bytes = shape[-1] * itemsize # 128, 256, 512
		if last_dim_bytes >= BLOCK_DEV_SIZE:
			assert last_dim_bytes % BLOCK_DEV_SIZE == 0, f"Last dim size {last_dim_bytes} not aligned to {BLOCK_DEV_SIZE}"
		else:
			assert BLOCK_DEV_SIZE % last_dim_bytes == 0, f"Block size {BLOCK_DEV_SIZE} not aligned to last dim size {last_dim_bytes}"     
		assert total_bytes <= MAX_KV_SIZE, f"{MAX_KV_SIZE} < {total_bytes}"
		need_alloc = (not os.path.exists(path)) or (os.path.getsize(path) < MAX_KV_SIZE)
		if need_alloc:
			print(f"Allocating {path} to {MAX_KV_SIZE} bytes")
			zero_out(path, MAX_KV_SIZE)
			
	shape_dict[path] = shape
	dtype_dict[path] = dtype
	attr_dict[path] = attr


libc.openat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
libc.openat.restype = ctypes.c_int

def openat(dirfd, path, flags):
	fd = libc.openat(dirfd, path.encode(), flags)
	if fd == -1:
		errno = ctypes.get_errno()
		raise OSError(errno, os.strerror(errno))
	return fd

class DiskInterface():
	def __init__(self, path, disk_cfg, token_group, use_mmap, disk_io_list=[None]):
		self.path = path
		self.use_mmap = use_mmap
		self.shape_ = [shape_dict[path_] for path_ in path]
		# assume all paths have the same dtype and attr
		self.dtype = np_dtype_to_torch_dtype[dtype_dict[path[0]]]
		# self.dtype = torch.bfloat16
		self.attr = attr_dict[path[0]]
		self.dk_rd, self.dk_wr = disk_cfg
		self.token_group = token_group
		if use_mmap:
			self.fd = [os.open(path_, os.O_RDWR) for path_ in path]
			self.mmap_handle = [np.lib.format.open_memmap(path_) for path_ in path]
			self.data = [torch.from_numpy(handle_) for handle_ in self.mmap_handle]
			if self.attr == 'bn2ghd': # b, s, 2hd -> view -> b, s//g, 2, ghd
				self.data = [data_.view(data_.shape[0], 
										data_.shape[1] // self.token_group, 
										2, self.token_group * data_.shape[2] // 2) for data_ in self.data]
			self.addr = None     
		else:
			flags = os.O_RDWR
			if self.dk_rd == 'clear':
				flags |= os.O_DIRECT
			if self.dk_wr == 'flush':
				flags |= os.O_SYNC
				print("make sure you want to use O_SYNC!")
			self.fd = [os.open(path_, flags) for path_ in path]
			element_size = np.dtype(dtype_dict[path[0]]).itemsize
			self.element_bytes = self.shape_[0][-1] * element_size  
			self.disk_io_list = disk_io_list
			
		# if self.validate_fd(self.fd) == False:
		#     raise ValueError(f"Invalid file descriptor {self.fd}, path={self.path}")
	
	# def validate_fd(self, fd_list):
	#     for fd in fd_list:
	#         try:
	#             os.fstat(fd)
	#         except OSError as e:
	#             print(f"File descriptor {fd} is invalid. e={e}")
	#             return False
	#     return True

	def reopen(self):
		if self.fd is not None:
			return
		if self.use_mmap:
			self.fd = [os.open(path_, os.O_RDWR) for path_ in self.path]
			self.mmap_handle = [np.lib.format.open_memmap(path_) for path_ in self.path]
			self.data = [torch.from_numpy(handle_) for handle_ in self.mmap_handle]
			if self.attr == 'bn2ghd': # b, s, 2hd -> view -> b, s//g, 2ghd
				self.data = [data_.view(data_.shape[0], 
										data_.shape[1] // self.token_group, 
										self.token_group * data_.shape[2]) for data_ in self.data]
			self.addr = None               
		else:
			flags = os.O_RDWR
			if self.dk_rd == 'clear':
				flags |= os.O_DIRECT
			if self.dk_wr == 'flush':
				flags |= os.O_SYNC
				print("make sure you want to use O_SYNC")
			self.fd = [os.open(path_, flags) for path_ in self.path]

	@property
	def shape(self):
		return self.shape_

	def read(self, indices, fd_id=0, diskio_id=0, **kwargs):
		if self.use_mmap:
			if type(indices) == torch.Tensor:
				assert self.attr == 'bn2ghd', f"attr is {self.attr}"
				# indices: b, n
				# self.data: b, s//g, 2, ghd
				output = self.data[fd_id].gather(1, indices.cpu().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.data[fd_id].shape[-2:]))
			else:
				assert self.attr == 'bs2hd', f"attr is {self.attr}"
				# indices: b, s
				output = self.data[fd_id][indices]
			if self.dk_rd == 'clear':
				self.clear_cache(fd_id)
			return output
		else:
			output = self.disk_io_list[diskio_id].read(self.fd[fd_id], indices, attr=self.attr, dtype=self.dtype, **kwargs)
			return output
	
	def write(self, indices, values, fd_id=0, diskio_id=0, **kwargs):
		if self.use_mmap:
			if self.attr == 'bn2ghd':
				if type(values) == tuple:
					# self.data[fd_id]: b, s//g, 2, ghd
					# values[0]: b, s//g, ghd
					self.data[fd_id][indices][..., 0, :].copy_(values[0])
					self.data[fd_id][indices][..., 1, :].copy_(values[1])
				else:
					# values: (b, 1, 2, ghd)
					self.data[fd_id][indices].copy_(values)
			else: # bs2hd
				# values: (b, s, 2hd)
				self.data[fd_id][indices][..., :self.shape_[fd_id][-1]//2].copy_(values[0])
				self.data[fd_id][indices][..., self.shape_[fd_id][-1]//2:].copy_(values[1])
			if self.dk_wr == 'flush':
				self.flush(fd_id)
		else:
			self.disk_io_list[diskio_id].write(self.fd[fd_id], indices, values, attr=self.attr, **kwargs)
			# os.fsync(self.fd[fd_id])
			
	def close(self):
		if self.use_mmap:
			for handle_ in self.mmap_handle:
				handle_.flush()
			for fd_ in self.fd:
				os.fsync(fd_)
			self.mmap_handle = None
			self.data = None
			self.addr = None
		for fd_ in self.fd:
			os.close(fd_)
		self.fd = None
	
	def get_mmap_addr(self):
		addr = [ctypes.c_void_p(data_.data_ptr()) for data_ in self.data]
		length = [data_.numel() * data_.element_size() for data_ in self.data]
		self.addr = []
		self.length = []
		self.data_offset = []
		for addr_, length_ in zip(addr, length):
			if addr_.value % PAGE_SIZE != 0:
				print("Address is not page aligned. Adjusting...")
				aligned_addr = ctypes.c_void_p(addr_.value - (addr_.value % PAGE_SIZE))
				length_ += addr_.value - aligned_addr.value
				data_offset = addr_.value - aligned_addr.value
				addr_ = aligned_addr
			else:
				data_offset = 0
			self.addr.append(addr_)
			self.length.append(length_)
			self.data_offset.append(data_offset)

	def clear_cache(self, fd_id=None):
		if not self.use_mmap:
			return
		if self.addr is None:
			self.get_mmap_addr()
		def clear_file_cache(fd, offset, length):
			assert offset % PAGE_SIZE == 0, "Offset is not page aligned"
			# assert length % PAGE_SIZE == 0, "Length is not page aligned"
			result = libc.posix_fadvise(fd, offset, length, POSIX_FADV_DONTNEED)
			assert result == 0, f"posix_fadvise failed with error code {result}"
		def clear_map_cache(addr, length):
			assert addr.value % PAGE_SIZE == 0, "Address is not page aligned"
			# assert length % PAGE_SIZE == 0, "Length is not page aligned"
			result = libc.madvise(addr, length, MADV_DONTNEED)
			assert result == 0, f"madvise failed with error code {result}"
		if fd_id is None:
			for addr_, length_, fd_ in zip(self.addr, self.length, self.fd):
				clear_map_cache(addr_, length_)
				clear_file_cache(fd_, 0, 0)
		else:
			clear_map_cache(self.addr[fd_id], self.length[fd_id])
			clear_file_cache(self.fd[fd_id], 0, 0)

	def flush(self, fd_id=None):
		if not self.use_mmap:
			return
		if fd_id is None:
			for handle_ in self.mmap_handle:
				handle_.flush()
			for fd_ in self.fd:
				os.fsync(fd_)
		else:
			self.mmap_handle[fd_id].flush()
			os.fsync(self.fd[fd_id])
		

