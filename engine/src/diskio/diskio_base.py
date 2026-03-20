import ctypes
import torch 
import numpy as np

BLOCK_DEV_SIZE = 512
PAGE_SIZE = 4096
libc = ctypes.CDLL("libc.so.6")  
libc.malloc.restype = ctypes.c_void_p
libc.malloc.argtypes = [ctypes.c_size_t]

libc.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # void **memptr
    ctypes.c_size_t,                  # alignment
    ctypes.c_size_t,                  # size
]
libc.posix_memalign.restype = ctypes.c_int  

libcudart = ctypes.CDLL("libcudart.so")
cudaHostRegisterDefault = 0

# def check_cuda(status):
#     if status != 0:
#         err_str = ctypes.c_char_p()
#         libcudart.cudaGetErrorString(status, ctypes.byref(err_str))
#         raise RuntimeError(f"CUDA Error: {err_str.value.decode()}")

# cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
libcudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
libcudart.cudaHostRegister.restype  = ctypes.c_int

# const char* cudaGetErrorString(cudaError_t error);
libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
libcudart.cudaGetErrorString.restype  = ctypes.c_char_p

def check_cuda(status: int):
    if status != 0:
        msg = libcudart.cudaGetErrorString(status)
        raise RuntimeError(f"CUDA Error {status}: {msg.decode() if msg else 'Unknown error'}")

class DiskIO_Base():
    def __init__(self, hd_size, max_kv_num, batch_size, pin_memory, itemsize=2):
        self.hd_bytes = hd_size * itemsize   
        self.pin_memory = pin_memory
        if max_kv_num > 0:
            self.read_tensor, _, self.tensor_buf, self.tensor_mv = self.allocate_tensor(max_kv_num*batch_size*self.hd_bytes)
        self.wr_blk_size = None
        self.prefill_buffer = None
        self.wr_buf = None

    def buffer_to_tensor(self, buffer, size):
        aligned_array = (ctypes.c_uint8 * size).from_address(buffer.value)
        mv = memoryview(aligned_array)
        np_array = np.ctypeslib.as_array(aligned_array)
        tensor = torch.from_numpy(np_array)
        if self.pin_memory:
            assert tensor.is_pinned() == True, f"{tensor.is_pinned()}"
        return tensor, aligned_array, buffer, mv
    
    def allocate_tensor(self, size):
        buf, _ = self.allocate_buffer(size)
        return self.buffer_to_tensor(buf, size)
        
    def allocate_buffer(self, size):
        buffer = ctypes.c_void_p()
        ret = libc.posix_memalign(ctypes.byref(buffer), PAGE_SIZE, size)
        # buffer = libc.malloc(size)
        # assert buffer is not None, f"malloc failed for size {size}"
        assert ret == 0, f"posix_memalign failed, returned ret={ret}, size={size}, PAGE_SIZE={PAGE_SIZE}"    
        
        mlock = libc.mlock
        mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        mlock.restype = ctypes.c_int
        ret = mlock(buffer, size)
        assert ret == 0, f"mlock failed with code {ret}. Hint: check `ulimit -l`."
        
        if self.pin_memory:
            check_cuda(libcudart.cudaHostRegister(buffer, size, cudaHostRegisterDefault))
        
        buffer_mv = memoryview((ctypes.c_char * size).from_address(buffer.value))       
        return buffer, buffer_mv
    
    def free_prefill_buffer(self):
        if self.prefill_buffer is not None:
            libc.free(self.prefill_buffer)
            self.prefill_buffer = None
            self.prefill_buf_mv = None
            self.prefill_tensor = None
            print("Prefill buffer is freed!")
        else:
            print("Prefill buffer is already freed or not inited")
    
    def write(self, fd, indices, values, **kwargs):     
        raise NotImplementedError

    def read(self, fd, indices, **kwargs):
        raise NotImplementedError

    def close(self):
        if hasattr(self, "tensor_buf"):
            self.read_tensor = None
            libc.free(self.tensor_buf)
            self.tensor_buf = None
            self.tensor_mv = None
        self.free_prefill_buffer()
        if self.wr_buf is not None:
            libc.free(self.wr_buf)
        self.wr_tensor = None
        self.wr_buf = None
        self.mv_wr = None
        self.wr_blk_size = None
