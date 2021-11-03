import torch


class Memory:
    str_to_torch_dtype_dict = {
        'bool': torch.bool,
        'uint8': torch.uint8,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'complex64': torch.complex64,
        'complex128': torch.complex128
    }
    _tensor_storage = {}
    _k = 0

    @staticmethod
    def alloc(shape, dtype='float32', device='cpu'):
        if not isinstance(dtype, torch.dtype):
            dtype = Memory.str_to_torch_dtype_dict[dtype]
        t = torch.empty(shape, dtype=dtype, device=device)
        key = Memory._k
        setattr(t, "__litKey", key)
        Memory._tensor_storage[key] = t
        Memory._k += 1
        return key, t

    @staticmethod
    def deallocKey(key):
        del Memory._tensor_storage[key]

    @staticmethod
    def dealloc(tensor):
        key = getattr(tensor, "__litKey", -1)
        if key == -1:
            return
        Memory.deallocKey(key)

    @staticmethod
    def obtain(key):
        if key not in Memory._tensor_storage:
            raise RuntimeError("key not in memory")
        return Memory._tensor_storage[key]
