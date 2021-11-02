import torch


class Memory:
    _tensor_storage = {}
    _k = 0

    @staticmethod
    def alloc(shape, device):
        t = torch.empty(shape, dtype=torch.float, device=device)
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
