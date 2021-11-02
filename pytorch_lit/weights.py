import json
import numpy as np
import torch


class WeightLoader:
    def __init__(self, mem_path, meta_path, device="cpu"):
        self._readMeta(meta_path)
        self._initMem(mem_path)
        self.device = device
        pass

    def _readMeta(self, meta_path):
        f = open(meta_path, "r")
        self.meta = json.load(f)

    def _initMem(self, mem_path):
        dtype, size = self.meta["_info"]["type"], self.meta["_info"]["size"]
        self.mem = np.memmap(mem_path, dtype=dtype, mode='r', shape=(size,))

    def basic_load(self, key):
        meta = self.meta[key]
        b = meta["bound"]
        s = meta["shape"]
        t = torch.tensor(self.mem[b[0]:b[1]], device=self.device)
        if len(s) == 0:
            t = t[0]
        else:
            t = t.reshape(tuple(s))
        return t
