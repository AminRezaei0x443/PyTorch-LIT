import numpy as np
import torch
from os import path

from pytorch_lit.util import read_json


class WeightLoader:
    ALLOWED_METHODS = ["np-memmap"]

    def __init__(self, info, device="cpu"):
        self.device = device
        self.info = info
        self._initialize()
        pass

    def _initialize(self):
        self.method = self.info["method"]
        if self.method not in WeightLoader.ALLOWED_METHODS:
            raise RuntimeError(f"Unsupported method: {self.method}")
        if self.method == "np-memmap":
            self._init_np_memmap()

    def _init_np_memmap(self):
        meta_p = path.join(self.info["__base_path"], self.info["meta"])
        mem_p = path.join(self.info["__base_path"], self.info["model"])
        self.meta = read_json(meta_p)
        dtype, size = self.meta["_info"]["type"], self.meta["_info"]["size"]
        self.mem = np.memmap(mem_p, dtype=dtype, mode='r', shape=(size,))

    def _np_memmap_load(self, key):
        meta = self.meta[key]
        b = meta["bound"]
        s = meta["shape"]
        t = torch.tensor(self.mem[b[0]:b[1]], device=self.device)
        if len(s) == 0:
            t = t[0]
        else:
            t = t.reshape(tuple(s))
        return t

    def load(self, key):
        if self.method == "np-memmap":
            return self._np_memmap_load(key)
        else:
            raise RuntimeError(f"Unsupported method: {self.method}")

    def largest_param_size(self):
        if self.method == "np-memmap":
            return self.meta["_info"]["largest_param_size"]
        else:
            return -1

    def data_type(self):
        return self.meta["_info"]["type"]