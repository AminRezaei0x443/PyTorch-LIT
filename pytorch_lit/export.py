from functools import reduce
import numpy as np
import json
from tqdm import tqdm


def params_to_memmap(params, mem_file=None, meta_file=None, dtype='float32', progress=True):
    paramSize = lambda tensor: reduce(lambda x, y: x * y, tensor.shape, 1)
    startIndex = 0
    meta = {}
    for i, k in enumerate(params):
        end = startIndex + paramSize(params[k])
        meta[k] = {
            "bound": (startIndex, end),
            "shape": list(params[k].shape)
        }
        startIndex = end
    it = meta
    if progress:
        print("creating memmap ...")
        it = tqdm(it)
    fp = np.memmap(mem_file, dtype=dtype, mode='write', shape=(startIndex,))
    for k in it:
        m = meta[k]
        tV = params[k]
        b = m["bound"]
        fp[b[0]:b[1]] = tV.reshape(-1)
        fp.flush()
    meta["_info"] = {
        "size": startIndex,
        "type": dtype
    }
    if meta_file:
        if progress:
            print("writing meta ...")
        f = open(meta_file, "w")
        json.dump(meta, f)
        f.close()
    if progress:
        print("done export.")
    return meta
