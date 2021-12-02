from functools import reduce
import numpy as np
from tqdm import tqdm
from os import path
import os

from .util import write_json
from .partial_loader import PartialLoader


def prepare_params(parameters, save_dir, method="np-memmap", dtype='float32', progress=True):
    state = {
        "method": method
    }
    if not path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    parameters = _standardize_params(parameters)
    if method == "np-memmap":
        _params_to_memmap(parameters, path.join(save_dir, "model.bin"),
                          path.join(save_dir, "model-meta.json"), dtype, progress)
        state.update({
            "model": "model.bin",
            "meta": "model-meta.json"
        })
    else:
        raise RuntimeError(f"Unknown method: {method}")
    write_json(path.join(save_dir, "meta.json"), state)


def _standardize_params(parameters):
    if isinstance(parameters, dict):
        return {
            "keys": parameters.keys(),
            "get": lambda k: parameters[k]
        }
    if isinstance(parameters, PartialLoader):
        return {
            "keys": parameters.keys(),
            "get": lambda k: parameters.read_key(k)
        }
    raise RuntimeError(f"unsupported parameters type: {type(parameters)}")


def _params_to_memmap(params, mem_file=None, meta_file=None, dtype='float32', progress=True):
    paramSize = lambda tensor: reduce(lambda x, y: x * y, tensor.shape, 1)
    startIndex = 0
    biggest = 0
    meta = {}
    # TODO: Merge Loops
    get_param = lambda key: params["get"](key)
    for i, k in enumerate(params["keys"]):
        param = get_param(k)
        size = paramSize(param)
        end = startIndex + size
        if size > biggest:
            biggest = size
        meta[k] = {
            "bound": (startIndex, end),
            "shape": list(param.shape)
        }
        startIndex = end
    it = meta
    if progress:
        print("creating memmap ...")
        it = tqdm(it)
    fp = np.memmap(mem_file, dtype=dtype, mode='write', shape=(startIndex,))
    for k in it:
        m = meta[k]
        tV = get_param(k)
        b = m["bound"]
        fp[b[0]:b[1]] = tV.reshape(-1)
        fp.flush()
    meta["_info"] = {
        "size": startIndex,
        "type": dtype,
        "largest_param_size": biggest
    }
    if meta_file:
        if progress:
            print("writing meta ...")
        write_json(meta_file, meta)
    if progress:
        print("done export.")
    return meta
