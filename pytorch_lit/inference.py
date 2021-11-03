from .memory import Memory
from .shared_params import SharedParameterUtil
import torch
from os import path

from .util import read_json
from .weights import WeightLoader


class LitModule:
    def __init__(self, model_closure):
        self.model_closure = model_closure
        self._hooks = []

    def attach_loader(self, loader):
        self.loader = loader
        _, self._dummy = Memory.alloc((1,), device=loader.device)

    def init_model(self, memory_key=None):
        if memory_key:
            SharedParameterUtil.hijackParameters(memory_key)
        self.module = self.model_closure()
        self.module = self.module.to(self.loader.device)
        self._init_module_map()
        self._install_hooks()
        if memory_key:
            SharedParameterUtil.resetParameters()

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @staticmethod
    def _list_modules(module, prefixes=None, mods=None):
        if mods is None:
            mods = []
        if prefixes is None:
            prefixes = []
        children = list(module.named_children())
        mods.append((".".join(prefixes), module))
        for n, c in children:
            p = prefixes.copy()
            p.append(n)
            LitModule._list_modules(c, p, mods)

    def _init_module_map(self):
        mods = []
        LitModule._list_modules(self.module, mods=mods)
        self.module_map = {m: name for name, m in mods}

    def _install_hooks(self):
        for module in self.module_map.keys():
            module: torch.nn.Module
            setattr(module, "_lit_inference", self)
            h1 = module.register_forward_pre_hook(LitModule._w_pre_hook)
            h2 = module.register_forward_hook(LitModule._w_hook)
            self._hooks.append(h1)
            self._hooks.append(h2)

    def detach_hooks(self):
        for h in self._hooks:
            h.remove()

    @staticmethod
    def _w_pre_hook(module, _):
        if not hasattr(module, "_lit_inference"):
            raise RuntimeError("hooks not installed on model")
        inference: "LitModule" = getattr(module, "_lit_inference")
        if module not in inference.module_map:
            return
        name = inference.module_map[module]
        loader = inference.loader
        params = dict(module.named_parameters(recurse=False))
        for p in params:
            k = name + "." + p
            d = loader.load(k)
            setattr(params[p], "data", d)

    @staticmethod
    def _w_hook(module, _, __):
        if not hasattr(module, "_lit_inference"):
            raise RuntimeError("hooks not installed on model")
        inference: "LitModule" = getattr(module, "_lit_inference")
        params = dict(module.named_parameters(recurse=False))
        for p in params:
            setattr(params[p], "data", inference._dummy)

    @staticmethod
    def from_params(params_path, model_closure, shared_memory_shape=None, device="cuda"):
        model = LitModule(model_closure)
        meta = read_json(path.join(params_path, "meta.json"))
        meta["__base_path"] = params_path
        loader = WeightLoader(meta, device)
        model.attach_loader(loader)
        lp = loader.largest_param_size()
        # 5% margin
        target_shape = (int(lp * 1.05),)
        if shared_memory_shape is None and lp == -1:
            raise RuntimeError("Failed inferring largest parameter size,"
                               " provide it via share_memory_shape (If you don't know, provide just an upper bound)")
        if shared_memory_shape is not None:
            target_shape = shared_memory_shape
        k, _ = Memory.alloc(target_shape, dtype=loader.data_type(), device=device)
        model.init_model(memory_key=k)
        return model

