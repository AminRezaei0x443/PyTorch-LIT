from .memory import Memory
from .shared_params import SharedParameterUtil
import torch


class Inference:
    def __init__(self, model_closure):
        self.model_closure = model_closure
        self._hooks = []

    def attach_loader(self, loader):
        self.loader = loader
        _, self._dummy = Memory.alloc((1,), device=loader.device)

    def init_model(self, memory_key=None):
        if memory_key:
            SharedParameterUtil.hijackParameters(memory_key)
        self.model = self.model_closure()
        self.model = self.model.to(self.loader.device)
        self._init_module_map()
        self._install_hooks()
        if memory_key:
            SharedParameterUtil.resetParameters()

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)

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
            Inference._list_modules(c, p, mods)

    def _init_module_map(self):
        mods = []
        Inference._list_modules(self.model, mods=mods)
        self.modMap = {m: name for name, m in mods}

    def _install_hooks(self):
        for module in self.modMap.keys():
            module: torch.nn.Module
            setattr(module, "_lit_inference", self)
            h1 = module.register_forward_pre_hook(Inference._w_pre_hook)
            h2 = module.register_forward_hook(Inference._w_hook)
            self._hooks.append(h1)
            self._hooks.append(h2)

    def detach_hooks(self):
        for h in self._hooks:
            h.remove()

    @staticmethod
    def _w_pre_hook(module, _):
        if not hasattr(module, "_lit_inference"):
            raise RuntimeError("hooks not installed on model")
        inference: "Inference" = getattr(module, "_lit_inference")
        if module not in inference.modMap:
            return
        name = inference.modMap[module]
        loader = inference.loader
        params = dict(module.named_parameters(recurse=False))
        for p in params:
            k = name + "." + p
            d = loader.basic_load(k)
            setattr(params[p], "data", d)

    @staticmethod
    def _w_hook(module, _, __):
        if not hasattr(module, "_lit_inference"):
            raise RuntimeError("hooks not installed on model")
        inference: "Inference" = getattr(module, "_lit_inference")
        params = dict(module.named_parameters(recurse=False))
        for p in params:
            setattr(params[p], "data", inference._dummy)
