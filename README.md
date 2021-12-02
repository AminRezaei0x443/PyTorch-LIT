# PyTorch-LIT
[![PyPI version](https://img.shields.io/badge/pytorch--lit-0.1.7-informational?style=flat-square&color=C51BA3)](https://pypi.org/project/pytorch-lit/)

**PyTorch-LIT** is the Lite Inference Toolkit (LIT) for PyTorch which focuses on easy and fast inference of large models on end-devices.

With the rapid growth of deep learning research, models are becoming increasingly complex in terms of parameters and complexity, making it difficult to run the models on currently available end devices. For example, GPT-J with 6B parameters only needs 24 GB of RAM in full-precision mode to be ready for execution, which may be impossible in most systems; even a powerful GPU like the RTX 2060 with 6 GB of memory can't even contain GPT-J in half-precision mode, making direct inference impossible.

To address this issue when training large models, libraries such as DeepSpeed use offload techniques (e.g., ZeRO) to handle the parameters and make training possible by dividing the weights between devices. In contrast, there is no direct library/framework available for inference.

**PyTorch-LIT** allows the inference of large models by loading weights as needed from secondary specified memory, which could be disk, CPU, or GPU, allowing the inference of models that do not even fit in the system's main memory simply by trading off time.

## Quick Start
1. Install the library

```bash
pip install pytorch-lit
```

2. You have to save the model's weight in a way that toolkit can use

```python
from pytorch_lit import prepare_params

weights = {} # your model's parameters (state_dict)
# change the directory to save your model and specify data-type
prepare_params(weights, ".models/my-model", dtype="float32")
```

**Note:** If you have trouble loading large state_dict in small RAM, use `PartialLoader` instead of `torch.load`:
```python
from pytorch_lit import prepare_params, PartialLoader

weights = PartialLoader("state_dict.bin") # your model's parameters (state_dict)
# change the directory to save your model and specify data-type
prepare_params(weights, ".models/my-model", dtype="float32")
```
3. After preparing the weights, you can infer your model

```python
from pytorch_lit import LitModule

# pass your model construction as a closure, 
# specify weights path and inference device 
model = LitModule.from_params(".models/my-model",
                                  lambda: MyModel(),
                                  device="cuda")
result = model(*arg, **kwargs)
```

4. Have fun enjoying the inference of the large model on a lower memory device:)

## Examples
The repo's `examples` directory contains examples. There are currently two examples of `GPT-J`, one for text generation and the other for extracting hidden states as feature representations.

## Development
This is a work in progress that will require further development before it can be considered a stable inference toolkit. Here is a list of potential future developments:

- [ ] Caching and batch loading as many weights as memory allows, with weights being replaced in parallel with future ones (through the order of the execution graph)
- [ ] C++ extension for PyTorch jit, so the solution applies to the majority of production end devices
- [ ] Add functions to make it easier to export large models to onnx or trace with jit
- [ ] Use better and faster format than numpy memmap
- [x] Load large state_dict partially when memory is not enough

Contributions are welcome; to discuss your idea further, open an issue with the `discussion` tag. Finally, you can submit a pull request to merge your fork.

## How does it work?
This implementation was made possible primarily by two ideas:

- The first issue was that PyTorch initialized the model object's parameters when constructing it, causing the construction to fail when the model couldn't fit into memory. To address this, we proposed temporarily hijacking PyTorch's `Parameter` class's `__new__` method during model construction, allowing us to replace the parameter's tensor with a view from a shared global tensor immediately after creation.
  By doing so, all parameters use the same shared big tensor as their primary storage, allowing the model to be built and tested with inputs to follow and trace the execution graph.
- The second issue was the large size of model parameters; in the preparation step, we built a numpy memmap`(np.memmap)` and saved metadata that provided us with the location of each key in the memmap. This allowed us to read parameters from the memmap as needed.
Following that, we use the PyTorch hooks (`forward` and `pre_forward`) to load and unload a module's parameters before and after execution.
## Citation
Please cite PyTorch-LIT if it helps your research. You can use the following BibTeX entry:

```bibtex
@misc{pytorch_lit,
	title = {PyTorch-LIT},
	author = {Rezaei, Amin},
	howpublished = {\url{github.com/AminRezaei0x443/PyTorch-LIT}},
	year = {2021}
}
```
