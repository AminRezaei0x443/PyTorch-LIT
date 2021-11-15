# PyTorch-LIT
**PyTorch-LIT** is the Lite Inference Toolkit (LIT) for PyTorch which focuses on easy and fast inference of large models on end-devices.

With the fast growth of deep learning research, models are growing in terms of parameters and complexity which makes it difficult to run the models on available end devices. For example, GPT-J which has 6B parameters just requires 24 GB of RAM in full-precision mode to get ready for execution which may be impossible in most systems; 
Even a powerful GPU like RTX 2060 which comes with 6 GB of memory can't even contain GPT-J in half-precision mode and makes it impossible to infer directly.

To overcome this issue in training large models, libraries like DeepSpeed handle the parameters with offload techniques (e.g. ZeRO) and make the training possible by dividing the weights between devices. In contrast for inference, there is no direct library/framework available. 

**PyTorch-LIT** comes to provide the inference of large models by loading the weights as needed from secondary specified memory that could be disk, CPU, or GPU making it possible to infer models that don't even fit in systems main memory just by trading off the time.

## Quick Start
1. Install the library

```bash
pip install pytorch-lit
```

2. You have to save the model's weight in a way that toolkit can use

```python
from pytorch_lit.export import prepare_params

weights = {} # your model's parameters (state_dict)
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
Examples are provided in the `examples` directory of the repo. Currently, there are two examples of `GPT-J` one for text generation and one for extracting hidden states as feature representations.

## Development
This is a WIP and needs further development to become a stable inference toolkit. Here is a checklist of future developments:

- [ ] Caching and batch loading the weights as much as memory can handle, with parallel-wise replacing the weights with future ones (through the order of the execution graph)
- [ ] C++ extension for PyTorch jit, so the solution applies to most of the end devices on production
- [ ] Add functions to ease exporting large models to onnx or tracing with jit
- [ ] Better and faster format than numpy memmap

Contributions are welcome, you can open an issue with the `discussion` tag to discuss your idea further. Finally, you can open a pull request to merge your implemented fork.

## How does it work?
There are mainly two ideas that made this implementation possible:

- The first issue was that when constructing the model object, It initialized its parameters making the construction fail when the model couldn't fit into the memory. To resolve this, the idea was to temporarily hijack PyTorch's `Parameter` class's `__new__` method during model construction, so we could replace the parameter's tensor with a view from a shared global tensor, right after the creation.
  With doing this, all parameters use the same shared big tensor as their main storage; making it possible to construct the model and do a test run with inputs to follow and trace the execution graph.
  
- The second issue was the large size of model parameters, we construct a numpy memmap`(np.memmap)` in the preparation step and save metadata that provide us the location of each key in the memmap. This way we could read parameters as needed from the memmap.
After that, we use the PyTorch hooks (`forward` and `pre_forward`) to load and unload the parameters of a module before and after execution.
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