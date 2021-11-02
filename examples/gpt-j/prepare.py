from pytorch_lit.export import params_to_memmap
import torch

if __name__ == "__main__":
    weights = torch.load("../../../lab/gpt-j-6B-f16/pytorch_model.bin")
    params_to_memmap(weights, mem_file=".models/gpt.npx", meta_file=".models/gpt.meta", dtype="float16")
