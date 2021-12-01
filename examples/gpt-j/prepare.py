from pytorch_lit import prepare_params
import torch

if __name__ == "__main__":
    weights = torch.load("../../../lab/gpt-j-6B-f16/pytorch_model.bin")
    prepare_params(weights, ".models/gpt-j-6b-lit", dtype="float16")
