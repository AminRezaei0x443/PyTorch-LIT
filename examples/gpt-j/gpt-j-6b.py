from transformers import AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers import AutoConfig
from pytorch_lit.inference import Inference
from pytorch_lit.memory import Memory
from pytorch_lit.weights import WeightLoader

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../../../lab/gpt-j-6B-f16")
    config = AutoConfig.from_pretrained("../../../lab/gpt-j-6B-f16")
    inference = Inference(lambda: GPTJForCausalLM(config))
    loader = WeightLoader(".models/gpt.npx", ".models/gpt.meta", device="cuda")
    inference.attach_loader(loader)
    k, _ = Memory.alloc((30000, 10000), device="cuda")
    inference.init_model(memory_key=k)
    tokens = tokenizer("hello from pytorch-lit", return_tensors="pt")
    tokens = {k: tokens[k].cuda() for k in tokens}
    result = inference.forward(**tokens)
    print(result)
