from transformers import AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers import AutoConfig
from pytorch_lit import LitModule

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../../../lab/gpt-j-6B-f16")
    config = AutoConfig.from_pretrained("../../../lab/gpt-j-6B-f16")
    model = LitModule.from_params(".models/gpt-j-6b-lit",
                                  lambda: GPTJForCausalLM(config),
                                  device="cuda")
    tokens = tokenizer("hello from pytorch-lit", return_tensors="pt")
    tokens = {k: tokens[k].cuda() for k in tokens}
    result = model(**tokens)
    print(result)
