from transformers import AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers import AutoConfig
from transformers import pipeline
from pytorch_lit import LitModule

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../../../lab/gpt-j-6B-f16")
    config = AutoConfig.from_pretrained("../../../lab/gpt-j-6B-f16")
    model = LitModule.from_params(".models/gpt-j-6b-lit",
                                  lambda: GPTJForCausalLM(config),
                                  device="cuda")
    generator = pipeline('text-generation', model=model.module, tokenizer=tokenizer, device=0)
    result = generator("Hello I am gpt-j model and ", max_length=14, num_return_sequences=1)
    print(result)
