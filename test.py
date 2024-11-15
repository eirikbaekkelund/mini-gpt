import torch
import tiktoken
from gpt2 import GPT
from contextlib import nullcontext


def generate_text(
    input_text: str, 
    model: GPT, 
    encoder_model: str = "gpt2",
    num_samples: int = 5,
    temperature: float = 0.2,
    top_k: int = 50, 
    max_length: int = 100
) -> None:
    encoder = tiktoken.get_encoding(encoder_model)
    
    def enc(s):
        return encoder.encode(s)
    def dec(s):
        return encoder.decode(s)

    input_ids = enc(input_text)
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    context = nullcontext() if torch.cuda.is_available() else torch.amp.autocast("cuda")
    # run generation
    with torch.no_grad():
        with context:
            for _ in range(num_samples):
                out = model.generate(x, max_length=max_length, temperature=temperature, top_k=top_k)
                print(dec(out[0].tolist()))
                print("="*80)
                print()