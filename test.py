import argparse
import torch
import tiktoken
from gpt2 import GPT
from config import GPTConfig
from contextlib import nullcontext

def load_model(dir: str, name: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    checkpoint = torch.load(f"{dir}/{name}.pt", weights_only=True)
    model.load_state_dict(checkpoint)

    return model

def generate_text(
    input_text: str, 
    model: GPT, 
    encoder_model: str = "gpt2",
    num_samples: int = 5,
    temperature: float = 0.01,
    top_k: int = 200, 
    max_length: int = 150,
    stream: bool = True
) -> None:
    encoder = tiktoken.get_encoding(encoder_model)
    
    def enc(s):
        return encoder.encode(s)
    def dec(s):
        return encoder.decode(s)
    
    model.eval()
    input_ids = enc(input_text)
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    context = nullcontext() if torch.cuda.is_available() else torch.amp.autocast("cuda")
    # run generation
    print("Input:", input_text)
    
    if stream:
        # it is a yield function generate_stream that we can use to stream the generation
        with context:
            print(f"Output: {dec(x[0].tolist())}....", end='')
            for out in model.generate_stream(x, max_length=max_length, temperature=temperature, top_k=top_k):
                print(dec(out[-1].tolist()), end='')
    else:
        with context:
            for _ in range(num_samples):
                out = model.generate(x, max_length=max_length, temperature=temperature, top_k=top_k)
        print(dec(out[0].tolist()))

if __name__ == "__main__":
    model = load_model("models", "gpt2", GPTConfig())
    argparser = argparse.ArgumentParser()   
    argparser.add_argument("--text", type=str, default="China's production rate in 2024 is")
    text = argparser.parse_args().text
    generate_text(text, model)
