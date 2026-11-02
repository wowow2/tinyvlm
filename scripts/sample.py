import torch
from tinyvlm.data.tokenizer import CharTokenizer
from tinyvlm.model.gpt import GPT, GPTConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint and config
    ckpt = torch.load("checkpoints/tiny.pt", map_location=device, weights_only=False)
    model_cfg = GPTConfig(**ckpt["model_cfg"])

    # Prefer the vocab stored in the checkpoint; fall back to the corpus for old ckpts
    if "tokenizer" in ckpt:
        tok = CharTokenizer.from_chars(ckpt["tokenizer"]["chars"])
    else:
        with open("data/input.txt") as f:
            tok = CharTokenizer(f.read())

    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Choose your starting prompt (or leave empty for random start)
    prompt = "To be, or not to be\n"
    start_ids = tok.encode(prompt) if prompt else [0]
    context = torch.tensor([start_ids], dtype=torch.long, device=device)

    max_tokens = 1000
    print(f"--- Generating {max_tokens} tokens ---\n")
    out = model.generate(context, max_new_tokens=max_tokens)

    # Decode and print
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()