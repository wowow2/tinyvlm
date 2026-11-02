import torch
import yaml

from tinyvlm.data.tokenizer import CharTokenizer
from tinyvlm.data.datasets import make_datasets
from tinyvlm.model.gpt import GPT, GPTConfig
from tinyvlm.train.trainer import Trainer, TrainConfig


def main():
    with open("configs/gpt_tiny.yaml") as f:
        raw = yaml.safe_load(f)

    with open("data/input.txt") as f:
        text = f.read()
    tok = CharTokenizer(text)

    model_cfg = GPTConfig(**raw["model"], vocab_size=tok.vocab_size)
    train_cfg = TrainConfig(**raw["train"])

    train_ds, val_ds = make_datasets(text, tok, block_size=model_cfg.block_size)
    model = GPT(model_cfg)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    trainer = Trainer(model, train_ds, val_ds, train_cfg, model_cfg)
    trainer.train()

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=train_cfg.device)
    out = model.generate(context, max_new_tokens=300)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()