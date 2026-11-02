import os
from dataclasses import asdict

import torch

def save_checkpoint(path, model, optimizer, step, model_cfg, train_cfg,
                    tokenizer=None, best_val_loss=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_version": str(torch.__version__),
    }
    if tokenizer is not None:
        state["tokenizer"] = {"chars": list(tokenizer.chars)}
    if best_val_loss is not None:
        state["best_val_loss"] = best_val_loss
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["step"], state
