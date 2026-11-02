import os
from dataclasses import asdict

import torch

def save_checkpoint(path, model, optimizer, step, model_cfg, train_cfg):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
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