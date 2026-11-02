from dataclasses import dataclass
import torch
import os
from .checkpoint import save_checkpoint, load_checkpoint

@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    max_steps: int = 3000
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 500
    ckpt_path: str = "checkpoints/latest.pt"
    best_ckpt_path: str = "checkpoints/best.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    resume: bool = False


class Trainer:
    def __init__(self, model, train_ds, val_ds, cfg: TrainConfig, model_cfg, tokenizer=None):
        self.model = model.to(cfg.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.step = 0
        self.best_val_loss = float("inf")

    def save(self, path: str = None):
        target_path = path or self.cfg.ckpt_path
        save_checkpoint(
            target_path,
            self.model,
            self.optimizer,
            self.step,
            self.model_cfg,
            self.cfg,
            tokenizer=self.tokenizer,
            best_val_loss=self.best_val_loss,
        )

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        for name, ds in [("train", self.train_ds), ("val", self.val_ds)]:
            total_loss = torch.tensor(0.0, device=self.cfg.device)
            for _ in range(self.cfg.eval_iters):
                x, y = ds.get_batch(self.cfg.batch_size)
                _, loss = self.model(x, y)
                total_loss += loss
            out[name] = (total_loss / self.cfg.eval_iters).item()
        self.model.train()
        return out

    def train(self):
        torch.manual_seed(self.cfg.seed)
        if self.cfg.resume and os.path.exists(self.cfg.ckpt_path):
            self.step, state = load_checkpoint(
                self.cfg.ckpt_path, self.model, self.optimizer, self.cfg.device
            )
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            print(f"resumed from step {self.step}")
        self.model.train()
        while self.step < self.cfg.max_steps:
            if self.step % self.cfg.eval_interval == 0:
                metrics = self.estimate_loss()
                print(self.step, metrics)
                if self.step > 0 and metrics["val"] < self.best_val_loss:
                    self.best_val_loss = metrics["val"]
                    self.save(self.cfg.best_ckpt_path)
            x, y = self.train_ds.get_batch(self.cfg.batch_size, self.cfg.device)
            _, loss = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.step += 1
            if self.step % self.cfg.save_interval == 0:
                self.save()
        self.save()