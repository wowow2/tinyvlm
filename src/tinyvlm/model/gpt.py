import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .attention import MultiHeadAttention

@dataclass
class GPTConfig:
    vocab_size: int = 0        # filled in from the tokenizer
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
        self.ffwd = FeedForward(cfg.n_embd, cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # residual connection around attention
        x = x + self.ffwd(self.ln2(x))   # residual connection around the MLP
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.block_size = cfg.block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self.forward(idx_cond, targets=None)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx