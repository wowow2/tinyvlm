import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Single head of causal self-attention.

        Each token builds a query, key and value from its embedding. Its query is
        scored against the keys of itself and all earlier tokens, the scores are
        turned into weights, and the token's output is the weighted average of
        those tokens' values. Later tokens are masked out, so position t only sees
        positions 0..t.
        """
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)        # (B,T,head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5        # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # a token can't see later tokens
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v                                             # (B,T,head_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))