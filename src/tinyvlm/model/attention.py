import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        # single projection matrix computes Q, K, and V for all heads at once
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Time (sequence length), Channels (n_embd)

        # Project to Q, K, V in one shot -> Shape: (B, T, 3 * C)
        qkv = self.qkv_proj(x)

        # Split into separate Q, K, V tensors -> Each is: (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape so 'n_head' becomes a batch dimension -> Shape: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Fused GPU FlashAttention kernel
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Concatenate heads back together -> Shape: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))