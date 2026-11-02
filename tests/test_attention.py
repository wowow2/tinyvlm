import torch
from tinyvlm.model.attention import MultiHeadAttention

def test_causal():
    torch.manual_seed(0)
    mha = MultiHeadAttention(n_embd=16, n_head=2, block_size=8, dropout=0.0).eval()
    x = torch.randn(1, 8, 16)
    y1 = mha(x)

    x2 = x.clone()
    x2[:, -1] += 1.0  # change only the last (future) token
    y2 = mha(x2)

    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-6)
    assert not torch.allclose(y1[:, -1], y2[:, -1])

if __name__ == '__main__':
    test_causal()