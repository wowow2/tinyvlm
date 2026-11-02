import torch
from tinyvlm.model.attention import Head


def test_causal():
    torch.manual_seed(0)
    h = Head(n_embd=16, head_size=8, block_size=8, dropout=0.0).eval()
    x = torch.randn(1, 8, 16)
    y1 = h(x)
    x2 = x.clone()
    x2[:, -1] += 1.0            # change only the last (future) token
    y2 = h(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1])   # earlier outputs unchanged
    assert not torch.allclose(y1[:, -1], y2[:, -1]) # the last output does change

if __name__ == '__main__':
    test_causal()