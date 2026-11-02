import torch

from tinyvlm.data.datasets import TokenDataset
from tinyvlm.model.gpt import GPT, GPTConfig
from tinyvlm.train.trainer import Trainer, TrainConfig

VOCAB = 17
BLOCK = 8


def make_data():
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (2000,), generator=g)
    return TokenDataset(tokens[:1800], BLOCK), TokenDataset(tokens[1800:], BLOCK)


def make_trainer(init_seed, ckpt_path, max_steps, resume):
    torch.manual_seed(init_seed)
    model_cfg = GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=2,
                          n_head=2, n_embd=16, dropout=0.0)
    model = GPT(model_cfg)
    train_ds, val_ds = make_data()
    cfg = TrainConfig(
        batch_size=4, lr=1e-3, max_steps=max_steps,
        eval_interval=1000, eval_iters=1, save_interval=10,
        ckpt_path=str(ckpt_path), device="cpu", seed=1337, resume=resume,
    )
    return Trainer(model, train_ds, val_ds, cfg, model_cfg)


def test_resume_matches_uninterrupted(tmp_path):
    # A: 20 steps straight through
    a = make_trainer(0, tmp_path / "a.pt", max_steps=20, resume=False)
    a.train()

    # B1: 10 steps, which saves a checkpoint at step 10
    b1 = make_trainer(0, tmp_path / "b.pt", max_steps=10, resume=False)
    b1.train()

    # B2: brand-new model with a different init, resumes from step 10 to 20
    b2 = make_trainer(99, tmp_path / "b.pt", max_steps=20, resume=True)
    b2.train()

    assert b2.step == 20
    for (name, pa), (_, pb) in zip(a.model.named_parameters(),
                                   b2.model.named_parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), f"{name} differs after resume"