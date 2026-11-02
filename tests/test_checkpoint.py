import torch

from tinyvlm.data.datasets import TokenDataset
from tinyvlm.data.tokenizer import CharTokenizer
from tinyvlm.model.gpt import GPT, GPTConfig
from tinyvlm.train.trainer import Trainer, TrainConfig
from tinyvlm.train.checkpoint import save_checkpoint, load_checkpoint

VOCAB = 17
BLOCK = 8


def make_data():
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (2000,), generator=g)
    return TokenDataset(tokens[:1800], BLOCK), TokenDataset(tokens[1800:], BLOCK)


def make_trainer(init_seed, ckpt_path, max_steps, resume, tokenizer=None,
                 best_ckpt_path=None, eval_interval=1000):
    torch.manual_seed(init_seed)
    model_cfg = GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=2,
                          n_head=2, n_embd=16, dropout=0.0)
    model = GPT(model_cfg)
    train_ds, val_ds = make_data()
    cfg = TrainConfig(
        batch_size=4, lr=1e-3, max_steps=max_steps,
        eval_interval=eval_interval, eval_iters=1, save_interval=10,
        ckpt_path=str(ckpt_path), device="cpu", seed=1337, resume=resume,
        best_ckpt_path=str(best_ckpt_path) if best_ckpt_path else "checkpoints/best.pt",
    )
    return Trainer(model, train_ds, val_ds, cfg, model_cfg, tokenizer=tokenizer)


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


def test_best_checkpoint_has_tokenizer_and_best_loss(tmp_path):
    tok = CharTokenizer("abcdefghij")
    best = tmp_path / "best.pt"
    t = make_trainer(0, tmp_path / "c.pt", max_steps=3, resume=False,
                     tokenizer=tok, best_ckpt_path=best, eval_interval=1)
    t.train()

    assert best.exists()
    step, state = load_checkpoint(str(best), t.model, device="cpu")
    assert state["tokenizer"]["chars"] == tok.chars
    assert state["best_val_loss"] < float("inf")
    restored = CharTokenizer.from_chars(state["tokenizer"]["chars"])
    assert restored.stoi == tok.stoi


def test_save_checkpoint_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tok = CharTokenizer("abc")
    model_cfg = GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=1,
                          n_head=1, n_embd=8, dropout=0.0)
    model = GPT(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = TrainConfig(device="cpu")

    save_checkpoint("tiny.pt", model, optimizer, 7, model_cfg, cfg,
                    tokenizer=tok, best_val_loss=1.5)
    step, state = load_checkpoint("tiny.pt", model, device="cpu")
    assert step == 7
    assert state["best_val_loss"] == 1.5
    assert state["tokenizer"]["chars"] == tok.chars