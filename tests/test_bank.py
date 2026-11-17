"""Teacher-bank terms: (A) relational KL over a bank of cached teacher vectors, (B) contrastive InfoNCE against it."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from conftest import CharTokenizer, make_texts
from test_train_smoke import build
from turkish_distill.bank import TeacherBank
from turkish_distill.config import LossConfig, TeacherConfig, load_config
from turkish_distill.logging_utils import read_metrics
from turkish_distill.losses import MultiTeacherLoss, bank_kl_loss, bank_nce_loss
from turkish_distill.train import run_training


class ArrayCache:
    def __init__(self, x):
        self.x = x

    def get(self, indices):
        return self.x[np.asarray(indices)]


def unit(n, d, seed):
    return F.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_bank_draws_training_texts_deterministically_per_round():
    x = np.random.default_rng(0).standard_normal((100, 6)).astype(np.float32)
    train_idx = np.arange(80)
    bank = TeacherBank({"t": ArrayCache(x)}, train_idx, size=30, seed=5, refresh_every=10)
    assert bank.at_step(0) and not bank.at_step(9)
    first = bank.indices.clone()
    assert len(set(first.tolist())) == 30 and first.max() < 80  # no repeats, never a held-out text
    torch.testing.assert_close(bank.vectors["t"], F.normalize(torch.from_numpy(x[first.numpy()]), dim=-1))
    assert bank.at_step(10) and not torch.equal(bank.indices, first)
    again = TeacherBank({"t": ArrayCache(x)}, train_idx, size=30, seed=5, refresh_every=10)
    again.at_step(3)
    assert torch.equal(again.indices, first)  # depends on the step only: a resumed run sees the same bank
    fixed = TeacherBank({"t": ArrayCache(x)}, train_idx, size=500, seed=5, refresh_every=0)
    assert fixed.at_step(0) and not fixed.at_step(10_000) and fixed.size == 80  # capped at the training texts


def test_kl_is_zero_at_the_teacher_and_ignores_the_own_text():
    target, bank = unit(6, 8, 0), unit(20, 8, 1)
    assert float(bank_kl_loss(target.clone(), target, bank, None, 0.05, 0.05)) == pytest.approx(0.0, abs=1e-6)
    pred = unit(6, 8, 2).requires_grad_(True)
    assert float(bank_kl_loss(pred, target, bank, None, 0.05, 0.05)) > 0
    # put row 0's own teacher vector in the bank: masked, the loss equals the one without that bank entry
    with_self = torch.cat([bank, target[:1]])
    mask = torch.zeros(6, 21, dtype=torch.bool)
    mask[0, 20] = True
    masked = bank_kl_loss(pred, target, with_self, mask, 0.02, 0.05)
    plain0 = bank_kl_loss(pred[:1], target[:1], bank, None, 0.02, 0.05)
    rest = bank_kl_loss(pred[1:], target[1:], with_self, None, 0.02, 0.05)
    torch.testing.assert_close(masked, (plain0 + 5 * rest) / 6)
    masked.backward()
    assert torch.isfinite(pred.grad).all()


def test_nce_matches_a_hand_computed_cross_entropy():
    pred, target, bank = unit(4, 8, 3), unit(4, 8, 4), unit(10, 8, 5)
    loss, acc = bank_nce_loss(pred, target, bank, None, 0.1)
    logits = torch.cat([(pred * target).sum(-1, keepdim=True), pred @ bank.T], dim=-1) / 0.1
    torch.testing.assert_close(loss, F.cross_entropy(logits, torch.zeros(4, dtype=torch.long)))
    loss, acc = bank_nce_loss(target.clone(), target, bank, None, 0.1)
    assert float(acc) == 1.0  # a perfect student ranks its own teacher vector first


def teachers2():
    return [TeacherConfig(name="a", hf_id="x", dim=8), TeacherConfig(name="b", hf_id="y", dim=6,
                                                                       loss_weights={"bank_nce": 0})]


def test_loss_adds_the_bank_terms_per_teacher_and_needs_a_bank():
    cfg = LossConfig(bank_size=10, bank_kl_weight=2.0, bank_nce_weight=1.0, sim_weight=0, resim_weight=0, cosine_weight=0)
    crit = MultiTeacherLoss(cfg, teachers2())
    pooled = torch.randn(5, 4)
    proj = {"a": torch.randn(5, 8), "b": torch.randn(5, 6)}
    targets = {"a": unit(5, 8, 1), "b": unit(5, 6, 2)}
    with pytest.raises(ValueError, match="no loaded bank"):
        crit(pooled, proj, targets)
    bank = TeacherBank({"a": ArrayCache(np.random.randn(50, 8)), "b": ArrayCache(np.random.randn(50, 6))},
                       np.arange(40), 10, seed=0)
    bank.at_step(0)
    crit.set_bank(bank)
    crit.set_batch_indices(bank.indices[:5].tolist())  # every text of the batch is in the bank: all masked
    total, logs = crit(pooled, proj, targets)
    assert {"bank_kl/a", "bank_nce/a", "bank_acc/a", "bank_kl/b"} <= set(logs) and "bank_nce/b" not in logs
    expect = (2 * logs["bank_kl/a"] + logs["bank_nce/a"] + 2 * logs["bank_kl/b"]) / 2
    assert float(total) == pytest.approx(expect, rel=1e-5)


def test_training_with_the_bank_logs_its_terms(experiment, tmp_path):
    exp = copy.deepcopy(experiment)
    exp.loss.bank_size = 40
    exp.loss.bank_refresh_every = 3
    exp.loss.bank_kl_weight = 1.0
    exp.loss.bank_nce_weight = 1.0
    exp.train.chunk_size = 4
    texts = make_texts(120)
    model, caches = build(exp, texts)
    result = run_training(exp, tmp_path / "run", model, CharTokenizer(), texts, caches, torch.device("cpu"))
    assert result["final_step"] == 6
    records = [r for r in read_metrics(tmp_path / "run" / "metrics.jsonl") if "train/total" in r]
    names = [t.name for t in exp.teachers]
    assert records and all(f"train/bank_kl/{n}" in r and f"train/bank_nce/{n}" in r for r in records for n in names)
    assert all(np.isfinite(r["train/total"]) for r in records)


def test_config_validation(tmp_path):
    raw = {"name": "x", "output_dir": str(tmp_path), "teachers": [{"name": "t", "hf_id": "h", "dim": 4}]}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    off = load_config(path)
    on = load_config(path, ["loss.bank_size=1000", "loss.bank_kl_weight=1"])
    assert on.fingerprint() != off.fingerprint()
    for bad in (["loss.bank_kl_weight=1"], ["loss.bank_size=10", "loss.bank_temperature=0"],
                ["loss.bank_size=-1"], ["teachers.0.loss_weights.bank_x=0"]):
        with pytest.raises(ValueError):
            load_config(path, bad)
    load_config(path, ["teachers.0.loss_weights.bank_nce=0"])
