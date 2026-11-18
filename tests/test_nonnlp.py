"""Methods adapted from outside NLP: weight averaging (EMA/SWA), data pruning and importance sampling, progressive
input length, SAM, uncertainty weighting and one teacher per step."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from conftest import CharTokenizer, make_texts
from test_train_smoke import build
from turkish_distill.averaging import WeightAverager, make_averagers
from turkish_distill.config import LossConfig, TeacherConfig, TrainConfig, load_config
from turkish_distill.data import Collator, sampling_plan, weighted_epoch_batches
from turkish_distill.logging_utils import read_metrics
from turkish_distill.losses import MultiTeacherLoss
from turkish_distill.score_texts import load_scores, save_scores, score_corpus
from turkish_distill.teachers import corpus_fingerprint
from turkish_distill.train import pick_teacher, run_training, sam_perturb


def test_ema_and_uniform_averages():
    lin = nn.Linear(2, 1, bias=False)
    ema = WeightAverager("ema", decay=0.5)
    swa = WeightAverager("uniform", start_step=2, every=2)
    seen = []
    for step in range(1, 7):
        with torch.no_grad():
            lin.weight.fill_(float(step))
        ema.update(lin, step)
        if swa.update(lin, step):
            seen.append(step)
    assert seen == [2, 4, 6]
    torch.testing.assert_close(swa.avg["weight"], torch.full((1, 2), 4.0))  # mean of 2, 4, 6
    # ema: first update copies, then d = min(0.5, (1+n)/(10+n)) = 0.1818.., 0.25, 0.2857.., 0.333.., 0.3636..
    expect = 1.0
    for n, w in zip(range(2, 7), range(2, 7)):
        d = min(0.5, (1 + n) / (10 + n))
        expect = d * expect + (1 - d) * w
    assert float(ema.avg["weight"][0, 0]) == pytest.approx(expect, rel=1e-6)
    with swa.applied(lin):
        assert float(lin.weight[0, 0]) == 4.0
    assert float(lin.weight[0, 0]) == 6.0  # restored
    again = WeightAverager("uniform", start_step=2, every=2)
    again.load_state_dict(swa.state_dict())
    assert again.count == 3 and torch.equal(again.avg["weight"], swa.avg["weight"])
    late = WeightAverager("uniform", start_step=4, every=10)
    assert not late.update(lin, 3, force=True) and late.update(lin, 5, force=True)  # force only once started
    assert set(make_averagers(TrainConfig(ema_decay=0.99, swa_start=0.5, max_steps=100))) == {"ema", "swa"}
    assert make_averagers(TrainConfig()) == {}


def test_sampling_plans_and_weighted_batches():
    idx = np.arange(100)
    scores = np.linspace(0.0, 1.0, 120)
    kept, probs, _ = sampling_plan(idx, scores, "prune_easy", 0.3, 1.0)
    assert probs is None and kept.tolist() == list(range(70, 100))  # the hardest 30 of the training texts
    kept, _, _ = sampling_plan(idx, scores, "prune_hard", 0.3, 1.0)
    assert kept.tolist() == list(range(30))
    same, probs, summary = sampling_plan(idx, scores, "weighted", 0.3, 1.0)
    assert np.array_equal(same, idx) and probs.sum() == pytest.approx(1.0) and probs.min() > 0
    assert probs[99] > probs[50] > probs[1] and summary["score_mean_sampled"] > summary["score_mean_all"]
    a = weighted_epoch_batches(idx, probs, 10, seed=3, epoch=1)
    assert len(a) == 10 and all(len(b) == 10 for b in a)
    assert weighted_epoch_batches(idx, probs, 10, seed=3, epoch=1, start_batch=4) == a[4:]  # resumable
    drawn = np.concatenate([np.array(b) for e in range(30) for b in weighted_epoch_batches(idx, probs, 10, 3, e)])
    assert (drawn >= 50).mean() > 0.6  # high-score texts are drawn more often


def test_scores_file_roundtrip_and_corpus_check(tmp_path):
    scores = {"score": np.array([0.1, 0.2, 0.3], dtype=np.float32), "err_a": np.zeros(3, dtype=np.float32)}
    summary = save_scores(tmp_path / "s.npz", scores, "abc", ["a"])
    assert summary["n"] == 3 and (tmp_path / "s.json").exists()
    np.testing.assert_allclose(load_scores(tmp_path / "s.npz", "abc", 3), [0.1, 0.2, 0.3], rtol=1e-6)
    with pytest.raises(ValueError, match="another corpus"):
        load_scores(tmp_path / "s.npz", "other", 3)


def test_score_corpus_uses_the_heads(experiment):
    texts = make_texts(40)
    model, caches = build(experiment, texts)
    collator = Collator(CharTokenizer(), texts, caches, 64)
    weights = {"t_small": 1.0, "t_big": 3.0}
    out = score_corpus(model, collator, len(texts), weights, torch.device("cpu"), batch_size=16)
    assert out["score"].shape == (40,) and set(out) == {"score", "err_t_small", "err_t_big"}
    np.testing.assert_allclose(out["score"], (out["err_t_small"] + 3 * out["err_t_big"]) / 4, rtol=1e-5)
    assert ((out["score"] >= 0) & (out["score"] <= 2)).all()


def test_pick_teacher_is_deterministic_and_weighted():
    weights = {"a": 1.0, "b": 3.0, "c": 0.0}
    picks = [pick_teacher(weights, 7, s) for s in range(2000)]
    assert picks == [pick_teacher(weights, 7, s) for s in range(2000)] and "c" not in picks
    assert 0.7 < picks.count("b") / len(picks) < 0.8


def test_sam_perturb_moves_along_the_normalised_gradient():
    lin = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    lin.weight.grad = torch.tensor([[3.0, 0.0, 4.0]])
    moved = sam_perturb(lin, 0.5)
    torch.testing.assert_close(lin.weight, torch.tensor([[1.3, 2.0, 3.4]]))
    for p, e in moved:
        p.data.sub_(e)
    torch.testing.assert_close(lin.weight, torch.tensor([[1.0, 2.0, 3.0]]))


def test_uncertainty_weighting_and_active_teachers():
    teachers = [TeacherConfig(name="a", hf_id="x", dim=4), TeacherConfig(name="b", hf_id="y", dim=4)]
    pooled = torch.randn(6, 5)
    proj = {"a": torch.randn(6, 4), "b": torch.randn(6, 4)}
    targets = {"a": torch.randn(6, 4), "b": torch.randn(6, 4)}
    plain = MultiTeacherLoss(LossConfig(resim_weight=0), teachers)
    uw = MultiTeacherLoss(LossConfig(resim_weight=0, uncertainty_weighting=True), teachers)
    assert len(plain.log_vars) == 0 and len(uw.log_vars) == 10
    torch.testing.assert_close(uw(pooled, proj, targets)[0], plain(pooled, proj, targets)[0])  # s = 0: identical
    with torch.no_grad():
        uw.log_vars["a__cosine"].fill_(1.0)
    total, logs = uw(pooled, proj, targets)
    assert logs["uw/a/cosine"] == pytest.approx(float(torch.exp(torch.tensor(-1.0))))
    assert float(total) != pytest.approx(float(plain(pooled, proj, targets)[0]))
    total.backward()
    assert uw.log_vars["a__cosine"].grad is not None and uw.log_vars["a__bank"].grad is None
    plain.set_active(["b"])
    assert plain.current_weights() == {"a": 0.0, "b": 1.0}
    _, logs = plain(pooled, proj, targets)
    assert "cos/b" in logs and "cos/a" not in logs
    plain.set_active(None)
    assert plain.current_weights() == {"a": 1.0, "b": 1.0}


def run(exp, tmp_path, name, texts=None):
    texts = texts or make_texts(120)
    model, caches = build(exp, texts)
    result = run_training(exp, tmp_path / name, model, CharTokenizer(), texts, caches, torch.device("cpu"))
    return result, model, [r for r in read_metrics(tmp_path / name / "metrics.jsonl") if "train/total" in r]


def test_training_with_averaging_length_sam_uncertainty_and_one_teacher(experiment, tmp_path):
    exp = copy.deepcopy(experiment)
    exp.train.ema_decay = 0.9
    exp.train.swa_start = 0.5
    exp.train.swa_every = 1
    exp.train.short_max_length = 10
    exp.train.short_until = 0.5
    exp.train.sam_rho = 0.05
    exp.train.teacher_sampling = "one"
    exp.loss.uncertainty_weighting = True
    exp.loss.uncertainty_lr = 0.1
    result, model, records = run(exp, tmp_path, "all")
    assert result["final_step"] == 6
    run_dir = tmp_path / "all"
    assert (run_dir / "final_ema" / "teacher_heads.pt").exists() and (run_dir / "final_swa" / "teacher_heads.pt").exists()
    metrics = read_metrics(run_dir / "metrics.jsonl")
    assert any("diag_ema/mean_pairwise_cosine" in r for r in metrics) and any("diag_swa/effective_rank" in r for r in metrics)
    assert all(np.isfinite(r["train/total"]) and "train/sam_total" in r for r in records)
    assert any(k.startswith("train/uw/") for r in records for k in r)
    # resuming the finished run restores the averagers and the log-variances from the last checkpoint
    exp2 = copy.deepcopy(exp)
    exp2.train.max_steps = 8
    texts = make_texts(120)
    model2, caches = build(exp2, texts)
    out = run_training(exp2, run_dir, model2, CharTokenizer(), texts, caches, torch.device("cpu"))
    assert out["final_step"] == 8


def test_progressive_length_changes_the_inputs(experiment, tmp_path):
    exp = copy.deepcopy(experiment)
    exp.train.short_max_length = 5
    exp.train.short_until = 0.5
    _, _, short = run(exp, tmp_path, "short")
    _, _, full = run(copy.deepcopy(experiment), tmp_path, "full")
    assert short[0]["train/total"] != pytest.approx(full[0]["train/total"])  # steps 0-1 see 5-token inputs
    assert len(short) == len(full) == 3


def test_training_with_pruned_and_weighted_sampling(experiment, tmp_path):
    texts = make_texts(120)
    fp = corpus_fingerprint(texts)
    save_scores(tmp_path / "scores.npz", {"score": np.linspace(0, 1, 120).astype(np.float32)}, fp, ["t_small"])
    for mode in ("prune_easy", "weighted"):
        exp = copy.deepcopy(experiment)
        exp.train.sampling = mode
        exp.train.sample_scores = str(tmp_path / "scores.npz")
        exp.train.keep_fraction = 0.5
        result, _, records = run(exp, tmp_path, mode, texts)
        assert result["final_step"] == 6 and all(np.isfinite(r["train/total"]) for r in records)
    exp = copy.deepcopy(experiment)
    exp.train.sampling = "prune_easy"
    exp.train.sample_scores = str(tmp_path / "scores.npz")
    with pytest.raises(ValueError, match="another corpus"):
        run(exp, tmp_path, "wrong", make_texts(121))


def test_config_validation_and_unchanged_fingerprint(tmp_path):
    raw = {"name": "x", "output_dir": str(tmp_path), "teachers": [{"name": "t", "hf_id": "h", "dim": 4},
                                                                  {"name": "u", "hf_id": "h", "dim": 4}]}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    off = load_config(path)
    assert load_config(path, ["train.ema_decay=0.999"]).fingerprint() != off.fingerprint()
    assert load_config(path, ["train.ema_decay=0.0"]).fingerprint() == off.fingerprint()  # default: left out
    load_config(path, ["train.sampling=weighted", "train.sample_scores=s.npz"])
    load_config(path, ["train.teacher_sampling=one", "loss.uncertainty_weighting=true"])
    for bad in (["train.ema_decay=1.5"], ["train.swa_start=1.0"], ["train.sampling=weighted"], ["train.sampling=random"],
                ["train.short_max_length=1000"], ["train.sam_rho=-1"], ["train.teacher_sampling=two"],
                ["loss.uncertainty_weighting=true", "train.grad_probe_every=10"],
                ["train.sampling=prune_easy", "train.sample_scores=s", "train.neighbor_fraction=0.5"]):
        with pytest.raises(ValueError):
            load_config(path, bad)
