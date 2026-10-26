"""End-to-end smoke test on CPU with a tiny backbone and fake teachers."""

import copy

import numpy as np
import pytest
import torch

from conftest import CharTokenizer, TinyBackbone, fake_encode_fn, learnable_encode_fn, make_texts
from turkish_distill.checkpoint import list_checkpoints
from turkish_distill.logging_utils import read_metrics
from turkish_distill.models import StudentEncoder
from turkish_distill.teachers import TeacherCache, corpus_fingerprint, encode_teacher
from turkish_distill.train import run_training


def build(experiment, texts, encoder_factory=fake_encode_fn):
    fp = corpus_fingerprint(texts)
    caches = {}
    for i, t in enumerate(experiment.teachers):
        encode_teacher(
            t, texts, experiment.data.cache_dir, experiment.data.shard_size,
            encode_fn=encoder_factory(t.dim, i), fingerprint=fp, verbose=False,
        )
        caches[t.name] = TeacherCache.open(experiment.data.cache_dir, t, fp, len(texts))
    torch.manual_seed(0)
    model = StudentEncoder(TinyBackbone(), 16, {t.name: t.dim for t in experiment.teachers})
    return model, caches


def test_training_runs_logs_and_checkpoints(experiment, tmp_path):
    texts = make_texts(120)
    model, caches = build(experiment, texts)
    run_dir = tmp_path / "run"
    result = run_training(experiment, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"))
    assert result["final_step"] == 6
    records = read_metrics(run_dir / "metrics.jsonl")
    train = [r for r in records if "train/total" in r]
    diag = [r for r in records if "diag/mean_pairwise_cosine" in r]
    assert [r["step"] for r in train] == [2, 4, 6]
    assert [r["step"] for r in diag] == [0, 3, 6]  # step-0 baseline + periodic
    assert "diag/teacher_cosine/t_big" in diag[0] and "diag/geometry_agreement/t_small" in diag[0]
    assert all(torch.isfinite(torch.tensor(r["train/total"])) for r in train)
    assert [s for s, _ in list_checkpoints(run_dir / "checkpoints")] == [3, 6]
    assert (run_dir / "final" / "teacher_heads.pt").exists()


def test_loss_decreases_on_a_tiny_problem(experiment, tmp_path):
    experiment.train.max_steps = 150
    experiment.train.log_every = 25
    experiment.train.ckpt_every = 1000
    texts = make_texts(120)
    model, caches = build(experiment, texts, learnable_encode_fn)
    run_dir = tmp_path / "run"
    run_training(experiment, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"))
    records = read_metrics(run_dir / "metrics.jsonl")
    train = [r for r in records if "train/total" in r]
    assert train[-1]["train/total"] < 0.5 * train[0]["train/total"]
    diag = [r for r in records if "diag/teacher_cosine/t_small" in r]
    assert diag[-1]["diag/teacher_cosine/t_small"] > diag[0]["diag/teacher_cosine/t_small"]


def test_resume_continues_from_last_checkpoint(experiment, tmp_path):
    texts = make_texts(120)
    run_dir = tmp_path / "run"

    # First run stops at step 3 (as if the session died).
    first = copy.deepcopy(experiment)
    first.train.max_steps = 3
    model, caches = build(first, texts)
    run_training(first, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"))
    assert [s for s, _ in list_checkpoints(run_dir / "checkpoints")] == [3]

    # Second run with the full schedule resumes instead of restarting.
    model2, caches2 = build(experiment, texts)
    result = run_training(experiment, run_dir, model2, CharTokenizer(), texts, caches2, torch.device("cpu"))
    assert result["final_step"] == 6
    records = read_metrics(run_dir / "metrics.jsonl")
    # No second step-0 baseline, and training steps continue after 3.
    assert sum(1 for r in records if r["step"] == 0) == 1
    assert max(r["step"] for r in records if "train/total" in r) == 6
    assert [s for s, _ in list_checkpoints(run_dir / "checkpoints")] == [3, 6]


def test_validation_retrieval_is_logged_like_the_contrastive_runs(experiment, tmp_path):
    texts = make_texts(120)
    model, caches = build(experiment, texts)
    val_texts = make_texts(30)[10:]
    val = {"question": [t[:25] for t in val_texts], "passage": val_texts, "passage_id": [f"p{i}" for i in range(len(val_texts))]}
    run_dir = tmp_path / "run"
    run_training(experiment, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"), retrieval_val=val)
    records = read_metrics(run_dir / "metrics.jsonl")
    steps = [r["step"] for r in records if "val/mrr@10" in r]
    assert steps == [0, 3, 6]  # the same cadence as the diagnostics
    last = next(r for r in records if r["step"] == 6 and "val/mrr@10" in r)
    assert 0.0 < last["val/mrr@10"] <= 1.0 and last["val/num_passages"] == 20

    plain_run = tmp_path / "plain"
    model2, caches2 = build(experiment, texts)
    run_training(experiment, plain_run, model2, CharTokenizer(), texts, caches2, torch.device("cpu"))
    assert not any("val/mrr@10" in r for r in read_metrics(plain_run / "metrics.jsonl"))


def test_cached_step_matches_the_plain_step_for_the_distillation_loss(experiment):
    from turkish_distill.config import LossConfig
    from turkish_distill.losses import MultiTeacherLoss
    from turkish_distill.train import distill_step

    tok = CharTokenizer()
    enc = tok([f"ornek {i} metni, biraz uzunca {i * 7 % 5}" for i in range(12)], 64)
    gen = torch.Generator().manual_seed(1)
    targets = {t.name: torch.nn.functional.normalize(torch.randn(12, t.dim, generator=gen), dim=-1) for t in experiment.teachers}

    def run(chunk):
        torch.manual_seed(0)
        model = StudentEncoder(TinyBackbone(), 16, {t.name: t.dim for t in experiment.teachers})
        criterion = MultiTeacherLoss(LossConfig(resim_num_samples=64), experiment.teachers, seed=3)
        logs = distill_step(model, criterion, enc["input_ids"], enc["attention_mask"], targets, chunk_size=chunk)
        return logs, {n: p.grad.clone() for n, p in model.named_parameters()}

    plain_logs, plain = run(None)
    for chunk in (1, 5, 12):
        logs, cached = run(chunk)
        assert logs["total"] == pytest.approx(plain_logs["total"], rel=1e-5)
        for name, grad in cached.items():
            torch.testing.assert_close(grad, plain[name], rtol=1e-4, atol=1e-6, msg=f"{name} chunk={chunk}")



def test_a_phased_in_teacher_ramps_up_during_training_and_is_logged(experiment, tmp_path):
    experiment.teachers[1].weight_start = 0.0  # t_big (weight 2.0) starts at 0 and reaches 2.0 at step 6
    experiment.teachers[1].ramp_start_step = 0
    experiment.teachers[1].ramp_end_step = 6
    texts = make_texts(120)
    model, caches = build(experiment, texts)
    run_dir = tmp_path / "run"
    run_training(experiment, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"))
    train = [r for r in read_metrics(run_dir / "metrics.jsonl") if "train/total" in r]
    ramp = [r["train/weight/t_big"] for r in train]  # means over steps (0,1), (2,3), (4,5) of 0, 1/3, 2/3, ... * 2
    assert ramp == sorted(ramp) and 0.0 < ramp[0] < ramp[-1] < 2.0
    assert all(r["train/weight/t_small"] == 1.0 for r in train)


def test_training_runs_with_only_the_vector_term_or_only_the_relation_terms(experiment, tmp_path):
    texts = make_texts(120)
    results = {}
    for name, weights in {"vector": (10.0, 0.0, 0.0), "relation": (0.0, 200.0, 20.0)}.items():
        exp = copy.deepcopy(experiment)
        exp.loss.cosine_weight, exp.loss.sim_weight, exp.loss.resim_weight = weights
        model, caches = build(exp, texts)
        heads_before = {k: v.clone() for k, v in model.heads.state_dict().items()}
        run_dir = tmp_path / name
        run_training(exp, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"))
        train = [r for r in read_metrics(run_dir / "metrics.jsonl") if "train/total" in r]
        assert all(torch.isfinite(torch.tensor(r["train/total"])) for r in train)
        heads_changed = any(not torch.equal(heads_before[k], v) for k, v in model.heads.state_dict().items())
        results[name] = (heads_changed, set(k.split("/")[1] for r in train for k in r if k.startswith("train/") and "/" in k[6:]))
    assert results["vector"][0] is True and results["relation"][0] is False  # no vector term, no gradient to the heads
    assert "cos" in results["vector"][1] and "sim" not in results["vector"][1] and "resim" not in results["vector"][1]
    assert "cos" not in results["relation"][1] and {"sim", "resim"} <= results["relation"][1]


@pytest.mark.parametrize(
    "method",
    ["center", "gram_standardize", "shared_relation", "fusion", "pca", "listwise", "neighbor"],
)
def test_training_runs_with_each_target_method(experiment, tmp_path, method):
    """Methods a-d run through the real loop (heads, loss, diagnostics, checkpoints) and stay finite."""
    import dataclasses

    from turkish_distill.config import FusionConfig
    from turkish_distill.train import apply_target_methods

    texts = make_texts(120)
    model, caches = build(experiment, texts)  # the fake teachers t_big and t_small
    cfg = experiment
    if method == "center":
        cfg.teachers[0].center = True
    elif method == "gram_standardize":
        cfg.loss.gram_standardize = True
    elif method == "shared_relation":
        cfg.loss.relation_mode = "shared"
    elif method == "listwise":
        cfg.loss.listwise_weight = 5.0
    elif method == "neighbor":
        cfg.train.neighbor_fraction = 0.5
        cfg.train.neighbor_group_size = 4
        cfg.loss.listwise_weight = 5.0
    elif method == "pca":
        cfg.teachers[0].pca_dim = 4
        cfg.teachers[0].pca_fit_samples = 60
    else:
        cfg.fusion = FusionConfig(mode="aligned_mean", reference=cfg.teachers[0].name, dim=8, fit_samples=60)
    cfg, caches = apply_target_methods(cfg, caches)
    if method == "fusion":
        model = StudentEncoder(TinyBackbone(), 16, {t.name: t.dim for t in cfg.teachers})
        assert [t.name for t in cfg.teachers] == ["fused"]
    run_dir = tmp_path / "run"
    labels = np.arange(len(texts)) % 6 if method == "neighbor" else None  # six neighbourhoods
    result = run_training(cfg, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"), neighbor_labels=labels)
    assert result["final_step"] == 6 and result["last_loss"] is not None and torch.isfinite(torch.tensor(result["last_loss"]))
    records = read_metrics(run_dir / "metrics.jsonl")
    assert all(torch.isfinite(torch.tensor(r["train/total"])) for r in records if "train/total" in r)
    if method == "shared_relation":
        assert any("train/sim/shared" in r for r in records)
