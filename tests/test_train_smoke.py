"""End-to-end smoke test on CPU with a tiny backbone and fake teachers."""

import copy

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
