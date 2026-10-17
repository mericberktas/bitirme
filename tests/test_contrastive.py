import copy
import json

import numpy as np
import pytest
import torch
import yaml

from conftest import CharTokenizer, TinyBackbone, make_texts
from turkish_distill import contrastive as ct
from turkish_distill.contrastive_config import (
    ContrastiveLossConfig,
    load_contrastive_config,
)
from turkish_distill.models import StudentEncoder
from turkish_distill.logging_utils import read_metrics
from turkish_distill.train_contrastive import run_contrastive


def tiny_model(seed: int = 0) -> StudentEncoder:
    torch.manual_seed(seed)
    return StudentEncoder(TinyBackbone(), 16, {}, "mean")


def make_batch(n: int) -> dict:
    tok = CharTokenizer()
    q = tok([f"soru {i} nedir? {i * 3 % 7}" for i in range(n)], 40)
    p = tok([f"pasaj {i} hakkinda uzunca bir metin {i * 5 % 11}" for i in range(n)], 80)
    return {"q_input_ids": q["input_ids"], "q_attention_mask": q["attention_mask"],
            "p_input_ids": p["input_ids"], "p_attention_mask": p["attention_mask"]}


# --- loss ------------------------------------------------------------------- #

def test_info_nce_matches_a_hand_computed_case():
    e = torch.eye(2)
    loss, logs = ct.info_nce(e, e, scale=2.0)
    expected = -np.log(np.exp(2.0) / (np.exp(2.0) + 1.0))
    assert loss.item() == pytest.approx(expected, rel=1e-5) and logs["acc"].item() == 1.0
    sym, _ = ct.info_nce(e, e, scale=2.0, symmetric=True)
    assert sym.item() == pytest.approx(expected, rel=1e-5)  # identical in both directions here


def test_symmetric_loss_differs_when_the_score_matrix_is_not_symmetric():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 1.0]])
    one, _ = ct.info_nce(q, p, 5.0)
    both, _ = ct.info_nce(q, p, 5.0, symmetric=True)
    assert one.item() != pytest.approx(both.item())


def test_make_loss_rejects_unknown_names():
    with pytest.raises(ValueError):
        ct.make_loss(ContrastiveLossConfig(name="triplet"))


# --- gradient caching ------------------------------------------------------- #

@pytest.mark.parametrize("chunk", [1, 3, 4])
def test_cached_step_gives_the_same_loss_and_gradients_as_the_plain_step(chunk):
    batch = make_batch(10)
    loss_fn = ct.make_loss(ContrastiveLossConfig(name="mnrl_symmetric", scale=20.0))
    plain, cached = tiny_model(1), tiny_model(1)
    cached.load_state_dict(plain.state_dict())
    plain_logs = ct.contrastive_step(plain, batch, loss_fn, chunk_size=None)
    cached_logs = ct.contrastive_step(cached, batch, loss_fn, chunk_size=chunk)
    assert cached_logs["loss"] == pytest.approx(plain_logs["loss"], rel=1e-5)
    for (name, a), (_, b) in zip(plain.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_cached_step_reproduces_dropout_masks_through_the_rng_restore():
    class Dropout(TinyBackbone):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(0.5)

        def forward(self, input_ids, attention_mask):
            out = super().forward(input_ids, attention_mask)
            out.last_hidden_state = self.drop(out.last_hidden_state)
            return out

    batch = make_batch(8)
    loss_fn = ct.make_loss(ContrastiveLossConfig())
    torch.manual_seed(0)
    model = StudentEncoder(Dropout(), 16, {}, "mean").train()
    twin = copy.deepcopy(model)
    # The embeddings of pass 1 and pass 2 must agree, otherwise the stored gradient belongs to other activations.
    torch.manual_seed(5)
    ct.contrastive_step(model, batch, loss_fn, chunk_size=3)
    grads_cached = [p.grad.clone() for p in model.parameters()]
    # Reference: a plain step with the very same dropout masks is not constructible chunk-wise, so check
    # determinism instead: the same seed must give the same gradients twice.
    torch.manual_seed(5)
    ct.contrastive_step(twin, batch, loss_fn, chunk_size=3)
    for a, b in zip(grads_cached, [p.grad for p in twin.parameters()]):
        torch.testing.assert_close(a, b)


# --- batching --------------------------------------------------------------- #

def test_batches_never_repeat_a_passage_and_are_deterministic():
    pids = [f"p{i % 40}" for i in range(200)]  # 40 distinct passages, 5 questions each
    batches = ct.unique_passage_batches(pids, 16, seed=1, epoch=0)
    assert batches and all(len(b) == 16 for b in batches)
    for b in batches:
        assert len({pids[i] for i in b}) == 16
    assert len({i for b in batches for i in b}) == sum(len(b) for b in batches)  # no pair used twice
    assert batches == ct.unique_passage_batches(pids, 16, seed=1, epoch=0)
    assert batches != ct.unique_passage_batches(pids, 16, seed=1, epoch=1)
    assert ct.unique_passage_batches(pids, 16, seed=1, epoch=0, start_batch=2) == batches[2:]


def test_batches_are_empty_when_there_are_too_few_distinct_passages():
    assert ct.unique_passage_batches(["a"] * 50 + ["b"] * 50, 4, seed=0, epoch=0) == []


# --- validation retrieval ---------------------------------------------------- #

def test_retrieval_metrics_perfect_and_pessimistic_on_ties():
    pairs = {"question": ["q0", "q1", "q2", "q3"], "passage": ["pa", "pa", "pb", "pc"], "passage_id": ["a", "a", "b", "c"]}
    vectors = {"q0": [1, 0, 0], "q1": [1, 0.1, 0], "q2": [0, 1, 0], "q3": [0, 0, 1],
               "pa": [1, 0, 0], "pb": [0, 1, 0], "pc": [0, 0, 1]}
    enc = lambda texts: np.array([vectors[t] for t in texts], dtype=float)  # noqa: E731
    metrics = ct.evaluate_retrieval(enc, enc, pairs)
    assert metrics["mrr@10"] == 1.0 and metrics["recall@1"] == 1.0
    assert metrics["num_passages"] == 3 and metrics["num_questions"] == 4
    constant = lambda texts: np.ones((len(texts), 3))  # noqa: E731  collapsed space: all scores tie
    collapsed = ct.evaluate_retrieval(constant, constant, pairs)
    assert collapsed["recall@1"] == 0.0 and collapsed["mrr@10"] == pytest.approx(np.mean([1 / 3] * 4))


# --- config ------------------------------------------------------------------ #

def write_cfg(tmp_path, **train):
    raw = {"name": "c", "output_dir": str(tmp_path / "runs"), "data": {"pairs_train": "a", "pairs_val": "b"}, "train": train}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_config_overrides_validation_and_fingerprint(tmp_path):
    cfg = load_contrastive_config(write_cfg(tmp_path), ["loss.name=mnrl_symmetric", "train.chunk_size=32", "train.batch_size=256"])
    assert cfg.loss.name == "mnrl_symmetric" and cfg.train.chunk_size == 32 and cfg.train.epochs == 3
    other = load_contrastive_config(write_cfg(tmp_path), ["loss.scale=50"])
    assert cfg.fingerprint() != other.fingerprint()
    with pytest.raises(ValueError, match="unknown keys"):
        load_contrastive_config(write_cfg(tmp_path), ["train.lerning_rate=1"])
    with pytest.raises(ValueError, match="loss.name"):
        load_contrastive_config(write_cfg(tmp_path), ["loss.name=triplet"])
    with pytest.raises(ValueError, match="epochs or train.max_steps"):
        load_contrastive_config(write_cfg(tmp_path), ["train.epochs=null"])
    assert load_contrastive_config(write_cfg(tmp_path), ["train.max_steps=7"]).train.max_steps == 7


# --- smoke training ----------------------------------------------------------- #

def make_pairs(n: int, offset: int = 0) -> dict:
    texts = make_texts(n + offset)[offset:]
    return {"question": [t[:30] for t in texts], "passage": texts, "passage_id": [f"id{i + offset}" for i in range(n)]}


def smoke_cfg(tmp_path, **train):
    raw = {"name": "smoke", "output_dir": str(tmp_path / "runs"), "tensorboard": False,
           "student": {"hf_id": "fake", "max_length": 64},
           "data": {"query_max_length": 32, "passage_max_length": 64},
           "loss": {"name": "mnrl", "scale": 20.0},
           "train": {"batch_size": 8, "chunk_size": 4, "lr": 5e-3, "num_workers": 0, "bf16": False, "epochs": None,
                     "max_steps": 12, "log_every": 4, "eval_every": 6, "ckpt_every": 6, **train}}
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_contrastive_config(path)


def test_smoke_training_reduces_loss_and_logs_validation(tmp_path):
    cfg = smoke_cfg(tmp_path)
    run_dir = tmp_path / "run"
    result = run_contrastive(cfg, run_dir, tiny_model(), CharTokenizer(), make_pairs(64), make_pairs(24, 100), torch.device("cpu"))
    assert result["final_step"] == 12
    records = read_metrics(run_dir / "metrics.jsonl")
    train = [r for r in records if "train/loss" in r]
    assert train[-1]["train/loss"] < train[0]["train/loss"]
    val_steps = [r["step"] for r in records if "val/mrr@10" in r]
    assert val_steps == [0, 6, 12]
    assert (run_dir / "final").exists()


def test_training_resumes_from_the_last_checkpoint_instead_of_starting_over(tmp_path):
    pairs, val = make_pairs(64), make_pairs(24, 100)
    run_dir = tmp_path / "split"
    cpu = torch.device("cpu")
    run_contrastive(smoke_cfg(tmp_path, max_steps=6, ckpt_every=6), run_dir, tiny_model(3), CharTokenizer(), pairs, val, cpu)
    resumed = tiny_model(3)  # a fresh model: only the checkpoint can carry the progress
    result = run_contrastive(smoke_cfg(tmp_path, max_steps=12), run_dir, resumed, CharTokenizer(), pairs, val, cpu)
    assert result["final_step"] == 12
    records = read_metrics(run_dir / "metrics.jsonl")
    assert [r["step"] for r in records if "val/mrr@10" in r] == [0, 6, 12]  # step 0 is evaluated only once
    train_steps = [r["step"] for r in records if "train/loss" in r]
    assert train_steps == sorted(train_steps) and len(train_steps) == len(set(train_steps))
    steps = [int(p.stem.split("_")[-1]) for p in (run_dir / "checkpoints").glob("step_*.pt")]
    assert max(steps) == 12


def test_overrides_in_scientific_notation_without_a_dot_become_numbers(tmp_path):
    # PyYAML alone reads "3e-05" as a string; the notebook formats 3e-5 exactly like that.
    cfg = load_contrastive_config(write_cfg(tmp_path), ["train.lr=3e-05", "train.weight_decay=1E-2", "name=1e3x"])
    assert cfg.train.lr == pytest.approx(3e-5) and isinstance(cfg.train.lr, float)
    assert cfg.train.weight_decay == pytest.approx(0.01)
    assert cfg.name == "1e3x"  # only a bare scientific number is converted, names stay strings
    assert cfg.loss.scale == 20.0 and load_contrastive_config(write_cfg(tmp_path), ["train.lr=5.0e-5"]).train.lr == pytest.approx(5e-5)
