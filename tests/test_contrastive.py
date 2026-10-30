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


# --- anchor (distillation regulariser) -------------------------------------- #

ANCHOR = ContrastiveLossConfig(name="mnrl", scale=20.0, anchor_cosine_weight=10.0, anchor_gram_weight=200.0)


def test_anchor_is_a_frozen_independent_copy():
    model = tiny_model(0)
    anchor = ct.make_anchor(model)
    assert not any(p.requires_grad for p in anchor.parameters()) and not anchor.training
    before = [p.clone() for p in anchor.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for a, b in zip(anchor.parameters(), before):
        torch.testing.assert_close(a, b)


def test_anchor_term_is_zero_and_changes_nothing_while_the_model_equals_its_anchor():
    batch = make_batch(8)
    plain, anchored = tiny_model(2), tiny_model(2)
    ct.contrastive_step(plain, batch, ct.make_loss(ANCHOR), chunk_size=None)
    logs = ct.contrastive_step(anchored, batch, ct.make_loss(ANCHOR), chunk_size=None,
                               anchor_model=ct.make_anchor(anchored), anchor_fn=ct.make_anchor_loss(ANCHOR))
    assert logs["anchor_cos"] == pytest.approx(1.0, abs=1e-5) and logs["anchor_gram"] == pytest.approx(0.0, abs=1e-8)
    assert logs["loss"] == pytest.approx(logs["nce_loss"], rel=1e-6)
    for (name, a), (_, b) in zip(plain.named_parameters(), anchored.named_parameters()):
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("chunk", [3, 5])
def test_cached_anchored_step_matches_the_plain_anchored_step(chunk):
    batch = make_batch(10)
    reference = tiny_model(4)
    plain, cached = copy.deepcopy(tiny_model(7)), copy.deepcopy(tiny_model(7))  # drifted away from the anchor
    anchor = ct.make_anchor(reference)
    fn = ct.make_anchor_loss(ANCHOR)
    a = ct.contrastive_step(plain, batch, ct.make_loss(ANCHOR), chunk_size=None, anchor_model=anchor, anchor_fn=fn)
    b = ct.contrastive_step(cached, batch, ct.make_loss(ANCHOR), chunk_size=chunk, anchor_model=anchor, anchor_fn=fn)
    assert a["anchor_gram"] > 0
    for key in ("loss", "nce_loss", "anchor_loss", "anchor_cos"):
        assert b[key] == pytest.approx(a[key], rel=1e-4)
    for (name, x), (_, y) in zip(plain.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_anchor_term_keeps_the_embeddings_closer_to_the_anchor_during_training():
    batch = make_batch(12)

    def drift(weighted: bool) -> float:
        model = tiny_model(0)
        anchor = ct.make_anchor(model)
        cfg = ANCHOR if weighted else ContrastiveLossConfig(name="mnrl", scale=20.0)
        fn = ct.make_anchor_loss(ANCHOR)  # always measured, only added to the loss when `weighted`
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        last = {}
        for _ in range(15):
            optimizer.zero_grad()
            last = ct.contrastive_step(model, batch, ct.make_loss(cfg), chunk_size=4,
                                       anchor_model=anchor if weighted else None, anchor_fn=fn if weighted else None)
            optimizer.step()
        with torch.no_grad():
            q = model.encode(batch["q_input_ids"], batch["q_attention_mask"])
            p = model.encode(batch["p_input_ids"], batch["p_attention_mask"])
            aq = anchor.encode(batch["q_input_ids"], batch["q_attention_mask"])
            ap = anchor.encode(batch["p_input_ids"], batch["p_attention_mask"])
            return float(fn(q, p, aq, ap)[1]["anchor_cos"])

    assert drift(True) > drift(False)


def test_anchor_weights_must_not_be_negative(tmp_path):
    path = write_cfg(tmp_path)
    assert load_contrastive_config(path).loss.anchor_cosine_weight == 0.0  # off by default
    with pytest.raises(ValueError):
        load_contrastive_config(path, ["loss.anchor_gram_weight=-1"])


def test_smoke_training_with_the_anchor_logs_it_and_still_learns(tmp_path):
    cfg = smoke_cfg(tmp_path)
    cfg.loss.anchor_cosine_weight, cfg.loss.anchor_gram_weight = 10.0, 200.0
    run_dir = tmp_path / "anchored"
    run_contrastive(cfg, run_dir, tiny_model(), CharTokenizer(), make_pairs(64), make_pairs(24, 100), torch.device("cpu"))
    train = [r for r in read_metrics(run_dir / "metrics.jsonl") if "train/loss" in r]
    assert all("train/anchor_cos" in r and "train/nce_loss" in r for r in train)
    assert train[0]["train/anchor_cos"] <= 1.0 + 1e-6 and train[-1]["train/nce_loss"] < train[0]["train/nce_loss"]


# --- hard negatives --------------------------------------------------------- #

def test_info_nce_with_extra_negatives_matches_a_hand_computed_case():
    q = torch.eye(2)
    p = torch.cat([torch.eye(2), torch.tensor([[1.0, 0.0]])])  # third row: an extra negative equal to question 0
    s = 2.0
    loss, logs = ct.info_nce(q, p, scale=s, neg_mask=torch.tensor([[False], [False]]))
    e = np.exp
    expected = (-np.log(e(s) / (e(s) + 1 + e(s))) - np.log(e(s) / (1 + e(s) + 1))) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    # masking the cell (question 0, extra 0): the extra passage IS question 0's positive and must be ignored
    masked, _ = ct.info_nce(q, p, scale=s, neg_mask=torch.tensor([[True], [False]]))
    expected_masked = (-np.log(e(s) / (e(s) + 1)) - np.log(e(s) / (1 + e(s) + 1))) / 2
    assert masked.item() == pytest.approx(expected_masked, rel=1e-5)


def hn_pairs(n=12, negs=2):
    pairs = make_pairs(n)
    pool = {i: t for i, t in zip(pairs["passage_id"], pairs["passage"])}
    ids = pairs["passage_id"]
    pairs["neg_ids"] = [[ids[(i + 1 + j) % n] for j in range(negs)] for i in range(n)]
    pairs["pool"] = pool
    return pairs


def test_collator_appends_the_negatives_and_masks_cells_where_a_negative_is_the_questions_own_positive():
    pairs = hn_pairs(6, negs=1)
    pairs["neg_ids"][1] = [pairs["passage_id"][0]]  # pair 1's negative is pair 0's positive
    col = ct.PairCollator(CharTokenizer(), pairs, 32, 64, hard_negatives=1)
    batch = col([0, 1, 2])
    assert batch["p_input_ids"].shape[0] == 6 and batch["q_input_ids"].shape[0] == 3
    mask = batch["neg_mask"]
    # pair 0's negative is pair 1's positive (cell [1, 0]) and pair 1's negative is pair 0's positive (cell [0, 1])
    assert mask.shape == (3, 3) and bool(mask[1, 0]) and bool(mask[0, 1]) and int(mask.sum()) == 2


def test_collator_fills_missing_negatives_without_using_the_own_positive_and_rejects_unmined_files():
    pairs = hn_pairs(8, negs=1)
    pairs["neg_ids"][3] = []
    col = ct.PairCollator(CharTokenizer(), pairs, 32, 64, hard_negatives=2)
    filled = col._negative_ids(3)
    assert len(filled) == 2 and len(set(filled)) == 2 and pairs["passage_id"][3] not in filled
    batch = col([2, 3])
    assert batch["p_input_ids"].shape[0] == 2 + 4
    assert col([2, 3])["p_input_ids"].equal(batch["p_input_ids"])  # deterministic fill
    with pytest.raises(ValueError):
        ct.PairCollator(CharTokenizer(), make_pairs(4), 32, 64, hard_negatives=1)


@pytest.mark.parametrize("chunk", [3, 5])
def test_cached_step_with_hard_negatives_and_anchor_matches_the_plain_step(chunk):
    pairs = hn_pairs(10, negs=1)
    batch = ct.PairCollator(CharTokenizer(), pairs, 32, 64, hard_negatives=1)(list(range(10)))
    anchor = ct.make_anchor(tiny_model(9))
    fn = ct.make_anchor_loss(ANCHOR)
    plain, cached = tiny_model(5), tiny_model(5)
    a = ct.contrastive_step(plain, batch, ct.make_loss(ANCHOR), None, anchor_model=anchor, anchor_fn=fn)
    b = ct.contrastive_step(cached, batch, ct.make_loss(ANCHOR), chunk, anchor_model=anchor, anchor_fn=fn)
    assert b["loss"] == pytest.approx(a["loss"], rel=1e-4)
    for (name, x), (_, y) in zip(plain.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_load_pairs_reads_mined_negatives_and_keeps_the_pool_when_subsampling(tmp_path):
    pairs = hn_pairs(10, negs=2)
    path = tmp_path / "hn.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"question": pairs["question"][i], "passage": pairs["passage"][i],
                                "passage_id": pairs["passage_id"][i], "neg_passage_ids": pairs["neg_ids"][i]}) + "\n")
    full = ct.load_pairs(path)
    assert full["neg_ids"] == pairs["neg_ids"] and set(full["pool"]) == set(pairs["passage_id"])
    sub = ct.load_pairs(path, max_pairs=4, seed=1)
    assert len(sub["question"]) == 4 and len(sub["neg_ids"]) == 4 and len(sub["pool"]) == 10
    assert "neg_ids" not in ct.load_pairs(write_plain_pairs(tmp_path))  # a file without negatives stays plain


def write_plain_pairs(tmp_path):
    path = tmp_path / "plain.jsonl"
    pairs = make_pairs(4)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"question": pairs["question"][i], "passage": pairs["passage"][i],
                                "passage_id": pairs["passage_id"][i]}) + "\n")
    return path


def test_smoke_training_with_hard_negatives_and_the_anchor(tmp_path):
    cfg = smoke_cfg(tmp_path)
    cfg.data.hard_negatives = 1
    cfg.loss.anchor_cosine_weight, cfg.loss.anchor_gram_weight = 10.0, 200.0
    run_dir = tmp_path / "hn"
    result = run_contrastive(cfg, run_dir, tiny_model(), CharTokenizer(), hn_pairs(64, negs=2), make_pairs(24, 100),
                             torch.device("cpu"))
    assert result["final_step"] == 12
    train = [r for r in read_metrics(run_dir / "metrics.jsonl") if "train/loss" in r]
    assert train[-1]["train/nce_loss"] < train[0]["train/nce_loss"]


def test_hard_negative_count_must_not_be_negative(tmp_path):
    with pytest.raises(ValueError):
        load_contrastive_config(write_cfg(tmp_path), ["data.hard_negatives=-1"])


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


def test_describe_device_says_plainly_when_there_is_no_gpu():
    from turkish_distill.utils import describe_device

    assert "NO GPU" in describe_device(torch.device("cpu"))
