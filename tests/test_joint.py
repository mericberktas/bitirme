"""Contrastive loss trained together with the distillation (`joint` config section)."""

import copy
import hashlib
import json

import pytest
import torch
import yaml

from conftest import CharTokenizer, make_texts
from test_answers import answered_pairs
from test_contrastive import make_pairs, tiny_model
from test_train_smoke import build
from turkish_distill import contrastive as ct
from turkish_distill.config import JointContrastiveConfig, load_config
from turkish_distill.joint import JointContrastive, joint_loss_config
from turkish_distill.logging_utils import read_metrics
from turkish_distill.train import run_training


def joint_cfg(**kw) -> JointContrastiveConfig:
    base = dict(pairs="pairs.jsonl", weight=0.3, batch_size=4, every=2, start_step=1,
                query_max_length=32, passage_max_length=64, answer_max_length=24)
    base.update(kw)
    return JointContrastiveConfig(**base)


def test_schedule_and_batches_are_deterministic_and_never_repeat_a_passage():
    pairs = make_pairs(12)
    pairs["passage_id"][5] = pairs["passage_id"][4]  # two questions on one passage
    joint = JointContrastive(joint_cfg(), CharTokenizer(), pairs, seed=3)
    assert [s for s in range(8) if joint.applies(s)] == [1, 3, 5, 7]
    assert joint.pair_steps(8) == 4 and joint.pair_steps(1) == 0 and joint.pair_steps(2) == 1
    seen = [joint.batch_indices(s) for s in range(1, 40, 2)]  # several epochs
    for indices in seen:
        assert len({pairs["passage_id"][i] for i in indices}) == len(indices) == 4
    again = JointContrastive(joint_cfg(), CharTokenizer(), pairs, seed=3)
    assert [again.batch_indices(s) for s in reversed(range(1, 40, 2))] == list(reversed(seen))  # order-free


def test_step_adds_weight_times_the_contrastive_gradient_to_what_is_already_there():
    pairs = make_pairs(10)
    joint = JointContrastive(joint_cfg(batch_size=6), CharTokenizer(), pairs, seed=0, chunk_size=4)
    a, b = tiny_model(2), tiny_model(2)
    for p in a.parameters():
        p.grad = torch.ones_like(p)  # stands in for the distillation gradient of the same step
    logs = joint.step(a, 1, torch.device("cpu"))
    batch = joint.collator(joint.batch_indices(1))
    ct.contrastive_step(b, batch, ct.make_loss(joint_loss_config(joint.cfg)), None)
    for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(x.grad, 1 + 0.3 * y.grad, rtol=1e-4, atol=1e-6, msg=name)
    assert logs["loss"] == pytest.approx(0.3 * logs["nce"], rel=1e-5)


def test_step_with_answers_includes_the_weighted_answer_term():
    pairs = answered_pairs(10, missing=(2,))
    joint = JointContrastive(joint_cfg(batch_size=6, answer_weight_mode="words", answer_ref_words=4), CharTokenizer(),
                             pairs, seed=0, chunk_size=3)
    logs = joint.step(tiny_model(1), 1, torch.device("cpu"))
    assert "ans_loss" in logs and logs["loss"] == pytest.approx(0.3 * (logs["nce"] + logs["ans_loss"]), rel=1e-5)
    with pytest.raises(ValueError, match="answer"):
        JointContrastive(joint_cfg(answer_weight_mode="fixed"), CharTokenizer(), make_pairs(10), seed=0)


def test_too_few_distinct_passages_fail_early():
    pairs = make_pairs(6)
    pairs["passage_id"] = ["same"] * 6
    with pytest.raises(ValueError, match="distinct passages"):
        JointContrastive(joint_cfg(), CharTokenizer(), pairs, seed=0)


def test_training_with_the_joint_term_logs_it_and_still_distils(experiment, tmp_path):
    experiment.joint = joint_cfg(start_step=0, every=2, answer_weight_mode="chars")
    experiment.train.chunk_size = 4
    texts = make_texts(120)
    model, caches = build(experiment, texts)
    run_dir = tmp_path / "run"
    result = run_training(experiment, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"),
                          joint_pairs=answered_pairs(24, offset=200))
    assert result["final_step"] == 6
    records = read_metrics(run_dir / "metrics.jsonl")
    ctr = [r for r in records if "train/ctr_loss" in r]
    assert [r["step"] for r in ctr] == [2, 4, 6]  # pair batches at steps 0, 2, 4: one per logging window
    assert all("train/ctr_nce" in r and "train/ctr_ans_loss" in r and "train/ctr_acc" in r for r in ctr)
    assert all("train/total" in r for r in records if r["step"] in (2, 4, 6) and "train/lr" in r)
    with pytest.raises(ValueError, match="joint"):
        run_training(experiment, tmp_path / "other", model, CharTokenizer(), texts, caches, torch.device("cpu"))


def test_joint_term_changes_the_trained_weights_and_weight_zero_does_not(experiment, tmp_path):
    texts = make_texts(120)
    pairs = make_pairs(24, offset=200)

    def train(joint, name):
        exp = copy.deepcopy(experiment)
        if joint is not None:
            exp.joint = joint
        model, caches = build(exp, texts)
        run_training(exp, tmp_path / name, model, CharTokenizer(), texts, caches, torch.device("cpu"),
                     joint_pairs=pairs if joint is not None else None)
        return torch.cat([p.detach().flatten() for p in model.backbone.parameters()])

    plain = train(None, "plain")
    zero = train(joint_cfg(weight=0.0, start_step=0), "zero")
    on = train(joint_cfg(weight=1.0, start_step=0), "on")
    torch.testing.assert_close(zero, plain)
    assert not torch.allclose(on, plain)


def test_config_section_validation_and_fingerprint(tmp_path):
    raw = {"name": "x", "output_dir": str(tmp_path), "teachers": [{"name": "t", "hf_id": "h", "dim": 4}]}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.joint.pairs is None  # off by default
    without = cfg.to_dict()
    without.pop("joint")
    blob = json.dumps(without, sort_keys=True, ensure_ascii=False)
    assert cfg.fingerprint() == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]  # unchanged while off
    on = load_config(path, ["joint.pairs=p.jsonl", "joint.weight=0.1"])
    assert on.joint.weight == 0.1 and on.fingerprint() != cfg.fingerprint()
    for bad in (["joint.every=0"], ["joint.weight=-1"], ["joint.loss=cosent"], ["joint.answer_weight_mode=len"],
                ["joint.answer_weight_min=0.6", "joint.answer_weight_max=0.5"]):
        with pytest.raises(ValueError):
            load_config(path, ["joint.pairs=p.jsonl"] + bad)
    load_config(path, ["joint.every=0"])  # not validated while off
