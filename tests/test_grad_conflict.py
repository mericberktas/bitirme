"""Gradient-conflict probe: term specs, the metrics, and that the probe sums to the real gradient without side effects."""

import copy
import dataclasses

import pytest
import torch

from conftest import CharTokenizer, TinyBackbone, make_texts
from test_train_smoke import build
from turkish_distill.config import LossConfig, TeacherConfig, load_config, validate
from turkish_distill.grad_conflict import (
    average_metrics,
    build_term_specs,
    conflict_metrics,
    format_report,
    probe_gradients,
    select_params,
)
from turkish_distill.logging_utils import read_metrics
from turkish_distill.losses import MultiTeacherLoss
from turkish_distill.models import StudentEncoder
from turkish_distill.train import distill_step, run_training


def tiny_setup(teachers, batch=8, seed=0):
    torch.manual_seed(seed)
    model = StudentEncoder(TinyBackbone(), 16, {t.name: t.dim for t in teachers})
    model.train()
    ids = torch.randint(1, 255, (batch, 12))
    mask = torch.ones(batch, 12, dtype=torch.long)
    targets = {t.name: torch.randn(batch, t.dim) for t in teachers}
    return model, ids, mask, targets


def true_gradient(model, loss_cfg, teachers, ids, mask, targets, chunk_size=None):
    """Parameter gradient of the full training loss, flattened over the backbone."""
    for p in model.parameters():
        p.grad = None
    criterion = MultiTeacherLoss(loss_cfg, teachers, seed=0)
    distill_step(model, criterion, ids, mask, targets, chunk_size)
    flat = torch.cat([p.grad.reshape(-1) for p in model.backbone.parameters() if p.grad is not None])
    for p in model.parameters():
        p.grad = None
    return flat


def test_conflict_metrics_known_geometry():
    vectors = {"a.cosine": torch.tensor([1.0, 0.0]), "a.sim": torch.tensor([0.0, 2.0]), "b.cosine": torch.tensor([-3.0, 0.0])}
    groups = {"a.cosine": "a", "a.sim": "a", "b.cosine": "b"}
    m = conflict_metrics(vectors, groups, "par")
    assert m["par_cos/a.cosine~a.sim"] == pytest.approx(0.0)
    assert m["par_cos/a.cosine~b.cosine"] == pytest.approx(-1.0)
    assert m["par_norm/a.sim"] == pytest.approx(2.0)
    total = torch.tensor([-2.0, 2.0])
    assert m["par_norm/total"] == pytest.approx(float(total.norm()))
    assert m["par_cos_total/a.sim"] == pytest.approx(float(2.0 * 2.0 / (2.0 * total.norm())))
    # group a = (1, 2), group b = (-3, 0): cosine = -3 / (sqrt(5) * 3)
    assert m["par_cos_group/a~b"] == pytest.approx(-1.0 / 5**0.5)


def test_conflict_metrics_zero_vector_gives_zero_cosine():
    m = conflict_metrics({"x": torch.zeros(3), "y": torch.ones(3)}, {"x": "x", "y": "y"}, "emb")
    assert m["emb_cos/x~y"] == 0.0


def test_term_specs_per_teacher_and_options(teachers):
    specs = build_term_specs(LossConfig(), teachers)
    assert [s.label for s in specs] == [f"{t}.{k}" for k in ("cosine", "sim", "resim") for t in ("t_small", "t_big")]
    assert {s.group for s in specs} == {"t_small", "t_big"}

    off = [dataclasses.replace(t, loss_weights={"resim": 0.0}) if t.name == "t_big" else t for t in teachers]
    labels = [s.label for s in build_term_specs(LossConfig(), off)]
    assert "t_big.resim" not in labels and "t_small.resim" in labels
    assert [s.label for s in build_term_specs(LossConfig(resim_weight=0.0), teachers)] == [
        f"{t}.{k}" for k in ("cosine", "sim") for t in ("t_small", "t_big")
    ]
    assert build_term_specs(LossConfig(listwise_weight=1.0), teachers)[-1].label == "listwise"
    shared = [s.label for s in build_term_specs(LossConfig(relation_mode="shared"), teachers)]
    assert shared == ["t_small.cosine", "t_big.cosine", "relation"]


def test_term_specs_skip_teachers_without_weight(teachers):
    ramping = [dataclasses.replace(teachers[0], weight_start=0.0, ramp_start_step=0, ramp_end_step=10), teachers[1]]
    assert all(s.group == "t_big" for s in build_term_specs(LossConfig(), ramping, step=0))
    assert any(s.group == "t_small" for s in build_term_specs(LossConfig(), ramping, step=10))


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_term_gradients_add_up_to_the_training_gradient(teachers, chunk_size):
    loss_cfg = LossConfig(resim_weight=0.0)  # the ranking hinge draws random pairs, so it cannot be compared exactly
    model, ids, mask, targets = tiny_setup(teachers)
    metrics = probe_gradients(model, loss_cfg, teachers, ids, mask, targets, chunk_size=chunk_size)
    assert all(p.grad is None for p in model.parameters())  # the probe leaves .grad alone
    full = true_gradient(model, loss_cfg, teachers, ids, mask, targets, chunk_size=chunk_size)
    assert metrics["par_norm/total"] == pytest.approx(float(full.norm()), rel=1e-4)
    for label in ("t_small.cosine", "t_big.sim"):
        assert metrics[f"par_norm/{label}"] > 0


def test_chunked_probe_matches_the_unchunked_one(teachers):
    loss_cfg = LossConfig(resim_weight=0.0)
    model, ids, mask, targets = tiny_setup(teachers)
    whole = probe_gradients(model, loss_cfg, teachers, ids, mask, targets)
    parts = probe_gradients(model, loss_cfg, teachers, ids, mask, targets, chunk_size=3)
    assert whole.keys() == parts.keys()
    for key in whole:
        assert parts[key] == pytest.approx(whole[key], rel=1e-4, abs=1e-6)


def test_single_term_gives_nothing_to_compare(teachers):
    model, ids, mask, targets = tiny_setup(teachers[:1])
    loss_cfg = LossConfig(sim_weight=0.0, resim_weight=0.0)
    assert probe_gradients(model, loss_cfg, teachers[:1], ids, mask, targets) == {}


def test_levels_and_param_selection(teachers):
    model, ids, mask, targets = tiny_setup(teachers)
    emb_only = probe_gradients(model, LossConfig(), teachers, ids, mask, targets, levels=("emb",))
    assert emb_only and all(k.startswith("emb_") for k in emb_only)
    with pytest.raises(ValueError):
        probe_gradients(model, LossConfig(), teachers, ids, mask, targets, levels=("nope",))
    with pytest.raises(ValueError, match="last:N"):
        select_params(model, "top3")
    with pytest.raises(ValueError, match="layers"):
        select_params(model, "last:2")  # the tiny backbone has no numbered layers
    assert sum(p.numel() for p in select_params(model, "all")) == sum(p.numel() for p in model.backbone.parameters())


def test_shared_relation_mode_probes_the_relation_term(teachers):
    loss_cfg = LossConfig(relation_mode="shared", resim_weight=0.0)
    model, ids, mask, targets = tiny_setup(teachers)
    metrics = probe_gradients(model, loss_cfg, teachers, ids, mask, targets)
    assert "par_cos/t_small.cosine~relation" in metrics
    full = true_gradient(model, loss_cfg, teachers, ids, mask, targets)
    assert metrics["par_norm/total"] == pytest.approx(float(full.norm()), rel=1e-4)


def test_average_and_report(teachers):
    model, ids, mask, targets = tiny_setup(teachers)
    one = probe_gradients(model, LossConfig(resim_weight=0.0), teachers, ids, mask, targets)
    mean = average_metrics([one, one])
    assert mean == pytest.approx(one)
    report = format_report(mean, "par")
    assert "t_small.cosine~t_big.sim" in report and "cosine between teachers" in report


def test_probe_during_training_logs_and_does_not_change_the_run(experiment, tmp_path):
    texts = make_texts(120)
    probed = copy.deepcopy(experiment)
    probed.train.grad_probe_every = 3
    plain_dir, probed_dir = tmp_path / "plain", tmp_path / "probed"
    for cfg, run_dir in ((experiment, plain_dir), (probed, probed_dir)):
        model, caches = build(cfg, texts)
        run_training(cfg, run_dir, model, CharTokenizer(), texts, caches, torch.device("cpu"))
    plain = [r["train/total"] for r in read_metrics(plain_dir / "metrics.jsonl") if "train/total" in r]
    with_probe = [r["train/total"] for r in read_metrics(probed_dir / "metrics.jsonl") if "train/total" in r]
    assert with_probe == pytest.approx(plain)  # probing does not move the RNG streams or the gradients
    logged = [r for r in read_metrics(probed_dir / "metrics.jsonl") if any(k.startswith("gradprobe/") for k in r)]
    assert [r["step"] for r in logged] == [0, 3]
    assert "gradprobe/par_cos/t_small.cosine~t_big.cosine" in logged[0]
    assert not any(k.startswith("gradprobe/") for r in read_metrics(plain_dir / "metrics.jsonl") for k in r)


def test_config_validation_of_the_probe_options(experiment):
    validate(experiment)
    for bad in ({"grad_probe_every": -1}, {"grad_probe_params": "top3"}, {"grad_probe_params": "last:0"}):
        broken = copy.deepcopy(experiment)
        for k, v in bad.items():
            setattr(broken.train, k, v)
        with pytest.raises(ValueError):
            validate(broken)
    ok = copy.deepcopy(experiment)
    ok.train.grad_probe_every, ok.train.grad_probe_params = 200, "last:4"
    validate(ok)
