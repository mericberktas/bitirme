"""Per-teacher loss multipliers (`teachers[].loss_weights`): cosine / sim / resim / listwise."""

import dataclasses

import pytest
import torch

from turkish_distill.config import ExperimentConfig, LossConfig, TeacherConfig, load_config, teacher_loss_scale, validate
from turkish_distill.losses import MultiTeacherLoss


def make_teachers(a_scale=None, b_scale=None, w=(1.0, 1.0)):
    return [
        TeacherConfig(name="a", hf_id="a", dim=10, weight=w[0], loss_weights=dict(a_scale or {})),
        TeacherConfig(name="b", hf_id="b", dim=8, weight=w[1], loss_weights=dict(b_scale or {})),
    ]


def inputs(seed=0, n=24):
    g = torch.Generator().manual_seed(seed)
    pooled = torch.randn(n, 12, generator=g).requires_grad_(True)
    proj = {"a": torch.randn(n, 10, generator=g).requires_grad_(True), "b": torch.randn(n, 8, generator=g).requires_grad_(True)}
    targets = {"a": torch.randn(n, 10, generator=g), "b": torch.randn(n, 8, generator=g)}
    return pooled, proj, targets


def run(teachers, **loss_kw):
    pooled, proj, targets = inputs()
    loss = MultiTeacherLoss(LossConfig(**loss_kw), teachers, seed=3)
    total, logs = loss(pooled, proj, targets)
    return total, logs, pooled, proj


def test_default_scales_are_one_and_change_nothing():
    plain, _, _, _ = run(make_teachers())
    explicit, _, _, _ = run(make_teachers({"cosine": 1.0, "sim": 1.0, "resim": 1.0}, {"cosine": 1, "sim": 1, "resim": 1}))
    assert float(plain) == pytest.approx(float(explicit), rel=1e-6)
    assert teacher_loss_scale(make_teachers()[0], "resim") == 1.0


def test_a_scale_multiplies_exactly_that_teachers_term():
    base_total, base_logs, _, _ = run(make_teachers(), listwise_weight=0.0)
    total, logs, _, _ = run(make_teachers({"resim": 3.0}), listwise_weight=0.0)
    cfg = LossConfig()
    # teacher a has weight 1 of 2: its resim term enters with weight/weight_sum = 0.5
    expected = float(base_total) + 0.5 * cfg.resim_weight * (3.0 - 1.0) * base_logs["resim/a"]
    assert float(total) == pytest.approx(expected, rel=1e-5)
    assert logs["resim/a"] == pytest.approx(base_logs["resim/a"], rel=1e-6)  # the logged value is the unscaled term


def test_zero_switches_a_term_off_for_one_teacher_only():
    _, logs, _, _ = run(make_teachers({"resim": 0.0}))
    assert "resim/a" not in logs and "resim/b" in logs
    assert "sim/a" in logs and "cos/a" in logs
    _, logs2, _, _ = run(make_teachers({"cosine": 0.0, "sim": 0.0, "resim": 0.0}))
    assert not any(k.endswith("/a") and k.split("/")[0] in ("cos", "sim", "resim") for k in logs2)


def test_a_teacher_without_a_vector_term_gets_no_gradient_in_its_head_output():
    pooled, proj, targets = inputs()
    loss = MultiTeacherLoss(LossConfig(), make_teachers({"cosine": 0.0}), seed=3)
    total, _ = loss(pooled, proj, targets)
    total.backward()
    assert proj["a"].grad is None or float(proj["a"].grad.abs().sum()) == 0.0
    assert float(proj["b"].grad.abs().sum()) > 0.0


def test_listwise_scale_removes_a_teacher_from_the_mixture():
    kw = dict(listwise_weight=1.0, cosine_weight=0.0, sim_weight=0.0, resim_weight=0.0)
    only_b = MultiTeacherLoss(LossConfig(**kw), [TeacherConfig(name="b", hf_id="b", dim=8)], seed=3)
    pooled, proj, targets = inputs()
    _, single = only_b(pooled, {"b": proj["b"]}, {"b": targets["b"]})
    _, mixed = MultiTeacherLoss(LossConfig(**kw), make_teachers({"listwise": 0.0}), seed=3)(pooled, proj, targets)
    assert mixed["listwise"] == pytest.approx(single["listwise"], rel=1e-5)
    _, full = MultiTeacherLoss(LossConfig(**kw), make_teachers(), seed=3)(pooled, proj, targets)
    assert abs(full["listwise"] - single["listwise"]) > 1e-3   # the second teacher does change the mixture


def test_validation_of_loss_weights():
    def cfg(teachers, **loss_kw):
        return ExperimentConfig(name="x", output_dir="o", teachers=teachers, loss=LossConfig(**loss_kw))

    validate(cfg(make_teachers({"resim": 0.0})))
    with pytest.raises(ValueError):
        validate(cfg(make_teachers({"nope": 1.0})))
    with pytest.raises(ValueError):
        validate(cfg(make_teachers({"resim": -1.0})))
    validate(cfg(make_teachers({"cosine": 0.0}), relation_mode="shared"))            # cosine is per teacher even when shared
    with pytest.raises(ValueError):
        validate(cfg(make_teachers({"resim": 0.0}), relation_mode="shared"))        # one shared relation target


def test_loss_weights_can_be_set_from_the_command_line():
    config = load_config(
        "configs/stage1_two_bge_m3_modernbert_tr_embed.yaml",
        ["teachers.0.loss_weights.resim=0", "teachers.1.loss_weights.cosine=2"],
    )
    assert config.teachers[0].loss_weights == {"resim": 0}
    assert config.teachers[1].loss_weights == {"cosine": 2}
    assert dataclasses.asdict(config)["teachers"][0]["loss_weights"] == {"resim": 0}
