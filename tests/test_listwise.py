"""Listwise (neighbour-distribution) term: KL from the mixture of the teachers' row distributions to the student's."""

import dataclasses

import pytest
import torch

from turkish_distill.config import ExperimentConfig, LossConfig, TeacherConfig, validate
from turkish_distill.losses import MultiTeacherLoss, listwise_kl_loss, teacher_row_probs


def vectors(n=24, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_teacher_row_probs_are_distributions_over_the_other_texts():
    p = teacher_row_probs(vectors(), temperature=0.05, ref_std=0.07)
    assert p.shape == (24, 24)
    assert torch.allclose(p.sum(dim=-1), torch.ones(24), atol=1e-5)
    assert torch.all(p.diagonal() == 0) and torch.all(p >= 0)


def test_a_lower_temperature_makes_the_neighbour_distribution_sharper():
    x = vectors(seed=1)
    soft = teacher_row_probs(x, temperature=0.2, ref_std=0.07)
    sharp = teacher_row_probs(x, temperature=0.02, ref_std=0.07)
    assert sharp.max(dim=-1).values.mean() > soft.max(dim=-1).values.mean()


def test_the_spread_of_the_teacher_does_not_change_its_row_distribution():
    # the cosines are z-scored and re-scaled, so two teachers that differ only by a monotone-affine map of their
    # cosines (here: the same vectors) give identical targets whatever their natural spread
    x = vectors(seed=2)
    a = teacher_row_probs(x, 0.05, 0.07)
    b = teacher_row_probs(x * 3.0, 0.05, 0.07)  # cosines are scale invariant: a sanity check of the normalisation
    assert torch.allclose(a, b, atol=1e-6)


def test_listwise_kl_is_non_negative_and_lower_for_a_student_that_looks_like_the_teacher():
    teacher = vectors(seed=3)
    target = teacher_row_probs(teacher, 0.05, 0.07)
    close = teacher + 0.05 * vectors(seed=4)
    far = vectors(seed=5)
    kl_close, kl_far = listwise_kl_loss(close, target, 0.05), listwise_kl_loss(far, target, 0.05)
    assert kl_close >= 0 and kl_far >= 0
    assert kl_close < kl_far


def test_listwise_kl_gives_a_finite_gradient():
    student = vectors(seed=6).requires_grad_(True)
    target = teacher_row_probs(vectors(seed=7), 0.05, 0.07)
    loss = listwise_kl_loss(student, target, 0.05)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


def make_loss(teachers, **loss_kw):
    cfg = LossConfig(**loss_kw)
    return MultiTeacherLoss(cfg, teachers)


def test_two_identical_teachers_give_the_same_listwise_term_as_one():
    pooled = vectors(seed=8)
    target = vectors(seed=9, d=12)
    one = make_loss([TeacherConfig(name="a", hf_id="a", dim=12)], listwise_weight=1.0, cosine_weight=0, sim_weight=0, resim_weight=0)
    two = make_loss(
        [TeacherConfig(name="a", hf_id="a", dim=12), TeacherConfig(name="b", hf_id="b", dim=12)],
        listwise_weight=1.0, cosine_weight=0, sim_weight=0, resim_weight=0,
    )
    _, l1 = one(pooled, {"a": pooled}, {"a": target})
    _, l2 = two(pooled, {"a": pooled, "b": pooled}, {"a": target, "b": target.clone()})
    assert l1["listwise"] == pytest.approx(l2["listwise"], rel=1e-5)


def test_the_listwise_term_is_off_by_default_and_logged_when_on():
    teachers = [TeacherConfig(name="a", hf_id="a", dim=12)]
    pooled, target = vectors(seed=10), vectors(seed=11)
    _, off_logs = make_loss(teachers)(pooled, {"a": pooled}, {"a": target})
    assert "listwise" not in off_logs
    total_on, on_logs = make_loss(teachers, listwise_weight=2.0)(pooled, {"a": pooled}, {"a": target})
    total_off, _ = make_loss(teachers)(pooled, {"a": pooled}, {"a": target})
    assert "listwise" in on_logs and on_logs["listwise"] > 0
    assert float(total_on) == pytest.approx(float(total_off) + 2.0 * on_logs["listwise"], rel=1e-5)


def test_the_mixture_target_covers_both_teachers_neighbours():
    # two teachers with unrelated neighbour structure: the mixture puts mass on both neighbour sets, so a student
    # copying teacher A pays for the neighbours of B that it ignores (and the other way round), and a student halfway
    # whose cosine matrix is the average of both teachers' (the concatenation of the two unit vectors) costs much
    # less than either copy
    a, b = vectors(seed=12), vectors(seed=13)
    mixture = 0.5 * teacher_row_probs(a, 0.05, 0.07) + 0.5 * teacher_row_probs(b, 0.05, 0.07)
    cost_a, cost_b = listwise_kl_loss(a, mixture, 0.05), listwise_kl_loss(b, mixture, 0.05)
    both = torch.cat([torch.nn.functional.normalize(a, dim=-1), torch.nn.functional.normalize(b, dim=-1)], dim=1)
    assert listwise_kl_loss(both, mixture, 0.05) < 0.6 * min(cost_a, cost_b)


def test_config_validation_of_the_listwise_settings():
    cfg = ExperimentConfig(name="x", output_dir="o", teachers=[TeacherConfig(name="a", hf_id="a", dim=4)])
    validate(cfg)
    validate(dataclasses.replace(cfg, loss=dataclasses.replace(cfg.loss, listwise_weight=5.0)))
    for bad in (dict(listwise_weight=-1.0), dict(listwise_temperature=0.0)):
        with pytest.raises(ValueError):
            validate(dataclasses.replace(cfg, loss=dataclasses.replace(cfg.loss, **bad)))
