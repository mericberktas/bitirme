import pytest
import torch

from turkish_distill.config import LossConfig, TeacherConfig
from turkish_distill.losses import (
    MultiTeacherLoss,
    cosine_loss,
    gor_loss,
    relative_similarity_loss,
    similarity_loss,
)


def test_cosine_loss_zero_for_identical_and_scale_invariant():
    x = torch.randn(5, 8)
    assert cosine_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert cosine_loss(3.0 * x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert cosine_loss(-x, x).item() == pytest.approx(2.0, abs=1e-5)


def test_similarity_loss_is_dimension_independent_and_zero_on_same_geometry():
    x = torch.randn(6, 8)
    # Same geometry in a different dimension: an orthogonal embedding of x.
    q, _ = torch.linalg.qr(torch.randn(20, 8))
    y = x @ q.T
    assert similarity_loss(x, y).item() == pytest.approx(0.0, abs=1e-5)
    assert similarity_loss(x, torch.randn(6, 20)).item() > 1e-3


def test_relative_similarity_loss_prefers_correct_order():
    torch.manual_seed(0)
    teacher = torch.randn(16, 8)
    good = teacher.clone()
    bad = teacher[torch.randperm(16)]  # student geometry unrelated to the teacher's
    # Without a margin, a student that reproduces the teacher's geometry has no violations.
    g = torch.Generator().manual_seed(1)
    assert relative_similarity_loss(good, teacher, 0.0, 512, g).item() == pytest.approx(0.0, abs=1e-6)
    # With a margin, near-ties are penalised slightly, but a wrong geometry is penalised far more.
    g = torch.Generator().manual_seed(1)
    l_good = relative_similarity_loss(good, teacher, 0.015, 512, g)
    g = torch.Generator().manual_seed(1)
    l_bad = relative_similarity_loss(bad, teacher, 0.015, 512, g)
    assert l_bad.item() > 10 * l_good.item()


def test_gor_loss_lower_for_spread_embeddings():
    spread = torch.linalg.qr(torch.randn(16, 16))[0]  # orthonormal rows
    collapsed = torch.ones(16, 16) + 0.01 * torch.randn(16, 16)
    assert gor_loss(spread).item() < gor_loss(collapsed).item()


def make_loss(**kw):
    teachers = [TeacherConfig("a", "x/a", 8, weight=1.0), TeacherConfig("b", "x/b", 12, weight=3.0)]
    return MultiTeacherLoss(LossConfig(resim_num_samples=64, **kw), teachers, seed=0)


def test_multi_teacher_loss_logs_and_gradients():
    torch.manual_seed(0)
    pooled = torch.randn(10, 16, requires_grad=True)
    heads = {"a": torch.nn.Linear(16, 8), "b": torch.nn.Linear(16, 12)}
    projections = {k: h(pooled) for k, h in heads.items()}
    targets = {"a": torch.randn(10, 8), "b": torch.randn(10, 12)}
    total, logs = make_loss()(pooled, projections, targets)
    assert total.requires_grad
    total.backward()
    assert pooled.grad is not None and torch.isfinite(pooled.grad).all()
    for key in ("cos/a", "cos/b", "sim/a", "sim/b", "resim/a", "resim/b", "total"):
        assert key in logs


def test_zero_weight_teacher_is_ignored():
    teachers = [TeacherConfig("a", "x/a", 8, weight=1.0), TeacherConfig("b", "x/b", 12, weight=0.0)]
    loss = MultiTeacherLoss(LossConfig(resim_num_samples=32), teachers)
    pooled = torch.randn(6, 16)
    projections = {"a": torch.randn(6, 8), "b": torch.randn(6, 12)}
    targets = {"a": torch.randn(6, 8), "b": torch.randn(6, 12)}
    _, logs = loss(pooled, projections, targets)
    assert "cos/a" in logs and "cos/b" not in logs


def test_all_zero_weights_rejected():
    with pytest.raises(ValueError):
        MultiTeacherLoss(LossConfig(), [TeacherConfig("a", "x/a", 8, weight=0.0)])


def test_loss_normalised_by_weight_sum():
    """Scaling every teacher weight by the same factor must not change the loss."""
    torch.manual_seed(1)
    pooled = torch.randn(8, 16)
    projections = {"a": torch.randn(8, 8), "b": torch.randn(8, 12)}
    targets = {"a": torch.randn(8, 8), "b": torch.randn(8, 12)}
    cfg = LossConfig(resim_num_samples=32)
    t1 = [TeacherConfig("a", "x", 8, weight=1.0), TeacherConfig("b", "x", 12, weight=2.0)]
    t2 = [TeacherConfig("a", "x", 8, weight=10.0), TeacherConfig("b", "x", 12, weight=20.0)]
    l1, _ = MultiTeacherLoss(cfg, t1, seed=3)(pooled, projections, targets)
    l2, _ = MultiTeacherLoss(cfg, t2, seed=3)(pooled, projections, targets)
    assert l1.item() == pytest.approx(l2.item(), rel=1e-5)


def test_teacher_weight_ramps_linearly_and_is_constant_without_a_start():
    from turkish_distill.config import teacher_weight

    constant = TeacherConfig("a", "x/a", 8, weight=2.0)
    assert [teacher_weight(constant, s) for s in (0, 500, 10_000)] == [2.0, 2.0, 2.0]
    ramp = TeacherConfig("b", "x/b", 8, weight=1.0, weight_start=0.0, ramp_start_step=100, ramp_end_step=300)
    assert [teacher_weight(ramp, s) for s in (0, 100, 200, 300, 5000)] == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    step = TeacherConfig("c", "x/c", 8, weight=1.0, weight_start=0.0, ramp_start_step=50, ramp_end_step=50)
    assert [teacher_weight(step, s) for s in (49, 50, 51)] == [0.0, 1.0, 1.0]  # no ramp: switched on AT step 50


def test_a_ramped_teacher_has_no_effect_until_its_weight_is_positive_and_the_loss_stays_normalised():
    torch.manual_seed(0)
    pooled = torch.randn(10, 8)
    teachers = [TeacherConfig("a", "x/a", 6), TeacherConfig("b", "x/b", 6, weight_start=0.0, ramp_start_step=0, ramp_end_step=100)]
    projections = {t.name: torch.randn(10, 6) for t in teachers}
    targets = {t.name: torch.nn.functional.normalize(torch.randn(10, 6), dim=-1) for t in teachers}
    cfg = LossConfig(resim_num_samples=32)

    ramped = MultiTeacherLoss(cfg, teachers, seed=1)
    only_a = MultiTeacherLoss(cfg, teachers[:1], seed=1)
    ramped.set_step(0)
    l0, logs0 = ramped(pooled, projections, targets)
    la, _ = only_a(pooled, {"a": projections["a"]}, {"a": targets["a"]})
    assert l0.item() == pytest.approx(la.item()) and logs0["weight/b"] == 0.0 and "cos/b" not in logs0  # step 0 == single teacher
    ramped.set_step(50)
    _, logs50 = ramped(pooled, projections, targets)
    assert logs50["weight/a"] == 1.0 and logs50["weight/b"] == pytest.approx(0.5) and "cos/b" in logs50
    # once the ramp is over the loss equals the equal-weight loss (fresh instances: the relative-similarity sampler
    # advances with every call, so compare first calls)
    done = MultiTeacherLoss(cfg, teachers, seed=1)
    done.set_step(100)
    equal = MultiTeacherLoss(cfg, [TeacherConfig("a", "x/a", 6), TeacherConfig("b", "x/b", 6)], seed=1)
    assert done(pooled, projections, targets)[0].item() == pytest.approx(equal(pooled, projections, targets)[0].item())


def test_ramp_config_is_validated():
    from turkish_distill.config import ExperimentConfig, validate

    def cfg(*teachers):
        return ExperimentConfig(name="x", output_dir="o", teachers=list(teachers))

    validate(cfg(TeacherConfig("a", "x/a", 4), TeacherConfig("b", "x/b", 4, weight_start=0.0, ramp_end_step=10)))
    with pytest.raises(ValueError, match="ramp_start_step"):
        validate(cfg(TeacherConfig("a", "x/a", 4, weight_start=0.0, ramp_start_step=20, ramp_end_step=10)))
    with pytest.raises(ValueError, match="at step 0"):  # every teacher starts at weight 0: nothing to imitate
        validate(cfg(TeacherConfig("a", "x/a", 4, weight_start=0.0, ramp_end_step=10)))
