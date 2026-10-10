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
