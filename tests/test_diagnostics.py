import pytest
import torch

from turkish_distill.diagnostics import (
    effective_rank,
    geometry_agreement,
    mean_pairwise_cosine,
    teacher_cosine,
)


def test_mean_pairwise_cosine_extremes():
    same = torch.ones(10, 4)
    assert mean_pairwise_cosine(same) == pytest.approx(1.0, abs=1e-5)
    torch.manual_seed(0)
    rand = torch.randn(400, 64)
    assert abs(mean_pairwise_cosine(rand)) < 0.05


def test_effective_rank_orders_collapsed_below_spread():
    torch.manual_seed(0)
    spread = torch.randn(200, 32)
    collapsed = torch.randn(200, 1) @ torch.randn(1, 32) + 0.01 * torch.randn(200, 32)
    assert effective_rank(spread) > 20
    assert effective_rank(collapsed) < 5


def test_geometry_agreement_perfect_and_dimension_free():
    torch.manual_seed(0)
    x = torch.randn(20, 8)
    q, _ = torch.linalg.qr(torch.randn(30, 8))
    assert geometry_agreement(x, x @ q.T) == pytest.approx(1.0, abs=1e-4)
    assert abs(geometry_agreement(x, torch.randn(20, 30))) < 0.5


def test_teacher_cosine():
    x = torch.randn(7, 5)
    assert teacher_cosine(2 * x, x) == pytest.approx(1.0, abs=1e-5)
