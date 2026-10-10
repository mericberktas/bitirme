"""Embedding-space diagnostics (anisotropy, rank, agreement with teachers)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    """Average off-diagonal cosine similarity. Close to 1 means a collapsed (anisotropic) space."""
    x = F.normalize(x.float(), dim=-1)
    n = x.shape[0]
    if n < 2:
        return float("nan")
    sims = x @ x.T
    total = sims.sum() - sims.diagonal().sum()
    return float(total / (n * (n - 1)))


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy) of the normalised singular values of the centred embedding matrix."""
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    total = s.sum()
    if float(total) == 0.0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def geometry_agreement(student: torch.Tensor, teacher: torch.Tensor) -> float:
    """Pearson correlation between the off-diagonal Gram entries of student and teacher.

    Independent of the embedding dimension, so it is comparable across teachers.
    """
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.float(), dim=-1)
    n = s.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=s.device)
    a = (s @ s.T)[iu[0], iu[1]]
    b = (t @ t.T)[iu[0], iu[1]]
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def teacher_cosine(projection: torch.Tensor, target: torch.Tensor) -> float:
    """Mean cosine between the student's projected output and the teacher vector."""
    p = F.normalize(projection.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    return float((p * t).sum(dim=-1).mean())
