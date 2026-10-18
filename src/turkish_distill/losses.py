"""Distillation losses.

The cosine loss compares a per-teacher projection of the student embedding with
the teacher vector, so teachers of different dimensions are supported. The
similarity and relative-similarity losses (Jasper & Stella, arXiv 2412.19048)
operate on in-batch similarity matrices, so they are independent of the
teacher dimension. They use the shared student embedding directly, which makes
the student geometry a consensus of all teachers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig, TeacherConfig, teacher_weight


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos(pred, target), averaged over the batch."""
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def similarity_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE between the in-batch cosine-similarity matrices of student and teacher."""
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.float(), dim=-1)
    return F.mse_loss(s @ s.T, t @ t.T)


def relative_similarity_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    margin: float,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Hinge loss on pair-of-pairs ordering.

    For sampled pairs of text pairs (a, b), if the teacher considers pair a more
    similar than pair b, the student must too, by at least `margin`. Sampling
    keeps the cost linear in `num_samples` instead of quadratic in batch pairs.
    """
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.float(), dim=-1)
    batch = s.shape[0]
    iu = torch.triu_indices(batch, batch, offset=1, device=s.device)
    sp = (s @ s.T)[iu[0], iu[1]]
    tp = (t @ t.T)[iu[0], iu[1]]
    num_pairs = sp.shape[0]
    a = torch.randint(0, num_pairs, (num_samples,), device=s.device, generator=generator)
    b = torch.randint(0, num_pairs, (num_samples,), device=s.device, generator=generator)
    teacher_prefers_a = tp[a] > tp[b]
    if not bool(teacher_prefers_a.any()):
        return sp.sum() * 0.0
    violation = F.relu(sp[b] - sp[a] + margin)
    return violation[teacher_prefers_a].mean()


def gor_loss(emb: torch.Tensor) -> torch.Tensor:
    """Spread-out regulariser: mean squared off-diagonal cosine (second moment only)."""
    x = F.normalize(emb.float(), dim=-1)
    sims = x @ x.T
    n = x.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return (sims[off] ** 2).mean()


class MultiTeacherLoss(nn.Module):
    """Weighted sum of per-teacher losses, normalised by the sum of teacher weights.

    Normalising keeps the gradient scale constant when teachers are added or
    removed, so single- and multi-teacher runs are comparable.
    """

    def __init__(self, loss_cfg: LossConfig, teachers: list[TeacherConfig], seed: int = 0):
        super().__init__()
        self.cfg = loss_cfg
        self.teachers = list(teachers)
        self._step = 0
        if sum(self.current_weights().values()) <= 0:
            raise ValueError("sum of teacher weights must be positive")
        self._seed = seed
        self._generators: dict[str, torch.Generator] = {}

    def set_step(self, step: int) -> None:
        """Tell the loss which training step it is computed for (teacher weights may ramp with the step)."""
        self._step = int(step)

    def current_weights(self) -> dict[str, float]:
        return {t.name: teacher_weight(t, self._step) for t in self.teachers}

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            g = torch.Generator(device=device)
            g.manual_seed(self._seed)
            self._generators[key] = g
        return self._generators[key]

    def forward(
        self,
        pooled: torch.Tensor,
        projections: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        total = pooled.new_zeros((), dtype=torch.float32)
        logs: dict[str, float] = {}
        weights = self.current_weights()
        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            raise ValueError(f"sum of teacher weights is not positive at step {self._step}")
        for name, weight in weights.items():
            logs[f"weight/{name}"] = weight
            if weight == 0.0:
                continue  # also leaves that teacher's head without gradient (no weight decay either)
            target = targets[name].to(pooled.device)
            parts = pooled.new_zeros((), dtype=torch.float32)
            if cfg.cosine_weight:
                c = cosine_loss(projections[name], target)
                parts = parts + cfg.cosine_weight * c
                logs[f"cos/{name}"] = float(c.detach())
            if cfg.sim_weight:
                s = similarity_loss(pooled, target)
                parts = parts + cfg.sim_weight * s
                logs[f"sim/{name}"] = float(s.detach())
            if cfg.resim_weight:
                r = relative_similarity_loss(
                    pooled,
                    target,
                    cfg.resim_margin,
                    cfg.resim_num_samples,
                    self._generator(pooled.device),
                )
                parts = parts + cfg.resim_weight * r
                logs[f"resim/{name}"] = float(r.detach())
            total = total + (weight / weight_sum) * parts
        if cfg.gor_weight:
            g = gor_loss(pooled)
            total = total + cfg.gor_weight * g
            logs["gor"] = float(g.detach())
        logs["total"] = float(total.detach())
        return total, logs
