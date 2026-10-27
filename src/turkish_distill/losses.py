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

from .config import LossConfig, TeacherConfig, teacher_loss_scale, teacher_weight


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos(pred, target), averaged over the batch."""
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def offdiag_similarities(vectors: torch.Tensor) -> torch.Tensor:
    """Cosine similarities of all distinct pairs in a batch (upper triangle, row-major order)."""
    v = F.normalize(vectors.float(), dim=-1)
    n = v.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=v.device)
    return (v @ v.T)[iu[0], iu[1]]


def zscore(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean 0 and (population) std 1; the MSE of two z-scored vectors is 2 * (1 - Pearson correlation)."""
    centred = values - values.mean()
    return centred / (centred.pow(2).mean().sqrt() + eps)


def standardized_teacher_pairs(
    teacher: torch.Tensor, ref_mean: float, ref_std: float
) -> torch.Tensor:
    """The teacher's pairwise cosines, standardised and re-scaled to a fixed reference mean and std.

    The pattern of which pairs are more similar than others is the teacher's; the offset and spread are the
    reference's, so a teacher whose vectors sit in a narrow cone (mE5) asks the student for the same absolute
    geometry as a spread-out one (BGE-M3: mean 0.29, std 0.07).
    """
    return ref_mean + ref_std * zscore(offdiag_similarities(teacher))


def similarity_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    standardize: bool = False,
    ref_mean: float = 0.30,
    ref_std: float = 0.07,
) -> torch.Tensor:
    """MSE between the in-batch cosine-similarity matrices of student and teacher.

    With `standardize` (method b) only the TEACHER side is standardised and re-scaled to (`ref_mean`, `ref_std`);
    the student's own cosines stay as they are. Standardising the student too (Pearson matching) leaves its offset
    and scale free, and a student that is free to stay in a narrow cone did (mE5: 52.85 against 58.46).
    """
    if standardize:
        return F.mse_loss(offdiag_similarities(student), standardized_teacher_pairs(teacher, ref_mean, ref_std))
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.float(), dim=-1)
    return F.mse_loss(s @ s.T, t @ t.T)


def ranking_hinge(
    student_pairs: torch.Tensor,
    teacher_pairs: torch.Tensor,
    margin: float,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pair-of-pairs hinge on similarity vectors (one entry per text pair); only the teacher's order is used."""
    num_pairs = student_pairs.shape[0]
    a = torch.randint(0, num_pairs, (num_samples,), device=student_pairs.device, generator=generator)
    b = torch.randint(0, num_pairs, (num_samples,), device=student_pairs.device, generator=generator)
    teacher_prefers_a = teacher_pairs[a] > teacher_pairs[b]
    if not bool(teacher_prefers_a.any()):
        return student_pairs.sum() * 0.0
    violation = F.relu(student_pairs[b] - student_pairs[a] + margin)
    return violation[teacher_prefers_a].mean()


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
    return ranking_hinge(
        offdiag_similarities(student), offdiag_similarities(teacher), margin, num_samples, generator
    )


_MASKED = -1e4  # logit of the diagonal (a text is not its own neighbour); exp() of it is exactly 0 in float32


def teacher_row_probs(teacher: torch.Tensor, temperature: float, ref_std: float) -> torch.Tensor:
    """Each text's distribution over the other texts of the batch under one teacher (rows sum to 1, diagonal 0).

    The teacher's in-batch cosines are z-scored over the off-diagonal entries and given the reference spread
    `ref_std` before the softmax with `temperature`: a teacher with a wide spread (8K-embed: std 0.087) and one with a
    narrow spread (BGE-M3: 0.071) then ask for the same sharpness, and the softmax itself ignores the offset.
    """
    t = F.normalize(teacher.float(), dim=-1)
    sims = t @ t.T
    n = sims.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=sims.device)
    values = sims[off]
    z = (sims - values.mean()) / (values.std(unbiased=False) + 1e-6)
    return torch.softmax((z * ref_std / temperature).masked_fill(~off, _MASKED), dim=-1)


def listwise_kl_loss(student: torch.Tensor, target_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Mean over the rows of KL(target || student): every text's neighbour distribution under the student (softmax of
    its raw in-batch cosines / temperature) must cover the target's. With a mixture of the teachers' row distributions
    as the target both teachers' neighbours stay plausible, instead of being averaged into a list neither has."""
    s = F.normalize(student.float(), dim=-1)
    n = s.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=s.device)
    log_p = torch.log_softmax((s @ s.T / temperature).masked_fill(~off, _MASKED), dim=-1)
    q = target_probs
    return (q * (torch.log(q.clamp_min(1e-12)) - log_p)).sum(dim=-1).mean()


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
        self._by_name = {t.name: t for t in self.teachers}
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
        shared = cfg.relation_mode == "shared"
        shared_gram = None  # (c): the weighted average of the teachers' z-scored Gram vectors
        mixture = None  # listwise term: the weighted mixture of the teachers' row distributions
        mixture_weight = 0.0
        for name, weight in weights.items():
            logs[f"weight/{name}"] = weight
            if weight == 0.0:
                continue  # also leaves that teacher's head without gradient (no weight decay either)
            target = targets[name].to(pooled.device)
            scale = {k: teacher_loss_scale(self._by_name[name], k) for k in ("cosine", "sim", "resim", "listwise")}
            if cfg.listwise_weight and scale["listwise"] > 0:
                share = weight * scale["listwise"]
                probs = share * teacher_row_probs(target, cfg.listwise_temperature, cfg.gram_target_std)
                mixture = probs if mixture is None else mixture + probs
                mixture_weight += share
            parts = pooled.new_zeros((), dtype=torch.float32)
            if cfg.cosine_weight and scale["cosine"] > 0:
                c = cosine_loss(projections[name], target)
                parts = parts + cfg.cosine_weight * scale["cosine"] * c
                logs[f"cos/{name}"] = float(c.detach())
            if shared:
                z = zscore(offdiag_similarities(target))
                shared_gram = (weight / weight_sum) * z if shared_gram is None else shared_gram + (weight / weight_sum) * z
            else:
                if cfg.sim_weight and scale["sim"] > 0:
                    s = similarity_loss(
                        pooled, target, cfg.gram_standardize, cfg.gram_target_mean, cfg.gram_target_std
                    )
                    parts = parts + cfg.sim_weight * scale["sim"] * s
                    logs[f"sim/{name}"] = float(s.detach())
                if cfg.resim_weight and scale["resim"] > 0:
                    r = relative_similarity_loss(
                        pooled,
                        target,
                        cfg.resim_margin,
                        cfg.resim_num_samples,
                        self._generator(pooled.device),
                    )
                    parts = parts + cfg.resim_weight * scale["resim"] * r
                    logs[f"resim/{name}"] = float(r.detach())
            total = total + (weight / weight_sum) * parts
        if shared and shared_gram is not None:
            student_pairs = offdiag_similarities(pooled)
            # averaging disagreeing teachers shrinks the spread: re-standardise, then give it the reference geometry
            target_pairs = cfg.gram_target_mean + cfg.gram_target_std * zscore(shared_gram)
            if cfg.sim_weight:
                s = F.mse_loss(student_pairs, target_pairs)
                total = total + cfg.sim_weight * s
                logs["sim/shared"] = float(s.detach())
            if cfg.resim_weight:
                r = ranking_hinge(
                    student_pairs, target_pairs, cfg.resim_margin, cfg.resim_num_samples, self._generator(pooled.device)
                )
                total = total + cfg.resim_weight * r
                logs["resim/shared"] = float(r.detach())
        if cfg.listwise_weight and mixture is not None:
            lw = listwise_kl_loss(pooled, mixture / mixture_weight, cfg.listwise_temperature)
            total = total + cfg.listwise_weight * lw
            logs["listwise"] = float(lw.detach())
        if cfg.gor_weight:
            g = gor_loss(pooled)
            total = total + cfg.gor_weight * g
            logs["gor"] = float(g.detach())
        logs["total"] = float(total.detach())
        return total, logs
