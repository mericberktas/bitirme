"""Gradient surgery between the teachers (PCGrad-style), applied at the pooled embedding of a batch.

Each teacher's loss gives the shared encoder a gradient. Where two of them point against each other (negative inner
product) PCGrad (Yu et al., 2020) removes from each gradient its component along the other one before they are
summed; Road MOPD and AE-KD use the same idea for teachers of different domains. Our measurements (grad_conflict.py)
show that the Gram terms of two teachers are exactly opposed at the end of a distillation run (cosine -1.0) and the
vector terms only mildly so (-0.28), so what this removes is mostly the part of the objective in which the teachers
cancel each other.
"""

from __future__ import annotations

from typing import Optional

import torch


def pcgrad_combine(
    grads: list[torch.Tensor], generator: Optional[torch.Generator] = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of the gradients after PCGrad: every gradient, taken in turn, loses its component along every other
    gradient (visited in random order) whose inner product with it is negative.

    The other gradients are the original ones, not already projected. Returns the combined gradient (shape of the
    inputs) and statistics: how many (gradient, other) pairs conflicted and the smallest pairwise cosine.
    """
    if not grads:
        raise ValueError("need at least one gradient")
    shape = grads[0].shape
    flat = [g.reshape(-1).float() for g in grads]
    if any(f.shape != flat[0].shape for f in flat):
        raise ValueError("all gradients must have the same shape")
    m = len(flat)
    norms = [float(f.norm()) for f in flat]
    conflicts = 0
    min_cos = 1.0
    for i in range(m):
        for j in range(i + 1, m):
            if norms[i] > 0 and norms[j] > 0:
                min_cos = min(min_cos, float((flat[i] * flat[j]).sum()) / (norms[i] * norms[j]))
    projected = []
    for i in range(m):
        g = flat[i].clone()
        others = [j for j in range(m) if j != i]
        if len(others) > 1:
            order = torch.randperm(len(others), generator=generator).tolist()
            others = [others[k] for k in order]
        for j in others:
            if norms[j] == 0.0:
                continue
            dot = float((g * flat[j]).sum())
            if dot < 0.0:
                g = g - dot / (norms[j] ** 2) * flat[j]
                conflicts += 1
        projected.append(g)
    combined = torch.stack(projected).sum(dim=0).reshape(shape)
    return combined, {"conflicts": float(conflicts), "min_cos": min_cos if m > 1 else 1.0}
