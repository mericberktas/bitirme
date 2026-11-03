"""Weight-space edits between a finished student and the base it was distilled from.

`theta = base + gamma * (student - base)` per tensor: gamma = 1 is the student, 0 the base, below 1 pulls toward the
base (WiSE-FT style interpolation), above 1 extrapolates past the student. A *profile* says which tensors get which
gamma, selected by layer range and module group. Tensors are matched by name, so the student's and the base's
safetensors must use the same names (the base's `model.` prefix is stripped by the caller).
"""
from __future__ import annotations

import re
from typing import Optional

import torch

LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")

# name: dict(gamma, layers (inclusive range or None = every layer), groups (subset of attn/mlp/norm or None = all),
#            outside_layers (also scale embeddings and the final norm))
PROFILES = {
    "wise_late_g75": dict(gamma=0.75, layers=(16, 21), groups=None, outside_layers=False),
    "wise_mlp_late_g75": dict(gamma=0.75, layers=(16, 21), groups={"mlp"}, outside_layers=False),
    "wise_all_g90": dict(gamma=0.90, layers=None, groups=None, outside_layers=True),
    "extrap_all_g110": dict(gamma=1.10, layers=None, groups=None, outside_layers=True),
}


def module_group(rest: str) -> str:
    """'attn.Wqkv.weight' -> 'attn', 'mlp.Wi.weight' -> 'mlp', 'attn_norm.weight' / 'mlp_norm.weight' -> 'norm'."""
    head = rest.split(".")[0]
    if head.endswith("_norm"):
        return "norm"
    return head if head in ("attn", "mlp") else "other"


def gamma_for(name: str, spec: dict) -> float:
    """The gamma a profile assigns to one tensor (1.0 = leave it as the student has it)."""
    m = LAYER_RE.match(name)
    if m is None:
        return spec["gamma"] if spec["outside_layers"] else 1.0
    layer, rest = int(m.group(1)), m.group(2)
    lo, hi = spec["layers"] if spec["layers"] else (0, 10**9)
    if not lo <= layer <= hi:
        return 1.0
    if spec["groups"] is not None and module_group(rest) not in spec["groups"]:
        return 1.0
    return spec["gamma"]


def scale_delta(student: dict[str, torch.Tensor], base: dict[str, torch.Tensor], spec: dict) -> dict[str, torch.Tensor]:
    """New state dict with theta = base + gamma * (student - base) where the profile selects a tensor, else the
    student's tensor unchanged (same values, not recomputed)."""
    missing = sorted(set(student) - set(base))
    if missing:
        raise KeyError(f"{len(missing)} student tensors are not in the base, e.g. {missing[:3]}")
    out = {}
    for name, w in student.items():
        g = gamma_for(name, spec)
        if g == 1.0:
            out[name] = w.clone()
            continue
        b = base[name]
        if b.shape != w.shape:
            raise ValueError(f"{name}: shape {tuple(w.shape)} vs base {tuple(b.shape)}")
        out[name] = (b.float() + g * (w.float() - b.float())).to(w.dtype)
    return out


def relative_change(edited: dict[str, torch.Tensor], student: dict[str, torch.Tensor]) -> dict[str, float]:
    """||edited - student|| / ||student|| per tensor (0 where nothing changed)."""
    return {k: ((edited[k].float() - student[k].float()).norm() / student[k].float().norm().clamp_min(1e-12)).item()
            for k in student}


def summarise(edited, student, spec) -> str:
    """One line per profile: how many tensors were touched and the mean relative change of those."""
    rc = relative_change(edited, student)
    touched = [k for k in student if gamma_for(k, spec) != 1.0]
    mean = sum(rc[k] for k in touched) / max(len(touched), 1)
    return f"{len(touched)} of {len(student)} tensors scaled by {spec['gamma']}, mean relative change {100 * mean:.2f}%"
