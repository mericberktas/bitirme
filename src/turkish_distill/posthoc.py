"""Post-hoc geometric edits of a finished embedding model, folded into one extra linear layer.

No training: statistics of the model's own embeddings on (training) texts give a linear map y = x W^T + b that is
appended as an identity-activation Dense module (optionally followed by Normalize). Variants:

  center  subtract the mean embedding
  abtt    centre, then project out the top-k principal directions ("all but the top", Mu and Viswanath 2018)
  zca     centre, then ZCA-whiten with eigenvalue shrinkage (W = V diag((lambda + eps * mean(lambda))^-1/2) V^T)
"""
from __future__ import annotations

import numpy as np


def fit_stats(emb: np.ndarray) -> dict:
    """Mean and the eigendecomposition of the covariance of `emb` (N, D), float64, eigenvalues descending."""
    x = np.asarray(emb, dtype=np.float64)
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / max(len(x) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(-vals)
    return {"mean": mean, "eigvals": np.clip(vals[order], 0.0, None), "eigvecs": vecs[:, order]}


def linear_map(kind: str, stats: dict, k: int = 5, eps: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(W, b) with y = x @ W.T + b, for kind in {'identity', 'center', 'abtt', 'zca'}."""
    mean, vals, vecs = stats["mean"], stats["eigvals"], stats["eigvecs"]
    d = len(mean)
    if kind == "identity":
        return np.eye(d), np.zeros(d)
    if kind == "center":
        w = np.eye(d)
    elif kind == "abtt":
        top = vecs[:, :k]
        w = np.eye(d) - top @ top.T
    elif kind == "zca":
        scale = 1.0 / np.sqrt(vals + eps * vals.mean())
        w = (vecs * scale) @ vecs.T
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return w, -(w @ mean)


def apply_map(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ w.T + b


def mean_pairwise_cosine(x: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """Average cosine over distinct pairs (a measure of anisotropy; a raw encoder is near 0.93)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) > max_n:
        x = x[np.random.default_rng(seed).choice(len(x), max_n, replace=False)]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n = len(x)
    total = (x.sum(0) ** 2).sum() - n          # sum over ordered pairs i != j of x_i . x_j
    return float(total / (n * (n - 1)))


def append_linear_head(st_model, w: np.ndarray, b: np.ndarray, normalize: bool):
    """New SentenceTransformer = the model's modules + Dense(identity activation) holding (w, b) [+ Normalize]."""
    import torch
    from sentence_transformers import SentenceTransformer, models

    d = w.shape[0]
    dense = models.Dense(d, d, bias=True, activation_function=torch.nn.Identity())
    with torch.no_grad():
        dense.linear.weight.copy_(torch.tensor(w, dtype=dense.linear.weight.dtype))
        dense.linear.bias.copy_(torch.tensor(b, dtype=dense.linear.bias.dtype))
    modules = [m for m in st_model] + [dense] + ([models.Normalize()] if normalize else [])
    return SentenceTransformer(modules=modules)
