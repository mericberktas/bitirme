"""Target transformations applied on top of the cached teacher vectors.

* `CenteredCache` (method a): subtract the corpus mean of a teacher's vectors and re-normalise.
* `PCACache`: keep only the top principal components of a teacher's vectors (what a student of that size can represent).
* `fuse_teachers` (method d): reduce every teacher to a common number of PCA components, rotate them onto one
  reference teacher with an orthogonal Procrustes fit and average the aligned unit vectors into a single target.

Both only read from the caches (which are L2-normalised float16 shards); the caches themselves never change, so
every teacher is encoded once and re-used by any number of target recipes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

CHUNK = 8192


def _unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def compute_mean(cache, chunk: int = CHUNK) -> np.ndarray:
    """Mean of all cached vectors of a teacher (float64 accumulation, chunked)."""
    total = np.zeros(cache.dim, dtype=np.float64)
    n = len(cache)
    for lo in range(0, n, chunk):
        total += cache.get(np.arange(lo, min(lo + chunk, n))).astype(np.float64).sum(axis=0)
    return (total / n).astype(np.float32)


class CenteredCache:
    """A teacher cache whose vectors are centred on the corpus mean and re-normalised (method a)."""

    def __init__(self, cache, mean: Optional[np.ndarray] = None):
        self.cache = cache
        self.mean = compute_mean(cache) if mean is None else np.asarray(mean, dtype=np.float32)
        self.dim = cache.dim
        self.n_texts = cache.n_texts

    def __len__(self) -> int:
        return self.n_texts

    def get(self, indices) -> np.ndarray:
        return _unit(self.cache.get(indices) - self.mean)


class PCACache:
    """A teacher cache whose vectors are replaced by their projection onto the top `n_components` principal
    components, mapped back to the original space and re-normalised (capacity-aware target).

    The PCA is fitted on a random subset of the corpus; the target keeps the teacher's dimension, mean (offset) and
    cosine geometry, but loses the variance that a student with `n_components` free directions cannot represent.
    """

    def __init__(self, cache, n_components: int, fit_samples: int = 20000, seed: int = 0):
        n = len(cache)
        rng = np.random.default_rng(seed)
        fit_idx = np.sort(rng.choice(n, size=min(fit_samples, n), replace=False))
        x = cache.get(fit_idx).astype(np.float64)
        self.mean = x.mean(axis=0).astype(np.float32)
        xc = x - self.mean
        values, vectors = np.linalg.eigh(xc.T @ xc / len(xc))  # ascending
        order = np.argsort(values)[::-1][:n_components]
        self.kept_variance = float(values[order].sum() / max(values.sum(), 1e-12))
        self.basis = vectors[:, order].astype(np.float32)  # (dim, n_components)
        self.cache = cache
        self.n_components = n_components
        self.dim = cache.dim
        self.n_texts = cache.n_texts

    def __len__(self) -> int:
        return self.n_texts

    def get(self, indices) -> np.ndarray:
        x = self.cache.get(indices)
        centred = x - self.mean
        return _unit(self.mean + (centred @ self.basis) @ self.basis.T)


class ArrayCache:
    """An in-memory array with the cache interface (used for fused targets)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.n_texts, self.dim = vectors.shape

    def __len__(self) -> int:
        return self.n_texts

    def get(self, indices) -> np.ndarray:
        return self.vectors[np.asarray(indices, dtype=np.int64)].astype(np.float32)


def _pca_basis(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean, the top `dim` principal directions (d x dim) and the share of variance they keep."""
    mean = x.mean(axis=0)
    xc = (x - mean).astype(np.float64)
    cov = xc.T @ xc / len(xc)
    values, vectors = np.linalg.eigh(cov)  # ascending
    order = np.argsort(values)[::-1][:dim]
    kept = float(values[order].sum() / max(values.sum(), 1e-12))
    return mean.astype(np.float32), vectors[:, order].astype(np.float32), kept


def fuse_teachers(
    caches: dict,
    weights: dict[str, float],
    reference: str,
    dim: int,
    fit_samples: int,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Fused target vectors (float16, n x dim) and a report (variance kept, alignment cosine per teacher).

    1. For every teacher a PCA to `dim` components is fitted on a random subset of the corpus (this also centres
       the teacher, so each space starts with mean 0).
    2. Every non-reference teacher is rotated onto the reference's reduced space by the orthogonal Procrustes
       solution on the same texts (the texts are paired: every teacher encoded the same corpus).
    3. The aligned vectors are L2-normalised and averaged with the teacher weights, then L2-normalised again.
    """
    names = list(caches)
    if reference not in caches:
        raise ValueError(f"reference teacher {reference!r} is not among {names}")
    n = len(caches[reference])
    if any(len(c) != n for c in caches.values()):
        raise ValueError("all teacher caches must cover the same texts")
    active = [k for k in names if weights.get(k, 0.0) > 0.0]
    if reference not in active:
        raise ValueError("the reference teacher needs a positive weight")
    rng = np.random.default_rng(seed)
    fit_idx = np.sort(rng.choice(n, size=min(fit_samples, n), replace=False))

    bases = {}
    reduced_fit = {}
    report: dict = {"kept_variance": {}, "alignment_cosine": {}}
    for k in active:
        x = caches[k].get(fit_idx)
        mean, basis, kept = _pca_basis(x, dim)
        bases[k] = (mean, basis)
        reduced_fit[k] = (x - mean) @ basis
        report["kept_variance"][k] = kept

    rotations = {reference: np.eye(dim, dtype=np.float32)}
    for k in active:
        if k == reference:
            continue
        u, _, vt = np.linalg.svd(reduced_fit[k].T.astype(np.float64) @ reduced_fit[reference].astype(np.float64))
        rotations[k] = (u @ vt).astype(np.float32)
        aligned = _unit(reduced_fit[k] @ rotations[k])
        report["alignment_cosine"][k] = float((aligned * _unit(reduced_fit[reference])).sum(axis=1).mean())

    total_weight = sum(weights[k] for k in active)
    fused = np.empty((n, dim), dtype=np.float16)
    for lo in range(0, n, CHUNK):
        idx = np.arange(lo, min(lo + CHUNK, n))
        acc = np.zeros((len(idx), dim), dtype=np.float32)
        for k in active:
            mean, basis = bases[k]
            aligned = ((caches[k].get(idx) - mean) @ basis) @ rotations[k]
            acc += (weights[k] / total_weight) * _unit(aligned)
        fused[lo : lo + len(idx)] = _unit(acc).astype(np.float16)
    return fused, report


METHODS = ("center", "gram", "shared", "fusion")


def method_overrides(method: Optional[str], teacher_names: list[str]) -> tuple[list[str], str]:
    """Config overrides and a run-name suffix for a notebook `METHOD` string.

    `method` is None or one or more of center, gram, shared, fusion joined with '+' (for example 'center+gram').
    Returns (overrides, suffix); both are empty for None. Raises ValueError for unknown or incompatible choices.
    """
    if not method:
        return [], ""
    chosen = [m.strip() for m in method.split("+") if m.strip()]
    unknown = [m for m in chosen if m not in METHODS]
    if unknown or len(set(chosen)) != len(chosen):
        raise ValueError(f"METHOD must combine {METHODS} once each with '+', got {method!r}")
    if "gram" in chosen and "shared" in chosen:
        raise ValueError("'shared' already standardises the Gram matrices: drop 'gram'")
    if "fusion" in chosen:
        if "center" in chosen:
            raise ValueError("'fusion' centres the teachers itself (PCA): drop 'center'")
        if len(teacher_names) < 2:
            raise ValueError("'fusion' needs a config with at least two teachers")
    overrides: list[str] = []
    for m in METHODS:  # fixed order, so the suffix does not depend on how the string was written
        if m not in chosen:
            continue
        if m == "center":
            overrides += [f"teachers.{i}.center=true" for i in range(len(teacher_names))]
        elif m == "gram":
            overrides += ["loss.gram_standardize=true"]
        elif m == "shared":
            overrides += ["loss.relation_mode=shared"]
        else:
            overrides += [
                "fusion.mode=aligned_mean", f"fusion.reference={teacher_names[0]}", "fusion.dim=512",
                "fusion.fit_samples=20000",
            ]
    return overrides, "_".join(m for m in METHODS if m in chosen)


def geometry_report(cache, sample: int = 3000, seed: int = 0, centered: bool = False) -> dict[str, float]:
    """Spread of a teacher's space: mean, std and 5th/95th percentile of the pairwise cosines of a text sample.

    `centered=True` reports the same after subtracting the corpus mean (what `CenteredCache` would serve), so the
    effect of method (a) can be read off before any training. The cache is only read.
    """
    n = len(cache)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    x = cache.get(idx)
    if centered:
        x = _unit(x - compute_mean(cache))
    sims = x @ x.T
    off = sims[~np.eye(len(x), dtype=bool)]
    return {
        "mean_cosine": float(off.mean()),
        "std": float(off.std()),
        "p5": float(np.percentile(off, 5)),
        "p95": float(np.percentile(off, 95)),
    }
