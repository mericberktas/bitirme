"""Compare several teachers' cached vectors on the same texts: per-teacher spread and dimensionality, pairwise agreement (no model is loaded).

    python -m turkish_distill.teacher_compare --config configs/three_bge_8k_voyage.yaml --n 10000

For each teacher: mean / std of the off-diagonal pairwise cosines, how many principal components hold 90% and 99% of
the variance and what share the first `--student-dim` components hold (a 768-d student cannot represent the rest).
For each pair of teachers: the Pearson correlation of their off-diagonal Gram entries ("geometry agreement") and the
overlap of their top-k neighbour lists. These are the numbers that were measured for Qwen3-8B before it failed as a
third teacher (its 90%-variance PC count was 787 against 319 / 211 for BGE-M3 / 8K-embed); they are a screen, not a
prediction of the trained student's score.
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Optional

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _offdiag(g: np.ndarray) -> np.ndarray:
    return g[~np.eye(len(g), dtype=bool)]


def _topk(g: np.ndarray, k: int) -> np.ndarray:
    g = g.copy()
    np.fill_diagonal(g, -np.inf)
    return np.argpartition(-g, k, axis=1)[:, :k]


def geometry_report(vectors: dict[str, np.ndarray], k: int = 10, student_dim: int = 768) -> dict:
    """`vectors`: teacher name -> (n, dim) array, rows aligned (same texts in the same order)."""
    names = list(vectors)
    n = {len(v) for v in vectors.values()}
    if len(n) != 1:
        raise ValueError(f"teachers have different numbers of rows: { {a: len(v) for a, v in vectors.items()} }")
    vecs = {a: _normalize(v) for a, v in vectors.items()}
    grams = {a: v @ v.T for a, v in vecs.items()}
    out: dict = {"n": n.pop(), "k": k, "teachers": {}, "pairs": {}}
    for a in names:
        off = _offdiag(grams[a])
        centred = vecs[a] - vecs[a].mean(axis=0, keepdims=True)
        var = np.linalg.svd(centred, compute_uv=False) ** 2
        share = np.cumsum(var) / var.sum()
        out["teachers"][a] = {
            "dim": int(vecs[a].shape[1]),
            "mean_cos": float(off.mean()),
            "std_cos": float(off.std()),
            "pcs_90": int(np.searchsorted(share, 0.90) + 1),
            "pcs_99": int(np.searchsorted(share, 0.99) + 1),
            f"var_first_{student_dim}": float(share[min(student_dim, len(share)) - 1]),
        }
    tops = {a: _topk(grams[a], k) for a in names}
    for a, b in itertools.combinations(names, 2):
        pearson = float(np.corrcoef(_offdiag(grams[a]), _offdiag(grams[b]))[0, 1])
        overlap = np.mean([len(set(x) & set(y)) / k for x, y in zip(tops[a], tops[b])])
        out["pairs"][f"{a}~{b}"] = {"gram_pearson": pearson, f"top{k}_overlap": float(overlap)}
    return out


def format_report(rep: dict) -> str:
    lines = [f"{rep['n']} texts, top-{rep['k']} neighbours"]
    for a, t in rep["teachers"].items():
        extra = " ".join(f"{key}={val:.3f}" for key, val in t.items() if key.startswith("var_first"))
        lines.append(f"  {a:24s} dim {t['dim']:5d}  mean cos {t['mean_cos']:.3f}  std {t['std_cos']:.3f}  "
                     f"PCs 90% {t['pcs_90']:5d}  99% {t['pcs_99']:5d}  {extra}")
    for pair, p in rep["pairs"].items():
        vals = "  ".join(f"{key} {val:.3f}" for key, val in p.items())
        lines.append(f"  {pair:50s} {vals}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    from .config import load_config
    from .data import load_corpus
    from .teachers import TeacherCache, corpus_fingerprint

    p = argparse.ArgumentParser(description="Compare teacher geometries from cached vectors")
    p.add_argument("--config", required=True)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--n", type=int, default=10000, help="first n texts of the corpus")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--student-dim", type=int, default=768)
    p.add_argument("--out", default=None, help="write the report as JSON")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.override)
    texts = load_corpus(cfg.data, cfg.train.seed)
    fp = corpus_fingerprint(texts)
    idx = np.arange(min(a.n, len(texts)))
    vectors = {t.name: TeacherCache.open(cfg.data.cache_dir, t, fp, len(texts)).get(idx) for t in cfg.teachers}
    rep = geometry_report(vectors, a.k, a.student_dim)
    print(format_report(rep))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
