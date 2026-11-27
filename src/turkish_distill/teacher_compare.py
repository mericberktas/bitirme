"""Compare several teachers' cached vectors on the same texts: per-teacher spread and dimensionality, pairwise agreement (no model is loaded).

    python -m turkish_distill.teacher_compare --config configs/three_bge_8k_voyage.yaml --n 10000 [--group-column bucket]

For each teacher: mean / std of the off-diagonal pairwise cosines, how many principal components hold 90% and 99% of
the variance and what share the first `--student-dim` components hold (a 768-d student cannot represent the rest).
For each pair of teachers:
- the Pearson correlation of their off-diagonal Gram entries ("geometry agreement") and the Spearman correlation of
  the same entries (on a random sample of pairs);
- the overlap of their top-k neighbour lists;
- how often they order random pairs of pairs (i, j) vs (k, l) the same way ("rank agreement"). The ranking hinge
  pulls the student towards each teacher's order, so where both teachers are decisive (|difference| above the hinge
  margin, `loss.resim_margin` by default) and disagree, the two hinge terms conflict; `conflict_rate` is that share
  of all sampled pairs of pairs.
With `--group-column` (e.g. the mix's `bucket`), the neighbour overlap and the Gram Pearson within each group are
also reported, to see where the teachers disagree. These are the numbers that were measured for Qwen3-8B before it
failed as a third teacher (its 90%-variance PC count was 787 against 319 / 211 for BGE-M3 / 8K-embed); they are a
screen, not a prediction of the trained student's score.
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


def _ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x), dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def _sample_pairs(n: int, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """m random index pairs (i, j) with i != j."""
    i = rng.integers(0, n, size=m)
    j = (i + rng.integers(1, n, size=m)) % n
    return i, j


def rank_agreement(ga: np.ndarray, gb: np.ndarray, num_quads: int, margin: float,
                   rng: np.random.Generator) -> dict:
    """How two Gram matrices order random pairs of pairs (i, j) vs (k, l)."""
    n = len(ga)
    i, j = _sample_pairs(n, num_quads, rng)
    k, l = _sample_pairs(n, num_quads, rng)
    da = ga[i, j] - ga[k, l]
    db = gb[i, j] - gb[k, l]
    same = np.sign(da) == np.sign(db)
    decisive = (np.abs(da) > margin) & (np.abs(db) > margin)
    return {
        "rank_agreement": float(same.mean()),
        "decisive_share": float(decisive.mean()),
        "rank_agreement_decisive": float(same[decisive].mean()) if decisive.any() else float("nan"),
        "conflict_rate": float((decisive & ~same).mean()),
    }


def geometry_report(vectors: dict[str, np.ndarray], k: int = 10, student_dim: int = 768,
                    groups: Optional[list] = None, num_quads: int = 200_000, margin: float = 0.015,
                    spearman_pairs: int = 2_000_000, seed: int = 0, min_group: int = 50) -> dict:
    """`vectors`: teacher name -> (n, dim) array, rows aligned (same texts in the same order).

    `groups`: optional label per row (e.g. the mix bucket); groups with at least `min_group` rows get their own
    neighbour overlap and within-group Gram Pearson for every teacher pair."""
    names = list(vectors)
    n = {len(v) for v in vectors.values()}
    if len(n) != 1:
        raise ValueError(f"teachers have different numbers of rows: { {a: len(v) for a, v in vectors.items()} }")
    vecs = {a: _normalize(v) for a, v in vectors.items()}
    grams = {a: v @ v.T for a, v in vecs.items()}
    out: dict = {"n": n.pop(), "k": k, "margin": margin, "teachers": {}, "pairs": {}}
    rows = out["n"]
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
    if groups is not None and len(groups) != rows:
        raise ValueError(f"groups has {len(groups)} labels for {rows} rows")
    group_rows: dict[str, np.ndarray] = {}
    if groups is not None:
        labels = np.asarray([str(g) for g in groups])
        for g in sorted(set(labels.tolist())):
            idx = np.flatnonzero(labels == g)
            if len(idx) >= min_group:
                group_rows[g] = idx
        out["groups"] = {g: int(len(idx)) for g, idx in group_rows.items()}
    tops = {a: _topk(grams[a], k) for a in names}
    for pi, (a, b) in enumerate(itertools.combinations(names, 2)):
        pearson = float(np.corrcoef(_offdiag(grams[a]), _offdiag(grams[b]))[0, 1])
        row_overlap = np.array([len(set(x) & set(y)) / k for x, y in zip(tops[a], tops[b])])
        rng = np.random.default_rng(seed + pi)
        si, sj = _sample_pairs(rows, min(spearman_pairs, rows * (rows - 1)), rng)
        spearman = float(np.corrcoef(_ranks(grams[a][si, sj]), _ranks(grams[b][si, sj]))[0, 1])
        pair = {"gram_pearson": pearson, "gram_spearman": spearman, f"top{k}_overlap": float(row_overlap.mean())}
        pair.update(rank_agreement(grams[a], grams[b], num_quads, margin, rng))
        if group_rows:
            per = {}
            for g, idx in group_rows.items():
                sub_a, sub_b = grams[a][np.ix_(idx, idx)], grams[b][np.ix_(idx, idx)]
                per[g] = {f"top{k}_overlap": float(row_overlap[idx].mean()),
                          "gram_pearson_within": float(np.corrcoef(_offdiag(sub_a), _offdiag(sub_b))[0, 1])}
            pair["by_group"] = per
        out["pairs"][f"{a}~{b}"] = pair
    return out


def format_report(rep: dict) -> str:
    lines = [f"{rep['n']} texts, top-{rep['k']} neighbours, decisive margin {rep.get('margin', float('nan')):.3f}"]
    for a, t in rep["teachers"].items():
        extra = " ".join(f"{key}={val:.3f}" for key, val in t.items() if key.startswith("var_first"))
        lines.append(f"  {a:24s} dim {t['dim']:5d}  mean cos {t['mean_cos']:.3f}  std {t['std_cos']:.3f}  "
                     f"PCs 90% {t['pcs_90']:5d}  99% {t['pcs_99']:5d}  {extra}")
    for pair, p in rep["pairs"].items():
        vals = "  ".join(f"{key} {val:.3f}" for key, val in p.items() if not isinstance(val, dict))
        lines.append(f"  {pair}\n      {vals}")
    if rep.get("groups"):
        k = rep["k"]
        pairs = list(rep["pairs"])
        lines.append(f"  per group: top-{k} overlap / Gram Pearson within the group")
        lines.append("  " + f"{'group':26s}{'n':>6s}" + "".join(f"  {p[:30]:>30s}" for p in pairs))
        for g, cnt in rep["groups"].items():
            cells = "".join(
                f"  {rep['pairs'][p]['by_group'][g][f'top{k}_overlap']:>15.3f} / "
                f"{rep['pairs'][p]['by_group'][g]['gram_pearson_within']:<12.3f}" for p in pairs)
            lines.append(f"  {g[:26]:26s}{cnt:6d}{cells}")
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
    p.add_argument("--group-column", default=None,
                   help="corpus column with a group label per text (e.g. bucket): adds per-group agreement")
    p.add_argument("--quads", type=int, default=200_000, help="random pairs of pairs for the rank agreement")
    p.add_argument("--margin", type=float, default=None,
                   help="decisive-difference threshold for the rank agreement (default: loss.resim_margin)")
    p.add_argument("--out", default=None, help="write the report as JSON")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.override)
    texts = load_corpus(cfg.data, cfg.train.seed)
    fp = corpus_fingerprint(texts)
    idx = np.arange(min(a.n, len(texts)))
    vectors = {t.name: TeacherCache.open(cfg.data.cache_dir, t, fp, len(texts)).get(idx) for t in cfg.teachers}
    groups = None
    if a.group_column:
        label: dict = {}
        with open(cfg.data.corpus_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                label.setdefault(row.get(cfg.data.text_column), row.get(a.group_column))
        groups = [label.get(texts[i]) for i in idx]
    margin = cfg.loss.resim_margin if a.margin is None else a.margin
    rep = geometry_report(vectors, a.k, a.student_dim, groups=groups, num_quads=a.quads, margin=margin,
                          seed=cfg.train.seed)
    print(format_report(rep))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
