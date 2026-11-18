"""Score every corpus text by how far a trained student still is from its teachers (for data pruning / sampling).

    python -m turkish_distill.score_texts --config C --model RUN_DIR_OR_FINAL --out scores.npz [--override a.b=c]

The score of a text is the student's vector-term error, 1 - cos(head_t(student(text)), teacher_t(text)), averaged over
the teachers with their weights (at step 0). The student is a finished run of the SAME config family: its `final/`
folder (backbone + teacher_heads.pt with one head per teacher of this config). The corpus, teacher caches and target
methods (centring, PCA) are those of the config, so the scores line up with the training texts one to one; the file
records the corpus fingerprint and training refuses a file made for another corpus.

Output (.npz): `score` (float32, one per corpus text, validation texts included), `err_<teacher>` per teacher,
`fingerprint`, `teachers`. A summary is printed and written next to it as .json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .config import ExperimentConfig, load_config, teacher_weight
from .data import Collator, load_corpus, student_texts
from .models import StudentEncoder
from .teachers import TeacherCache, corpus_fingerprint
from .utils import get_device


def final_dir(path: str | Path) -> Path:
    path = Path(path)
    return path / "final" if (path / "final" / "teacher_heads.pt").exists() else path


@torch.no_grad()
def score_corpus(
    model: StudentEncoder,
    collator: Collator,
    n: int,
    weights: dict[str, float],
    device: torch.device,
    batch_size: int = 64,
    use_amp: bool = False,
) -> dict[str, np.ndarray]:
    """Per-text errors of the student against every teacher with a positive weight, and their weighted mean."""
    model.eval()
    active = [name for name, w in weights.items() if w > 0]
    errs = {name: np.zeros(n, dtype=np.float32) for name in active}
    for start in range(0, n, batch_size):
        idx = list(range(start, min(n, start + batch_size)))
        batch = collator(idx)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, projections = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for name in active:
            pred = torch.nn.functional.normalize(projections[name].float(), dim=-1)
            target = torch.nn.functional.normalize(batch["targets"][name].to(device).float(), dim=-1)
            errs[name][start : start + len(idx)] = (1.0 - (pred * target).sum(dim=-1)).cpu().numpy()
    total = sum(weights[name] for name in active)
    score = sum(weights[name] * errs[name] for name in active) / total
    out = {"score": score.astype(np.float32)}
    out.update({f"err_{name}": e for name, e in errs.items()})
    return out


def save_scores(path: str | Path, scores: dict[str, np.ndarray], fingerprint: str, teachers: list[str]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, fingerprint=np.array(fingerprint), teachers=np.array(teachers), **scores)
    s = scores["score"]
    summary = {
        "n": int(len(s)),
        "fingerprint": fingerprint,
        "teachers": teachers,
        "mean": float(s.mean()),
        "quantiles": {f"p{q}": float(np.quantile(s, q / 100)) for q in (1, 10, 25, 50, 75, 90, 99)},
    }
    for key, values in scores.items():
        if key.startswith("err_"):
            summary[f"mean_{key}"] = float(values.mean())
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def load_scores(path: str | Path, fingerprint: str, n: int) -> np.ndarray:
    """The `score` array of a scores file, checked against the corpus it is used with."""
    with np.load(path, allow_pickle=False) as f:
        stored = str(f["fingerprint"])
        score = np.asarray(f["score"], dtype=np.float64)
    if stored != fingerprint or len(score) != n:
        raise ValueError(
            f"scores file {path} was made for another corpus (fingerprint {stored}, {len(score)} texts; "
            f"this corpus: {fingerprint}, {n} texts)"
        )
    if not np.isfinite(score).all():
        raise ValueError(f"scores file {path} has non-finite values")
    return score


def run(cfg: ExperimentConfig, model_path: str | Path, out: str | Path, batch_size: int = 64) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from .train import apply_target_methods

    texts = load_corpus(cfg.data, cfg.train.seed)
    fingerprint = corpus_fingerprint(texts)
    caches = {t.name: TeacherCache.open(cfg.data.cache_dir, t, fingerprint, len(texts)) for t in cfg.teachers}
    cfg, caches = apply_target_methods(cfg, caches)
    folder = final_dir(model_path)
    student = dataclasses.replace(cfg.student, hf_id=str(folder), revision=None, heads_path=str(folder / "teacher_heads.pt"))
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    model = StudentEncoder.from_pretrained(student, cfg.teachers)
    device = get_device()
    model.to(device)
    queries: set[str] = set()
    if cfg.student.query_prefix:
        from .teachers import load_query_set

        queries = load_query_set(cfg.student.query_sources)
    seen, _ = student_texts(texts, cfg.student.query_prefix, queries)
    collator = Collator(tokenizer, seen, caches, cfg.student.max_length)
    weights = {t.name: teacher_weight(t, 0) for t in cfg.teachers}
    scores = score_corpus(model, collator, len(texts), weights, device, batch_size, use_amp=device.type == "cuda")
    summary = save_scores(out, scores, fingerprint, [n for n, w in weights.items() if w > 0])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score corpus texts by the student's remaining error")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="a finished run directory (or its final/ folder) of this config family")
    parser.add_argument("--out", required=True, help="output .npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    args = parser.parse_args(argv)
    run(load_config(args.config, args.override), args.model, args.out, args.batch_size)


if __name__ == "__main__":
    main()
