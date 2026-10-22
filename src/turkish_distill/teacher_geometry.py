"""Print how spread out every teacher's cached vectors are, with and without centring (no model, no training).

    python -m turkish_distill.teacher_geometry --config configs/stage1_single_cosmos_e5.yaml

Use it right after encoding a new teacher: a teacher whose pairwise cosines sit in a narrow cone (mE5-instruct:
mean 0.82, std 0.04) is the case where `center` / `gram` can matter; a spread-out one (BGE-M3: mean 0.29) is not.
"""

from __future__ import annotations

import argparse
from typing import Optional

from .config import load_config
from .data import load_corpus
from .targets import geometry_report
from .teachers import TeacherCache, corpus_fingerprint


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Pairwise-cosine statistics of the cached teacher vectors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    parser.add_argument("--sample", type=int, default=3000)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    texts = load_corpus(cfg.data, cfg.train.seed)
    fingerprint = corpus_fingerprint(texts)
    print(f"{'teacher':24s} {'view':9s} {'mean':>7s} {'std':>7s} {'p5':>7s} {'p95':>7s}")
    for teacher in cfg.teachers:
        cache = TeacherCache.open(cfg.data.cache_dir, teacher, fingerprint, len(texts))
        for label, centered in (("raw", False), ("centred", True)):
            r = geometry_report(cache, args.sample, centered=centered)
            print(f"{teacher.name:24s} {label:9s} {r['mean_cosine']:7.3f} {r['std']:7.3f} {r['p5']:7.3f} {r['p95']:7.3f}")


if __name__ == "__main__":
    main()
