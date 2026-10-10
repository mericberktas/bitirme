"""Encode the corpus with every teacher into the shard cache.

    python -m turkish_distill.encode_teachers --config configs/multi_teacher.yaml [--only NAME]
"""

from __future__ import annotations

import argparse
from typing import Optional

from .config import load_config
from .data import load_corpus
from .teachers import corpus_fingerprint, encode_teacher


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Encode the corpus with the configured teachers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", action="append", default=[], help="teacher name (repeatable); default: all")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    if not cfg.data.cache_dir:
        raise ValueError("data.cache_dir is empty")
    texts = load_corpus(cfg.data, cfg.train.seed)
    fingerprint = corpus_fingerprint(texts)
    print(f"corpus: {len(texts)} texts, fingerprint {fingerprint}", flush=True)
    selected = [t for t in cfg.teachers if not args.only or t.name in args.only]
    if not selected:
        raise ValueError(f"no teacher matches --only {args.only}")
    for teacher in selected:
        path = encode_teacher(teacher, texts, cfg.data.cache_dir, cfg.data.shard_size, fingerprint=fingerprint)
        print(f"[{teacher.name}] cache ready: {path}", flush=True)


if __name__ == "__main__":
    main()
