"""Encode the corpus with the configured teachers into the shard cache.

    python -m turkish_distill.encode_teachers --config configs/multi_teacher.yaml [--only NAME] [--status]

Several sessions can run at the same time on different teachers (`--only`): every teacher has its own cache
folder. Two sessions on the same teacher are stopped by a lock file (`--ignore-lock` overrides a dead one).
`--status` loads no model; it only reports how many shards of each teacher exist.
"""

from __future__ import annotations

import argparse
from typing import Optional

from .config import load_config
from .data import load_corpus
from .teachers import cache_status, corpus_fingerprint, encode_teacher


def format_status(status: dict) -> str:
    line = f"[{status['teacher']}] {status['shards_done']}/{status['num_shards']} shards"
    line += ", complete" if status["complete"] else ", incomplete"
    if status["lock_age_s"] is not None:
        line += f", lock refreshed {status['lock_age_s'] / 60:.0f} min ago (a session may be encoding)"
    return line


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Encode the corpus with the configured teachers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", action="append", default=[], help="teacher name (repeatable); default: all")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--status", action="store_true", help="report the cache progress and exit (no model is loaded)")
    parser.add_argument("--ignore-lock", action="store_true", help="encode even if another session's lock is fresh")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    if not cfg.data.cache_dir:
        raise ValueError("data.cache_dir is empty")
    texts = load_corpus(cfg.data, cfg.train.seed)
    fingerprint = corpus_fingerprint(texts)
    print(f"corpus: {len(texts)} texts from {cfg.data.corpus_path}, fingerprint {fingerprint}", flush=True)
    selected = [t for t in cfg.teachers if not args.only or t.name in args.only]
    if not selected:
        raise ValueError(f"no teacher matches --only {args.only}")
    if args.status:
        for teacher in selected:
            print(format_status(cache_status(teacher, cfg.data.cache_dir, fingerprint, len(texts), cfg.data.shard_size)), flush=True)
        return
    for teacher in selected:
        path = encode_teacher(
            teacher, texts, cfg.data.cache_dir, cfg.data.shard_size, fingerprint=fingerprint, ignore_lock=args.ignore_lock
        )
        print(f"[{teacher.name}] cache ready: {path}", flush=True)


if __name__ == "__main__":
    main()
