"""Build a training corpus (JSONL) from a Hugging Face dataset, streaming.

    python -m turkish_distill.prepare_corpus --dataset ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0 \
        --text-column text --max-samples 1000000 --out corpus.jsonl

Long documents are cut into passages of at most --max-chars characters at a
whitespace boundary. Exact duplicates are removed later by `load_texts`.
Decontamination against benchmark test sets is NOT done here.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterator, Optional


def split_passages(text: str, max_chars: int, min_chars: int) -> Iterator[str]:
    text = " ".join(text.split())
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        cut = cut if cut > min_chars else max_chars
        yield text[:cut].strip()
        text = text[cut:].strip()
    if len(text) >= min_chars:
        yield text


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-passages-per-doc", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config_name, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get(args.text_column)
            if not isinstance(text, str):
                continue
            for k, passage in enumerate(split_passages(text, args.max_chars, args.min_chars)):
                if k >= args.max_passages_per_doc:
                    break
                f.write(json.dumps({"text": passage}, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.max_samples:
                    break
            if written >= args.max_samples:
                break
    print(f"wrote {written} passages to {args.out}")


if __name__ == "__main__":
    main()
