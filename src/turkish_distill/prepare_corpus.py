"""Build a training corpus (JSONL) from a Hugging Face dataset, streaming.

    python -m turkish_distill.prepare_corpus --dataset ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0 \
        --text-column text --max-samples 1000000 --out corpus.jsonl

Long documents are cut into passages of at most --max-chars characters at a
whitespace boundary. Each line is {"text": ...} plus {"url": ...} when the dataset has a URL column
(--url-column, default "url"; pass "" to drop it): the source site is needed to balance a corpus by domain.
Exact duplicates are removed later by `load_texts`. Decontamination against benchmark test sets is NOT done
here (see `decontaminate_corpus`). The output is written to a temporary file and moved into place at the
end, with a progress line every --progress-every passages.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional


def split_passages(text: str, max_chars: int, min_chars: int) -> Iterator[str]:
    text = " ".join(text.split())
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        cut = cut if cut > min_chars else max_chars
        yield text[:cut].strip()
        text = text[cut:].strip()
    if len(text) >= min_chars:
        yield text


def passages_from_rows(
    rows: Iterable[dict],
    text_column: str,
    url_column: Optional[str],
    max_chars: int,
    min_chars: int,
    max_passages_per_doc: int,
    max_samples: int,
) -> Iterator[dict]:
    """{"text": passage[, "url": url]} for the first `max_samples` passages of the rows."""
    written = 0
    for row in rows:
        text = row.get(text_column)
        if not isinstance(text, str):
            continue
        url = row.get(url_column) if url_column else None
        for k, passage in enumerate(split_passages(text, max_chars, min_chars)):
            if k >= max_passages_per_doc:
                break
            item = {"text": passage}
            if isinstance(url, str) and url:
                item["url"] = url
            yield item
            written += 1
            if written >= max_samples:
                return


def write_jsonl_atomic(path: str | Path, items: Iterable[dict], progress_every: int = 100_000) -> int:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"  {written:,} passages written", flush=True)
    os.replace(tmp, path)  # an interrupted run never leaves a half-written file under the final name
    return written


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--url-column", default="url", help="kept as a 'url' field when present; '' drops it")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-passages-per-doc", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config_name, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    items = passages_from_rows(
        ds, args.text_column, args.url_column or None, args.max_chars, args.min_chars,
        args.max_passages_per_doc, args.max_samples,
    )
    written = write_jsonl_atomic(args.out, items, args.progress_every)
    print(f"wrote {written} passages to {args.out}")


if __name__ == "__main__":
    main()
