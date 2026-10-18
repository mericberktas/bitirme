"""Remove benchmark test texts' near-copies from a large corpus (streaming, bounded memory).

    python -m turkish_distill.decontaminate_corpus --corpus data/corpus.jsonl \
        --forbidden data/benchmark_texts.jsonl --out data/corpus_clean.jsonl

`--forbidden` is the file written by `benchmark_texts export` (its `.stats.json` next to it says which dataset each
line came from, so the report can name the benchmarks that overlap with the corpus). A corpus passage is dropped if
it equals a benchmark text after whitespace/case normalisation, or shares at least `--min-hits` word n-grams
(default: one 13-gram, as `prepare_pairs` does) with any benchmark text.

Memory: `prepare_pairs.decontaminate` indexes the n-grams of the texts being cleaned, which works for tens of
thousands of texts. A corpus of ~1M passages has hundreds of millions of n-grams, so here the benchmark side is
indexed instead: every benchmark n-gram hash goes into one sorted int64 array (about 1 GB for the 2M benchmark
texts) and each corpus passage is looked up with a binary search. Python's `hash()` is randomised per process, so the
index cannot be saved; building it takes a couple of minutes.

The output is written to a temporary file and moved into place at the end, so an interrupted run never leaves a
half-written corpus under the final name.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from array import array
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .prepare_pairs import _ngram_hashes, normalize


def _dataset_names_and_bounds(forbidden_path: Path) -> tuple[list[str], list[int]]:
    """Dataset names and the line number each one starts at, from `benchmark_texts export`'s stats file."""
    stats_path = forbidden_path.with_suffix(".stats.json")
    if stats_path.exists():
        counts = json.loads(stats_path.read_text(encoding="utf-8")).get("new_texts_per_dataset", {})
        if counts:
            names = list(counts)
            bounds, total = [], 0
            for name in names:
                bounds.append(total)
                total += counts[name]
            return names, bounds
    return ["benchmark"], [0]


def _iter_texts(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]


class BenchmarkIndex:
    """Sorted n-gram hashes (with the dataset each came from) and the hashes of whole normalised texts."""

    def __init__(self, texts: Iterator[str], bounds: list[int], n: int):
        self.n = n
        grams, gram_ds, exact, exact_ds = array("q"), array("B"), array("q"), array("B")
        dataset = 0
        for line_no, text in enumerate(texts):
            while dataset + 1 < len(bounds) and line_no >= bounds[dataset + 1]:
                dataset += 1
            exact.append(hash(normalize(text).lower()))
            exact_ds.append(dataset)
            found = _ngram_hashes(text, n)
            grams.extend(found)
            gram_ds.extend([dataset] * len(found))
        self.grams, self.gram_ds = self._sorted(grams, gram_ds)
        self.exact, self.exact_ds = self._sorted(exact, exact_ds)

    @staticmethod
    def _sorted(values: array, datasets: array) -> tuple[np.ndarray, np.ndarray]:
        v, d = np.frombuffer(values, dtype=np.int64), np.frombuffer(datasets, dtype=np.uint8)
        order = np.argsort(v, kind="stable")
        return v[order], d[order]

    @staticmethod
    def _lookup(sorted_values: np.ndarray, datasets: np.ndarray, query: np.ndarray):
        """(hit mask, dataset of each hit) for the query values."""
        if len(query) == 0 or len(sorted_values) == 0:
            return np.zeros(len(query), dtype=bool), np.zeros(len(query), dtype=np.uint8)
        pos = np.minimum(np.searchsorted(sorted_values, query), len(sorted_values) - 1)
        hit = sorted_values[pos] == query
        return hit, datasets[pos]

    def check(self, text: str) -> tuple[int, Optional[int], bool]:
        """(number of shared n-grams, dataset of the first shared one, equals a whole benchmark text)."""
        exact_hit, exact_ds = self._lookup(self.exact, self.exact_ds, np.array([hash(normalize(text).lower())], dtype=np.int64))
        grams = np.fromiter(_ngram_hashes(text, self.n), dtype=np.int64)
        hit, ds = self._lookup(self.grams, self.gram_ds, grams)
        hits = int(hit.sum())
        dataset = int(ds[hit][0]) if hits else (int(exact_ds[0]) if exact_hit[0] else None)
        return hits, dataset, bool(exact_hit[0])


def decontaminate_corpus(
    corpus_path: str | Path,
    forbidden_path: str | Path,
    out_path: str | Path,
    n: int = 13,
    min_hits: int = 1,
    progress_every: int = 100_000,
) -> dict:
    corpus_path, forbidden_path, out_path = Path(corpus_path), Path(forbidden_path), Path(out_path)
    started = time.time()
    names, bounds = _dataset_names_and_bounds(forbidden_path)
    index = BenchmarkIndex(_iter_texts(forbidden_path), bounds, n)
    print(f"indexed {len(index.grams):,} benchmark {n}-grams and {len(index.exact):,} texts in {time.time() - started:.0f}s", flush=True)

    removed_by_dataset = {name: 0 for name in names}
    histogram = {"0-1": 0, "2-4": 0, "5-19": 0, "20+": 0}  # 0 = equal to a whole (short) benchmark text
    total = kept = removed_exact = 0
    tmp = out_path.with_name(out_path.name + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            hits, dataset, exact = index.check(json.loads(line)["text"])
            if exact or hits >= min_hits:
                removed_exact += int(exact)
                removed_by_dataset[names[dataset if dataset is not None else 0]] += 1
                bucket = "0-1" if hits <= 1 else "2-4" if hits < 5 else "5-19" if hits < 20 else "20+"
                histogram[bucket] += 1
            else:
                kept += 1
                dst.write(line if line.endswith("\n") else line + "\n")
            if progress_every and total % progress_every == 0:
                print(f"  {total:,} passages scanned, {total - kept:,} removed", flush=True)
    os.replace(tmp, out_path)
    stats = {
        "corpus": str(corpus_path), "forbidden": str(forbidden_path), "ngram": n, "min_hits": min_hits,
        "passages": total, "kept": kept, "removed": total - kept, "removed_equal_to_a_benchmark_text": removed_exact,
        "removed_by_first_matching_dataset": {k: v for k, v in removed_by_dataset.items() if v},
        "removed_by_number_of_shared_ngrams": histogram, "seconds": round(time.time() - started),
    }
    out_path.with_suffix(".decontam.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Remove benchmark test texts and their near-copies from a corpus")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--forbidden", required=True, help="JSONL from `benchmark_texts export`")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--min-hits", type=int, default=1, help="shared n-grams needed to drop a passage")
    args = parser.parse_args(argv)
    print(json.dumps(decontaminate_corpus(args.corpus, args.forbidden, args.out, args.ngram, args.min_hits), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
