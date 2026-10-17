"""Download the TR-MTEB test data straight from the Hugging Face Hub and export its texts.

    python -m turkish_distill.benchmark_texts download --dir data/benchmark
    python -m turkish_distill.benchmark_texts export --dir data/benchmark --out data/benchmark_texts.jsonl

The exported JSONL (one {"text": ...} per unique text) feeds `prepare_pairs --forbidden`, which drops
training pairs that share a passage or question with a benchmark text (exact match or a 13-word n-gram).

Only what the evaluation reads is fetched: the test split of every task, and for the retrieval tasks the
whole corpus and the queries (qrels are not needed). The file list below was taken from the task metadata
of the `mteb_tr` fork (MTEB(Turkish), 26 tasks). Many of those datasets are not pinned to a commit there,
so `download` resolves the revision to a commit hash and records it in `manifest.json`.
Train/validation splits are NOT included; texts from them are not evaluation data.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

_RETRIEVAL = ("corpus/corpus-00000-of-00001.parquet", "queries/queries-00000-of-00001.parquet")
_TEST = ("data/test-00000-of-00001.parquet",)

# repo -> (revision named in mteb_tr, files)            # tasks that read it
BENCHMARK_FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "trmteb/wmt16_en_tr": ("main", _TEST),  # WMT16BitextMining
    "asparius/Turkish-Movie-Review": ("409a4415cce5f6bcfca6d5f3ca3c408211ca00b3", ("test.csv",)),  # TurkishMovieSentimentClassification
    "trmteb/offenseval": ("main", _TEST),  # TurkishOffensiveLanguageClassification
    "asparius/Turkish-Product-Review": ("ad861e463abda351ff65ca5ac0cc5985afe9eb99", ("test.csv",)),  # TurkishProductSentimentClassification
    "trmteb/news-cat": ("main", _TEST),  # TurkishNewsCategoryClassification
    "trmteb/75haber": ("main", _TEST),  # Turkish75NewsClassification
    "trmteb/ts_timeline_news_category": ("main", _TEST),  # TSTimelineNewsCategoryClassification
    "trmteb/thy_sa": ("main", _TEST),  # THYSentimentClassification
    "trmteb/irony-tr": ("main", _TEST),  # TurkishIronyClassification
    "trmteb/ts_abstract_corpus_p2p": ("main", _TEST),  # TurkishAbstractCorpusClustering
    "trmteb/630koseyazisi_p2p": ("main", _TEST),  # TurkishColumnWritingClustering
    "trmteb/multinli_tr": ("main", ("data/test_matched-00000-of-00001.parquet", "data/test_mismatched-00000-of-00001.parquet")),  # MnliTr
    "trmteb/snli_tr": ("main", _TEST),  # SnliTr
    "mteb/xnli": ("09698e0180d87dc247ca447d3a1248b931ac0cdb", ("test/tr.parquet",)),  # XNLI
    "trmteb/stsb-tr": ("main", _TEST),  # STSbTR
    "trmteb/squad-tr": ("main", _RETRIEVAL),  # SquadTRRetrieval
    "trmteb/tquad": ("main", _RETRIEVAL),  # TQuadRetrieval
    "google/xquad": ("51adfef1c1287aab1d2d91b5bead9bcfb9c68583", ("xquad.tr/validation-00000-of-00001.parquet",)),  # XQuADRetrieval
    "trmteb/msmarco-tr": ("main", _RETRIEVAL),  # MSMarcoTRRetrieval
    "trmteb/nfcorpus-tr": ("main", _RETRIEVAL),  # NFCorpusTR
    "trmteb/arguana-tr": ("main", _RETRIEVAL),  # ArguAnaTR
    "trmteb/cqadupstack-gaming-tr": ("main", _RETRIEVAL),  # CQADupstackGamingRetrievalTR
    "trmteb/fiqa-tr": ("main", _RETRIEVAL),  # FiQA2018TR
    "trmteb/quora-tr": ("main", _RETRIEVAL),  # QuoraRetrievalTR
    "trmteb/scidocs-tr": ("main", _RETRIEVAL),  # SCIDOCSTR
    "trmteb/scifact-tr": ("main", _RETRIEVAL),  # SciFactTR
}

HUB = "https://huggingface.co"
Opener = Callable[[str], object]


def _urlopen(url: str):
    return urllib.request.urlopen(url, timeout=120)


def resolve_commit(repo: str, revision: str, opener: Opener = _urlopen) -> str:
    with opener(f"{HUB}/api/datasets/{repo}/revision/{revision}") as response:
        return json.loads(response.read())["sha"]


def download(out_dir: str | Path, opener: Opener = _urlopen, repos: Optional[Iterable[str]] = None) -> dict:
    """Fetch the listed files (skipping finished ones) and write manifest.json with the commit hashes."""
    out_dir = Path(out_dir)
    manifest: dict[str, dict] = {}
    for repo in repos if repos is not None else BENCHMARK_FILES:
        revision, files = BENCHMARK_FILES[repo]
        commit = resolve_commit(repo, revision, opener)
        sizes = {}
        for rel in files:
            dest = out_dir / repo / rel
            if not (dest.exists() and dest.stat().st_size > 0):
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")
                with opener(f"{HUB}/datasets/{repo}/resolve/{commit}/{rel}") as response, open(part, "wb") as f:
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
                part.replace(dest)  # a half-written file never has the final name
            sizes[rel] = dest.stat().st_size
        manifest[repo] = {"requested_revision": revision, "commit": commit, "files": sizes}
        print(f"{repo}: {sum(sizes.values()) / 1e6:.1f} MB @ {commit[:8]}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _strings(value) -> Iterator[str]:
    """Every string inside a cell: plain strings, dicts (e.g. translation pairs) and lists."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def iter_file_texts(path: Path) -> Iterator[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=20_000):
            for column in batch.to_pydict().values():
                for cell in column:
                    yield from _strings(cell)
    elif path.suffix == ".csv":
        csv.field_size_limit(1 << 30)
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for cell in row.values():
                    yield from _strings(cell)
    else:
        raise ValueError(f"unsupported file type: {path}")


def export(in_dir: str | Path, out_path: str | Path, min_chars: int = 20) -> dict:
    """Unique texts of at least `min_chars` characters (ids, labels and other short cells are skipped)."""
    in_dir = Path(in_dir)
    seen: set[str] = set()
    per_repo: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for repo, (_, files) in BENCHMARK_FILES.items():
            added = 0
            for rel in files:
                path = in_dir / repo / rel
                if not path.exists():
                    raise FileNotFoundError(f"{path} is missing; run `download` first")
                for text in iter_file_texts(path):
                    text = " ".join(text.split())
                    if len(text) >= min_chars and text not in seen:
                        seen.add(text)
                        out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        added += 1
            per_repo[repo] = added
    stats = {"unique_texts": len(seen), "new_texts_per_dataset": per_repo, "min_chars": min_chars}
    Path(out_path).with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dir", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--min-chars", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.dir)
    else:
        print(json.dumps(export(args.dir, args.out, args.min_chars), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
