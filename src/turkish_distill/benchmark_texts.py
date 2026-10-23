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
import gzip
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

# The test data of the Turkish tasks of the Mizan leaderboard (mteb 1.38.51; revisions pinned as in that release).
# The nine tasks that TR-MTEB does not contain are our development benchmark; the five shared ones (Movie and
# Product sentiment, XQuAD, XNLI, TurHistQuAD = TQuAD) are already covered by BENCHMARK_FILES. All splits an
# evaluation reads are included (Massive also reads validation; SIB200 trains its classifier on the train split).
MIZAN_FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "mteb/amazon_massive_intent": ("4672e20407010da34463acc759c162ca9734bca6", ("test/tr.json.gz", "validation/tr.json.gz")),  # MassiveIntentClassification
    "mteb/amazon_massive_scenario": ("fad2c6e8459f9e1c45d9315f4953d921437d70f8", ("test/tr.json.gz", "validation/tr.json.gz")),  # MassiveScenarioClassification
    "mteb/multilingual-sentiment-classification": ("2b9b4d10fc589af67794141fe8cbd3739de1eb33", ("test/tur.parquet",)),  # MultilingualSentimentClassification
    "mteb/sib200": ("a74d7350ea12af010cfb1c21e34f1f81fd2e615b", ("test/tur_Latn.parquet", "train/tur_Latn.parquet", "validation/tur_Latn.parquet")),  # SIB200Classification, SIB200ClusteringS2S
    "PaDaS-Lab/webfaq-retrieval": ("c3262adb1c32ac0c3ea8de6393a44366edaa62e1", ("tur/queries-test.jsonl", "tur/corpus.jsonl")),  # WebFAQRetrieval
    "apple/mkqa": ("main", ("mkqa.jsonl.gz",)),  # MKQARetrieval (a loading script on the Hub; the data file is on GitHub, see DATA_URLS)
    "mteb/xnli2.0-multi-pair": ("5b7d477a8c62cdd18e2fed7e015497c20b4371ad", ("turkish/test-00000-of-00001.parquet",)),  # XNLIV2
    "mteb/sts22-crosslingual-sts": ("d31f33a128469b20e357535c39b82fb3c3f6f2bd", ("test/tr.jsonl.gz",)),  # STS22.v2
}
SUITES = {"trmteb": BENCHMARK_FILES, "mizan": MIZAN_FILES}
# repos whose files do not live on the Hub: URL template with {file}
DATA_URLS = {"apple/mkqa": "https://github.com/apple/ml-mkqa/raw/main/dataset/{file}"}

HUB = "https://huggingface.co"
Opener = Callable[[str], object]


def _urlopen(url: str):
    return urllib.request.urlopen(url, timeout=120)


def resolve_commit(repo: str, revision: str, opener: Opener = _urlopen) -> str:
    with opener(f"{HUB}/api/datasets/{repo}/revision/{revision}") as response:
        return json.loads(response.read())["sha"]


def download(
    out_dir: str | Path,
    opener: Opener = _urlopen,
    repos: Optional[Iterable[str]] = None,
    files_map: Optional[dict] = None,
) -> dict:
    """Fetch the listed files (skipping finished ones) and write manifest.json with the commit hashes."""
    out_dir = Path(out_dir)
    files_map = BENCHMARK_FILES if files_map is None else files_map
    manifest: dict[str, dict] = {}
    for repo in repos if repos is not None else files_map:
        revision, files = files_map[repo]
        commit = revision if repo in DATA_URLS else resolve_commit(repo, revision, opener)
        sizes = {}
        for rel in files:
            dest = out_dir / repo / rel
            if not (dest.exists() and dest.stat().st_size > 0):
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")
                url = DATA_URLS[repo].format(file=rel) if repo in DATA_URLS else f"{HUB}/datasets/{repo}/resolve/{commit}/{rel}"
                with opener(url) as response, open(part, "wb") as f:
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
    elif path.suffix in (".jsonl", ".gz", ".json"):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield from _json_row_texts(json.loads(line))
    else:
        raise ValueError(f"unsupported file type: {path}")


def _json_row_texts(row) -> Iterator[str]:
    """Texts of one JSON row. An MKQA row holds 26 languages: only its Turkish query and answers are evaluated."""
    if isinstance(row, dict) and isinstance(row.get("queries"), dict) and isinstance(row.get("answers"), dict):
        yield from _strings(row["queries"].get("tr"))
        for answer in row["answers"].get("tr") or []:
            yield from _strings(answer.get("text") if isinstance(answer, dict) else None)
    else:
        yield from _strings(row)


def export(in_dir: str | Path, out_path: str | Path, min_chars: int = 20, files_map: Optional[dict] = None) -> dict:
    """Unique texts of at least `min_chars` characters (ids, labels and other short cells are skipped)."""
    in_dir = Path(in_dir)
    files_map = BENCHMARK_FILES if files_map is None else files_map
    seen: set[str] = set()
    per_repo: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for repo, (_, files) in files_map.items():
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


def merge(base_path: str | Path, extra_path: str | Path, out_path: str | Path) -> dict:
    """One forbidden list out of two exports (for example TR-MTEB + Mizan): the base texts first, then the extra texts
    the base does not have. `.stats.json` keeps the dataset order, so `decontaminate_corpus` can still name the
    benchmark a copy came from; the extra datasets' counts are the NEW texts they add."""
    base_path, extra_path, out_path = Path(base_path), Path(extra_path), Path(out_path)
    base_stats = json.loads(base_path.with_suffix(".stats.json").read_text(encoding="utf-8"))
    extra_stats = json.loads(extra_path.with_suffix(".stats.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    names = list(extra_stats["new_texts_per_dataset"])
    bounds = []
    total = 0
    for name in names:
        bounds.append(total)
        total += extra_stats["new_texts_per_dataset"][name]
    new_per_extra = {name: 0 for name in names}
    with open(out_path, "w", encoding="utf-8") as out:
        with open(base_path, encoding="utf-8") as f:
            for line in f:
                seen.add(json.loads(line)["text"])
                out.write(line if line.endswith(chr(10)) else line + chr(10))
        dataset = 0
        with open(extra_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                while dataset + 1 < len(bounds) and line_no >= bounds[dataset + 1]:
                    dataset += 1
                text = json.loads(line)["text"]
                if text not in seen:
                    seen.add(text)
                    new_per_extra[names[dataset]] += 1
                    out.write(line if line.endswith(chr(10)) else line + chr(10))
    stats = {
        "unique_texts": len(seen),
        "new_texts_per_dataset": {**base_stats["new_texts_per_dataset"], **new_per_extra},
        "min_chars": min(base_stats.get("min_chars", 20), extra_stats.get("min_chars", 20)),
    }
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dir", required=True)
    d.add_argument("--suite", choices=sorted(SUITES), default="trmteb")
    e = sub.add_parser("export")
    e.add_argument("--dir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--min-chars", type=int, default=20)
    e.add_argument("--suite", choices=sorted(SUITES), default="trmteb")
    m = sub.add_parser("merge", help="combine two exports into one forbidden list")
    m.add_argument("--base", required=True)
    m.add_argument("--extra", required=True)
    m.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.dir, files_map=SUITES[args.suite])
    elif args.command == "merge":
        print(json.dumps(merge(args.base, args.extra, args.out), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(export(args.dir, args.out, args.min_chars, SUITES[args.suite]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
