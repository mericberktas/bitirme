"""Add the datasets' answers to an existing pairs file, keeping its pairs exactly as they are.

    python -m turkish_distill.attach_answers --pairs data/stage1/pairs_train_clean.jsonl \
        --out data/stage1/pairs_train_clean_ans.jsonl --raw-dir data/raw --download \
        --forbidden data/benchmark_texts_all.jsonl

`prepare_pairs` now writes an `answer` field itself, but the pairs that the earlier runs used (`pairs_train_clean`:
decontaminated, then filtered against the RAG corpora) were made before that. Re-running `prepare_pairs` would
reproduce them only up to that later filter, so this script looks the answers up instead: a pair is matched on
(source, question, passage_id) against the raw datasets, read with the same loaders, so normalisation is identical.

Answers that equal a forbidden (benchmark test) text or share a 13-word n-gram with one are removed (the pair stays,
without an answer). Rows without an answer are kept; the answer-weighted loss skips them. Writes `<out>` and
`<out stem>.stats.json` (matches per source, conflicts, removed answers, answer lengths per source).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable, Optional

from . import prepare_pairs as pp

# The raw files the stage-1 pairs were made from; revisions pinned (HF commit hashes, 2026-11-12).
RAW_SOURCES = {
    "turhistquad": ("asparius/TurHistQuAD", "2a2b8ddecf1189f530676244d0751e1d0a569e03", ["train.json", "test.json"]),
    "wikirag": ("Metin/WikiRAG-TR", "ce6af3f7cbc8fceab80c821038af9caf346eb20b", ["data/train.csv"]),
    "wikiqa": ("hcsolakoglu/turkish-wikipedia-qa-4-million", "d979eb01bc0de4907b23ef4473f04ef13f8a9bf0",
               ["data/turkish-wiki-qa-4mil.parquet"]),
}
Key = tuple[str, str, str]  # (source, question, passage_id)


def download_raw(raw_dir: str | Path) -> dict[str, Path]:
    """Download the three raw datasets (pinned revisions) into raw_dir/<source>/; returns the paths the loaders take."""
    from huggingface_hub import hf_hub_download

    raw_dir = Path(raw_dir)
    for source, (repo, revision, files) in RAW_SOURCES.items():
        for name in files:
            hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=raw_dir / source)
    return raw_paths(raw_dir)


def raw_paths(raw_dir: str | Path) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    return {
        "turhistquad": raw_dir / "turhistquad",
        "wikirag": raw_dir / "wikirag" / "data" / "train.csv",
        "wikiqa": raw_dir / "wikiqa" / "data" / "turkish-wiki-qa-4mil.parquet",
    }


def _key(pair: dict) -> Key:
    return pair["source"], pair["question"], pair["passage_id"]


def _add(index: dict[Key, str], conflicts: dict[str, int], pair: Optional[dict]) -> None:
    if not pair or not pair.get("answer"):
        return
    key = _key(pair)
    if key in index:
        if index[key] != pair["answer"]:
            conflicts[pair["source"]] = conflicts.get(pair["source"], 0) + 1  # first answer wins
        return
    index[key] = pair["answer"]


def wikiqa_answers(path: str | Path, passage_ids: set[str]) -> Iterable[dict]:
    """Every (question, answer) of the Wikipedia-QA articles whose passage id is in `passage_ids`."""
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=20_000, columns=["original_text", "answers"]):
        columns = batch.to_pydict()
        for text, answers in zip(columns["original_text"], columns["answers"]):
            passage = pp.normalize(text or "")
            if not passage or pp.passage_id(passage) not in passage_ids:
                continue
            try:
                qas = answers if isinstance(answers, list) else ast.literal_eval(answers)
            except (ValueError, SyntaxError):
                continue
            for qa in qas:
                if isinstance(qa, dict) and qa.get("questionText"):
                    pair = pp.make_pair("wikiqa", qa["questionText"], passage, pp.answer_text(qa.get("answerText")))
                    if pair:
                        yield pair


def answer_index(pairs: list[dict], paths: dict[str, Path], stats: dict) -> dict[Key, str]:
    """(source, question, passage_id) -> answer, for the sources that occur in `pairs` and have a raw path."""
    sources = {p["source"] for p in pairs}
    index: dict[Key, str] = {}
    conflicts: dict[str, int] = {}
    if "turhistquad" in sources and paths.get("turhistquad"):
        train, test = pp.load_turhistquad(paths["turhistquad"], {})
        for pair in train + test:
            _add(index, conflicts, pair)
    if "wikirag" in sources and paths.get("wikirag"):
        for pair in pp.load_wikirag(paths["wikirag"], {}):
            _add(index, conflicts, pair)
    if "wikiqa" in sources and paths.get("wikiqa"):
        wanted = {p["passage_id"] for p in pairs if p["source"] == "wikiqa"}
        for pair in wikiqa_answers(paths["wikiqa"], wanted):
            _add(index, conflicts, pair)
    stats["answer_conflicts"] = conflicts
    return index


def attach(pairs: list[dict], index: dict[Key, str], forbidden: Optional[Iterable[str]], stats: dict) -> list[dict]:
    """Copies of `pairs` with `answer` set from `index` (and removed where contaminated). Order and pairs unchanged."""
    out = []
    per_source: dict[str, dict[str, int]] = {}
    for pair in pairs:
        row = dict(pair)
        row.pop("answer", None)
        answer = index.get(_key(pair))
        counts = per_source.setdefault(pair["source"], {"pairs": 0, "matched": 0})
        counts["pairs"] += 1
        if answer:
            row["answer"] = answer
            counts["matched"] += 1
        out.append(row)
    removed = 0
    if forbidden is not None:
        hits = pp.contaminated_texts([r["answer"] for r in out if "answer" in r], forbidden)
        for row in out:
            if row.get("answer") in hits:
                del row["answer"]
                removed += 1
    stats["by_source"] = per_source
    stats["answers_removed_decontamination"] = removed
    stats["with_answer"] = sum("answer" in r for r in out)
    stats["answer_length"] = length_summary(out)
    return out


def length_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Per source: answer count and the 10th / 50th / 90th percentile of the word and character counts."""

    def pct(values: list[int], q: float) -> float:
        values = sorted(values)
        return float(values[min(len(values) - 1, int(q * len(values)))]) if values else 0.0

    out = {}
    for source in sorted({r["source"] for r in rows}):
        answers = [r["answer"] for r in rows if r["source"] == source and "answer" in r]
        words = [len(a.split()) for a in answers]
        chars = [len(a) for a in answers]
        out[source] = {"n": len(answers), **{f"words_p{int(q * 100)}": pct(words, q) for q in (0.1, 0.5, 0.9)},
                       **{f"chars_p{int(q * 100)}": pct(chars, q) for q in (0.1, 0.5, 0.9)}}
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", required=True, help="pairs JSONL (prepare_pairs / mine_negatives output)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-dir", required=True, help="raw datasets, laid out as download_raw writes them")
    parser.add_argument("--download", action="store_true", help="download the raw datasets into --raw-dir first")
    parser.add_argument("--forbidden", nargs="*", default=[], help="JSONL files with a 'text' field (benchmark test texts)")
    args = parser.parse_args(argv)

    paths = download_raw(args.raw_dir) if args.download else raw_paths(args.raw_dir)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"raw data not found: {missing} (use --download)")
    pairs = read_jsonl(args.pairs)
    stats: dict = {"pairs_file": str(args.pairs), "pairs": len(pairs),
                   "raw_revisions": {s: v[1] for s, v in RAW_SOURCES.items()}, "forbidden": args.forbidden}
    index = answer_index(pairs, paths, stats)

    def forbidden_texts():
        for path in args.forbidden:
            yield from pp.JsonlTexts(path)

    rows = attach(pairs, index, forbidden_texts() if args.forbidden else None, stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    pp._write_jsonl(tmp, rows)
    tmp.replace(out)
    stats_path = out.with_name(out.stem + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
