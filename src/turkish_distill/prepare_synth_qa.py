"""Clean, decontaminate and split the synthetic FineWeb QA file into training pairs.

    python -m turkish_distill.prepare_synth_qa --csv data/extra/fineweb_qa/fineweb1_merged_dataset.csv \
        --forbidden data/forbidden/benchmark_texts_all.jsonl data/forbidden/rag_texts.jsonl \
        --out-dir data/extra/fineweb_qa/clean

The input is a one-column CSV (`Text`). Every row is a Turkish web document (FineWeb-2 style) followed by six blocks an
LLM generated for it, each introduced by a fixed Turkish header: a summary, keywords, sentiment, category, three
question/answer pairs and three search queries that should retrieve the document.

Cleaning (counters for every rule go to `stats.json`):
- the row must have the document and the summary block; the six blocks are cut at their headers;
- keywords are dropped (lists of one- or two-word tags: neither natural text nor a useful query next to the three
  generated search queries);
- documents: 200 to 50,000 characters, at most 5% of their letters in a non-Latin script, de-duplicated (whole
  normalised text and first 300 normalised characters);
- generated texts (summary, questions, answers, queries) are dropped one by one when they contain a non-Latin script
  (Cyrillic/CJK/Arabic/... leaks of the generator) or fall outside length bounds; a question also needs its answer;
- sentiment is mapped to pozitif / negatif / nötr (anything else: null); the category is kept as free text.

Decontamination: a document is dropped when it equals a forbidden text after normalisation or shares at least one
word 13-gram with one (the rule of prepare_pairs / decontaminate_corpus); a generated text is dropped when it does
(short texts can only match exactly). The forbidden files are JSONL with a `text` field: the TR-MTEB + Mizan test
texts (`benchmark_texts export`, its `.stats.json` names the datasets) and any other evaluation set (RAG corpora).

Outputs in `--out-dir`: `docs.jsonl` (one record per kept document), pairs in the `prepare_pairs` format
(`question`, `passage`, `passage_id`, `source`, optional `answer`) for search query -> document
(`pairs_query_*.jsonl`), question -> document with the answer (`pairs_qa_*.jsonl`) and summary -> document
(`pairs_summary_*.jsonl`), each split into train / val by document (`--val-fraction` of the documents), and
`stats.json`. The passages of the pair files are cut to `--max-passage-chars` (2,500) characters at a word boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from .decontaminate_corpus import BenchmarkIndex, _dataset_names_and_bounds
from .prepare_pairs import make_pair, normalize

HEADERS = {
    "summary": "Yukarıdaki metnin özeti nedir?",
    "keywords": "Yukarıdaki metnin anahtar kelimeleri nelerdir?",
    "sentiment": "Yukarıdaki metnin duygudurumu nedir?",
    "category": "Yukarıdaki metnin kategorisi nedir?",
    "qa": "Yukarıdaki metin ile ilgili 3 soru ve cevabı nelerdir?",
    "queries": "Yukarıdaki metni bulmayı sağlayacak 3 arama sorgusu nelerdir?",
}
# Scripts a Turkish text should not contain (generator leaks): Greek, Cyrillic, Armenian, Hebrew, Arabic, Devanagari,
# Thai, Hangul, CJK, kana.
NON_LATIN = re.compile(
    "[Ͱ-ϿЀ-ԯ԰-֏֐-׿؀-ۿऀ-ॿ฀-๿"
    "ᄀ-ᇿ぀-ヿ㐀-䶿一-鿿가-힯]"
)
LETTER = re.compile(r"[^\W\d_]")
SENTIMENTS = {"pozitif": "pozitif", "olumlu": "pozitif", "negatif": "negatif", "olumsuz": "negatif", "nötr": "nötr",
              "notr": "nötr", "nötral": "nötr", "tarafsız": "nötr"}
LIMITS = {"summary": (20, 800), "question": (8, 300), "answer": (2, 800), "query": (8, 300)}


def parse_record(text: str) -> Optional[dict]:
    """Cut one row into the document and its generated blocks (None if the summary header is missing)."""
    text = text.replace("\r\n", "\n")
    positions = sorted((text.find(h), key) for key, h in HEADERS.items() if text.find(h) >= 0)
    if not positions or positions[0][1] != "summary":
        return None
    out = {"doc": text[: positions[0][0]].strip()}
    for i, (pos, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        out[key] = text[pos + len(HEADERS[key]) : end].strip()
    return out


def split_qa(block: str) -> list[tuple[str, str]]:
    """'1. Soru: ... 1. Cevap: ...' blocks (questions and answers may sit on the next line)."""
    pairs = []
    for m in re.finditer(r"(\d+)\.\s*Soru\s*:\s*(.+?)\s*\1\.\s*Cevap\s*:\s*(.+?)(?=\s*\d+\.\s*Soru\s*:|\Z)", block, flags=re.S):
        pairs.append((normalize(m.group(2)), normalize(m.group(3))))
    return pairs


def split_list(block: str) -> list[str]:
    """Numbered lines '1. ...' of a block."""
    return [normalize(m.group(1)) for m in re.finditer(r"^\s*\d+[.)]\s*(.+)$", block, flags=re.M)]


def sentiment_label(value: str) -> Optional[str]:
    words = re.findall(r"\w+", value.lower())
    for w in words:
        if w in SENTIMENTS:
            return SENTIMENTS[w]
    return None


def non_latin_share(text: str) -> float:
    letters = LETTER.findall(text)
    return len(NON_LATIN.findall(text)) / len(letters) if letters else 0.0


def _ok(kind: str, text: str, counts: Counter) -> bool:
    lo, hi = LIMITS[kind]
    if not lo <= len(text) <= hi:
        counts[f"{kind}_length"] += 1
        return False
    if NON_LATIN.search(text):
        counts[f"{kind}_non_latin"] += 1
        return False
    return True


def clean_record(rec: dict, counts: Counter) -> Optional[dict]:
    """The cleaned record (keywords dropped), or None when the document itself is rejected."""
    doc = rec["doc"].strip()
    if not 200 <= len(doc) <= 50_000:
        counts["doc_length"] += 1
        return None
    if non_latin_share(doc) > 0.05:
        counts["doc_non_latin"] += 1
        return None
    summary = normalize(rec.get("summary", ""))
    out = {
        "doc": doc,
        "summary": summary if summary and _ok("summary", summary, counts) else None,
        "sentiment": sentiment_label(rec.get("sentiment", "")),
        "category": normalize(rec.get("category", ""))[:60] or None,
        "qa": [],
        "queries": [],
    }
    if out["sentiment"] is None:
        counts["sentiment_unmapped"] += 1
    seen = set()
    for q, a in split_qa(rec.get("qa", "")):
        if q.lower() in seen or q.lower() == a.lower():
            counts["question_duplicate"] += 1
            continue
        if _ok("question", q, counts) and _ok("answer", a, counts):
            out["qa"].append({"question": q, "answer": a})
            seen.add(q.lower())
    for query in split_list(rec.get("queries", "")):
        if query.lower() in seen:
            counts["query_duplicate"] += 1
            continue
        if _ok("query", query, counts):
            out["queries"].append(query)
            seen.add(query.lower())
    return out


def iter_rows(path: str | Path) -> Iterator[str]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header "Text"
        for row in reader:
            if row and row[0].strip():
                yield row[0]


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_index(paths: list[str | Path], n: int = 13) -> tuple[BenchmarkIndex, list[str]]:
    """One n-gram index over several forbidden JSONL files; dataset names from their .stats.json or file names."""
    names, bounds, offset = [], [], 0
    for path in map(Path, paths):
        file_names, file_bounds = _dataset_names_and_bounds(path)
        if file_names == ["benchmark"]:
            file_names = [path.stem]
        names += file_names
        bounds += [offset + b for b in file_bounds]
        offset += _count_lines(path)

    def texts():
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)["text"]

    return BenchmarkIndex(texts(), bounds, n), names


def truncate_passage(text: str, max_chars: int) -> str:
    """Cut a (normalised) passage to at most `max_chars` characters at a word boundary (0 = keep it whole). The
    student reads at most 512 tokens (~2,000-2,500 characters of Turkish), so longer passages only make the pair files
    large; the full documents stay in docs.jsonl."""
    text = normalize(text)
    if not max_chars or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > max_chars // 2 else cut


def doc_id(doc: str) -> str:
    return hashlib.sha1(normalize(doc).encode("utf-8")).hexdigest()[:12]


def is_val(identifier: str, fraction: float) -> bool:
    return int(identifier[:8], 16) / 0xFFFFFFFF < fraction


def prepare(
    csv_path: str | Path,
    forbidden: list[str | Path],
    out_dir: str | Path,
    val_fraction: float = 0.01,
    limit: Optional[int] = None,
    max_passage_chars: int = 2500,
    progress_every: int = 50_000,
) -> dict:
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index, names = build_index(forbidden) if forbidden else (None, [])
    if index is not None:
        print(f"forbidden index: {len(index.grams):,} 13-grams, {len(index.exact):,} texts, {len(names)} datasets "
              f"({time.time() - started:.0f}s)", flush=True)
    counts: Counter = Counter()
    removed_by: Counter = Counter()
    seen_full, seen_head = set(), set()
    files = {}
    for kind in ("query", "qa", "summary"):
        for split in ("train", "val"):
            files[(kind, split)] = open(out_dir / f"pairs_{kind}_{split}.jsonl.tmp", "w", encoding="utf-8")
    docs_out = open(out_dir / "docs.jsonl.tmp", "w", encoding="utf-8")
    sentiment, categories = Counter(), Counter()

    def contaminated(text: str) -> Optional[str]:
        if index is None:
            return None
        hits, dataset, exact = index.check(text)
        if exact or hits:
            return names[dataset if dataset is not None else 0]
        return None

    try:
        for row_no, text in enumerate(iter_rows(csv_path)):
            if limit is not None and row_no >= limit:
                break
            counts["rows"] += 1
            rec = parse_record(text)
            if rec is None:
                counts["unparsed"] += 1
                continue
            rec = clean_record(rec, counts)
            if rec is None:
                continue
            norm = normalize(rec["doc"]).lower()
            if norm in seen_full or norm[:300] in seen_head:
                counts["doc_duplicate"] += 1
                continue
            seen_full.add(norm)
            seen_head.add(norm[:300])
            hit = contaminated(rec["doc"])
            if hit:
                counts["doc_contaminated"] += 1
                removed_by[hit] += 1
                continue
            if rec["summary"] and contaminated(rec["summary"]):
                counts["summary_contaminated"] += 1
                rec["summary"] = None
            kept_qa = []
            for item in rec["qa"]:
                if contaminated(item["question"]) or contaminated(item["answer"]):
                    counts["qa_contaminated"] += 1
                else:
                    kept_qa.append(item)
            rec["qa"] = kept_qa
            kept_q = []
            for query in rec["queries"]:
                if contaminated(query):
                    counts["query_contaminated"] += 1
                else:
                    kept_q.append(query)
            rec["queries"] = kept_q
            identifier = doc_id(rec["doc"])
            passage = truncate_passage(rec["doc"], max_passage_chars)
            split = "val" if is_val(identifier, val_fraction) else "train"
            rec = {"id": identifier, "split": split, **rec}
            docs_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[f"docs_{split}"] += 1
            sentiment[rec["sentiment"]] += 1
            if rec["category"]:
                categories[rec["category"].lower()] += 1
            for query in rec["queries"]:
                pair = make_pair("fwqa_query", query, passage)
                files[("query", split)].write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts[f"pairs_query_{split}"] += 1
            for item in rec["qa"]:
                pair = make_pair("fwqa_qa", item["question"], passage, item["answer"])
                files[("qa", split)].write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts[f"pairs_qa_{split}"] += 1
            if rec["summary"]:
                pair = make_pair("fwqa_summary", rec["summary"], passage)
                files[("summary", split)].write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts[f"pairs_summary_{split}"] += 1
            if progress_every and counts["rows"] % progress_every == 0:
                print(f"  {counts['rows']:,} rows, {counts['docs_train'] + counts['docs_val']:,} documents kept "
                      f"({time.time() - started:.0f}s)", flush=True)
    finally:
        for f in [docs_out, *files.values()]:
            f.close()
    for path in out_dir.glob("*.jsonl.tmp"):
        path.replace(path.with_suffix(""))
    stats = {
        "csv": str(csv_path), "forbidden": [str(p) for p in forbidden], "val_fraction": val_fraction,
        "max_passage_chars": max_passage_chars,
        "counts": dict(sorted(counts.items())),
        "docs_removed_by_first_matching_dataset": dict(removed_by.most_common()),
        "sentiment": {str(k): v for k, v in sentiment.most_common()},
        "categories_distinct": len(categories), "categories_top": dict(categories.most_common(40)),
        "seconds": round(time.time() - started),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Clean and decontaminate the synthetic FineWeb QA CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--forbidden", nargs="*", default=[], help="JSONL files with a `text` field")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    parser.add_argument("--max-passage-chars", type=int, default=2500, help="passages in the pair files (0 = whole)")
    args = parser.parse_args(argv)
    stats = prepare(args.csv, args.forbidden, args.out_dir, args.val_fraction, args.limit, args.max_passage_chars)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
