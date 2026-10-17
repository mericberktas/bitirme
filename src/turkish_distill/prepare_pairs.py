"""Merge Turkish QA datasets into one (question, passage) format.

    python -m turkish_distill.prepare_pairs \
        --turhistquad-dir data/raw/turhistquad --wikirag-csv data/raw/wikirag/train.csv \
        --wikiqa-parquet data/raw/wikiqa/turkish-wiki-qa-4mil.parquet --out-dir data/stage1

Writes, in --out-dir:
    pairs_train.jsonl / pairs_val.jsonl   {id, source, question, passage, passage_id}  (contrastive training)
    corpus_train.jsonl                    {"text": ...}, unique questions and passages of the training
                                          pairs (distillation corpus, same format as prepare_corpus.py)
    stats.json                            counts per source and what was dropped

Rules that keep the comparison clean:
  * The validation pairs come from TurHistQuAD's test split plus a seeded slice of WikiRAG-TR; every
    training pair whose passage also occurs in the validation set is dropped (no passage leakage).
  * TurHistQuAD rows with SQuAD-style 24-hex ids are dropped (SQuAD translations could overlap with the
    SQuAD-TR retrieval test set of TR-MTEB); only the natively Turkish rows remain.
  * WikiRAG-TR negative rows (no answer) and rows whose `correct_intro_idx` is invalid are dropped.
  * Wikipedia-QA is sampled (reservoir) with a cap on questions per article, because several questions
    on one passage inside a batch would be false negatives for in-batch-negative losses.
Decontamination against TR-MTEB test sets is optional (--forbidden) and needs those texts exported first.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable, Optional

SQUAD_ID = re.compile(r"[0-9a-f]{24}")
Pair = dict


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def passage_id(passage: str) -> str:
    return hashlib.sha1(passage.encode("utf-8")).hexdigest()[:12]


def make_pair(source: str, question: str, passage: str) -> Optional[Pair]:
    question, passage = normalize(question), normalize(passage)
    if not question or not passage:
        return None
    return {"source": source, "question": question, "passage": passage, "passage_id": passage_id(passage)}


# --------------------------------------------------------------------------- #
# Loaders (one per dataset)
# --------------------------------------------------------------------------- #

def load_turhistquad(directory: str | Path, stats: dict) -> tuple[list[Pair], list[Pair]]:
    """(train, val) pairs; val is the dataset's own test split."""
    directory = Path(directory)
    out: dict[str, list[Pair]] = {}
    squad_dropped = 0
    for split in ("train", "test"):
        rows = json.loads((directory / f"{split}.json").read_text(encoding="utf-8"))
        pairs = []
        for row in rows:
            if SQUAD_ID.fullmatch(str(row.get("id", ""))):
                squad_dropped += 1
                continue
            pair = make_pair("turhistquad", row["question"], row["context"])
            if pair:
                pairs.append(pair)
        out[split] = pairs
    stats["turhistquad_squad_style_dropped"] = squad_dropped
    return out["train"], out["test"]


def load_wikirag(path: str | Path, stats: dict) -> list[Pair]:
    """Positive question -> correct-article pairs. Negative rows and rows with a bad index are dropped."""
    csv.field_size_limit(1 << 30)
    pairs: list[Pair] = []
    dropped = {"negative": 0, "bad_index": 0}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row["is_negative_response"]).strip() not in ("0", "False", "false"):
                dropped["negative"] += 1
                continue
            try:
                points = json.loads(row["ctx_split_points"])
                idx = int(row["correct_intro_idx"])
                context = row["context"]
                start = points[idx - 1] if idx > 0 else 0
                passage = context[start:points[idx]]
            except (ValueError, IndexError, KeyError, TypeError):
                dropped["bad_index"] += 1
                continue
            if idx < 0 or not passage.strip():
                dropped["bad_index"] += 1
                continue
            pair = make_pair("wikirag", row["question"], passage)
            if pair:
                pairs.append(pair)
    stats["wikirag_dropped"] = dropped
    return pairs


def load_wikiqa(
    path: str | Path,
    max_pairs: int,
    max_chars: int,
    min_chars: int,
    max_q_per_article: int,
    seed: int,
    stats: dict,
) -> list[Pair]:
    """Reservoir sample of at most `max_pairs` (question, whole article) pairs from the large parquet.

    Only articles of `min_chars` to `max_chars` characters are used: the generated answers are sentences,
    not spans, so for a long article we could not tell which chunk supports the question.
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    seen: set[int] = set()
    reservoir: list[Pair] = []
    seen_pairs = skipped_long = skipped_short = duplicate = parse_failed = 0
    for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=20_000, columns=["original_text", "answers"]):
        columns = batch.to_pydict()
        for text, answers in zip(columns["original_text"], columns["answers"]):
            passage = normalize(text or "")
            if len(passage) > max_chars:
                skipped_long += 1
                continue
            if len(passage) < min_chars:
                skipped_short += 1
                continue
            key = hash(passage)
            if key in seen:
                duplicate += 1
                continue
            seen.add(key)
            try:
                qas = answers if isinstance(answers, list) else ast.literal_eval(answers)
                questions = [normalize(q["questionText"]) for q in qas if q.get("questionText")]
            except (ValueError, SyntaxError, TypeError, AttributeError):
                parse_failed += 1
                continue
            questions = [q for q in questions if len(q) >= 10]
            if len(questions) > max_q_per_article:
                questions = rng.sample(questions, max_q_per_article)
            for question in questions:
                pair = {"source": "wikiqa", "question": question, "passage": passage, "passage_id": passage_id(passage)}
                if len(reservoir) < max_pairs:
                    reservoir.append(pair)
                else:
                    j = rng.randint(0, seen_pairs)
                    if j < max_pairs:
                        reservoir[j] = pair
                seen_pairs += 1
    stats["wikiqa"] = {
        "eligible_pairs_seen": seen_pairs, "kept": len(reservoir), "articles_too_long": skipped_long,
        "articles_too_short": skipped_short, "duplicate_articles": duplicate, "answers_unparsable": parse_failed,
    }
    return reservoir


# --------------------------------------------------------------------------- #
# Merging, decontamination and output
# --------------------------------------------------------------------------- #

def _ngram_hashes(text: str, n: int) -> set[int]:
    words = normalize(text).lower().split()
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def decontaminate(pairs: list[Pair], forbidden: Iterable[str], n: int = 13) -> tuple[list[Pair], int]:
    """Drop pairs whose passage or question equals / shares a word n-gram with a forbidden text.

    Texts shorter than n words (typical benchmark queries) can only match exactly after normalisation.
    """
    exact: set[str] = set()
    grams: set[int] = set()
    for text in forbidden:
        exact.add(normalize(text).lower())
        grams |= _ngram_hashes(text, n)

    def hit(text: str) -> bool:
        return normalize(text).lower() in exact or bool(_ngram_hashes(text, n) & grams)

    kept = [p for p in pairs if not hit(p["passage"]) and not hit(p["question"])]
    return kept, len(pairs) - len(kept)


def merge(
    train_parts: list[list[Pair]],
    val_parts: list[list[Pair]],
    wikirag_val_size: int = 0,
    wikirag_pairs: Optional[list[Pair]] = None,
    seed: int = 42,
    stats: Optional[dict] = None,
    max_q_per_passage: Optional[int] = None,
) -> tuple[list[Pair], list[Pair]]:
    """Combine sources; guarantee that no validation passage occurs in the training pairs.

    `max_q_per_passage` caps the training questions per passage (seeded random subset): some TurHistQuAD
    passages carry 100+ questions, which would be false negatives for each other inside one batch.
    """
    stats = stats if stats is not None else {}
    train = [p for part in train_parts for p in part]
    val = [p for part in val_parts for p in part]
    if wikirag_pairs:
        shuffled = list(wikirag_pairs)
        random.Random(seed).shuffle(shuffled)
        val += shuffled[:wikirag_val_size]
        train += shuffled[wikirag_val_size:]
    val_passages = {p["passage_id"] for p in val}
    before = len(train)
    train = [p for p in train if p["passage_id"] not in val_passages]
    stats["train_dropped_for_val_passage_overlap"] = before - len(train)

    def dedup(pairs: list[Pair]) -> list[Pair]:
        seen, out = set(), []
        for p in pairs:
            key = (p["question"].lower(), p["passage_id"])
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    train, val = dedup(train), dedup(val)
    if max_q_per_passage is not None:
        rng = random.Random(seed)
        by_passage: dict[str, list[int]] = {}
        for i, p in enumerate(train):
            by_passage.setdefault(p["passage_id"], []).append(i)
        keep = set(range(len(train)))
        for indices in by_passage.values():
            if len(indices) > max_q_per_passage:
                keep -= set(indices) - set(rng.sample(indices, max_q_per_passage))
        stats["train_dropped_for_passage_cap"] = len(train) - len(keep)
        train = [p for i, p in enumerate(train) if i in keep]
    counters: dict[str, int] = {}
    for split in (train, val):
        for p in split:
            counters[p["source"]] = counters.get(p["source"], 0) + 1
            p["id"] = f"{p['source']}:{counters[p['source']]:07d}"
    return train, val


def corpus_texts(pairs: list[Pair], exclude: Iterable[Pair] = ()) -> list[str]:
    """Unique questions and passages in order of first appearance, minus any text used by `exclude` pairs."""
    banned = {t for p in exclude for t in (p["question"], p["passage"])}
    seen: dict[str, None] = {}
    for p in pairs:
        for text in (p["question"], p["passage"]):
            if text not in banned:
                seen.setdefault(text, None)
    return list(seen)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turhistquad-dir", default=None)
    parser.add_argument("--wikirag-csv", default=None)
    parser.add_argument("--wikiqa-parquet", default=None)
    parser.add_argument("--wikiqa-pairs", type=int, default=25_000)
    parser.add_argument("--wikiqa-max-chars", type=int, default=2000)
    parser.add_argument("--wikiqa-min-chars", type=int, default=100)
    parser.add_argument("--max-q-per-article", type=int, default=3)
    parser.add_argument("--max-q-per-passage", type=int, default=10, help="cap on training questions per passage, all sources")
    parser.add_argument("--wikirag-val-size", type=int, default=500)
    parser.add_argument("--forbidden", default=None, help="JSONL with a 'text' field of benchmark test texts to exclude")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    stats: dict = {}
    train_parts: list[list[Pair]] = []
    val_parts: list[list[Pair]] = []
    wikirag_pairs = None
    if args.turhistquad_dir:
        tr, va = load_turhistquad(args.turhistquad_dir, stats)
        train_parts.append(tr)
        val_parts.append(va)
    if args.wikirag_csv:
        wikirag_pairs = load_wikirag(args.wikirag_csv, stats)
    if args.wikiqa_parquet:
        train_parts.append(load_wikiqa(args.wikiqa_parquet, args.wikiqa_pairs, args.wikiqa_max_chars,
                                       args.wikiqa_min_chars, args.max_q_per_article, args.seed, stats))
    train, val = merge(train_parts, val_parts, args.wikirag_val_size, wikirag_pairs, args.seed, stats,
                       max_q_per_passage=args.max_q_per_passage)
    if args.forbidden:
        forbidden = [json.loads(line)["text"] for line in open(args.forbidden, encoding="utf-8") if line.strip()]
        train, stats["train_dropped_decontamination"] = decontaminate(train, forbidden)
        val, stats["val_dropped_decontamination"] = decontaminate(val, forbidden)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = corpus_texts(train, exclude=val)
    stats["pairs_train"] = _write_jsonl(out / "pairs_train.jsonl", train)
    stats["pairs_val"] = _write_jsonl(out / "pairs_val.jsonl", val)
    stats["corpus_train_texts"] = _write_jsonl(out / "corpus_train.jsonl", ({"text": t} for t in texts))
    stats["train_by_source"] = {s: sum(p["source"] == s for p in train) for s in sorted({p["source"] for p in train})}
    stats["decontaminated"] = bool(args.forbidden)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
