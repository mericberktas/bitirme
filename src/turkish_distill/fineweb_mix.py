"""Mix v3: the 200K distillation mix (v2) plus texts taken from the cleaned synthetic FineWeb QA data.

    python -m turkish_distill.fineweb_mix --base data/mix/mix_v2_200k.jsonl \
        --base-questions data/mix/mix_v2_200k_questions.jsonl \
        --docs data/extra/fineweb_qa/clean/docs.jsonl \
        --out data/mix/mix_v3_400k.jsonl --questions-out data/mix/mix_v3_400k_questions.jsonl

Distillation needs no labels: every Turkish text with teacher vectors is a training example. So the FineWeb file is
used for its texts, not its pairs: paragraph and long chunks of the web documents, and the short generated texts
(search queries, questions, answers, summaries). The short ones fill a gap of the mix (benchmark texts are mostly
short: NLI/STS medians ~50 characters). The selection is a hypothesis, like the v2 shares.

Layout of the output: the first `len(base) - tail` base texts, then the FineWeb texts, then the last `tail` base texts.
The base mix's tail is the validation split of training (`data.val_size` = 2000 last texts), so the in-run diagnostics
stay comparable to the 200K runs, and the base prefix keeps its order, so the teacher cache shards of the 200K mix
that fall inside it can be copied instead of re-encoded (`encode_teachers --seed-from-corpus`).

Only documents of the FineWeb train split are used, each document gives at most one text per kind, the texts pass
`build_mix.clean_reason` and are deduplicated against the base mix and each other. The data was decontaminated against
TR-MTEB + Mizan + the two RAG sets by `prepare_synth_qa` (13-gram); chunks of a clean document are clean.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterator, Optional

from .build_mix import clean_reason, normalize, split_lines

# kind: (share of the FineWeb texts, min chars, max chars). Shares are a hypothesis.
KINDS: dict[str, tuple[float, int, int]] = {
    "fw_paragraphs": (0.30, 150, 600),
    "fw_long": (0.15, 800, 1500),
    "fw_queries": (0.15, 15, 200),
    "fw_questions": (0.15, 15, 200),
    "fw_answers": (0.125, 20, 400),
    "fw_summaries": (0.125, 40, 600),
}
QUESTION_KINDS = ("fw_queries", "fw_questions")  # get the 8K teacher's query instruction


def _read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def candidates(doc: dict, rng: random.Random) -> dict[str, str]:
    """At most one text per kind from one FineWeb document (chosen at random among the ones that fit)."""
    out: dict[str, list[str]] = {k: [] for k in KINDS}
    for line in split_lines(doc.get("doc", "")):
        for kind in ("fw_paragraphs", "fw_long"):
            if KINDS[kind][1] <= len(line) <= KINDS[kind][2]:
                out[kind].append(line)
    out["fw_queries"] = [normalize(q) for q in doc.get("queries", [])]
    out["fw_questions"] = [normalize(x["question"]) for x in doc.get("qa", []) if x.get("question")]
    out["fw_answers"] = [normalize(x["answer"]) for x in doc.get("qa", []) if x.get("answer")]
    if doc.get("summary"):
        out["fw_summaries"] = [normalize(doc["summary"])]
    picked = {}
    for kind, texts in out.items():
        lo, hi = KINDS[kind][1], KINDS[kind][2]
        texts = [t for t in texts if lo <= len(t) <= hi and clean_reason(t) is None]
        if texts:
            picked[kind] = rng.choice(texts)
    return picked


def quotas(n_total: int) -> dict[str, int]:
    q = {k: int(round(share * n_total)) for k, (share, _, _) in KINDS.items()}
    q[next(iter(q))] += n_total - sum(q.values())  # rounding remainder to the first kind
    return q


def select_fineweb(docs_path: str | Path, n_total: int, exclude: set[str], seed: int = 42) -> tuple[list[dict], dict]:
    """`n_total` texts (dicts with text, bucket, doc_id) from the train documents, in a seeded random document order."""
    docs = [d for d in _read_jsonl(docs_path) if d.get("split", "train") == "train"]
    rng = random.Random(seed)
    rng.shuffle(docs)
    want = quotas(n_total)
    got: dict[str, list[dict]] = {k: [] for k in KINDS}
    seen = set(exclude)
    for doc in docs:
        if all(len(got[k]) >= want[k] for k in KINDS):
            break
        for kind, text in candidates(doc, rng).items():
            if len(got[kind]) < want[kind] and text not in seen:
                seen.add(text)
                got[kind].append({"text": text, "bucket": kind, "source": "fineweb_qa", "doc_id": doc.get("id")})
    short = {k: want[k] - len(got[k]) for k in KINDS if len(got[k]) < want[k]}
    if short:
        raise ValueError(f"FineWeb ran out of texts for {short} (documents: {len(docs)})")
    rows = [r for k in KINDS for r in got[k]]
    rng.shuffle(rows)
    stats = {"documents": len(docs), "quotas": want, "selected": {k: len(v) for k, v in got.items()}}
    return rows, stats


def _write_jsonl(path: Path, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_v3(
    base: str | Path,
    docs: str | Path,
    out: str | Path,
    questions_out: str | Path,
    base_questions: Optional[str | Path] = None,
    n_fineweb: int = 200_000,
    tail: int = 2000,
    seed: int = 42,
) -> dict:
    base_rows = list(_read_jsonl(base))
    base_texts = {r["text"].strip() for r in base_rows}
    fw_rows, stats = select_fineweb(docs, n_fineweb, base_texts, seed)
    rows = base_rows[: len(base_rows) - tail] + fw_rows + base_rows[len(base_rows) - tail:]
    out, questions_out = Path(out), Path(questions_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, rows)
    questions = [r["question"] for r in _read_jsonl(base_questions)] if base_questions else []
    questions += [r["text"] for r in fw_rows if r["bucket"] in QUESTION_KINDS]
    _write_jsonl(questions_out, ({"question": q} for q in questions))
    stats.update({
        "base": str(base), "base_texts": len(base_rows), "tail": tail, "fineweb_texts": len(fw_rows),
        "total": len(rows), "questions": len(questions), "seed": seed,
        "fineweb_chars_median": {
            k: sorted(len(r["text"]) for r in fw_rows if r["bucket"] == k)[stats["selected"][k] // 2] for k in KINDS
        },
    })
    with open(out.with_suffix(".mix.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Mix v3: the 200K mix plus FineWeb QA texts")
    p.add_argument("--base", required=True)
    p.add_argument("--base-questions", default=None)
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--questions-out", required=True)
    p.add_argument("--n-fineweb", type=int, default=200_000)
    p.add_argument("--tail", type=int, default=2000, help="base texts kept at the end (the training validation split)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    stats = build_v3(a.base, a.docs, a.out, a.questions_out, a.base_questions, a.n_fineweb, a.tail, a.seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
