"""Convert the downloaded candidate datasets into the JSONL files that `build_mix` reads.

    python -m turkish_distill.prepare_extra --raw data/extra --out data/extra/prepared

Every output row is {"text", "url"}: a real URL where the dataset has one, otherwise a pseudo URL
`https://<domain>/` built from the page's domain, so `build_mix` can report (and cap) domains. Spam, boilerplate and
page-furniture filters are NOT applied here: `build_mix` runs `clean_reason` on every bucket. This module only does
what the raw formats need: parse the nested structures, drop obviously unusable sources, and cap the rows per domain
(the Q&A and news corpora are dominated by a few sites: the first ten domains hold 53-65% of the MQA questions, and
three of 18 news sites hold more than 65% of the news).

Outputs (one file per source of configs/data_mix_v2.yaml):
    wikipedia.jsonl         full article text of Turkish Wikipedia (paragraphs are split later)
    wikipedia_leads.jsonl   the first paragraph of every article (150-1,200 characters)
    news.jsonl              news-tr-1.8M articles, a random ~12% and at most `news_cap` per site
    news_titles.jsonl       MLSUM + XL-Sum headlines, at most `title_cap` (50,000) per site: MLSUM is one newspaper
    tapaco.jsonl            TaPaCo sentences of 30+ characters
    mqa_questions.jsonl     MQA community Q&A questions, without the excluded sites, at most `qa_cap` per site
    mqa_answers.jsonl       the first answer of those questions
    mfaq_questions.jsonl    MFAQ questions, at most `faq_cap` per site
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import urlparse

# Health and personal-advice sites: medical questions of private people (personal data, medical claims) and
# a girls' forum with intimate topics. They are 37% of the MQA community questions and are left out on purpose.
EXCLUDED_DOMAINS = {"doktorsitesi.com", "kizlarsoruyor.com"}


def domain_of(url: Optional[str]) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


class DomainCap:
    """Allow at most `cap` rows per domain (first come, first served)."""

    def __init__(self, cap: int):
        self.cap, self.counts = cap, collections.Counter()

    def allow(self, domain: str) -> bool:
        if self.counts[domain] >= self.cap:
            return False
        self.counts[domain] += 1
        return True


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def parquet_rows(paths: list[Path], columns: list[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
            cols = batch.to_pydict()
            for i in range(batch.num_rows):
                yield {c: cols[c][i] for c in columns}


def pseudo_url(domain: str) -> str:
    return f"https://{domain}/"


_EMPTY_PARENS = re.compile(r"\s*\(\s*\)")


def tidy(text: str) -> str:
    """Two artefacts of the raw data: a combining dot above (U+0307) left behind where a dotted capital I was
    lower-cased (MQA: "i" + U+0307 + "stiyorum"), and empty parentheses where a pronunciation or non-Latin
    spelling was removed (Wikipedia: "... ilcesi ()")."""
    return _EMPTY_PARENS.sub("", text.replace("̇", ""))


# --------------------------------------------------------------------------- #
# One converter per source
# --------------------------------------------------------------------------- #

def wikipedia(raw: Path, out: Path) -> dict[str, int]:
    paths = sorted((raw / "wikipedia_tr").rglob("*.parquet"))

    def articles() -> Iterator[dict]:
        for row in parquet_rows(paths, ["url", "text"]):
            if row["text"] and row["text"].strip():
                yield {"text": tidy(row["text"]), "url": row["url"]}

    def leads() -> Iterator[dict]:
        for row in parquet_rows(paths, ["url", "text"]):
            lines = [line.strip() for line in tidy(row["text"] or "").split("\n") if line.strip()]
            if lines and 150 <= len(lines[0]) <= 1200:
                yield {"text": lines[0], "url": row["url"]}

    return {"wikipedia": write_jsonl(articles(), out / "wikipedia.jsonl"),
            "wikipedia_leads": write_jsonl(leads(), out / "wikipedia_leads.jsonl")}


def news(raw: Path, out: Path, news_cap: int = 20_000, keep: float = 0.12, seed: int = 0) -> int:
    rng, cap = random.Random(seed), DomainCap(news_cap)

    def rows() -> Iterator[dict]:
        for row in parquet_rows(sorted((raw / "news_tr_1_8m").rglob("*.parquet")), ["text", "url"]):
            if rng.random() < keep and row["text"] and cap.allow(domain_of(row["url"])):
                yield {"text": row["text"], "url": row["url"]}

    return write_jsonl(rows(), out / "news.jsonl")


def news_titles(raw: Path, out: Path, title_cap: int = 50_000) -> int:
    cap = DomainCap(title_cap)

    def rows() -> Iterator[dict]:
        for row in parquet_rows(sorted((raw / "mlsum_tu").rglob("train/*.parquet")) + sorted((raw / "mlsum_tu").rglob("validation/*.parquet")), ["title", "url"]):
            if row["title"] and cap.allow(domain_of(row["url"])):
                yield {"text": row["title"].strip(), "url": row["url"]}
        xlsum = raw / "xlsum_tr" / "extracted"
        for name in ("turkish_train.jsonl", "turkish_val.jsonl"):
            path = xlsum / name
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("title") and cap.allow("bbc.com/turkce"):
                            yield {"text": r["title"].strip(), "url": r.get("url") or "https://www.bbc.com/turkce"}

    return write_jsonl(rows(), out / "news_titles.jsonl")


def tapaco(raw: Path, out: Path, min_chars: int = 30) -> int:
    rows = ({"text": r["paraphrase"].strip()} for r in parquet_rows(sorted((raw / "tapaco_tr").rglob("*.parquet")), ["paraphrase"])
            if r["paraphrase"] and len(r["paraphrase"].strip()) >= min_chars)
    return write_jsonl(rows, out / "tapaco.jsonl")


def mqa(raw: Path, out: Path, qa_cap: int = 12_000) -> dict[str, int]:
    """The community Q&A file: one row per question with its first answer, without excluded sites."""
    q_cap, a_cap = DomainCap(qa_cap), DomainCap(qa_cap)
    questions, answers = [], []
    with gzip.open(raw / "mqa_tr" / "data" / "data.tr.cqa.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            domain = (page.get("domain") or "").lower()
            domain = domain[4:] if domain.startswith("www.") else domain
            if not domain or domain in EXCLUDED_DOMAINS:
                continue
            for q in page.get("questions", []):
                name = tidy((q.get("name") or q.get("text") or "")).strip()
                first = next((tidy(a["text"]) for a in q.get("answers", []) if a.get("text")), None)
                if name and first and q_cap.allow(domain):
                    questions.append({"text": name, "url": pseudo_url(domain)})
                if first and a_cap.allow(domain):
                    answers.append({"text": first.strip(), "url": pseudo_url(domain)})
    return {"mqa_questions": write_jsonl(questions, out / "mqa_questions.jsonl"),
            "mqa_answers": write_jsonl(answers, out / "mqa_answers.jsonl")}


def mfaq(raw: Path, out: Path, faq_cap: int = 4_000) -> int:
    cap = DomainCap(faq_cap)

    def rows() -> Iterator[dict]:
        with open(raw / "mfaq_tr" / "data" / "tr" / "train.jsonl", encoding="utf-8") as f:
            for line in f:
                page = json.loads(line)
                domain = (page.get("domain") or "").lower()
                domain = domain[4:] if domain.startswith("www.") else domain
                for pair in page.get("qa_pairs", []):
                    question = tidy(pair.get("question") or "").strip()
                    if question and domain not in EXCLUDED_DOMAINS and cap.allow(domain):
                        yield {"text": question, "url": pseudo_url(domain)}

    return write_jsonl(rows(), out / "mfaq_questions.jsonl")


def extract_xlsum(raw: Path) -> None:
    """The XL-Sum archive holds three JSONL files; extract them once (data filter: no links, no absolute paths)."""
    dest = raw / "xlsum_tr" / "extracted"
    archive = raw / "xlsum_tr" / "data" / "turkish_XLSum_v2.0.tar.bz2"
    if archive.exists() and not (dest / "turkish_train.jsonl").exists():
        dest.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert the downloaded datasets into build_mix sources")
    parser.add_argument("--raw", default="data/extra")
    parser.add_argument("--out", default="data/extra/prepared")
    parser.add_argument("--only", nargs="*", default=None, help="subset of: wikipedia news news_titles tapaco mqa mfaq")
    args = parser.parse_args(argv)
    raw, out = Path(args.raw), Path(args.out)
    steps = {"tapaco": tapaco, "mfaq": mfaq, "news_titles": lambda r, o: (extract_xlsum(r), news_titles(r, o))[1],
             "mqa": mqa, "wikipedia": wikipedia, "news": news}
    for name in args.only or list(steps):
        print(name, steps[name](raw, out), flush=True)


if __name__ == "__main__":
    main()
