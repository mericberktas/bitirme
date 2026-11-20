"""Domain mix: the 200K distillation mix plus Turkish scientific and tech/gaming forum texts.

    python -m turkish_distill.domain_mix --base data/mix/mix_v2_200k.jsonl \
        --theses data/extra/domain/theses/tezler.parquet \
        --forum data/extra/domain/forum/technopatsosyal data/extra/domain/forum/donanimhaber ... \
        --forbidden data/forbidden/benchmark_texts_all.jsonl data/forbidden/rag_texts.jsonl \
        --out data/mix/mix_domain_250k.jsonl

Why: the student is far below Qwen3-8B / voyage-4-nano on SciFact-TR, SCIDOCS-TR and CQADupstack-Gaming-TR, and a
voyage-only student did not inherit voyage's SciFact strength (69.9 against the teacher's 86.4) although it imitated
voyage well: a teacher's knowledge only reaches the student through the texts it is shown, and the mix has almost no
scientific or forum Q&A text. This mix adds such text (no labels: the teachers encode it).

Sources (both native Turkish, checked on their Hugging Face cards, 20 Nov 2026):
- Turkish academic theses abstracts (`umutertugrul/turkish-academic-theses-dataset`, YÖK thesis centre, CC-BY-4.0,
  gated): abstracts and titles by subject group: life/health sciences 50% (SciFact, NFCorpus), economics/finance
  20% (FiQA; no ready native Turkish finance corpus exists), other fields 30% (SCIDOCS). Neither is a translation of the BEIR sources (S2ORC, Semantic Scholar, PubMed).
- ForumSohbetleri (`turkish-nlp-suite/ForumSohbetleri`, CC-BY-SA-4.0): tech and gaming forum threads; per thread one
  question-like post (contains "?") and one other post. Not StackExchange, so no overlap with the CQADupstack / FiQA sources.

Every text passes `build_mix.clean_reason`, is deduplicated against the base mix and the other new texts, and is
dropped if it shares a 13-gram with (or equals) a text of the forbidden lists (TR-MTEB + Mizan tests, RAG sets).
Forum signatures (hardware lists, slogans, link lists) are stripped first: a word tail that recurs in at least
`min_tail_count` posts of the sampled pool is treated as a signature.

Layout as in mix v3: the first `len(base) - tail` base texts, the new texts, then the last `tail` base texts (the
training validation split), so the base mix's teacher cache shards can be copied (`encode_teachers
--seed-from-corpus`).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .build_mix import clean_reason, normalize, split_sentences
from .fineweb_mix import _read_jsonl, _write_jsonl

# kind: (share of the new texts, min chars, max chars). Shares are a hypothesis.
KINDS: dict[str, tuple[float, int, int]] = {
    "thesis_abstracts": (0.50, 300, 2000),
    "thesis_titles": (0.10, 25, 250),
    "forum_questions": (0.20, 40, 1500),
    "forum_replies": (0.20, 40, 1500),
}
# Forum furniture: merge notices, edit stamps, attachment buttons.
_FURNITURE = re.compile(r"Mesaj otomatik birleştirildi:?\s*\d{1,2}\s+\w+\s+\d{4}|"
                        r"Son düzenleyen(?: moderatör)?:?\s*(?:[\w.\-]+:\s*)?\d{1,2}\s+\w+\s+\d{4}|"
                        r"Dosya Ekle|Genişletmek için tıkla\.*", re.IGNORECASE)
# Thesis subject groups, matched on the English half of the subject field ("Tıp = Medicine; ..."). SciFact and
# NFCorpus are biomedical, FiQA is finance, SCIDOCS is mostly computer science / natural science (the "other" group).
LIFE_KEYS = ("medic", "health", "disease", "biolog", "biochem", "nursing", "dentist", "pharma", "microbio",
             "psychiatr", "genetic", "veterinar", "nutrition", "dietetic", "surgery", "obstetric", "gynecol",
             "cardiol", "oncolog", "neurol", "immunol", "physiolog", "anatom", "radiolog", "anesthes", "pediatr",
             "dermatol", "urolog", "orthop", "otorhinolaryn", "gastro", "endocrin", "hematol", "infectious",
             "patholog", "biotech", "bioengineering", "molecular", "histolog", "emergency", "chest", "nephrol",
             "rheumatol", "geriatr", "parasitol", "biophysic", "biostatist", "midwifery", "eye diseases",
             "physiotherapy", "rehabilitation", "toxicol", "hospitals", "forensic", "public health", "allergy")
FINANCE_KEYS = ("finance", "banking", "economics", "econometrics", "insurance", "actuarial", "international trade",
                "business administration")
GROUP_SHARES = {"life": 0.5, "finance": 0.2, "other": 0.3}


def quotas(n_total: int) -> dict[str, int]:
    q = {k: int(round(share * n_total)) for k, (share, _, _) in KINDS.items()}
    q[next(iter(q))] += n_total - sum(q.values())
    return q


def cut_at_sentence(text: str, max_chars: int) -> str:
    """The longest prefix made of whole sentences that fits `max_chars` (the text itself if it fits)."""
    if len(text) <= max_chars:
        return text
    out = ""
    for sentence in split_sentences(text):
        candidate = f"{out} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        out = candidate
    return out


def _fits(kind: str, text: str) -> bool:
    lo, hi = KINDS[kind][1], KINDS[kind][2]
    return lo <= len(text) <= hi and clean_reason(text) is None


# --- theses -------------------------------------------------------------------------------------------------

def subject_group(subject: Optional[str]) -> str:
    """"life", "finance" or "other" from a thesis subject field (several subjects joined by ";")."""
    english = " ".join(part.split("=")[-1] for part in (subject or "").lower().split(";"))
    if any(k in english for k in LIFE_KEYS):
        return "life"
    if any(k in english for k in FINANCE_KEYS):
        return "finance"
    return "other"


_GLUED_SENTENCE = re.compile(r"([a-zçğıöşü0-9][.!?;:])([A-ZÇĞİÖŞÜ])")


def thesis_candidates(rows: Iterable[dict], rng: random.Random) -> Iterator[tuple[str, str, str]]:
    """(kind, text, subject group) for each usable abstract and title. `rows` have abstract_tr, title_tr, subject.
    Abstracts often lack the space after a full stop ("planlandı.Araştırmanın"); it is restored first."""
    for row in rows:
        group = subject_group(row.get("subject"))
        abstract = normalize(_GLUED_SENTENCE.sub(r"\1 \2", row.get("abstract_tr") or ""))
        if abstract:
            abstract = cut_at_sentence(abstract, KINDS["thesis_abstracts"][2])
            if _fits("thesis_abstracts", abstract):
                yield "thesis_abstracts", abstract, group
        title = normalize(row.get("title_tr") or "")
        if title and _fits("thesis_titles", title):
            yield "thesis_titles", title, group


def read_theses(path: str | Path, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["title_tr", "abstract_tr", "subject"])
    rows = table.to_pylist()
    random.Random(seed).shuffle(rows)
    return rows


# --- forum ----------------------------------------------------------------------------------------------------

def frequent_tails(posts: Iterable[str], min_count: int, k_min: int = 6, k_max: int = 40) -> set[str]:
    """Word tails (k_min..k_max words, step 2) that end at least `min_count` posts: signatures."""
    cnt: Counter = Counter()
    for post in posts:
        words = post.split()
        for k in range(k_min, min(k_max, len(words) - 1) + 1, 2):
            cnt[" ".join(words[-k:])] += 1
    return {s for s, c in cnt.items() if c >= min_count}


def strip_post(post: str, tails: set[str], k_min: int = 6, k_max: int = 40) -> str:
    post = _FURNITURE.sub(" ", post)
    words = post.split()
    for k in range(min(k_max, len(words) - 1), k_min - 1, -1):
        if (k - k_min) % 2 == 0 and " ".join(words[-k:]) in tails:
            words = words[:-k]
            break
    return normalize(" ".join(words))


def iter_threads(dirs: Iterable[str | Path]) -> Iterator[dict]:
    for d in dirs:
        for path in sorted(Path(d).glob("*.jsonl")):
            yield from _read_jsonl(path)


def sample_threads(dirs: list[str | Path], n: int, seed: int) -> list[dict]:
    """A uniform random sample of `n` threads over all forum files (reservoir; one pass, bounded memory)."""
    rng = random.Random(seed)
    sample: list[dict] = []
    for i, thread in enumerate(iter_threads(dirs)):
        if len(sample) < n:
            sample.append(thread)
        else:
            j = rng.randint(0, i)
            if j < n:
                sample[j] = thread
    rng.shuffle(sample)
    return sample


def forum_candidates(threads: list[dict], rng: random.Random, min_tail_count: int = 8) -> Iterator[tuple[str, str]]:
    """(kind, text): per thread at most one question-like post (it contains "?") and one other post, signatures and
    furniture stripped. The post order of the scraped threads is not reliable (the first post is often a reply), so
    the split is by form, not by position."""
    tails = frequent_tails((p for t in threads for p in t.get("texts", [])), min_tail_count)
    for thread in threads:
        posts = [strip_post(p, tails) for p in thread.get("texts", [])]
        questions = [p for p in posts if "?" in p and _fits("forum_questions", p)]
        others = [p for p in posts if "?" not in p and _fits("forum_replies", p)]
        if questions:
            yield "forum_questions", rng.choice(questions)
        if others:
            yield "forum_replies", rng.choice(others)


# --- assembly -------------------------------------------------------------------------------------------------

def select(candidates: Iterable[tuple], want: dict[str, int], exclude: set[str], index=None,
           group_shares: Optional[dict[str, float]] = None) -> tuple[dict[str, list[str]], Counter]:
    """Fill the quotas from (kind, text[, group]) candidates in order. With `group_shares`, each thesis kind takes
    those shares of its quota from the subject groups (the rounding remainder goes to the first group)."""
    got: dict[str, list[str]] = {k: [] for k in want}
    split: dict[tuple[str, str], int] = {}
    if group_shares:
        for k in want:
            if k.startswith("thesis"):
                parts = {g: int(round(want[k] * share)) for g, share in group_shares.items()}
                parts[next(iter(parts))] += want[k] - sum(parts.values())
                split.update({(k, g): n for g, n in parts.items()})
    used: Counter = Counter()
    drops: Counter = Counter()
    seen = set(exclude)
    for cand in candidates:
        kind, text = cand[0], cand[1]
        if kind not in want or len(got[kind]) >= want[kind]:
            continue
        key = (kind, cand[2]) if len(cand) > 2 and (kind, cand[2]) in split else None
        if key is not None and used[key] >= split[key]:
            continue
        if text in seen:
            drops["duplicate"] += 1
            continue
        if index is not None:
            shared, _, equal = index.check(text)
            if equal or shared >= 1:
                drops["benchmark_overlap"] += 1
                continue
        seen.add(text)
        got[kind].append(text)
        if key is not None:
            used[key] += 1
        if all(len(got[k]) >= want[k] for k in want):
            break
    return got, drops


def build_domain_mix(
    base: str | Path,
    theses: str | Path,
    forum_dirs: list[str | Path],
    out: str | Path,
    forbidden: Optional[list[str | Path]] = None,
    n_new: int = 50_000,
    tail: int = 2000,
    group_shares: Optional[dict[str, float]] = None,
    forum_threads: int = 60_000,
    min_tail_count: int = 8,
    seed: int = 42,
) -> dict:
    base_rows = list(_read_jsonl(base))
    exclude = {r["text"].strip() for r in base_rows}
    index = None
    if forbidden:
        from .prepare_synth_qa import build_index

        index, _ = build_index(list(forbidden))
    want = quotas(n_new)
    rng = random.Random(seed)
    thesis_want = {k: v for k, v in want.items() if k.startswith("thesis")}
    forum_want = {k: v for k, v in want.items() if k.startswith("forum")}
    group_shares = GROUP_SHARES if group_shares is None else group_shares
    got_t, drops_t = select(thesis_candidates(read_theses(theses, seed), rng), thesis_want, exclude, index, group_shares)
    exclude |= {t for v in got_t.values() for t in v}
    threads = sample_threads(forum_dirs, forum_threads, seed)
    got_f, drops_f = select(forum_candidates(threads, rng, min_tail_count), forum_want, exclude, index)
    got = {**got_t, **got_f}
    short = {k: want[k] - len(got[k]) for k in want if len(got[k]) < want[k]}
    if short:
        raise ValueError(f"ran out of texts for {short} (raise forum_threads or lower n_new)")
    new_rows = [{"text": t, "bucket": k, "source": "theses" if k.startswith("thesis") else "forum"}
                for k in KINDS for t in got[k]]
    rng.shuffle(new_rows)
    rows = base_rows[: len(base_rows) - tail] + new_rows + base_rows[len(base_rows) - tail:]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, rows)
    stats = {
        "base": str(base), "base_texts": len(base_rows), "tail": tail, "new_texts": len(new_rows), "total": len(rows),
        "quotas": want, "group_shares": group_shares, "forum_threads_sampled": len(threads), "seed": seed,
        "drops": {"theses": dict(drops_t), "forum": dict(drops_f)},
        "chars_median": {k: sorted(map(len, got[k]))[len(got[k]) // 2] for k in KINDS},
    }
    with open(out.with_suffix(".mix.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="200K mix plus Turkish thesis abstracts and tech/gaming forum texts")
    p.add_argument("--base", required=True)
    p.add_argument("--theses", required=True, help="tezler.parquet of umutertugrul/turkish-academic-theses-dataset")
    p.add_argument("--forum", nargs="+", required=True, help="ForumSohbetleri forum directories (JSONL files)")
    p.add_argument("--forbidden", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--n-new", type=int, default=50_000)
    p.add_argument("--tail", type=int, default=2000)
    p.add_argument("--forum-threads", type=int, default=60_000)
    p.add_argument("--min-tail-count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    stats = build_domain_mix(a.base, a.theses, a.forum, a.out, a.forbidden, a.n_new, a.tail, None,
                             a.forum_threads, a.min_tail_count, a.seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
