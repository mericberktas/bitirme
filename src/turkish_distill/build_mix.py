"""Build a mixed distillation corpus from several sources with a quota per bucket.

    python -m turkish_distill.build_mix --config configs/data_mix_v1.yaml \
        --forbidden data/benchmark_texts.jsonl --out data/mix/mix_v1.jsonl

Why: a distillation corpus needs no labels, so its composition is a free design choice. Raw web chunks are long
(the Cosmos chunks are cut at 1,500 characters) while most TR-MTEB texts are sentences, reviews and queries, and
the QA datasets are a single genre. A *bucket* takes texts of one kind from one source (whole passages, sentences
or lines of a passage, questions or passages of a pairs file), within a length range and behind a noise filter,
and gets a share of the target size.

Config (YAML):
    name, seed, target_total, oversample (candidates kept per bucket = oversample x quota; room for the
    duplicates and benchmark copies removed at the end)
    sources: {<name>: {type: jsonl | pairs, path: ...}}     jsonl rows: {"text", optional "url"}; pairs: prepare_pairs
    buckets: [{name, source, transform, share, min_chars, max_chars, clean, max_per_doc, include_regex,
               exclude_regex, url_regex, where_source, allow_shortfall}]
    transform: whole | sentences | lines (jsonl) ; questions | passages (pairs)

Everything is sampled in one pass over each source (reservoir sampling, seeded). Afterwards each bucket's candidates
are shuffled and accepted in turn unless they duplicate an earlier text (case/whitespace-insensitive) or, with
--forbidden, equal / share a 13-gram with a benchmark test text. A bucket that cannot fill its quota raises an
error unless it has allow_shortfall. The output rows are {"text", "bucket"[, "url"]}, shuffled; a statistics file
next to it records the rejections per reason, the lengths and the top domains of every bucket.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import yaml

from .prepare_pairs import normalize

LOWER, UPPER = "a-zçğıöşü", "A-ZÇĞİÖŞÜ"
_GLUE = re.compile(f"[{LOWER}][{UPPER}]")  # "TepkisaadetPartisi": words glued together (menus, headline fragments)
_URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|info)\b", re.I)
_SPLIT = re.compile(rf"(?<=[.!?…])[\"”’')\]]*\s+(?=[\"“‘'(\[]?[{UPPER}0-9])")
_ABBREVIATIONS = {"dr", "prof", "doç", "yrd", "av", "sn", "sayın", "vb", "vs", "bkz", "örn", "no", "st", "gen", "alb",
                  "müh", "uzm", "öğr", "gör", "dz", "tuğ", "ltd", "şti", "a.ş", "mah", "cad", "sok"}


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")


def _fold(text: str) -> str:
    """Lower-case ASCII form, so patterns match with or without Turkish diacritics (web text often has none)."""
    return text.translate(_FOLD).lower()


# Page furniture seen in the Cosmos chunks: cookie banners, copyright footers, "you may also like" blocks, buttons.
_BOILERPLATE = re.compile(
    r"(cerez\w*.{0,60}(kullan|kabul|tercih)|tum haklari saklidir|haklari\b.{0,40}\baittir|bunlar da ilginizi cekebilir|"
    r"devamini oku|abone ol\b|gizlilik politikasi|tiklayin|giris yap\b|uye ol\b|sepete ekle|sosyal medyada paylas|"
    r"yorum yap\b|paylas\b.{0,10}tweetle)")
# Adult and gambling spam. Gambling terms are specific on purpose: "bahse konu" ("the matter at hand") is ordinary
# official Turkish. Mild profanity is NOT listed: offensive-language texts are part of what the student must embed.
_UNWANTED = re.compile(
    r"\b(escort\w*|porno\w*|pornografi\w*|erotik\w*|hentai|seks\w*|sex\w*|sikis\w*|orospu\w*|"
    r"canli bahis|bahis sitesi\w*|bahis oyna\w*|bahis oranlari|casino\w*|kumar\w*|iddaa\w*|bet365|slot oyun\w*)\b")


def clean_reason(text: str) -> Optional[str]:
    """Why `text` looks like page furniture, spam or foreign-script junk rather than prose (None = clean).
    Thresholds come from samples of the Cosmos corpus and are deliberately simple."""
    words = text.split()
    n = len(words)
    if n == 0:
        return "empty"
    if _URL.search(text):
        return "url"
    folded = _fold(text)
    if _UNWANTED.search(folded):
        return "unwanted_content"
    if _BOILERPLATE.search(folded):
        return "boilerplate"
    foreign = sum(1 for c in text if c.isalpha() and ord(c) > 0x24F)  # beyond Latin Extended-B: Arabic, Cyrillic, CJK ...
    if foreign > 2 and foreign / len(text) > 0.02:
        return "foreign_script"
    non_space = sum(not c.isspace() for c in text)
    if sum(c.isalpha() for c in text) / max(non_space, 1) < 0.70:
        return "alpha_ratio"
    if sum(c.isdigit() for c in text) / max(non_space, 1) > 0.25:
        return "digit_ratio"
    glued = len(_GLUE.findall(text))
    if glued >= 3 and glued / n > 0.03:
        return "glued_words"
    enders = sum(text.count(c) for c in ".!?…")
    if n >= 8 and text.count(",") / n >= 0.30 and enders <= n / 40:
        return "tag_list"
    if n >= 30 and len({w.lower() for w in words}) / n < 0.35:
        return "repetition"
    if len(text) >= 200 and enders == 0:
        return "no_sentence_punctuation"
    if sum(1 for c in text if ord(c) > 0x2FFF) > 5:
        return "symbols"
    return None


def _ends_with_abbreviation(piece: str) -> bool:
    last = piece.rstrip().split(" ")[-1] if piece.strip() else ""
    stem = last.rstrip(".").lower()
    # ordinal numbers are small ("1. Dünya Savaşı"); a four-digit year before a full stop ends the sentence
    return stem in _ABBREVIATIONS or (len(stem) == 1 and stem.isalpha()) or bool(re.fullmatch(r"\d{1,2}\.", last))


def split_sentences(text: str) -> list[str]:
    """Sentences of a passage: split at line breaks and after . ! ? before a capital letter or digit, without
    cutting after common abbreviations, single-letter initials or ordinal numbers of up to two digits
    ("1. Dünya Savaşı"); "…yıl 1923. Sonra" is split."""
    out: list[str] = []
    for block in re.split(r"\n+", text):
        merged: list[str] = []
        for piece in _SPLIT.split(block):
            if merged and _ends_with_abbreviation(merged[-1]):
                merged[-1] += " " + piece
            else:
                merged.append(piece)
        out += [normalize(m) for m in merged if m.strip()]
    return out


def unescape_breaks(text: str) -> str:
    """The Cosmos texts contain the two characters backslash + n where a line break was (and prepare_corpus
    flattened real breaks into spaces), so the escape sequences are the only structure left; make them real."""
    return text.replace("\\n", "\n").replace("\\r", "\n").replace("\\t", " ")


def split_lines(text: str) -> list[str]:
    return [normalize(line) for line in re.split(r"\n+", text) if line.strip()]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

TRANSFORMS = {"jsonl": ("whole", "sentences", "lines"), "pairs": ("questions", "passages")}


@dataclass
class Bucket:
    name: str
    source: str
    transform: str
    share: float
    min_chars: int = 20
    max_chars: int = 2000
    clean: bool = True
    max_per_doc: int = 1
    include_regex: Optional[str] = None
    exclude_regex: Optional[str] = None
    url_regex: Optional[str] = None
    where_source: list[str] = field(default_factory=list)  # pairs only: keep rows whose `source` is listed
    allow_shortfall: bool = False


@dataclass
class MixConfig:
    name: str
    seed: int
    target_total: int
    sources: dict[str, dict[str, str]]
    buckets: list[Bucket]
    oversample: float = 1.5


def load_mix_config(path: str | Path, overrides: Optional[dict[str, str]] = None) -> MixConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {"name", "seed", "target_total", "sources", "buckets", "oversample"}
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    sources = {k: dict(v) for k, v in raw["sources"].items()}
    for name, new_path in (overrides or {}).items():
        if name not in sources:
            raise ValueError(f"--source {name}: not a source of the config ({sorted(sources)})")
        sources[name]["path"] = new_path
    buckets = []
    for b in raw["buckets"]:
        bad = set(b) - set(Bucket.__dataclass_fields__)
        if bad:
            raise ValueError(f"bucket {b.get('name')}: unknown keys {sorted(bad)}")
        buckets.append(Bucket(**b))
    cfg = MixConfig(raw["name"], int(raw["seed"]), int(raw["target_total"]), sources, buckets, float(raw.get("oversample", 1.5)))
    validate_mix(cfg)
    return cfg


def validate_mix(cfg: MixConfig) -> None:
    names = [b.name for b in cfg.buckets]
    if len(set(names)) != len(names):
        raise ValueError(f"bucket names must be unique, got {names}")
    if abs(sum(b.share for b in cfg.buckets) - 1.0) > 1e-6:
        raise ValueError(f"bucket shares must sum to 1, got {sum(b.share for b in cfg.buckets):.4f}")
    for b in cfg.buckets:
        if b.source not in cfg.sources:
            raise ValueError(f"bucket {b.name}: unknown source {b.source!r}")
        kind = cfg.sources[b.source]["type"]
        if kind not in TRANSFORMS or b.transform not in TRANSFORMS[kind]:
            raise ValueError(f"bucket {b.name}: transform {b.transform!r} does not fit a {kind!r} source {TRANSFORMS.get(kind)}")
        if b.min_chars > b.max_chars or b.share <= 0:
            raise ValueError(f"bucket {b.name}: need min_chars <= max_chars and a positive share")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

class Reservoir:
    """Uniform random sample of fixed size from a stream (Algorithm R), seeded."""

    def __init__(self, size: int, rng: random.Random):
        self.size, self.rng, self.seen, self.items = size, rng, 0, []

    def add(self, item: Any) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.size:
                self.items[j] = item


def iter_source(spec: dict[str, str]) -> Iterator[dict]:
    with open(spec["path"], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class BucketSampler:
    def __init__(self, bucket: Bucket, quota: int, oversample: float, seed: int):
        self.bucket = bucket
        self.rng = random.Random(f"{seed}|{bucket.name}")
        self.reservoir = Reservoir(max(quota, math.ceil(quota * oversample)), self.rng)
        self.rejected: collections.Counter = collections.Counter()
        self.include = re.compile(bucket.include_regex, re.I) if bucket.include_regex else None
        self.exclude = re.compile(bucket.exclude_regex, re.I) if bucket.exclude_regex else None
        self.url_re = re.compile(bucket.url_regex, re.I) if bucket.url_regex else None
        self._seen_passages: set[str] = set()

    def candidates(self, row: dict, kind: str, cache: dict) -> list[str]:
        """Texts of this bucket's kind in the row; `cache` (one per row, shared by all buckets) holds the splits."""
        b = self.bucket
        if kind == "pairs":
            if b.where_source and row.get("source") not in b.where_source:
                return []
            if b.transform == "questions":
                return [row["question"]]
            pid = row.get("passage_id", row["passage"])
            if pid in self._seen_passages:
                return []
            self._seen_passages.add(pid)
            return [row["passage"]]
        if self.url_re and not self.url_re.search(row.get("url", "")):
            return []
        if "text" not in cache:
            cache["text"] = unescape_breaks(row["text"])
        text = cache["text"]
        if b.transform == "whole":
            return [normalize(text)]  # line breaks become spaces
        if b.transform not in cache:
            cache[b.transform] = split_sentences(text) if b.transform == "sentences" else split_lines(text)
        return cache[b.transform]

    def feed(self, row: dict, kind: str, cache: dict) -> None:
        b = self.bucket
        valid = []
        for text in self.candidates(row, kind, cache):
            if not b.min_chars <= len(text) <= b.max_chars:
                self.rejected["length"] += 1
            elif self.include and not self.include.search(text):
                self.rejected["include_regex"] += 1
            elif self.exclude and self.exclude.search(text):
                self.rejected["exclude_regex"] += 1
            else:
                reason = clean_reason(text) if b.clean else None
                if reason:
                    self.rejected[reason] += 1
                else:
                    valid.append(text)
        if len(valid) > b.max_per_doc:
            valid = self.rng.sample(valid, b.max_per_doc)
        for text in valid:
            self.reservoir.add({"text": text, "url": row.get("url")} if row.get("url") else {"text": text})


def _is_benchmark_copy(index, text: str) -> bool:
    shared_ngrams, _, equals_a_text = index.check(text)
    return equals_a_text or shared_ngrams >= 1


def _domain(url: Optional[str]) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def build_mix(cfg: MixConfig, out_path: str | Path, forbidden: Optional[str | Path] = None, progress_every: int = 200_000) -> dict:
    started = time.time()
    quotas = {b.name: round(b.share * cfg.target_total) for b in cfg.buckets}
    samplers = {b.name: BucketSampler(b, quotas[b.name], cfg.oversample, cfg.seed) for b in cfg.buckets}
    for source_name, spec in cfg.sources.items():
        mine = [samplers[b.name] for b in cfg.buckets if b.source == source_name]
        if not mine:
            continue
        for i, row in enumerate(iter_source(spec), 1):
            cache: dict = {}
            for sampler in mine:
                sampler.feed(row, spec["type"], cache)
            if progress_every and i % progress_every == 0:
                print(f"  {source_name}: {i:,} rows read", flush=True)

    index = None
    if forbidden:
        from .decontaminate_corpus import BenchmarkIndex, _dataset_names_and_bounds, _iter_texts

        _, bounds = _dataset_names_and_bounds(Path(forbidden))
        index = BenchmarkIndex(_iter_texts(Path(forbidden)), bounds, 13)
        print(f"indexed {len(index.grams):,} benchmark n-grams", flush=True)

    rng = random.Random(cfg.seed)
    taken: set[str] = set()
    selected: list[dict] = []
    stats: dict[str, Any] = {"name": cfg.name, "target_total": cfg.target_total, "buckets": {}}
    for b in cfg.buckets:
        sampler, quota = samplers[b.name], quotas[b.name]
        pool = list(sampler.reservoir.items)
        rng.shuffle(pool)
        accepted, dup, contaminated = [], 0, 0
        for item in pool:
            if len(accepted) >= quota:
                break
            key = normalize(item["text"]).lower()
            if key in taken:
                dup += 1
            elif index is not None and _is_benchmark_copy(index, item["text"]):
                contaminated += 1
            else:
                taken.add(key)
                accepted.append({**item, "bucket": b.name})
        lengths = sorted(len(x["text"]) for x in accepted)
        domains = collections.Counter(_domain(x.get("url")) for x in accepted if x.get("url"))
        stats["buckets"][b.name] = {
            "quota": quota, "selected": len(accepted), "candidates_seen_after_filters": sampler.reservoir.seen,
            "rejected_before_sampling": dict(sampler.rejected), "dropped_duplicates": dup, "dropped_benchmark_copies": contaminated,
            "chars_p10_p50_p90": [lengths[len(lengths) // 10], lengths[len(lengths) // 2], lengths[int(len(lengths) * .9)]] if lengths else None,
            "top_domains": domains.most_common(8) or None,
        }
        if len(accepted) < quota:
            stats["buckets"][b.name]["shortfall"] = quota - len(accepted)
            if not b.allow_shortfall:
                raise ValueError(f"bucket {b.name}: only {len(accepted)} of {quota} texts available "
                                 f"(candidates after filters: {sampler.reservoir.seen}); relax the filters, change the share or set allow_shortfall")
        selected += accepted
    rng.shuffle(selected)
    stats["total"] = len(selected)
    stats["seconds"] = round(time.time() - started)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    out_path.with_suffix(".mix.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a mixed distillation corpus with a quota per bucket")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--forbidden", default=None, help="benchmark_texts.jsonl: drop copies of benchmark test texts")
    parser.add_argument("--source", action="append", default=[], help="name=path: replace a source's path (repeatable)")
    parser.add_argument("--target-total", type=int, default=None, help="replace target_total (the shares stay)")
    args = parser.parse_args(argv)
    overrides = dict(item.split("=", 1) for item in args.source)
    cfg = load_mix_config(args.config, overrides)
    if args.target_total is not None:
        cfg.target_total = args.target_total
    stats = build_mix(cfg, args.out, args.forbidden)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
