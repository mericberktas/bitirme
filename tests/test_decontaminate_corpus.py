import json

import pytest

from turkish_distill.decontaminate_corpus import BenchmarkIndex, decontaminate_corpus


def words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(n))


def write_jsonl(path, texts):
    path.write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n", encoding="utf-8")


def setup(tmp_path):
    # benchmark file with two datasets: lines 0-1 belong to "qa", line 2 to "nli"
    forbidden = tmp_path / "bench.jsonl"
    leaked_a, leaked_b = words("a", 30), words("b", 30)
    write_jsonl(forbidden, [leaked_a, leaked_b, "Kisa bir benchmark cumlesi burada"])
    forbidden.with_suffix(".stats.json").write_text(json.dumps({"new_texts_per_dataset": {"qa": 2, "nli": 1}}), encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, [
        "tamamen temiz bir pasaj metni, benchmark ile ilgisi yok",
        "giris " + leaked_a + " cikis",                 # shares 13-grams with qa
        "bastan " + words("b", 14) + " sondan",        # shares exactly two 13-grams with qa (b0..b12, b1..b13)
        "  KISA bir   benchmark cumlesi BURADA ",       # equals the nli text after normalisation
        "temiz " + words("z", 40),
    ])
    return corpus, forbidden


def test_removes_copies_exact_matches_and_names_the_datasets(tmp_path):
    corpus, forbidden = setup(tmp_path)
    out = tmp_path / "clean.jsonl"
    stats = decontaminate_corpus(corpus, forbidden, out, progress_every=0)
    kept = [json.loads(l)["text"] for l in open(out, encoding="utf-8")]
    assert kept == ["tamamen temiz bir pasaj metni, benchmark ile ilgisi yok", "temiz " + words("z", 40)]
    assert (stats["passages"], stats["kept"], stats["removed"]) == (5, 2, 3)
    assert stats["removed_by_first_matching_dataset"] == {"qa": 2, "nli": 1}
    assert stats["removed_equal_to_a_benchmark_text"] == 1
    # exact match (0 shared n-grams, too short for one), two shared n-grams, and the 30-word copy: 30 - 13 + 1 = 18
    assert stats["removed_by_number_of_shared_ngrams"] == {"0-1": 1, "2-4": 1, "5-19": 1, "20+": 0}
    assert (out.with_suffix(".decontam.json")).exists() and not list(tmp_path.glob("*.tmp"))


def test_min_hits_keeps_passages_that_only_share_a_phrase(tmp_path):
    corpus, forbidden = setup(tmp_path)
    out = tmp_path / "clean.jsonl"
    stats = decontaminate_corpus(corpus, forbidden, out, min_hits=3, progress_every=0)
    kept = [json.loads(l)["text"] for l in open(out, encoding="utf-8")]
    # the passage sharing two 13-grams survives min_hits=3; the full copy and the exact match are still removed
    assert any(t.startswith("bastan") for t in kept) and len(kept) == 3
    assert stats["removed"] == 2


def test_check_counts_shared_ngrams(tmp_path):
    corpus, forbidden = setup(tmp_path)
    index = BenchmarkIndex((json.loads(l)["text"] for l in open(forbidden, encoding="utf-8")), [0, 2], 13)
    hits, dataset, exact = index.check("x " + words("a", 15) + " y")
    assert (hits, dataset, exact) == (3, 0, False)  # a0..a12, a1..a13, a2..a14
    assert index.check("hicbir ortak kelime yok burada")[0] == 0
    assert index.check("kisa BIR benchmark cumlesi   burada")[2] is True


def test_works_without_a_stats_file_and_agrees_with_prepare_pairs(tmp_path):
    from turkish_distill.prepare_pairs import decontaminate, make_pair

    forbidden = tmp_path / "bench.jsonl"
    write_jsonl(forbidden, [words("q", 25)])  # no .stats.json next to it
    texts = ["on " + words("q", 14), "bambaska bir metin, ortak 13 kelimelik dizi yok", words("q", 25)]
    corpus = tmp_path / "c.jsonl"
    write_jsonl(corpus, texts)
    stats = decontaminate_corpus(corpus, forbidden, tmp_path / "o.jsonl", progress_every=0)
    assert stats["removed_by_first_matching_dataset"] == {"benchmark": 2}
    pairs = [make_pair("s", "soru metni nedir?", t) for t in texts]
    kept, dropped = decontaminate(pairs, [words("q", 25)])  # the small-scale implementation must agree
    assert dropped == stats["removed"] and [p["passage"] for p in kept] == [texts[1]]
