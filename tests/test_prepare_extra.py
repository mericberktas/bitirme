"""prepare_extra converters (parsing, exclusions, per-domain caps) and nested mixes (`subsample_mix`)."""

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from turkish_distill.build_mix import Bucket, MixConfig, build_mix, subsample_mix
from turkish_distill.prepare_extra import DomainCap, domain_of, mfaq, mqa, tapaco, wikipedia


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_domain_helpers():
    assert domain_of("https://www.haberler.com/a/b") == "haberler.com" and domain_of(None) == ""
    cap = DomainCap(2)
    assert [cap.allow("a"), cap.allow("a"), cap.allow("a"), cap.allow("b")] == [True, True, False, True]


def test_mqa_drops_excluded_sites_caps_domains_and_pairs_question_with_first_answer(tmp_path):
    def page(domain, n):
        return {"domain": domain, "questions": [
            {"name": f"Soru {domain} {i}?", "text": "", "answers": [{"text": None}, {"text": f"Cevap {domain} {i} burada uzun bir cevap."}]}
            for i in range(n)]}

    path = tmp_path / "mqa_tr" / "data"
    path.mkdir(parents=True)
    with gzip.open(path / "data.tr.cqa.json.gz", "wt", encoding="utf-8") as f:
        for p in (page("forum.example", 5), page("www.doktorsitesi.com", 3), page("kizlarsoruyor.com", 2), page("other.example", 1)):
            f.write(json.dumps(p) + "\n")
    counts = mqa(tmp_path, tmp_path / "out", qa_cap=3)
    questions = read_jsonl(tmp_path / "out" / "mqa_questions.jsonl")
    answers = read_jsonl(tmp_path / "out" / "mqa_answers.jsonl")
    assert counts == {"mqa_questions": 4, "mqa_answers": 4}  # forum.example capped at 3, plus other.example
    assert {q["url"] for q in questions} == {"https://forum.example/", "https://other.example/"}
    assert questions[0]["text"] == "Soru forum.example 0?" and answers[0]["text"].startswith("Cevap forum.example 0")


def test_mfaq_and_tapaco_and_wikipedia_leads(tmp_path):
    faq = tmp_path / "mfaq_tr" / "data" / "tr"
    faq.mkdir(parents=True)
    rows = [{"domain": "a.example", "qa_pairs": [{"question": f"Soru {i}?", "answer": "x"} for i in range(5)]},
            {"domain": "doktorsitesi.com", "qa_pairs": [{"question": "Dışlanan soru?", "answer": "y"}]}]
    (faq / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert mfaq(tmp_path, tmp_path / "out", faq_cap=2) == 2

    (tmp_path / "tapaco_tr" / "tr").mkdir(parents=True)
    pq.write_table(pa.table({"paraphrase": ["Kısa.", "Bu cümle otuz karakterden daha uzundur, evet."]}), tmp_path / "tapaco_tr" / "tr" / "a.parquet")
    assert tapaco(tmp_path, tmp_path / "out") == 1
    assert read_jsonl(tmp_path / "out" / "tapaco.jsonl")[0]["text"].startswith("Bu cümle")

    (tmp_path / "wikipedia_tr" / "20231101.tr").mkdir(parents=True)
    lead = "Bu, yüz elli karakterden uzun bir giriş paragrafıdır. " * 4
    pq.write_table(pa.table({"url": ["https://tr.wikipedia.org/wiki/A", "https://tr.wikipedia.org/wiki/B"],
                             "text": [lead + "\nBölüm başlığı\nİkinci paragraf.", "Kısa giriş.\nBaşka satır."]}),
                   tmp_path / "wikipedia_tr" / "20231101.tr" / "t.parquet")
    counts = wikipedia(tmp_path, tmp_path / "out")
    assert counts == {"wikipedia": 2, "wikipedia_leads": 1}
    assert read_jsonl(tmp_path / "out" / "wikipedia_leads.jsonl")[0]["text"] == lead.strip()


def test_subsample_is_a_subset_with_the_same_bucket_proportions(tmp_path):
    src = tmp_path / "s.jsonl"
    src.write_text("\n".join(json.dumps({"text": f"Bu bir örnek cümledir numara {i} ve yeterince uzundur."}) for i in range(2000)), encoding="utf-8")
    other = tmp_path / "o.jsonl"
    other.write_text("\n".join(json.dumps({"text": f"Bambaşka bir kaynaktan gelen metin sayı {i} ve o da uzun."}) for i in range(2000)), encoding="utf-8")
    cfg = MixConfig("m", 1, 400, {"a": {"type": "jsonl", "path": str(src)}, "b": {"type": "jsonl", "path": str(other)}},
                    [Bucket("one", "a", "whole", 0.75, 20, 200), Bucket("two", "b", "whole", 0.25, 20, 200)])
    full = tmp_path / "full.jsonl"
    build_mix(cfg, full)
    stats = subsample_mix(full, cfg, 100, tmp_path / "small.jsonl")
    big, small = read_jsonl(full), read_jsonl(tmp_path / "small.jsonl")
    assert len(big) == 400 and len(small) == 100 and stats["total"] == 100
    assert {r["text"] for r in small} <= {r["text"] for r in big}  # nested
    assert sum(r["bucket"] == "one" for r in small) == 75 and sum(r["bucket"] == "two" for r in small) == 25
    assert all(r["rank"] < 75 for r in small if r["bucket"] == "one")


def test_tidy_removes_the_combining_dot_and_empty_parentheses():
    from turkish_distill.prepare_extra import tidy

    assert tidy("bir kedi i̇stiyorum") == "bir kedi istiyorum"
    assert tidy("Stará Ľubovňa ilçesi (), Slovakya'dadır. Tomita (d. 1967) futbolcu.") == "Stará Ľubovňa ilçesi, Slovakya'dadır. Tomita (d. 1967) futbolcu."


def test_v2_config_excludes_bibliography_lines_and_promotional_answers():
    import re

    from turkish_distill.build_mix import load_mix_config

    cfg = load_mix_config("configs/data_mix_v2.yaml")
    wiki = next(b for b in cfg.buckets if b.name == "wiki_paragraphs")
    ref = "Polozov S.A., Pereladova O. B. “Syunt-Khasardag Reserve” // “SSCB Rezervleri”. - Moskova : “ Düşünce ”. S. 156-175."
    assert re.search(wiki.exclude_regex, ref, re.I) and not re.search(wiki.exclude_regex, "Dolunay, 1988 yapımı bir filmdir.", re.I)
    answers = next(b for b in cfg.buckets if b.name == "informal_answers")
    assert re.search(answers.exclude_regex, "konya'dan op. dr. sinan doktorumuzun cevaplarını görüntüleyin.", re.I)
    assert abs(sum(b.share for b in cfg.buckets) - 1.0) < 1e-9


def test_refilter_drops_copies_of_a_second_forbidden_list_and_keeps_ranks_contiguous(tmp_path):
    from turkish_distill.build_mix import refilter_mix

    src = tmp_path / "s.jsonl"
    src.write_text("\n".join(json.dumps({"text": f"Bu bir örnek cümledir numara {i} ve yeterince uzundur."}) for i in range(60)), encoding="utf-8")
    cfg = MixConfig("m", 1, 40, {"a": {"type": "jsonl", "path": str(src)}}, [Bucket("one", "a", "whole", 1.0, 20, 200)])
    full = tmp_path / "full.jsonl"
    build_mix(cfg, full)
    forbidden = tmp_path / "forbidden.jsonl"
    banned = [r["text"] for r in read_jsonl(full) if r["rank"] in (0, 5)]
    forbidden.write_text("\n".join(json.dumps({"text": t}) for t in banned), encoding="utf-8")
    forbidden.with_suffix(".stats.json").write_text(json.dumps({"unique_texts": 2, "new_texts_per_dataset": {"other/bench": 2}}), encoding="utf-8")
    stats = refilter_mix(full, forbidden, tmp_path / "clean.jsonl")
    rows = read_jsonl(tmp_path / "clean.jsonl")
    assert stats["rows_in"] == 40 and stats["rows_out"] == 38 and stats["removed_by_first_matching_dataset"] == {"other/bench": 2}
    assert sorted(r["rank"] for r in rows) == list(range(38)) and not ({r["text"] for r in rows} & set(banned))
    small = subsample_mix(tmp_path / "clean.jsonl", cfg, 20, tmp_path / "small.jsonl")
    assert small["total"] == 20


def test_mizan_suite_files_and_json_parsing(tmp_path):
    import gzip as gz

    from turkish_distill import benchmark_texts as bt

    assert set(bt.MIZAN_FILES) >= {"mteb/sib200", "PaDaS-Lab/webfaq-retrieval", "apple/mkqa", "mteb/sts22-crosslingual-sts"}
    assert all(len(rev) >= 4 for rev, _ in bt.MIZAN_FILES.values()) and "apple/mkqa" in bt.DATA_URLS
    path = tmp_path / "x.jsonl.gz"
    with gz.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": "1", "text": "Bu bir Türkçe kullanıcı cümlesidir."}) + "\n")
        f.write(json.dumps({"queries": {"tr": "Türkçe sorgu metni nedir?", "de": "Deutsche Frage"}, "answers": {"tr": [{"text": "Türkçe cevap metni"}], "de": [{"text": "Deutsch"}]}}) + "\n")
    texts = list(bt.iter_file_texts(path))
    assert "Bu bir Türkçe kullanıcı cümlesidir." in texts and "Türkçe sorgu metni nedir?" in texts and "Türkçe cevap metni" in texts
    assert "Deutsche Frage" not in texts and "Deutsch" not in texts  # MKQA rows: only the Turkish side is evaluated


def test_merge_attributes_new_texts_to_the_extra_dataset_that_adds_them(tmp_path):
    from turkish_distill.benchmark_texts import merge

    def make(name, texts, per):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps({"text": t}) for t in texts), encoding="utf-8")
        p.with_suffix(".stats.json").write_text(json.dumps({"unique_texts": len(texts), "new_texts_per_dataset": per, "min_chars": 10}), encoding="utf-8")
        return p

    base = make("b.jsonl", ["bir iki üç dört beş altı", "yedi sekiz dokuz on"], {"base/x": 2})
    extra = make("e.jsonl", ["yedi sekiz dokuz on", "yeni metin bir birinci", "yeni metin iki ikinci"], {"extra/a": 2, "extra/b": 1})
    stats = merge(base, extra, tmp_path / "all.jsonl")
    assert stats["unique_texts"] == 4 and stats["new_texts_per_dataset"] == {"base/x": 2, "extra/a": 1, "extra/b": 1}
    assert len(read_jsonl(tmp_path / "all.jsonl")) == 4
