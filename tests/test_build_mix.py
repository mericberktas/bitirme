import json

import pytest
import yaml

from turkish_distill import build_mix as bm

VOCAB = ("ev okul yol deniz dağ nehir köprü şehir köy kasaba bahçe pazar kitap defter kalem masa sandalye pencere kapı "
         "çocuk öğretmen doktor mühendis çiftçi işçi sporcu yazar ressam müzisyen toplantı sınav yarışma festival konser sergi "
         "tören gezi ziyaret görüşme karar proje rapor bütçe takvim program plan öneri sonuç haber makale yorum soru cevap "
         "ilkbahar yaz sonbahar kış sabah akşam gece hafta").split()


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


# --- noise filter ------------------------------------------------------------ #

@pytest.mark.parametrize("text, reason", [
    ("Ankara, Türkiye'nin başkenti olan ve nüfusu en kalabalık ikinci büyük şehridir.", None),
    ("Toplantıda alınan kararlar, yarın sabah basın mensuplarına açıklanacak.", None),
    ("HaberlerSiyasetSaadet Partisi Genel Başkanı Arıkan'dan Memur Zammı Kararına TepkisaadetPartisiPAYLAŞKaydet", "glued_words"),
    ("Krem, Koltuk takımı, Orta sehpa, Sehpa, Halı, Salon halısı, Ev bitkisi, Avize, Aydınlatma, Tül perde, Perde", "tag_list"),
    ("Daha fazla bilgi için www.ornek.com adresini ziyaret edin ve kayıt olun.", "url"),
    ("12 34 56 78 90 12 34 56 78 90 ab", "alpha_ratio"),
    ("Evet evet evet evet evet evet evet evet evet evet evet evet evet evet evet evet evet evet evet evet. " * 2, "repetition"),
    (" ".join(VOCAB[:40]), "no_sentence_punctuation"),
    ("Sitemiz çerezleri kullanır; devam ederek çerez kullanımını kabul etmiş olursunuz, teşekkürler.", "boilerplate"),
    ("Tüm hakları saklıdır. Bu içeriğin hakları Örnek Medya Grubu'na aittir.", "boilerplate"),
    ("Canlı bahis ve casino sitelerinde yüksek oranlar sizi bekliyor, hemen kaydolun.", "unwanted_content"),
    ("Kadıköy escort ilanları ile ilgili gençlerle sohbet imkanı sunuluyor.", "unwanted_content"),
    ("İDDAA kuponu hazırlayan kullanıcılar sonucu merakla bekledi.", "unwanted_content"),
    ("Bahse konu araç, mahkeme kararıyla satışa çıkarıldı ve ihale sonuçlandı.", None),
    ("Sınır, تعريف a bound border extent utmost reach في الإنجليزية التركية القاموس kelimesi demektir.", "foreign_script"),
    ("Yunan alfabesinin son harfi ω olarak yazılır ve ohm birimini simgeler.", None),
])
def test_clean_reason(text, reason):
    assert bm.clean_reason(text) == reason


# --- sentence splitting ------------------------------------------------------- #

def test_sentences_split_on_boundaries_but_not_after_abbreviations_initials_or_ordinals():
    text = ("Prof. Dr. Ayşe Yılmaz konferansa katıldı. Toplantı 1. Dünya Savaşı üzerineydi! Sonra A. Kaya konuştu. "
            "Soruları cevapladı?\nYeni satır başka bir cümledir.")
    assert bm.split_sentences(text) == [
        "Prof. Dr. Ayşe Yılmaz konferansa katıldı.",
        "Toplantı 1. Dünya Savaşı üzerineydi!",
        "Sonra A. Kaya konuştu.",
        "Soruları cevapladı?",
        "Yeni satır başka bir cümledir.",
    ]
    assert bm.split_sentences("Kuruluş yılı 1923. Sonra büyüdü. 15. Yüzyılda başka bir olay oldu.") == [
        "Kuruluş yılı 1923.", "Sonra büyüdü.", "15. Yüzyılda başka bir olay oldu."]
    assert bm.split_lines("birinci satır\n\n  ikinci   satır \n") == ["birinci satır", "ikinci satır"]
    # the corpus stores a line break as the two characters backslash + n
    assert bm.split_lines(bm.unescape_breaks("birinci satır\\n\\nikinci satır")) == ["birinci satır", "ikinci satır"]
    assert bm.split_sentences(bm.unescape_breaks("Başlık metni\\n\\nİlk cümle burada. İkinci cümle de burada.")) == [
        "Başlık metni", "İlk cümle burada.", "İkinci cümle de burada."]


# --- the builder --------------------------------------------------------------- #

def sentence(i):
    return " ".join(VOCAB[(i * 7 + j * 5) % len(VOCAB)] for j in range(9)).capitalize() + f" {VOCAB[i % len(VOCAB)]}{VOCAB[(i // len(VOCAB)) % len(VOCAB)]}."  # unique per i < 3600


def make_config(tmp_path, **changes):
    sentences = [sentence(i) for i in range(400)]
    docs = [{"text": " ".join(sentences[i * 4:(i + 1) * 4]) + "\nKrem, Koltuk, Sehpa, Halı, Avize, Perde, Tül, Lamba, Yatak, Dolap",
             "url": f"https://site{i % 4}.example/x{i}"} for i in range(100)]
    write_jsonl(tmp_path / "web.jsonl", docs)
    pairs = [{"source": "wikirag" if i % 2 else "wikiqa", "question": f"{i}. örnek soru nedir?",
              "passage": f"{i // 2} numaralı pasaj metni, oldukça uzun bir açıklama içerir ve cümle sonu vardır.",
              "passage_id": f"p{i // 2}"} for i in range(60)]
    write_jsonl(tmp_path / "pairs.jsonl", pairs)
    raw = {
        "name": "t", "seed": 7, "target_total": 100, "oversample": 2.0,
        "sources": {"web": {"type": "jsonl", "path": str(tmp_path / "web.jsonl")},
                    "qa": {"type": "pairs", "path": str(tmp_path / "pairs.jsonl")}},
        "buckets": [
            {"name": "sentences", "source": "web", "transform": "sentences", "share": 0.5, "min_chars": 30, "max_chars": 150},
            {"name": "questions", "source": "qa", "transform": "questions", "share": 0.2, "min_chars": 10, "max_chars": 100},
            {"name": "wiki_passages", "source": "qa", "transform": "passages", "share": 0.2, "min_chars": 20, "max_chars": 300,
             "where_source": ["wikiqa"]},
            {"name": "web_lines", "source": "web", "transform": "lines", "share": 0.1, "min_chars": 20, "max_chars": 400, "max_per_doc": 2},
        ],
    }
    raw.update(changes)
    path = tmp_path / "mix.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def read_out(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_quotas_dedup_filtering_and_determinism(tmp_path):
    path = make_config(tmp_path)
    stats = bm.build_mix(bm.load_mix_config(path), tmp_path / "out.jsonl", progress_every=0)
    rows = read_out(tmp_path / "out.jsonl")
    by_bucket = {b: sum(r["bucket"] == b for r in rows) for b in stats["buckets"]}
    assert by_bucket == {"sentences": 50, "questions": 20, "wiki_passages": 20, "web_lines": 10} and len(rows) == 100
    assert len({r["text"].lower() for r in rows}) == 100  # no duplicate across buckets
    assert all(30 <= len(r["text"]) <= 150 for r in rows if r["bucket"] == "sentences")
    assert not any("Krem, Koltuk" in r["text"] for r in rows)  # the tag list line was filtered out
    assert stats["buckets"]["web_lines"]["rejected_before_sampling"]["tag_list"] >= 1
    assert all(r["url"].startswith("https://site") for r in rows if r["bucket"] == "sentences")  # URL kept for web texts
    assert {d for d, _ in stats["buckets"]["sentences"]["top_domains"]} <= {f"site{i}.example" for i in range(4)}
    assert not (tmp_path / "out.jsonl.tmp").exists() and (tmp_path / "out.mix.json").exists()
    bm.build_mix(bm.load_mix_config(path), tmp_path / "again.jsonl", progress_every=0)
    assert read_out(tmp_path / "again.jsonl") == rows  # same seed, same corpus


def test_a_pairs_bucket_uses_one_text_per_passage_and_the_source_filter(tmp_path):
    path = make_config(tmp_path)
    bm.build_mix(bm.load_mix_config(path), tmp_path / "out.jsonl", progress_every=0)
    passages = [r["text"] for r in read_out(tmp_path / "out.jsonl") if r["bucket"] == "wiki_passages"]
    assert len(passages) == len(set(passages)) == 20  # 30 distinct passages exist for source wikiqa, 20 taken


def test_a_bucket_that_cannot_be_filled_raises_unless_it_allows_a_shortfall(tmp_path):
    raw = yaml.safe_load(make_config(tmp_path).read_text(encoding="utf-8"))
    raw["buckets"][2].update(where_source=["nowhere"])
    path = tmp_path / "short.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="wiki_passages: only 0 of 20"):
        bm.build_mix(bm.load_mix_config(path), tmp_path / "o1.jsonl", progress_every=0)
    raw["buckets"][2]["allow_shortfall"] = True
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    stats = bm.build_mix(bm.load_mix_config(path), tmp_path / "o2.jsonl", progress_every=0)
    assert stats["buckets"]["wiki_passages"]["shortfall"] == 20 and stats["total"] == 80


def test_benchmark_copies_are_dropped(tmp_path):
    path = make_config(tmp_path)
    leaked = sentence(5)
    forbidden = tmp_path / "bench.jsonl"
    write_jsonl(forbidden, [{"text": leaked}])
    forbidden.with_suffix(".stats.json").write_text(json.dumps({"new_texts_per_dataset": {"qa": 1}}), encoding="utf-8")
    stats = bm.build_mix(bm.load_mix_config(path), tmp_path / "out.jsonl", forbidden, progress_every=0)
    assert leaked not in {r["text"] for r in read_out(tmp_path / "out.jsonl")}
    assert stats["total"] == 100  # the oversampled pool refilled the quota


def test_config_validation(tmp_path):
    raw = yaml.safe_load(make_config(tmp_path).read_text(encoding="utf-8"))
    for mutate, message in (
        (lambda r: r["buckets"][0].update(share=0.9), "sum to 1"),
        (lambda r: r["buckets"][0].update(transform="questions"), "does not fit"),
        (lambda r: r["buckets"][0].update(source="nope"), "unknown source"),
        (lambda r: r["buckets"][0].update(shares=0.5), "unknown keys"),
        (lambda r: r["buckets"].append(dict(r["buckets"][0])), "unique"),
    ):
        bad = json.loads(json.dumps(raw))
        mutate(bad)
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            bm.load_mix_config(path)
    with pytest.raises(ValueError, match="not a source"):
        bm.load_mix_config(make_config(tmp_path), {"zzz": "x"})
    assert bm.load_mix_config(make_config(tmp_path), {"web": "elsewhere.jsonl"}).sources["web"]["path"] == "elsewhere.jsonl"
