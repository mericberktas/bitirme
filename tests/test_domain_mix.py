"""Domain mix: thesis abstracts/titles and forum posts added to the base mix (signature stripping, quotas, layout)."""

import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from turkish_distill import domain_mix as dm

SIG = "Sistemim: Ryzen 5 5600 Asus B550 16GB DDR4 RTX 3060 Kioxia 1TB NVMe Corsair 650W kasa soğutma"
SYL = ("ka", "le", "mi", "to", "ru", "sa", "ne", "po", "di", "ya", "bu", "ze", "ço", "ğa", "şi")


def words(i, n):
    vocab = [a + b + c for a in SYL for b in SYL for c in SYL[:3]]
    r = random.Random(i)
    return " ".join(r.choice(vocab) for _ in range(n))


def sentences(i, n_sent, n_words=12):
    return " ".join(words(i * 100 + s, n_words).capitalize() + "." for s in range(n_sent))


def test_signature_tail_is_stripped():
    posts = [f"{sentences(i, 2)} {SIG}" for i in range(20)] + [sentences(100 + i, 2) for i in range(20)]
    tails = dm.frequent_tails(posts, min_count=8)
    stripped = dm.strip_post(posts[0], tails)
    assert "Ryzen" not in stripped and stripped.startswith(sentences(0, 2)[:20])
    assert dm.strip_post(posts[25], tails) == posts[25]  # no signature, unchanged
    assert "birleştirildi" not in dm.strip_post("Merhaba. Mesaj otomatik birleştirildi: 10 Ağustos 2023 Tamam.", set())
    assert dm.strip_post("Hız kaç MB? Son düzenleyen: Moderatör: 13 Mart 2020", set()) == "Hız kaç MB?"


def test_cut_at_sentence_and_life_science():
    text = sentences(1, 30)
    cut = dm.cut_at_sentence(text, 300)
    assert len(cut) <= 300 and cut.endswith(".") and text.startswith(cut)
    assert dm.subject_group("Tıp = Medicine; Halk Sağlığı = Public Health") == "life"
    assert dm.subject_group("Bankacılık = Banking") == "finance"
    assert dm.subject_group("Elektrik ve Elektronik Mühendisliği = Electrical and Electronics Engineering") == "other"
    rows = [{"abstract_tr": "Araştırma planlandı.Araştırmanın evreni 246 doktordur. " + sentences(7, 6), "title_tr": "Kısa",
             "subject": "Tıp = Medicine"}]
    (kind, text, group), = dm.thesis_candidates(rows, random.Random(0))
    assert kind == "thesis_abstracts" and group == "life" and "planlandı. Araştırmanın" in text


def test_select_respects_quotas_life_split_and_forbidden():
    cands = [("thesis_abstracts", f"metin {i}", "life" if i % 2 == 0 else "other") for i in range(40)]
    cands.insert(0, ("thesis_abstracts", "yasak metin", "life"))

    class Index:
        def check(self, text):
            return (1, 0, False) if text == "yasak metin" else (0, None, False)

    got, drops = dm.select(cands, {"thesis_abstracts": 10}, {"metin 0"}, Index(), {"life": 0.5, "other": 0.5})
    texts = got["thesis_abstracts"]
    assert len(texts) == 10 and "yasak metin" not in texts and "metin 0" not in texts
    assert sum(int(t.split()[1]) % 2 == 0 for t in texts) == 5  # half life-science
    assert drops["benchmark_overlap"] == 1 and drops["duplicate"] == 1


def test_build_domain_mix_layout(tmp_path):
    base = tmp_path / "base.jsonl"
    base_rows = [{"text": f"temel karma metni numara {i} burada"} for i in range(60)]
    base.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base_rows), encoding="utf-8")
    theses = tmp_path / "tezler.parquet"
    pq.write_table(pa.table({
        "title_tr": [f"Tez başlığı {words(i, 4)} üzerine bir inceleme" for i in range(80)],
        "abstract_tr": [sentences(i, 6) for i in range(80)],
        "subject": [["Tıp = Medicine", "Fizik = Physics", "Bankacılık = Banking"][i % 3] for i in range(80)],
    }), theses)
    forum = tmp_path / "forum"
    forum.mkdir()
    threads = [{"url": f"https://f/{i}", "texts": [sentences(1000 + i, 2) + " ne yapmalıyım?",
                                                   sentences(2000 + i, 2), sentences(3000 + i, 3)]} for i in range(80)]
    (forum / "train.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in threads), encoding="utf-8")
    out = tmp_path / "mix.jsonl"
    stats = dm.build_domain_mix(base, theses, [forum], out, n_new=40, tail=5, forum_threads=80,
                                group_shares={"life": 0.5, "finance": 0.25, "other": 0.25})
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert len(rows) == 100 and stats["new_texts"] == 40
    assert rows[:55] == base_rows[:55] and rows[-5:] == base_rows[-5:]
    new = rows[55:95]
    assert {r["bucket"] for r in new} == set(dm.KINDS) and len({r["text"] for r in new}) == 40
    assert stats["quotas"] == dm.quotas(40)
    with pytest.raises(ValueError, match="ran out"):
        dm.build_domain_mix(base, theses, [forum], tmp_path / "x.jsonl", n_new=2000, tail=5, forum_threads=80)


def test_extend_mix_moves_domain_rows_into_another_base(tmp_path):
    def write(path, rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    base_rows = [{"text": f"v3 metni {i}"} for i in range(30)]
    donor_rows = [{"text": f"eski karma {i}"} for i in range(10)] + [
        {"text": "tez özeti bir", "bucket": "thesis_abstracts", "source": "theses"},
        {"text": "forum sorusu nasıl?", "bucket": "forum_questions", "source": "forum"},
        {"text": "forum cevabı", "bucket": "forum_replies", "source": "forum"},
        {"text": "v3 metni 3", "bucket": "forum_replies", "source": "forum"},  # already in the base: dropped
    ]
    base, donor, bq = tmp_path / "base.jsonl", tmp_path / "donor.jsonl", tmp_path / "bq.jsonl"
    write(base, base_rows)
    write(donor, donor_rows)
    write(bq, [{"question": "eski soru?"}])
    out, qout = tmp_path / "out.jsonl", tmp_path / "q.jsonl"
    stats = dm.extend_mix(base, donor, out, qout, bq, tail=4)
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert len(rows) == 33 and stats["dropped_duplicates"] == 1
    assert rows[:26] == base_rows[:26] and rows[-4:] == base_rows[-4:]
    assert [r["text"] for r in rows[26:29]] == ["tez özeti bir", "forum sorusu nasıl?", "forum cevabı"]
    assert [json.loads(line)["question"] for line in open(qout, encoding="utf-8")] == ["eski soru?", "forum sorusu nasıl?"]
