"""prepare_synth_qa: parsing, cleaning and decontamination of the synthetic FineWeb QA rows."""

import csv
import json
from collections import Counter

from turkish_distill.contrastive import load_pairs
from turkish_distill.prepare_synth_qa import (
    clean_record,
    parse_record,
    prepare,
    sentiment_label,
    split_list,
    split_qa,
)

DOC = ("Burs vermek için İzmir'deki ilçelerden birinde annesinin yanında kalan ve fakülte okuyan kız öğrenciler "
       "tercih edildi. Yirmi kişilik burs için yüz yirmi altı öğrenci başvurdu ve milletvekili maaşını bursa ayırdı. ") * 2


def row(doc=DOC, qa_block=None, extra=""):
    qa_block = qa_block or ("1. Soru:\nKaç öğrenci başvurdu?\n\n1. Cevap:\nYüz yirmi altı öğrenci başvurdu.\n\n"
                            "2. Soru: Burs kimlere verildi?\n2. Cevap: Annesinin yanında kalan kız öğrencilere.\n\n"
                            "3. Soru: Почему это произошло?\n3. Cevap: Bilinmiyor.")
    return (f"{doc}\n\nYukarıdaki metnin özeti nedir?\nYirmi burs için yüz yirmi altı başvuru yapıldı.\n\n"
            "Yukarıdaki metnin anahtar kelimeleri nelerdir?\n1. burs\n2. kız öğrenciler\n\n"
            "Yukarıdaki metnin duygudurumu nedir?\np pozitif\n\n"
            "Yukarıdaki metnin kategorisi nedir?\nEğitim\n\n"
            f"Yukarıdaki metin ile ilgili 3 soru ve cevabı nelerdir?\n{qa_block}\n\n"
            "Yukarıdaki metni bulmayı sağlayacak 3 arama sorgusu nelerdir?\n1. İzmir kız öğrenci bursu\n"
            f"2. milletvekili burs başvurusu sayısı\n3. Kaç öğrenci başvurdu?{extra}\n")


def test_parse_and_split():
    rec = parse_record(row())
    assert set(rec) == {"doc", "summary", "keywords", "sentiment", "category", "qa", "queries"}
    assert rec["doc"] == DOC.strip()
    assert split_qa(rec["qa"])[:2] == [("Kaç öğrenci başvurdu?", "Yüz yirmi altı öğrenci başvurdu."),
                                       ("Burs kimlere verildi?", "Annesinin yanında kalan kız öğrencilere.")]
    assert split_list(rec["queries"])[0] == "İzmir kız öğrenci bursu"
    assert parse_record("no headers here") is None
    assert sentiment_label("p pozitif") == "pozitif" and sentiment_label("Olumsuz") == "negatif"
    assert sentiment_label("pazartesi") is None


def test_clean_drops_keywords_foreign_script_and_duplicates():
    counts = Counter()
    rec = clean_record(parse_record(row()), counts)
    assert "keywords" not in rec and rec["sentiment"] == "pozitif" and rec["category"] == "Eğitim"
    assert [x["question"] for x in rec["qa"]] == ["Kaç öğrenci başvurdu?", "Burs kimlere verildi?"]
    assert counts["question_non_latin"] == 1
    assert rec["queries"] == ["İzmir kız öğrenci bursu", "milletvekili burs başvurusu sayısı"]  # 3rd = a question
    assert counts["query_duplicate"] == 1
    assert clean_record(parse_record(row(doc="kısa")), counts) is None and counts["doc_length"] == 1
    assert clean_record(parse_record(row(doc="Привет мир " * 40)), counts) is None and counts["doc_non_latin"] == 1


def test_prepare_end_to_end_with_decontamination(tmp_path):
    other = DOC.replace("İzmir", "Ankara")
    contaminated_doc = "Bu metin bir test kümesinden alınmıştır ve on üç kelimeden uzun bir parçayı aynen içerir. " * 4
    path = tmp_path / "in.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Text"])
        for text in (row(), row(), row(doc=other), row(doc=contaminated_doc)):  # second is a duplicate
            w.writerow([text])
    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text(
        json.dumps({"text": "Bu metin bir test kümesinden alınmıştır ve on üç kelimeden uzun bir parçayı aynen içerir."},
                   ensure_ascii=False) + "\n" + json.dumps({"text": "İzmir kız öğrenci bursu"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stats = prepare(path, [forbidden], tmp_path / "out", val_fraction=0.0)
    c = stats["counts"]
    assert c["rows"] == 4 and c["doc_duplicate"] == 1 and c["doc_contaminated"] == 1 and c["docs_train"] == 2
    assert c["query_contaminated"] == 2  # the forbidden query, exact match, in both kept documents
    assert stats["docs_removed_by_first_matching_dataset"] == {"forbidden": 1}
    pairs = load_pairs(tmp_path / "out" / "pairs_qa_train.jsonl")
    assert len(pairs["question"]) == 4 and pairs["answer"][0] == "Yüz yirmi altı öğrenci başvurdu."
    queries = load_pairs(tmp_path / "out" / "pairs_query_train.jsonl")
    assert "İzmir kız öğrenci bursu" not in queries["question"] and len(queries["question"]) == 2
    docs = [json.loads(line) for line in open(tmp_path / "out" / "docs.jsonl", encoding="utf-8")]
    assert len(docs) == 2 and all("keywords" not in d for d in docs)
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_truncate_passage():
    from turkish_distill.prepare_synth_qa import truncate_passage

    text = "kelime " * 1000
    cut = truncate_passage(text, 2500)
    assert len(cut) <= 2500 and not cut.endswith(" ") and cut.endswith("kelime")
    assert truncate_passage("kısa  metin", 2500) == "kısa metin" and truncate_passage(text, 0) == " ".join(text.split())
