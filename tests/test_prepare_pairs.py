import csv
import json

import pytest

from turkish_distill import prepare_pairs as pp


def write_turhistquad(directory, train_rows, test_rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.json").write_text(json.dumps(train_rows), encoding="utf-8")
    (directory / "test.json").write_text(json.dumps(test_rows), encoding="utf-8")


def thq(row_id, question, context):
    return {"id": row_id, "question": question, "context": context, "answers": {"answer_start": 0, "text": "x"}}


def write_wikirag(path, rows):
    fields = ["id", "question", "answer", "context", "is_negative_response", "number_of_articles", "ctx_split_points", "correct_intro_idx"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rag_row(i, question, articles, correct, negative=0):
    context, points = "", []
    for k, article in enumerate(articles):
        last = k == len(articles) - 1
        context += article + ("" if last else "\n")
        # as in the real data: each end offset includes the joining newline, the last one is len(context) + 1
        points.append(len(context) + (1 if last else 0))
    return {"id": str(i), "question": question, "answer": "a", "context": context, "is_negative_response": negative,
            "number_of_articles": len(articles), "ctx_split_points": json.dumps(points), "correct_intro_idx": correct}


def test_turhistquad_drops_squad_ids_and_uses_test_as_validation(tmp_path):
    stats = {}
    write_turhistquad(
        tmp_path / "thq",
        [thq("1234", "Soru bir mi?", "Pasaj bir"), thq("56dde1d966d3e219004dad8d", "Normanlar kimdir?", "Rollo"),
         thq("1235", "Soru iki mi?", "Pasaj iki")],
        [thq("77", "Test sorusu mu?", "Pasaj test")],
    )
    train, val = pp.load_turhistquad(tmp_path / "thq", stats)
    assert [p["question"] for p in train] == ["Soru bir mi?", "Soru iki mi?"]
    assert [p["question"] for p in val] == ["Test sorusu mu?"]
    assert stats["turhistquad_squad_style_dropped"] == 1


def test_wikirag_selects_the_correct_article_and_drops_bad_rows(tmp_path):
    arts = ["Birinci makale metni.", "Ikinci makale metni burada.", "Ucuncu makale."]
    rows = [
        rag_row(1, "Soru 1?", arts, 0), rag_row(2, "Soru 2?", arts, 1), rag_row(3, "Soru 3?", arts, 2),
        rag_row(4, "Cevapsiz?", arts, -1, negative=1),
        rag_row(5, "Bozuk indeks?", arts, 7),
    ]
    write_wikirag(tmp_path / "w.csv", rows)
    stats = {}
    pairs = pp.load_wikirag(tmp_path / "w.csv", stats)
    assert [p["passage"] for p in pairs] == arts
    assert stats["wikirag_dropped"] == {"negative": 1, "bad_index": 1}


def test_wikiqa_reservoir_cap_length_filter_and_article_dedup(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    long_article = "uzun " * 1000
    rows = []
    for i in range(40):
        qas = [{"questionNumber": k, "questionText": f"Makale {i} hakkinda soru {k} nedir?", "answerText": "a"} for k in range(6)]
        rows.append({"original_text": f"Makale {i} metni " + "bilgi " * 30, "answers": str(qas)})
    rows.append({"original_text": long_article, "answers": str([{"questionText": "Uzun makale sorusu nedir?", "answerText": "a"}])})
    rows.append(dict(rows[0]))  # exact duplicate article text
    pq.write_table(pa.table({"original_text": [r["original_text"] for r in rows], "answers": [r["answers"] for r in rows]}), tmp_path / "q.parquet")
    stats = {}
    pairs = pp.load_wikiqa(tmp_path / "q.parquet", max_pairs=50, max_chars=2000, min_chars=50, max_q_per_article=2, seed=0, stats=stats)
    assert len(pairs) == 50  # reservoir cap; 40 articles x 2 questions = 80 eligible
    assert stats["wikiqa"]["eligible_pairs_seen"] == 80
    assert stats["wikiqa"]["articles_too_long"] == 1 and stats["wikiqa"]["duplicate_articles"] == 1
    per_article = {}
    for p in pairs:
        per_article[p["passage_id"]] = per_article.get(p["passage_id"], 0) + 1
    assert max(per_article.values()) <= 2
    again = pp.load_wikiqa(tmp_path / "q.parquet", 50, 2000, 50, 2, 0, {})
    assert [p["question"] for p in again] == [p["question"] for p in pairs]  # deterministic for a seed


def pair(source, question, passage):
    return pp.make_pair(source, question, passage)


def test_merge_removes_training_pairs_that_share_a_passage_with_validation():
    shared = "Ortak pasaj metni"
    train = [pair("a", "Soru bir mi?", shared), pair("a", "Soru iki mi?", "Baska pasaj"), pair("a", "soru iki mi?", "Baska pasaj")]
    val = [pair("b", "Dogrulama sorusu mu?", shared)]
    stats = {}
    tr, va = pp.merge([train], [val], stats=stats)
    assert [p["question"] for p in tr] == ["Soru iki mi?"]  # shared-passage pair dropped, case-insensitive duplicate removed
    assert len(va) == 1 and stats["train_dropped_for_val_passage_overlap"] == 1
    assert tr[0]["id"] == "a:0000001" and va[0]["id"] == "b:0000001"


def test_merge_splits_wikirag_into_val_and_train_by_seed():
    rag = [pair("wikirag", f"Soru {i}?", f"Pasaj numara {i}") for i in range(20)]
    tr1, va1 = pp.merge([], [], wikirag_val_size=5, wikirag_pairs=rag, seed=1)
    tr2, va2 = pp.merge([], [], wikirag_val_size=5, wikirag_pairs=rag, seed=1)
    assert len(va1) == 5 and len(tr1) == 15
    assert [p["question"] for p in va1] == [p["question"] for p in va2]
    assert not {p["passage_id"] for p in va1} & {p["passage_id"] for p in tr1}


def test_merge_caps_questions_per_passage_in_training_only():
    train = [pair("s", f"Soru {i} nedir?", "Cok sorulu pasaj") for i in range(8)] + [pair("s", "Tek soru mu?", "Tek pasaj")]
    val = [pair("v", f"Dogrulama {i} mi?", "Dogrulama pasaji") for i in range(6)]
    stats = {}
    tr, va = pp.merge([train], [val], seed=3, stats=stats, max_q_per_passage=3)
    assert len(tr) == 4 and stats["train_dropped_for_passage_cap"] == 5
    assert sum(p["passage"] == "Cok sorulu pasaj" for p in tr) == 3
    assert len(va) == 6  # validation is not capped
    tr2, _ = pp.merge([train], [val], seed=3, max_q_per_passage=3)
    assert [p["question"] for p in tr2] == [p["question"] for p in tr]  # deterministic


def test_corpus_texts_are_unique_in_first_seen_order():
    pairs = [pair("s", "Soru bir?", "Pasaj"), pair("s", "Soru iki?", "Pasaj")]
    assert pp.corpus_texts(pairs) == ["Soru bir?", "Pasaj", "Soru iki?"]


def test_corpus_texts_exclude_validation_texts():
    train = [pair("s", "Ortak soru mu?", "Pasaj bir"), pair("s", "Ozel soru mu?", "Pasaj iki")]
    val = [pair("v", "Ortak soru mu?", "Baska pasaj")]
    assert pp.corpus_texts(train, exclude=val) == ["Pasaj bir", "Ozel soru mu?", "Pasaj iki"]


def test_decontaminate_exact_and_ngram_overlap():
    leaked_passage = " ".join(f"kelime{i}" for i in range(30))
    pairs = [
        pair("s", "Temiz soru nedir?", "Tamamen baska bir pasaj metni"),
        pair("s", "Temiz soru iki nedir?", "bir iki " + leaked_passage + " uc dort"),  # shares a 13-gram with the forbidden text
        pair("s", "Test sorusunun kendisi", "Alakasiz pasaj"),
    ]
    kept, dropped = pp.decontaminate(pairs, [leaked_passage, "test sorusunun   KENDISI"])
    assert [p["question"] for p in kept] == ["Temiz soru nedir?"] and dropped == 2


def test_main_writes_all_outputs(tmp_path):
    write_turhistquad(tmp_path / "thq", [thq("1", "Soru bir mi?", "Pasaj bir")], [thq("2", "Test mi?", "Pasaj test")])
    arts = ["Birinci makale metni.", "Ikinci makale metni burada."]
    write_wikirag(tmp_path / "w.csv", [rag_row(i, f"Rag soru {i}?", arts, i % 2) for i in range(4)])
    out = tmp_path / "out"
    pp.main(["--turhistquad-dir", str(tmp_path / "thq"), "--wikirag-csv", str(tmp_path / "w.csv"),
             "--wikirag-val-size", "1", "--out-dir", str(out)])
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["pairs_val"] == 2 and stats["decontaminated"] is False
    corpus = [json.loads(line)["text"] for line in open(out / "corpus_train.jsonl", encoding="utf-8")]
    assert len(corpus) == len(set(corpus)) == stats["corpus_train_texts"]
    val_questions = {json.loads(line)["question"] for line in open(out / "pairs_val.jsonl", encoding="utf-8")}
    assert not val_questions & set(corpus)  # no validation question leaks into the distillation corpus


def test_main_with_forbidden_file_drops_leaked_pairs_in_both_splits(tmp_path):
    leaked = " ".join(f"sizinti{i}" for i in range(20))
    write_turhistquad(
        tmp_path / "thq",
        [thq("1", "Temiz soru bir mi?", "Temiz pasaj bir"), thq("2", "Sizan pasajli soru mu?", "giris " + leaked + " cikis")],
        [thq("3", "Dogrulama sorusu mu?", "Dogrulama pasaji"), thq("4", "Sizan dogrulama mi?", "x " + leaked)],
    )
    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text(json.dumps({"text": leaked}) + "\n" + json.dumps({"text": "alakasiz benchmark metni burada"}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    pp.main(["--turhistquad-dir", str(tmp_path / "thq"), "--forbidden", str(forbidden), "--out-dir", str(out)])
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["decontaminated"] is True
    assert stats["train_dropped_decontamination"] == 1 and stats["val_dropped_decontamination"] == 1
    assert [json.loads(l)["question"] for l in open(out / "pairs_train.jsonl", encoding="utf-8")] == ["Temiz soru bir mi?"]
