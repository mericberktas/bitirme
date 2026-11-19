"""Mix v3 (200K mix + FineWeb QA texts) and seeding a bigger corpus's teacher cache from an older one."""

import dataclasses
import json

import numpy as np
import pytest

from conftest import fake_encode_fn, make_texts
from turkish_distill import fineweb_mix
from turkish_distill.fineweb_mix import KINDS, build_v3, candidates, quotas
from turkish_distill.teachers import TeacherCache, corpus_fingerprint, encode_teacher, seed_cache_from

SENT = "Belediye bu yıl kentin dört farklı noktasında yeni parklar açmayı planladığını duyurdu ve çalışmalar başladı."


def make_doc(i, split="train"):
    para = f"{SENT} Bu {i}. haber metnidir ve ayrıntılar sonraki günlerde açıklanacak."
    syl = ("ka", "le", "mi", "to", "ru", "sa", "ne", "po", "di", "ya", "bu", "ze")
    words = [x + y + z for x in syl for y in syl for z in syl[:2]][(i % 7) * 10:(i % 7) * 10 + 140]
    long = " ".join(" ".join(words[k:k + 10]) + "." for k in range(0, len(words), 10)) + f" Haber numarası {i}."
    return {
        "id": f"d{i}", "split": split, "doc": f"{para}\n{long}\nkısa satır",
        "summary": f"Belediye {i} numaralı haberde yeni parklar açacağını duyurdu.",
        "qa": [{"question": f"Belediye kaç park açacak {i}?", "answer": f"Belediye dört yeni park açacak, haber {i}."}],
        "queries": [f"belediye yeni park haberi {i}", f"kent park planı {i}"],
    }


def test_candidates_one_text_per_kind():
    import random

    picked = candidates(make_doc(1), random.Random(0))
    assert set(picked) == set(KINDS)
    for kind, text in picked.items():
        assert KINDS[kind][1] <= len(text) <= KINDS[kind][2], kind
    assert quotas(1000) == {"fw_paragraphs": 300, "fw_long": 150, "fw_queries": 150, "fw_questions": 150,
                            "fw_answers": 125, "fw_summaries": 125}
    assert sum(quotas(997).values()) == 997


def test_build_v3_layout(tmp_path):
    base = tmp_path / "base.jsonl"
    base_rows = [{"text": f"temel karma metni numara {i} burada"} for i in range(50)]
    base.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base_rows), encoding="utf-8")
    bq = tmp_path / "bq.jsonl"
    bq.write_text(json.dumps({"question": "temel soru?"}, ensure_ascii=False) + "\n", encoding="utf-8")
    docs = tmp_path / "docs.jsonl"
    docs.write_text("".join(json.dumps(make_doc(i, "val" if i % 10 == 0 else "train"), ensure_ascii=False) + "\n"
                            for i in range(200)), encoding="utf-8")
    out, qout = tmp_path / "mix.jsonl", tmp_path / "q.jsonl"
    stats = build_v3(base, docs, out, qout, bq, n_fineweb=100, tail=5)
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert len(rows) == 150 and stats["total"] == 150
    assert rows[:45] == base_rows[:45] and rows[-5:] == base_rows[-5:]  # prefix and validation tail unchanged
    fw = rows[45:145]
    assert all(r["source"] == "fineweb_qa" for r in fw) and len({r["text"] for r in fw}) == 100
    assert not any(r["doc_id"] in {f"d{i}" for i in range(0, 200, 10)} for r in fw)  # val documents unused
    assert stats["selected"] == quotas(100)
    questions = [json.loads(line)["question"] for line in open(qout, encoding="utf-8")]
    assert questions[0] == "temel soru?"
    assert set(questions[1:]) == {r["text"] for r in fw if r["bucket"] in fineweb_mix.QUESTION_KINDS}
    with pytest.raises(ValueError, match="ran out"):
        build_v3(base, docs, tmp_path / "x.jsonl", tmp_path / "xq.jsonl", n_fineweb=10_000, tail=5)


def test_seed_cache_copies_matching_shards_only(teachers, tmp_path):
    teacher = teachers[0]
    old = make_texts(35)
    new = old[:20] + make_texts(30)[25:] + ["yeni metin a", "yeni metin b"] + old[20:]
    encode_teacher(teacher, old, tmp_path, 10, encode_fn=fake_encode_fn(teacher.dim, 0), verbose=False)
    assert seed_cache_from(teacher, new, old, tmp_path, 10, verbose=False) == 2  # shards 0 and 1 match
    calls = []

    def counting(batch):
        calls.append(len(batch))
        return fake_encode_fn(teacher.dim, 0)(batch)

    encode_teacher(teacher, new, tmp_path, 10, encode_fn=counting, verbose=False)
    assert sum(calls) == len(new) - 20
    fresh = tmp_path / "fresh"
    encode_teacher(teacher, new, fresh, 10, encode_fn=fake_encode_fn(teacher.dim, 0), verbose=False)
    a = TeacherCache.open(tmp_path, teacher, corpus_fingerprint(new), len(new)).get(np.arange(len(new)))
    b = TeacherCache.open(fresh, teacher, corpus_fingerprint(new), len(new)).get(np.arange(len(new)))
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="plain teachers"):
        seed_cache_from(dataclasses.replace(teacher, query_prefix="Q: "), new, old, tmp_path, 10, verbose=False)
    with pytest.raises(ValueError, match="shard_size"):
        seed_cache_from(teacher, new, old, tmp_path, 5, verbose=False)
