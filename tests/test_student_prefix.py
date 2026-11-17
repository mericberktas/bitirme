"""The student sees a query instruction on the question texts (`student.query_prefix`)."""

import copy
import hashlib
import json

import pytest
import torch
import yaml

from conftest import CharTokenizer, make_texts
from test_train_smoke import build
from turkish_distill.config import BANK_FIELDS, load_config
from turkish_distill.data import student_texts
from turkish_distill.train import run_training

PREFIX = "Instruct: ara\nQuery:"


class RecordingTokenizer(CharTokenizer):
    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts, *args, **kwargs):
        self.seen.extend(texts)
        return super().__call__(texts, *args, **kwargs)


def test_student_texts_prefixes_only_the_questions():
    texts = ["soru bir?", "pasaj", "soru iki?", "pasaj"]
    out, n = student_texts(texts, PREFIX, {"soru bir?", "soru iki?", "baska"})
    assert out == [PREFIX + "soru bir?", "pasaj", PREFIX + "soru iki?", "pasaj"] and n == 2
    assert student_texts(texts, "", {"soru bir?"}) == (texts, 0)


def write_questions(path, questions):
    path.write_text("".join(json.dumps({"question": q}, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    return str(path)


def test_training_feeds_the_prefixed_questions_and_the_same_targets(experiment, tmp_path):
    texts = make_texts(120)
    questions = texts[:30:3]  # 10 training texts are questions
    exp = copy.deepcopy(experiment)
    exp.student.query_prefix = PREFIX
    exp.student.query_sources = [write_questions(tmp_path / "q.jsonl", questions)]
    model, caches = build(exp, texts)
    tok = RecordingTokenizer()
    run_training(exp, tmp_path / "run", model, tok, texts, caches, torch.device("cpu"))
    prefixed = {t for t in tok.seen if t.startswith(PREFIX)}
    assert prefixed and prefixed <= {PREFIX + q for q in questions}
    assert not any(t in questions for t in tok.seen)  # a question is never fed plain
    assert all(t.startswith(PREFIX) or t not in questions for t in tok.seen)
    with pytest.raises(ValueError, match="none of the corpus"):
        bad = copy.deepcopy(exp)
        bad.student.query_sources = [write_questions(tmp_path / "other.jsonl", ["yok boyle bir soru"])]
        run_training(bad, tmp_path / "run2", model, tok, texts, caches, torch.device("cpu"))


def test_retrieval_check_prefixes_the_validation_questions(experiment, tmp_path):
    texts = make_texts(120)
    exp = copy.deepcopy(experiment)
    exp.student.query_prefix = PREFIX
    exp.student.query_sources = [write_questions(tmp_path / "q.jsonl", texts[:5])]
    model, caches = build(exp, texts)
    tok = RecordingTokenizer()
    val = {"question": ["ilk soru", "ikinci soru"], "passage": ["ilk pasaj", "ikinci pasaj"],
           "passage_id": ["p1", "p2"]}
    run_training(exp, tmp_path / "run", model, tok, texts, caches, torch.device("cpu"), retrieval_val=val)
    assert PREFIX + "ilk soru" in tok.seen and "ilk soru" not in tok.seen
    assert "ilk pasaj" in tok.seen and PREFIX + "ilk pasaj" not in tok.seen


def test_config_validation_and_fingerprint(tmp_path):
    raw = {"name": "x", "output_dir": str(tmp_path), "teachers": [{"name": "t", "hf_id": "h", "dim": 4}]}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    data = cfg.to_dict()
    data.pop("joint")
    for key in BANK_FIELDS:  # off by default too (test_bank.py)
        data["loss"].pop(key)
    for key in ("query_prefix", "query_sources"):
        data["student"].pop(key)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
    assert cfg.fingerprint() == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]  # unchanged while off
    on = load_config(path, ["student.query_prefix=Q:", "student.query_sources=[q.jsonl]"])
    assert on.student.query_sources == ["q.jsonl"] and on.fingerprint() != cfg.fingerprint()
    with pytest.raises(ValueError, match="together"):
        load_config(path, ["student.query_prefix=Q:"])
