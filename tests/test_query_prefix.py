"""Query-instruction encoding: only question texts get the query prefix; the rest can be copied from a plain cache."""

import json

import numpy as np
import pytest

from turkish_distill.config import TeacherConfig, load_config
from turkish_distill.teachers import TeacherCache, corpus_fingerprint, encode_teacher, load_query_set

QUESTION = "Kim doğdu?"
PASSAGE = "Bir pasaj metni."
TEXTS = [QUESTION, PASSAGE, "İkinci soru nedir?", "Başka bir pasaj.", "Üçüncü soru?"]
QUESTIONS = {QUESTION, "İkinci soru nedir?", "Üçüncü soru?"}


def write_pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps({"question": q, "passage": "p"}) for q in sorted(QUESTIONS)), encoding="utf-8")
    return str(path)


class RecordingEncoder:
    """Deterministic fake model: the vector depends on the exact string it is given."""

    def __init__(self, dim=6):
        self.dim, self.seen = dim, []

    def __call__(self, batch):
        self.seen.extend(batch)
        out = np.zeros((len(batch), self.dim), dtype=np.float32)
        for i, text in enumerate(batch):
            out[i] = np.random.default_rng(abs(hash(text)) % (2**32)).normal(size=self.dim)
        return out


def teacher(tmp_path, **kwargs):
    return TeacherConfig(name="t", hf_id="x/t", dim=6, **kwargs)


def test_only_questions_get_the_query_prefix(tmp_path):
    fn = RecordingEncoder()
    t = teacher(tmp_path, query_prefix="Q: ", doc_prefix="D: ", query_sources=[write_pairs(tmp_path)])
    encode_teacher(t, TEXTS, tmp_path / "cache", 4, encode_fn=fn, verbose=False)
    assert sorted(fn.seen) == sorted(["Q: " + x for x in TEXTS if x in QUESTIONS] + ["D: " + x for x in TEXTS if x not in QUESTIONS])
    meta = json.loads((tmp_path / "cache" / "t" / corpus_fingerprint(TEXTS) / "meta.json").read_text(encoding="utf-8"))
    assert meta["query_prefix"] == "Q: " and meta["n_query_texts"] == 3


def test_reusing_a_plain_cache_only_encodes_the_questions(tmp_path):
    cache_root = tmp_path / "cache"
    plain_fn = RecordingEncoder()
    plain = TeacherConfig(name="plain", hf_id="x/t", dim=6)
    encode_teacher(plain, TEXTS, cache_root, 4, encode_fn=plain_fn, verbose=False)
    fp = corpus_fingerprint(TEXTS)
    plain_cache = TeacherCache.open(cache_root, plain, fp, len(TEXTS))

    fn = RecordingEncoder()
    instr = TeacherConfig(name="instr", hf_id="x/t", dim=6, query_prefix="Q: ", query_sources=[write_pairs(tmp_path)],
                          reuse_plain_from="plain")
    encode_teacher(instr, TEXTS, cache_root, 4, encode_fn=fn, verbose=False)
    assert sorted(fn.seen) == sorted("Q: " + x for x in TEXTS if x in QUESTIONS)  # passages never re-encoded
    new = TeacherCache.open(cache_root, instr, fp, len(TEXTS)).get(np.arange(len(TEXTS)))
    old = plain_cache.get(np.arange(len(TEXTS)))
    for i, text in enumerate(TEXTS):
        same = np.allclose(new[i], old[i], atol=1e-3)
        assert same == (text not in QUESTIONS), text  # passages identical, questions changed
    assert np.allclose(np.linalg.norm(new, axis=1), 1.0, atol=1e-2)


def test_reuse_requires_the_matching_plain_cache(tmp_path):
    instr = TeacherConfig(name="instr", hf_id="x/t", dim=6, query_prefix="Q: ", query_sources=[write_pairs(tmp_path)],
                          reuse_plain_from="missing")
    with pytest.raises(FileNotFoundError):
        encode_teacher(instr, TEXTS, tmp_path / "cache", 4, encode_fn=RecordingEncoder(), verbose=False)


def test_config_validation_of_the_query_options(tmp_path):
    import yaml

    base = {"name": "x", "output_dir": "o"}
    bad = [
        {"name": "t", "hf_id": "h", "dim": 4, "query_prefix": "Q: "},  # prefix without sources
        {"name": "t", "hf_id": "h", "dim": 4, "reuse_plain_from": "other"},  # reuse without prefix
        {"name": "t", "hf_id": "h", "dim": 4, "query_prefix": "Q: ", "query_sources": ["p"], "reuse_plain_from": "t"},
    ]
    for entry in bad:
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({**base, "teachers": [entry]}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


def test_load_query_set_reads_the_question_field(tmp_path):
    assert load_query_set([write_pairs(tmp_path)]) == QUESTIONS
