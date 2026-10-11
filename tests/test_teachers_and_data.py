import json

import numpy as np
import pytest

from conftest import fake_encode_fn, make_texts
from turkish_distill.config import TeacherConfig
from turkish_distill.data import epoch_batches, load_texts
from turkish_distill.teachers import TeacherCache, corpus_fingerprint, encode_teacher


def test_cache_roundtrip_matches_direct_encoding(tmp_path):
    texts = make_texts(95)  # 3 shards of 40, last one partial
    teacher = TeacherConfig("t", "fake/t", 8)
    enc = fake_encode_fn(8, seed=1)
    path = encode_teacher(teacher, texts, tmp_path, 40, encode_fn=enc, verbose=False)
    cache = TeacherCache(path)
    assert len(cache) == 95
    idx = [94, 0, 41, 40, 39, 80]
    got = cache.get(idx)
    expected = enc([texts[i] for i in idx])
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(got, expected, atol=2e-3)  # float16 storage
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=2e-3)


def test_encoding_resumes_and_skips_existing_shards(tmp_path):
    texts = make_texts(90)
    teacher = TeacherConfig("t", "fake/t", 4)
    calls = []

    def counting(batch):
        calls.append(len(batch))
        return fake_encode_fn(4, 0)(batch)

    path = encode_teacher(teacher, texts, tmp_path, 40, encode_fn=counting, verbose=False)
    assert calls == [40, 40, 10]
    (path / "meta.json").unlink()
    (path / "shard_00002.npy").unlink()
    calls.clear()
    encode_teacher(teacher, texts, tmp_path, 40, encode_fn=counting, verbose=False)
    assert calls == [10]  # only the missing shard is recomputed


def test_wrong_dim_is_reported(tmp_path):
    teacher = TeacherConfig("t", "fake/t", 5)
    with pytest.raises(ValueError, match="expected shape"):
        encode_teacher(teacher, make_texts(10), tmp_path, 10, encode_fn=fake_encode_fn(4, 0), verbose=False)


def test_open_validates_cache(tmp_path):
    texts = make_texts(30)
    teacher = TeacherConfig("t", "fake/t", 4)
    fp = corpus_fingerprint(texts)
    encode_teacher(teacher, texts, tmp_path, 40, encode_fn=fake_encode_fn(4, 0), fingerprint=fp, verbose=False)
    TeacherCache.open(tmp_path, teacher, fp, 30)  # fine
    with pytest.raises(ValueError, match="texts"):
        TeacherCache.open(tmp_path, teacher, fp, 31)
    with pytest.raises(ValueError, match="dim"):
        TeacherCache.open(tmp_path, TeacherConfig("t", "fake/t", 6), fp, 30)
    with pytest.raises(ValueError, match="built with"):
        TeacherCache.open(tmp_path, TeacherConfig("t", "other/model", 4), fp, 30)
    with pytest.raises(FileNotFoundError):
        TeacherCache.open(tmp_path, teacher, "0" * 16, 30)


def test_revision_is_recorded_and_checked(tmp_path):
    texts = make_texts(20)
    pinned = TeacherConfig("t", "fake/t", 4, revision="abc123")
    fp = corpus_fingerprint(texts)
    path = encode_teacher(pinned, texts, tmp_path, 40, encode_fn=fake_encode_fn(4, 0), fingerprint=fp, verbose=False)
    assert json.loads((path / "meta.json").read_text(encoding="utf-8"))["revision"] == "abc123"
    TeacherCache.open(tmp_path, pinned, fp, 20)  # matches
    # Config without a pin is accepted (nothing to compare against).
    TeacherCache.open(tmp_path, TeacherConfig("t", "fake/t", 4), fp, 20)
    # A different pinned revision must be rejected.
    with pytest.raises(ValueError, match="revision"):
        TeacherCache.open(tmp_path, TeacherConfig("t", "fake/t", 4, revision="def456"), fp, 20)


def test_fingerprint_depends_on_content_and_order():
    a = make_texts(10)
    assert corpus_fingerprint(a) == corpus_fingerprint(list(a))
    assert corpus_fingerprint(a) != corpus_fingerprint(a[::-1])
    assert corpus_fingerprint(a) != corpus_fingerprint(a[:-1])


def test_load_texts_filters_dedups_and_subsamples_deterministically(tmp_path):
    p = tmp_path / "c.jsonl"
    rows = [{"text": f"metin numarasi {i} yeterince uzun"} for i in range(50)]
    rows += [{"text": "kisa"}, {"text": rows[3]["text"]}, {"other": "x"}, {"text": None}]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    full = load_texts(p, min_chars=10)
    assert len(full) == 50
    sub1 = load_texts(p, min_chars=10, max_samples=20, seed=7)
    sub2 = load_texts(p, min_chars=10, max_samples=20, seed=7)
    assert sub1 == sub2 and len(sub1) == 20
    assert [full.index(t) for t in sub1] == sorted(full.index(t) for t in sub1)  # order preserved


def test_epoch_batches_deterministic_and_resumable():
    idx = np.arange(100)
    full = epoch_batches(idx, 8, seed=5, epoch=2)
    assert len(full) == 12 and all(len(b) == 8 for b in full)
    assert epoch_batches(idx, 8, seed=5, epoch=2) == full
    assert epoch_batches(idx, 8, seed=5, epoch=2, start_batch=5) == full[5:]
    assert epoch_batches(idx, 8, seed=5, epoch=3) != full
    flat = [i for b in full for i in b]
    assert len(set(flat)) == len(flat)  # no repeats within an epoch


def test_resuming_with_a_different_shard_size_is_refused(tmp_path):
    texts = make_texts(90)
    teacher = TeacherConfig("t", "fake/t", 4)
    # Interrupted run: shard 0 exists (40 rows), no meta.json yet.
    encode_teacher(teacher, texts[:40], tmp_path, 40, encode_fn=fake_encode_fn(4, 0), fingerprint="fp", verbose=False)
    directory = tmp_path / "t" / "fp"
    (directory / "meta.json").unlink()
    with pytest.raises(ValueError, match="different shard_size"):
        encode_teacher(teacher, texts, tmp_path, 25, encode_fn=fake_encode_fn(4, 0), fingerprint="fp", verbose=False)


def test_finished_cache_with_other_shard_size_is_refused_via_meta(tmp_path):
    texts = make_texts(60)
    teacher = TeacherConfig("t", "fake/t", 4)
    encode_teacher(teacher, texts, tmp_path, 30, encode_fn=fake_encode_fn(4, 0), fingerprint="fp", verbose=False)
    with pytest.raises(ValueError, match="built with shard_size=30"):
        encode_teacher(teacher, texts, tmp_path, 20, encode_fn=fake_encode_fn(4, 0), fingerprint="fp", verbose=False)


def test_damaged_shard_is_not_silently_accepted(tmp_path):
    texts = make_texts(50)
    teacher = TeacherConfig("t", "fake/t", 4)
    path = encode_teacher(teacher, texts, tmp_path, 25, encode_fn=fake_encode_fn(4, 0), fingerprint="fp", verbose=False)
    (path / "meta.json").unlink()
    (path / "shard_00001.npy").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="unreadable"):
        encode_teacher(teacher, texts, tmp_path, 25, encode_fn=fake_encode_fn(4, 0), fingerprint="fp", verbose=False)


def test_default_shard_size_keeps_the_work_lost_per_interruption_small():
    from turkish_distill.config import DataConfig
    assert DataConfig().shard_size == 10_000
