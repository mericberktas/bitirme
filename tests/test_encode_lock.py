import json
import os
import time

import pytest

from conftest import fake_encode_fn, make_texts
from turkish_distill import encode_teachers
from turkish_distill.config import dump_config
from turkish_distill.data import load_corpus
from turkish_distill.teachers import LOCK_NAME, cache_dir_for, cache_status, corpus_fingerprint, encode_teacher


def encode(teacher, texts, root, **kwargs):
    return encode_teacher(teacher, texts, root, 10, encode_fn=fake_encode_fn(teacher.dim, 0), verbose=False, **kwargs)


def test_a_fresh_lock_from_another_session_blocks_encoding(teachers, tmp_path):
    teacher, texts = teachers[0], make_texts(25)
    directory = cache_dir_for(tmp_path, teacher, corpus_fingerprint(texts))
    directory.mkdir(parents=True)
    (directory / LOCK_NAME).write_text(json.dumps({"host": "colab-vm-2"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="another session.*colab-vm-2"):
        encode(teacher, texts, tmp_path)
    assert not list(directory.glob("shard_*.npy"))  # nothing was encoded
    encode(teacher, texts, tmp_path, ignore_lock=True)  # a dead session's lock can be overridden
    assert len(list(directory.glob("shard_*.npy"))) == 3


def test_a_stale_lock_is_ignored_and_the_lock_is_always_released(teachers, tmp_path):
    teacher, texts = teachers[0], make_texts(25)
    directory = cache_dir_for(tmp_path, teacher, corpus_fingerprint(texts))
    directory.mkdir(parents=True)
    lock = directory / LOCK_NAME
    lock.write_text("{}", encoding="utf-8")
    old = time.time() - 2 * 3600
    os.utime(lock, (old, old))
    encode(teacher, texts, tmp_path)
    assert not lock.exists()  # released after a successful run

    def boom(batch):
        raise RuntimeError("out of memory")

    other = make_texts(40)
    with pytest.raises(RuntimeError, match="out of memory"):
        encode_teacher(teacher, other, tmp_path, 10, encode_fn=boom, verbose=False)
    assert not (cache_dir_for(tmp_path, teacher, corpus_fingerprint(other)) / LOCK_NAME).exists()  # and after a crash


def test_two_teachers_do_not_lock_each_other(teachers, tmp_path):
    texts = make_texts(25)
    directory = cache_dir_for(tmp_path, teachers[0], corpus_fingerprint(texts))
    directory.mkdir(parents=True)
    (directory / LOCK_NAME).write_text("{}", encoding="utf-8")  # teacher 0 is being encoded elsewhere
    encode(teachers[1], texts, tmp_path)  # teacher 1 has its own folder and is free
    assert cache_status(teachers[1], tmp_path, corpus_fingerprint(texts), 25, 10)["complete"]


def test_cache_status_counts_shards(teachers, tmp_path):
    teacher, texts = teachers[0], make_texts(25)
    fp = corpus_fingerprint(texts)
    empty = cache_status(teacher, tmp_path, fp, 25, 10)
    assert (empty["shards_done"], empty["num_shards"], empty["complete"], empty["lock_age_s"]) == (0, 3, False, None)
    encode(teacher, texts, tmp_path)
    done = cache_status(teacher, tmp_path, fp, 25, 10)
    assert (done["shards_done"], done["complete"]) == (3, True)
    (cache_dir_for(tmp_path, teacher, fp) / "shard_00002.npy").unlink()
    partial = cache_status(teacher, tmp_path, fp, 25, 10)
    assert (partial["shards_done"], partial["complete"]) == (2, False)


def test_cli_status_reports_without_loading_a_model(experiment, tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    texts = make_texts(25)
    corpus.write_text("\n".join(json.dumps({"text": t}) for t in texts), encoding="utf-8")
    experiment.data.corpus_path = str(corpus)
    experiment.data.shard_size = 10
    experiment.data.min_chars = 5
    config = tmp_path / "c.yaml"
    dump_config(experiment, config)
    loaded = load_corpus(experiment.data, experiment.train.seed)
    encode(experiment.teachers[0], loaded, experiment.data.cache_dir)
    encode_teachers.main(["--config", str(config), "--status"])
    out = capsys.readouterr().out
    assert "25 texts" in out and f"[{experiment.teachers[0].name}] 3/3 shards, complete" in out
    assert f"[{experiment.teachers[1].name}] 0/3 shards, incomplete" in out
    encode_teachers.main(["--config", str(config), "--status", "--only", experiment.teachers[1].name])
    only = capsys.readouterr().out
    assert f"[{experiment.teachers[1].name}]" in only and f"[{experiment.teachers[0].name}]" not in only
    with pytest.raises(ValueError, match="no teacher matches"):
        encode_teachers.main(["--config", str(config), "--status", "--only", "nope"])
