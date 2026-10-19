import json

from turkish_distill import prepare_corpus as pc


def rows():
    return [
        {"text": "Kisa bir belge metni, yeterince uzun degil mi? " * 3, "url": "https://haber.example/a"},
        {"text": "uzun " * 400, "url": "https://forum.example/b"},
        {"text": 12345, "url": "https://bad.example/c"},          # not a string: skipped
        {"text": "Adressiz belge metni, ama yeterince uzun bir metin."},
    ]


def test_passages_keep_the_url_and_respect_the_limits():
    items = list(pc.passages_from_rows(rows(), "text", "url", max_chars=300, min_chars=20, max_passages_per_doc=2, max_samples=100))
    assert [i.get("url") for i in items] == ["https://haber.example/a", "https://forum.example/b", "https://forum.example/b", None]
    assert all(len(i["text"]) <= 300 for i in items)  # the long document gives two passages, not seven
    assert len(list(pc.passages_from_rows(rows(), "text", "url", 300, 20, 2, max_samples=2))) == 2
    assert "url" not in list(pc.passages_from_rows(rows(), "text", None, 300, 20, 2, 10))[0]


def test_write_is_atomic_and_counts(tmp_path, capsys):
    out = tmp_path / "c.jsonl"
    n = pc.write_jsonl_atomic(out, ({"text": f"metin {i}"} for i in range(5)), progress_every=2)
    assert n == 5 and len(out.read_text(encoding="utf-8").splitlines()) == 5
    assert not list(tmp_path.glob("*.tmp"))
    assert "2 passages written" in capsys.readouterr().out

    def broken():
        yield {"text": "ilk"}
        raise ConnectionError("stream dropped")

    final = tmp_path / "d.jsonl"
    try:
        pc.write_jsonl_atomic(final, broken())
    except ConnectionError:
        pass
    assert not final.exists()  # the half-written file never gets the final name
