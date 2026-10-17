import io
import json

import pytest

from turkish_distill import benchmark_texts as bt

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def parquet_bytes(table: dict) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.table(table), buffer)
    return buffer.getvalue()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_opener(files: dict[str, bytes], commit: str = "c0ffee" * 6 + "ab"):
    requested = []

    def opener(url: str):
        requested.append(url)
        if "/api/datasets/" in url:
            return FakeResponse(json.dumps({"sha": commit}).encode())
        return FakeResponse(files[url.split("/resolve/")[1]])

    opener.requested = requested
    return opener, commit


def test_download_pins_the_commit_skips_finished_files_and_writes_a_manifest(tmp_path):
    repo = "trmteb/stsb-tr"
    payload = parquet_bytes({"sentence1": ["bir cumle"], "sentence2": ["baska cumle"], "score": [3.5]})
    opener, commit = make_opener({f"{'c0ffee' * 6 + 'ab'}/data/test-00000-of-00001.parquet": payload})
    manifest = bt.download(tmp_path, opener, repos=[repo])
    assert manifest[repo]["commit"] == commit and manifest[repo]["requested_revision"] == "main"
    assert (tmp_path / repo / "data/test-00000-of-00001.parquet").read_bytes() == payload
    assert json.loads((tmp_path / "manifest.json").read_text())[repo]["commit"] == commit
    assert not list(tmp_path.rglob("*.part"))

    before = len(opener.requested)
    bt.download(tmp_path, opener, repos=[repo])  # a second run downloads nothing new
    assert [u for u in opener.requested[before:] if "/resolve/" in u] == []


def test_a_half_written_download_never_gets_the_final_name(tmp_path):
    class Broken(FakeResponse):
        def read(self, n=-1):
            raise ConnectionError("dropped")

    def opener(url):
        return FakeResponse(json.dumps({"sha": "abc"}).encode()) if "/api/" in url else Broken(b"x")

    with pytest.raises(ConnectionError):
        bt.download(tmp_path, opener, repos=["trmteb/stsb-tr"])
    assert not (tmp_path / "trmteb/stsb-tr/data/test-00000-of-00001.parquet").exists()


def test_texts_are_collected_from_nested_cells_and_short_cells_are_skipped(tmp_path):
    path = tmp_path / "t.parquet"
    path.write_bytes(parquet_bytes({
        "id": ["q1", "q2"], "label": ["pos", "neg"],
        "translation": [{"en": "an english sentence long enough", "tr": "yeterince uzun bir turkce cumle"}, {"en": "short", "tr": "kisa"}],
        "tags": [["ilk uzun etiket metni burada", "kisa"], []],
    }))
    texts = set(bt.iter_file_texts(path))
    assert {"an english sentence long enough", "yeterince uzun bir turkce cumle", "ilk uzun etiket metni burada"} <= texts
    csv_path = tmp_path / "t.csv"
    csv_path.write_text('text,label\n"Bu bir yorum metnidir, uzun",1\n', encoding="utf-8")
    assert "Bu bir yorum metnidir, uzun" in set(bt.iter_file_texts(csv_path))


def test_export_deduplicates_normalises_whitespace_and_needs_every_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "BENCHMARK_FILES", {
        "a/one": ("main", ("data/test.parquet",)),
        "b/two": ("main", ("data/test.parquet",)),
    })
    for repo, rows in {"a/one": ["ayni uzun cumle metni", "yalnizca ilkinde olan metin"], "b/two": ["ayni   uzun\ncumle metni", "ikincide olan baska metin"]}.items():
        (tmp_path / repo / "data").mkdir(parents=True)
        (tmp_path / repo / "data/test.parquet").write_bytes(parquet_bytes({"text": rows}))
    stats = bt.export(tmp_path, tmp_path / "out.jsonl")
    lines = [json.loads(line)["text"] for line in open(tmp_path / "out.jsonl", encoding="utf-8")]
    assert lines == ["ayni uzun cumle metni", "yalnizca ilkinde olan metin", "ikincide olan baska metin"]
    assert stats["new_texts_per_dataset"] == {"a/one": 2, "b/two": 1}

    (tmp_path / "b/two/data/test.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        bt.export(tmp_path, tmp_path / "out2.jsonl")


def test_file_map_covers_all_26_tasks_with_retrieval_corpora_and_queries():
    assert len(bt.BENCHMARK_FILES) == 26
    retrieval = [files for _, files in bt.BENCHMARK_FILES.values() if any(f.startswith("corpus/") for f in files)]
    assert len(retrieval) == 10 and all(any(f.startswith("queries/") for f in files) for files in retrieval)
