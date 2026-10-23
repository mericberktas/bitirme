"""Teacher encoding with an on-disk shard cache (offline distillation).

Every teacher vector is L2-normalised and stored as float16 in `.npy` shards:

    <cache_dir>/<teacher>/<corpus_fingerprint>/shard_00000.npy
                                              /meta.json   (written last)

Shards are written atomically and skipped if present, so an interrupted
encoding job (e.g. a Colab disconnect) resumes where it stopped. The corpus
fingerprint ties a cache to the exact text list it was computed from.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import TeacherConfig

EncodeFn = Callable[[list[str]], np.ndarray]


def corpus_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


def cache_dir_for(cache_root: str | Path, teacher: TeacherConfig, fingerprint: str) -> Path:
    return Path(cache_root) / teacher.name / fingerprint


def _shard_path(directory: Path, shard: int) -> Path:
    return directory / f"shard_{shard:05d}.npy"


# A lock file stops two sessions (e.g. two Colab notebooks) from encoding the same teacher into the same cache
# at once, which would waste a GPU. It is refreshed after every shard and ignored once it is older than the
# timeout, so a crashed session does not block the cache for long. Shards are written atomically, so the lock is
# a courtesy, not what keeps the cache intact; note that Drive may show a new file to another session late.
LOCK_NAME = "encoding.lock"
LOCK_TIMEOUT_S = 3600.0  # longer than one shard of the slowest teacher (Qwen3-8B: ~26 min per 10K texts on an L4)


def _acquire_lock(directory: Path, teacher: TeacherConfig, timeout_s: float, ignore: bool) -> Path:
    lock = directory / LOCK_NAME
    if lock.exists() and not ignore:
        age = time.time() - lock.stat().st_mtime
        if age < timeout_s:
            try:
                holder = json.loads(lock.read_text(encoding="utf-8")).get("host", "unknown")
            except (OSError, ValueError):
                holder = "unknown"
            raise RuntimeError(
                f"teacher {teacher.name}: another session (host {holder}) is encoding into {directory}; its lock was "
                f"refreshed {age / 60:.0f} min ago. Wait for it, or pass --ignore-lock if that session is dead."
            )
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started": time.time()}), encoding="utf-8")
    return lock


def _touch_lock(lock: Path) -> None:
    try:
        os.utime(lock)
    except OSError:
        pass


def _release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def cache_status(teacher: TeacherConfig, cache_root: str | Path, fingerprint: str, n_texts: int, shard_size: int) -> dict:
    """How far a teacher's cache for this corpus is (file existence only; shard contents are not validated)."""
    directory = cache_dir_for(cache_root, teacher, fingerprint)
    num_shards = math.ceil(n_texts / shard_size)
    done = sum(1 for shard in range(num_shards) if _shard_path(directory, shard).exists())
    lock = directory / LOCK_NAME
    return {
        "teacher": teacher.name,
        "directory": str(directory),
        "shards_done": done,
        "num_shards": num_shards,
        "complete": (directory / "meta.json").exists() and done == num_shards,
        "lock_age_s": (time.time() - lock.stat().st_mtime) if lock.exists() else None,
    }


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def make_sentence_transformer_encoder(teacher: TeacherConfig) -> tuple[EncodeFn, Callable[[], None]]:
    """Build an encode function backed by sentence-transformers. Returns (fn, close)."""
    import torch
    from sentence_transformers import SentenceTransformer

    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if teacher.dtype not in dtypes:
        raise ValueError(f"teacher {teacher.name}: unknown dtype {teacher.dtype!r}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        teacher.hf_id,
        device=device,
        trust_remote_code=teacher.trust_remote_code,
        revision=teacher.revision,
        model_kwargs={"torch_dtype": dtypes[teacher.dtype]},
    )
    model.max_seq_length = teacher.max_length

    def encode(batch: list[str]) -> np.ndarray:
        return model.encode(
            batch,
            batch_size=teacher.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def close() -> None:
        nonlocal model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return encode, close


def load_query_set(paths: list[str]) -> set[str]:
    """The question texts listed in the `question` field of JSONL files (the prepare_pairs outputs)."""
    questions: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.add(json.loads(line)["question"])
    return questions


def _encode_chunk(
    teacher: TeacherConfig,
    chunk: list[str],
    is_query: Optional[list[bool]],
    encode_fn: EncodeFn,
    base_vectors: Optional[np.ndarray],
) -> np.ndarray:
    """Vectors of one shard of texts: questions with the query prefix, the rest plain (or copied from `base_vectors`)."""
    if is_query is None:
        return _normalize(np.asarray(encode_fn([teacher.doc_prefix + t for t in chunk])))
    out = np.empty((len(chunk), teacher.dim), dtype=np.float32)
    q_pos = [i for i, flag in enumerate(is_query) if flag]
    o_pos = [i for i, flag in enumerate(is_query) if not flag]
    if q_pos:
        out[q_pos] = _normalize(np.asarray(encode_fn([teacher.query_prefix + chunk[i] for i in q_pos])))
    if o_pos:
        if base_vectors is not None:
            out[o_pos] = base_vectors[o_pos]
        else:
            out[o_pos] = _normalize(np.asarray(encode_fn([teacher.doc_prefix + chunk[i] for i in o_pos])))
    return out


def encode_teacher(
    teacher: TeacherConfig,
    texts: list[str],
    cache_root: str | Path,
    shard_size: int,
    encode_fn: Optional[EncodeFn] = None,
    fingerprint: Optional[str] = None,
    verbose: bool = True,
    lock_timeout_s: float = LOCK_TIMEOUT_S,
    ignore_lock: bool = False,
) -> Path:
    """Encode `texts` with one teacher into the shard cache. Resumable."""
    fingerprint = fingerprint or corpus_fingerprint(texts)
    directory = cache_dir_for(cache_root, teacher, fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    num_shards = math.ceil(len(texts) / shard_size)

    # Shards are addressed by index, so resuming with a different shard_size would silently pair
    # vectors with the wrong texts. Refuse to continue a cache built with another shard_size.
    meta_path = directory / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8")).get("shard_size")
        if previous is not None and previous != shard_size:
            raise ValueError(
                f"teacher {teacher.name}: {directory} was built with shard_size={previous}, "
                f"now shard_size={shard_size}. Use the old value or delete that directory."
            )

    query_mask: Optional[list[bool]] = None
    base_cache = None
    if teacher.query_prefix:
        queries = load_query_set(teacher.query_sources)
        query_mask = [t in queries for t in texts]
        if teacher.reuse_plain_from:
            base_cfg = dataclasses.replace(
                teacher, name=teacher.reuse_plain_from, query_prefix="", query_sources=[], reuse_plain_from=None
            )
            base_cache = TeacherCache.open(cache_root, base_cfg, fingerprint, len(texts))
            if base_cache.meta.get("doc_prefix", "") != teacher.doc_prefix:
                raise ValueError(
                    f"teacher {teacher.name}: the plain cache {teacher.reuse_plain_from} used doc_prefix "
                    f"{base_cache.meta.get('doc_prefix')!r}, this teacher uses {teacher.doc_prefix!r}"
                )
        if verbose:
            print(f"[{teacher.name}] {sum(query_mask)} of {len(texts)} texts are questions and get the query prefix", flush=True)

    lock = _acquire_lock(directory, teacher, lock_timeout_s, ignore_lock)
    close = lambda: None  # noqa: E731
    try:
        if encode_fn is None:
            encode_fn, close = make_sentence_transformer_encoder(teacher)
        for shard in range(num_shards):
            path = _shard_path(directory, shard)
            lo, hi = shard * shard_size, min((shard + 1) * shard_size, len(texts))
            if path.exists():
                # A shard written under another shard_size (a partial cache has no meta.json yet)
                # or a damaged file must not be silently accepted.
                try:
                    shape = np.load(path, mmap_mode="r").shape
                except Exception as exc:
                    raise ValueError(f"teacher {teacher.name}: shard {shard} ({path}) is unreadable: {exc}") from exc
                if shape != (hi - lo, teacher.dim):
                    raise ValueError(
                        f"teacher {teacher.name}: existing shard {shard} has shape {shape}, expected "
                        f"{(hi - lo, teacher.dim)}. Was the cache built with a different shard_size? "
                        f"Use the original value or delete {directory}."
                    )
                continue
            base_vectors = base_cache.get(np.arange(lo, hi)) if base_cache is not None else None
            vectors = _encode_chunk(
                teacher, texts[lo:hi], query_mask[lo:hi] if query_mask is not None else None, encode_fn, base_vectors
            )
            if vectors.shape != (hi - lo, teacher.dim):
                raise ValueError(
                    f"teacher {teacher.name}: expected shape {(hi - lo, teacher.dim)}, got {vectors.shape}. "
                    "Check `dim` in the config."
                )
            tmp = path.with_suffix(".npy.tmp")
            with open(tmp, "wb") as f:
                np.save(f, vectors.astype(np.float16))
            os.replace(tmp, path)
            _touch_lock(lock)
            if verbose:
                print(f"[{teacher.name}] shard {shard + 1}/{num_shards} done", flush=True)
    finally:
        close()
        _release_lock(lock)

    meta = {
        "teacher": teacher.name,
        "hf_id": teacher.hf_id,
        "revision": teacher.revision,
        "dim": teacher.dim,
        "doc_prefix": teacher.doc_prefix,
        "query_prefix": teacher.query_prefix,
        "n_query_texts": sum(query_mask) if query_mask is not None else 0,
        "reuse_plain_from": teacher.reuse_plain_from,
        "max_length": teacher.max_length,
        "dtype": teacher.dtype,
        "n_texts": len(texts),
        "shard_size": shard_size,
        "num_shards": num_shards,
        "fingerprint": fingerprint,
    }
    with open(directory / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return directory


class TeacherCache:
    """Read-only access to a completed shard cache (memory-mapped)."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        meta_path = self.directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{self.directory} has no meta.json: the cache is missing or incomplete. "
                "Run `python -m turkish_distill.encode_teachers` first."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.shard_size = int(self.meta["shard_size"])
        self.n_texts = int(self.meta["n_texts"])
        self.dim = int(self.meta["dim"])
        self._shards: dict[int, np.ndarray] = {}
        for shard in range(int(self.meta["num_shards"])):
            if not _shard_path(self.directory, shard).exists():
                raise FileNotFoundError(f"missing shard {shard} in {self.directory}")

    @classmethod
    def open(
        cls, cache_root: str | Path, teacher: TeacherConfig, fingerprint: str, n_texts: int
    ) -> "TeacherCache":
        cache = cls(cache_dir_for(cache_root, teacher, fingerprint))
        if cache.dim != teacher.dim:
            raise ValueError(f"teacher {teacher.name}: cache dim {cache.dim} != config dim {teacher.dim}")
        if cache.n_texts != n_texts:
            raise ValueError(f"teacher {teacher.name}: cache has {cache.n_texts} texts, corpus has {n_texts}")
        if cache.meta.get("hf_id") != teacher.hf_id:
            raise ValueError(
                f"teacher {teacher.name}: cache was built with {cache.meta.get('hf_id')}, config says {teacher.hf_id}"
            )
        if teacher.revision is not None and cache.meta.get("revision") != teacher.revision:
            raise ValueError(
                f"teacher {teacher.name}: cache was built from revision {cache.meta.get('revision')}, "
                f"config pins {teacher.revision}"
            )
        return cache

    def __len__(self) -> int:
        return self.n_texts

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shards"] = {}
        return state

    def _shard(self, shard: int) -> np.ndarray:
        if shard not in self._shards:
            self._shards[shard] = np.load(_shard_path(self.directory, shard), mmap_mode="r")
        return self._shards[shard]

    def get(self, indices) -> np.ndarray:
        """Return float32 vectors for the given corpus indices (order preserved)."""
        indices = np.asarray(indices, dtype=np.int64)
        out = np.empty((len(indices), self.dim), dtype=np.float32)
        shard_ids = indices // self.shard_size
        for shard in np.unique(shard_ids):
            mask = shard_ids == shard
            local = indices[mask] - shard * self.shard_size
            out[mask] = self._shard(int(shard))[local]
        return out
