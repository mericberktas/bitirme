"""Corpus loading and batching."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _iter_raw(path: Path, column: str):
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line).get(column)
    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield row.get(column)
    elif suffix == ".parquet":
        import pandas as pd

        for value in pd.read_parquet(path, columns=[column])[column]:
            yield value
    else:
        raise ValueError(f"unsupported corpus format: {path.suffix} (use .jsonl, .txt, .csv or .parquet)")


def load_texts(
    path: str | Path,
    column: str = "text",
    min_chars: int = 20,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> list[str]:
    """Load, filter and deduplicate (exact match) a corpus, deterministically.

    When `max_samples` is smaller than the corpus, a seeded random subset is
    taken and the original order is preserved, so the validation tail is stable.
    """
    seen: set[str] = set()
    texts: list[str] = []
    for value in _iter_raw(Path(path), column):
        if not isinstance(value, str):
            continue
        value = value.strip()
        if len(value) < min_chars or value in seen:
            continue
        seen.add(value)
        texts.append(value)
    if max_samples is not None and len(texts) > max_samples:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(texts), size=max_samples, replace=False))
        texts = [texts[i] for i in keep]
    return texts


def load_corpus(data_cfg, seed: int) -> list[str]:
    """Load the corpus described by a DataConfig (shared by encoding and training)."""
    if not data_cfg.corpus_path:
        raise ValueError("data.corpus_path is empty")
    return load_texts(
        data_cfg.corpus_path,
        column=data_cfg.text_column,
        min_chars=data_cfg.min_chars,
        max_samples=data_cfg.max_samples,
        seed=seed,
    )


class ListDataset(torch.utils.data.Dataset):
    """Each item is a precomputed list of corpus indices (one batch)."""

    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> list[int]:
        return self.batches[i]


class Collator:
    """Turns a list of corpus indices into a tokenised batch plus teacher targets."""

    def __init__(self, tokenizer: Any, texts: list[str], caches: dict[str, Any], max_length: int):
        self.tokenizer = tokenizer
        self.texts = texts
        self.caches = caches
        self.max_length = max_length

    def __call__(self, indices: list[int]) -> dict[str, Any]:
        enc = self.tokenizer(
            [self.texts[i] for i in indices],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        targets = {name: torch.from_numpy(cache.get(indices)).float() for name, cache in self.caches.items()}
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "targets": targets,
            "indices": indices,
        }


def epoch_batches(train_idx: np.ndarray, batch_size: int, seed: int, epoch: int, start_batch: int = 0) -> list[list[int]]:
    """Deterministic shuffled batches for one epoch (last partial batch dropped)."""
    perm = np.random.default_rng(seed + epoch).permutation(train_idx)
    steps = len(perm) // batch_size
    return [perm[i * batch_size : (i + 1) * batch_size].tolist() for i in range(start_batch, steps)]
