"""Fast STS evaluation used for monitoring during training.

This is not a replacement for TR-MTEB; it is a cheap sanity signal.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # average rank for ties
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rank(a), _rank(b))


def evaluate_sts(
    encode_fn: Callable[[list[str]], np.ndarray],
    sentences1: list[str],
    sentences2: list[str],
    scores: list[float],
) -> dict[str, float]:
    e1 = encode_fn(sentences1)
    e2 = encode_fn(sentences2)
    e1 = e1 / np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12, None)
    e2 = e2 / np.clip(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12, None)
    cos = (e1 * e2).sum(axis=1)
    gold = np.asarray(scores, dtype=np.float64)
    return {"sts_pearson": pearson(cos, gold), "sts_spearman": spearman(cos, gold)}


@torch.no_grad()
def make_student_encode_fn(model, tokenizer, device, max_length: int, batch_size: int = 128):
    def encode(texts: list[str]) -> np.ndarray:
        was_training = model.training
        model.eval()
        chunks = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            pooled = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            chunks.append(pooled.float().cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(chunks, axis=0)

    return encode


def load_sts_pairs(dataset_id: str, split: str, columns: Optional[list[str]] = None):
    """Load (sentence1, sentence2, score) lists from a HF dataset.

    `columns` must name the three columns explicitly unless the dataset uses
    the common names sentence1/sentence2/score. Column names of Turkish STS
    datasets on the Hub have not been verified here.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    if columns is None:
        columns = ["sentence1", "sentence2", "score"]
    missing = [c for c in columns if c not in ds.column_names]
    if missing:
        raise KeyError(
            f"dataset {dataset_id} has columns {ds.column_names}; "
            f"set eval.sts_columns explicitly (missing: {missing})"
        )
    c1, c2, cs = columns
    return list(ds[c1]), list(ds[c2]), [float(x) for x in ds[cs]]
