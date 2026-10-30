"""Contrastive baseline: in-batch-negative loss, gradient caching, batching and validation retrieval.

The data is a list of positive (question, passage) pairs, so the losses are the InfoNCE / "multiple negatives
ranking" family: every other passage in the batch is a negative for a question. Two things protect that
assumption:
  * `unique_passage_batches` never puts two pairs with the same passage into one batch (their questions
    would be false negatives for each other);
  * `contrastive_step` can encode the batch in mini-batches (gradient caching, as in GradCache / sentence-
    transformers' CachedMultipleNegativesRankingLoss), so a large batch does not need large memory.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .contrastive_config import ContrastiveLossConfig

LossFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]]


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def info_nce(
    q: torch.Tensor, p: torch.Tensor, scale: float, symmetric: bool = False
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy over cosine similarities; the i-th passage is the positive of the i-th question."""
    q = F.normalize(q.float(), dim=-1)
    p = F.normalize(p.float(), dim=-1)
    scores = q @ p.T * scale
    labels = torch.arange(scores.size(0), device=scores.device)
    loss = F.cross_entropy(scores, labels)
    if symmetric:
        loss = (loss + F.cross_entropy(scores.T, labels)) / 2
    accuracy = (scores.argmax(dim=1) == labels).float().mean()
    return loss, {"acc": accuracy.detach()}


def make_loss(cfg: ContrastiveLossConfig) -> LossFn:
    symmetric = cfg.name == "mnrl_symmetric"
    if cfg.name not in ("mnrl", "mnrl_symmetric"):
        raise ValueError(f"unknown contrastive loss {cfg.name!r}")
    return lambda q, p: info_nce(q, p, cfg.scale, symmetric)


def make_anchor_loss(cfg: ContrastiveLossConfig) -> Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Distillation toward a frozen reference model, on the pooled embeddings of the batch's questions and passages."""

    def fn(q: torch.Tensor, p: torch.Tensor, aq: torch.Tensor, ap: torch.Tensor):
        s = F.normalize(torch.cat([q, p]).float(), dim=-1)
        a = F.normalize(torch.cat([aq, ap]).float(), dim=-1)
        cosine = (s * a).sum(-1)
        gram = F.mse_loss(s @ s.T, a @ a.T)
        loss = cfg.anchor_cosine_weight * (1 - cosine).mean() + cfg.anchor_gram_weight * gram
        return loss, {"anchor_cos": cosine.mean().detach(), "anchor_gram": gram.detach(), "anchor_loss": loss.detach()}

    return fn


def make_anchor(model) -> torch.nn.Module:
    """Frozen copy of `model` (eval mode, no gradients): the reference the anchor loss pulls toward."""
    anchor = copy.deepcopy(model).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    return anchor


# --------------------------------------------------------------------------- #
# Training step (plain or gradient-cached)
# --------------------------------------------------------------------------- #

def _encode(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_amp: bool) -> torch.Tensor:
    device_type = input_ids.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        return model.encode(input_ids, attention_mask).float()


def _capture_rng(device: torch.device):
    return torch.get_rng_state(), (torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


@contextmanager
def _restore_rng(state, device: torch.device):
    cpu_state, cuda_state = state
    with torch.random.fork_rng(devices=[device] if cuda_state is not None else []):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        yield


def contrastive_step(
    model,
    batch: dict[str, torch.Tensor],
    loss_fn: LossFn,
    chunk_size: Optional[int] = None,
    use_amp: bool = False,
    anchor_model=None,
    anchor_fn=None,
) -> dict[str, float]:
    """One forward/backward pass; gradients are left on the parameters (the caller runs the optimiser).

    With `anchor_model` and `anchor_fn` (see `make_anchor`, `make_anchor_loss`) the loss also contains the anchor
    term; the anchor embeddings are computed without gradients, so the cached and plain steps still agree.

    With `chunk_size`, the embeddings of the whole batch are computed without gradients first, the loss and
    its gradient w.r.t. the embeddings are computed on the full batch, and then each mini-batch is encoded
    again with gradients and back-propagated with that stored embedding gradient. The result equals the plain
    step (up to numerical noise); the RNG state is restored so dropout masks are identical in both passes.
    """
    q_ids, q_mask = batch["q_input_ids"], batch["q_attention_mask"]
    p_ids, p_mask = batch["p_input_ids"], batch["p_attention_mask"]
    size = q_ids.size(0)
    anchored = anchor_model is not None and anchor_fn is not None

    def total_loss(q, p, aq=None, ap=None):
        loss, logs = loss_fn(q, p)
        if not anchored:
            return loss, logs
        anchor_term, anchor_logs = anchor_fn(q, p, aq, ap)
        return loss + anchor_term, {**logs, **anchor_logs, "nce_loss": loss.detach()}

    if chunk_size is None or chunk_size >= size:
        aq = ap = None
        if anchored:
            with torch.no_grad():
                aq, ap = _encode(anchor_model, q_ids, q_mask, use_amp), _encode(anchor_model, p_ids, p_mask, use_amp)
        loss, logs = total_loss(_encode(model, q_ids, q_mask, use_amp), _encode(model, p_ids, p_mask, use_amp), aq, ap)
        loss.backward()
        return {"loss": float(loss.detach()), **{k: float(v) for k, v in logs.items()}}

    device = q_ids.device
    slices = [slice(i, i + chunk_size) for i in range(0, size, chunk_size)]
    embeddings: dict[str, torch.Tensor] = {}
    states: dict[str, list] = {}
    with torch.no_grad():
        for side, ids, mask in (("q", q_ids, q_mask), ("p", p_ids, p_mask)):
            states[side], parts = [], []
            for sl in slices:
                states[side].append(_capture_rng(device))
                parts.append(_encode(model, ids[sl], mask[sl], use_amp))
            embeddings[side] = torch.cat(parts).requires_grad_(True)
        anchors = {}
        if anchored:
            for side, ids, mask in (("q", q_ids, q_mask), ("p", p_ids, p_mask)):
                anchors[side] = torch.cat([_encode(anchor_model, ids[sl], mask[sl], use_amp) for sl in slices])
    loss, logs = total_loss(embeddings["q"], embeddings["p"], anchors.get("q"), anchors.get("p"))
    loss.backward()  # fills .grad of the two embedding tensors only

    for side, ids, mask in (("q", q_ids, q_mask), ("p", p_ids, p_mask)):
        grad = embeddings[side].grad
        for sl, state in zip(slices, states[side]):
            with _restore_rng(state, device):
                chunk = _encode(model, ids[sl], mask[sl], use_amp)
            (chunk * grad[sl]).sum().backward()
    return {"loss": float(loss.detach()), **{k: float(v) for k, v in logs.items()}}


# --------------------------------------------------------------------------- #
# Data: pairs, batching without duplicate passages, collation
# --------------------------------------------------------------------------- #

def load_pairs(path: str | Path, max_pairs: Optional[int] = None, seed: int = 42) -> dict[str, list[str]]:
    """Read a pairs JSONL written by `prepare_pairs` into parallel lists."""
    questions, passages, ids = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                questions.append(row["question"])
                passages.append(row["passage"])
                ids.append(row["passage_id"])
    if max_pairs is not None and len(questions) > max_pairs:
        keep = np.sort(np.random.default_rng(seed).choice(len(questions), size=max_pairs, replace=False))
        questions, passages, ids = ([x[i] for i in keep] for x in (questions, passages, ids))
    return {"question": questions, "passage": passages, "passage_id": ids}


def unique_passage_batches(
    passage_ids: Sequence[str], batch_size: int, seed: int, epoch: int, start_batch: int = 0
) -> list[list[int]]:
    """Deterministic shuffled batches in which no passage occurs twice (the leftover tail is dropped)."""
    order = np.random.default_rng(seed + epoch).permutation(len(passage_ids)).tolist()
    batches: list[list[int]] = []
    remaining = order
    while len(remaining) >= batch_size:
        batch: list[int] = []
        seen: set[str] = set()
        rest: list[int] = []
        for position, index in enumerate(remaining):
            if len(batch) == batch_size:
                rest.extend(remaining[position:])
                break
            if passage_ids[index] in seen:
                rest.append(index)
            else:
                batch.append(index)
                seen.add(passage_ids[index])
        if len(batch) < batch_size:
            break  # not enough distinct passages left to fill another batch
        batches.append(batch)
        remaining = rest
    return batches[start_batch:]


class PairCollator:
    """Tokenises a list of pair indices into question and passage tensors (different max lengths)."""

    def __init__(self, tokenizer: Any, pairs: dict[str, list[str]], query_max_length: int, passage_max_length: int):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        def tok(texts: list[str], max_length: int):
            return self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

        q = tok([self.pairs["question"][i] for i in indices], self.query_max_length)
        p = tok([self.pairs["passage"][i] for i in indices], self.passage_max_length)
        return {
            "q_input_ids": q["input_ids"], "q_attention_mask": q["attention_mask"],
            "p_input_ids": p["input_ids"], "p_attention_mask": p["attention_mask"],
        }


# --------------------------------------------------------------------------- #
# Validation: retrieval over the validation passages
# --------------------------------------------------------------------------- #

def evaluate_retrieval(
    encode_queries: Callable[[list[str]], np.ndarray],
    encode_passages: Callable[[list[str]], np.ndarray],
    pairs: dict[str, list[str]],
) -> dict[str, float]:
    """Rank all distinct validation passages for every question (cosine); MRR@10 and recall@k."""
    unique: dict[str, int] = {}
    gold = []
    for passage_id in pairs["passage_id"]:
        gold.append(unique.setdefault(passage_id, len(unique)))
    first_text: dict[str, str] = {}
    for passage_id, passage in zip(pairs["passage_id"], pairs["passage"]):
        first_text.setdefault(passage_id, passage)
    passages = list(first_text.values())  # insertion order == index order of `unique`

    def unit(x: np.ndarray) -> np.ndarray:
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

    scores = unit(encode_queries(pairs["question"])) @ unit(encode_passages(passages)).T
    gold_arr = np.asarray(gold)
    gold_scores = scores[np.arange(len(gold_arr)), gold_arr]
    # rank 1 = the gold passage is the best-scored; ties count against it (a collapsed space must not look good)
    rank = (scores >= gold_scores[:, None]).sum(axis=1)
    return {
        "mrr@10": float(np.where(rank <= 10, 1.0 / rank, 0.0).mean()),
        "recall@1": float((rank <= 1).mean()),
        "recall@10": float((rank <= 10).mean()),
        "num_questions": float(len(gold_arr)),
        "num_passages": float(len(passages)),
    }
