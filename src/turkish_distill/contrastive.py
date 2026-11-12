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
    q: torch.Tensor, p: torch.Tensor, scale: float, symmetric: bool = False, neg_mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy over cosine similarities; the i-th passage is the positive of the i-th question.

    `p` may hold more rows than `q`: the first len(q) are the positives, the rest are extra (hard) negatives that
    only act as negatives for every question. `neg_mask` (len(q) x extra, True = ignore) removes the entries where
    an extra passage is, in fact, the positive of that question. The symmetric direction (passage -> question)
    covers the positives only.
    """
    q = F.normalize(q.float(), dim=-1)
    p = F.normalize(p.float(), dim=-1)
    scores = q @ p.T * scale
    n = q.size(0)
    if neg_mask is not None:
        scores = torch.cat([scores[:, :n], scores[:, n:].masked_fill(neg_mask, float("-inf"))], dim=1)
    labels = torch.arange(n, device=scores.device)
    loss = F.cross_entropy(scores, labels)
    if symmetric:
        loss = (loss + F.cross_entropy(scores[:, :n].T, labels)) / 2
    accuracy = (scores.argmax(dim=1) == labels).float().mean()
    return loss, {"acc": accuracy.detach()}


def make_loss(cfg: ContrastiveLossConfig) -> LossFn:
    symmetric = cfg.name == "mnrl_symmetric"
    if cfg.name not in ("mnrl", "mnrl_symmetric"):
        raise ValueError(f"unknown contrastive loss {cfg.name!r}")
    return lambda q, p, neg_mask=None: info_nce(q, p, cfg.scale, symmetric, neg_mask)


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


def answer_weight(text: str, cfg: ContrastiveLossConfig) -> float:
    """Weight of one pair's answer term (see `ContrastiveLossConfig.answer_weight_mode`)."""
    if cfg.answer_weight_mode == "fixed":
        return float(cfg.answer_weight)
    if cfg.answer_weight_mode == "words":
        fraction = len(text.split()) / cfg.answer_ref_words
    elif cfg.answer_weight_mode == "chars":
        fraction = len(text) / cfg.answer_ref_chars
    else:
        raise ValueError(f"no answer weight for mode {cfg.answer_weight_mode!r}")
    fraction = min(1.0, fraction)
    if cfg.answer_weight_inverse:
        fraction = 1.0 - fraction
    return float(cfg.answer_weight_min + (cfg.answer_weight_max - cfg.answer_weight_min) * fraction)


def make_answer_weight_fn(cfg: ContrastiveLossConfig) -> Optional[Callable[[str], float]]:
    """text -> weight for the collator, or None when the answer term is off."""
    if cfg.answer_weight_mode == "none":
        return None
    return lambda text: answer_weight(text, cfg)


def answer_info_nce(
    q: torch.Tensor, a: torch.Tensor, weights: torch.Tensor, scale: float, mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Question -> answer InfoNCE over the batch's answers, mean of the per-pair weighted cross-entropies.

    Row i of `q` is the question of answer i; `mask` (k x k, True = ignore) removes the cells where another pair's
    answer is the same text (it would be a false negative). The diagonal is never masked.
    """
    q = F.normalize(q.float(), dim=-1)
    a = F.normalize(a.float(), dim=-1)
    scores = q @ a.T * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    labels = torch.arange(q.size(0), device=scores.device)
    ce = F.cross_entropy(scores, labels, reduction="none")
    w = weights.to(ce.dtype)
    loss = (w * ce).mean()
    accuracy = (scores.argmax(dim=1) == labels).float().mean()
    return loss, {
        "ans_loss": loss.detach(), "ans_ce": ce.mean().detach(), "ans_weight": w.mean().detach(),
        "ans_acc": accuracy.detach(), "ans_pairs": torch.tensor(float(q.size(0))),
    }


def make_answer_loss(cfg: ContrastiveLossConfig) -> Optional[Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]]]:
    """The answer term for `contrastive_step`, or None when it is off."""
    if cfg.answer_weight_mode == "none":
        return None
    return lambda q, a, weights, mask=None: answer_info_nce(q, a, weights, cfg.scale, mask)


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
    answer_fn=None,
) -> dict[str, float]:
    """One forward/backward pass; gradients are left on the parameters (the caller runs the optimiser).

    With `anchor_model` and `anchor_fn` (see `make_anchor`, `make_anchor_loss`) the loss also contains the anchor
    term; the anchor embeddings are computed without gradients, so the cached and plain steps still agree.
    With `answer_fn` (see `make_answer_loss`) and answers in the batch (`a_input_ids`, `a_rows`, `answer_weight`,
    `answer_mask` from `PairCollator`), the weighted question -> answer term is added; the answers are a third side,
    encoded like the questions and passages (and not anchored).

    With `chunk_size`, the embeddings of the whole batch are computed without gradients first, the loss and
    its gradient w.r.t. the embeddings are computed on the full batch, and then each mini-batch is encoded
    again with gradients and back-propagated with that stored embedding gradient. The result equals the plain
    step (up to numerical noise); the RNG state is restored so dropout masks are identical in both passes.
    """
    sides = [
        ("q", batch["q_input_ids"], batch["q_attention_mask"]),
        ("p", batch["p_input_ids"], batch["p_attention_mask"]),
    ]
    answered = answer_fn is not None and "a_input_ids" in batch
    if answered:
        sides.append(("a", batch["a_input_ids"], batch["a_attention_mask"]))
    size = max(ids.size(0) for _, ids, _ in sides)  # p holds the positives and, if mined, the hard negatives
    anchored = anchor_model is not None and anchor_fn is not None
    neg_mask = batch.get("neg_mask")

    def total_loss(emb: dict[str, torch.Tensor], anchors: dict[str, torch.Tensor]):
        q, p = emb["q"], emb["p"]
        nce, logs = loss_fn(q, p) if neg_mask is None else loss_fn(q, p, neg_mask)
        loss = nce
        if anchored:
            term, extra = anchor_fn(q, p, anchors["q"], anchors["p"])
            loss, logs = loss + term, {**logs, **extra}
        if answered:
            term, extra = answer_fn(q[batch["a_rows"]], emb["a"], batch["answer_weight"], batch.get("answer_mask"))
            loss, logs = loss + term, {**logs, **extra}
        if anchored or answered:
            logs["nce_loss"] = nce.detach()
        return loss, logs

    if chunk_size is None or chunk_size >= size:
        anchors = {}
        if anchored:
            with torch.no_grad():
                anchors = {side: _encode(anchor_model, ids, mask, use_amp) for side, ids, mask in sides[:2]}
        emb = {side: _encode(model, ids, mask, use_amp) for side, ids, mask in sides}
        loss, logs = total_loss(emb, anchors)
        loss.backward()
        return {"loss": float(loss.detach()), **{k: float(v) for k, v in logs.items()}}

    device = batch["q_input_ids"].device
    slices_of = {side: [slice(i, i + chunk_size) for i in range(0, ids.size(0), chunk_size)] for side, ids, _ in sides}
    embeddings: dict[str, torch.Tensor] = {}
    states: dict[str, list] = {}
    with torch.no_grad():
        for side, ids, mask in sides:
            states[side], parts = [], []
            for sl in slices_of[side]:
                states[side].append(_capture_rng(device))
                parts.append(_encode(model, ids[sl], mask[sl], use_amp))
            embeddings[side] = torch.cat(parts).requires_grad_(True)
        anchors = {}
        if anchored:
            for side, ids, mask in sides[:2]:
                anchors[side] = torch.cat(
                    [_encode(anchor_model, ids[sl], mask[sl], use_amp) for sl in slices_of[side]]
                )
    loss, logs = total_loss(embeddings, anchors)
    loss.backward()  # fills .grad of the embedding tensors only

    for side, ids, mask in sides:
        grad = embeddings[side].grad
        for sl, state in zip(slices_of[side], states[side]):
            with _restore_rng(state, device):
                chunk = _encode(model, ids[sl], mask[sl], use_amp)
            (chunk * grad[sl]).sum().backward()
    return {"loss": float(loss.detach()), **{k: float(v) for k, v in logs.items()}}


# --------------------------------------------------------------------------- #
# Data: pairs, batching without duplicate passages, collation
# --------------------------------------------------------------------------- #

def load_pairs(path: str | Path, max_pairs: Optional[int] = None, seed: int = 42) -> dict[str, Any]:
    """Read a pairs JSONL written by `prepare_pairs` (or `mine_negatives`) into parallel lists.

    If the rows carry `answer` (see attach_answers), the result has `answer` (None for rows without one).
    If the rows carry `neg_passage_ids`, the result also has `neg_ids` (one list per pair) and `pool`, a
    passage-id -> text map over all rows (also those dropped by `max_pairs`), which the negatives refer to.
    """
    questions, passages, ids, negs, answers = [], [], [], [], []
    pool: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                questions.append(row["question"])
                passages.append(row["passage"])
                ids.append(row["passage_id"])
                pool[row["passage_id"]] = row["passage"]
                negs.append(row.get("neg_passage_ids"))
                answers.append(row.get("answer") or None)
    mined = any(n is not None for n in negs)
    has_answers = any(a is not None for a in answers)
    if max_pairs is not None and len(questions) > max_pairs:
        keep = np.sort(np.random.default_rng(seed).choice(len(questions), size=max_pairs, replace=False))
        questions, passages, ids, negs, answers = (
            [x[i] for i in keep] for x in (questions, passages, ids, negs, answers)
        )
    out: dict[str, Any] = {"question": questions, "passage": passages, "passage_id": ids}
    if has_answers:
        out["answer"] = answers  # None where the pair has no answer
    if mined:
        out["neg_ids"] = [n or [] for n in negs]
        out["pool"] = pool
    return out


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
    """Tokenises a list of pair indices into question and passage tensors (different max lengths).

    With `hard_negatives` > 0 the passage tensors hold the batch's positives first and then `hard_negatives`
    mined negatives per pair (pair-major); `neg_mask` marks the (question, negative) cells where the negative is
    that question's own positive passage. A pair without a mined negative gets a deterministic random pool
    passage instead (an easy negative, never its own positive).

    With `answer_weight_fn` (text -> weight, see `make_answer_weight_fn`) the batch also carries the answers of the
    pairs that have one: `a_input_ids` / `a_attention_mask`, `a_rows` (their positions in the batch), `answer_weight`
    and `answer_mask` (True where two pairs have the same answer text, off the diagonal). A batch without any answer
    has none of these keys.
    """

    def __init__(
        self, tokenizer: Any, pairs: dict[str, Any], query_max_length: int, passage_max_length: int,
        hard_negatives: int = 0, answer_weight_fn: Optional[Callable[[str], float]] = None,
        answer_max_length: int = 128,
    ):
        if hard_negatives > 0 and "neg_ids" not in pairs:
            raise ValueError("data.hard_negatives > 0 needs a pairs file with `neg_passage_ids` (see mine_negatives)")
        if answer_weight_fn is not None and "answer" not in pairs:
            raise ValueError("the answer term needs a pairs file with an `answer` field (see attach_answers)")
        self.answer_weight_fn = answer_weight_fn
        self.answer_max_length = answer_max_length
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.hard_negatives = hard_negatives
        self._pool_ids = sorted(pairs["pool"]) if hard_negatives > 0 else []

    def _negative_ids(self, index: int) -> list[str]:
        own = self.pairs["passage_id"][index]
        mined = [n for n in self.pairs["neg_ids"][index] if n != own][: self.hard_negatives]
        rng = np.random.default_rng(index)
        while len(mined) < self.hard_negatives:  # fill with easy negatives so every batch has the same shape
            candidate = self._pool_ids[int(rng.integers(len(self._pool_ids)))]
            if candidate != own and candidate not in mined:
                mined.append(candidate)
        return mined

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        def tok(texts: list[str], max_length: int):
            return self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

        q = tok([self.pairs["question"][i] for i in indices], self.query_max_length)
        passages = [self.pairs["passage"][i] for i in indices]
        extra: dict[str, torch.Tensor] = {}
        if self.hard_negatives > 0:
            neg_ids = [n for i in indices for n in self._negative_ids(i)]
            passages = passages + [self.pairs["pool"][n] for n in neg_ids]
            own = [self.pairs["passage_id"][i] for i in indices]
            extra["neg_mask"] = torch.tensor([[n == o for n in neg_ids] for o in own], dtype=torch.bool)
        p = tok(passages, self.passage_max_length)
        if self.answer_weight_fn is not None:
            rows = [k for k, i in enumerate(indices) if self.pairs["answer"][i]]
            if rows:
                texts = [self.pairs["answer"][indices[k]] for k in rows]
                a = tok(texts, self.answer_max_length)
                keys = [" ".join(t.lower().split()) for t in texts]
                same = torch.tensor([[x == y for y in keys] for x in keys], dtype=torch.bool)
                extra.update({
                    "a_input_ids": a["input_ids"], "a_attention_mask": a["attention_mask"],
                    "a_rows": torch.tensor(rows, dtype=torch.long),
                    "answer_weight": torch.tensor([self.answer_weight_fn(t) for t in texts], dtype=torch.float32),
                    "answer_mask": same & ~torch.eye(len(rows), dtype=torch.bool),
                })
        return {
            "q_input_ids": q["input_ids"], "q_attention_mask": q["attention_mask"],
            "p_input_ids": p["input_ids"], "p_attention_mask": p["attention_mask"], **extra,
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
