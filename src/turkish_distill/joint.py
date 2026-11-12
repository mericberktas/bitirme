"""Contrastive loss on (question, passage) pairs, trained together with the distillation (`joint` config section).

Each step on which the joint term applies (`JointContrastive.applies`), `JointContrastive.step` runs one
`contrastive.contrastive_step` on a pair batch with the loss multiplied by `joint.weight`. It only accumulates gradients;
the caller has already run the distillation step and runs the optimiser afterwards, so the two losses form one objective.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import JointContrastiveConfig
from .contrastive import (
    PairCollator,
    contrastive_step,
    make_answer_loss,
    make_answer_weight_fn,
    make_loss,
    unique_passage_batches,
)
from .contrastive_config import ContrastiveLossConfig


def joint_loss_config(cfg: JointContrastiveConfig) -> ContrastiveLossConfig:
    return ContrastiveLossConfig(
        name=cfg.loss, scale=cfg.scale, answer_weight_mode=cfg.answer_weight_mode, answer_weight=cfg.answer_weight,
        answer_weight_min=cfg.answer_weight_min, answer_weight_max=cfg.answer_weight_max,
        answer_ref_words=cfg.answer_ref_words, answer_ref_chars=cfg.answer_ref_chars,
        answer_weight_inverse=cfg.answer_weight_inverse,
    )


class JointContrastive:
    def __init__(
        self, cfg: JointContrastiveConfig, tokenizer: Any, pairs: dict[str, Any], seed: int,
        chunk_size: Optional[int] = None, use_amp: bool = False,
    ):
        self.cfg = cfg
        self.seed = seed
        self.chunk_size = chunk_size
        self.use_amp = use_amp
        self.passage_ids = pairs["passage_id"]
        loss_cfg = joint_loss_config(cfg)
        self.collator = PairCollator(
            tokenizer, pairs, cfg.query_max_length, cfg.passage_max_length, cfg.hard_negatives,
            answer_weight_fn=make_answer_weight_fn(loss_cfg), answer_max_length=cfg.answer_max_length,
        )
        weight = float(cfg.weight)
        nce = make_loss(loss_cfg)

        def loss_fn(q, p, neg_mask=None):
            loss, logs = nce(q, p) if neg_mask is None else nce(q, p, neg_mask)
            return weight * loss, {**logs, "nce": loss.detach()}

        self.loss_fn = loss_fn
        answer = make_answer_loss(loss_cfg)
        self.answer_fn = None
        if answer is not None:
            def answer_fn(q, a, weights, mask=None):
                loss, logs = answer(q, a, weights, mask)
                return weight * loss, logs

            self.answer_fn = answer_fn
        self._epoch: Optional[int] = None
        self._batches: list[list[int]] = []
        first = unique_passage_batches(self.passage_ids, cfg.batch_size, seed, 0)
        if not first:
            raise ValueError(
                f"joint: cannot form a batch of {cfg.batch_size} pairs with distinct passages from "
                f"{len(self.passage_ids)} pairs"
            )
        self.steps_per_epoch = len(first)
        self._epoch, self._batches = 0, first

    def applies(self, step: int) -> bool:
        """Whether training step number `step` (0-based) gets a pair batch."""
        return step >= self.cfg.start_step and (step - self.cfg.start_step) % self.cfg.every == 0

    def pair_steps(self, total_steps: int) -> int:
        """How many pair batches a run of `total_steps` steps uses."""
        if total_steps <= self.cfg.start_step:
            return 0
        return (total_steps - self.cfg.start_step - 1) // self.cfg.every + 1

    def batch_indices(self, step: int) -> list[int]:
        """The pair indices of training step `step` (deterministic in the step, so resuming sees the same batches)."""
        k = (step - self.cfg.start_step) // self.cfg.every
        epoch, offset = divmod(k, self.steps_per_epoch)
        if epoch != self._epoch:
            # a later epoch can yield a batch fewer than epoch 0 (cycled, as in train_contrastive)
            self._batches = unique_passage_batches(self.passage_ids, self.cfg.batch_size, self.seed, epoch) or self._batches
            self._epoch = epoch
        return self._batches[offset % len(self._batches)]

    def step(self, model, step: int, device: torch.device) -> dict[str, float]:
        """Accumulate the weighted contrastive gradient of step `step` on the model's parameters; returns its logs."""
        batch = {k: v.to(device) for k, v in self.collator(self.batch_indices(step)).items()}
        return contrastive_step(model, batch, self.loss_fn, self.chunk_size, self.use_amp, answer_fn=self.answer_fn)
