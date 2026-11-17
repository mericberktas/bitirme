"""A bank of cached teacher vectors for the relational and contrastive bank terms (`loss.bank_*`).

A batch of 128 random texts holds almost none of a text's real neighbours (measured: 1.2% of its top-10 neighbours),
so in-batch relation terms only ever compare it with far-away texts. The teacher vectors of the whole corpus are
already cached, so a large random subset of them (tens of thousands) can serve as reference points at almost no cost:
no student encoding, only a (batch x bank) matrix product per teacher.

The bank is drawn from the TRAINING texts only, without replacement, from a generator seeded by (seed, round), and is
redrawn every `refresh_every` steps (round = step // refresh_every; 0 = one bank for the whole run). The draw depends
only on the step, so a resumed run sees the same banks. The vectors come from the same caches the collator reads
(after the target methods, e.g. centring), so the bank and the targets are in the same space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class TeacherBank:
    def __init__(
        self,
        caches: dict[str, Any],
        train_idx: np.ndarray,
        size: int,
        seed: int,
        refresh_every: int = 0,
        device: torch.device | str = "cpu",
        names: list[str] | None = None,
    ):
        if size < 1:
            raise ValueError("bank size must be >= 1")
        self.caches = {n: caches[n] for n in (names if names is not None else list(caches))}
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.size = min(int(size), len(self.train_idx))
        self.seed = int(seed)
        self.refresh_every = int(refresh_every)
        self.device = torch.device(device)
        self._round: int | None = None
        self.indices: torch.Tensor | None = None
        self.vectors: dict[str, torch.Tensor] = {}

    def round_of(self, step: int) -> int:
        return step // self.refresh_every if self.refresh_every > 0 else 0

    def at_step(self, step: int) -> bool:
        """Make sure the bank of training step `step` is loaded; returns True when it was (re)drawn."""
        r = self.round_of(step)
        if r == self._round:
            return False
        rng = np.random.default_rng([self.seed, 7919, r])
        idx = np.sort(rng.choice(self.train_idx, size=self.size, replace=False))
        self.indices = torch.from_numpy(idx).to(self.device)
        self.vectors = {
            name: F.normalize(torch.from_numpy(np.asarray(cache.get(idx), dtype=np.float32)), dim=-1).to(self.device)
            for name, cache in self.caches.items()
        }
        self._round = r
        return True
