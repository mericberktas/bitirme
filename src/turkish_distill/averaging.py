"""Weight averaging during training: EMA (Mean Teacher / model EMA) and SWA (stochastic weight averaging).

Both are borrowed from vision. Averaging the weights of models trained with different losses (soups) gained 0.25-0.34
TR-MTEB points here; these are the cheap in-run versions. The averages never feed back into training: the optimiser
keeps working on the model's own weights, and the averages are saved as extra models at the end (`final_ema/`,
`final_swa/`) and kept in the checkpoints, so a resumed run continues them.

Only floating-point entries of the state dict are averaged (integer buffers are copied from the model when an average
is applied). The averages live on the model's device in float32.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn


class WeightAverager:
    """`mode` "ema": avg = d * avg + (1 - d) * w after every update, d = min(decay, (1 + n) / (10 + n)).
    `mode` "uniform": avg = mean of the weights at every update (SWA).

    `update(model, step)` is called after optimiser step number `step` (1-based: `step` steps are done); it only
    records the weights when `step >= start_step` and `(step - start_step) % every == 0`.
    """

    def __init__(self, mode: str, decay: float = 0.999, start_step: int = 0, every: int = 1):
        if mode not in ("ema", "uniform"):
            raise ValueError(f"mode must be 'ema' or 'uniform', got {mode!r}")
        self.mode = mode
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.every = max(1, int(every))
        self.avg: Optional[dict[str, torch.Tensor]] = None
        self.count = 0

    @staticmethod
    def _float_entries(model: nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def due(self, step: int) -> bool:
        return step >= self.start_step and (step - self.start_step) % self.every == 0

    @torch.no_grad()
    def update(self, model: nn.Module, step: int, force: bool = False) -> bool:
        """`force`: record even off the schedule (used for the last step, once the averaging has started)."""
        if not self.due(step) and not (force and step >= self.start_step):
            return False
        weights = self._float_entries(model)
        if self.avg is None:
            self.avg = {k: v.float().clone() for k, v in weights.items()}
            self.count = 1
            return True
        self.count += 1
        keys = list(self.avg)
        avg = [self.avg[k] for k in keys]
        new = [weights[k].float() for k in keys]
        if self.mode == "ema":
            d = min(self.decay, (1.0 + self.count) / (10.0 + self.count))
            torch._foreach_mul_(avg, d)
            torch._foreach_add_(avg, new, alpha=1.0 - d)
        else:
            delta = torch._foreach_sub(new, avg)
            torch._foreach_add_(avg, delta, alpha=1.0 / self.count)
        return True

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "count": self.count,
            "avg": None if self.avg is None else {k: v.cpu() for k, v in self.avg.items()},
        }

    def load_state_dict(self, state: dict, device: torch.device | str = "cpu") -> None:
        if state.get("mode") != self.mode:
            raise ValueError(f"averager state is {state.get('mode')!r}, expected {self.mode!r}")
        self.count = int(state["count"])
        self.avg = None if state["avg"] is None else {k: v.to(device) for k, v in state["avg"].items()}

    @contextmanager
    def applied(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily load the average into `model` (its own weights are restored afterwards)."""
        if self.avg is None:
            raise ValueError("no weights averaged yet")
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            with torch.no_grad():
                current = model.state_dict()
                for k, v in self.avg.items():
                    current[k].copy_(v.to(current[k].dtype))
            yield model
        finally:
            with torch.no_grad():
                current = model.state_dict()
                for k, v in backup.items():
                    current[k].copy_(v)


def make_averagers(train_cfg) -> dict[str, WeightAverager]:
    """The averagers a training config asks for, by output suffix ("ema", "swa")."""
    out: dict[str, WeightAverager] = {}
    if train_cfg.ema_decay > 0:
        out["ema"] = WeightAverager("ema", decay=train_cfg.ema_decay)
    if train_cfg.swa_start > 0:
        start = max(1, int(round(train_cfg.swa_start * train_cfg.max_steps)))
        out["swa"] = WeightAverager("uniform", start_step=start, every=train_cfg.swa_every)
    return out
