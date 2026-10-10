"""Metric logging: JSONL (always, source of truth) + optional TensorBoard."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional


class MetricsLogger:
    def __init__(self, run_dir: str | Path, tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception as exc:  # tensorboard not installed, etc.
                warnings.warn(f"TensorBoard disabled: {exc}")

    def log(self, step: int, metrics: dict[str, float], prefix: str = "") -> None:
        clean = {}
        for key, value in metrics.items():
            try:
                clean[f"{prefix}{key}"] = float(value)
            except (TypeError, ValueError):
                continue
        record = {"step": int(step), "time": time.time(), **clean}
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        if self._tb is not None:
            for key, value in clean.items():
                self._tb.add_scalar(key, value, step)

    def flush(self) -> None:
        self._jsonl.flush()
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        self.flush()
        self._jsonl.close()
        if self._tb is not None:
            self._tb.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[dict]:
    """Read a metrics.jsonl file (tolerates a truncated last line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def last_logged_step(path: str | Path) -> Optional[int]:
    if not Path(path).exists():
        return None
    records = read_metrics(path)
    return records[-1]["step"] if records else None
