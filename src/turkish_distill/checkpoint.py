"""Atomic, resumable checkpoints.

A checkpoint is a single file. It is first written to a local scratch
directory (fast disk), then copied next to its destination under a `.tmp`
name and renamed. A crash or Colab disconnect can therefore never leave a
half-written `.pt` file in the destination (e.g. on Google Drive).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch

_PATTERN = re.compile(r"^step_(\d{8})\.pt$")


def checkpoint_name(step: int) -> str:
    return f"step_{step:08d}.pt"


def list_checkpoints(ckpt_dir: str | Path) -> list[tuple[int, Path]]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []
    found = []
    for p in ckpt_dir.iterdir():
        m = _PATTERN.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_checkpoint(ckpt_dir: str | Path) -> Optional[Path]:
    found = list_checkpoints(ckpt_dir)
    return found[-1][1] if found else None


def cleanup_stale(ckpt_dir: str | Path) -> None:
    """Remove leftovers of interrupted saves."""
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("*.tmp"):
            p.unlink(missing_ok=True)


def save_checkpoint(
    state: dict[str, Any],
    ckpt_dir: str | Path,
    step: int,
    keep_last: int = 2,
    local_tmp_dir: Optional[str | Path] = None,
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / checkpoint_name(step)
    staging_root = Path(local_tmp_dir) if local_tmp_dir else None
    if staging_root is not None:
        staging_root.mkdir(parents=True, exist_ok=True)
    fd, local_path = tempfile.mkstemp(suffix=".pt", dir=str(staging_root) if staging_root else str(ckpt_dir))
    os.close(fd)
    try:
        torch.save(state, local_path)
        if staging_root is None:
            # Already in the destination directory: just rename atomically.
            os.replace(local_path, final)
        else:
            tmp_dest = final.with_suffix(".pt.tmp")
            shutil.copyfile(local_path, tmp_dest)
            os.replace(tmp_dest, final)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    _rotate(ckpt_dir, keep_last)
    return final


def _rotate(ckpt_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    found = list_checkpoints(ckpt_dir)
    for _, path in found[:-keep_last]:
        path.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
