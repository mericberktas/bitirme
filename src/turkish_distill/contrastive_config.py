"""Configuration of the contrastive baseline (YAML <-> dataclasses).

Separate from `config.py` because a contrastive run has no teachers; the student section is shared.
Unknown keys raise an error, as in the distillation configs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import StudentConfig, _set_dotted, parse_override_value

LOSS_NAMES = ("mnrl", "mnrl_symmetric")


@dataclass
class ContrastiveDataConfig:
    pairs_train: str = ""
    pairs_val: str = ""
    max_train_pairs: Optional[int] = None  # seeded subset, for quick debugging runs
    query_max_length: int = 64
    passage_max_length: int = 512


@dataclass
class ContrastiveLossConfig:
    # mnrl: question -> passage cross-entropy over the in-batch passages (InfoNCE).
    # mnrl_symmetric: the average of that and passage -> question.
    name: str = "mnrl"
    scale: float = 20.0  # 1 / temperature; the lab's 0.02 corresponds to 50
    # Anchor (distillation regulariser): keep the pooled embeddings of the batch texts close to those of a frozen copy
    # of the model the run starts from (cosine to the anchor, and MSE between the two cosine-similarity matrices over
    # the questions and passages together; the same two terms as the distillation loss, so 10 / 200 are the usual
    # weights). 0 = off. Meant for fine-tuning an already distilled student without forgetting what distillation gave.
    anchor_cosine_weight: float = 0.0
    anchor_gram_weight: float = 0.0


@dataclass
class ContrastiveTrainConfig:
    batch_size: int = 128
    # None: plain training. Otherwise gradient caching (GradCache): the batch is encoded in mini-batches of
    # this size, so a large effective batch needs little memory. Same maths as the plain step.
    chunk_size: Optional[int] = None
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: Optional[int] = 3  # used when max_steps is None
    max_steps: Optional[int] = None  # takes precedence over epochs
    grad_clip: float = 1.0
    bf16: bool = True
    seed: int = 42
    num_workers: int = 2
    log_every: int = 20
    eval_every: int = 100
    ckpt_every: int = 200
    keep_last: int = 2


@dataclass
class ContrastiveConfig:
    name: str
    output_dir: str
    student: StudentConfig = field(default_factory=StudentConfig)
    data: ContrastiveDataConfig = field(default_factory=ContrastiveDataConfig)
    loss: ContrastiveLossConfig = field(default_factory=ContrastiveLossConfig)
    train: ContrastiveTrainConfig = field(default_factory=ContrastiveTrainConfig)
    tensorboard: bool = True
    local_tmp_dir: Optional[str] = None  # checkpoints are written here first, then moved to output_dir

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


_SECTIONS = {
    "student": StudentConfig,
    "data": ContrastiveDataConfig,
    "loss": ContrastiveLossConfig,
    "train": ContrastiveTrainConfig,
}


def _build(cls, data: dict[str, Any], path: str = ""):
    if not isinstance(data, dict):
        raise TypeError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{path or 'config'}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        section = _SECTIONS.get(key) if cls is ContrastiveConfig else None
        kwargs[key] = _build(section, value or {}, f"{path}.{key}" if path else key) if section else value
    return cls(**kwargs)


def validate(cfg: ContrastiveConfig) -> None:
    t = cfg.train
    if cfg.loss.name not in LOSS_NAMES:
        raise ValueError(f"loss.name must be one of {LOSS_NAMES}, got {cfg.loss.name!r}")
    if cfg.loss.scale <= 0:
        raise ValueError("loss.scale must be positive")
    if cfg.loss.anchor_cosine_weight < 0 or cfg.loss.anchor_gram_weight < 0:
        raise ValueError("loss.anchor_cosine_weight and loss.anchor_gram_weight must be >= 0")
    if t.batch_size < 2:
        raise ValueError("train.batch_size must be >= 2 (in-batch negatives need other pairs)")
    if t.chunk_size is not None and t.chunk_size < 1:
        raise ValueError("train.chunk_size must be >= 1 or null")
    if t.max_steps is None and t.epochs is None:
        raise ValueError("set train.epochs or train.max_steps")
    if t.max_steps is not None and t.max_steps < 1:
        raise ValueError("train.max_steps must be >= 1")
    if not 0 <= t.warmup_ratio < 1:
        raise ValueError("train.warmup_ratio must be in [0, 1)")


def load_contrastive_config(path: str | Path, overrides: Optional[list[str]] = None) -> ContrastiveConfig:
    """Load a YAML config. `overrides` are 'a.b.c=value' strings (value parsed as YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(raw, key.strip(), parse_override_value(value))
    cfg = _build(ContrastiveConfig, raw)
    validate(cfg)
    return cfg
