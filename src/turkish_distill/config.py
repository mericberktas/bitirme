"""Experiment configuration: YAML <-> typed dataclasses.

Unknown keys raise an error so that typos in a YAML file never go unnoticed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TeacherConfig:
    name: str
    hf_id: str
    dim: int
    # Prefix prepended to every corpus text when the teacher encodes it.
    # The student always sees the raw text.
    doc_prefix: str = ""
    max_length: int = 512
    batch_size: int = 32
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    # Hugging Face commit hash of the model. Pinning it makes the teacher
    # vectors reproducible; it is stored in the cache's meta.json and checked
    # when the cache is opened.
    revision: Optional[str] = None
    # Relative weight of this teacher in the total loss (only the ratio between teachers matters).
    weight: float = 1.0
    # Optional linear ramp: the weight is `weight_start` until step `ramp_start_step`, rises linearly to `weight`
    # at step `ramp_end_step` and stays there. `weight_start: null` keeps `weight` constant. Example: a teacher the
    # student imitates poorly can be phased in after the others (weight_start 0, ramp 0 -> 900).
    weight_start: Optional[float] = None
    ramp_start_step: int = 0
    ramp_end_step: int = 0


def teacher_weight(teacher: TeacherConfig, step: int) -> float:
    """The loss weight of a teacher at a training step (constant unless `weight_start` is set)."""
    if teacher.weight_start is None or step >= teacher.ramp_end_step:
        return float(teacher.weight)
    if step <= teacher.ramp_start_step:
        return float(teacher.weight_start)
    fraction = (step - teacher.ramp_start_step) / (teacher.ramp_end_step - teacher.ramp_start_step)
    return float(teacher.weight_start + fraction * (teacher.weight - teacher.weight_start))


@dataclass
class StudentConfig:
    hf_id: str = "ytu-ce-cosmos/modernbert-tr-base-1k"
    max_length: int = 512
    pooling: str = "mean"
    revision: Optional[str] = None  # HF commit hash of the student backbone/tokenizer


@dataclass
class DataConfig:
    corpus_path: str = ""
    text_column: str = "text"
    cache_dir: str = ""
    shard_size: int = 10_000
    max_samples: Optional[int] = None
    min_chars: int = 20
    val_size: int = 2000


@dataclass
class LossConfig:
    cosine_weight: float = 10.0
    sim_weight: float = 200.0
    resim_weight: float = 20.0
    resim_margin: float = 0.015
    resim_num_samples: int = 4096
    gor_weight: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 128
    # None: the whole batch goes through the encoder at once. Otherwise gradient caching: the batch is encoded in
    # mini-batches of this size (same maths, memory of one mini-batch). batch_size 128 with 512-token texts ran
    # out of memory on an L4 (22 GB) without it.
    chunk_size: Optional[int] = None
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_steps: int = 10_000
    grad_clip: float = 1.0
    bf16: bool = True
    seed: int = 42
    num_workers: int = 2
    log_every: int = 50
    diag_every: int = 500
    ckpt_every: int = 1000
    keep_last: int = 2


@dataclass
class EvalConfig:
    # Optional fast STS evaluation during training (HF dataset id).
    sts_dataset: Optional[str] = None
    sts_split: str = "validation"
    sts_columns: Optional[list[str]] = None  # [sentence1, sentence2, score]
    # Optional validation retrieval (the same metric the contrastive runs log, so the methods are comparable):
    # a pairs JSONL written by prepare_pairs. Passages are encoded with student.max_length.
    pairs_val: Optional[str] = None
    query_max_length: int = 64


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    teachers: list[TeacherConfig]
    student: StudentConfig = field(default_factory=StudentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tensorboard: bool = True
    # Optional local scratch directory; checkpoints are written here first
    # and then moved to output_dir (e.g. Colab /content -> Drive).
    local_tmp_dir: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _build(cls, data: dict[str, Any], path: str = ""):
    if not isinstance(data, dict):
        raise TypeError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{path or 'config'}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        sub = f"{path}.{key}" if path else key
        if key == "teachers":
            if not isinstance(value, list) or not value:
                raise ValueError("teachers: expected a non-empty list")
            kwargs[key] = [_build(TeacherConfig, t, f"teachers[{i}]") for i, t in enumerate(value)]
        elif key in _SECTIONS and cls is ExperimentConfig:
            kwargs[key] = _build(_SECTIONS[key], value or {}, sub)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_SECTIONS = {
    "student": StudentConfig,
    "data": DataConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


_SCIENTIFIC_WITHOUT_DOT = re.compile(r"^[+-]?\d+[eE][+-]?\d+$")


def parse_override_value(text: str) -> Any:
    """YAML value of a command-line override.

    PyYAML (YAML 1.1) reads `3e-05` as a string because the mantissa has no dot, while `3.0e-05` is a float;
    notebooks and shells naturally produce the first form, so it is converted here.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str) and _SCIENTIFIC_WITHOUT_DOT.match(value.strip()):
        return float(value)
    return value


def _set_dotted(d: Any, dotted: str, value: Any) -> None:
    """Set `a.b.c` in nested dicts; a numeric part indexes a list (`teachers.0.weight`)."""
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d[int(p)] if isinstance(d, list) else d.setdefault(p, {})
    if isinstance(d, list):
        d[int(parts[-1])] = value
    else:
        d[parts[-1]] = value


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> ExperimentConfig:
    """Load a YAML config. `overrides` are 'a.b.c=value' strings (value parsed as YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(raw, key.strip(), parse_override_value(value))
    cfg = _build(ExperimentConfig, raw)
    validate(cfg)
    return cfg


def validate(cfg: ExperimentConfig) -> None:
    names = [t.name for t in cfg.teachers]
    if len(set(names)) != len(names):
        raise ValueError(f"teacher names must be unique, got {names}")
    for t in cfg.teachers:
        if t.dim <= 0:
            raise ValueError(f"teacher {t.name}: dim must be positive")
        if t.weight < 0:
            raise ValueError(f"teacher {t.name}: weight must be >= 0")
        if t.weight_start is not None:
            if t.weight_start < 0:
                raise ValueError(f"teacher {t.name}: weight_start must be >= 0")
            if t.ramp_end_step < t.ramp_start_step or t.ramp_start_step < 0:
                raise ValueError(f"teacher {t.name}: need 0 <= ramp_start_step <= ramp_end_step")
    if sum(teacher_weight(t, 0) for t in cfg.teachers) <= 0:
        raise ValueError("at step 0 at least one teacher must have a positive weight")
    if cfg.train.warmup_ratio < 0 or cfg.train.warmup_ratio >= 1:
        raise ValueError("train.warmup_ratio must be in [0, 1)")
    if cfg.train.batch_size < 2:
        raise ValueError("train.batch_size must be >= 2 (similarity losses need pairs)")
    if cfg.train.chunk_size is not None and cfg.train.chunk_size < 1:
        raise ValueError("train.chunk_size must be >= 1 or null")


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
