"""Shared test helpers: a tiny backbone, a char-level tokenizer and fake teachers."""

import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from turkish_distill.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
)


class TinyBackbone(nn.Module):
    def __init__(self, vocab: int = 260, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden, padding_idx=0)
        self.mix = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.mix(self.embed(input_ids)))


class CharTokenizer:
    """Mimics the HF tokenizer call signature used by the code base."""

    def __call__(self, texts, max_length, padding=True, truncation=True, return_tensors="pt"):
        ids = [[(ord(c) % 255) + 1 for c in t][:max_length] or [1] for t in texts]
        width = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def make_texts(n: int) -> list[str]:
    return [f"Bu {i} numarali ornek cumle, biraz daha uzun bir metin {i * 7 % 13}." for i in range(n)]


def fake_encode_fn(dim: int, seed: int):
    """Deterministic fake teacher: hashes each text into a random-looking vector."""

    def encode(batch):
        out = np.empty((len(batch), dim), dtype=np.float32)
        for i, text in enumerate(batch):
            rng = np.random.default_rng(zlib.crc32(f"{seed}|{text}".encode("utf-8")))
            out[i] = rng.standard_normal(dim)
        return out

    return encode


def learnable_encode_fn(dim: int, seed: int):
    """Fake teacher whose vectors are a fixed linear map of the character histogram.

    Unlike `fake_encode_fn`, a small student can actually learn this target, so
    it is suitable for checking that the loss goes down.
    """
    proj = np.random.default_rng(seed).standard_normal((260, dim)).astype(np.float32)

    def encode(batch):
        out = np.zeros((len(batch), 260), dtype=np.float32)
        for i, text in enumerate(batch):
            for c in text:
                out[i, (ord(c) % 255) + 1] += 1.0
            out[i] /= max(len(text), 1)
        return out @ proj

    return encode


@pytest.fixture
def teachers():
    return [
        TeacherConfig(name="t_small", hf_id="fake/small", dim=8),
        TeacherConfig(name="t_big", hf_id="fake/big", dim=24, weight=2.0),
    ]


@pytest.fixture
def experiment(tmp_path, teachers):
    return ExperimentConfig(
        name="smoke",
        output_dir=str(tmp_path / "runs"),
        teachers=teachers,
        student=StudentConfig(hf_id="fake", max_length=64),
        data=DataConfig(cache_dir=str(tmp_path / "cache"), shard_size=40, val_size=20),
        loss=LossConfig(resim_num_samples=64),
        train=TrainConfig(
            batch_size=8, max_steps=6, lr=1e-3, num_workers=0, log_every=2, diag_every=3, ckpt_every=3, bf16=False
        ),
        tensorboard=False,
    )
