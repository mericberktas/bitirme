"""Student encoder: HF backbone + pooling + one projection head per teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import StudentConfig, TeacherConfig


def pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "cls":
        return hidden[:, 0]
    if mode != "mean":
        raise ValueError(f"unknown pooling {mode!r}")
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class StudentEncoder(nn.Module):
    """`backbone(input_ids=..., attention_mask=...)` must expose `.last_hidden_state`."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        teacher_dims: dict[str, int],
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, dim) for name, dim in teacher_dims.items()})

    @classmethod
    def from_pretrained(cls, student: StudentConfig, teachers: list[TeacherConfig]) -> "StudentEncoder":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(student.hf_id, revision=student.revision)
        hidden = backbone.config.hidden_size
        return cls(backbone, hidden, {t.name: t.dim for t in teachers}, student.pooling)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return pool(out.last_hidden_state, attention_mask, self.pooling)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled = self.encode(input_ids, attention_mask)
        return pooled, {name: head(pooled) for name, head in self.heads.items()}

    def save_backbone(self, path: str | Path, tokenizer: Optional[object] = None) -> None:
        """Save the backbone (HF format) so it can be reloaded with AutoModel."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(path)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(path)
        torch.save(self.heads.state_dict(), path / "teacher_heads.pt")
