"""Projection heads of the student: linear (default) and MLP."""

import pytest
import torch
import torch.nn as nn

from turkish_distill.config import ExperimentConfig, StudentConfig, TeacherConfig, load_config, validate
from turkish_distill.models import StudentEncoder, make_head


class TinyBackbone(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.emb = nn.Embedding(50, hidden)

    def forward(self, input_ids, attention_mask):
        class Out:
            pass

        out = Out()
        out.last_hidden_state = self.emb(input_ids)
        return out


def encoder(head="linear", head_hidden=0, dims=None):
    return StudentEncoder(TinyBackbone(), 16, dims or {"a": 10, "b": 8}, "mean", head, head_hidden)


def batch():
    ids = torch.randint(0, 50, (4, 6))
    return ids, torch.ones_like(ids)


def test_default_head_is_the_old_linear_head():
    model = encoder()
    assert all(isinstance(h, nn.Linear) for h in model.heads.values())
    assert set(model.heads.state_dict()) == {"a.weight", "a.bias", "b.weight", "b.bias"}   # unchanged keys: old heads still load


def test_mlp_head_shapes_and_hidden_width():
    model = encoder("mlp")
    pooled, proj = model(*batch())
    assert pooled.shape == (4, 16) and proj["a"].shape == (4, 10) and proj["b"].shape == (4, 8)
    assert [m.out_features for m in model.heads["a"] if isinstance(m, nn.Linear)] == [16, 10]   # 0 = backbone width
    wide = encoder("mlp", head_hidden=40)
    assert [m.out_features for m in wide.heads["a"] if isinstance(m, nn.Linear)] == [40, 10]


def test_mlp_head_gets_gradient_in_both_layers():
    model = encoder("mlp")
    _, proj = model(*batch())
    proj["a"].sum().backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.heads["a"].parameters())


def test_heads_roundtrip_through_state_dict():
    src, dst = encoder("mlp", 24), encoder("mlp", 24)
    dst.heads.load_state_dict(src.heads.state_dict())
    x = torch.randn(3, 16)
    assert torch.allclose(src.heads["a"](x), dst.heads["a"](x))


def test_make_head_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_head("conv", 16, 8)


def test_validation_and_command_line_override():
    def cfg(**kw):
        return ExperimentConfig(name="x", output_dir="o", student=StudentConfig(**kw), teachers=[TeacherConfig(name="a", hf_id="a", dim=4)])

    validate(cfg(head="mlp", head_hidden=64))
    with pytest.raises(ValueError):
        validate(cfg(head="conv"))
    with pytest.raises(ValueError):
        validate(cfg(head="mlp", head_hidden=-1))
    loaded = load_config("configs/stage1_two_bge_m3_modernbert_tr_embed.yaml", ["student.head=mlp", "student.head_hidden=1536"])
    assert loaded.student.head == "mlp" and loaded.student.head_hidden == 1536
