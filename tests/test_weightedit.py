import pytest
import torch

from turkish_distill.weightedit import PROFILES, gamma_for, module_group, relative_change, scale_delta


def toy(seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * scale
    names = {
        "embeddings.tok_embeddings.weight": (10, 4), "embeddings.norm.weight": (4,), "final_norm.weight": (4,),
    }
    for layer in (0, 15, 16, 21):
        names.update({f"layers.{layer}.attn.Wqkv.weight": (12, 4), f"layers.{layer}.attn.Wo.weight": (4, 4),
                      f"layers.{layer}.attn_norm.weight": (4,), f"layers.{layer}.mlp.Wi.weight": (6, 4),
                      f"layers.{layer}.mlp.Wo.weight": (4, 3), f"layers.{layer}.mlp_norm.weight": (4,)})
    return {k: r(*s) for k, s in names.items()}


STUDENT, BASE = toy(1), toy(2)


def test_module_groups():
    assert module_group("attn.Wqkv.weight") == "attn" and module_group("mlp.Wo.weight") == "mlp"
    assert module_group("attn_norm.weight") == "norm" and module_group("mlp_norm.weight") == "norm"


def test_gamma_selection_by_layer_and_group():
    late = PROFILES["wise_late_g75"]
    assert gamma_for("layers.16.attn.Wo.weight", late) == 0.75 and gamma_for("layers.21.mlp_norm.weight", late) == 0.75
    assert gamma_for("layers.15.mlp.Wi.weight", late) == 1.0
    assert gamma_for("embeddings.tok_embeddings.weight", late) == 1.0 and gamma_for("final_norm.weight", late) == 1.0
    mlp = PROFILES["wise_mlp_late_g75"]
    assert gamma_for("layers.17.mlp.Wi.weight", mlp) == 0.75 and gamma_for("layers.17.attn.Wo.weight", mlp) == 1.0
    assert gamma_for("layers.17.mlp_norm.weight", mlp) == 1.0
    allp = PROFILES["wise_all_g90"]
    assert gamma_for("embeddings.tok_embeddings.weight", allp) == 0.9 and gamma_for("layers.0.attn.Wo.weight", allp) == 0.9


def test_gamma_one_returns_the_student_exactly_and_zero_the_base():
    ident = scale_delta(STUDENT, BASE, dict(gamma=1.0, layers=None, groups=None, outside_layers=True))
    assert all(torch.equal(ident[k], STUDENT[k]) for k in STUDENT)
    zero = scale_delta(STUDENT, BASE, dict(gamma=0.0, layers=None, groups=None, outside_layers=True))
    assert all(torch.allclose(zero[k], BASE[k], atol=1e-6) for k in STUDENT)


def test_formula_and_untouched_tensors():
    spec = PROFILES["wise_late_g75"]
    out = scale_delta(STUDENT, BASE, spec)
    k = "layers.16.mlp.Wi.weight"
    assert torch.allclose(out[k], BASE[k] + 0.75 * (STUDENT[k] - BASE[k]), atol=1e-6)
    for k in STUDENT:
        if gamma_for(k, spec) == 1.0:
            assert torch.equal(out[k], STUDENT[k])
    assert not torch.equal(out["layers.16.attn.Wo.weight"], STUDENT["layers.16.attn.Wo.weight"])
    assert out["layers.16.attn.Wo.weight"] is not STUDENT["layers.16.attn.Wo.weight"]      # a copy, the student is not edited in place


def test_extrapolation_goes_past_the_student():
    out = scale_delta(STUDENT, BASE, PROFILES["extrap_all_g110"])
    k = "layers.15.attn.Wqkv.weight"
    assert torch.allclose(out[k] - BASE[k], 1.1 * (STUDENT[k] - BASE[k]), atol=1e-5)


def test_relative_change_is_proportional_to_one_minus_gamma():
    out = scale_delta(STUDENT, BASE, PROFILES["wise_all_g90"])
    rc = relative_change(out, STUDENT)
    k = "layers.0.mlp.Wo.weight"
    want = 0.1 * (BASE[k] - STUDENT[k]).norm() / STUDENT[k].norm()
    assert rc[k] == pytest.approx(want.item(), rel=1e-4)


def test_mismatched_inputs_fail_loudly():
    short = {k: v for k, v in BASE.items() if k != "final_norm.weight"}
    with pytest.raises(KeyError):
        scale_delta(STUDENT, short, PROFILES["wise_all_g90"])
    bad = dict(BASE, **{"final_norm.weight": torch.zeros(5)})
    with pytest.raises(ValueError):
        scale_delta(STUDENT, bad, PROFILES["wise_all_g90"])


def test_dtype_is_preserved():
    half = {k: v.half() for k, v in STUDENT.items()}
    out = scale_delta(half, BASE, PROFILES["wise_all_g90"])
    assert all(out[k].dtype == torch.float16 for k in half)
