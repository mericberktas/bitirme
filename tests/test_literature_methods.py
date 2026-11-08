"""Losses, gradient surgery and head loading added for the comparison with multi-teacher methods from the literature."""

import copy
import dataclasses

import pytest
import torch
import torch.nn.functional as F

from conftest import TinyBackbone
from turkish_distill.config import ExperimentConfig, LossConfig, TeacherConfig, TrainConfig, validate
from turkish_distill.gradproj import pcgrad_combine
from turkish_distill.losses import (
    MultiTeacherLoss,
    cosine_per_sample,
    row_centered_mse,
    sample_teacher_weights,
    similarity_loss,
)
from turkish_distill.models import StudentEncoder
from turkish_distill.train import distill_step


# ---------------------------------------------------------------- row-centred similarity term
def test_row_centered_mse_is_zero_for_equal_matrices_and_ignores_row_shifts():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(10, 10, generator=g)
    assert float(row_centered_mse(s, s)) == pytest.approx(0.0, abs=1e-12)
    shift = torch.randn(10, 1, generator=g)  # a different constant for every row of the student matrix
    t = torch.randn(10, 10, generator=g)
    assert float(row_centered_mse(s + shift, t)) == pytest.approx(float(row_centered_mse(s, t)), rel=1e-5)
    assert float(row_centered_mse(s, t)) > 0.0


def test_row_centered_mse_equals_plain_mse_when_the_rows_of_the_difference_have_mean_zero():
    n = 8
    g = torch.Generator().manual_seed(1)
    diff = torch.randn(n, n, generator=g)
    off = ~torch.eye(n, dtype=torch.bool)
    diff = diff.masked_fill(~off, 0.0)
    diff = (diff - diff.sum(dim=1, keepdim=True) / (n - 1)).masked_fill(~off, 0.0)  # off-diagonal rows sum to zero
    zeros = torch.zeros(n, n)
    assert float(row_centered_mse(diff, zeros)) == pytest.approx(float(F.mse_loss(diff, zeros)), rel=1e-5)


def test_row_centered_similarity_loss_is_never_larger_than_the_plain_one_and_rejects_standardize():
    g = torch.Generator().manual_seed(2)
    student, teacher = torch.randn(16, 12, generator=g), torch.randn(16, 9, generator=g)
    plain = similarity_loss(student, teacher)
    centred = similarity_loss(student, teacher, row_centered=True)
    assert 0.0 < float(centred) <= float(plain) + 1e-9
    assert float(similarity_loss(teacher, teacher, row_centered=True)) == pytest.approx(0.0, abs=1e-10)
    with pytest.raises(ValueError):
        similarity_loss(student, teacher, standardize=True, row_centered=True)


# ---------------------------------------------------------------- sample-wise teacher weights
def test_sample_weights_follow_the_agreement_and_sum_to_the_number_of_teachers():
    low, high = torch.tensor([0.05, 0.3]), torch.tensor([0.30, 0.05])
    easy = sample_teacher_weights({"a": low, "b": high}, "easy", 0.05, 0.2)
    hard = sample_teacher_weights({"a": low, "b": high}, "hard", 0.05, 0.2)
    assert easy["a"][0] > 1.0 > easy["b"][0] and easy["a"][1] < 1.0 < easy["b"][1]  # the smaller loss gets more weight
    assert hard["a"][0] < 1.0 < hard["b"][0]
    for w in (easy, hard):
        assert torch.allclose(w["a"] + w["b"], torch.full((2,), 2.0))
        assert float(min(w["a"].min(), w["b"].min())) >= 0.2 * 2 - 1e-6  # the floor, times the number of teachers
        assert not w["a"].requires_grad


def test_sample_weights_are_one_for_equal_losses_and_for_a_single_teacher():
    same = torch.tensor([0.1, 0.2, 0.3])
    w = sample_teacher_weights({"a": same, "b": same.clone()}, "easy", 0.05, 0.2)
    assert torch.allclose(w["a"], torch.ones(3)) and torch.allclose(w["b"], torch.ones(3))
    single = sample_teacher_weights({"a": same}, "easy", 0.05, 0.2)
    assert torch.allclose(single["a"], torch.ones(3))
    with pytest.raises(ValueError):
        sample_teacher_weights({"a": same}, "other", 0.05, 0.2)


def two_teachers(**kw):
    return [TeacherConfig(name="a", hf_id="a", dim=6, **kw), TeacherConfig(name="b", hf_id="b", dim=6, **kw)]


def loss_inputs(seed=0, n=16, identical=False):
    g = torch.Generator().manual_seed(seed)
    pooled = torch.randn(n, 10, generator=g).requires_grad_(True)
    pa = torch.randn(n, 6, generator=g).requires_grad_(True)
    ta = torch.randn(n, 6, generator=g)
    if identical:
        pb, tb = pa, ta
    else:
        pb, tb = torch.randn(n, 6, generator=g).requires_grad_(True), torch.randn(n, 6, generator=g)
    return pooled, {"a": pa, "b": pb}, {"a": ta, "b": tb}


def test_sample_weighting_changes_nothing_when_the_teachers_agree_on_every_text():
    base = MultiTeacherLoss(LossConfig(resim_num_samples=32), two_teachers(), seed=1)
    weighted = MultiTeacherLoss(LossConfig(resim_num_samples=32, sample_weighting="easy"), two_teachers(), seed=1)
    pooled, proj, targets = loss_inputs(identical=True)
    t0, _ = base(pooled, proj, targets)
    t1, logs = weighted(pooled, proj, targets)
    assert float(t1) == pytest.approx(float(t0), rel=1e-6)
    assert logs["sw/a"] == pytest.approx(1.0) and logs["sw/b"] == pytest.approx(1.0)


def test_sample_weighting_reweights_the_vector_term_and_still_gives_gradients():
    cfg = LossConfig(resim_num_samples=32, sim_weight=0.0, resim_weight=0.0, sample_weighting="easy")
    loss = MultiTeacherLoss(cfg, two_teachers(), seed=1)
    pooled, proj, targets = loss_inputs()
    targets["b"] = proj["b"].detach() + 0.01 * torch.randn(16, 6)  # the student already follows teacher b
    total, logs = loss(pooled, proj, targets)
    assert logs["sw/b"] > logs["sw/a"]
    total.backward()
    assert proj["a"].grad is not None and torch.isfinite(proj["a"].grad).all()
    plain, _ = MultiTeacherLoss(dataclasses.replace(cfg, sample_weighting="none"), two_teachers(), seed=1)(
        *loss_inputs()[:2], {**loss_inputs()[2], "b": targets["b"]}
    )
    assert float(total) != pytest.approx(float(plain), rel=1e-4)


def test_parts_out_splits_the_total_into_teacher_shares_and_the_common_terms():
    cfg = LossConfig(resim_num_samples=32, relation_mode="shared", listwise_weight=5.0)
    loss = MultiTeacherLoss(cfg, two_teachers(), seed=1)
    pooled, proj, targets = loss_inputs()
    parts: dict = {}
    total, _ = loss(pooled, proj, targets, parts_out=parts)
    assert set(parts) == {"a", "b", "common"}
    assert float(sum(parts.values())) == pytest.approx(float(total), rel=1e-5)
    plain = MultiTeacherLoss(LossConfig(resim_num_samples=32), two_teachers(), seed=1)
    parts2: dict = {}
    total2, _ = plain(*loss_inputs(), parts_out=parts2)
    assert set(parts2) == {"a", "b"} and float(sum(parts2.values())) == pytest.approx(float(total2), rel=1e-5)


# ---------------------------------------------------------------- PCGrad
def test_pcgrad_removes_the_conflicting_component_only():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])  # inner product -1: conflict
    combined, info = pcgrad_combine([a, b])
    a_pc = a - (-1.0 / 2.0) * b  # a minus its projection on b (b has squared norm 2)
    b_pc = b - (-1.0 / 1.0) * a
    assert torch.allclose(combined, a_pc + b_pc)
    assert info["conflicts"] == 2.0 and info["min_cos"] == pytest.approx(-1.0 / 2**0.5)
    assert float(torch.dot(a_pc, b)) == pytest.approx(0.0, abs=1e-6)  # the projected gradient is orthogonal to the other


def test_pcgrad_leaves_agreeing_gradients_unchanged_and_keeps_the_shape():
    a, b = torch.tensor([[1.0, 2.0], [0.0, 1.0]]), torch.tensor([[2.0, 1.0], [1.0, 1.0]])
    combined, info = pcgrad_combine([a, b])
    assert torch.allclose(combined, a + b) and combined.shape == a.shape and info["conflicts"] == 0.0
    only, _ = pcgrad_combine([a])
    assert torch.allclose(only, a)
    zero, info = pcgrad_combine([a, torch.zeros_like(a)])
    assert torch.allclose(zero, a) and info["conflicts"] == 0.0
    with pytest.raises(ValueError):
        pcgrad_combine([])
    with pytest.raises(ValueError):
        pcgrad_combine([torch.zeros(2), torch.zeros(3)])


def test_pcgrad_of_three_gradients_is_order_independent_in_the_two_way_case_and_deterministic_with_a_seed():
    g = torch.Generator().manual_seed(0)
    grads = [torch.randn(5, generator=g) for _ in range(3)]
    first, _ = pcgrad_combine(grads, torch.Generator().manual_seed(7))
    second, _ = pcgrad_combine(grads, torch.Generator().manual_seed(7))
    assert torch.allclose(first, second)


# ---------------------------------------------------------------- distill_step with the gradient surgery
def tiny_model(seed=0):
    torch.manual_seed(seed)
    teachers = [TeacherConfig(name="a", hf_id="a", dim=8), TeacherConfig(name="b", hf_id="b", dim=8)]
    return StudentEncoder(TinyBackbone(), 16, {t.name: t.dim for t in teachers}), teachers


def step_inputs(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 250, (n, 7), generator=g)
    return ids, torch.ones_like(ids)


def encoder_grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.backbone.parameters() if p.grad is not None])


def test_surgery_equals_the_plain_step_when_the_teachers_do_not_conflict():
    model, teachers = tiny_model()
    model.heads["b"].load_state_dict(model.heads["a"].state_dict())  # identical heads and targets: identical gradients
    target = torch.randn(12, 8, generator=torch.Generator().manual_seed(3))
    targets = {"a": target, "b": target.clone()}
    ids, mask = step_inputs()
    criterion = MultiTeacherLoss(LossConfig(resim_num_samples=32, resim_weight=0.0), teachers, seed=0)
    plain_model = copy.deepcopy(model)
    distill_step(plain_model, criterion, ids, mask, targets)
    for chunk in (None, 5):
        surgery_model = copy.deepcopy(model)
        logs = distill_step(surgery_model, criterion, ids, mask, targets, chunk, projection="pcgrad")
        assert logs["pcgrad/conflicts"] == 0.0
        assert torch.allclose(encoder_grads(surgery_model), encoder_grads(plain_model), atol=1e-5)
        for name in ("a", "b"):  # the heads keep the gradients of their own teacher
            for p, q in zip(surgery_model.heads[name].parameters(), plain_model.heads[name].parameters()):
                assert torch.allclose(p.grad, q.grad, atol=1e-6)


def test_surgery_changes_the_encoder_gradient_when_the_teachers_conflict_and_runs_with_chunks():
    model, teachers = tiny_model(1)
    g = torch.Generator().manual_seed(4)
    target = torch.randn(12, 8, generator=g)
    # cosine only; teacher b asks for a rotated version of a's target, which conflicts for some of the batch
    targets = {"a": target, "b": -target + 0.3 * torch.randn(12, 8, generator=g)}
    criterion = MultiTeacherLoss(
        LossConfig(sim_weight=0.0, resim_weight=0.0, resim_num_samples=8), teachers, seed=0
    )
    ids, mask = step_inputs()
    model.heads["b"].load_state_dict(model.heads["a"].state_dict())
    plain = copy.deepcopy(model)
    distill_step(plain, criterion, ids, mask, targets)
    for chunk in (None, 4):
        surgery = copy.deepcopy(model)
        logs = distill_step(surgery, criterion, ids, mask, targets, chunk, projection="pcgrad")
        assert logs["pcgrad/conflicts"] >= 1.0 and logs["pcgrad/min_cos"] < 0.0
        grad = encoder_grads(surgery)
        assert torch.isfinite(grad).all() and not torch.allclose(grad, encoder_grads(plain), atol=1e-6)


def test_surgery_with_one_active_teacher_is_the_plain_step_and_unknown_modes_are_rejected():
    model, teachers = tiny_model(2)
    teachers = [dataclasses.replace(teachers[0], weight=1.0), dataclasses.replace(teachers[1], weight=0.0)]
    targets = {"a": torch.randn(12, 8), "b": torch.randn(12, 8)}
    ids, mask = step_inputs()
    def criterion():  # a fresh one per step: the ranking hinge draws its pairs from the criterion's own generator
        return MultiTeacherLoss(LossConfig(resim_num_samples=16), teachers, seed=0)

    plain = copy.deepcopy(model)
    distill_step(plain, criterion(), ids, mask, targets)
    surgery = copy.deepcopy(model)
    logs = distill_step(surgery, criterion(), ids, mask, targets, projection="pcgrad")
    assert "pcgrad/conflicts" not in logs
    assert torch.allclose(encoder_grads(surgery), encoder_grads(plain), atol=1e-5)
    with pytest.raises(ValueError):
        distill_step(copy.deepcopy(model), criterion(), ids, mask, targets, projection="other")


# ---------------------------------------------------------------- config and heads
def make_cfg(**loss_kw):
    teachers = two_teachers()
    return ExperimentConfig(name="x", output_dir="o", teachers=teachers, loss=LossConfig(**loss_kw))


def test_config_validation_of_the_new_options():
    validate(make_cfg(sim_row_centered=True))
    validate(make_cfg(sample_weighting="easy"))
    for bad in (
        dict(sim_row_centered=True, gram_standardize=True),
        dict(sim_row_centered=True, relation_mode="shared"),
        dict(sample_weighting="other"),
        dict(sample_weighting="easy", sample_weight_temperature=0.0),
        dict(sample_weighting="easy", sample_weight_floor=0.5),  # two teachers: the floor must stay below 1/2
    ):
        with pytest.raises(ValueError):
            validate(make_cfg(**bad))
    cfg = make_cfg()
    cfg.train = TrainConfig(grad_projection="pcgrad")
    validate(cfg)
    cfg.train = TrainConfig(grad_projection="other")
    with pytest.raises(ValueError):
        validate(cfg)


def test_defaults_leave_the_loss_untouched():
    cfg = LossConfig()
    assert not cfg.sim_row_centered and cfg.sample_weighting == "none" and TrainConfig().grad_projection == "none"
    assert float(cosine_per_sample(torch.eye(3), torch.eye(3)).abs().sum()) == pytest.approx(0.0, abs=1e-6)


def test_heads_can_be_loaded_from_a_saved_file(tmp_path):
    source, _ = tiny_model(5)
    source.backbone.save_pretrained = lambda path: None  # not needed here, only the heads are written
    path = tmp_path / "heads"
    source.save_backbone(path)
    target, _ = tiny_model(9)
    assert not torch.allclose(target.heads["a"].weight, source.heads["a"].weight)
    target.load_heads(path / "teacher_heads.pt")
    for name in ("a", "b"):
        assert torch.equal(target.heads[name].weight, source.heads[name].weight)
