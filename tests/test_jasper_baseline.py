"""Jasper-style baseline: concatenated teacher target, relation terms on the head output, staged freezing, head export."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from conftest import CharTokenizer, make_texts
from test_train_smoke import build
from turkish_distill.config import LossConfig, TeacherConfig, TrainConfig, load_config
from turkish_distill.export_head import head_weights
from turkish_distill.losses import MultiTeacherLoss
from turkish_distill.models import StudentEncoder
from turkish_distill.targets import ArrayCache, ConcatCache
from turkish_distill.train import apply_target_methods, freeze_stage, run_training, stage_parameters


def unit_rows(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_concat_target_is_unit_length_and_its_gram_is_the_weighted_teacher_average():
    rng = np.random.default_rng(0)
    a, b = unit_rows(rng, 30, 6), 3.0 * unit_rows(rng, 30, 4)  # b not normalised: the cache normalises it
    cache = ConcatCache({"a": ArrayCache(a), "b": ArrayCache(b)}, {"a": 1.0, "b": 2.0})
    assert len(cache) == 30 and cache.dim == 10
    v = cache.get(np.arange(30))
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    np.testing.assert_allclose(v @ v.T, (a @ a.T) / 3 + 2 * (bn @ bn.T) / 3, atol=1e-5)
    with pytest.raises(ValueError):
        ConcatCache({"a": ArrayCache(a), "b": ArrayCache(b)}, {"a": 1.0, "b": 0.0})


def test_apply_target_methods_replaces_the_teachers_by_one_concat_target(experiment):
    exp = copy.deepcopy(experiment)
    exp.fusion.mode = "concat"
    caches = {t.name: ArrayCache(np.random.default_rng(i).standard_normal((50, t.dim)).astype(np.float32))
              for i, t in enumerate(exp.teachers)}
    cfg, out = apply_target_methods(exp, caches)
    assert [t.name for t in cfg.teachers] == ["concat"] and cfg.teachers[0].dim == 8 + 24
    assert set(out) == {"concat"} and out["concat"].get([0, 1]).shape == (2, 32)
    # teacher weights 1 and 2 (fixture): the concat blocks carry sqrt(1/3) and sqrt(2/3)
    assert out["concat"].scales == pytest.approx({"t_small": (1 / 3) ** 0.5, "t_big": (2 / 3) ** 0.5})


def test_relation_on_head_moves_the_relation_gradient_from_pooled_to_the_head():
    torch.manual_seed(0)
    teachers = [TeacherConfig(name="a", hf_id="x", dim=5)]
    targets = {"a": torch.randn(12, 5)}
    for where in ("pooled", "head"):
        cfg = LossConfig(cosine_weight=0.0, sim_weight=200.0, resim_weight=20.0, resim_num_samples=64, relation_on=where)
        pooled = torch.randn(12, 7, requires_grad=True)
        proj = {"a": torch.randn(12, 5, requires_grad=True)}
        loss, logs = MultiTeacherLoss(cfg, teachers)(pooled, proj, targets)
        loss.backward()
        assert "sim/a" in logs and "resim/a" in logs
        if where == "pooled":
            assert pooled.grad is not None and proj["a"].grad is None
        else:
            assert pooled.grad is None and proj["a"].grad is not None and float(proj["a"].grad.abs().sum()) > 0


class LayeredBackbone(nn.Module):
    """Embeddings, two numbered layers and a final norm, named like a HF encoder."""

    def __init__(self, vocab: int = 260, hidden: int = 16):
        super().__init__()
        self.embeddings = nn.Embedding(vocab, hidden, padding_idx=0)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, attention_mask):
        h = self.embeddings(input_ids)
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return SimpleNamespace(last_hidden_state=self.final_norm(h))


def test_freeze_stages_and_their_parameters():
    t = TrainConfig(max_steps=10, head_only_until=0.3, last_layers_until=0.8, last_layers=1)
    assert [freeze_stage(t, s) for s in (0, 2, 3, 7, 8, 9)] == ["heads", "heads", "last", "last", "all", "all"]
    assert freeze_stage(TrainConfig(max_steps=10), 0) == "all"
    model = StudentEncoder(LayeredBackbone(), 16, {"a": 4})
    names = {id(p): n for n, p in model.named_parameters()}
    assert stage_parameters(model, "all", 1) is None
    assert {names[i] for i in stage_parameters(model, "heads", 1)} == {"heads.a.weight", "heads.a.bias"}
    assert {names[i] for i in stage_parameters(model, "last", 1)} == {
        "heads.a.weight", "heads.a.bias", "backbone.layers.1.weight", "backbone.layers.1.bias",
        "backbone.final_norm.weight", "backbone.final_norm.bias",
    }


def test_staged_jasper_run_trains_only_the_allowed_parameters(experiment, tmp_path):
    texts = make_texts(120)
    for until, frozen_prefixes in ((0.99, ("backbone.",)), (0.0, ())):
        exp = copy.deepcopy(experiment)
        exp.fusion.mode = "concat"
        exp.loss.relation_on = "head"
        exp.train.max_steps = 4
        exp.train.head_only_until = until
        _, caches = build(exp, texts)
        cfg, caches = apply_target_methods(exp, caches)
        torch.manual_seed(0)
        model = StudentEncoder(LayeredBackbone(), 16, {t.name: t.dim for t in cfg.teachers})
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        out = run_training(cfg, tmp_path / f"run{until}", model, CharTokenizer(), texts, caches, torch.device("cpu"))
        assert out["final_step"] == 4
        for n, p in model.named_parameters():
            same = torch.equal(before[n], p.detach())
            if n.startswith(frozen_prefixes):
                assert same, n
            elif n.startswith("heads.") or n.startswith("backbone.layers."):
                assert not same, n
    # last-layers stage: the first layer and the embeddings stay, the last layer moves
    exp = copy.deepcopy(experiment)
    exp.fusion.mode = "concat"
    exp.loss.relation_on = "head"
    exp.train.max_steps = 4
    exp.train.last_layers_until = 0.99
    exp.train.last_layers = 1
    _, caches = build(exp, texts)
    cfg, caches = apply_target_methods(exp, caches)
    torch.manual_seed(0)
    model = StudentEncoder(LayeredBackbone(), 16, {t.name: t.dim for t in cfg.teachers})
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    run_training(cfg, tmp_path / "last", model, CharTokenizer(), texts, caches, torch.device("cpu"))
    moved = {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}
    assert "backbone.layers.1.weight" in moved and "heads.concat.weight" in moved
    assert not moved & {"backbone.layers.0.weight", "backbone.embeddings.weight"}


def test_head_export_weights_and_rectangular_dense(tmp_path):
    model = StudentEncoder(LayeredBackbone(), 16, {"concat": 32})
    torch.save(model.heads.state_dict(), tmp_path / "teacher_heads.pt")
    w, b = head_weights(tmp_path / "teacher_heads.pt", "concat")
    assert w.shape == (32, 16) and b.shape == (32,)
    with pytest.raises(ValueError, match="no linear head"):
        head_weights(tmp_path / "teacher_heads.pt", "other")
    st = pytest.importorskip("sentence_transformers")
    from turkish_distill.posthoc import append_linear_head

    class Identity(nn.Module):
        def forward(self, features):
            return features

    combined = append_linear_head(st.SentenceTransformer(modules=[Identity()], device="cpu"), w, b, normalize=True).to("cpu")
    x = torch.randn(3, 16)
    out = combined[1]({"sentence_embedding": x})["sentence_embedding"]
    np.testing.assert_allclose(out.detach().numpy(), x.numpy() @ w.T + b, atol=1e-5)


def test_config_validation_and_unchanged_fingerprint(tmp_path):
    raw = {"name": "x", "output_dir": str(tmp_path), "teachers": [{"name": "t", "hf_id": "h", "dim": 4},
                                                                  {"name": "u", "hf_id": "h", "dim": 6}]}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    off = load_config(path)
    same = load_config(path, ["loss.relation_on=pooled", "train.head_only_until=0.0", "train.last_layers=3"])
    assert same.fingerprint() == off.fingerprint()  # defaults are left out: old runs resume without a warning
    on = load_config(path, ["fusion.mode=concat", "loss.relation_on=head", "train.head_only_until=0.3",
                            "train.last_layers_until=0.83"])
    assert on.fingerprint() != off.fingerprint()
    for bad in (["loss.relation_on=both"], ["loss.relation_on=head", "loss.relation_mode=shared"],
                ["loss.relation_on=head", "loss.listwise_weight=1"], ["train.head_only_until=1.0"],
                ["train.head_only_until=0.5", "train.last_layers_until=0.3"], ["train.last_layers=0"],
                ["fusion.mode=stack"]):
        with pytest.raises(ValueError):
            load_config(path, bad)


def test_full_pair_hinge_matches_a_brute_force_loop():
    from turkish_distill.losses import ranking_hinge

    torch.manual_seed(1)
    s, t = torch.rand(7), torch.rand(7)
    got = ranking_hinge(s, t, 0.1, 0)
    terms = []
    for a in range(7):
        for b in range(a + 1, 7):
            label = 1.0 if t[b] - t[a] < 0 else -1.0  # Jasper: rank_label = where(t_diff < 0, 1, -1)
            terms.append(max(0.0, float(s[b] - s[a]) * label + 0.1))
    assert float(got) == pytest.approx(sum(terms) / len(terms), rel=1e-5)


def test_jasper_combine_averages_the_relation_terms_over_active_heads():
    torch.manual_seed(0)
    teachers = [TeacherConfig(name="concat", hf_id="x", dim=6),
                TeacherConfig(name="concat_r4", hf_id="x", dim=4, loss_weights={"cosine": 0.0})]
    pooled = torch.randn(10, 5)
    target = torch.randn(10, 6)
    proj = {"concat": torch.randn(10, 6), "concat_r4": torch.randn(10, 4)}
    targets = {"concat": target, "concat_r4": target}
    cfg = LossConfig(relation_on="head", teacher_combine="jasper", resim_num_samples=0)
    total, logs = MultiTeacherLoss(cfg, teachers)(pooled, proj, targets)
    expect = 10 * logs["cos/concat"] + (200 * (logs["sim/concat"] + logs["sim/concat_r4"])
                                       + 20 * (logs["resim/concat"] + logs["resim/concat_r4"])) / 2
    assert float(total) == pytest.approx(expect, rel=1e-5)
    single, slogs = MultiTeacherLoss(cfg, teachers[:1])(pooled, {"concat": proj["concat"]}, {"concat": target})
    assert float(single) == pytest.approx(10 * slogs["cos/concat"] + 200 * slogs["sim/concat"] + 20 * slogs["resim/concat"], rel=1e-5)


def test_staged_learning_rate_restarts_at_every_stage():
    from turkish_distill.train import staged_lr_factor, stage_bounds

    t = TrainConfig(max_steps=100, lr=1e-4, warmup_ratio=0.1, head_only_until=0.3, last_layers_until=0.8,
                    stage_lrs=[1e-4, 8e-5, 7e-5], stage_schedule="linear")
    assert stage_bounds(t) == [("heads", 0, 30), ("last", 30, 80), ("all", 80, 100)]
    f = staged_lr_factor(t)
    assert f(0) == pytest.approx(1 / 3)          # warmup 3 steps of 30
    assert f(2) == pytest.approx(1.0)
    assert f(29) < 0.05 and f(30) == pytest.approx(0.8 / 5)  # restart: 8e-5, warmup 5 steps of 50
    assert f(34) == pytest.approx(0.8) and f(80) == pytest.approx(0.7 / 2) and f(81) == pytest.approx(0.7)
    assert f(99) == pytest.approx(0.7 * (1 - 17 / 18))
    cos = staged_lr_factor(TrainConfig(max_steps=100, lr=1e-4, warmup_ratio=0.1, head_only_until=0.3,
                                       stage_lrs=[1e-4, 1e-4, 1e-4], stage_schedule="cosine"))
    assert cos(3) == pytest.approx(1.0) and cos(29) == pytest.approx(0.5 * (1 + np.cos(np.pi * 26 / 27)))
    assert [freeze_stage(t, s) for s in (29, 30, 79, 80)] == ["heads", "last", "last", "all"]


def test_final_stage_layers_limits_the_last_stage():
    model = StudentEncoder(LayeredBackbone(), 16, {"a": 4})
    names = {id(p): n for n, p in model.named_parameters()}
    kept = {names[i] for i in stage_parameters(model, "all", 1, final_layers=1)}
    assert "backbone.layers.1.weight" in kept and "backbone.layers.0.weight" not in kept
    assert "backbone.embeddings.weight" not in kept and "backbone.final_norm.weight" in kept


def test_faithful_jasper_run_with_stage_lrs_and_reduction_heads(experiment, tmp_path, capsys):
    texts = make_texts(120)
    exp = copy.deepcopy(experiment)
    exp.fusion.mode = "concat"
    exp.fusion.reduce_dims = [8]
    exp.loss.relation_on = "head"
    exp.loss.teacher_combine = "jasper"
    exp.loss.resim_num_samples = 0
    exp.train.max_steps = 6
    exp.train.head_only_until = 0.34
    exp.train.last_layers_until = 0.67
    exp.train.last_layers = 1
    exp.train.final_stage_layers = 1
    exp.train.stage_lrs = [1e-3, 8e-4, 7e-4]
    exp.train.stage_schedule = "cosine"
    _, caches = build(exp, texts)
    cfg, caches = apply_target_methods(exp, caches)
    assert [t.name for t in cfg.teachers] == ["concat", "concat_r8"]
    assert cfg.teachers[1].ramp_start_step == 4 and cfg.teachers[1].loss_weights == {"cosine": 0.0}
    torch.manual_seed(0)
    model = StudentEncoder(LayeredBackbone(), 16, {t.name: t.dim for t in cfg.teachers})
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = run_training(cfg, tmp_path / "faithful", model, CharTokenizer(), texts, caches, torch.device("cpu"))
    assert out["final_step"] == 6
    assert "reduction heads ['concat_r8'] initialised" in capsys.readouterr().out
    moved = {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}
    assert {"heads.concat_r8.weight", "heads.concat.weight", "backbone.layers.1.weight"} <= moved
    assert not moved & {"backbone.layers.0.weight", "backbone.embeddings.weight"}  # frozen in every stage here
    w, b = head_weights(tmp_path / "faithful" / "final" / "teacher_heads.pt", "concat_r8")
    assert w.shape == (8, 16)


def test_reduction_head_init_averages_row_groups():
    from turkish_distill.train import init_reduction_heads

    model = StudentEncoder(LayeredBackbone(), 16, {"concat": 10, "concat_r4": 4})
    assert init_reduction_heads(model) == ["concat_r4"]
    full = model.heads["concat"].weight.detach()
    groups = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]  # np.array_split of 10 rows into 4
    for i, g in enumerate(groups):
        torch.testing.assert_close(model.heads["concat_r4"].weight[i].detach(), full[g].mean(0))


def test_new_jasper_options_keep_the_fingerprint_and_validate(tmp_path):
    raw = {"name": "x", "output_dir": str(tmp_path), "teachers": [{"name": "t", "hf_id": "h", "dim": 4},
                                                                  {"name": "u", "hf_id": "h", "dim": 6}]}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    off = load_config(path)
    same = load_config(path, ["loss.cosine_reduction=mean", "loss.teacher_combine=weighted_mean", "fusion.reduce_from=-1",
                              "train.stage_schedule=linear", "train.final_stage_layers=0", "fusion.reduce_weight=1.0"])
    assert same.fingerprint() == off.fingerprint()
    ok = ["fusion.mode=concat", "loss.relation_on=head", "fusion.reduce_dims=[768]", "train.head_only_until=0.3",
          "train.last_layers_until=0.83", "train.stage_lrs=[1e-4, 8e-5, 7e-5]", "loss.teacher_combine=jasper",
          "loss.resim_num_samples=0"]
    assert load_config(path, ok).fingerprint() != off.fingerprint()
    for bad in (["train.stage_lrs=[1e-4, 8e-5, 7e-5]"], ["train.head_only_until=0.3", "train.stage_lrs=[1e-4]"],
                ["fusion.reduce_dims=[768]"], ["loss.teacher_combine=max"], ["loss.resim_num_samples=-1"],
                ["train.final_stage_layers=-1"], ["train.stage_schedule=step"]):
        with pytest.raises(ValueError):
            load_config(path, bad)
