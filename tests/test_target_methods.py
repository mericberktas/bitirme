"""Methods a-d: vector centring, Gram standardisation, shared relation target, aligned teacher fusion."""

import dataclasses

import numpy as np
import pytest
import torch

from turkish_distill.config import ExperimentConfig, FusionConfig, LossConfig, TeacherConfig, validate
from turkish_distill.losses import MultiTeacherLoss, similarity_loss, zscore
from turkish_distill.targets import ArrayCache, CenteredCache, compute_mean, fuse_teachers
from turkish_distill.train import apply_target_methods


class FakeCache(ArrayCache):
    """In-memory teacher cache of L2-normalised vectors."""

    def __init__(self, vectors):
        v = np.asarray(vectors, dtype=np.float32)
        super().__init__((v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16))


def cone_vectors(n=400, d=16, offset=6.0, seed=0):
    """Vectors in a narrow cone: a large common direction plus a small spread (like mE5)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=d)
    return offset * common / np.linalg.norm(common) + rng.normal(size=(n, d)) * 0.5


def mean_offdiag_cosine(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sims = x @ x.T
    n = len(x)
    return float((sims.sum() - n) / (n * (n - 1)))


# ---- (a) centring -------------------------------------------------------------------------------------------

def test_centred_cache_opens_up_a_cone_and_keeps_unit_norm():
    cache = FakeCache(cone_vectors())
    centred = CenteredCache(cache)
    raw, new = cache.get(np.arange(400)), centred.get(np.arange(400))
    assert mean_offdiag_cosine(raw) > 0.8
    assert abs(mean_offdiag_cosine(new)) < 0.05
    assert np.allclose(np.linalg.norm(new, axis=1), 1.0, atol=1e-5)
    assert centred.dim == cache.dim and len(centred) == len(cache)
    np.testing.assert_allclose(centred.mean, compute_mean(cache), atol=1e-6)


def test_apply_target_methods_only_touches_flagged_teachers():
    cfg = ExperimentConfig(
        name="x", output_dir="o",
        teachers=[TeacherConfig(name="a", hf_id="a", dim=16, center=True), TeacherConfig(name="b", hf_id="b", dim=16)],
    )
    caches = {"a": FakeCache(cone_vectors(seed=1)), "b": FakeCache(cone_vectors(seed=2))}
    cfg2, out = apply_target_methods(cfg, caches)
    assert cfg2 is cfg and isinstance(out["a"], CenteredCache) and out["b"] is caches["b"]


# ---- capacity-aware PCA target ---------------------------------------------------------------------------------

def low_rank_vectors(n=500, d=24, rank=6, noise=0.05, offset=2.0, seed=0):
    """Vectors that live (almost) in a `rank`-dimensional subspace around a common offset direction."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, rank)))[0]
    common = rng.normal(size=d)
    return offset * common / np.linalg.norm(common) + rng.normal(size=(n, rank)) @ basis.T + rng.normal(size=(n, d)) * noise


def test_pca_cache_keeps_a_low_rank_target_and_removes_the_rest():
    from turkish_distill.targets import PCACache

    cache = FakeCache(low_rank_vectors())
    idx = np.arange(500)
    full = cache.get(idx)
    exact = PCACache(cache, n_components=7, fit_samples=500)   # rank 6 + the offset direction: nothing is lost
    cos_exact = (exact.get(idx) * full).sum(axis=1).mean()
    assert cos_exact > 0.995 and exact.kept_variance > 0.99   # what is left is the noise and float16 rounding
    tight = PCACache(cache, n_components=2, fit_samples=500)   # fewer components than the signal has
    cos_tight = (tight.get(idx) * full).sum(axis=1).mean()
    assert 0.5 < cos_tight < cos_exact and tight.kept_variance < exact.kept_variance
    out = tight.get(idx)
    assert out.shape == full.shape and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert tight.dim == cache.dim and len(tight) == len(cache)
    # the part of a vector outside the kept subspace is gone: projecting the target again changes (almost) nothing
    # (not exactly: the target is re-normalised, which moves the mean the second fit centres on)
    again = PCACache(ArrayCache(out), n_components=2, fit_samples=500)
    assert (again.get(idx) * out).sum(axis=1).mean() > 0.99


def test_pca_cache_is_deterministic_for_a_seed():
    from turkish_distill.targets import PCACache

    cache = FakeCache(low_rank_vectors(seed=3))
    a, b = PCACache(cache, 3, fit_samples=200, seed=1), PCACache(cache, 3, fit_samples=200, seed=1)
    np.testing.assert_allclose(a.get(np.arange(50)), b.get(np.arange(50)), atol=1e-6)


def test_pca_dim_is_applied_to_the_flagged_teacher_and_validated():
    from turkish_distill.targets import PCACache

    cfg = ExperimentConfig(
        name="x", output_dir="o",
        teachers=[TeacherConfig(name="a", hf_id="a", dim=24, pca_dim=4, pca_fit_samples=300), TeacherConfig(name="b", hf_id="b", dim=24)],
    )
    validate(cfg)
    caches = {"a": FakeCache(low_rank_vectors(seed=1)), "b": FakeCache(low_rank_vectors(seed=2))}
    cfg2, out = apply_target_methods(cfg, caches)
    assert cfg2 is cfg and isinstance(out["a"], PCACache) and out["b"] is caches["b"]
    for bad in (dict(pca_dim=24), dict(pca_dim=1), dict(pca_dim=4, pca_fit_samples=3), dict(pca_dim=4, center=True)):
        broken = dataclasses.replace(cfg, teachers=[TeacherConfig(name="a", hf_id="a", dim=24, **bad)])
        with pytest.raises(ValueError):
            validate(broken)
    fused = dataclasses.replace(cfg, fusion=FusionConfig(mode="aligned_mean", reference="a", dim=4, fit_samples=50))
    with pytest.raises(ValueError):
        validate(fused)


# ---- (b) Gram standardisation ---------------------------------------------------------------------------------

def test_gram_standardisation_gives_the_teacher_the_reference_geometry_and_keeps_the_student_raw():
    from turkish_distill.losses import offdiag_similarities, standardized_teacher_pairs

    torch.manual_seed(0)
    x = torch.randn(12, 8)
    teacher = torch.randn(12, 5)
    cone_teacher = teacher + 6.0 * torch.randn(1, 5)  # one shared direction: a cone, large offset, small spread
    for t in (teacher, cone_teacher):
        pairs = standardized_teacher_pairs(t, 0.30, 0.07)
        # whatever the teacher's own offset and spread, the target has the reference mean and std
        assert pairs.mean().item() == pytest.approx(0.30, abs=1e-5)
        assert pairs.std(unbiased=False).item() == pytest.approx(0.07, abs=1e-4)
        # and the teacher's own order of pairs is kept
        assert torch.equal(torch.argsort(pairs), torch.argsort(offdiag_similarities(t)))
    # The plain loss reacts to a cone teacher; the standardised one gives both teachers a target of the same
    # geometry, so the two losses of the same student differ only through the pattern.
    assert similarity_loss(x, cone_teacher).item() > 0.05
    # z-scoring really gives mean 0 and std 1, and is unchanged by an affine transformation
    v = torch.randn(100)
    z = zscore(v * 3 + 7)
    assert z.mean().item() == pytest.approx(0.0, abs=1e-5) and z.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)
    assert torch.allclose(z, zscore(v), atol=1e-4)


def test_standardised_similarity_loss_still_pins_down_the_student_offset_and_scale():
    """The point of the fix: the student's own cosines are NOT standardised, so a student that stays in a narrow
    cone is penalised (a Pearson-style loss, which z-scores both sides, would not)."""
    torch.manual_seed(1)
    teacher = torch.randn(16, 8)
    ref_mean, ref_std = 0.30, 0.07
    spread_student = torch.randn(16, 8)  # unrelated pattern, wide spread: a large loss
    cone_student = torch.randn(16, 8) + 8.0 * torch.randn(1, 8)  # narrow cone
    loss_cone = similarity_loss(cone_student, teacher, True, ref_mean, ref_std).item()
    loss_spread = similarity_loss(spread_student, teacher, True, ref_mean, ref_std).item()
    assert loss_cone > 0.3  # a cone student has cosines near 0.9, far from the reference mean of 0.3
    assert loss_cone > loss_spread


# ---- (c) shared relation target --------------------------------------------------------------------------------

def _two_teacher_loss(mode, standardize=False, weights=(1.0, 1.0)):
    teachers = [
        TeacherConfig(name="a", hf_id="a", dim=8, weight=weights[0]),
        TeacherConfig(name="b", hf_id="b", dim=12, weight=weights[1]),
    ]
    cfg = LossConfig(relation_mode=mode, gram_standardize=standardize)
    return MultiTeacherLoss(cfg, teachers, seed=0)


def test_shared_relation_with_identical_teachers_matches_the_standardised_per_teacher_loss():
    torch.manual_seed(2)
    pooled = torch.randn(10, 6, requires_grad=True)
    t = torch.randn(10, 8)
    projections = {"a": torch.randn(10, 8), "b": torch.randn(10, 8)}
    targets = {"a": t, "b": t.clone()}
    shared, logs_shared = _two_teacher_loss("shared")(pooled, projections, targets)
    per, logs_per = _two_teacher_loss("per_teacher", standardize=True)(pooled, projections, targets)
    assert logs_shared["sim/shared"] == pytest.approx(logs_per["sim/a"], abs=1e-5)
    # The ranking hinge samples its pairs with its own generator stream; the sample counts differ between the
    # two modes (one draw vs two), so compare the relation-free part and the standardised similarity term only.
    assert "resim/shared" in logs_shared and "resim/a" in logs_per
    assert torch.isfinite(shared) and torch.isfinite(per)


def test_shared_relation_is_logged_once_and_skips_zero_weight_teachers():
    torch.manual_seed(3)
    pooled = torch.randn(8, 5, requires_grad=True)
    projections = {"a": torch.randn(8, 8), "b": torch.randn(8, 12)}
    targets = {"a": torch.randn(8, 8), "b": torch.randn(8, 12)}
    total, logs = _two_teacher_loss("shared", weights=(1.0, 0.0))(pooled, projections, targets)
    assert "sim/shared" in logs and "sim/a" not in logs and "cos/b" not in logs
    total.backward()
    assert pooled.grad is not None and torch.isfinite(pooled.grad).all()


def test_defaults_keep_the_old_behaviour():
    cfg = LossConfig()
    assert cfg.relation_mode == "per_teacher" and cfg.gram_standardize is False
    assert FusionConfig().mode == "none" and TeacherConfig(name="a", hf_id="a", dim=4).center is False


# ---- (d) fusion ------------------------------------------------------------------------------------------------

def test_fusion_recovers_a_rotated_copy_and_averages_aligned_vectors():
    rng = np.random.default_rng(0)
    n, d, dim = 600, 24, 12
    base = rng.normal(size=(n, d)) @ np.diag(np.linspace(3, 0.3, d)) + 2.0
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    caches = {"ref": FakeCache(base), "rot": FakeCache(base @ q + 5.0 * rng.normal(size=(1, d)))}
    fused, report = fuse_teachers(caches, {"ref": 1.0, "rot": 1.0}, "ref", dim, fit_samples=400, seed=0)
    assert fused.shape == (n, dim) and fused.dtype == np.float16
    assert np.allclose(np.linalg.norm(fused.astype(np.float32), axis=1), 1.0, atol=1e-2)
    # the rotated teacher is an (almost exact) rotation of the reference after centring, so it aligns very well
    assert report["alignment_cosine"]["rot"] > 0.95
    assert 0.5 < report["kept_variance"]["ref"] <= 1.0


def test_fusion_config_validation_and_pipeline():
    teachers = [TeacherConfig(name="a", hf_id="a", dim=24), TeacherConfig(name="b", hf_id="b", dim=32)]
    ok = ExperimentConfig(name="x", output_dir="o", teachers=teachers, fusion=FusionConfig(mode="aligned_mean", reference="a", dim=8, fit_samples=50))
    validate(ok)
    for bad in (
        dataclasses.replace(ok, fusion=FusionConfig(mode="aligned_mean", reference="zzz", dim=8, fit_samples=50)),
        dataclasses.replace(ok, fusion=FusionConfig(mode="aligned_mean", reference="a", dim=40, fit_samples=50)),
        dataclasses.replace(ok, teachers=teachers[:1]),
        dataclasses.replace(ok, loss=LossConfig(relation_mode="nope")),
    ):
        with pytest.raises(ValueError):
            validate(bad)
    rng = np.random.default_rng(1)
    caches = {"a": FakeCache(rng.normal(size=(120, 24)) + 1.0), "b": FakeCache(rng.normal(size=(120, 32)) - 1.0)}
    cfg2, out = apply_target_methods(ok, caches)
    assert [t.name for t in cfg2.teachers] == ["fused"] and cfg2.teachers[0].dim == 8
    assert list(out) == ["fused"] and out["fused"].get(np.arange(5)).shape == (5, 8)


# ---- notebook METHOD string -----------------------------------------------------------------------------------

def test_method_overrides_map_to_config_values_and_a_stable_name_suffix(tmp_path):
    import yaml

    from turkish_distill.config import load_config
    from turkish_distill.targets import method_overrides

    assert method_overrides(None, ["a"]) == ([], "") and method_overrides("", ["a"]) == ([], "")
    ov, suffix = method_overrides("gram+center", ["a", "b"])
    assert suffix == "center_gram"  # fixed order whatever the writing order
    assert ov == ["teachers.0.center=true", "teachers.1.center=true", "loss.gram_standardize=true"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"name": "x", "output_dir": "o", "teachers": [
        {"name": "a", "hf_id": "a", "dim": 24}, {"name": "b", "hf_id": "b", "dim": 32}]}), encoding="utf-8")
    cfg = load_config(path, ov)
    assert all(t.center for t in cfg.teachers) and cfg.loss.gram_standardize and cfg.loss.sim_weight == 200.0
    ov, suffix = method_overrides("shared", ["a", "b"])
    assert suffix == "shared" and load_config(path, ov).loss.relation_mode == "shared"
    path.write_text(yaml.safe_dump({"name": "x", "output_dir": "o", "teachers": [
        {"name": "a", "hf_id": "a", "dim": 1024}, {"name": "b", "hf_id": "b", "dim": 768}]}), encoding="utf-8")
    ov, suffix = method_overrides("fusion", ["a", "b"])
    cfg = load_config(path, ov)
    assert suffix == "fusion" and cfg.fusion.mode == "aligned_mean" and cfg.fusion.reference == "a" and cfg.fusion.dim == 512


@pytest.mark.parametrize("bad, teachers", [("nope", ["a", "b"]), ("gram+shared", ["a", "b"]), ("fusion+center", ["a", "b"]),
                                           ("fusion", ["a"]), ("center+center", ["a"])])
def test_method_overrides_reject_incompatible_choices(bad, teachers):
    from turkish_distill.targets import method_overrides

    with pytest.raises(ValueError):
        method_overrides(bad, teachers)


def test_geometry_report_shows_the_cone_and_centring_opens_it():
    from turkish_distill.targets import geometry_report

    cache = FakeCache(cone_vectors(n=500, d=16, offset=6.0, seed=3))
    raw, new = geometry_report(cache, sample=300), geometry_report(cache, sample=300, centered=True)
    assert raw["mean_cosine"] > 0.8 and raw["std"] < 0.1 and raw["p5"] < raw["mean_cosine"] < raw["p95"]
    assert abs(new["mean_cosine"]) < 0.1 and new["std"] > 2 * raw["std"]
