"""Teacher comparison report (per-teacher spread and dimensionality, pairwise Gram agreement and neighbour overlap)."""

import numpy as np
import pytest

from turkish_distill.teacher_compare import format_report, geometry_report


def test_identical_and_rotated_teachers_agree():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    rep = geometry_report({"a": a, "rot": a @ q, "noise": rng.normal(size=(200, 32))}, k=5, student_dim=8)
    assert rep["pairs"]["a~rot"]["gram_pearson"] == pytest.approx(1.0, abs=1e-5)  # rotation keeps the geometry
    assert rep["pairs"]["a~rot"]["top5_overlap"] == pytest.approx(1.0)
    assert abs(rep["pairs"]["a~noise"]["gram_pearson"]) < 0.2
    t = rep["teachers"]["a"]
    assert t["dim"] == 16 and 1 <= t["pcs_90"] <= t["pcs_99"] <= 16 and 0 < t["var_first_8"] < 1
    assert "a~noise" in format_report(rep)


def test_low_rank_teacher_needs_few_components():
    rng = np.random.default_rng(1)
    low = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 64))
    rep = geometry_report({"low": low, "full": rng.normal(size=(300, 64))}, k=5, student_dim=3)
    assert rep["teachers"]["low"]["pcs_99"] <= 3 and rep["teachers"]["low"]["var_first_3"] == pytest.approx(1.0)
    assert rep["teachers"]["full"]["pcs_90"] > 30
    with pytest.raises(ValueError, match="different numbers of rows"):
        geometry_report({"x": low, "y": low[:10]})


def test_rank_agreement_spearman_and_groups():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(120, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    groups = ["x"] * 60 + ["y"] * 50 + ["tiny"] * 10
    rep = geometry_report({"a": a, "rot": a @ q, "neg": -a[:, ::-1] @ q + 0 * a, "noise": rng.normal(size=(120, 16))},
                          k=5, groups=groups, num_quads=20000, margin=0.01, spearman_pairs=5000, min_group=20)
    same = rep["pairs"]["a~rot"]
    assert same["gram_spearman"] == pytest.approx(1.0, abs=1e-6)
    assert same["rank_agreement"] == pytest.approx(1.0) and same["conflict_rate"] == 0.0
    assert same["rank_agreement_decisive"] == pytest.approx(1.0)
    noise = rep["pairs"]["a~noise"]
    assert 0.4 < noise["rank_agreement"] < 0.6 and noise["conflict_rate"] > 0.2
    assert noise["conflict_rate"] <= noise["decisive_share"]
    assert set(rep["groups"]) == {"x", "y"}  # "tiny" is below min_group
    assert same["by_group"]["x"]["gram_pearson_within"] == pytest.approx(1.0, abs=1e-5)
    assert same["by_group"]["y"]["top5_overlap"] == pytest.approx(1.0)
    text = format_report(rep)
    assert "per group" in text and "conflict_rate" in text
    with pytest.raises(ValueError, match="labels for"):
        geometry_report({"a": a, "b": a}, groups=["x"] * 5)


def test_rank_agreement_reversed_order_conflicts():
    from turkish_distill.teacher_compare import rank_agreement

    rng = np.random.default_rng(3)
    v = rng.normal(size=(80, 8))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    g = v @ v.T
    rep = rank_agreement(g, -g, 5000, 0.01, np.random.default_rng(0))
    # only the rare draws of the same pair twice (difference 0 in both) count as agreeing
    assert rep["rank_agreement"] < 0.005 and rep["conflict_rate"] == pytest.approx(rep["decisive_share"])
