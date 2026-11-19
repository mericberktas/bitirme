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
