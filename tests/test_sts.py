import numpy as np
import pytest

from turkish_distill.eval.sts import evaluate_sts, pearson, spearman


def test_rank_correlations():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(a, 2 * a + 1) == pytest.approx(1.0)
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    tied = np.array([1.0, 1.0, 2.0, 3.0])
    assert -1.0 <= spearman(tied, a) <= 1.0


def test_evaluate_sts_with_controlled_embeddings():
    # Embeddings whose cosine equals the gold ordering.
    angles = np.array([0.0, 0.3, 0.6, 0.9, 1.2])
    s1 = [str(i) for i in range(5)]
    s2 = [str(i) for i in range(5)]
    base = {t: np.array([1.0, 0.0]) for t in s1}

    def encode(texts):
        # first call: s1 -> fixed vector; second call: s2 -> rotated by angle
        encode.calls += 1
        if encode.calls == 1:
            return np.stack([base[t] for t in texts])
        return np.stack([[np.cos(angles[int(t)]), np.sin(angles[int(t)])] for t in texts])

    encode.calls = 0
    gold = [5.0, 4.0, 3.0, 2.0, 1.0]  # larger angle -> lower similarity
    out = evaluate_sts(encode, s1, s2, gold)
    assert out["sts_spearman"] == pytest.approx(1.0)
