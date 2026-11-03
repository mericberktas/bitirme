import numpy as np
import pytest

from turkish_distill.posthoc import apply_map, fit_stats, linear_map, mean_pairwise_cosine


def anisotropic(n=4000, d=16, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.array([5.0, 3.0] + [1.0] * (d - 2))
    return rng.normal(size=(n, d)) * scales + 4.0 * rng.normal(size=d) + 10.0     # big common offset


def test_center_removes_mean_and_nothing_else():
    x = anisotropic()
    w, b = linear_map("center", fit_stats(x))
    y = apply_map(x, w, b)
    assert np.allclose(y.mean(0), 0, atol=1e-9)
    assert np.allclose(y - y.mean(0), x - x.mean(0))


def test_identity_is_identity():
    x = anisotropic()
    w, b = linear_map("identity", fit_stats(x))
    assert np.allclose(apply_map(x, w, b), x)


def test_abtt_removes_top_directions():
    x = anisotropic()
    st = fit_stats(x)
    y = apply_map(x, *linear_map("abtt", st, k=2))
    top = st["eigvecs"][:, :2]
    assert np.allclose(y @ top, 0, atol=1e-8)                  # nothing left along the two strongest directions
    assert np.allclose(y.mean(0), 0, atol=1e-8)
    rest = st["eigvecs"][:, 2:]
    assert np.allclose(y @ rest, (x - x.mean(0)) @ rest)       # the other directions are untouched


def test_zca_whitens_with_shrinkage_toward_identity_covariance():
    x = anisotropic()
    st = fit_stats(x)
    y = apply_map(x, *linear_map("zca", st, eps=1e-6))
    cov = np.cov(y.T)
    assert np.allclose(cov, np.eye(16), atol=1e-3)
    y_shrunk = apply_map(x, *linear_map("zca", st, eps=10.0))   # heavy shrinkage: close to a plain rescaling
    ev = np.linalg.eigvalsh(np.cov(y_shrunk.T))
    assert ev.max() / ev.min() < np.linalg.eigvalsh(np.cov(x.T)).max() / np.linalg.eigvalsh(np.cov(x.T)).min()


def test_anisotropy_measure_drops_after_centering():
    x = anisotropic()
    before = mean_pairwise_cosine(x)
    after = mean_pairwise_cosine(apply_map(x, *linear_map("center", fit_stats(x))))
    assert before > 0.8 and abs(after) < 0.05


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        linear_map("nope", fit_stats(anisotropic()))


def test_dense_head_matches_the_numpy_map(tmp_path):
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("sentence_transformers").models
    x = anisotropic(n=200)
    w, b = linear_map("abtt", fit_stats(x), k=3)
    dense = models.Dense(16, 16, bias=True, activation_function=torch.nn.Identity())
    with torch.no_grad():
        dense.linear.weight.copy_(torch.tensor(w, dtype=torch.float32))
        dense.linear.bias.copy_(torch.tensor(b, dtype=torch.float32))
    dense.save(str(tmp_path))
    loaded = models.Dense.load(str(tmp_path))                  # the saved-and-reloaded layer is what evaluation sees
    out = loaded({"sentence_embedding": torch.tensor(x, dtype=torch.float32)})["sentence_embedding"].detach().numpy()
    assert np.allclose(out, apply_map(x, w, b), atol=1e-3)
