"""Neighbour-rich batches: teacher-space clusters and an epoch sampler that builds batches from cluster groups."""

import numpy as np
import pytest

from turkish_distill.config import ExperimentConfig, TeacherConfig, validate
from turkish_distill.data import epoch_batches, neighbor_epoch_batches
from turkish_distill.targets import ArrayCache, cached_neighbor_clusters, neighbor_clusters


def blob_cache(n_blobs=6, per_blob=60, d=16, noise=0.15, seed=0):
    """Unit vectors in `n_blobs` tight groups (the true neighbourhoods) and the group of every text."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_blobs, d))
    truth = np.repeat(np.arange(n_blobs), per_blob)
    x = centres[truth] + rng.normal(size=(len(truth), d)) * noise
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return ArrayCache(x.astype(np.float16)), truth


def purity(labels, truth):
    return sum(np.bincount(truth[labels == c]).max() for c in np.unique(labels)) / len(truth)


def test_neighbor_clusters_recover_tight_groups_and_are_deterministic():
    cache, truth = blob_cache()
    a = neighbor_clusters({"t": cache}, n_clusters=6, iters=10, seed=1, proj_dim=8)
    b = neighbor_clusters({"t": cache}, n_clusters=6, iters=10, seed=1, proj_dim=8)
    assert a.shape == (len(truth),) and purity(a, truth) > 0.9
    np.testing.assert_array_equal(a, b)


def test_neighbor_clusters_use_every_teacher():
    cache_a, truth = blob_cache(seed=0)
    cache_b, _ = blob_cache(seed=1)
    labels = neighbor_clusters({"a": cache_a, "b": cache_b}, n_clusters=6, iters=8, seed=0, proj_dim=8)
    assert labels.shape == (len(truth),) and len(np.unique(labels)) > 1


def test_cluster_labels_are_cached_and_reloaded(tmp_path):
    cache, truth = blob_cache()
    first = cached_neighbor_clusters({"t": cache}, str(tmp_path), "fp", 6, 5)
    files = list((tmp_path / "neighbor_clusters").glob("*.npy"))
    assert len(files) == 1
    second = cached_neighbor_clusters({"t": cache}, str(tmp_path), "fp", 6, 5)
    np.testing.assert_array_equal(first, second)


def batches_for(labels, fraction, group=8, batch=32, seed=3, epoch=0, start=0):
    return neighbor_epoch_batches(np.arange(len(labels)), labels, batch, seed, epoch, fraction, group, start)


def test_neighbor_batches_have_the_right_size_and_never_repeat_a_text_in_an_epoch():
    labels = np.repeat(np.arange(10), 60)  # 600 texts
    batches = batches_for(labels, 0.5)
    assert len(batches) == 600 // 32 and all(len(b) == 32 for b in batches)
    flat = [i for b in batches for i in b]
    assert len(set(flat)) == len(flat)


def test_neighbor_batches_put_groups_from_one_cluster_into_a_batch():
    labels = np.repeat(np.arange(10), 60)
    group, batch = 8, 32
    batches = batches_for(labels, 0.5, group, batch)
    for b in batches:
        counts = np.bincount(labels[b], minlength=10)
        assert counts[counts >= group].sum() >= 16   # 0.5 * 32 / 8 = 2 groups of 8, each from one cluster
    random_counts = [np.bincount(labels[b], minlength=10).max() for b in epoch_batches(np.arange(600), 32, 3, 0)]
    neighbour_counts = [np.bincount(labels[b], minlength=10).max() for b in batches]
    assert np.mean(neighbour_counts) > np.mean(random_counts)


def test_neighbor_batches_are_deterministic_and_resumable():
    labels = np.repeat(np.arange(8), 70)
    full = batches_for(labels, 0.5)
    assert full == batches_for(labels, 0.5)
    assert batches_for(labels, 0.5, start=5) == full[5:]
    assert full != batches_for(labels, 0.5, epoch=1)


def test_a_fraction_of_zero_groups_nothing_and_small_clusters_still_fill_the_batches():
    labels = np.arange(300) % 150          # clusters of only 2 texts: no full group of 8 exists
    batches = batches_for(labels, 0.5)
    assert len(batches) == 300 // 32 and all(len(b) == 32 for b in batches)
    assert len({i for b in batches for i in b}) == len(batches) * 32
    batches0 = batches_for(np.repeat(np.arange(10), 60), 0.0)
    assert all(len(b) == 32 for b in batches0)


def test_config_validation_of_the_neighbour_batch_settings():
    import dataclasses

    cfg = ExperimentConfig(name="x", output_dir="o", teachers=[TeacherConfig(name="a", hf_id="a", dim=4)])
    validate(dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, neighbor_fraction=0.5)))
    for bad in (dict(neighbor_fraction=1.5), dict(neighbor_fraction=0.5, neighbor_group_size=1),
                dict(neighbor_fraction=0.5, neighbor_clusters=1)):
        with pytest.raises(ValueError):
            validate(dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **bad)))
