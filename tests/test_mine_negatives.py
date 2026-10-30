"""Multi-miner hard-negative mining: candidates, consensus, veto, near-duplicates."""

import json

import numpy as np
import pytest

from turkish_distill.mine_negatives import mine_negatives


def unit(deg):
    a = np.deg2rad(np.asarray(deg, dtype=np.float32))
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


# pool order: 0 = positive, 1 = A, 2 = B, 3 = C, 4 = D; one question at angle 0
Q = unit([0.0] * 1)  # one question at angle 0, shape (1, 2)
MINER_1 = unit([20, 25, 40, 60, 170])
MINER_2 = unit([20, 22, 45, 150, 70])
POS = np.array([0])


def test_veto_drops_candidates_almost_as_close_as_the_positive_and_orders_by_hardness():
    # A is at 25 degrees against the positive's 20: 0.906 / 0.940 = 0.96 of the positive's score -> vetoed.
    out, stats = mine_negatives([Q], [MINER_1], POS, per_question=2, skip=0, top_k=4, mode="union")
    assert out[0] == [2, 3]  # B (hardest allowed) then C; A vetoed, D beyond per_question
    assert stats["vetoed"] == 1


def test_skip_removes_the_top_ranked_passages_from_the_candidates():
    out, _ = mine_negatives([Q], [MINER_1], POS, per_question=3, skip=2, top_k=4, mode="union")
    assert out[0] == [3, 4]  # ranks 0 and 1 (A, B) are skipped


def test_consensus_needs_every_miner_to_rank_the_candidate_high():
    # Rank window (top 3 non-positives): miner 1 {A, B, C}, miner 2 {A, B, D}; consensus {A, B}, A is vetoed.
    out, _ = mine_negatives([Q, Q], [MINER_1, MINER_2], POS, per_question=3, skip=0, top_k=3, mode="consensus")
    assert out[0] == [2]
    union, _ = mine_negatives([Q, Q], [MINER_1, MINER_2], POS, per_question=3, skip=0, top_k=3, mode="union")
    assert set(union[0]) == {2, 3, 4}  # A is vetoed by either miner; B, C, D remain, B hardest
    assert union[0][0] == 2


def test_one_miner_veto_is_enough():
    # Miner 2 finds A almost identical to the positive (22 vs 20 degrees): vetoed although miner 1 alone would
    # also veto it; make miner 1 lenient instead by moving A far away for miner 1.
    lenient = unit([20, 90, 40, 60, 170])  # A at 90 degrees for miner 1: clearly not a positive
    out, stats = mine_negatives([Q, Q], [lenient, MINER_2], POS, per_question=4, skip=0, top_k=4, mode="union")
    assert 1 not in out[0] and stats["vetoed"] >= 1


def test_near_duplicates_of_the_positive_are_dropped():
    # B (index 2) shares all of its 5 13-grams with the positive: a near-duplicate, not a negative.
    grams = [set(range(100)), set(range(1000, 1100)), set(range(5)), set(), set()]
    out, stats = mine_negatives([Q], [MINER_1], POS, per_question=3, skip=0, top_k=4, mode="union",
                                passage_grams=grams, near_duplicate=0.1)
    assert 2 not in out[0] and stats["near_duplicate"] == 1


def test_positive_is_never_returned_and_blocks_do_not_change_the_result():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(60, 8)).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    questions = pool[:20] + 0.3 * rng.normal(size=(20, 8)).astype(np.float32)
    questions /= np.linalg.norm(questions, axis=1, keepdims=True)
    positive = np.arange(20)
    a, _ = mine_negatives([questions], [pool], positive, per_question=2, skip=1, top_k=10, mode="union", block=4)
    b, _ = mine_negatives([questions], [pool], positive, per_question=2, skip=1, top_k=10, mode="union", block=64)
    assert a == b and all(i not in n for i, n in enumerate(a))


def test_argument_checks():
    with pytest.raises(ValueError):
        mine_negatives([Q], [MINER_1], POS, mode="intersection")
    with pytest.raises(ValueError):
        mine_negatives([Q], [MINER_1], POS, skip=5, top_k=5)


def test_cli_writes_negatives_and_stats_and_reuses_the_embedding_cache(tmp_path, monkeypatch):
    from turkish_distill import mine_negatives as mn

    calls = []

    def fake_encode(miner, texts, prefix):
        calls.append(miner)
        rng = np.random.default_rng(abs(hash(miner)) % 1000)
        vecs = np.stack([np.random.default_rng(sum(map(ord, t)) + int(rng.integers(3))).normal(size=16) for t in texts])
        return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)

    monkeypatch.setattr(mn, "_encode", fake_encode)
    rows = []
    for i in range(40):
        rows.append({"source": "x", "question": f"soru {i} nedir", "passage": f"pasaj numarasi {i} " + "kelime " * 20 + str(i),
                     "passage_id": f"id{i}"})
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "hn.jsonl"
    argv = ["--pairs", str(pairs), "--out", str(out), "--miners", "bge_m3", "--mode", "union", "--skip", "1",
            "--top-k", "8", "--per-question", "2", "--emb-dir", str(tmp_path / "emb")]
    mn.main(argv)
    written = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 40 and all(set(r["neg_passage_ids"]) <= {f"id{i}" for i in range(40)} for r in written)
    assert all(r["passage_id"] not in r["neg_passage_ids"] and len(r["neg_passage_ids"]) <= 2 for r in written)
    stats = json.loads(out.with_suffix(".stats.json").read_text(encoding="utf-8"))
    assert stats["pairs"] == 40 and stats["miners"] == ["bge_m3"]
    n_calls = len(calls)
    mn.main(argv)  # second run: embeddings come from the cache
    assert len(calls) == n_calls
