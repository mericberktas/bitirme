"""Answers in the QA pairs: extraction (prepare_pairs), lookup for existing pairs (attach_answers) and the
answer-weighted contrastive term (fixed / word-count / character-count weights)."""

import json

import numpy as np
import pytest
import torch

from conftest import CharTokenizer
from test_contrastive import ANCHOR, make_pairs, smoke_cfg, tiny_model, write_cfg
from test_prepare_pairs import rag_row, write_turhistquad, write_wikirag
from turkish_distill import attach_answers as aa
from turkish_distill import contrastive as ct
from turkish_distill import prepare_pairs as pp
from turkish_distill.contrastive_config import ContrastiveLossConfig, load_contrastive_config
from turkish_distill.logging_utils import read_metrics
from turkish_distill.train_contrastive import run_contrastive


def loss_cfg(**kw) -> ContrastiveLossConfig:
    return ContrastiveLossConfig(name="mnrl", scale=20.0, **kw)


def answered_pairs(n: int, offset: int = 0, missing=(), duplicate=()) -> dict:
    pairs = make_pairs(n, offset)
    answers = [f"cevap {i + offset} " + "kelime " * (i % 5) for i in range(n)]
    for i in missing:
        answers[i] = None
    for i in duplicate:
        answers[i] = answers[duplicate[0]]
    pairs["answer"] = answers
    return pairs


# --- weights -------------------------------------------------------------------- #

def test_fixed_weight_ignores_the_answer_length():
    cfg = loss_cfg(answer_weight_mode="fixed", answer_weight=0.3)
    assert ct.answer_weight("1923", cfg) == ct.answer_weight("uzun " * 50, cfg) == pytest.approx(0.3)


def test_word_and_char_ramps_go_from_min_to_max_and_clip():
    words = loss_cfg(answer_weight_mode="words", answer_weight_min=0.1, answer_weight_max=0.5, answer_ref_words=20)
    assert ct.answer_weight("bir", words) == pytest.approx(0.1 + 0.4 / 20)
    assert ct.answer_weight(" ".join(["k"] * 10), words) == pytest.approx(0.3)
    assert ct.answer_weight(" ".join(["k"] * 80), words) == pytest.approx(0.5)  # clipped at the reference length
    chars = loss_cfg(answer_weight_mode="chars", answer_ref_chars=100)
    assert ct.answer_weight("x" * 50, chars) == pytest.approx(0.3)
    inverse = loss_cfg(answer_weight_mode="chars", answer_ref_chars=100, answer_weight_inverse=True)
    assert ct.answer_weight("x" * 100, inverse) == pytest.approx(0.1)  # long answers get the low end
    assert ct.make_answer_weight_fn(loss_cfg()) is None and ct.make_answer_loss(loss_cfg()) is None


def test_answer_config_validation(tmp_path):
    path = write_cfg(tmp_path)
    cfg = load_contrastive_config(path)
    assert cfg.loss.answer_weight_mode == "none" and cfg.data.answer_max_length == 128  # off by default
    for bad in (["loss.answer_weight_mode=length"], ["loss.answer_weight=-0.1"],
                ["loss.answer_weight_min=0.6", "loss.answer_weight_max=0.5"], ["loss.answer_ref_words=0"]):
        with pytest.raises(ValueError):
            load_contrastive_config(path, bad)
    assert load_contrastive_config(path, ["loss.answer_weight_mode=words"]).fingerprint() != cfg.fingerprint()


# --- loss ----------------------------------------------------------------------- #

def test_answer_info_nce_matches_a_hand_computed_weighted_case():
    q = torch.eye(2)
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # both answers point like question 0
    s, w = 2.0, torch.tensor([0.2, 0.4])
    loss, logs = ct.answer_info_nce(q, a, w, scale=s)
    e = np.exp
    ce0 = -np.log(e(s) / (e(s) + e(s)))  # question 0: its own answer and the other one score the same
    ce1 = -np.log(e(0) / (e(0) + e(0)))
    assert loss.item() == pytest.approx((0.2 * ce0 + 0.4 * ce1) / 2, rel=1e-5)
    assert logs["ans_ce"].item() == pytest.approx((ce0 + ce1) / 2, rel=1e-5)
    # the second answer is the same text as the first: masked, both rows are now trivially right
    masked, _ = ct.answer_info_nce(q, a, w, scale=s, mask=torch.tensor([[False, True], [True, False]]))
    assert masked.item() == pytest.approx(0.0, abs=1e-6)


# --- data ----------------------------------------------------------------------- #

def test_collator_carries_only_the_answered_pairs_with_weights_and_a_duplicate_mask():
    pairs = answered_pairs(6, missing=(2,), duplicate=(0, 4))
    fn = ct.make_answer_weight_fn(loss_cfg(answer_weight_mode="fixed", answer_weight=0.25))
    batch = ct.PairCollator(CharTokenizer(), pairs, 32, 64, answer_weight_fn=fn, answer_max_length=16)([0, 1, 2, 3, 4])
    assert batch["a_rows"].tolist() == [0, 1, 3, 4]
    assert batch["a_input_ids"].shape[0] == 4 and batch["a_input_ids"].shape[1] <= 16
    torch.testing.assert_close(batch["answer_weight"], torch.full((4,), 0.25))
    mask = batch["answer_mask"]
    assert bool(mask[0, 3]) and bool(mask[3, 0]) and int(mask.sum()) == 2 and not mask.diagonal().any()
    none = ct.PairCollator(CharTokenizer(), pairs, 32, 64, answer_weight_fn=fn)([2])
    assert "a_input_ids" not in none  # no answer in the batch: no answer keys
    plain = ct.PairCollator(CharTokenizer(), pairs, 32, 64)([0, 1])
    assert "a_input_ids" not in plain  # the term is off: answers are ignored
    with pytest.raises(ValueError, match="answer"):
        ct.PairCollator(CharTokenizer(), make_pairs(4), 32, 64, answer_weight_fn=fn)


def test_load_pairs_reads_answers_and_keeps_them_aligned_when_subsampling(tmp_path):
    pairs = answered_pairs(10, missing=(3,))
    path = tmp_path / "ans.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(10):
            row = {"question": pairs["question"][i], "passage": pairs["passage"][i], "passage_id": pairs["passage_id"][i]}
            if pairs["answer"][i]:
                row["answer"] = pairs["answer"][i]
            f.write(json.dumps(row) + "\n")
    full = ct.load_pairs(path)
    assert full["answer"] == pairs["answer"]
    sub = ct.load_pairs(path, max_pairs=5, seed=2)
    index = {q: i for i, q in enumerate(pairs["question"])}
    assert [pairs["answer"][index[q]] for q in sub["question"]] == sub["answer"]


# --- training step -------------------------------------------------------------- #

def answer_batch(n=10, hard_negatives=0, mode="words"):
    pairs = answered_pairs(n, missing=(1,), duplicate=(0, 5))
    if hard_negatives:
        ids = pairs["passage_id"]
        pairs["neg_ids"] = [[ids[(i + 1) % n]] for i in range(n)]
        pairs["pool"] = dict(zip(ids, pairs["passage"]))
    cfg = loss_cfg(answer_weight_mode=mode, answer_ref_words=4)
    col = ct.PairCollator(CharTokenizer(), pairs, 32, 64, hard_negatives, ct.make_answer_weight_fn(cfg), 24)
    return col(list(range(n))), cfg


@pytest.mark.parametrize("chunk", [1, 3, 4])
def test_cached_step_with_answers_matches_the_plain_step(chunk):
    batch, cfg = answer_batch()
    fn = ct.make_answer_loss(cfg)
    plain, cached = tiny_model(3), tiny_model(3)
    a = ct.contrastive_step(plain, batch, ct.make_loss(cfg), None, answer_fn=fn)
    b = ct.contrastive_step(cached, batch, ct.make_loss(cfg), chunk, answer_fn=fn)
    assert b["loss"] == pytest.approx(a["loss"], rel=1e-5) and a["ans_pairs"] == 9
    assert a["loss"] == pytest.approx(a["nce_loss"] + a["ans_loss"], rel=1e-5)
    for (name, x), (_, y) in zip(plain.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("chunk", [3, 5])
def test_cached_step_with_answers_hard_negatives_and_anchor_matches_the_plain_step(chunk):
    batch, cfg = answer_batch(hard_negatives=1)
    anchor = ct.make_anchor(tiny_model(9))
    kw = dict(anchor_model=anchor, anchor_fn=ct.make_anchor_loss(ANCHOR), answer_fn=ct.make_answer_loss(cfg))
    plain, cached = tiny_model(4), tiny_model(4)
    a = ct.contrastive_step(plain, batch, ct.make_loss(cfg), None, **kw)
    b = ct.contrastive_step(cached, batch, ct.make_loss(cfg), chunk, **kw)
    assert b["loss"] == pytest.approx(a["loss"], rel=1e-4)
    for (name, x), (_, y) in zip(plain.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_zero_answer_weight_gives_the_same_gradients_as_no_answer_term():
    batch, _ = answer_batch(mode="fixed")
    batch["answer_weight"] = torch.zeros_like(batch["answer_weight"])
    cfg = loss_cfg(answer_weight_mode="fixed", answer_weight=0.0)
    with_term, without = tiny_model(6), tiny_model(6)
    ct.contrastive_step(with_term, batch, ct.make_loss(cfg), 4, answer_fn=ct.make_answer_loss(cfg))
    ct.contrastive_step(without, batch, ct.make_loss(cfg), 4)
    for (name, x), (_, y) in zip(without.named_parameters(), with_term.named_parameters()):
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-7, msg=name)


def test_smoke_training_with_answers_logs_the_term_and_it_decreases(tmp_path):
    cfg = smoke_cfg(tmp_path, max_steps=16)
    cfg.loss.answer_weight_mode = "chars"
    cfg.loss.anchor_cosine_weight, cfg.loss.anchor_gram_weight = 10.0, 200.0
    run_dir = tmp_path / "answers"
    run_contrastive(cfg, run_dir, tiny_model(), CharTokenizer(), answered_pairs(64, missing=(5, 9)),
                    make_pairs(24, 100), torch.device("cpu"))
    train = [r for r in read_metrics(run_dir / "metrics.jsonl") if "train/loss" in r]
    assert all("train/ans_loss" in r and "train/anchor_cos" in r and "train/nce_loss" in r for r in train)
    assert 0.1 <= train[0]["train/ans_weight"] <= 0.5
    assert train[-1]["train/ans_ce"] < train[0]["train/ans_ce"]


def test_turning_on_answers_without_an_answer_field_fails_early(tmp_path):
    cfg = smoke_cfg(tmp_path)
    cfg.loss.answer_weight_mode = "fixed"
    with pytest.raises(ValueError, match="answer"):
        run_contrastive(cfg, tmp_path / "x", tiny_model(), CharTokenizer(), make_pairs(32), make_pairs(8, 100),
                        torch.device("cpu"))


# --- prepare_pairs and attach_answers ------------------------------------------- #

def thq_row(row_id, question, context, answer):
    return {"id": row_id, "question": question, "context": context, "answers": {"answer_start": 0, "text": answer}}


def write_raw(tmp_path):
    """Raw files in the layout `attach_answers.raw_paths` expects; returns that layout's root."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    raw = tmp_path / "raw"
    write_turhistquad(raw / "turhistquad",
                      [thq_row("1", "Normanlar kimdir?", "Normanlar pasaji", "Rollo"),
                       thq_row("2", "Ikinci soru nedir?", "Ikinci pasaj", "Bir  cevap")],
                      [thq_row("3", "Test sorusu mu?", "Test pasaji", "test")])
    arts = ["Birinci makale metni.", "Ikinci makale metni burada."]
    rows = [rag_row(i, f"Rag soru {i}?", arts, i % 2) for i in range(3)]
    for i, row in enumerate(rows):
        row["answer"] = f"Rag cevap {i} uzun bir cumle."
    (raw / "wikirag" / "data").mkdir(parents=True)
    write_wikirag(raw / "wikirag" / "data" / "train.csv", rows)
    qas = [{"questionText": f"Makale sorusu {k} nedir?", "answerText": f"Uretilmis cevap {k}."} for k in range(3)]
    (raw / "wikiqa" / "data").mkdir(parents=True)
    pq.write_table(pa.table({"original_text": ["Makale metni " + "bilgi " * 30], "answers": [str(qas)]}),
                   raw / "wikiqa" / "data" / "turkish-wiki-qa-4mil.parquet")
    return raw


def test_loaders_keep_the_answers():
    pair = pp.make_pair("s", " Soru? ", "Pasaj", "  iki   bosluklu  cevap ")
    assert pair["answer"] == "iki bosluklu cevap"
    assert "answer" not in pp.make_pair("s", "Soru?", "Pasaj", "  ")
    assert pp.answer_text({"text": ["ilk", "ikinci"]}) == "ilk" and pp.answer_text([]) is None


def test_attach_answers_restores_answers_on_answerless_pairs_and_decontaminates_them(tmp_path):
    raw = write_raw(tmp_path)
    paths = aa.raw_paths(raw)
    train, _ = pp.load_turhistquad(paths["turhistquad"], {})
    wiki = list(aa.wikiqa_answers(paths["wikiqa"], {pp.passage_id(pp.normalize("Makale metni " + "bilgi " * 30))}))
    with_answers = train + pp.load_wikirag(paths["wikirag"], {}) + wiki
    assert len(with_answers) == 2 + 3 + 3 and all("answer" in p for p in with_answers)
    stripped = [{k: v for k, v in p.items() if k != "answer"} for p in with_answers]
    stripped.append(pp.make_pair("turhistquad", "Ham veride olmayan soru?", "Yok pasaj"))

    pairs_path = tmp_path / "pairs.jsonl"
    pp._write_jsonl(pairs_path, stripped)
    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text(json.dumps({"text": "rollo"}) + "\n", encoding="utf-8")  # a benchmark text equal to one answer
    out = tmp_path / "out" / "pairs_ans.jsonl"
    aa.main(["--pairs", str(pairs_path), "--out", str(out), "--raw-dir", str(raw), "--forbidden", str(forbidden)])

    rows = aa.read_jsonl(out)
    assert [(r["question"], r["passage_id"]) for r in rows] == [(p["question"], p["passage_id"]) for p in stripped]
    expected = {(p["question"], p["passage_id"]): p["answer"] for p in with_answers}
    expected[("Normanlar kimdir?", pp.passage_id("Normanlar pasaji"))] = None  # removed: equals a forbidden text
    assert {(r["question"], r["passage_id"]): r.get("answer") for r in rows[:-1]} == expected
    assert "answer" not in rows[-1]  # not in the raw data
    stats = json.loads(out.with_name("pairs_ans.stats.json").read_text(encoding="utf-8"))
    assert stats["answers_removed_decontamination"] == 1 and stats["with_answer"] == 7
    assert stats["by_source"]["turhistquad"] == {"pairs": 3, "matched": 2}
    assert stats["answer_length"]["wikirag"]["n"] == 3


def test_decontaminate_drops_a_leaked_answer_but_keeps_its_pair():
    leaked = " ".join(f"kelime{i}" for i in range(20))
    pairs = [pp.make_pair("s", "Temiz soru mu?", "Temiz pasaj", "x " + leaked),
             pp.make_pair("s", "Ikinci soru mu?", "Ikinci pasaj", "temiz cevap")]
    stats = {}
    kept, dropped = pp.decontaminate(pairs, [leaked], stats=stats)
    assert dropped == 0 and len(kept) == 2 and "answer" not in kept[0] and kept[1]["answer"] == "temiz cevap"
    assert stats["answers_removed"] == 1
