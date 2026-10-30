"""Hard-negative mining with several miners (veto + consensus), for the contrastive fine-tune.

Each question of a pairs file gets up to `per_question` mined negatives, chosen from the pool of the file's
distinct passages. Several embedding models ("miners") vote:

  * candidates: passages ranked between `skip` and `top_k` (excluding the positive) by the miners; with
    mode "consensus" a candidate has to be in that rank window for EVERY miner (hard for all of them), with
    "union" for any of them;
  * veto (false-negative guard): a candidate is dropped when ANY miner scores it at least `pos_frac` times the
    positive's score (positive-aware filtering: a passage that a miner finds almost as good as the positive may
    be a false negative);
  * near-duplicates of the positive (shared 13-grams) are dropped;
  * the remaining candidates are ranked by their mean score relative to the positive, hardest first.

With one miner the veto and the consensus reduce to ordinary positive-aware mining, which is the ablation.

    python -m turkish_distill.mine_negatives --pairs P.jsonl --out P_hn.jsonl --miners bge_m3 modernbert_tr_embed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import TeacherConfig

QUERY_PREFIX_8K = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

# name -> (teacher definition as in the stage-1 configs, prefix for the QUESTIONS only; passages are encoded plain)
MINERS: dict[str, tuple[TeacherConfig, str]] = {
    "bge_m3": (
        TeacherConfig(name="bge_m3", hf_id="BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181",
                      dim=1024, max_length=512, batch_size=64, dtype="float16"),
        "",
    ),
    "modernbert_tr_embed": (
        TeacherConfig(name="modernbert_tr_embed", hf_id="ytu-ce-cosmos/modernbert-tr-embed",
                      revision="855820e0079102805908d60d32cd9fd59c970b80", dim=768, max_length=512, batch_size=64,
                      dtype="float32"),
        QUERY_PREFIX_8K,
    ),
}


def _grams(text: str, n: int = 13) -> set[int]:
    from .prepare_pairs import _ngram_hashes

    return _ngram_hashes(text, n)


def mine_negatives(
    q_emb: list[np.ndarray],
    p_emb: list[np.ndarray],
    positive: np.ndarray,
    *,
    per_question: int = 3,
    skip: int = 3,
    top_k: int = 30,
    pos_frac: float = 0.95,
    mode: str = "consensus",
    passage_grams: Optional[list[set[int]]] = None,
    near_duplicate: float = 0.1,
    block: int = 256,
) -> tuple[list[list[int]], dict[str, float]]:
    """Mined negatives (pool indices, hardest first) per question and counters of what was filtered.

    q_emb[m] is (n_questions, d_m) and p_emb[m] (n_pool, d_m), both L2-normalised, one entry per miner;
    positive[i] is the pool index of question i's positive passage.
    """
    if mode not in ("consensus", "union"):
        raise ValueError("mode must be 'consensus' or 'union'")
    if not 0 <= skip < top_k:
        raise ValueError("need 0 <= skip < top_k")
    n_miners, n_q = len(q_emb), len(positive)
    pools = [torch.as_tensor(p, dtype=torch.float32) for p in p_emb]
    pos_t = torch.as_tensor(positive, dtype=torch.long)
    out: list[list[int]] = [[] for _ in range(n_q)]
    stats = {"candidates": 0, "vetoed": 0, "near_duplicate": 0, "no_consensus": 0}
    for start in range(0, n_q, block):
        rows = torch.arange(start, min(start + block, n_q))
        b = len(rows)
        scores, pos_scores, windows = [], [], []
        for m in range(n_miners):
            s = torch.as_tensor(q_emb[m][start:start + b], dtype=torch.float32) @ pools[m].T
            ps = s[torch.arange(b), pos_t[rows]].clone()
            s[torch.arange(b), pos_t[rows]] = float("-inf")  # the positive is never a negative
            order = torch.topk(s, top_k, dim=1).indices  # (b, top_k) best first
            scores.append(s)
            pos_scores.append(ps)
            windows.append(order[:, skip:])
        for r in range(b):
            sets = [set(w[r].tolist()) for w in windows]
            cands = set.intersection(*sets) if mode == "consensus" else set.union(*sets)
            if mode == "consensus" and not cands:
                stats["no_consensus"] += 1
            stats["candidates"] += len(cands)
            pos_i = int(pos_t[rows[r]])
            kept = []
            for c in cands:
                if any(scores[m][r, c] >= pos_frac * pos_scores[m][r] for m in range(n_miners)):
                    stats["vetoed"] += 1
                    continue
                if passage_grams is not None and passage_grams[pos_i] and passage_grams[c]:
                    shared = len(passage_grams[pos_i] & passage_grams[c])
                    if shared / min(len(passage_grams[pos_i]), len(passage_grams[c])) >= near_duplicate:
                        stats["near_duplicate"] += 1
                        continue
                hardness = float(np.mean([float(scores[m][r, c]) / max(float(pos_scores[m][r]), 1e-6) for m in range(n_miners)]))
                kept.append((hardness, c))
            kept.sort(key=lambda x: (-x[0], x[1]))  # hardest first, ties by index: deterministic
            out[start + r] = [c for _, c in kept[:per_question]]
    return out, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _encode(miner: str, texts: list[str], prefix: str) -> np.ndarray:
    from .teachers import _normalize, make_sentence_transformer_encoder

    teacher, _ = MINERS[miner]
    encode, close = make_sentence_transformer_encoder(teacher)
    try:
        return _normalize(encode([prefix + t for t in texts]))
    finally:
        close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for a (question, passage) pairs file")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--miners", nargs="+", default=["bge_m3", "modernbert_tr_embed"], choices=sorted(MINERS))
    parser.add_argument("--mode", default="consensus", choices=["consensus", "union"])
    parser.add_argument("--per-question", type=int, default=3)
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--pos-frac", type=float, default=0.95)
    parser.add_argument("--emb-dir", default=None, help="cache of the embeddings (re-used if present)")
    args = parser.parse_args(argv)

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    pool_ids: list[str] = []
    pool_text: list[str] = []
    index: dict[str, int] = {}
    for r in rows:
        if r["passage_id"] not in index:
            index[r["passage_id"]] = len(pool_ids)
            pool_ids.append(r["passage_id"])
            pool_text.append(r["passage"])
    positive = np.array([index[r["passage_id"]] for r in rows])
    questions = [r["question"] for r in rows]
    print(f"{len(rows)} pairs, {len(pool_ids)} distinct passages", flush=True)

    emb_dir = Path(args.emb_dir) if args.emb_dir else None
    q_emb, p_emb = [], []
    for miner in args.miners:
        cached = emb_dir / f"{miner}.npz" if emb_dir else None
        if cached is not None and cached.exists():
            z = np.load(cached)
            q, p = z["q"], z["p"]
            if q.shape[0] == len(questions) and p.shape[0] == len(pool_text):
                print(f"{miner}: embeddings from {cached}", flush=True)
                q_emb.append(q)
                p_emb.append(p)
                continue
        print(f"{miner}: encoding", flush=True)
        p = _encode(miner, pool_text, "")
        q = _encode(miner, questions, MINERS[miner][1])
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cached, q=q, p=p)
        q_emb.append(q)
        p_emb.append(p)

    negatives, stats = mine_negatives(
        q_emb, p_emb, positive, per_question=args.per_question, skip=args.skip, top_k=args.top_k,
        pos_frac=args.pos_frac, mode=args.mode, passage_grams=[_grams(t) for t in pool_text],
    )
    counts = np.array([len(n) for n in negatives])
    stats.update({
        "pairs": len(rows), "pool": len(pool_ids), "miners": args.miners, "mode": args.mode, "skip": args.skip,
        "top_k": args.top_k, "pos_frac": args.pos_frac, "per_question": args.per_question,
        "questions_without_negative": int((counts == 0).sum()), "mean_negatives": float(counts.mean()),
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r, neg in zip(rows, negatives):
            f.write(json.dumps({**r, "neg_passage_ids": [pool_ids[i] for i in neg]}, ensure_ascii=False) + "\n")
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
