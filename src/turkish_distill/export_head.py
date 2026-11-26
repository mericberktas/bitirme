"""Export a trained student WITH one of its projection heads as a SentenceTransformer folder.

    python -m turkish_distill.export_head --model RUN/final --head concat --out RUN/final_head

Our recipe evaluates the pooled backbone vector and drops the heads. A Jasper-style run (`fusion.mode: concat`,
`loss.relation_on: head`) shapes the HEAD output with all three loss terms, and Jasper uses that output (FC1) at
inference, so its fair evaluation needs the head: backbone + mean pooling + Dense(identity activation) holding the
head's weights + Normalize. Only linear heads can be exported this way.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .posthoc import append_linear_head


def head_weights(heads_path: str | Path, head: str) -> tuple:
    """(weight (out, in), bias (out,)) of a linear head in a saved `teacher_heads.pt`."""
    state = torch.load(str(heads_path), map_location="cpu")
    w_key, b_key = f"{head}.weight", f"{head}.bias"
    if w_key not in state:
        names = sorted({k.split(".")[0] for k in state})
        raise ValueError(f"no linear head {head!r} in {heads_path} (heads: {names}; MLP heads cannot be exported)")
    return state[w_key].float().numpy(), state[b_key].float().numpy()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="a saved student folder (final/) with teacher_heads.pt")
    parser.add_argument("--head", required=True, help="teacher (head) name, e.g. concat")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    w, b = head_weights(Path(args.model) / "teacher_heads.pt", args.head)
    base = SentenceTransformer(args.model, device="cpu")
    model = append_linear_head(base, w, b, normalize=not args.no_normalize)
    model.save(args.out)
    print(f"saved {args.model} + head {args.head} ({w.shape[1]} -> {w.shape[0]}) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
