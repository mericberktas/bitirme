"""Weight averaging ("soup") of backbones that share the same architecture and initialisation.

Used to blend a fine-tuned model back toward the model it started from (e.g. the distilled student and its
contrastive fine-tune). Usage:

    python -m turkish_distill.soup --models A/final B/final --weights 0.5 0.5 --out models/soup_a50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch


def weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float]
) -> dict[str, torch.Tensor]:
    """Weighted mean of floating-point tensors (weights are normalised); other tensors come from the first model."""
    if not state_dicts or len(state_dicts) != len(weights):
        raise ValueError("need one weight per model")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    total = float(sum(weights))
    keys = list(state_dicts[0])
    for sd in state_dicts[1:]:
        if list(sd) != keys:
            raise ValueError("the models do not have the same parameters")
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        first = state_dicts[0][k]
        if not first.is_floating_point():
            out[k] = first.clone()
            continue
        acc = torch.zeros_like(first, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            if sd[k].shape != first.shape:
                raise ValueError(f"shape mismatch for {k}")
            acc += sd[k].to(torch.float32) * (w / total)
        out[k] = acc.to(first.dtype)
    return out


def soup_models(paths: list[str], weights: list[float], out_dir: str, label: Optional[str] = None) -> Path:
    """Average the HF backbones at `paths`, save model, tokenizer (from the first path) and SOUP.json to `out_dir`."""
    from transformers import AutoModel, AutoTokenizer

    models = [AutoModel.from_pretrained(p, torch_dtype=torch.float32) for p in paths]
    merged = weighted_average_state_dicts([m.state_dict() for m in models], weights)
    models[0].load_state_dict(merged)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models[0].save_pretrained(out)
    AutoTokenizer.from_pretrained(paths[0]).save_pretrained(out)
    total = float(sum(weights))
    (out / "SOUP.json").write_text(
        json.dumps(
            {"label": label, "models": paths, "weights": [w / total for w in weights]}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return out


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Weight-average HF backbones")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(soup_models(args.models, args.weights, args.out))


if __name__ == "__main__":
    main()
