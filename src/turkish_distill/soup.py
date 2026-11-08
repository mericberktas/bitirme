"""Weight averaging ("soup") of backbones that share the same architecture and initialisation.

Used to blend a fine-tuned model back toward the model it started from (e.g. the distilled student and its
contrastive fine-tune). Usage:

    python -m turkish_distill.soup --models A/final B/final --weights 0.5 0.5 --out models/soup_a50

Other ways of merging models with the same initialisation (`--method`; the model they started from is `--base`):

* `linear` (default): the weighted mean above.
* `slerp`: spherical interpolation of two models (weight of the second = t). With `--base` the interpolation is done
  between the two models' changes from the base (task vectors), which is where the models actually differ; without
  it between the raw weights, which for two models of one initialisation is nearly linear. Used for Qwen3-Embedding.
* `ties`: TIES merging (Yadav et al., 2023, not in our literature survey): per tensor the changes from the base are
  trimmed to the `--density` fraction of largest magnitude, the sign of every entry is elected by the summed trimmed
  changes, and only the changes that agree with it are averaged; the result is added to the base, scaled by `--scale`.
"""

from __future__ import annotations

import argparse
import json
import math
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


def slerp_tensors(a: torch.Tensor, b: torch.Tensor, t: float, parallel: float = 0.9995) -> torch.Tensor:
    """Spherical linear interpolation of two tensors read as flat vectors (t = 0 gives `a`, t = 1 gives `b`).

    As in mergekit the sines weight the ORIGINAL vectors, so lengths are interpolated along with the direction; for
    zero vectors or directions closer than `parallel` (cosine) it falls back to the linear interpolation.
    """
    af, bf = a.float().reshape(-1), b.float().reshape(-1)
    na, nb = float(af.norm()), float(bf.norm())
    linear = (1.0 - t) * a.float() + t * b.float()
    if na < 1e-12 or nb < 1e-12:
        return linear
    cos = float((af / na).dot(bf / nb))
    if abs(cos) > parallel:
        return linear
    omega = math.acos(max(-1.0, min(1.0, cos)))
    s0 = math.sin((1.0 - t) * omega) / math.sin(omega)
    s1 = math.sin(t * omega) / math.sin(omega)
    return s0 * a.float() + s1 * b.float()


def slerp_state_dicts(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], t: float, base: Optional[dict[str, torch.Tensor]] = None
) -> dict[str, torch.Tensor]:
    """Per-tensor SLERP of two models; with `base` the interpolation runs on the changes from the base and the result
    is base + interpolated change. Non-floating-point tensors come from `a`."""
    if list(a) != list(b) or (base is not None and list(base) != list(a)):
        raise ValueError("the models do not have the same parameters")
    out: dict[str, torch.Tensor] = {}
    for k, first in a.items():
        if not first.is_floating_point():
            out[k] = first.clone()
            continue
        if base is None:
            out[k] = slerp_tensors(first, b[k], t).to(first.dtype)
        else:
            out[k] = (base[k].float() + slerp_tensors(first.float() - base[k].float(), b[k].float() - base[k].float(), t)).to(first.dtype)
    return out


def _trim_to_density(delta: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the `density` fraction of entries with the largest magnitude, zero the rest."""
    if density >= 1.0 or delta.numel() == 0:
        return delta
    keep = max(1, int(round(delta.numel() * density)))
    threshold = delta.abs().reshape(-1).kthvalue(delta.numel() - keep + 1).values
    return delta * (delta.abs() >= threshold)


def ties_state_dicts(
    base: dict[str, torch.Tensor],
    models: list[dict[str, torch.Tensor]],
    density: float = 0.5,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """TIES merging: trim every model's change from `base` to its largest `density` fraction, elect the sign of every
    entry from the summed trimmed changes (ties go positive), average only the changes that agree with the elected
    sign (disjoint mean) and add `scale` times that to the base. Non-floating-point tensors come from the base."""
    if not models or not 0.0 < density <= 1.0:
        raise ValueError("need at least one model and a density in (0, 1]")
    if any(list(m) != list(base) for m in models):
        raise ValueError("the models do not have the same parameters")
    out: dict[str, torch.Tensor] = {}
    for k, b in base.items():
        if not b.is_floating_point():
            out[k] = b.clone()
            continue
        stack = torch.stack([_trim_to_density(m[k].float() - b.float(), density) for m in models])
        elected = torch.where(stack.sum(dim=0) >= 0, 1.0, -1.0)
        agree = (torch.sign(stack) == elected) & (stack != 0)
        merged = (stack * agree).sum(dim=0) / agree.sum(dim=0).clamp(min=1)
        out[k] = (b.float() + scale * merged).to(b.dtype)
    return out


def merge_models(
    paths: list[str],
    out_dir: str,
    method: str = "linear",
    weights: Optional[list[float]] = None,
    base: Optional[str] = None,
    density: float = 0.5,
    scale: float = 1.0,
    label: Optional[str] = None,
) -> Path:
    """Merge the HF backbones at `paths` (`linear`, `slerp` or `ties`, see the module docstring); save model, tokenizer
    (from the first path) and SOUP.json (method and settings) to `out_dir`."""
    from transformers import AutoModel, AutoTokenizer

    weights = list(weights) if weights is not None else [1.0] * len(paths)
    models = [AutoModel.from_pretrained(p, torch_dtype=torch.float32) for p in paths]
    sds = [m.state_dict() for m in models]
    base_sd = AutoModel.from_pretrained(base, torch_dtype=torch.float32).state_dict() if base else None
    if method == "linear":
        merged = weighted_average_state_dicts(sds, weights)
    elif method == "slerp":
        if len(paths) != 2:
            raise ValueError("slerp merges exactly two models")
        t = weights[1] / float(sum(weights))
        merged = slerp_state_dicts(sds[0], sds[1], t, base_sd)
    elif method == "ties":
        if base_sd is None:
            raise ValueError("ties needs --base (the model the others were trained from)")
        merged = ties_state_dicts(base_sd, sds, density, scale)
    else:
        raise ValueError(f"unknown method {method!r}")
    models[0].load_state_dict(merged)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models[0].save_pretrained(out)
    AutoTokenizer.from_pretrained(paths[0]).save_pretrained(out)
    total = float(sum(weights))
    (out / "SOUP.json").write_text(
        json.dumps(
            {
                "label": label,
                "method": method,
                "models": paths,
                "weights": [w / total for w in weights],
                "base": base,
                "density": density if method == "ties" else None,
                "scale": scale if method == "ties" else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out


def soup_models(paths: list[str], weights: list[float], out_dir: str, label: Optional[str] = None) -> Path:
    """Average the HF backbones at `paths`, save model, tokenizer (from the first path) and SOUP.json to `out_dir`."""
    return merge_models(paths, out_dir, "linear", weights, label=label)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Merge HF backbones (weight average, SLERP, TIES)")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="default: equal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["linear", "slerp", "ties"], default="linear")
    parser.add_argument("--base", default=None, help="the model the others were trained from (needed for ties)")
    parser.add_argument("--density", type=float, default=0.5, help="ties: fraction of each change that is kept")
    parser.add_argument("--scale", type=float, default=1.0, help="ties: scale of the merged change")
    args = parser.parse_args(argv)
    print(merge_models(args.models, args.out, args.method, args.weights, args.base, args.density, args.scale))


if __name__ == "__main__":
    main()
