"""Gradient-conflict probe: do the loss terms of the teachers push the shared encoder in agreeing directions?

Every (teacher, loss term) pair is a separate objective on the same encoder: BGE cosine, BGE similarity (Gram),
8K cosine, 8K similarity, ... For each batch this module measures the gradient each term would give

* at the pooled embedding (`emb`): cheap, it is exactly what gradient caching hands to the encoder, and
* at the encoder parameters (`par`): the real update direction (one extra backward pass per term and chunk),

and reports the norm of every term's gradient, the cosine between every pair of terms (positive: they push the same
way, negative: they conflict, about zero: roughly orthogonal), the cosine of every term with the sum (the actual
update) and the same between the teachers (sum of their terms). Nothing here changes training: gradients are taken
with `torch.autograd.grad`, so `.grad` of the parameters stays untouched, and the random-number state is restored.

Inside training: `train.grad_probe_every: 200` logs `gradprobe/...` metrics every 200 steps. Stand-alone, on a finished
or running run (the latest checkpoint of its run folder), averaged over some batches:

    python -m turkish_distill.grad_conflict --config CONFIG --run-dir RUN_DIR [--batches 10] [--params last:4]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .config import LossConfig, TeacherConfig, teacher_loss_scale, teacher_weight
from .contrastive import _capture_rng, _restore_rng
from .losses import MultiTeacherLoss
from .models import StudentEncoder

_KINDS = ("cosine", "sim", "resim")


@dataclass(frozen=True)
class TermSpec:
    """One objective to differentiate: a loss config and a teacher list under which the criterion is exactly that term."""

    label: str  # e.g. "bge_m3.cosine"
    group: str  # the teacher's name for per-teacher terms, "relation" or "listwise" otherwise
    loss: LossConfig
    teachers: tuple[TeacherConfig, ...]


def build_term_specs(loss_cfg: LossConfig, teachers: list[TeacherConfig], step: int = 0) -> list[TermSpec]:
    """The loss terms of the current config, each as its own criterion that keeps the real normalisation.

    Other teachers are silenced for the term by `loss_weights` (they stay in the weight sum, so the scale of a term is
    the one it has in training) and the other term kinds by zero global weights. The sum of the terms is the training
    loss (up to the ranking hinge's random pairs and the gor term, which is not probed).
    """
    all_off = dict(cosine_weight=0.0, sim_weight=0.0, resim_weight=0.0, listwise_weight=0.0, gor_weight=0.0)
    shared = loss_cfg.relation_mode == "shared"
    active = [t for t in teachers if teacher_weight(t, step) > 0]
    specs: list[TermSpec] = []
    for kind in ("cosine",) if shared else _KINDS:
        global_weight = getattr(loss_cfg, f"{kind}_weight")
        if not global_weight:
            continue
        cfg = dataclasses.replace(loss_cfg, **{**all_off, f"{kind}_weight": global_weight})
        for t in active:
            if teacher_loss_scale(t, kind) <= 0:
                continue
            silenced = tuple(
                other if other.name == t.name else dataclasses.replace(other, loss_weights={**other.loss_weights, kind: 0.0})
                for other in teachers
            )
            specs.append(TermSpec(f"{t.name}.{kind}", t.name, cfg, silenced))
    if shared and (loss_cfg.sim_weight or loss_cfg.resim_weight):
        cfg = dataclasses.replace(loss_cfg, **{**all_off, "sim_weight": loss_cfg.sim_weight, "resim_weight": loss_cfg.resim_weight})
        specs.append(TermSpec("relation", "relation", cfg, tuple(teachers)))
    if loss_cfg.listwise_weight:
        cfg = dataclasses.replace(loss_cfg, **{**all_off, "listwise_weight": loss_cfg.listwise_weight})
        specs.append(TermSpec("listwise", "listwise", cfg, tuple(teachers)))
    return specs


def select_params(model: StudentEncoder, mode: str = "all") -> list[torch.nn.Parameter]:
    """Encoder parameters whose gradients are compared: all of them, or only the top N layers ("last:4")."""
    named = [(n, p) for n, p in model.backbone.named_parameters() if p.requires_grad]
    if mode == "all":
        return [p for _, p in named]
    m = re.fullmatch(r"last:(\d+)", mode)
    if not m:
        raise ValueError(f"params must be 'all' or 'last:N', got {mode!r}")
    layer_of = {n: re.search(r"layers\.(\d+)\.", n) for n, _ in named}
    indices = [int(match.group(1)) for match in layer_of.values() if match]
    if not indices:
        raise ValueError("no 'layers.N.' parameters in this backbone; use params='all'")
    first = max(indices) - int(m.group(1)) + 1
    return [p for n, p in named if layer_of[n] and int(layer_of[n].group(1)) >= first]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a * b).sum() / (na * nb))


def conflict_metrics(vectors: dict[str, torch.Tensor], groups: dict[str, str], prefix: str) -> dict[str, float]:
    """Norms, pairwise cosines, cosine with the total and the same between groups (teachers) of per-term gradients."""
    out: dict[str, float] = {}
    labels = list(vectors)
    total = sum(vectors.values())
    out[f"{prefix}_norm/total"] = float(total.norm())
    for label in labels:
        out[f"{prefix}_norm/{label}"] = float(vectors[label].norm())
        out[f"{prefix}_cos_total/{label}"] = _cosine(vectors[label], total)
    for a, b in itertools.combinations(labels, 2):
        out[f"{prefix}_cos/{a}~{b}"] = _cosine(vectors[a], vectors[b])
    by_group: dict[str, torch.Tensor] = {}
    for label in labels:
        g = groups[label]
        by_group[g] = vectors[label] if g not in by_group else by_group[g] + vectors[label]
    for g, vec in by_group.items():
        out[f"{prefix}_norm_group/{g}"] = float(vec.norm())
    for a, b in itertools.combinations(list(by_group), 2):
        out[f"{prefix}_cos_group/{a}~{b}"] = _cosine(by_group[a], by_group[b])
    return out


def _pooled_gradients(
    model: StudentEncoder,
    specs: list[TermSpec],
    pooled: torch.Tensor,
    targets: dict[str, torch.Tensor],
    step: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Gradient of every term's loss w.r.t. the pooled embeddings of the batch."""
    grads: dict[str, torch.Tensor] = {}
    for spec in specs:
        criterion = MultiTeacherLoss(spec.loss, list(spec.teachers), seed=seed)
        criterion.set_step(step)
        leaf = pooled.detach().clone().requires_grad_(True)
        projections = {name: head(leaf) for name, head in model.heads.items()}
        loss, _ = criterion(leaf, projections, targets)
        (grad,) = torch.autograd.grad(loss, leaf, allow_unused=True)
        grads[spec.label] = torch.zeros_like(leaf) if grad is None else grad.detach()
    return grads


def probe_gradients(
    model: StudentEncoder,
    loss_cfg: LossConfig,
    teachers: list[TeacherConfig],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: dict[str, torch.Tensor],
    chunk_size: Optional[int] = None,
    use_amp: bool = False,
    step: int = 0,
    seed: int = 0,
    params: str = "all",
    levels: tuple[str, ...] = ("emb", "par"),
) -> dict[str, float]:
    """Gradient-conflict metrics of one batch (see the module docstring). Does not touch `.grad` or the RNG stream."""
    unknown = set(levels) - {"emb", "par"}
    if unknown:
        raise ValueError(f"levels must be 'emb' and/or 'par', got {sorted(unknown)}")
    specs = build_term_specs(loss_cfg, teachers, step)
    if len(specs) < 2:
        return {}
    device = input_ids.device

    def encode(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            return model.encode(ids, mask).float()

    size = input_ids.size(0)
    if chunk_size is None or chunk_size >= size:
        slices = [slice(0, size)]
    else:
        slices = [slice(i, i + chunk_size) for i in range(0, size, chunk_size)]
    states, parts = [], []
    with torch.no_grad():
        for sl in slices:
            states.append(_capture_rng(device))
            parts.append(encode(input_ids[sl], attention_mask[sl]))
    pooled = torch.cat(parts)
    pooled_grads = _pooled_gradients(model, specs, pooled, targets, step, seed)
    groups = {s.label: s.group for s in specs}

    metrics: dict[str, float] = {}
    if "emb" in levels:
        metrics.update(conflict_metrics({k: g.reshape(-1) for k, g in pooled_grads.items()}, groups, "emb"))
    if "par" in levels:
        plist = select_params(model, params)
        sizes = [p.numel() for p in plist]
        vectors = {label: torch.zeros(sum(sizes), device=device) for label in pooled_grads}
        for sl, state in zip(slices, states):
            with _restore_rng(state, device):
                chunk = encode(input_ids[sl], attention_mask[sl])
            for label, grad in pooled_grads.items():
                flat = torch.autograd.grad(chunk, plist, grad_outputs=grad[sl], retain_graph=True, allow_unused=True)
                offset = 0
                for g, n in zip(flat, sizes):
                    if g is not None:
                        vectors[label][offset : offset + n] += g.reshape(-1).float()
                    offset += n
            del chunk
        metrics.update(conflict_metrics(vectors, groups, "par"))
    return metrics


def average_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every metric over the batches (a metric missing from a batch is averaged over the batches that have it)."""
    keys = sorted({k for r in records for k in r})
    return {k: sum(r[k] for r in records if k in r) / sum(1 for r in records if k in r) for k in keys}


def format_report(metrics: dict[str, float], level: str = "par") -> str:
    """Plain-text summary: term norms, then the pairwise cosines sorted from most conflicting to most agreeing."""
    lines = [f"[{level}] gradient norms"]
    for k, v in sorted(metrics.items()):
        if k.startswith(f"{level}_norm/"):
            lines.append(f"  {k.split('/', 1)[1]:<44}{v:12.5f}")
    lines.append(f"[{level}] cosine between terms (negative = conflict)")
    pairs = sorted((v, k) for k, v in metrics.items() if k.startswith(f"{level}_cos/"))
    lines += [f"  {v:+.3f}  {k.split('/', 1)[1]}" for v, k in pairs]
    lines.append(f"[{level}] cosine between teachers")
    lines += [f"  {v:+.3f}  {k.split('/', 1)[1]}" for k, v in sorted(metrics.items()) if k.startswith(f"{level}_cos_group/")]
    lines.append(f"[{level}] cosine of each term with the total update")
    lines += [f"  {v:+.3f}  {k.split('/', 1)[1]}" for k, v in sorted(metrics.items()) if k.startswith(f"{level}_cos_total/")]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    import numpy as np
    from transformers import AutoTokenizer

    from .checkpoint import latest_checkpoint, load_checkpoint
    from .config import load_config
    from .data import Collator, epoch_batches, load_corpus
    from .teachers import TeacherCache, corpus_fingerprint
    from .train import apply_target_methods
    from .utils import get_device, set_seed

    parser = argparse.ArgumentParser(description="Gradient-conflict probe of the distillation loss terms")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    parser.add_argument("--run-dir", default=None, help="load the latest checkpoint of this run (default: the untrained student)")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--params", default="all", help="'all' or 'last:N' encoder layers for the parameter-level gradients")
    parser.add_argument("--levels", nargs="+", default=["emb", "par"], choices=["emb", "par"])
    parser.add_argument("--out", default=None, help="write the averaged metrics to this JSON file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    set_seed(cfg.train.seed)
    texts = load_corpus(cfg.data, cfg.train.seed)
    fingerprint = corpus_fingerprint(texts)
    caches = {t.name: TeacherCache.open(cfg.data.cache_dir, t, fingerprint, len(texts)) for t in cfg.teachers}
    cfg, caches = apply_target_methods(cfg, caches)
    tokenizer = AutoTokenizer.from_pretrained(cfg.student.hf_id, revision=cfg.student.revision)
    model = StudentEncoder.from_pretrained(cfg.student, cfg.teachers)
    step = 0
    if args.run_dir:
        path = latest_checkpoint(Path(args.run_dir) / "checkpoints")
        if path is None:
            raise FileNotFoundError(f"no checkpoint under {args.run_dir}/checkpoints")
        state = load_checkpoint(path, map_location="cpu")
        model.load_state_dict(state["model"])
        step = int(state["step"])
        print(f"loaded {path.name} (step {step})", flush=True)
    device = get_device()
    model.to(device).train()
    collator = Collator(tokenizer, texts, caches, cfg.student.max_length)
    n = len(texts)
    train_idx = np.arange(0, n - min(cfg.data.val_size, max(2, n // 10)))
    batches = epoch_batches(train_idx, cfg.train.batch_size, cfg.train.seed, epoch=0)[: args.batches]
    use_amp = bool(cfg.train.bf16 and device.type == "cuda")
    records = []
    for i, idx in enumerate(batches):
        batch = collator(idx)
        records.append(
            probe_gradients(
                model, cfg.loss, cfg.teachers, batch["input_ids"].to(device), batch["attention_mask"].to(device),
                {k: v.to(device) for k, v in batch["targets"].items()}, cfg.train.chunk_size, use_amp, step=step,
                seed=cfg.train.seed + i, params=args.params, levels=tuple(args.levels),
            )
        )
        print(f"batch {i + 1}/{len(batches)} done", flush=True)
    mean = average_metrics(records)
    for level in args.levels:
        print(format_report(mean, level), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(mean, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
