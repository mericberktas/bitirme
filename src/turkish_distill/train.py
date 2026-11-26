"""Distillation training loop (resumable) and CLI entry point.

    python -m turkish_distill.train --config configs/multi_teacher.yaml [--run-id ID] [--override a.b=c]
"""

from __future__ import annotations

import argparse
import math
import re
import dataclasses
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .averaging import make_averagers
from .checkpoint import cleanup_stale, latest_checkpoint, load_checkpoint, save_checkpoint
from .config import ExperimentConfig, TeacherConfig, dump_config, load_config, teacher_weight
from .contrastive import _capture_rng, _restore_rng, evaluate_retrieval, load_pairs
from .data import (
    Collator,
    ListDataset,
    epoch_batches,
    load_corpus,
    neighbor_epoch_batches,
    sampling_plan,
    student_texts,
    weighted_epoch_batches,
)
from .diagnostics import effective_rank, geometry_agreement, mean_pairwise_cosine, teacher_cosine
from .grad_conflict import probe_gradients
from .gradproj import pcgrad_combine
from .joint import JointContrastive
from .eval.sts import evaluate_sts, load_sts_pairs, make_student_encode_fn
from .logging_utils import MetricsLogger
from .losses import MultiTeacherLoss
from .models import StudentEncoder
from .targets import ArrayCache, CenteredCache, ConcatCache, PCACache, cached_neighbor_clusters, fuse_teachers, geometry_report
from .teachers import TeacherCache, corpus_fingerprint
from .utils import describe_device, get_device, git_sha, set_seed, write_env_info


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float, t=None):
    """Warmup then linear decay to 0 over all steps; with `t.stage_lrs` a restart at every freezing stage instead."""
    if t is not None and t.stage_lrs is not None:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, staged_lr_factor(t))
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def stage_bounds(t) -> list[tuple[str, int, int]]:
    """(stage, first step, end step) of the freezing stages that occur, in order."""
    total = t.max_steps
    cuts = [("heads", int(round(t.head_only_until * total)))]
    last_end = int(round(t.last_layers_until * total)) if t.last_layers_until else cuts[0][1]
    cuts.append(("last", last_end))
    cuts.append(("all", total))
    bounds, start = [], 0
    for name, end in cuts:
        if end > start:
            bounds.append((name, start, end))
            start = end
    return bounds


def staged_lr_factor(t):
    """LambdaLR factor (multiplies train.lr) restarting at every stage: stage lr, warmup over `warmup_ratio` of the
    stage, then `stage_schedule` decay. The stages match `freeze_stage` (same rounding of the step shares)."""
    lrs = dict(zip(("heads", "last", "all"), (float(x) for x in t.stage_lrs)))
    bounds = stage_bounds(t)

    def factor(step: int) -> float:
        for name, start, end in bounds:
            if step < end or name == bounds[-1][0]:
                break
        length = max(1, end - start)
        local = min(max(step - start, 0), length - 1)
        warmup = max(1, int(length * t.warmup_ratio))
        if local < warmup:
            shape = (local + 1) / warmup
        elif t.stage_schedule == "constant":
            shape = 1.0
        else:
            progress = (local - warmup) / max(1, length - warmup)
            shape = 1.0 - progress if t.stage_schedule == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))
        return lrs[name] / t.lr * max(0.0, shape)

    return factor


def distill_step(
    model: StudentEncoder,
    criterion: MultiTeacherLoss,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: dict[str, torch.Tensor],
    chunk_size: Optional[int] = None,
    use_amp: bool = False,
    projection: str = "none",
    generator: Optional[torch.Generator] = None,
) -> dict[str, float]:
    """One forward/backward pass of the distillation loss; gradients stay on the parameters.

    The losses compare every text with the others in the batch (similarity matrix, relative ranking), so the
    batch cannot simply be split. With `chunk_size` the pooled embeddings of the whole batch are computed in
    mini-batches without gradients, the loss and its gradient w.r.t. the pooled embeddings are computed on the
    full batch (the teacher heads get their gradients there), and each mini-batch is then encoded again with
    gradients and back-propagated with the stored embedding gradient. The result equals the plain step; the RNG
    state is restored so dropout masks are the same in both passes.

    With `projection` "pcgrad" the gradient each teacher's loss gives the pooled embeddings is first made free of its
    conflict with the other teachers' gradients (gradproj.pcgrad_combine) and the sum of those replaces `pooled.grad`;
    the heads keep the gradients of their own teacher. That needs the pooled embeddings as a leaf, so the batch always
    goes through the chunked path (one chunk when `chunk_size` is not set).
    """
    device = input_ids.device

    def encode(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            return model.encode(ids, mask).float()

    size = input_ids.size(0)
    if projection not in ("none", "pcgrad"):
        raise ValueError(f"projection must be 'none' or 'pcgrad', got {projection!r}")
    if projection == "none" and (chunk_size is None or chunk_size >= size):
        pooled = encode(input_ids, attention_mask)
        projections = {name: head(pooled) for name, head in model.heads.items()}
        loss, logs = criterion(pooled, projections, targets)
        loss.backward()
        return logs

    step_size = size if (chunk_size is None or chunk_size >= size) else chunk_size
    slices = [slice(i, i + step_size) for i in range(0, size, step_size)]
    states, parts = [], []
    with torch.no_grad():
        for sl in slices:
            states.append(_capture_rng(device))
            parts.append(encode(input_ids[sl], attention_mask[sl]))
    pooled = torch.cat(parts).requires_grad_(True)
    projections = {name: head(pooled) for name, head in model.heads.items()}
    if projection == "pcgrad":
        shares: dict[str, torch.Tensor] = {}
        loss, logs = criterion(pooled, projections, targets, parts_out=shares)
        teacher_grads = []
        for key, share in shares.items():
            if key == "common" or not share.requires_grad:
                continue
            (g,) = torch.autograd.grad(share, pooled, retain_graph=True, allow_unused=True)
            teacher_grads.append(torch.zeros_like(pooled) if g is None else g.detach())
        combined = None
        if len(teacher_grads) >= 2:
            combined, info = pcgrad_combine(teacher_grads, generator)
            logs.update({f"pcgrad/{k}": v for k, v in info.items()})
            common = shares.get("common")
            if common is not None and common.requires_grad:
                (g,) = torch.autograd.grad(common, pooled, retain_graph=True, allow_unused=True)
                if g is not None:
                    combined = combined + g.detach()
        loss.backward()  # teacher heads get their gradient here; `pooled` gets the plain sum, replaced below
        if combined is not None:
            pooled.grad = combined
    else:
        loss, logs = criterion(pooled, projections, targets)
        loss.backward()  # teacher heads get their gradient here; `pooled` gets .grad
    for sl, state in zip(slices, states):
        with _restore_rng(state, device):
            chunk = encode(input_ids[sl], attention_mask[sl])
        (chunk * pooled.grad[sl]).sum().backward()
    return logs


@torch.no_grad()
def sam_perturb(model: torch.nn.Module, rho: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """SAM's ascent step: move every parameter with a gradient by rho * g / ||g|| (global norm). Returns the
    (parameter, offset) pairs to subtract afterwards."""
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return []
    norm = torch.norm(torch.stack([p.grad.detach().float().norm() for p in params]))
    scale = rho / (float(norm) + 1e-12)
    moved = []
    for p in params:
        e = p.grad.detach() * scale
        p.add_(e)
        moved.append((p, e))
    return moved


_LAYER_INDEX = re.compile(r"(?:^|\.)layers?\.(\d+)\.")


def freeze_stage(t, step: int) -> str:
    """Jasper-style staged freezing (train.head_only_until / last_layers_until): "heads", "last" or "all"."""
    for name, start, end in stage_bounds(t):
        if step < end:
            return name
    return "all"


def stage_parameters(model: StudentEncoder, stage: str, last_layers: int, final_layers: int = 0) -> Optional[set[int]]:
    """ids of the parameters trained in a stage (None = all). "last": the heads, the backbone's last `last_layers`
    layers and its parameters outside the layers and the embeddings (the final norm). "all" with `final_layers` > 0:
    the same with `final_layers` layers (Jasper's released stage-3 code)."""
    if stage == "all" and not final_layers:
        return None
    if stage == "all":
        stage, last_layers = "last", final_layers
    keep = {id(p) for p in model.heads.parameters()}
    if stage == "heads":
        return keep
    named = list(model.backbone.named_parameters())
    indices = [int(m.group(1)) for n, _ in named if (m := _LAYER_INDEX.search(n))]
    if not indices:
        raise ValueError("could not find numbered encoder layers in the backbone's parameter names")
    first = max(indices) + 1 - last_layers
    for n, p in named:
        m = _LAYER_INDEX.search(n)
        if (m and int(m.group(1)) >= first) or (not m and "embed" not in n):
            keep.add(id(p))
    return keep


@torch.no_grad()
def init_reduction_heads(model: StudentEncoder, source: str = "concat") -> list[str]:
    """Jasper's stage-3 start: every reduction head `<source>_r<dim>` becomes the average of consecutive row groups of
    the trained full head (Jasper: w.reshape(dim, -1, hidden).mean(1); here np.array_split groups, so the full
    dimension need not be a multiple of `dim`). Linear heads only."""
    full = model.heads[source]
    done = []
    for name, head in model.heads.items():
        if not name.startswith(f"{source}_r"):
            continue
        groups = np.array_split(np.arange(full.weight.shape[0]), head.weight.shape[0])
        head.weight.copy_(torch.stack([full.weight[g].mean(dim=0) for g in groups]))
        head.bias.copy_(torch.stack([full.bias[g].mean() for g in groups]))
        done.append(name)
    return done


def drop_frozen_grads(model: StudentEncoder, keep: Optional[set[int]]) -> None:
    """Remove the gradients of the parameters outside `keep`: AdamW then neither updates nor decays them."""
    if keep is None:
        return
    for p in model.parameters():
        if id(p) not in keep:
            p.grad = None


def pick_teacher(weights: dict[str, float], seed: int, step: int) -> str:
    """train.teacher_sampling "one": the teacher of a step, drawn with probability proportional to its weight."""
    names = [n for n, w in weights.items() if w > 0]
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    rng = np.random.default_rng([seed, 31, step])
    return names[int(rng.choice(len(names), p=probs / probs.sum()))]


@torch.no_grad()
def run_diagnostics(
    model: StudentEncoder,
    collator: Collator,
    val_idx: np.ndarray,
    device: torch.device,
    max_items: int = 1000,
    batch_size: int = 64,
    sts_fn=None,
) -> dict[str, float]:
    """Geometry of the student space and agreement with every teacher on held-out texts."""
    was_training = model.training
    model.eval()
    idx = val_idx[:max_items].tolist()
    pooled_chunks: list[torch.Tensor] = []
    proj_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in model.heads}
    target_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in collator.caches}
    for i in range(0, len(idx), batch_size):
        batch = collator(idx[i : i + batch_size])
        pooled, projections = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        pooled_chunks.append(pooled.float().cpu())
        for name in proj_chunks:
            proj_chunks[name].append(projections[name].float().cpu())
        for name in target_chunks:
            target_chunks[name].append(batch["targets"][name])
    pooled = torch.cat(pooled_chunks)
    metrics = {
        "mean_pairwise_cosine": mean_pairwise_cosine(pooled),
        "effective_rank": effective_rank(pooled),
    }
    for name in target_chunks:
        target = torch.cat(target_chunks[name])
        proj = torch.cat(proj_chunks[name])
        if proj.shape[1] == target.shape[1]:  # a reduction head (concat_r<dim>) has no vector target
            metrics[f"teacher_cosine/{name}"] = teacher_cosine(proj, target)
        metrics[f"geometry_agreement/{name}"] = geometry_agreement(pooled, target)
    if sts_fn is not None:
        try:
            metrics.update(sts_fn())
        except Exception as exc:
            warnings.warn(f"STS evaluation failed: {exc}")
    if was_training:
        model.train()
    return metrics


def run_training(
    cfg: ExperimentConfig,
    run_dir: str | Path,
    model: StudentEncoder,
    tokenizer: Any,
    texts: list[str],
    caches: dict[str, TeacherCache],
    device: torch.device,
    logger: Optional[MetricsLogger] = None,
    sts_data: Optional[tuple[list[str], list[str], list[float]]] = None,
    resume: bool = True,
    retrieval_val: Optional[dict[str, list[str]]] = None,
    neighbor_labels: Optional[np.ndarray] = None,
    joint_pairs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    cleanup_stale(ckpt_dir)
    own_logger = logger is None
    logger = logger or MetricsLogger(run_dir, cfg.tensorboard)
    t = cfg.train

    n = len(texts)
    val_n = min(cfg.data.val_size, max(2, n // 10))
    train_idx = np.arange(0, n - val_n)
    val_idx = np.arange(n - val_n, n)
    sample_probs = None
    if t.sampling != "uniform":
        from .score_texts import load_scores
        from .teachers import corpus_fingerprint as _fingerprint

        scores = load_scores(t.sample_scores, _fingerprint(texts), n)
        train_idx, sample_probs, plan = sampling_plan(train_idx, scores, t.sampling, t.keep_fraction, t.sampling_power)
        print(f"sampling: {plan}", flush=True)
    steps_per_epoch = len(train_idx) // t.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"not enough training texts ({len(train_idx)}) for batch size {t.batch_size}")

    model.to(device)
    criterion = MultiTeacherLoss(cfg.loss, cfg.teachers, seed=t.seed).to(device)
    groups = [{"params": list(model.parameters()), "lr": t.lr, "weight_decay": t.weight_decay}]
    if len(criterion.log_vars):
        groups.append({"params": list(criterion.log_vars.parameters()), "lr": cfg.loss.uncertainty_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=t.lr, weight_decay=t.weight_decay)
    scheduler = build_scheduler(optimizer, t.max_steps, t.warmup_ratio, t)
    averagers = make_averagers(t)
    # fusion.reduce_dims: the step at which the reduction heads (concat_r<dim>) are switched on and initialised
    reduction_starts = {tc.ramp_start_step for tc in cfg.teachers if tc.name.startswith("concat_r")}

    step = 0
    if resume:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = int(state["step"])
            if "criterion" in state:
                criterion.load_state_dict(state["criterion"])
            for key, avg_state in state.get("averagers", {}).items():
                if key in averagers:
                    averagers[key].load_state_dict(avg_state, device)
            if state.get("config_fingerprint") != cfg.fingerprint():
                warnings.warn("config differs from the one the checkpoint was created with")
            print(f"resumed from {ckpt_path.name} (step {step})", flush=True)

    queries: set[str] = set()
    if cfg.student.query_prefix:
        from .teachers import load_query_set

        queries = load_query_set(cfg.student.query_sources)
    seen_texts, n_prefixed = student_texts(texts, cfg.student.query_prefix, queries)
    if cfg.student.query_prefix:
        print(f"student query prefix on {n_prefixed} of {len(texts)} texts: {cfg.student.query_prefix!r}", flush=True)
        if n_prefixed == 0:
            raise ValueError("student.query_prefix is set but none of the corpus texts is in student.query_sources")
    collator = Collator(tokenizer, seen_texts, caches, cfg.student.max_length)  # diagnostics: always full length
    train_collator = Collator(tokenizer, seen_texts, caches, cfg.student.max_length)
    short_steps = int(round(t.short_until * t.max_steps)) if t.short_max_length else 0

    def input_length(at_step: int) -> int:
        return t.short_max_length if at_step < short_steps else cfg.student.max_length

    if short_steps:
        print(f"progressive length: {t.short_max_length} tokens for steps 0-{short_steps - 1}, then {cfg.student.max_length}", flush=True)
    if averagers:
        print("weight averaging: " + ", ".join(
            f"{k} ({'decay ' + str(a.decay) if a.mode == 'ema' else f'every {a.every} steps from step {a.start_step}'})"
            for k, a in averagers.items()), flush=True)
    if t.sam_rho:
        print(f"SAM: rho {t.sam_rho} (two forward/backward passes per step)", flush=True)
    if t.teacher_sampling == "one":
        print("teacher sampling: one teacher per step, drawn with the teacher weights", flush=True)
    bank = None
    if criterion.uses_bank():
        from .bank import TeacherBank

        active = [tc.name for tc in cfg.teachers if tc.name in caches]
        bank = TeacherBank(caches, train_idx, cfg.loss.bank_size, t.seed, cfg.loss.bank_refresh_every, device, active)
        criterion.set_bank(bank)
        print(
            f"teacher bank: {bank.size} training texts per teacher, redrawn every "
            f"{cfg.loss.bank_refresh_every or 'never'} steps; KL weight {cfg.loss.bank_kl_weight}, "
            f"InfoNCE weight {cfg.loss.bank_nce_weight}",
            flush=True,
        )
    sts_fn = None
    if sts_data is not None:
        encode_fn = make_student_encode_fn(model, tokenizer, device, cfg.student.max_length)
        sts_fn = lambda: evaluate_sts(encode_fn, *sts_data)  # noqa: E731

    def diagnostics() -> dict[str, float]:
        return run_diagnostics(model, collator, val_idx, device, sts_fn=sts_fn)

    retrieval_fn = None
    if retrieval_val is not None:
        # Same metric and prefix as the contrastive runs ("val/mrr@10", ...), so all methods are comparable.
        encode_q = make_student_encode_fn(model, tokenizer, device, cfg.eval.query_max_length)
        if cfg.student.query_prefix:
            plain_q, prefix = encode_q, cfg.student.query_prefix
            encode_q = lambda qs: plain_q([prefix + q for q in qs])  # noqa: E731
        encode_p = make_student_encode_fn(model, tokenizer, device, cfg.student.max_length, batch_size=64)
        retrieval_fn = torch.no_grad()(lambda: evaluate_retrieval(encode_q, encode_p, retrieval_val))

    def log_validation(at_step: int) -> None:
        logger.log(at_step, diagnostics(), prefix="diag/")
        if retrieval_fn is not None:
            logger.log(at_step, retrieval_fn(), prefix="val/")
            model.train()  # the encode functions switch back, but be explicit

    if step == 0:
        log_validation(0)

    use_amp = bool(t.bf16 and device.type == "cuda")
    joint = None
    if cfg.joint.pairs is not None:
        if joint_pairs is None:
            raise ValueError("joint.pairs is set but no pairs were passed to run_training")
        joint = JointContrastive(cfg.joint, tokenizer, joint_pairs, t.seed, t.chunk_size, use_amp)
        used = joint.pair_steps(t.max_steps)
        print(
            f"joint contrastive: {len(joint_pairs['question'])} pairs, {joint.steps_per_epoch} batches of "
            f"{cfg.joint.batch_size} per epoch; every {cfg.joint.every} steps from step {cfg.joint.start_step}: "
            f"{used} pair batches = {used / joint.steps_per_epoch:.1f} epochs, weight {cfg.joint.weight}",
            flush=True,
        )
    joint_window: dict[str, float] = {}
    joint_count = 0
    projection_generator = torch.Generator().manual_seed(t.seed)  # visiting order of the gradient surgery (CPU, tiny)
    model.train()
    window: dict[str, float] = {}
    window_count = 0
    window_start = time.time()
    last_logs: dict[str, float] = {}

    def save(at_step: int) -> None:
        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": at_step,
                "config_fingerprint": cfg.fingerprint(),
                **({"criterion": criterion.state_dict()} if len(criterion.log_vars) else {}),
                **({"averagers": {k: a.state_dict() for k, a in averagers.items()}} if averagers else {}),
            },
            ckpt_dir,
            at_step,
            keep_last=t.keep_last,
            local_tmp_dir=cfg.local_tmp_dir,
        )

    last_saved = step
    # One line every ~20 s (carriage-return updates in a notebook cell): step, speed, time left, the latest loss and validation MRR.
    bar = tqdm(total=t.max_steps, initial=step, desc=cfg.name, mininterval=20, smoothing=0.05, dynamic_ncols=True)
    while step < t.max_steps:
        epoch, offset = divmod(step, steps_per_epoch)
        if neighbor_labels is not None and t.neighbor_fraction > 0:
            batches = neighbor_epoch_batches(
                train_idx, neighbor_labels, t.batch_size, t.seed, epoch, t.neighbor_fraction, t.neighbor_group_size, offset
            )
        elif sample_probs is not None:
            batches = weighted_epoch_batches(train_idx, sample_probs, t.batch_size, t.seed, epoch, offset)
        else:
            batches = epoch_batches(train_idx, t.batch_size, t.seed, epoch, offset)
        loader_length = input_length(step)
        train_collator.max_length = loader_length  # workers copy the collator when the loader starts iterating
        loader = DataLoader(ListDataset(batches), batch_size=None, collate_fn=train_collator, num_workers=t.num_workers)
        for batch in loader:
            if input_length(step) != loader_length:
                break  # length switch: rebuild the loader from this step on (the batch is not consumed)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            if step in reduction_starts:
                names = init_reduction_heads(model)
                print(f"step {step}: reduction heads {names} initialised from the concat head", flush=True)
            optimizer.zero_grad(set_to_none=True)
            criterion.set_step(step)  # `step` steps are done, this is step number `step` (teacher weights may ramp)
            if t.teacher_sampling == "one":
                criterion.set_active(None)
                criterion.set_active([pick_teacher(criterion.current_weights(), t.seed, step)])
            if bank is not None:
                bank.at_step(step)
                criterion.set_batch_indices(batch["indices"])
            if t.grad_probe_every and step % t.grad_probe_every == 0:
                # measurement only: autograd.grad leaves .grad alone and the RNG state is restored
                probe = probe_gradients(
                    model, cfg.loss, cfg.teachers, input_ids, attention_mask, targets, t.chunk_size, use_amp,
                    step=step, seed=t.seed, params=t.grad_probe_params,
                )
                if probe:
                    logger.log(step, probe, prefix="gradprobe/")
            logs = distill_step(
                model, criterion, input_ids, attention_mask, targets, t.chunk_size, use_amp,
                projection=t.grad_projection, generator=projection_generator,
            )
            if t.sam_rho:
                moved = sam_perturb(model, t.sam_rho)  # the gradient at the perturbed weights is the one applied
                optimizer.zero_grad(set_to_none=True)
                sam_logs = distill_step(
                    model, criterion, input_ids, attention_mask, targets, t.chunk_size, use_amp,
                    projection=t.grad_projection, generator=projection_generator,
                )
                with torch.no_grad():
                    for p, e in moved:
                        p.sub_(e)
                logs["sam_total"] = sam_logs["total"]
            if joint is not None and joint.applies(step):
                # adds its gradient to the distillation gradient: one objective, one optimiser step
                for k, v in joint.step(model, step, device).items():
                    joint_window[k] = joint_window.get(k, 0.0) + v
                joint_count += 1
            stage = freeze_stage(t, step)
            if stage != "all" or t.final_stage_layers:
                drop_frozen_grads(model, stage_parameters(model, stage, t.last_layers, t.final_stage_layers))
                logs["frozen_stage"] = {"heads": 0.0, "last": 1.0, "all": 2.0}[stage]
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            for averager in averagers.values():
                averager.update(model, step, force=step == t.max_steps)

            for k, v in logs.items():
                window[k] = window.get(k, 0.0) + v
            window["grad_norm"] = window.get("grad_norm", 0.0) + float(grad_norm)
            window_count += 1
            last_logs = logs
            bar.update(1)

            if step % t.log_every == 0 or step == t.max_steps:
                bar.set_postfix(loss=f"{window['total'] / window_count:.4f}" if "total" in window else None, refresh=False)
                elapsed = max(time.time() - window_start, 1e-9)
                record = {k: v / window_count for k, v in window.items()}
                record["lr"] = scheduler.get_last_lr()[0]
                record["samples_per_sec"] = window_count * t.batch_size / elapsed
                if device.type == "cuda":
                    record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                logger.log(step, record, prefix="train/")
                window, window_count, window_start = {}, 0, time.time()
                if joint_count:
                    logger.log(step, {k: v / joint_count for k, v in joint_window.items()}, prefix="train/ctr_")
                    joint_window, joint_count = {}, 0
            if step % t.diag_every == 0 or step == t.max_steps:
                log_validation(step)
            if step % t.ckpt_every == 0:
                save(step)
                last_saved = step
            if step >= t.max_steps:
                break

    bar.close()
    if last_saved != step:
        save(step)
    model.save_backbone(run_dir / "final", tokenizer)
    for key, averager in averagers.items():
        if averager.avg is None:
            continue
        with averager.applied(model):
            model.save_backbone(run_dir / f"final_{key}", tokenizer)
            logger.log(step, diagnostics(), prefix=f"diag_{key}/")
        print(f"saved the {key} average of {averager.count} weight snapshots to final_{key}/", flush=True)
    logger.flush()
    if own_logger:
        logger.close()
    return {"final_step": step, "last_loss": last_logs.get("total")}


def apply_target_methods(cfg: ExperimentConfig, caches: dict[str, Any]) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Methods (a) and (d) and the capacity-aware target: centre teachers (`teachers[].center`), project them onto their
    top principal components (`teachers[].pca_dim`) or fuse them into one target (`fusion`).

    Returns the config the training loop should use (with fusion the only teacher is the fused one) and the
    matching caches. Without either option both are returned unchanged.
    """
    if cfg.fusion.mode == "concat":
        weights = {t.name: teacher_weight(t, 0) for t in cfg.teachers}
        inner = {t.name: (CenteredCache(caches[t.name]) if t.center else caches[t.name]) for t in cfg.teachers}
        concat = ConcatCache(inner, weights)
        print(f"concat target (Jasper-style): {concat.names}, scales {concat.scales}, dim {concat.dim}", flush=True)
        teachers = [TeacherConfig(name="concat", hf_id="fusion:concat", dim=concat.dim, weight=1.0)]
        out = {"concat": concat}
        fz = cfg.fusion
        if fz.reduce_dims:
            share = cfg.train.last_layers_until if fz.reduce_from < 0 else fz.reduce_from
            start = int(round(share * cfg.train.max_steps))
            for d in fz.reduce_dims:
                # Jasper's FC2-FC4: same target, relation terms only, switched on at `start` (a step in the weight)
                teachers.append(TeacherConfig(
                    name=f"concat_r{int(d)}", hf_id="fusion:concat", dim=int(d), weight=float(fz.reduce_weight),
                    weight_start=0.0 if start > 0 else None, ramp_start_step=start, ramp_end_step=start,
                    loss_weights={"cosine": 0.0},
                ))
                out[f"concat_r{int(d)}"] = concat
            print(f"reduction heads {list(fz.reduce_dims)} from step {start}", flush=True)
        return dataclasses.replace(cfg, teachers=teachers), out
    if cfg.fusion.mode != "none":
        weights = {t.name: teacher_weight(t, 0) for t in cfg.teachers}
        fused, report = fuse_teachers(
            caches, weights, cfg.fusion.reference, cfg.fusion.dim, cfg.fusion.fit_samples, cfg.fusion.seed
        )
        print(f"fusion: {report}", flush=True)
        fused_teacher = TeacherConfig(name="fused", hf_id="fusion:aligned_mean", dim=cfg.fusion.dim, weight=1.0)
        return dataclasses.replace(cfg, teachers=[fused_teacher]), {"fused": ArrayCache(fused)}
    wrapped = dict(caches)
    for t in cfg.teachers:
        if t.center:
            wrapped[t.name] = CenteredCache(caches[t.name])
            raw, new = geometry_report(caches[t.name]), geometry_report(caches[t.name], centered=True)
            print(
                f"centred teacher {t.name}: mean vector norm {float(np.linalg.norm(wrapped[t.name].mean)):.3f}; "
                f"pairwise cosine mean/std {raw['mean_cosine']:.3f}/{raw['std']:.3f} -> {new['mean_cosine']:.3f}/{new['std']:.3f}",
                flush=True,
            )
        if t.pca_dim is not None:
            pca = PCACache(caches[t.name], t.pca_dim, t.pca_fit_samples, seed=cfg.train.seed)
            wrapped[t.name] = pca
            probe = np.arange(min(2000, len(pca)))
            agreement = float((pca.get(probe) * caches[t.name].get(probe)).sum(axis=1).mean())
            print(
                f"PCA target for teacher {t.name}: top {t.pca_dim} of {t.dim} components keep {pca.kept_variance:.3f} of the "
                f"variance; mean cosine between target and original vector {agreement:.3f}",
                flush=True,
            )
    return cfg, wrapped


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-teacher distillation training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None, help="reuse an existing run id to resume it")
    parser.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    cfg = load_config(args.config, args.override)
    sha = git_sha()
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{(sha or 'nogit')[:7]}"
    run_dir = Path(cfg.output_dir) / cfg.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    env_path = run_dir / "env.json"
    write_env_info(env_path if not env_path.exists() else run_dir / f"env_resume_{int(time.time())}.json")

    set_seed(cfg.train.seed)
    texts = load_corpus(cfg.data, cfg.train.seed)
    fingerprint = corpus_fingerprint(texts)
    caches = {t.name: TeacherCache.open(cfg.data.cache_dir, t, fingerprint, len(texts)) for t in cfg.teachers}
    neighbor_labels = None
    if cfg.train.neighbor_fraction > 0:
        active = {t.name: caches[t.name] for t in cfg.teachers if teacher_weight(t, 0) > 0}
        n_clusters = cfg.train.neighbor_clusters or max(2, len(texts) // cfg.train.neighbor_group_size)
        neighbor_labels = cached_neighbor_clusters(active, cfg.data.cache_dir, fingerprint, n_clusters, cfg.train.neighbor_iters)
        sizes = np.bincount(neighbor_labels)
        print(
            f"neighbour batches: {cfg.train.neighbor_fraction:.2f} of every batch in groups of {cfg.train.neighbor_group_size}; "
            f"{int((sizes > 0).sum())} clusters, size median {int(np.median(sizes[sizes > 0]))}, max {int(sizes.max())}",
            flush=True,
        )
    cfg, caches = apply_target_methods(cfg, caches)
    tokenizer = AutoTokenizer.from_pretrained(cfg.student.hf_id, revision=cfg.student.revision)
    model = StudentEncoder.from_pretrained(cfg.student, cfg.teachers)

    sts_data = None
    if cfg.eval.sts_dataset:
        try:
            sts_data = load_sts_pairs(cfg.eval.sts_dataset, cfg.eval.sts_split, cfg.eval.sts_columns)
        except Exception as exc:
            warnings.warn(f"STS monitoring disabled: {exc}")

    retrieval_val = load_pairs(cfg.eval.pairs_val) if cfg.eval.pairs_val else None
    joint_pairs = load_pairs(cfg.joint.pairs) if cfg.joint.pairs else None

    print(f"run dir: {run_dir}", flush=True)
    print(f"device: {describe_device(get_device())}", flush=True)
    result = run_training(
        cfg, run_dir, model, tokenizer, texts, caches, get_device(), sts_data=sts_data, resume=not args.no_resume,
        retrieval_val=retrieval_val, neighbor_labels=neighbor_labels, joint_pairs=joint_pairs,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
