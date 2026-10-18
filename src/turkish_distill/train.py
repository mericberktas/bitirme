"""Distillation training loop (resumable) and CLI entry point.

    python -m turkish_distill.train --config configs/multi_teacher.yaml [--run-id ID] [--override a.b=c]
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import cleanup_stale, latest_checkpoint, load_checkpoint, save_checkpoint
from .config import ExperimentConfig, dump_config, load_config
from .contrastive import evaluate_retrieval, load_pairs
from .data import Collator, ListDataset, epoch_batches, load_corpus
from .diagnostics import effective_rank, geometry_agreement, mean_pairwise_cosine, teacher_cosine
from .eval.sts import evaluate_sts, load_sts_pairs, make_student_encode_fn
from .logging_utils import MetricsLogger
from .losses import MultiTeacherLoss
from .models import StudentEncoder
from .teachers import TeacherCache, corpus_fingerprint
from .utils import describe_device, get_device, git_sha, set_seed, write_env_info


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


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
    steps_per_epoch = len(train_idx) // t.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"not enough training texts ({len(train_idx)}) for batch size {t.batch_size}")

    model.to(device)
    criterion = MultiTeacherLoss(cfg.loss, cfg.teachers, seed=t.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = build_scheduler(optimizer, t.max_steps, t.warmup_ratio)

    step = 0
    if resume:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = int(state["step"])
            if state.get("config_fingerprint") != cfg.fingerprint():
                warnings.warn("config differs from the one the checkpoint was created with")
            print(f"resumed from {ckpt_path.name} (step {step})", flush=True)

    collator = Collator(tokenizer, texts, caches, cfg.student.max_length)
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
            },
            ckpt_dir,
            at_step,
            keep_last=t.keep_last,
            local_tmp_dir=cfg.local_tmp_dir,
        )

    last_saved = step
    while step < t.max_steps:
        epoch, offset = divmod(step, steps_per_epoch)
        batches = epoch_batches(train_idx, t.batch_size, t.seed, epoch, offset)
        loader = DataLoader(ListDataset(batches), batch_size=None, collate_fn=collator, num_workers=t.num_workers)
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pooled, projections = model(input_ids, attention_mask)
            loss, logs = criterion(pooled.float(), {k: v.float() for k, v in projections.items()}, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            for k, v in logs.items():
                window[k] = window.get(k, 0.0) + v
            window["grad_norm"] = window.get("grad_norm", 0.0) + float(grad_norm)
            window_count += 1
            last_logs = logs

            if step % t.log_every == 0 or step == t.max_steps:
                elapsed = max(time.time() - window_start, 1e-9)
                record = {k: v / window_count for k, v in window.items()}
                record["lr"] = scheduler.get_last_lr()[0]
                record["samples_per_sec"] = window_count * t.batch_size / elapsed
                if device.type == "cuda":
                    record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                logger.log(step, record, prefix="train/")
                window, window_count, window_start = {}, 0, time.time()
            if step % t.diag_every == 0 or step == t.max_steps:
                log_validation(step)
            if step % t.ckpt_every == 0:
                save(step)
                last_saved = step
            if step >= t.max_steps:
                break

    if last_saved != step:
        save(step)
    model.save_backbone(run_dir / "final", tokenizer)
    logger.flush()
    if own_logger:
        logger.close()
    return {"final_step": step, "last_loss": last_logs.get("total")}


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
    tokenizer = AutoTokenizer.from_pretrained(cfg.student.hf_id, revision=cfg.student.revision)
    model = StudentEncoder.from_pretrained(cfg.student, cfg.teachers)

    sts_data = None
    if cfg.eval.sts_dataset:
        try:
            sts_data = load_sts_pairs(cfg.eval.sts_dataset, cfg.eval.sts_split, cfg.eval.sts_columns)
        except Exception as exc:
            warnings.warn(f"STS monitoring disabled: {exc}")

    retrieval_val = load_pairs(cfg.eval.pairs_val) if cfg.eval.pairs_val else None

    print(f"run dir: {run_dir}", flush=True)
    print(f"device: {describe_device(get_device())}", flush=True)
    result = run_training(
        cfg, run_dir, model, tokenizer, texts, caches, get_device(), sts_data=sts_data, resume=not args.no_resume,
        retrieval_val=retrieval_val,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
