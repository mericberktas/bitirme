"""Contrastive baseline training (resumable) and CLI entry point.

    python -m turkish_distill.train_contrastive --config configs/contrastive_mnrl.yaml [--override a.b=c]

Same run layout as the distillation runs: <output_dir>/<name>/<run_id>/{config.yaml, env.json, metrics.jsonl,
tb/, checkpoints/, final/}. Model selection uses the validation retrieval metrics logged under `val/`;
TR-MTEB is not touched during training.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .checkpoint import cleanup_stale, latest_checkpoint, load_checkpoint, save_checkpoint
from .config import dump_config
from .contrastive import (
    PairCollator,
    contrastive_step,
    evaluate_retrieval,
    load_pairs,
    make_anchor,
    make_anchor_loss,
    make_loss,
    unique_passage_batches,
)
from .contrastive_config import ContrastiveConfig, load_contrastive_config
from .data import ListDataset
from .eval.sts import make_student_encode_fn
from .logging_utils import MetricsLogger
from .models import StudentEncoder
from .train import build_scheduler
from .utils import describe_device, get_device, git_sha, set_seed, write_env_info


def run_contrastive(
    cfg: ContrastiveConfig,
    run_dir: str | Path,
    model: StudentEncoder,
    tokenizer: Any,
    train_pairs: dict[str, list[str]],
    val_pairs: dict[str, list[str]],
    device: torch.device,
    logger: Optional[MetricsLogger] = None,
    resume: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    cleanup_stale(ckpt_dir)
    own_logger = logger is None
    logger = logger or MetricsLogger(run_dir, cfg.tensorboard)
    t = cfg.train
    passage_ids = train_pairs["passage_id"]

    epoch0 = unique_passage_batches(passage_ids, t.batch_size, t.seed, 0)
    steps_per_epoch = len(epoch0)
    if steps_per_epoch == 0:
        raise ValueError(
            f"cannot form a batch of {t.batch_size} pairs with distinct passages from {len(passage_ids)} pairs "
            f"({len(set(passage_ids))} distinct passages)"
        )
    total_steps = t.max_steps if t.max_steps is not None else t.epochs * steps_per_epoch
    print(f"{len(passage_ids)} pairs, {steps_per_epoch} steps/epoch, {total_steps} steps in total", flush=True)

    model.to(device)
    loss_fn = make_loss(cfg.loss)
    anchor_model = anchor_fn = None
    if cfg.loss.anchor_cosine_weight > 0 or cfg.loss.anchor_gram_weight > 0:
        anchor_model = make_anchor(model)  # taken BEFORE a checkpoint is loaded: the starting model, not a resumed one
        anchor_fn = make_anchor_loss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps, t.warmup_ratio)

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

    collator = PairCollator(tokenizer, train_pairs, cfg.data.query_max_length, cfg.data.passage_max_length)
    encode_q = make_student_encode_fn(model, tokenizer, device, cfg.data.query_max_length)
    encode_p = make_student_encode_fn(model, tokenizer, device, cfg.data.passage_max_length, batch_size=64)

    @torch.no_grad()  # the encode functions themselves do not disable autograd
    def validate() -> dict[str, float]:
        return evaluate_retrieval(encode_q, encode_p, val_pairs)

    if step == 0:
        logger.log(0, validate(), prefix="val/")  # the starting point: the raw student

    use_amp = bool(t.bf16 and device.type == "cuda")
    model.train()
    window: dict[str, float] = {}
    window_count = 0
    window_start = time.time()
    last_loss: Optional[float] = None

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
    while step < total_steps:
        epoch, offset = divmod(step, steps_per_epoch)
        batches = unique_passage_batches(passage_ids, t.batch_size, t.seed, epoch) or epoch0
        # A later epoch can yield a batch or two fewer than epoch 0; cycling keeps step -> batch deterministic.
        batches = [batches[i % len(batches)] for i in range(steps_per_epoch)][offset:]
        loader = DataLoader(ListDataset(batches), batch_size=None, collate_fn=collator, num_workers=t.num_workers)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logs = contrastive_step(model, batch, loss_fn, t.chunk_size, use_amp, anchor_model, anchor_fn)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            for k, v in logs.items():
                window[k] = window.get(k, 0.0) + v
            window["grad_norm"] = window.get("grad_norm", 0.0) + float(grad_norm)
            window_count += 1
            last_loss = logs["loss"]

            if step % t.log_every == 0 or step == total_steps:
                elapsed = max(time.time() - window_start, 1e-9)
                record = {k: v / window_count for k, v in window.items()}
                record["lr"] = scheduler.get_last_lr()[0]
                record["pairs_per_sec"] = window_count * t.batch_size / elapsed
                if device.type == "cuda":
                    record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                logger.log(step, record, prefix="train/")
                window, window_count, window_start = {}, 0, time.time()
            if step % t.eval_every == 0 or step == total_steps:
                logger.log(step, validate(), prefix="val/")
                model.train()
            if step % t.ckpt_every == 0:
                save(step)
                last_saved = step
            if step >= total_steps:
                break

    if last_saved != step:
        save(step)
    model.save_backbone(run_dir / "final", tokenizer)
    logger.flush()
    if own_logger:
        logger.close()
    return {"final_step": step, "last_loss": last_loss}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Contrastive baseline training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None, help="reuse an existing run id to resume it")
    parser.add_argument("--override", action="append", default=[], help="dotted.key=value (repeatable)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    cfg = load_contrastive_config(args.config, args.override)
    sha = git_sha()
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{(sha or 'nogit')[:7]}"
    run_dir = Path(cfg.output_dir) / cfg.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    env_path = run_dir / "env.json"
    write_env_info(env_path if not env_path.exists() else run_dir / f"env_resume_{int(time.time())}.json")

    set_seed(cfg.train.seed)
    train_pairs = load_pairs(cfg.data.pairs_train, cfg.data.max_train_pairs, cfg.train.seed)
    val_pairs = load_pairs(cfg.data.pairs_val)
    tokenizer = AutoTokenizer.from_pretrained(cfg.student.hf_id, revision=cfg.student.revision)
    model = StudentEncoder.from_pretrained(cfg.student, [])  # no teachers, so no projection heads

    print(f"run dir: {run_dir}", flush=True)
    print(f"device: {describe_device(get_device())}", flush=True)
    print(run_contrastive(cfg, run_dir, model, tokenizer, train_pairs, val_pairs, get_device(), resume=not args.no_resume), flush=True)


if __name__ == "__main__":
    main()
