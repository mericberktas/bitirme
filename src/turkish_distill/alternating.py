"""Alternating train-and-merge for multi-teacher distillation (Merge-of-Thought style).

Instead of one run that sums the teachers' losses, every round trains one arm per teacher: the arm starts from the
model merged at the end of the previous round (round 1: the pre-trained student), learns from its own teacher only for
`steps_per_round` steps, and the arms' backbones are then averaged with equal weights into the next round's starting
point. After the last round the merged model is the result. Merge-of-Thought (Wang et al., LLM reasoning
distillation) found that most of the gain of merging comes from repeating this, not from one final merge; a one-shot
merge of students trained separately is our `soup` (66.52 at 5400 steps against 66.71 for the joint run).

Budget. Training cost per step does not depend on the number of teachers (their vectors are cached), so the joint
two-teacher run of 5400 steps is matched by `rounds * steps_per_round = 2700` steps per arm (two arms, 5400 steps in
all). `rounds * steps_per_round = 5400` gives every arm the joint run's step count at twice its cost.

Every arm is an ordinary `turkish_distill.train` run (name `<name>_r<round>_<teacher>`, resumable like any run), so
checkpoints, logs and diagnostics come for free; each arm keeps its own projection head from round to round
(`student.heads_path`), the optimiser and the learning-rate schedule restart in every round, and the data order of a
round differs from the previous one (`train.seed` + round - 1). Re-running the same command after a disconnect skips
finished arms and merges and resumes the unfinished one.

    python -m turkish_distill.alternating --config CONFIG --name NAME --rounds 4 --steps-per-round 675 \\
        [--arms 0 1] [--override a.b=c ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .config import load_config


def run_name(name: str, round_: int, teacher: str) -> str:
    return f"{name}_r{round_}_{teacher}"


def find_run(output_dir: str, name: str) -> tuple[Optional[str], bool]:
    """(run id, finished) of the latest run folder `output_dir/name/*`; (None, False) when there is none."""
    run_dirs = sorted(glob.glob(os.path.join(output_dir, name, "*", "")))
    if not run_dirs:
        return None, False
    last = run_dirs[-1].rstrip("/\\")
    return os.path.basename(last), os.path.isdir(os.path.join(last, "final"))


def arm_overrides(
    n_teachers: int,
    arm: int,
    run: str,
    steps: int,
    seed: int,
    round_: int,
    init_model: Optional[str] = None,
    heads_path: Optional[str] = None,
) -> list[str]:
    """Overrides that turn the joint config into one arm: only teacher `arm` has weight, `steps` steps, the round's
    seed, and from round 2 on the merged model (and the arm's own heads) as the starting point."""
    out = [f"name={run}", f"train.max_steps={steps}", f"train.seed={seed + round_ - 1}"]
    out += [f"teachers.{k}.weight={1 if k == arm else 0}" for k in range(n_teachers)]
    if init_model is not None:
        out += [f"student.hf_id={init_model}", "student.revision=null"]
    if heads_path is not None:
        out.append(f"student.heads_path={heads_path}")
    return out


def default_runner(command: list[str]) -> None:
    subprocess.run(command, check=True)


def default_merger(paths: list[str], weights: list[float], out_dir: str, label: Optional[str] = None) -> None:
    from .soup import soup_models

    soup_models(paths, weights, out_dir, label=label)


def run_alternating(
    config: str,
    name: str,
    rounds: int,
    steps_per_round: int,
    overrides: Optional[list[str]] = None,
    arms: Optional[list[int]] = None,
    runner: Callable[[list[str]], None] = default_runner,
    merger: Callable[..., None] = default_merger,
) -> str:
    """Run (or finish) the whole procedure and return the folder of the final merged model."""
    if rounds < 1 or steps_per_round < 1:
        raise ValueError("rounds and steps_per_round must be positive")
    overrides = list(overrides or [])
    cfg = load_config(config, overrides + [f"name={name}"])
    teachers = [t.name for t in cfg.teachers]
    arms = list(range(len(teachers))) if arms is None else list(arms)
    if len(arms) < 2 or any(not 0 <= a < len(teachers) for a in arms):
        raise ValueError(f"need at least two arms among the {len(teachers)} teachers, got {arms}")
    models_dir = Path(cfg.output_dir).parent / "models"
    seed = cfg.train.seed
    previous_merged: Optional[str] = None
    heads: dict[int, Optional[str]] = {a: None for a in arms}
    rounds_log = []
    for r in range(1, rounds + 1):
        finals = []
        for a in arms:
            rn = run_name(name, r, teachers[a])
            run_id, finished = find_run(cfg.output_dir, rn)
            run_dir = os.path.join(cfg.output_dir, rn, run_id) if run_id else None
            if not finished:
                command = [sys.executable, "-m", "turkish_distill.train", "--config", config]
                if run_id:
                    command += ["--run-id", run_id]
                arm_ov = arm_overrides(len(teachers), a, rn, steps_per_round, seed, r, previous_merged, heads[a])
                for item in overrides + arm_ov:  # the arm's own settings come last and win
                    command += ["--override", item]
                print(f"\n########## round {r}/{rounds}, arm {teachers[a]}: {rn} {run_id or '(new run)'}", flush=True)
                runner(command)
                run_id, finished = find_run(cfg.output_dir, rn)
                run_dir = os.path.join(cfg.output_dir, rn, run_id) if run_id else None
            else:
                print(f"\n########## {rn}: already finished, skipped", flush=True)
            if not run_dir or not os.path.isdir(os.path.join(run_dir, "final")):
                raise RuntimeError(f"{rn} did not produce a final model")
            finals.append(os.path.join(run_dir, "final"))
            heads[a] = os.path.join(run_dir, "final", "teacher_heads.pt")
        merged = models_dir / f"{name}_r{r}"
        if (merged / "SOUP.json").exists():
            print(f"merged model of round {r} exists: {merged}", flush=True)
        else:
            merger(finals, [1.0] * len(finals), str(merged), label=f"{name} round {r}")
        previous_merged = str(merged)
        rounds_log.append({"round": r, "arms": [run_name(name, r, teachers[a]) for a in arms], "merged": previous_merged})
    (Path(previous_merged) / "ALTERNATING.json").write_text(
        json.dumps(
            {
                "name": name,
                "config": config,
                "rounds": rounds,
                "steps_per_round": steps_per_round,
                "steps_per_arm": rounds * steps_per_round,
                "arms": [teachers[a] for a in arms],
                "log": rounds_log,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nfinal merged model: {previous_merged}", flush=True)
    return previous_merged


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Alternating train-and-merge over the teachers of a config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True, help="experiment name: runs are <name>_r<round>_<teacher>")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--steps-per-round", type=int, required=True)
    parser.add_argument("--arms", nargs="+", type=int, default=None, help="teacher indices that get an arm (default: all)")
    parser.add_argument("--override", action="append", default=[], help="dotted.key=value for every arm (repeatable)")
    args = parser.parse_args(argv)
    run_alternating(args.config, args.name, args.rounds, args.steps_per_round, args.override, args.arms)


if __name__ == "__main__":
    main()
