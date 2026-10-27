"""TR-MTEB evaluation of any sentence-embedding model (parametric).

Wraps the authors' fork of MTEB (https://github.com/selmanbaysan/mteb_tr), which
defines the benchmark "MTEB(Turkish)" (26 tasks). The fork pins `datasets<3`,
so install it in the evaluation runtime only:

    pip install "git+https://github.com/selmanbaysan/mteb_tr.git"

Command line:

    python -m turkish_distill.eval.trmteb run --model BAAI/bge-m3 --revision <hash> \
        --output-dir /content/drive/MyDrive/bitirme/eval --dtype float16
    python -m turkish_distill.eval.trmteb summarize /content/drive/MyDrive/bitirme/eval/*

Protocol notes (these affect comparability with published numbers):

* By default the model is encoded WITHOUT any instruction or prefix, which is how
  the TR-MTEB authors' own scripts evaluate (plain `SentenceTransformer`). Use
  `--query-prompt` / `--passage-prompt` to add prefixes explicitly; results are
  then stored under a different label.
* Scores are main scores x 100. "Mean (task)" is the mean over tasks (so the 11
  retrieval tasks weigh most); "Mean (type)" is the mean over the task categories.
  Both are reported, with and without bitext mining, which a monolingual model
  cannot be expected to solve.
* Results are written per task by MTEB, so an interrupted run resumes and
  skips finished tasks.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

BENCHMARK_NAME = "MTEB(Turkish)"
MTEB_TR_INSTALL = 'pip install "git+https://github.com/selmanbaysan/mteb_tr.git"'

# Task names as defined in mteb_tr (mteb/benchmarks/benchmarks.py, MTEB_TR).
TASK_CATEGORIES: dict[str, str] = {
    "WMT16BitextMining": "BitextMining",
    "TurkishMovieSentimentClassification": "Classification",
    "TurkishOffensiveLanguageClassification": "Classification",
    "TurkishProductSentimentClassification": "Classification",
    "TurkishNewsCategoryClassification": "Classification",
    "Turkish75NewsClassification": "Classification",
    "TSTimelineNewsCategoryClassification": "Classification",
    "THYSentimentClassification": "Classification",
    "TurkishIronyClassification": "Classification",
    "TurkishAbstractCorpusClustering": "Clustering",
    "TurkishColumnWritingClustering": "Clustering",
    "MnliTr": "PairClassification",
    "SnliTr": "PairClassification",
    "XNLI": "PairClassification",
    "STSbTR": "STS",
    "SquadTRRetrieval": "Retrieval",
    "TQuadRetrieval": "Retrieval",
    "XQuADRetrieval": "Retrieval",
    "MSMarcoTRRetrieval": "Retrieval",
    "NFCorpusTR": "Retrieval",
    "ArguAnaTR": "Retrieval",
    "CQADupstackGamingRetrievalTR": "Retrieval",
    "FiQA2018TR": "Retrieval",
    "QuoraRetrievalTR": "Retrieval",
    "SCIDOCSTR": "Retrieval",
    "SciFactTR": "Retrieval",
}
# Finer, skill-oriented grouping of the same 26 tasks (our own; not part of TR-MTEB). The official
# categories hide differences: hard scientific retrieval and QA retrieval behave very differently,
# and so do topic classification and sentiment. Used to see which teacher's strength a student inherits.
TASK_GROUPS: dict[str, list[str]] = {
    "RetrHard": ["ArguAnaTR", "SCIDOCSTR", "NFCorpusTR", "SciFactTR", "FiQA2018TR", "CQADupstackGamingRetrievalTR"],
    "RetrQA": ["MSMarcoTRRetrieval", "QuoraRetrievalTR", "SquadTRRetrieval", "TQuadRetrieval", "XQuADRetrieval"],
    "NLI": ["MnliTr", "SnliTr", "XNLI"],
    "STS": ["STSbTR"],
    "Sentiment": ["THYSentimentClassification", "TurkishMovieSentimentClassification", "TurkishProductSentimentClassification"],
    "Pragmatic": ["TurkishIronyClassification", "TurkishOffensiveLanguageClassification"],
    "Topic": ["Turkish75NewsClassification", "TurkishNewsCategoryClassification", "TSTimelineNewsCategoryClassification"],
    "Clustering": ["TurkishAbstractCorpusClustering", "TurkishColumnWritingClustering"],
    "Bitext": ["WMT16BitextMining"],
}
# Tasks whose score depends on the scale of the embeddings (a logistic regression / k-means is fitted on the raw
# vectors); retrieval, STS and pair classification use cosine similarity and do not.
SCALE_SENSITIVE_CATEGORIES = ("Classification", "Clustering")
CATEGORY_ORDER = ["Retrieval", "Classification", "Clustering", "PairClassification", "STS", "BitextMining"]
SCALE_SENSITIVE_TASKS = [t for t, c in TASK_CATEGORIES.items() if c in SCALE_SENSITIVE_CATEGORIES]
_SPLIT_PREFERENCE = ("test", "validation", "dev", "train")
_IGNORED_FILES = {"model_meta.json", "summary.json", "run_info.json"}


# --------------------------------------------------------------------------- #
# Pure functions: reading results and computing summaries (no mteb needed)
# --------------------------------------------------------------------------- #

def sanitize_label(model: str, revision: Optional[str] = None, suffix: str = "") -> str:
    """Folder-safe label for a model (and optional revision / protocol suffix)."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "__", model.strip("/\\")).strip("_")
    label = base or "model"
    if revision:
        label += f"_{revision[:8]}"
    if suffix:
        label += f"_{suffix}"
    return label


def task_score_from_result(data: dict) -> Optional[tuple[float, str]]:
    """Main score x 100 of a task result, averaged over subsets of the preferred split."""
    scores = data.get("scores")
    if not isinstance(scores, dict) or not scores:
        return None
    split = next((s for s in _SPLIT_PREFERENCE if s in scores), None)
    if split is None:
        split = sorted(scores)[0]
    entries = scores[split]
    if isinstance(entries, dict):
        entries = [entries]
    values = [e["main_score"] for e in entries if isinstance(e, dict) and "main_score" in e]
    if not values:
        return None
    return 100.0 * sum(values) / len(values), split


def collect_task_scores(results_dir: str | Path) -> dict[str, dict]:
    """Read every per-task result JSON below `results_dir` (newest file wins per task)."""
    found: dict[str, tuple[float, dict]] = {}
    for path in Path(results_dir).rglob("*.json"):
        if path.name in _IGNORED_FILES:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "task_name" not in data:
            continue
        parsed = task_score_from_result(data)
        if parsed is None:
            continue
        score, split = parsed
        mtime = path.stat().st_mtime
        name = data["task_name"]
        if name not in found or mtime > found[name][0]:
            found[name] = (mtime, {"score": score, "split": split, "file": str(path)})
    return {name: info for name, (_, info) in found.items()}


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summarize_scores(task_scores: dict[str, dict]) -> dict:
    """Aggregate per-task scores into the numbers reported in the TR-MTEB paper."""
    known = {t: v["score"] for t, v in task_scores.items() if t in TASK_CATEGORIES}
    missing = sorted(set(TASK_CATEGORIES) - set(known))
    by_category: dict[str, list[float]] = {c: [] for c in CATEGORY_ORDER}
    for task, score in known.items():
        by_category[TASK_CATEGORIES[task]].append(score)
    category_means = {c: _mean(v) for c, v in by_category.items() if v}
    no_bitext_tasks = [s for t, s in known.items() if TASK_CATEGORIES[t] != "BitextMining"]
    no_bitext_cats = [m for c, m in category_means.items() if c != "BitextMining"]
    return {
        "num_tasks": len(known),
        "complete": not missing,
        "missing_tasks": missing,
        "mean_task": _mean(list(known.values())),
        "mean_task_no_bitext": _mean(no_bitext_tasks),
        "mean_type": _mean(list(category_means.values())),
        "mean_type_no_bitext": _mean(no_bitext_cats),
        "categories": category_means,
        # a group mean is reported only when all its tasks are present (otherwise it is not comparable)
        "groups": {
            g: _mean([known[t] for t in tasks]) if all(t in known for t in tasks) else None
            for g, tasks in TASK_GROUPS.items()
        },
        "per_task": {t: known[t] for t in sorted(known)},
    }


def summarize_dir(results_dir: str | Path) -> dict:
    return summarize_scores(collect_task_scores(results_dir))


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_table(summaries: dict[str, dict]) -> str:
    """Plain-text comparison table, one row per model label."""
    columns = [
        ("Mean(task)", lambda s: s["mean_task"]),
        ("-bitext", lambda s: s["mean_task_no_bitext"]),
        ("Mean(type)", lambda s: s["mean_type"]),
        ("Retr", lambda s: s["categories"].get("Retrieval")),
        ("Clf", lambda s: s["categories"].get("Classification")),
        ("Clust", lambda s: s["categories"].get("Clustering")),
        ("Pair", lambda s: s["categories"].get("PairClassification")),
        ("STS", lambda s: s["categories"].get("STS")),
        ("Bitext", lambda s: s["categories"].get("BitextMining")),
    ]
    width = max([len("model")] + [len(k) for k in summaries]) + 2
    header = "model".ljust(width) + "".join(name.rjust(12) for name, _ in columns) + "  tasks"
    lines = [header, "-" * len(header)]
    for label, summary in summaries.items():
        row = label.ljust(width) + "".join(_fmt(get(summary)).rjust(12) for _, get in columns)
        row += f"  {summary['num_tasks']}/{len(TASK_CATEGORIES)}"
        if not summary["complete"]:
            row += "  (INCOMPLETE: averages not comparable)"
        lines.append(row)
    return "\n".join(lines)


def format_group_table(summaries: dict[str, dict]) -> str:
    """Plain-text table of the skill groups in TASK_GROUPS, one row per model label."""
    groups = list(TASK_GROUPS)
    width = max([len("model")] + [len(k) for k in summaries]) + 2
    header = "model".ljust(width) + "".join(g.rjust(11) for g in groups)
    lines = [header, "-" * len(header)]
    for label, summary in summaries.items():
        values = summary.get("groups", {})
        lines.append(label.ljust(width) + "".join(_fmt(values.get(g)).rjust(11) for g in groups))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Running the evaluation (needs the mteb_tr fork, sentence-transformers, torch)
# --------------------------------------------------------------------------- #

def _unescape(text: Optional[str]) -> Optional[str]:
    return None if text is None else text.replace("\\n", "\n").replace("\\t", "\t")


_PROTOCOL_KEYS = ("model", "revision", "dtype", "max_seq_length")

# Program run by `run_evaluation_isolated` in a fresh Python process.
_CHILD_CODE = (
    "import json, sys\n"
    "from turkish_distill.eval.trmteb import run_evaluation\n"
    "run_evaluation(**json.loads(sys.stdin.read()))\n"
)


def run_evaluation_isolated(label: str, output_dir: str | Path, **kwargs) -> Path:
    """Run `run_evaluation` in a separate Python process and stream its output.

    A CUDA error ("device-side assert") makes the whole process's CUDA context unusable, so in one
    process a single bad model (e.g. incompatible remote code) makes every later model fail too.
    Running each model in its own process confines such failures to that model.

    All arguments must be JSON-serialisable (they are for the notebook presets). Raises RuntimeError
    if the child exits with an error; its output (including the traceback) has already been printed.
    """
    import subprocess

    payload = json.dumps({"label": label, "output_dir": str(output_dir), **kwargs})
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(payload.encode("utf-8"))
    proc.stdin.close()
    while True:  # raw chunks, so progress bars (carriage returns) keep working in notebooks
        chunk = proc.stdout.read1(4096)
        if not chunk:
            break
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"evaluation of {label!r} failed in its own process (exit code {code}); see the output above")
    return Path(output_dir) / label


def prepare_reuse(
    output_dir: str | Path,
    label_dir: str | Path,
    reuse_from: str,
    protocol: dict,
    task_names: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    """Copy prompt-insensitive task results from an earlier plain run.

    Returns (tasks reused, tasks that still have to be evaluated). Retrieval tasks are never
    reused (queries and passages receive the prompts); a non-retrieval task missing from the
    source run is simply evaluated normally.
    """
    src_dir = Path(output_dir) / reuse_from
    info_path = src_dir / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"reuse_from={reuse_from!r}: no run_info.json in {src_dir}; run the plain evaluation first")
    src = json.loads(info_path.read_text(encoding="utf-8"))
    if src.get("prompts"):
        raise ValueError(f"reuse_from={reuse_from!r} used prompts itself; it must be a plain run")
    mismatched = {k: (src.get(k), protocol.get(k)) for k in _PROTOCOL_KEYS if src.get(k) != protocol.get(k)}
    if mismatched:
        raise ValueError(f"reuse_from={reuse_from!r} was run with a different protocol (source, requested): {mismatched}")

    wanted = list(task_names) if task_names else list(TASK_CATEGORIES)
    source_scores = collect_task_scores(src_dir)
    reuse_dir = Path(label_dir) / "reused"
    reused: list[str] = []
    remaining: list[str] = []
    for task in wanted:
        if TASK_CATEGORIES[task] != "Retrieval" and task in source_scores:
            reuse_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_scores[task]["file"], reuse_dir / f"{task}.json")
            reused.append(task)
        else:
            remaining.append(task)
    return reused, remaining


_RUN_PROTOCOL_KEYS = ("model", "revision", "dtype", "max_seq_length", "prompts", "normalize")


def check_label_protocol(label_dir: str | Path, protocol: dict, overwrite: bool = False) -> None:
    """Refuse to add results to a label that was produced under a different protocol.

    MTEB skips tasks that already have a result file, so re-running a label with another dtype,
    prompt or sequence length would silently keep the stale scores and mix two protocols.
    """
    info_path = Path(label_dir) / "run_info.json"
    if overwrite or not info_path.exists():
        return
    try:
        previous = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    changed = {
        k: (previous.get(k), protocol.get(k)) for k in _RUN_PROTOCOL_KEYS if k in previous and previous[k] != protocol.get(k)
    }
    if changed:
        raise ValueError(
            f"{label_dir} already holds results from a different protocol (previous, requested): {changed}. "
            "Use a different label, or overwrite=True to recompute."
        )


def pending_tasks(label_dir: str | Path, needed: list[str], overwrite: bool = False) -> list[str]:
    """Tasks of `needed` that have no result under `label_dir` yet (all of them when overwriting)."""
    if overwrite:
        return list(needed)
    have = set(collect_task_scores(label_dir))
    return [t for t in needed if t not in have]


def run_evaluation(
    model: str,
    output_dir: str | Path,
    revision: Optional[str] = None,
    label: Optional[str] = None,
    task_names: Optional[list[str]] = None,
    query_prompt: Optional[str] = None,
    passage_prompt: Optional[str] = None,
    dtype: str = "float32",
    max_seq_length: Optional[int] = 512,
    batch_size: int = 32,
    trust_remote_code: bool = False,
    overwrite: bool = False,
    verbosity: int = 1,
    reuse_from: Optional[str] = None,
    normalize: bool = False,
) -> Path:
    """Evaluate one model on TR-MTEB. Returns the label directory holding the results.

    `model` is a Hugging Face id or a local path (e.g. a trained student's `final/` directory).

    `reuse_from` (only with prompts): label of an earlier plain run of the SAME model/revision/
    dtype/max_seq_length in `output_dir`. Prompts change retrieval scores only, so the results of
    all non-retrieval tasks are copied from that run and only retrieval is evaluated.
    """
    if task_names:
        unknown = sorted(set(task_names) - set(TASK_CATEGORIES))
        if unknown:
            raise ValueError(f"unknown TR-MTEB tasks {unknown}; valid names: {sorted(TASK_CATEGORIES)}")
    dtypes = ("float32", "float16", "bfloat16")
    if dtype not in dtypes:
        raise ValueError(f"dtype must be one of {dtypes}, got {dtype!r}")
    prompted = query_prompt is not None or passage_prompt is not None
    if reuse_from and not prompted:
        raise ValueError("reuse_from only makes sense for a run with prompts (query_prompt/passage_prompt)")

    try:
        import mteb
        from mteb import MTEB
    except ImportError as exc:
        raise ImportError(f"mteb is not installed. Install the TR-MTEB fork with: {MTEB_TR_INSTALL}") from exc
    try:
        benchmark = mteb.get_benchmark(BENCHMARK_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"The installed mteb has no benchmark {BENCHMARK_NAME!r} (upstream mteb does not ship it). "
            f"Install the TR-MTEB fork: {MTEB_TR_INSTALL}"
        ) from exc

    import torch
    from sentence_transformers import SentenceTransformer

    label = label or sanitize_label(model, revision, "prompted" if prompted else "")
    label_dir = Path(output_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)

    # Explicit prompts only: the model's built-in prompt table is ignored so that every
    # model is evaluated under the same, recorded protocol.
    prompts: dict[str, str] = {}
    if query_prompt is not None:
        prompts["query"] = query_prompt
    if passage_prompt is not None:
        prompts["passage"] = passage_prompt

    check_label_protocol(
        label_dir,
        {"model": model, "revision": revision, "dtype": dtype, "max_seq_length": max_seq_length, "prompts": prompts,
         "normalize": normalize},
        overwrite,
    )

    # Prompts only reach retrieval tasks (queries vs passages), so the other tasks score exactly
    # the same as in a plain run of the same model: reuse them instead of recomputing.
    tasks_to_run: Optional[list[str]] = list(task_names) if task_names else None
    reused: list[str] = []
    if reuse_from:
        protocol = {"model": model, "revision": revision, "dtype": dtype, "max_seq_length": max_seq_length}
        reused, tasks_to_run = prepare_reuse(output_dir, label_dir, reuse_from, protocol, task_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = {
        "model": model,
        "revision": revision,
        "label": label,
        "tasks": task_names or "all",
        "prompts": prompts,
        "dtype": dtype,
        "max_seq_length": max_seq_length,
        "normalize": normalize,
        "batch_size": batch_size,
        "device": device,
        "reused_from": reuse_from,
        "reused_tasks": reused,
        "mteb_version": getattr(mteb, "__version__", None),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (label_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # Only load the model if some task still needs it: a finished (or fully reused) label costs nothing.
    needed = list(TASK_CATEGORIES) if tasks_to_run is None else list(tasks_to_run)
    todo = pending_tasks(label_dir, needed, overwrite)
    if not todo:
        print(f"[{label}] all {len(needed)} requested tasks already have results: nothing to evaluate", flush=True)
    else:
        if todo != needed:
            print(f"[{label}] {len(needed) - len(todo)} of {len(needed)} tasks already done, evaluating {len(todo)}", flush=True)
        tasks_to_run = todo
        st_model = SentenceTransformer(
            model,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
        )
        if max_seq_length:
            st_model.max_seq_length = max_seq_length
        if normalize:
            # A folder saved by `save_backbone` is loaded as transformer + mean pooling only, so its vectors are
            # not unit length; published embedding models (BGE-M3, mE5, Qwen3) ship a Normalize layer.
            from sentence_transformers.models import Normalize

            if not any(isinstance(module, Normalize) for module in st_model):
                st_model.append(Normalize())
        st_model.prompts = prompts
        st_model.default_prompt_name = None
        tasks = benchmark if len(tasks_to_run) == len(TASK_CATEGORIES) else mteb.get_tasks(tasks=tasks_to_run, languages=["tur"])
        try:
            run_with_write_retries(
                lambda: MTEB(tasks=tasks).run(
                    st_model,
                    output_folder=str(label_dir),
                    encode_kwargs={"batch_size": batch_size},
                    overwrite_results=overwrite,
                    verbosity=verbosity,
                ),
                label_dir,
            )
        finally:
            del st_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_dir(label_dir)
    summary["protocol"] = info
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return label_dir


def run_with_write_retries(run_once, label_dir: Path, attempts: int = 4, wait: float = 15.0, sleep=time.sleep):
    """Run `run_once()`; if a result file cannot be written under `label_dir`, recreate its folder and run it again.

    On a Colab Drive mount a folder that exists on Drive is sometimes not visible to the runtime for a while, and
    mteb then fails with FileNotFoundError when it writes a task result (the finished tasks keep their files, and
    mteb skips them on the next call, so a retry only redoes the task whose result was lost). Any other error, and a
    path outside `label_dir`, is raised at once.
    """
    label_dir = Path(label_dir).resolve()
    for attempt in range(1, attempts + 1):
        try:
            return run_once()
        except OSError as exc:
            target = Path(exc.filename).resolve() if getattr(exc, "filename", None) else None
            if target is None or label_dir not in target.parents or attempt == attempts:
                raise
            print(
                f"[{label_dir.name}] writing {target.name} failed ({type(exc).__name__}); "
                f"recreating its folder and retrying ({attempt}/{attempts - 1}) in {wait * attempt:.0f} s",
                flush=True,
            )
            sleep(wait * attempt)
            target.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_run(args: argparse.Namespace) -> None:
    label_dir = run_evaluation(
        model=args.model,
        output_dir=args.output_dir,
        revision=args.revision,
        label=args.label,
        task_names=args.tasks,
        query_prompt=_unescape(args.query_prompt),
        passage_prompt=_unescape(args.passage_prompt),
        dtype=args.dtype,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        overwrite=args.overwrite,
        reuse_from=args.reuse_from,
    )
    print(format_table({label_dir.name: summarize_dir(label_dir)}))
    print(f"results: {label_dir}")


def _cmd_summarize(args: argparse.Namespace) -> None:
    summaries = {Path(d).name: summarize_dir(d) for d in args.dirs}
    print(format_table(summaries))
    print()
    print(format_group_table(summaries))
    for label, summary in summaries.items():
        if summary["missing_tasks"]:
            print(f"\n{label}: missing tasks: {', '.join(summary['missing_tasks'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TR-MTEB evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate one model")
    run.add_argument("--model", required=True, help="HF id or local path")
    run.add_argument("--revision", default=None, help="HF commit hash (recommended)")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--label", default=None, help="results folder name (default: derived from model/revision)")
    run.add_argument("--tasks", nargs="+", default=None, help="subset of TR-MTEB task names (default: all 26)")
    run.add_argument("--query-prompt", default=None, help="prefix for queries; '\\n' is turned into a newline")
    run.add_argument("--passage-prompt", default=None, help="prefix for documents")
    run.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    run.add_argument("--max-seq-length", type=int, default=512)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--overwrite", action="store_true", help="recompute tasks that already have results")
    run.add_argument(
        "--reuse-from",
        default=None,
        help="label of an earlier PLAIN run of the same model: copy its non-retrieval results, evaluate only retrieval "
        "(requires --query-prompt/--passage-prompt)",
    )
    run.set_defaults(func=_cmd_run)

    summ = sub.add_parser("summarize", help="compare finished result folders")
    summ.add_argument("dirs", nargs="+", help="label directories produced by `run`")
    summ.set_defaults(func=_cmd_summarize)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
