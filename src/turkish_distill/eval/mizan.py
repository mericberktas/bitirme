"""Evaluation on the Turkish tasks of the Mizan leaderboard, with stock `mteb` 1.38.x (our secondary benchmark).

Why a second benchmark: every decision of this project (data mix, teacher combination, instruction, centring) was
made by looking at TR-MTEB, which makes TR-MTEB a poor judge of the final result (selection bias). Mizan
(https://newmindai-mizan.hf.space, 14 Turkish tasks, MTEB v1.38.51) shares only five of its tasks with TR-MTEB, so
the other nine are a development set that TR-MTEB-based choices have not touched.

    pip install --ignore-requires-python "mteb==1.38.51" "datasets<4"      # see the notebook: 1.38 asks for Python < 3.13

Differences to `eval.trmteb` (which uses the TR-MTEB fork, mteb 1.31.3):
* the tasks come from the stock mteb release the leaderboard uses, restricted to Turkish (`languages=["tur"]`);
* "MTEB Score" of the leaderboard is the mean over the task types (here `mean_type`); `mean_task` is also given;
* the nine tasks not shared with TR-MTEB are summarised on their own (`mean_independent`): THE development set;
* the protocol is plain encoding (no prompts), like our TR-MTEB runs; the leaderboard's own protocol (prompts,
  sequence length) is not documented on its page, so absolute numbers may differ slightly from the leaderboard's.
Everything else (labels, protocol check, resuming, isolated processes) is shared with `eval.trmteb`.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .trmteb import _fmt, _mean, check_label_protocol, collect_task_scores, pending_tasks, run_with_write_retries, sanitize_label

MIZAN_MTEB_VERSION = "1.38.51"
MTEB_INSTALL = 'pip install --ignore-requires-python "mteb==1.38.51" "datasets<4"'

TASK_CATEGORIES: dict[str, str] = {
    "MassiveIntentClassification": "Classification",
    "MassiveScenarioClassification": "Classification",
    "MultilingualSentimentClassification": "Classification",
    "SIB200Classification": "Classification",
    "TurkishMovieSentimentClassification": "Classification",
    "TurkishProductSentimentClassification": "Classification",
    "WebFAQRetrieval": "Retrieval",
    "XQuADRetrieval": "Retrieval",
    "TurHistQuadRetrieval": "Retrieval",
    "MKQARetrieval": "Retrieval",
    "XNLI": "PairClassification",
    "XNLIV2": "PairClassification",
    "SIB200ClusteringS2S": "Clustering",
    "STS22.v2": "STS",
}
# Also in TR-MTEB (TurHistQuAD is the same data as TR-MTEB's TQuAD): reported for the leaderboard, not for decisions.
SHARED_WITH_TRMTEB = [
    "TurkishMovieSentimentClassification",
    "TurkishProductSentimentClassification",
    "XQuADRetrieval",
    "XNLI",
    "TurHistQuadRetrieval",
]
# The development set. XNLIV2 re-translates the XNLI pairs, so it is only partly independent of TR-MTEB's XNLI.
INDEPENDENT_TASKS = [t for t in TASK_CATEGORIES if t not in SHARED_WITH_TRMTEB]
CATEGORY_ORDER = ["Classification", "Retrieval", "PairClassification", "Clustering", "STS"]
_RUN_PROTOCOL = ("model", "revision", "dtype", "max_seq_length", "prompts", "normalize")


def patch_transformers_for_mteb(module=None) -> bool:
    """Let `import mteb` (1.38.x) work with transformers 5.

    One of mteb's model modules does `from transformers import AutoModelForVision2Seq`; transformers 5 removed that name
    (it was merged into `AutoModelForImageTextToText`), which makes `import mteb` fail with an ImportError before any
    task is touched. It is the only transformers name of the 15 that mteb 1.38.51 imports that is missing in
    transformers 5.20.1 (checked against the source of both). Returns True when the alias was added. Call it BEFORE
    `import mteb`; `run_evaluation` does.

    The alias must be set AFTER `sentence_transformers` has been imported: importing it makes transformers 5 replace
    its own module object in `sys.modules` (measured in the Mizan environment: the module id changes while
    `transformers.video_processing_utils` loads), which silently drops an alias set earlier, and `import mteb`
    (which imports sentence_transformers) then fails with the same ImportError although the patch reported success.
    So the function imports sentence_transformers first (mteb needs it anyway)."""
    if module is None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pass
        import transformers as module
    if hasattr(module, "AutoModelForVision2Seq"):
        return False
    replacement = getattr(module, "AutoModelForImageTextToText", None)
    if replacement is None:
        return False
    module.AutoModelForVision2Seq = replacement
    return True


def patch_torchvision_for_datasets(io_module=None) -> bool:
    """Keep `datasets` 3.x working with a torchvision that no longer has `torchvision.io.VideoReader`.

    While `datasets` converts a column to numpy (`with_format("numpy")`, which `train_test_split(stratify_by_column=...)`
    does) its formatter runs `from torchvision.io import VideoReader` whenever torchvision has been imported. Newer
    torchvision builds removed that class, so the very first stratified subsampling fails with `ImportError: cannot
    import name 'VideoReader'` (seen on Colab in the Mizan run: TurkishMovieSentimentClassification). The formatter only
    uses the name for an `isinstance` check, so a placeholder class is enough. Returns True when the placeholder was
    added (False when torchvision is not installed or already has the class)."""
    if io_module is None:
        import importlib.util

        if importlib.util.find_spec("torchvision") is None:
            return False
        try:
            import torchvision.io as io_module
        except Exception:
            return False
    if hasattr(io_module, "VideoReader"):
        return False
    io_module.VideoReader = type("VideoReader", (), {})
    return True


def allow_long_http_headers(session_cls=None, limit: int = 65536) -> bool:
    """Raise aiohttp's default limit on response header size (8190 bytes) for every new `ClientSession`.

    `apple/mkqa` is a loading script that downloads `mkqa.jsonl.gz` from github.com; GitHub's redirect answers with a
    Content-Security-Policy header longer than 8190 bytes, aiohttp refuses it ("Got more than 8190 bytes") and
    `MKQARetrieval.load_data` raises FileNotFoundError. mteb lets that exception end the whole run, so every task after
    MKQARetrieval (XNLI, XNLIV2, SIB200 clustering, STS22.v2) is skipped. datasets passes its own `client_kwargs` to
    fsspec, which replaces an fsspec-level config, so the default is set on the session class itself. Returns True when
    the patch was installed (False when aiohttp is missing or it was installed before)."""
    if session_cls is None:
        try:
            import aiohttp
        except ImportError:
            return False
        session_cls = aiohttp.ClientSession
    init = session_cls.__init__
    if getattr(init, "_long_headers_patch", False):
        return False

    def patched(self, *args, **kwargs):
        kwargs.setdefault("max_field_size", limit)
        kwargs.setdefault("max_line_size", limit)
        init(self, *args, **kwargs)

    patched._long_headers_patch = True
    session_cls.__init__ = patched
    return True


def summarize_scores(task_scores: dict[str, dict]) -> dict:
    known = {t: v["score"] for t, v in task_scores.items() if t in TASK_CATEGORIES}
    by_category: dict[str, list[float]] = {c: [] for c in CATEGORY_ORDER}
    for task, score in known.items():
        by_category[TASK_CATEGORIES[task]].append(score)
    category_means = {c: _mean(v) for c, v in by_category.items() if v}
    independent = [known[t] for t in INDEPENDENT_TASKS if t in known]
    shared = [known[t] for t in SHARED_WITH_TRMTEB if t in known]
    missing = sorted(set(TASK_CATEGORIES) - set(known))
    return {
        "num_tasks": len(known),
        "complete": not missing,
        "missing_tasks": missing,
        "mean_task": _mean(list(known.values())),
        "mean_type": _mean(list(category_means.values())),  # the leaderboard's "MTEB Score"
        # a mean is reported only when all its tasks are present (otherwise it is not comparable)
        "mean_independent": _mean(independent) if len(independent) == len(INDEPENDENT_TASKS) else None,
        "mean_shared": _mean(shared) if len(shared) == len(SHARED_WITH_TRMTEB) else None,
        "categories": category_means,
        "per_task": {t: known[t] for t in sorted(known)},
    }


def summarize_dir(results_dir: str | Path) -> dict:
    return summarize_scores(collect_task_scores(results_dir))


def format_table(summaries: dict[str, dict]) -> str:
    columns = [
        ("Indep.(9)", lambda s: s["mean_independent"]),
        ("MTEB Score", lambda s: s["mean_type"]),
        ("Mean(task)", lambda s: s["mean_task"]),
        ("Shared(5)", lambda s: s["mean_shared"]),
        ("Clf", lambda s: s["categories"].get("Classification")),
        ("Retr", lambda s: s["categories"].get("Retrieval")),
        ("Pair", lambda s: s["categories"].get("PairClassification")),
        ("Clust", lambda s: s["categories"].get("Clustering")),
        ("STS", lambda s: s["categories"].get("STS")),
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


_CHILD_CODE = (
    "import json, sys\n"
    "from turkish_distill.eval.mizan import run_evaluation\n"
    "run_evaluation(**json.loads(sys.stdin.read()))\n"
)


def run_evaluation_isolated(label: str, output_dir: str | Path, **kwargs) -> Path:
    """`run_evaluation` in its own Python process (a CUDA error would otherwise poison every later model)."""
    payload = json.dumps({"label": label, "output_dir": str(output_dir), **kwargs})
    proc = subprocess.Popen([sys.executable, "-c", _CHILD_CODE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(payload.encode("utf-8"))
    proc.stdin.close()
    while True:
        chunk = proc.stdout.read1(4096)
        if not chunk:
            break
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Mizan evaluation of {label!r} failed in its own process (exit code {code}); see the output above")
    return Path(output_dir) / label


def run_evaluation(
    model: str,
    output_dir: str | Path,
    revision: Optional[str] = None,
    label: Optional[str] = None,
    task_names: Optional[list[str]] = None,
    dtype: str = "float32",
    max_seq_length: Optional[int] = 512,
    batch_size: int = 32,
    trust_remote_code: bool = False,
    overwrite: bool = False,
    verbosity: int = 1,
    normalize: bool = False,
) -> Path:
    """Evaluate one model on the Mizan Turkish tasks (all 14, or `task_names`). Plain encoding, no prompts.

    `model` is a Hugging Face id or a local path (e.g. a trained student's `final/` directory)."""
    if task_names:
        unknown = sorted(set(task_names) - set(TASK_CATEGORIES))
        if unknown:
            raise ValueError(f"unknown Mizan tasks {unknown}; valid names: {sorted(TASK_CATEGORIES)}")
    dtypes = ("float32", "float16", "bfloat16")
    if dtype not in dtypes:
        raise ValueError(f"dtype must be one of {dtypes}, got {dtype!r}")
    patch_transformers_for_mteb()
    allow_long_http_headers()
    patch_torchvision_for_datasets()
    try:
        import mteb
        from mteb import MTEB
    except ImportError as exc:
        raise ImportError(f"mteb is not installed. Install the Mizan version with: {MTEB_INSTALL}") from exc
    version = getattr(mteb, "__version__", "?")
    if not str(version).startswith("1.38"):
        print(f"WARNING: mteb {version} is installed, Mizan uses {MIZAN_MTEB_VERSION}; scores may differ ({MTEB_INSTALL})", flush=True)

    import torch
    from sentence_transformers import SentenceTransformer

    label = label or sanitize_label(model, revision)
    label_dir = Path(output_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"model": model, "revision": revision, "dtype": dtype, "max_seq_length": max_seq_length, "prompts": {}, "normalize": normalize}
    check_label_protocol(label_dir, protocol, overwrite)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    needed = list(task_names) if task_names else list(TASK_CATEGORIES)
    info = {**protocol, "label": label, "tasks": task_names or "all", "batch_size": batch_size, "device": device,
            "mteb_version": version, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (label_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    todo = pending_tasks(label_dir, needed, overwrite)
    if not todo:
        print(f"[{label}] all {len(needed)} requested tasks already have results: nothing to evaluate", flush=True)
    else:
        if todo != needed:
            print(f"[{label}] {len(needed) - len(todo)} of {len(needed)} tasks already done, evaluating {len(todo)}", flush=True)
        st_model = SentenceTransformer(
            model, revision=revision, device=device, trust_remote_code=trust_remote_code,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
        )
        if max_seq_length:
            st_model.max_seq_length = max_seq_length
        if normalize:
            from sentence_transformers.models import Normalize

            if not any(isinstance(module, Normalize) for module in st_model):
                st_model.append(Normalize())
        st_model.prompts = {}
        st_model.default_prompt_name = None
        tasks = mteb.get_tasks(tasks=todo, languages=["tur"])
        try:
            run_with_write_retries(
                lambda: MTEB(tasks=tasks).run(
                    st_model, output_folder=str(label_dir), encode_kwargs={"batch_size": batch_size},
                    overwrite_results=overwrite, verbosity=verbosity,
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
