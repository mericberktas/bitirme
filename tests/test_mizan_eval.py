"""Mizan evaluation: task lists and the summaries (no mteb needed)."""

import json

import pytest

from turkish_distill.eval import mizan


def test_task_lists_split_the_fourteen_tasks_into_nine_independent_and_five_shared():
    assert len(mizan.TASK_CATEGORIES) == 14 and len(mizan.INDEPENDENT_TASKS) == 9
    assert set(mizan.INDEPENDENT_TASKS) | set(mizan.SHARED_WITH_TRMTEB) == set(mizan.TASK_CATEGORIES)
    assert not set(mizan.INDEPENDENT_TASKS) & set(mizan.SHARED_WITH_TRMTEB)
    from turkish_distill.eval import trmteb

    shared_in_trmteb = {t for t in mizan.TASK_CATEGORIES if t in trmteb.TASK_CATEGORIES}
    # the tasks Mizan shares with TR-MTEB by name (TurHistQuAD is TR-MTEB's TQuAD under another name)
    assert shared_in_trmteb == set(mizan.SHARED_WITH_TRMTEB) - {"TurHistQuadRetrieval"}
    from turkish_distill.benchmark_texts import MIZAN_FILES

    assert {"mteb/sib200", "PaDaS-Lab/webfaq-retrieval", "mteb/sts22-crosslingual-sts"} <= set(MIZAN_FILES)


def fake_scores(value, missing=()):
    return {t: {"score": value + i, "split": "test", "file": "x"} for i, t in enumerate(mizan.TASK_CATEGORIES) if t not in missing}


def test_summaries_report_the_leaderboard_score_and_the_independent_mean():
    s = mizan.summarize_scores(fake_scores(50.0))
    assert s["complete"] and s["num_tasks"] == 14
    per_task = s["per_task"]
    assert s["mean_independent"] == pytest.approx(sum(per_task[t] for t in mizan.INDEPENDENT_TASKS) / 9)
    assert s["mean_shared"] == pytest.approx(sum(per_task[t] for t in mizan.SHARED_WITH_TRMTEB) / 5)
    cats = {c: [per_task[t] for t in per_task if mizan.TASK_CATEGORIES[t] == c] for c in mizan.CATEGORY_ORDER}
    assert s["mean_type"] == pytest.approx(sum(sum(v) / len(v) for v in cats.values()) / 5)  # mean of the task-type means
    partial = mizan.summarize_scores(fake_scores(50.0, missing=("STS22.v2",)))
    assert not partial["complete"] and partial["mean_independent"] is None and partial["missing_tasks"] == ["STS22.v2"]
    assert "INCOMPLETE" in mizan.format_table({"m": partial}) and f"{s['mean_task']:.2f}" in mizan.format_table({"m": s})


def test_results_are_read_like_trmteb_results(tmp_path):
    result = {"task_name": "XNLIV2", "scores": {"test": [{"main_score": 0.61, "hf_subset": "tr"}]}}
    (tmp_path / "XNLIV2.json").write_text(json.dumps(result), encoding="utf-8")
    assert mizan.summarize_dir(tmp_path)["per_task"] == {"XNLIV2": pytest.approx(61.0)}


def test_run_evaluation_validates_its_arguments_before_importing_mteb(tmp_path):
    with pytest.raises(ValueError, match="unknown Mizan tasks"):
        mizan.run_evaluation("some/model", tmp_path, task_names=["NotATask"])
    with pytest.raises(ValueError, match="dtype"):
        mizan.run_evaluation("some/model", tmp_path, dtype="int8")
