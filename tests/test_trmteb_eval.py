import json
import os
import sys
from collections import Counter

import pytest

from turkish_distill.eval import trmteb
from turkish_distill.eval.trmteb import (
    TASK_CATEGORIES,
    collect_task_scores,
    format_table,
    sanitize_label,
    summarize_dir,
    summarize_scores,
    task_score_from_result,
)


def write_result(directory, task, main_score, split="test", subsets=None, name=None):
    directory.mkdir(parents=True, exist_ok=True)
    entries = [{"main_score": s, "hf_subset": f"s{i}", "languages": ["tur-Latn"]} for i, s in enumerate(subsets or [main_score])]
    data = {"task_name": task, "scores": {split: entries}, "evaluation_time": 1.0}
    (directory / (name or f"{task}.json")).write_text(json.dumps(data), encoding="utf-8")


def write_all(directory, score_of):
    for task in TASK_CATEGORIES:
        write_result(directory, task, score_of(task))


def test_task_registry_matches_the_paper():
    assert len(TASK_CATEGORIES) == 26
    assert Counter(TASK_CATEGORIES.values()) == {
        "Retrieval": 11, "Classification": 8, "PairClassification": 3,
        "Clustering": 2, "STS": 1, "BitextMining": 1,
    }


def test_sanitize_label():
    assert sanitize_label("BAAI/bge-m3") == "BAAI__bge-m3"
    assert sanitize_label("BAAI/bge-m3", "5617a9f61b028005") == "BAAI__bge-m3_5617a9f6"
    assert sanitize_label("a/b", "abcdef012345", "prompted").endswith("_abcdef01_prompted")


def test_score_uses_test_split_and_averages_subsets():
    data = {"task_name": "X", "scores": {"validation": [{"main_score": 0.1}], "test": [{"main_score": 0.2}, {"main_score": 0.4}]}}
    score, split = task_score_from_result(data)
    assert split == "test" and score == pytest.approx(30.0)
    only_val = {"task_name": "X", "scores": {"validation": [{"main_score": 0.5}]}}
    assert task_score_from_result(only_val) == (pytest.approx(50.0), "validation")
    assert task_score_from_result({"task_name": "X", "scores": {}}) is None
    assert task_score_from_result({"task_name": "X", "scores": {"test": [{"foo": 1}]}}) is None


def test_summary_means_match_hand_computation(tmp_path):
    # Retrieval 0.6, Classification 0.8, Clustering 0.5, Pair 0.7, STS 0.9, Bitext 1.0
    level = {"Retrieval": 0.6, "Classification": 0.8, "Clustering": 0.5, "PairClassification": 0.7, "STS": 0.9, "BitextMining": 1.0}
    write_all(tmp_path, lambda t: level[TASK_CATEGORIES[t]])
    s = summarize_dir(tmp_path)
    assert s["complete"] and s["num_tasks"] == 26 and s["missing_tasks"] == []
    expected_task = (11 * 60 + 8 * 80 + 2 * 50 + 3 * 70 + 1 * 90 + 1 * 100) / 26
    assert s["mean_task"] == pytest.approx(expected_task)
    assert s["mean_task_no_bitext"] == pytest.approx((11 * 60 + 8 * 80 + 2 * 50 + 3 * 70 + 90) / 25)
    assert s["mean_type"] == pytest.approx((60 + 80 + 50 + 70 + 90 + 100) / 6)
    assert s["mean_type_no_bitext"] == pytest.approx((60 + 80 + 50 + 70 + 90) / 5)
    assert s["categories"]["Retrieval"] == pytest.approx(60.0)


def test_incomplete_results_are_flagged(tmp_path):
    write_result(tmp_path, "STSbTR", 0.8)
    write_result(tmp_path, "NFCorpusTR", 0.3)
    s = summarize_dir(tmp_path)
    assert not s["complete"] and s["num_tasks"] == 2
    assert "SciFactTR" in s["missing_tasks"] and len(s["missing_tasks"]) == 24
    assert "INCOMPLETE" in format_table({"m": s})


def test_unrelated_and_meta_files_are_ignored_and_newest_wins(tmp_path):
    write_result(tmp_path / "model" / "rev", "STSbTR", 0.5)
    (tmp_path / "model" / "rev" / "model_meta.json").write_text(json.dumps({"name": "x"}), encoding="utf-8")
    (tmp_path / "summary.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "other.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
    write_result(tmp_path / "model" / "rev2", "STSbTR", 0.9)
    old = tmp_path / "model" / "rev" / "STSbTR.json"
    os.utime(old, (1, 1))  # make the first result clearly older
    scores = collect_task_scores(tmp_path)
    assert list(scores) == ["STSbTR"] and scores["STSbTR"]["score"] == pytest.approx(90.0)


def test_non_benchmark_tasks_do_not_pollute_means():
    s = summarize_scores({"STSbTR": {"score": 80.0}, "SomeOtherTask": {"score": 10.0}})
    assert s["num_tasks"] == 1 and s["mean_task"] == pytest.approx(80.0)


def test_format_table_and_cli_summarize(tmp_path, capsys):
    a, b = tmp_path / "model_a", tmp_path / "model_b"
    write_all(a, lambda t: 0.5)
    write_all(b, lambda t: 0.7)
    trmteb.main(["summarize", str(a), str(b)])
    out = capsys.readouterr().out
    assert "model_a" in out and "model_b" in out and "50.00" in out and "70.00" in out and "26/26" in out


def test_run_validates_task_names_before_touching_mteb():
    with pytest.raises(ValueError, match="unknown TR-MTEB tasks"):
        trmteb.run_evaluation("x/y", "out", task_names=["NotATask"])
    with pytest.raises(ValueError, match="dtype"):
        trmteb.run_evaluation("x/y", "out", dtype="int8")


def test_run_explains_how_to_install_mteb_tr(monkeypatch):
    monkeypatch.setitem(sys.modules, "mteb", None)  # makes `import mteb` raise ImportError
    with pytest.raises(ImportError, match="mteb_tr"):
        trmteb.run_evaluation("x/y", "out")


def test_prompt_escape_handling():
    assert trmteb._unescape("Instruct: a\\nQuery: ") == "Instruct: a\nQuery: "
    assert trmteb._unescape(None) is None
