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


def test_task_groups_partition_the_benchmark():
    grouped = [t for tasks in trmteb.TASK_GROUPS.values() for t in tasks]
    assert sorted(grouped) == sorted(TASK_CATEGORIES)  # every task in exactly one group


def test_group_means_and_incomplete_groups(tmp_path):
    write_all(tmp_path, lambda t: 0.5 if t in trmteb.TASK_GROUPS["NLI"] else 0.8)
    s = summarize_dir(tmp_path)
    assert s["groups"]["NLI"] == pytest.approx(50.0) and s["groups"]["STS"] == pytest.approx(80.0)
    partial = summarize_scores({"STSbTR": {"score": 80.0}, "MnliTr": {"score": 60.0}})
    assert partial["groups"]["STS"] == pytest.approx(80.0)
    assert partial["groups"]["NLI"] is None  # SnliTr and XNLI are missing
    table = trmteb.format_group_table({"m": s, "p": partial})
    assert "NLI" in table and "50.00" in table


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


# --- reuse of prompt-insensitive results ---------------------------------------------------

PLAIN_INFO = {"model": "x/y", "revision": "abc", "dtype": "float16", "max_seq_length": 512, "prompts": {}}


def make_plain_run(root, label="plain", info=None, skip=()):
    run = root / label
    write_all_except(run / "inner" / "rev", skip)
    (run / "run_info.json").write_text(json.dumps(info or PLAIN_INFO), encoding="utf-8")
    return run


def write_all_except(directory, skip=()):
    for task in TASK_CATEGORIES:
        if task not in skip:
            write_result(directory, task, 0.5)


def test_prepare_reuse_copies_only_non_retrieval_tasks(tmp_path):
    make_plain_run(tmp_path)
    label_dir = tmp_path / "prompted"
    protocol = {k: PLAIN_INFO[k] for k in ("model", "revision", "dtype", "max_seq_length")}
    reused, remaining = trmteb.prepare_reuse(tmp_path, label_dir, "plain", protocol)
    assert len(reused) == 15 and len(remaining) == 11
    assert all(TASK_CATEGORIES[t] != "Retrieval" for t in reused)
    assert all(TASK_CATEGORIES[t] == "Retrieval" for t in remaining)
    assert sorted(p.stem for p in (label_dir / "reused").glob("*.json")) == sorted(reused)
    # the reused files are picked up by the summary together with newly evaluated tasks
    for task in remaining:
        write_result(label_dir / "inner" / "rev", task, 0.7)
    s = summarize_dir(label_dir)
    assert s["complete"] and s["num_tasks"] == 26
    assert s["categories"]["Retrieval"] == pytest.approx(70.0) and s["categories"]["STS"] == pytest.approx(50.0)


def test_prepare_reuse_respects_task_subset_and_missing_source_tasks(tmp_path):
    make_plain_run(tmp_path, skip=("STSbTR",))
    protocol = {k: PLAIN_INFO[k] for k in ("model", "revision", "dtype", "max_seq_length")}
    reused, remaining = trmteb.prepare_reuse(
        tmp_path, tmp_path / "p", "plain", protocol, task_names=["STSbTR", "MnliTr", "NFCorpusTR"]
    )
    assert reused == ["MnliTr"]
    assert remaining == ["STSbTR", "NFCorpusTR"]  # missing in the source -> evaluated normally; retrieval -> evaluated


def test_prepare_reuse_rejects_a_different_protocol_or_missing_or_prompted_source(tmp_path):
    protocol = {k: PLAIN_INFO[k] for k in ("model", "revision", "dtype", "max_seq_length")}
    with pytest.raises(FileNotFoundError, match="plain evaluation first"):
        trmteb.prepare_reuse(tmp_path, tmp_path / "p", "nope", protocol)
    make_plain_run(tmp_path, "other_dtype", info={**PLAIN_INFO, "dtype": "float32"})
    with pytest.raises(ValueError, match="different protocol"):
        trmteb.prepare_reuse(tmp_path, tmp_path / "p", "other_dtype", protocol)
    make_plain_run(tmp_path, "other_model", info={**PLAIN_INFO, "model": "a/b"})
    with pytest.raises(ValueError, match="different protocol"):
        trmteb.prepare_reuse(tmp_path, tmp_path / "p", "other_model", protocol)
    make_plain_run(tmp_path, "was_prompted", info={**PLAIN_INFO, "prompts": {"query": "q: "}})
    with pytest.raises(ValueError, match="plain run"):
        trmteb.prepare_reuse(tmp_path, tmp_path / "p", "was_prompted", protocol)


def test_run_evaluation_requires_prompts_for_reuse():
    with pytest.raises(ValueError, match="only makes sense"):
        trmteb.run_evaluation("x/y", "out", reuse_from="plain")


# --- running an evaluation in its own process -----------------------------------------------

def test_isolated_run_reports_failure_and_prints_the_childs_error(tmp_path, capsys):
    # mteb is not installed in the test environment, so the child fails with the ImportError.
    with pytest.raises(RuntimeError, match="failed in its own process"):
        trmteb.run_evaluation_isolated(label="x", output_dir=tmp_path, model="a/b")
    assert "mteb_tr" in capsys.readouterr().out


def test_isolated_run_streams_output_and_returns_the_label_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        trmteb, "_CHILD_CODE",
        "import json, sys\nkw = json.loads(sys.stdin.read())\nprint('child got', kw['model'], kw['label'], flush=True)\n",
    )
    out = trmteb.run_evaluation_isolated(label="lbl", output_dir=tmp_path, model="a/b")
    assert out == tmp_path / "lbl"
    assert "child got a/b lbl" in capsys.readouterr().out


def test_one_failing_child_does_not_affect_the_next_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        trmteb, "_CHILD_CODE",
        "import json, sys\nkw = json.loads(sys.stdin.read())\nsys.exit(3 if kw['model'] == 'bad' else 0)\n",
    )
    with pytest.raises(RuntimeError, match="exit code 3"):
        trmteb.run_evaluation_isolated(label="a", output_dir=tmp_path, model="bad")
    assert trmteb.run_evaluation_isolated(label="b", output_dir=tmp_path, model="good") == tmp_path / "b"


# --- skipping finished work, protecting a label from protocol mixing --------------------------

def test_pending_tasks(tmp_path):
    write_result(tmp_path, "STSbTR", 0.5)
    write_result(tmp_path, "MnliTr", 0.5)
    assert trmteb.pending_tasks(tmp_path, ["STSbTR", "MnliTr", "SciFactTR"]) == ["SciFactTR"]
    assert trmteb.pending_tasks(tmp_path, ["STSbTR"]) == []
    assert trmteb.pending_tasks(tmp_path, ["STSbTR", "MnliTr"], overwrite=True) == ["STSbTR", "MnliTr"]


def test_check_label_protocol(tmp_path):
    proto = {"model": "a/b", "revision": "r", "dtype": "float16", "max_seq_length": 512, "prompts": {}}
    trmteb.check_label_protocol(tmp_path, proto)  # no run_info yet: fine
    (tmp_path / "run_info.json").write_text(json.dumps({**proto, "batch_size": 8}), encoding="utf-8")
    trmteb.check_label_protocol(tmp_path, {**proto})  # same protocol (batch size is not part of it)
    for key, new in (("dtype", "float32"), ("max_seq_length", 256), ("revision", "other"), ("prompts", {"query": "q: "})):
        with pytest.raises(ValueError, match="different protocol"):
            trmteb.check_label_protocol(tmp_path, {**proto, key: new})
    trmteb.check_label_protocol(tmp_path, {**proto, "dtype": "float32"}, overwrite=True)  # explicit override


class _FakeEnv:
    """Fake mteb + sentence_transformers so run_evaluation's control flow can be tested on CPU."""

    def __init__(self, monkeypatch):
        import types

        self.loaded: list[str] = []
        self.ran: list[list[str]] = []
        env = self

        class FakeBenchmark:
            tasks = list(TASK_CATEGORIES)

        class FakeMTEB:
            def __init__(self, tasks):
                self.tasks = tasks

            def run(self, model, output_folder, **kwargs):
                names = [getattr(t, "name", t) for t in (self.tasks.tasks if hasattr(self.tasks, "tasks") else self.tasks)]
                env.ran.append(names)
                for name in names:
                    write_result(Path(output_folder) / "inner" / "rev", name, 0.6)

        class Normalize:  # stands for sentence_transformers.models.Normalize
            pass

        env.Normalize = Normalize
        env.initial_modules: list = []  # modules a "loaded" model already has (set by a test)
        env.model = None

        class FakeST(list):  # a SentenceTransformer is a sequence of modules
            def __init__(self, name, **kwargs):
                super().__init__(env.initial_modules)
                env.loaded.append(name)
                env.model = self

        fake_models = types.ModuleType("sentence_transformers.models")
        fake_models.Normalize = Normalize
        monkeypatch.setitem(sys.modules, "sentence_transformers.models", fake_models)
        fake_mteb = types.ModuleType("mteb")
        fake_mteb.get_benchmark = lambda n: FakeBenchmark()
        fake_mteb.get_tasks = lambda tasks, languages=None: [types.SimpleNamespace(name=t) for t in tasks]
        fake_mteb.MTEB = FakeMTEB
        fake_mteb.__version__ = "fake"
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = FakeST
        monkeypatch.setitem(sys.modules, "mteb", fake_mteb)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)


from pathlib import Path  # noqa: E402  (used by the fake above)


def test_finished_label_does_not_load_the_model_again(tmp_path, monkeypatch, capsys):
    env = _FakeEnv(monkeypatch)
    trmteb.run_evaluation("a/b", tmp_path, revision="r", label="m", dtype="float32")
    assert env.loaded == ["a/b"] and len(env.ran) == 1 and len(env.ran[0]) == 26
    out = trmteb.run_evaluation("a/b", tmp_path, revision="r", label="m", dtype="float32")
    assert env.loaded == ["a/b"], "the model must not be loaded a second time"
    assert len(env.ran) == 1
    assert "nothing to evaluate" in capsys.readouterr().out
    assert (out / "summary.json").exists()


def test_partial_label_only_evaluates_the_missing_tasks(tmp_path, monkeypatch):
    env = _FakeEnv(monkeypatch)
    label_dir = tmp_path / "m"
    for task in list(TASK_CATEGORIES)[:20]:
        write_result(label_dir / "inner" / "rev", task, 0.5)
    trmteb.run_evaluation("a/b", tmp_path, label="m")
    assert len(env.loaded) == 1
    assert sorted(env.ran[0]) == sorted(list(TASK_CATEGORIES)[20:])


def test_prompted_label_reuses_plain_results_and_loads_nothing_when_all_present(tmp_path, monkeypatch):
    env = _FakeEnv(monkeypatch)
    plain = {"model": "a/b", "revision": None, "dtype": "float32", "max_seq_length": 512, "prompts": {}}
    run = tmp_path / "plain"
    write_all_except(run / "inner" / "rev")
    (run / "run_info.json").write_text(json.dumps(plain), encoding="utf-8")
    trmteb.run_evaluation("a/b", tmp_path, label="prompted", query_prompt="q: ", reuse_from="plain")
    assert env.loaded == ["a/b"]
    assert len(env.ran) == 1 and all(TASK_CATEGORIES[t] == "Retrieval" for t in env.ran[0]) and len(env.ran[0]) == 11
    # second call: retrieval is now present too, so no model load at all
    trmteb.run_evaluation("a/b", tmp_path, label="prompted", query_prompt="q: ", reuse_from="plain")
    assert env.loaded == ["a/b"] and len(env.ran) == 1


def test_changing_dtype_for_an_existing_label_is_refused(tmp_path, monkeypatch):
    _FakeEnv(monkeypatch)
    trmteb.run_evaluation("a/b", tmp_path, label="m", dtype="float16")
    with pytest.raises(ValueError, match="different protocol"):
        trmteb.run_evaluation("a/b", tmp_path, label="m", dtype="float32")


def test_normalize_adds_a_normalize_layer_once_and_is_part_of_the_protocol(tmp_path, monkeypatch):
    env = _FakeEnv(monkeypatch)
    trmteb.run_evaluation("a/b", tmp_path, label="plain", normalize=False)
    assert not any(isinstance(m, env.Normalize) for m in env.model)  # a trained student is left as it is by default

    trmteb.run_evaluation("a/b", tmp_path, label="norm", normalize=True)
    assert [type(m) for m in env.model] == [env.Normalize]
    assert json.loads((tmp_path / "norm" / "run_info.json").read_text())["normalize"] is True

    env.initial_modules = [env.Normalize()]  # a packaged model that already normalises
    trmteb.run_evaluation("c/d", tmp_path, label="packaged", normalize=True)
    assert sum(isinstance(m, env.Normalize) for m in env.model) == 1

    with pytest.raises(ValueError):  # results with and without normalisation must never share a label
        trmteb.run_evaluation("a/b", tmp_path, label="plain", normalize=True)


def test_scale_sensitive_tasks_are_the_classification_and_clustering_tasks():
    assert len(trmteb.SCALE_SENSITIVE_TASKS) == 10
    assert {TASK_CATEGORIES[t] for t in trmteb.SCALE_SENSITIVE_TASKS} == {"Classification", "Clustering"}
