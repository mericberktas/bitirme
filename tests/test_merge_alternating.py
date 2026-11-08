"""SLERP and TIES merging, and the alternating train-and-merge procedure (with a fake trainer)."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

from turkish_distill.alternating import arm_overrides, find_run, run_alternating, run_name
from turkish_distill.config import dump_config
from turkish_distill.soup import merge_models, slerp_state_dicts, slerp_tensors, ties_state_dicts


# ---------------------------------------------------------------- SLERP
def test_slerp_endpoints_the_orthogonal_midpoint_and_the_parallel_fallback():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert torch.allclose(slerp_tensors(a, b, 0.0), a, atol=1e-6)
    assert torch.allclose(slerp_tensors(a, b, 1.0), b, atol=1e-6)
    assert torch.allclose(slerp_tensors(a, b, 0.5), torch.full((2,), math.sin(math.pi / 4)), atol=1e-6)
    p, q = torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])  # same direction: linear interpolation
    assert torch.allclose(slerp_tensors(p, q, 0.25), torch.tensor([1.25, 2.5]))
    assert torch.allclose(slerp_tensors(torch.zeros(2), q, 0.5), 0.5 * q)  # a zero vector: linear


def test_slerp_state_dicts_on_the_changes_from_a_base():
    base = {"w": torch.ones(2, 2), "n": torch.tensor(3)}
    a = {"w": base["w"] + torch.tensor([[1.0, 0.0], [0.0, 0.0]]), "n": torch.tensor(3)}
    b = {"w": base["w"] + torch.tensor([[0.0, 1.0], [0.0, 0.0]]), "n": torch.tensor(3)}
    out = slerp_state_dicts(a, b, 0.5, base)
    expected = base["w"] + math.sin(math.pi / 4) * torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert torch.allclose(out["w"], expected, atol=1e-6) and out["n"].item() == 3
    raw = slerp_state_dicts(a, b, 0.0)  # without a base: raw weights, t = 0 gives the first model
    assert torch.allclose(raw["w"], a["w"], atol=1e-6)
    with pytest.raises(ValueError):
        slerp_state_dicts(a, {"other": b["w"]}, 0.5)


# ---------------------------------------------------------------- TIES
def test_ties_elects_a_sign_and_averages_only_the_agreeing_changes():
    base = {"w": torch.zeros(4)}
    m1 = {"w": torch.tensor([1.0, -1.0, 0.1, 0.0])}
    m2 = {"w": torch.tensor([1.0, 1.0, 0.1, -3.0])}
    out = ties_state_dicts(base, [m1, m2], density=1.0)
    # entry 1: summed change 0 elects +, only m2 agrees; entry 3: elected -, only m2 has a change
    assert torch.allclose(out["w"], torch.tensor([1.0, 1.0, 0.1, -3.0]))
    assert torch.allclose(ties_state_dicts(base, [m1, m2], density=1.0, scale=0.5)["w"], 0.5 * out["w"])


def test_ties_trims_each_change_to_its_largest_entries_and_keeps_other_tensors():
    base = {"w": torch.zeros(4), "n": torch.tensor(7)}
    model = {"w": torch.tensor([4.0, 3.0, 2.0, 1.0]), "n": torch.tensor(9)}
    out = ties_state_dicts(base, [model], density=0.5)
    assert torch.allclose(out["w"], torch.tensor([4.0, 3.0, 0.0, 0.0])) and out["n"].item() == 7
    with pytest.raises(ValueError):
        ties_state_dicts(base, [model], density=0.0)
    with pytest.raises(ValueError):
        ties_state_dicts(base, [], density=0.5)


def test_merge_models_roundtrip_for_every_method(tmp_path):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.BertConfig(
        vocab_size=30, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32
    )
    paths = []
    for i in range(3):
        torch.manual_seed(i)
        model = transformers.BertModel(cfg)
        p = tmp_path / f"m{i}"
        model.save_pretrained(p)
        (p / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghij")))
        transformers.BertTokenizer(str(p / "vocab.txt")).save_pretrained(p)
        paths.append(str(p))
    base, a, b = paths
    out = merge_models([a, b], str(tmp_path / "slerp"), "slerp", base=base)
    meta = json.loads((out / "SOUP.json").read_text())
    assert meta["method"] == "slerp" and meta["weights"] == [0.5, 0.5]
    merged = transformers.AutoModel.from_pretrained(out)
    assert all(torch.isfinite(p).all() for p in merged.parameters())
    out = merge_models([a, b], str(tmp_path / "ties"), "ties", base=base, density=0.5)
    assert json.loads((out / "SOUP.json").read_text())["density"] == 0.5
    with pytest.raises(ValueError):
        merge_models([a, b], str(tmp_path / "x"), "ties")  # ties needs a base
    with pytest.raises(ValueError):
        merge_models([a, b, base], str(tmp_path / "y"), "slerp")  # slerp merges exactly two models
    linear = merge_models([a, b], str(tmp_path / "lin"), "linear", [1, 1])
    ma, mb, ml = (transformers.AutoModel.from_pretrained(p) for p in (a, b, linear))
    for k, v in ml.state_dict().items():
        if v.is_floating_point():
            assert torch.allclose(v, 0.5 * (ma.state_dict()[k] + mb.state_dict()[k]), atol=1e-6)


# ---------------------------------------------------------------- alternating train-and-merge
def test_arm_overrides_silence_the_other_teachers_and_start_later_rounds_from_the_merge():
    first = arm_overrides(2, 1, "alt_r1_b", 675, 42, 1)
    assert "teachers.0.weight=0" in first and "teachers.1.weight=1" in first
    assert "train.max_steps=675" in first and "train.seed=42" in first and "name=alt_r1_b" in first
    assert not any(o.startswith("student.") for o in first)
    later = arm_overrides(2, 0, "alt_r3_a", 675, 42, 3, "/m/alt_r2", "/r/final/teacher_heads.pt")
    assert "train.seed=44" in later
    assert "student.hf_id=/m/alt_r2" in later and "student.revision=null" in later
    assert "student.heads_path=/r/final/teacher_heads.pt" in later


def test_find_run_reports_the_latest_folder_and_whether_it_finished(tmp_path):
    assert find_run(str(tmp_path), "r") == (None, False)
    (tmp_path / "r" / "20260101-a").mkdir(parents=True)
    assert find_run(str(tmp_path), "r") == ("20260101-a", False)
    (tmp_path / "r" / "20260101-a" / "final").mkdir()
    (tmp_path / "r" / "20260102-b").mkdir()
    assert find_run(str(tmp_path), "r") == ("20260102-b", False)  # the latest folder counts
    assert run_name("alt", 2, "bge") == "alt_r2_bge"


class FakeTrainer:
    """Stands in for `python -m turkish_distill.train`: records the commands and creates the run folders."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.commands = []

    def __call__(self, command):
        overrides = [command[i + 1] for i, a in enumerate(command) if a == "--override"]
        name = next(o.split("=", 1)[1] for o in overrides if o.startswith("name="))
        run_id = command[command.index("--run-id") + 1] if "--run-id" in command else "run-1"
        final = self.output_dir / name / run_id / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "teacher_heads.pt").write_bytes(b"heads")
        self.commands.append((name, run_id, overrides))


class FakeMerger:
    def __init__(self):
        self.calls = []

    def __call__(self, paths, weights, out_dir, label=None):
        self.calls.append((list(paths), list(weights), out_dir))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "SOUP.json").write_text("{}")


def test_alternating_runs_every_arm_of_every_round_and_hands_the_merge_on(experiment, tmp_path):
    config = tmp_path / "cfg.yaml"
    dump_config(experiment, config)
    trainer, merger = FakeTrainer(experiment.output_dir), FakeMerger()
    final = run_alternating(str(config), "alt", 2, 5, ["train.lr=0.0001"], runner=trainer, merger=merger)
    models = Path(experiment.output_dir).parent / "models"
    assert final == str(models / "alt_r2")
    assert [c[0] for c in trainer.commands] == ["alt_r1_t_small", "alt_r1_t_big", "alt_r2_t_small", "alt_r2_t_big"]
    round1, round2 = trainer.commands[0][2], trainer.commands[2][2]
    assert "train.lr=0.0001" in round1 and "train.seed=42" in round1 and not any(o.startswith("student.hf_id") for o in round1)
    assert "train.seed=43" in round2 and f"student.hf_id={models / 'alt_r1'}" in round2
    own_heads = os.path.join(experiment.output_dir, "alt_r1_t_small", "run-1", "final", "teacher_heads.pt")
    assert f"student.heads_path={own_heads}" in round2
    assert "teachers.0.weight=1" in round2 and "teachers.1.weight=0" in round2
    assert len(merger.calls) == 2 and merger.calls[0][1] == [1.0, 1.0] and len(merger.calls[0][0]) == 2
    summary = json.loads((models / "alt_r2" / "ALTERNATING.json").read_text())
    assert summary["steps_per_arm"] == 10 and summary["arms"] == ["t_small", "t_big"]


def test_alternating_resumes_after_a_disconnect(experiment, tmp_path):
    config = tmp_path / "cfg.yaml"
    dump_config(experiment, config)
    trainer, merger = FakeTrainer(experiment.output_dir), FakeMerger()
    run_alternating(str(config), "alt", 1, 5, runner=trainer, merger=merger)
    assert len(trainer.commands) == 2 and len(merger.calls) == 1
    again = FakeTrainer(experiment.output_dir)
    run_alternating(str(config), "alt", 1, 5, runner=again, merger=merger)  # everything is finished: nothing runs
    assert again.commands == [] and len(merger.calls) == 1
    # one arm lost its final model (the session died during it): that arm resumes under its run id, the merge is redone
    import shutil

    shutil.rmtree(Path(experiment.output_dir) / "alt_r1_t_big" / "run-1" / "final")
    shutil.rmtree(Path(experiment.output_dir).parent / "models" / "alt_r1")
    resumed = FakeTrainer(experiment.output_dir)
    run_alternating(str(config), "alt", 1, 5, runner=resumed, merger=merger)
    assert [c[0] for c in resumed.commands] == ["alt_r1_t_big"] and resumed.commands[0][1] == "run-1"
    assert len(merger.calls) == 2


def test_alternating_needs_two_arms_and_positive_sizes(experiment, tmp_path):
    config = tmp_path / "cfg.yaml"
    dump_config(experiment, config)
    with pytest.raises(ValueError):
        run_alternating(str(config), "alt", 1, 5, arms=[0], runner=FakeTrainer(experiment.output_dir), merger=FakeMerger())
    with pytest.raises(ValueError):
        run_alternating(str(config), "alt", 0, 5, runner=FakeTrainer(experiment.output_dir), merger=FakeMerger())
    with pytest.raises(ValueError):
        run_alternating(str(config), "alt", 1, 5, arms=[0, 5], runner=FakeTrainer(experiment.output_dir), merger=FakeMerger())
