import textwrap

import pytest

from turkish_distill.config import load_config

YAML = textwrap.dedent(
    """
    name: demo
    output_dir: /tmp/runs
    teachers:
      - {name: a, hf_id: x/a, dim: 4}
      - {name: b, hf_id: x/b, dim: 6, weight: 0.5}
    train: {batch_size: 4, max_steps: 10}
    """
)


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_and_defaults(tmp_path):
    cfg = load_config(write(tmp_path, YAML))
    assert cfg.name == "demo"
    assert [t.name for t in cfg.teachers] == ["a", "b"]
    assert cfg.teachers[1].weight == 0.5
    assert cfg.student.hf_id == "ytu-ce-cosmos/modernbert-tr-base-1k"
    assert cfg.train.batch_size == 4


def test_overrides(tmp_path):
    cfg = load_config(write(tmp_path, YAML), ["train.max_steps=99", "loss.gor_weight=0.1"])
    assert cfg.train.max_steps == 99
    assert cfg.loss.gor_weight == 0.1


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(write(tmp_path, YAML + "bogus: 1\n"))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(write(tmp_path, YAML), ["train.learning_rate=1"])


def test_duplicate_teacher_names_rejected(tmp_path):
    bad = YAML.replace("name: b,", "name: a,")
    with pytest.raises(ValueError, match="unique"):
        load_config(write(tmp_path, bad))


def test_fingerprint_changes_with_config(tmp_path):
    a = load_config(write(tmp_path, YAML))
    b = load_config(write(tmp_path, YAML), ["train.max_steps=11"])
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == load_config(write(tmp_path, YAML)).fingerprint()


def test_distillation_overrides_accept_scientific_notation_without_a_dot(tmp_path):
    import yaml

    from turkish_distill.config import load_config, parse_override_value

    assert parse_override_value("3e-05") == pytest.approx(3e-5) and parse_override_value("abc") == "abc"
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"name": "x", "output_dir": "o", "teachers": [{"name": "t", "hf_id": "h", "dim": 4}]}), encoding="utf-8")
    assert load_config(path, ["train.lr=3e-05"]).train.lr == pytest.approx(3e-5)
