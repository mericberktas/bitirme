"""Weight averaging of backbones."""

import json

import pytest
import torch

from turkish_distill.soup import soup_models, weighted_average_state_dicts


def sd(value, step=3):
    return {"w": torch.full((2, 2), float(value)), "steps": torch.tensor(step)}


def test_weighted_mean_and_normalisation():
    out = weighted_average_state_dicts([sd(0), sd(10)], [1, 3])  # weights are normalised: 0.25 / 0.75
    assert torch.allclose(out["w"], torch.full((2, 2), 7.5))
    assert out["steps"].item() == 3  # integer tensors are copied from the first model


def test_identical_models_are_unchanged():
    out = weighted_average_state_dicts([sd(2.5), sd(2.5)], [0.3, 0.7])
    assert torch.allclose(out["w"], torch.full((2, 2), 2.5))


def test_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        weighted_average_state_dicts([sd(1)], [1, 2])
    with pytest.raises(ValueError):
        weighted_average_state_dicts([sd(1), sd(2)], [0, 0])
    with pytest.raises(ValueError):
        weighted_average_state_dicts([sd(1), {"other": torch.zeros(1)}], [1, 1])


def test_soup_models_roundtrip(tmp_path):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.BertConfig(
        vocab_size=30, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32
    )
    paths = []
    for i in range(2):
        torch.manual_seed(i)
        m = transformers.BertModel(cfg)
        p = tmp_path / f"m{i}"
        m.save_pretrained(p)
        (p / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghij")))
        transformers.BertTokenizer(str(p / "vocab.txt")).save_pretrained(p)
        paths.append(str(p))
    out = soup_models(paths, [0.5, 0.5], str(tmp_path / "soup"))
    a, b = (transformers.AutoModel.from_pretrained(p) for p in paths)
    s = transformers.AutoModel.from_pretrained(out)
    pa, pb, ps = a.state_dict()["embeddings.word_embeddings.weight"], b.state_dict()["embeddings.word_embeddings.weight"], s.state_dict()["embeddings.word_embeddings.weight"]
    assert torch.allclose(ps, (pa + pb) / 2, atol=1e-6)
    meta = json.loads((out / "SOUP.json").read_text())
    assert meta["weights"] == [0.5, 0.5]
