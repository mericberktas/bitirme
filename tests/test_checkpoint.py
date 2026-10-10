import torch

from turkish_distill.checkpoint import (
    cleanup_stale,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)


def test_save_load_and_latest(tmp_path):
    state = {"w": torch.arange(5), "step": 3}
    path = save_checkpoint(state, tmp_path / "ckpt", 3)
    assert path.name == "step_00000003.pt"
    assert latest_checkpoint(tmp_path / "ckpt") == path
    loaded = load_checkpoint(path)
    assert loaded["step"] == 3 and torch.equal(loaded["w"], torch.arange(5))


def test_rotation_keeps_last_n(tmp_path):
    for step in (1, 2, 3, 4):
        save_checkpoint({"step": step}, tmp_path / "c", step, keep_last=2)
    assert [s for s, _ in list_checkpoints(tmp_path / "c")] == [3, 4]


def test_staging_dir_is_used_and_cleaned(tmp_path):
    scratch = tmp_path / "scratch"
    path = save_checkpoint({"step": 1}, tmp_path / "drive", 1, local_tmp_dir=scratch)
    assert path.exists()
    assert list(scratch.iterdir()) == []  # staged file removed
    assert not list((tmp_path / "drive").glob("*.tmp"))


def test_stale_tmp_files_are_ignored_and_removed(tmp_path):
    d = tmp_path / "c"
    save_checkpoint({"step": 1}, d, 1)
    (d / "step_00000002.pt.tmp").write_bytes(b"partial")
    assert latest_checkpoint(d).name == "step_00000001.pt"
    cleanup_stale(d)
    assert not list(d.glob("*.tmp"))
