from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cascade.shared.config import RoundConfig
from cascade.trainer.continuation import (
    SnapshotStore,
    marketcast_data_files,
    restore_marketcast_data,
)
from cascade.trainer.continuation_transport import ContinuationMirror
from cascade.trainer.remote import RemoteHost


def test_mirror_keeps_last_verified_copy_after_partial_corrupt_or_old_update(tmp_path, monkeypatch):
    remote = SnapshotStore(tmp_path / "remote", {"init": "pinned", "seed": 7})
    remote.save(2, {"step": 2, "weights": torch.arange(8.0)})
    mirror = ContinuationMirror(
        RemoteHost("fixture", "127.0.0.1"),
        tmp_path / "local",
        str(remote.root),
        max_bytes=1024**2,
        interval=0,
    )
    monkeypatch.setattr(mirror, "_run", lambda *a, **kw: (remote.root / "CURRENT").read_text())

    def harvest(host, source, dest, **kwargs):
        shutil.copytree(source, dest)

    monkeypatch.setattr("cascade.trainer.remote.harvest_remote_dir", harvest)
    mirror.pull()
    local = SnapshotStore(mirror.root, remote.binding)
    assert local.load()["step"] == 2
    pointer = json.loads((remote.root / "CURRENT").read_text())
    tensor = remote.root / pointer["version"] / "state.safetensors"
    original = tensor.read_bytes()
    tensor.write_bytes(b"partial data")
    with pytest.raises(ValueError, match="tensor digest"):
        mirror.pull()
    assert local.load()["step"] == 2
    tensor.write_bytes(original)
    remote.save(3, {"step": 3, "weights": torch.arange(8.0)})
    mirror.pull()
    assert local.load()["step"] == 3
    (remote.root / "CURRENT").write_text(json.dumps(pointer))
    mirror.pull()
    assert local.load()["step"] == 3


def test_snapshot_rejects_wrong_base_partial_write_and_budget_overflow(tmp_path):
    store = SnapshotStore(tmp_path, {"init": "pinned"}, max_bytes=4096)
    store.save(1, {"step": 1, "weights": torch.ones(3)})
    with pytest.raises(ValueError, match="different job"):
        SnapshotStore(tmp_path, {"init": "wrong"}).load()
    with pytest.raises(ValueError, match="storage limit"):
        store.save(2, {"step": 2, "weights": torch.ones(10000)})
    assert store.load()["step"] == 1
    with pytest.raises(ValueError, match="completed optimizer step"):
        store.save(0, {"step": 0})


def test_marketcast_recovery_restores_original_data_without_fetching_revised_feed(
    tmp_path, monkeypatch
):
    from .test_marketcast_trainer import _train_snap

    snap = _train_snap(tmp_path / "initial")
    pin = "sha256:" + "a" * 64
    monkeypatch.setenv("CASCADE_TRAIN_IMAGE_DIGEST", pin)
    cfg = SimpleNamespace(
        round=replace(
            RoundConfig(), ops_continuation_interval_seconds=300, ops_continuation_image_digest=pin
        )
    )
    binding = {"backend": "marketcast", "base": "fixture", "seed": 7}
    store = SnapshotStore(tmp_path / "partial", {**binding, "data": snap.digest})
    store.save(2, {"step": 2, "data_files": marketcast_data_files(snap)})
    args = SimpleNamespace(continuation_root=str(store.root))
    assert restore_marketcast_data(args, cfg, binding, tmp_path / "replacement")
    from cascade.pool.marketcast import load_snapshot

    assert load_snapshot(tmp_path / "replacement", verify_signature=False).digest == snap.digest
    with pytest.raises(ValueError, match="different experiment"):
        restore_marketcast_data(args, cfg, {**binding, "seed": 8}, tmp_path / "wrong")


def test_pointer_publication_failure_never_deletes_committed_generation(tmp_path, monkeypatch):
    import cascade.trainer.continuation as module

    original = module.write_text_atomic
    store = SnapshotStore(tmp_path, {"init": "pinned"})

    def fail_after_rename(path, text, **kw):
        original(path, text, **kw)
        if Path(path).name == "CURRENT":
            raise OSError("simulated directory fsync failure after rename")

    from pathlib import Path

    monkeypatch.setattr(module, "write_text_atomic", fail_after_rename)
    with pytest.raises(OSError):
        store.save(1, {"step": 1, "weights": torch.arange(4.0)})
    assert store.load()["step"] == 1
