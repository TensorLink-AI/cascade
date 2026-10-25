"""Offline completion-race tests for the worker-side poll; no marketplace calls."""

import json
from pathlib import Path

from cascade.direct.runtime import LocalTorchDispatcher


def finish_during_listing(root, monkeypatch):
    (root / "checkpoints").mkdir(parents=True)
    original = Path.glob
    fired = False

    def glob(path, pattern):
        nonlocal fired
        snapshot = list(original(path, pattern))
        if path == root / "checkpoints" and not fired:
            fired = True
            (path / "checkpoint-1.json").write_text(json.dumps({"manifest": {"consumed": 1}}))
            (root / "done.json").write_text(json.dumps({"returncode": 0}))
        return iter(snapshot)

    monkeypatch.setattr(Path, "glob", glob)


def test_local_poll_does_not_publish_completion_ahead_of_final_checkpoint(tmp_path, monkeypatch):
    dispatcher = LocalTorchDispatcher(tmp_path)
    root = dispatcher.directory("job", 1)
    finish_during_listing(root, monkeypatch)
    assert dispatcher.poll("job", 1) == ([], None)
    assert dispatcher.poll("job", 1) == ([{"consumed": 1}], {"returncode": 0})
