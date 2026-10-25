"""T7/T8: the burn set's read-modify-write is locked and atomic, harvest
leftovers age out, the trainer log rotates, and only one trainer may run on a
work root."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
import time

import pytest

from cascade.trainer.main import _configure_logging, acquire_instance_lock
from cascade.trainer.runner_common import (
    HARVEST_RETENTION_SECONDS,
    _add_seen_hotkeys,
    _load_seen_hotkeys,
    prune_harvest_dirs,
)


def test_concurrent_burns_never_drop_each_other(tmp_path):
    path = tmp_path / "submissions.json"
    hotkeys = [f"hk{i}" for i in range(40)]
    threads = [threading.Thread(target=_add_seen_hotkeys, args=(path, [h])) for h in hotkeys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _load_seen_hotkeys(path) == set(hotkeys)
    assert json.loads(path.read_text()) == sorted(hotkeys)
    assert not list(tmp_path.glob("*.tmp.*"))          # atomic writes leave no temp


def test_harvest_leftovers_age_out(tmp_path):
    root = tmp_path / "_funded_harvest" / "777"
    old, fresh = root / "hkOld", root / "hkNew"
    for d in (old, fresh):
        d.mkdir(parents=True)
        (d / "weights.safetensors").write_bytes(b"W")
    rec = root / "hkOld.pending.json"
    rec.write_text("{}")
    past = time.time() - HARVEST_RETENTION_SECONDS - 60
    for p in (old, rec):
        os.utime(p, (past, past))
    assert prune_harvest_dirs(tmp_path) == 2
    assert not old.exists() and not rec.exists() and fresh.exists()
    os.utime(fresh, (past, past))
    assert prune_harvest_dirs(tmp_path) == 1
    assert not root.exists()                            # empty seed dir removed
    assert prune_harvest_dirs(tmp_path / "nowhere") == 0


def test_a_second_trainer_on_the_same_work_root_fails_fast(tmp_path):
    held = acquire_instance_lock(tmp_path)
    try:
        assert (tmp_path / "trainer.lock").read_text().strip() == str(os.getpid())
        with pytest.raises(RuntimeError, match="already holds"):
            acquire_instance_lock(tmp_path)
    finally:
        held.close()
    acquire_instance_lock(tmp_path).close()             # free once the holder exits


def test_log_file_is_size_rotated(tmp_path):
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        log_file = tmp_path / "trainer.log"
        _configure_logging("INFO", log_file, max_mb=1, backups=2)
        (handler,) = root.handlers
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 1024 * 1024 and handler.backupCount == 2
        lg = logging.getLogger("cascade.test_rotation")
        for _ in range(3000):
            lg.info("x" * 1000)
        handler.flush()
        assert (tmp_path / "trainer.log.1").is_file()
        assert not (tmp_path / "trainer.log.3").exists()
        assert log_file.stat().st_size <= 1024 * 1024 + 2000
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
