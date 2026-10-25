"""Real subprocess lifetime fault injection for the pinned torchrun path."""

import fcntl
import json
import os
import signal
import time
from contextlib import suppress
from pathlib import Path

import pytest

from cascade.direct.runtime import LocalTorchDispatcher
from tests.integration.test_direct_torch import training_spec


@pytest.mark.parametrize("kill_launcher", [False, True])
def test_surviving_ranks_keep_launch_lock(tmp_path, kill_launcher):
    spec, _ = training_spec(tmp_path)
    dispatcher = LocalTorchDispatcher(tmp_path / "workers")
    dispatcher.ensure_started(spec)
    supervisor = dispatcher.children[0]
    root = dispatcher.directory(spec["job"], spec["fence"])
    launcher = None
    ranks = []
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (root / "pid.json").exists():
                launcher = json.loads((root / "pid.json").read_text())["pid"]
                children = Path(f"/proc/{launcher}/task/{launcher}/children")
                if children.exists():
                    ranks = [int(p) for p in children.read_text().split()]
                    if len(ranks) == spec["world_size"]:
                        break
            assert supervisor.poll() is None, (root / "worker.log").read_text()
            time.sleep(0.02)
        assert len(ranks) == spec["world_size"], (root / "worker.log").read_text()
        # Freeze ranks before they can finish, then kill their ancestors.
        for pid in ranks:
            os.kill(pid, signal.SIGSTOP)
        supervisor.kill()
        supervisor.wait(timeout=10)
        if kill_launcher:
            os.kill(launcher, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while True:
                try:
                    state = Path(f"/proc/{launcher}/stat").read_text().split(")", 1)[1].split()[0]
                except FileNotFoundError:
                    break  # Exit/reaping can race with any preceding existence check.
                if state == "Z":
                    break
                assert time.monotonic() < deadline
                time.sleep(0.02)
        with (root / "launch.lock").open("a") as lock, pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        replacement = LocalTorchDispatcher(dispatcher.root)
        replacement.ensure_started(spec)
        assert replacement.children == []
    finally:
        for pid in ranks:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        if launcher:
            with suppress(ProcessLookupError):
                os.kill(launcher, signal.SIGKILL)
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=10)
    deadline = time.monotonic() + 10
    with (root / "launch.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline, "lock leaked after all workers exited"
                time.sleep(0.02)
