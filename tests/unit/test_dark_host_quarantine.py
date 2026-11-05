"""2026-11-05 5G6jNi6Y: a funded pod went dark for 49 min mid-run, came back
with its detached worker gone (step 15,600) and the host was never
quarantined — only OUR watchdog's hang abort quarantined a host. Now a run
host that goes dark past the reattach grace, or whose worker vanishes, is
quarantined; a healthy run quarantines nothing."""
from __future__ import annotations

import types

import pytest

import cascade.trainer.host_preflight as host_preflight
import cascade.trainer.remote as remote_mod
from cascade.trainer.remote import (
    DETACHED_LAUNCH_TOKEN,
    DETACHED_STDERR_MARK,
    RemoteDispatchError,
    RemoteHost,
    run_detached,
)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class _Scripted:
    def __init__(self, steps):
        self.steps = list(steps)

    def __call__(self, argv, timeout, stdin=None):
        step = self.steps.pop(0) if self.steps else _proc(rc=255, stderr="down")
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(remote_mod, "_SLEEP", lambda s: None)
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 400.0
        return clock["t"]

    monkeypatch.setattr(remote_mod, "time", types.SimpleNamespace(time=fake_time))


@pytest.fixture
def quarantined(monkeypatch):
    got: list[tuple[str, str]] = []
    monkeypatch.setattr(host_preflight, "quarantine_host_ip",
                        lambda ip, reason, **kw: got.append((ip, reason)))
    return got


def _run(steps, **kw):
    runner = _Scripted([_proc(stdout=DETACHED_LAUNCH_TOKEN + "\n")] + steps)
    host = RemoteHost(name="pod-a", host="195.242.10.42", workdir="/root/cascade")
    return run_detached(host, "true", None, "/rd", timeout=10**7, poll_seconds=1,
                        grace_seconds=900, runner=runner, describe="leg x", **kw)


FETCH = _proc(stdout="out\n" + DETACHED_STDERR_MARK + "\nerr\n")


def test_a_vanished_worker_quarantines_its_host(quarantined):
    proc = _run([_proc(stdout='{"step": 15600}\nRUNNING\n'), _proc(stdout="GONE\n"), FETCH])
    assert proc.returncode == 1 and "vanished without an exit code" in proc.stderr
    assert [ip for ip, _ in quarantined] == ["195.242.10.42"]
    assert "vanished mid-run" in quarantined[0][1]


def test_dark_past_grace_quarantines_once_even_while_the_provider_keeps_it(quarantined):
    down = [_proc(rc=255, stderr="down")] * 8               # ~3200 s dark
    proc = _run(down + [_proc(stdout="EXIT:0\n"), FETCH],
                alive=lambda: True, alive_extend_seconds=7200)
    assert proc.returncode == 0                              # the run itself is kept
    assert [ip for ip, _ in quarantined] == ["195.242.10.42"]
    assert "dark" in quarantined[0][1]


def test_dark_and_lost_quarantines_once(quarantined):
    with pytest.raises(RemoteDispatchError):
        _run([_proc(rc=255, stderr="down")] * 20, alive=lambda: False,
             alive_extend_seconds=7200)
    assert [ip for ip, _ in quarantined] == ["195.242.10.42"]


def test_a_brief_blip_inside_the_grace_quarantines_nothing(quarantined):
    proc = _run([_proc(rc=255, stderr="down"), _proc(stdout="EXIT:0\n"), FETCH])
    assert proc.returncode == 0 and quarantined == []


def test_a_clean_run_quarantines_nothing(quarantined):
    proc = _run([_proc(stdout="RUNNING\n"), _proc(stdout="EXIT:0\n"), FETCH])
    assert proc.returncode == 0 and quarantined == []


def test_a_quarantine_failure_never_fails_the_poll(monkeypatch):
    def boom(ip, reason, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(host_preflight, "quarantine_host_ip", boom)
    proc = _run([_proc(stdout="GONE\n"), FETCH])
    assert proc.returncode == 1
