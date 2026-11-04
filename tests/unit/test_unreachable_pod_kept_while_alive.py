"""2026-11-04 uid 78: a funded pod stopped answering ssh ~3.5 h into its run;
after the 900 s reattach grace the leg was declared lost and the pod torn
down unharvested. Now, past the grace, the PROVIDER's listing decides: while
it still lists the pod alive the leg keeps polling (up to the extension);
gone ⇒ lost at once; unknown ⇒ the old behaviour."""
from __future__ import annotations

import types

import pytest

import cascade.trainer.remote as remote_mod
from cascade.shared.config.rounds import RoundConfig
from cascade.trainer.remote import (
    DETACHED_LAUNCH_TOKEN,
    DETACHED_STDERR_MARK,
    RemoteDispatchError,
    RemoteHost,
    run_detached,
)


def _host():
    return RemoteHost(name="pod-a", host="10.0.0.1", workdir="/root/cascade")


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class _Scripted:
    def __init__(self, steps):
        self.steps, self.calls = list(steps), 0

    def __call__(self, argv, timeout, stdin=None):
        self.calls += 1
        step = self.steps.pop(0) if self.steps else _proc(rc=255, stderr="down")
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(remote_mod, "_SLEEP", lambda s: None)
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 400.0                       # each poll 400 s apart
        return clock["t"]

    monkeypatch.setattr(remote_mod, "time", types.SimpleNamespace(time=fake_time))


def _run(steps, **kw):
    runner = _Scripted([_proc(stdout=DETACHED_LAUNCH_TOKEN + "\n")] + steps)
    return run_detached(_host(), "true", None, "/rd", timeout=10**7, poll_seconds=1,
                        grace_seconds=900, runner=runner, **kw), runner


def test_a_pod_the_provider_still_lists_alive_is_kept_and_its_run_fetched():
    down = [_proc(rc=255, stderr="down")] * 8               # ~3200 s unreachable
    back = [_proc(stdout="EXIT:0\n"), _proc(stdout="ok\n" + DETACHED_STDERR_MARK + "\n")]
    checks = []
    proc, _ = _run(down + back, alive=lambda: checks.append(1) or True,
                   alive_extend_seconds=7200)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok"
    assert checks                                           # the provider was asked


def test_the_provider_saying_gone_ends_the_leg_at_the_grace():
    with pytest.raises(RemoteDispatchError) as ei:
        _run([_proc(rc=255, stderr="down")] * 20, alive=lambda: False,
             alive_extend_seconds=7200)
    assert "no longer lists it alive" in str(ei.value) and ei.value.returncode == 255


def test_unknown_liveness_or_no_check_keeps_the_old_give_up():
    for kw in ({}, {"alive": lambda: None, "alive_extend_seconds": 7200}):
        with pytest.raises(RemoteDispatchError) as ei:
            _run([_proc(rc=255, stderr="down")] * 20, **kw)
        assert "unreachable" in str(ei.value)


def test_the_extension_is_bounded():
    with pytest.raises(RemoteDispatchError):
        _run([_proc(rc=255, stderr="down")] * 100, alive=lambda: True,
             alive_extend_seconds=1200)


def test_a_failing_liveness_check_keeps_the_pod():
    def boom():
        raise RuntimeError("lium ps 502")

    down = [_proc(rc=255, stderr="down")] * 4
    back = [_proc(stdout="EXIT:0\n"), _proc(stdout="ok\n" + DETACHED_STDERR_MARK + "\n")]
    proc, _ = _run(down + back, alive=boom, alive_extend_seconds=7200)
    assert proc.returncode == 0


def test_knob_default():
    assert RoundConfig.unreachable_alive_extend_seconds == 7200
