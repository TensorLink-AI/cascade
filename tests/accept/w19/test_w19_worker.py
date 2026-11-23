"""W19: `cascade.validator.lane_d_worker.LaneDWorker`, the off-thread runner
the validator's lane D tick submits to. Contract: README.md."""

from __future__ import annotations

import threading
import time

import pytest

from . import helpers as h


@pytest.fixture
def step():
    s = h.HungStep()
    yield s
    s.release.set()


def _worker(step):
    from cascade.validator.lane_d_worker import LaneDWorker

    return LaneDWorker(step)


def test_submit_returns_without_waiting_for_a_hung_step(step):
    w = _worker(step)
    t0 = time.monotonic()
    assert w.submit(h.B1) is True
    assert time.monotonic() - t0 < h.TICK_BUDGET_S
    assert step.started.wait(5)
    assert w.running is True
    assert step.calls == [h.B1]


def test_the_step_runs_on_another_daemon_thread(step):
    w = _worker(step)
    w.submit(h.B1)
    assert step.started.wait(5)
    assert step.threads[0] != threading.get_ident()
    assert step.daemon == [True]


def test_a_busy_worker_refuses_a_second_step(step):
    w = _worker(step)
    assert w.submit(h.B1) is True
    assert step.started.wait(5)
    assert w.submit(h.B2) is False
    assert w.submit(h.B1) is False
    time.sleep(0.05)
    assert step.calls == [h.B1]


def test_join_times_out_while_hung_and_succeeds_after_release(step):
    w = _worker(step)
    w.submit(h.B1)
    assert step.started.wait(5)
    t0 = time.monotonic()
    assert w.join(0.2) is False
    assert time.monotonic() - t0 < h.TICK_BUDGET_S
    step.release.set()
    assert w.join(10) is True
    assert w.running is False


def test_a_completed_boundary_is_never_run_again(step):
    w = _worker(step)
    step.release.set()
    assert w.submit(h.B1) is True
    assert w.join(10)
    assert w.submit(h.B1) is False
    assert w.submit(h.B2) is True
    assert w.join(10)
    assert step.calls == [h.B1, h.B2]
    assert w.last_error is None


def test_a_raising_step_is_recorded_and_its_boundary_retried():
    err = RuntimeError("w19 fixture: boundary hash unreadable")
    s = h.HungStep(raises=err)
    w = _worker(s)
    assert w.submit(h.B1) is True           # never raises into the caller
    assert w.join(10)
    assert w.last_error is err
    assert w.running is False
    assert w.submit(h.B1) is True           # fail-closed: retried next tick
    assert w.join(10)
    assert s.calls == [h.B1, h.B1]


def test_a_success_after_a_failure_clears_the_error():
    s = h.HungStep(raises=RuntimeError("w19 fixture: transient"))
    w = _worker(s)
    w.submit(h.B1)
    assert w.join(10)
    s.raises = None
    s.release.set()
    w.submit(h.B1)
    assert w.join(10)
    assert w.last_error is None
    assert w.submit(h.B1) is False


def test_idle_worker_join_is_immediate():
    w = _worker(h.HungStep())
    assert w.running is False
    assert w.join(0) is True
