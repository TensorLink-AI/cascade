"""2026-11-27: a boundary the worker moves past unsettled is given up AND
recorded. Before, the poll loop simply submitted the next boundary: the
failed one was never retried, never recorded, and (with no lane D receipt
at it) CI receipt replay skipped that round as pending forever. Now the
worker calls ``on_abandon(boundary)`` (the lane D step's ``abandon``, which
publishes the W14 absent ``infra_deferred`` record) before the later step."""

from __future__ import annotations

import inspect
import threading

from . import helpers as h


class FlakyStep:
    """``step(b)`` raises for the boundaries in ``fail``; ``abandon`` records."""

    def __init__(self, fail=(), *, abandon_raises: bool = False):
        self.fail = set(fail)
        self.calls: list[int] = []
        self.abandoned: list[int] = []
        self.order: list[tuple[str, int]] = []
        self.abandon_raises = abandon_raises

    def __call__(self, boundary):
        self.calls.append(int(boundary))
        self.order.append(("step", int(boundary)))
        if int(boundary) in self.fail:
            raise RuntimeError(f"w19: settlement {boundary} inputs unreadable")

    def abandon(self, boundary):
        self.abandoned.append(int(boundary))
        self.order.append(("abandon", int(boundary)))
        if self.abandon_raises:
            raise RuntimeError("w19: absent record not published")


def _worker(step):
    from cascade.validator.offthread_worker import LaneDWorker

    assert "on_abandon" in inspect.signature(LaneDWorker).parameters, \
        "LaneDWorker(step, on_abandon=) is missing: a boundary moved past is never recorded"
    return LaneDWorker(step, on_abandon=step.abandon)


def _run(worker, b) -> bool:
    started = worker.submit(b)
    assert worker.join(10)
    return started


def test_moving_past_a_failed_boundary_abandons_it_before_the_next_step():
    step = FlakyStep(fail={h.B1})
    w = _worker(step)
    assert _run(w, h.B1)
    assert _run(w, h.B1), "a failed boundary is retried while the head is still on it"
    assert step.abandoned == []
    assert _run(w, h.B2)
    assert step.abandoned == [h.B1], "the boundary moved past was not recorded"
    assert step.order[-2:] == [("abandon", h.B1), ("step", h.B2)]
    assert not w.submit(h.B1), "an abandoned boundary must never run again"
    assert step.calls == [h.B1, h.B1, h.B2]


def test_a_settled_boundary_is_never_abandoned():
    step = FlakyStep()
    w = _worker(step)
    assert _run(w, h.B1)
    assert _run(w, h.B2)
    assert step.abandoned == []


def test_a_boundary_failed_then_settled_is_not_abandoned():
    step = FlakyStep(fail={h.B1})
    w = _worker(step)
    assert _run(w, h.B1)
    step.fail.clear()
    assert _run(w, h.B1)
    assert _run(w, h.B2)
    assert step.abandoned == []


def test_a_raising_abandon_never_blocks_the_next_step():
    step = FlakyStep(fail={h.B1}, abandon_raises=True)
    w = _worker(step)
    assert _run(w, h.B1)
    assert _run(w, h.B2)
    assert step.abandoned == [h.B1] and step.calls[-1] == h.B2
    assert w.last_error is None, "the later step succeeded"


def test_abandon_runs_off_the_poll_thread():
    seen: list[int] = []
    step = FlakyStep(fail={h.B1})
    inner = step.abandon

    def abandon(b):
        seen.append(threading.get_ident())
        inner(b)

    step.abandon = abandon
    w = _worker(step)
    _run(w, h.B1)
    _run(w, h.B2)
    assert seen and seen[0] != threading.get_ident()


def test_the_service_loop_wires_the_steps_abandon(tmp_path):
    step = FlakyStep(fail={h.B1})
    r = h.runner(tmp_path / "w19", armed=True, step=step)
    chain = h.FakeChain(h.B1)
    h.drive(r, chain, [h.B1])
    assert r._lane_d_worker.join(10)
    h.drive(r, chain, [h.B2 + 1])
    assert r._lane_d_worker.join(10)
    assert step.abandoned == [h.B1], "the loop's worker does not record a boundary moved past"
