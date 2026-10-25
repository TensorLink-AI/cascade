"""2026-09-19 incident: operator-lane FALLBACK legs waited on the final lane pool
with no deadline. The marketplace latest-safe-start (epoch end − leg wall −
publish margin) bounded every RENT, but a leg that had switched to an operator
lane blocked on ``_FinalLanePool.get()`` and dispatched whenever a lane freed —
two 5 h legs started 2 h before the epoch end (three more queued behind them)
and held the round's manifest past the boundary. Now the same latest safe start
bounds the lane wait: a fallback challenger without a lane by then requeues
unburned (sold-out taxonomy); a king aborts the round at once.
"""

from __future__ import annotations

import queue as _q
import threading
import time
from types import SimpleNamespace

import pytest

from cascade.trainer.loop import (
    TrainerRunner,
    _FinalLanePool,
    _LaneDeadlinePassed,
)

REF_A = "alice/gen-a@sha256:" + "a" * 64
VAULT_REF = "vault/direct@sha256:" + "d" * 64


class _Host:
    def __init__(self, name, stage="final"):
        self.name = name
        self.stage = stage


# ── the pool's deadline form ─────────────────────────────────────────────────


def test_pool_get_with_a_past_deadline_raises_instead_of_blocking():
    pool = _FinalLanePool([], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    t0 = time.monotonic()
    with pytest.raises(_LaneDeadlinePassed):
        pool.get(deadline=time.time() - 1.0)
    assert time.monotonic() - t0 < 1.0                 # no refresh-loop wait


def test_pool_get_past_the_deadline_serves_a_lane_that_is_free_now():
    # Owner 2026-09-20: a free rented lane is never withheld; only WAITING
    # for one past the deadline is refused (the empty-pool case above).
    pool = _FinalLanePool([_Host("a")], lambda: [])
    assert pool.get(deadline=time.time() - 1.0).name == "a"
    with pytest.raises(_LaneDeadlinePassed):
        pool.get(deadline=time.time() - 1.0)          # now empty: no wait


def test_pool_get_serves_a_free_lane_before_the_deadline():
    pool = _FinalLanePool([_Host("a")], lambda: [])
    assert pool.get(deadline=time.time() + 60.0).name == "a"


def test_pool_get_leaves_a_lane_that_freed_too_late_for_the_next_leg():
    pool = _FinalLanePool([], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    deadline = time.time() + 0.3
    late = _Host("late")
    threading.Timer(0.5, lambda: pool.put(late)).start()
    with pytest.raises(_LaneDeadlinePassed):
        pool.get(deadline=deadline)
    time.sleep(0.4)
    assert pool.get(timeout=0.5) is late               # still in the pool, not lost


def test_pool_get_without_a_deadline_is_unchanged():
    pool = _FinalLanePool([_Host("a")], lambda: [])
    assert pool.get().name == "a"
    with pytest.raises(_q.Empty):
        pool.get(timeout=0.01)


# ── the free-lane dispatch ───────────────────────────────────────────────────


class _Disp:
    def __init__(self, fail_first: bool = False):
        self.calls: list = []
        self.fail_first = fail_first

    def dispatch(self, host, **kw):
        self.calls.append(host.name)
        if self.fail_first and len(self.calls) == 1:
            time.sleep(0.05)                       # the deadline passes during the failure
            raise RuntimeError("torn dispatch")
        return SimpleNamespace(host=host.name)


def test_free_lane_dispatch_raises_past_the_deadline_with_no_lane():
    pool = _FinalLanePool([], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    d = _Disp()
    with pytest.raises(_LaneDeadlinePassed):
        TrainerRunner._dispatch_on_free_lane(d, pool, [], describe="x",
                                             deadline=time.time() - 1.0, role="challenger")
    assert d.calls == []


def test_free_lane_dispatch_retry_past_the_deadline_takes_the_freed_lane():
    lane = _Host("a")
    pool = _FinalLanePool([lane], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    d = _Disp(fail_first=True)
    # The one lane is handed back after the failure and is FREE at the
    # retry's get(): past the deadline it is still served (no wait needed).
    out = TrainerRunner._dispatch_on_free_lane(d, pool, [lane], describe="x",
                                               deadline=time.time() + 0.01, role="challenger")
    assert out.host == "a" and d.calls == ["a", "a"]


def test_free_lane_dispatch_retry_wait_is_bounded_too():
    lane = _Host("a")

    class _Taken(_FinalLanePool):
        def put(self, item, *a, **kw):        # a sibling grabs the freed lane
            pass

    pool = _Taken([lane], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    d = _Disp(fail_first=True)
    # No lane free at the retry and the deadline has passed: no wait — the
    # leg ends (requeue), nothing hangs on the pool.
    with pytest.raises(_LaneDeadlinePassed):
        TrainerRunner._dispatch_on_free_lane(d, pool, [lane], describe="x",
                                             deadline=time.time() + 0.01, role="challenger")
    assert d.calls == ["a"]


def test_free_lane_dispatch_with_no_deadline_uses_a_plain_queue():
    # Operator-fleet finals pass no deadline and may hand a plain queue.Queue
    # (no ``deadline`` kwarg) — that path is byte-for-byte the old one.
    pool: _q.Queue = _q.Queue()
    pool.put(_Host("a"))
    out = TrainerRunner._dispatch_on_free_lane(_Disp(), pool, [], describe="x",
                                               role="challenger")
    assert out.host == "a"
    assert pool.get(timeout=0.1).name == "a"


def _hold_every_lane_busy(monkeypatch):
    """Make every deadline-form pool get() find no free lane (all busy)."""
    from cascade.trainer import loop as loop_mod

    orig_get = loop_mod._FinalLanePool.get

    def busy_get(self, *a, **kw):
        if kw.get("deadline") is not None:
            self._absorb_new()
            while True:
                try:
                    self.get_nowait()
                except Exception:  # noqa: BLE001
                    break
        return orig_get(self, *a, **kw)

    monkeypatch.setattr(loop_mod._FinalLanePool, "get", busy_get)


def _leg_wall(cfg) -> float:
    return max(int(c.max_train_seconds) for c in cfg.throne_contracts())


def test_unknown_epoch_end_leaves_the_lane_wait_unbounded(tmp_path):
    # No leg target (unit fakes, --offline): the deadline helper answers
    # None and the fallback path waits unbounded.
    from tests.unit.test_funded_pod_wiring import _runner

    r = _runner(tmp_path)
    assert TrainerRunner._operator_lane_deadline(r) is None
