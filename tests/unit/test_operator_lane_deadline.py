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
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.trainer.loop import (
    ResolvedGenerator,
    TrainerRunner,
    _FinalLanePool,
    _FundedLegSkip,
    _FundedOperatorFallback,
    _LaneDeadlinePassed,
)
from cascade.trainer.remote import RemoteHost

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


def test_free_lane_dispatch_retry_wait_is_bounded_too():
    lane = _Host("a")
    pool = _FinalLanePool([lane], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    d = _Disp(fail_first=True)
    # The one lane is handed back after the failure, but the deadline has
    # passed by the retry's get(): the leg ends, the lane stays in the pool.
    with pytest.raises(_LaneDeadlinePassed):
        TrainerRunner._dispatch_on_free_lane(d, pool, [lane], describe="x",
                                             deadline=time.time() + 0.01, role="challenger")
    assert d.calls == ["a"]
    assert pool.get(timeout=0.1) is lane


def test_free_lane_dispatch_with_no_deadline_uses_a_plain_queue():
    # Operator-fleet finals pass no deadline and may hand a plain queue.Queue
    # (no ``deadline`` kwarg) — that path is byte-for-byte the old one.
    pool: _q.Queue = _q.Queue()
    pool.put(_Host("a"))
    out = TrainerRunner._dispatch_on_free_lane(_Disp(), pool, [], describe="x",
                                               role="challenger")
    assert out.host == "a"
    assert pool.get(timeout=0.1).name == "a"


# ── the round: fallback legs vs the latest safe start ────────────────────────


def _fallback_round(cfg, tmp_path, monkeypatch, *, epoch_end_wall: float,
                    king_falls_back: bool):
    from tests.unit.test_trainer_round import _FakeBaseTrainer

    rnd = replace(cfg.round, funded_mode="required", funded_pods="rent", funded_king_rent=True,
                  funded_activation_block=0, funded_operator_fallback=True)
    runner = TrainerRunner(cfg=replace(cfg, round=rnd), base_trainer=_FakeBaseTrainer(),
                           work_root=tmp_path, trainer_spec="m:C")
    lane = RemoteHost(name="sf-l40s-0", host="10.9.9.9", port=2222, stage="final",
                      workdir="/root/cascade")
    runner.remote_hosts = [RemoteHost(name="funded-pod-profile", host="127.0.0.1", stage="final"),
                           lane]
    runner._funded_field = {"c": VAULT_REF}
    runner._funded_leg_failures = {}
    runner._funded_roster = {"seated": [], "waiting": [], "terminal": [], "outcomes": []}
    runner._funded_king_lock = threading.Lock()
    runner._funded_king_host = None
    runner._final_role_hosts = {}
    runner._funded_epoch_end_wall = epoch_end_wall      # fixed per attempt, as live
    if king_falls_back:
        runner._rent_king_host = lambda rid: (_ for _ in ()).throw(_FundedOperatorFallback("king"))
    else:
        runner._rent_king_host = lambda rid: lane
        runner._stage_king_vault = lambda host, gen: host
        runner._refuse_diverged_king = lambda entry, contract: None
    runner._run_funded_leg = lambda *a, **k: (_ for _ in ()).throw(_FundedOperatorFallback("c"))
    runner._stage_vault_zip_on = lambda host, digest: host
    seen: list = []

    class _D:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, **kw):
            seen.append((host.name, kw["role"]))
            return SimpleNamespace(hotkey=kw["hotkey"], role=kw["role"])

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher", _D)
    contract = cfg.throne_contracts()[0]
    jobs = [(ResolvedGenerator(hotkey="k", uid=0, ref=REF_A), "king"),
            (ResolvedGenerator(hotkey="c", uid=1, ref=VAULT_REF), "challenger")]
    return runner, jobs, contract, seen


def _leg_wall(cfg) -> float:
    return max(int(c.max_train_seconds) for c in cfg.throne_contracts())


def test_fallback_challenger_past_the_latest_safe_start_requeues_unburned(
        cfg, tmp_path, monkeypatch):
    # Epoch ends in 1 h: a full leg + publish margin no longer fits, so the
    # deadline is already behind us — the leg must NOT start (it would end
    # hours past the boundary); it settles as sold-out (no attempt burned).
    runner, jobs, contract, seen = _fallback_round(
        cfg, tmp_path, monkeypatch, epoch_end_wall=time.time() + 3600.0, king_falls_back=False)
    out = runner._train_remote(jobs, SimpleNamespace(base_seed=1), 10, contract,
                               contract.train_tokens)
    assert [e.role for e in out] == ["king"]
    assert [r for _, r in seen] == ["king"]                 # challenger never dispatched
    msg, miner_fault, error_class, burn = runner._funded_leg_failures["c"]
    assert "latest safe start" in msg
    assert (miner_fault, error_class, burn) == (False, "no_capacity", False)


def test_fallback_challenger_inside_the_latest_safe_start_dispatches(
        cfg, tmp_path, monkeypatch):
    runner, jobs, contract, seen = _fallback_round(
        cfg, tmp_path, monkeypatch,
        epoch_end_wall=time.time() + _leg_wall(cfg) + TrainerRunner.FUNDED_PUBLISH_MARGIN_SECONDS
        + 3600.0,
        king_falls_back=False)
    out = runner._train_remote(jobs, SimpleNamespace(base_seed=1), 10, contract,
                               contract.train_tokens)
    assert {e.role for e in out} == {"king", "challenger"}
    assert ("sf-l40s-0", "challenger") in seen
    assert runner._funded_leg_failures == {}


def test_fallback_king_past_the_latest_safe_start_aborts_the_round(
        cfg, tmp_path, monkeypatch):
    runner, jobs, contract, seen = _fallback_round(
        cfg, tmp_path, monkeypatch, epoch_end_wall=time.time() + 3600.0, king_falls_back=True)
    with pytest.raises(RuntimeError, match="king training failed on remote.*latest safe start"):
        runner._train_remote(jobs, SimpleNamespace(base_seed=1), 10, contract,
                             contract.train_tokens)
    assert seen == []


def test_unknown_epoch_end_leaves_the_lane_wait_unbounded(tmp_path):
    # No round context (unit fakes, --offline): the deadline helper answers
    # None and the fallback path waits as it always did.
    from tests.unit.test_funded_pod_wiring import _runner

    r = _runner(tmp_path)
    for attr in ("_funded_epoch_end_wall", "_stage_ctx", "_funded_gate_block"):
        if hasattr(r, attr):
            delattr(r, attr)
    assert TrainerRunner._operator_lane_deadline(r) is None


def test_fallback_skip_is_the_funded_leg_skip(cfg, tmp_path, monkeypatch):
    # The dropped leg travels the same _FundedLegSkip route as a rent failure
    # (never retried on the operator fleet, settled by _settle_funded).
    runner, jobs, contract, _ = _fallback_round(
        cfg, tmp_path, monkeypatch, epoch_end_wall=time.time() + 3600.0, king_falls_back=False)
    raised = {}
    orig = runner._record_funded_failure

    def _rec(hotkey, msg, **kw):
        raised["hk"] = hotkey
        orig(hotkey, msg, **kw)

    runner._record_funded_failure = _rec
    runner._train_remote(jobs, SimpleNamespace(base_seed=1), 10, contract, contract.train_tokens)
    assert raised["hk"] == "c"
    assert isinstance(_FundedLegSkip("c"), Exception)
