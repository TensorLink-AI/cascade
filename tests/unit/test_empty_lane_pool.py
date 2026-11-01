"""2026-11-01: 8 legs (5 operator-billed make-goods + 3 miner-funded) sat in
``_FinalLanePool.get(deadline=…)`` from ~02:20 until their latest safe start
(~06:41) after every operator lane was deleted / drained — they never went
back to renting. A pool with ZERO live lanes now raises ``_NoOperatorLanes``
within one refresh; the leg requeues ``no_capacity`` (unburned, no backoff)
and its next try rents: an operator pod for an operator-billed entry, the
miner's own key for a miner-funded one."""
from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from cascade.funding.queue import FundedQueue
from cascade.trainer.faults import (
    _LaneDeadlinePassed,
    _NoOperatorLanes,
    record_no_lanes_verdict,
)
from cascade.trainer.loop import TrainerRunner, _FinalLanePool, _FundedOperatorRent
from cascade.trainer.remote import RemoteHost
from tests.unit.test_funded_expired_key_operator_rent import REF_A, _expired_runner
from tests.unit.test_funded_pod_wiring import _challenger, _runner, _vault
from tests.unit.test_funded_rent_wait import _arm_wait

LANE = RemoteHost(name="targon-lane-2", host="104.218.237.44", port=22, user="root",
                  workdir="/root/cascade", stage="final")


def _waiter(pool, deadline):
    out = {}

    def run():
        t0 = time.monotonic()
        try:
            out["got"] = pool.get(deadline=deadline)
        except Exception as e:  # noqa: BLE001 — the test reads it
            out["exc"] = e
        out["took"] = time.monotonic() - t0

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


# ── the pool ─────────────────────────────────────────────────────────────────

def test_an_empty_pool_raises_at_once_instead_of_waiting():
    pool = _FinalLanePool([], lambda: [])
    with pytest.raises(_NoOperatorLanes, match="no operator lanes left"):
        pool.get(deadline=time.time() + 3600)
    assert issubclass(_NoOperatorLanes, _LaneDeadlinePassed)     # old handlers still apply


def test_waiter_is_released_when_the_last_lane_leaves_the_hosts_file():
    on_file = [LANE]
    pool = _FinalLanePool([LANE], lambda: list(on_file), sync_from_file=True)
    pool.REFRESH_INTERVAL_S = 0.05
    busy = pool.get(block=False)                                 # a leg holds the lane
    t, out = _waiter(pool, time.time() + 3600)
    time.sleep(0.2)
    assert t.is_alive()                                          # a live busy lane: waits
    on_file.clear()                                              # every lane deleted
    t.join(timeout=2.0)
    assert not t.is_alive() and isinstance(out.get("exc"), _NoOperatorLanes)
    pool.put(busy)                                               # the busy leg ends: dropped
    assert pool.known_hosts() == []


def test_waiter_is_released_when_the_last_lane_is_marked_dead():
    pool = _FinalLanePool([LANE], lambda: [LANE])
    pool.REFRESH_INTERVAL_S = 0.05
    pool.get(block=False)
    t, out = _waiter(pool, time.time() + 3600)
    time.sleep(0.2)
    assert t.is_alive()
    pool.mark_dead(LANE, "pod unreachable")                      # drained (or breaker)
    t.join(timeout=2.0)
    assert not t.is_alive() and isinstance(out.get("exc"), _NoOperatorLanes)
    assert out["took"] < 1.5
    assert pool.unavailable_names() == {"targon-lane-2"}


def test_a_lane_freed_while_waiting_is_still_served():
    pool = _FinalLanePool([LANE], lambda: [LANE])
    pool.REFRESH_INTERVAL_S = 0.05
    held = pool.get(block=False)
    threading.Timer(0.15, lambda: pool.put(held)).start()
    assert pool.get(deadline=time.time() + 3600) is held


# ── the verdict: no_capacity, unburned, NO backoff ───────────────────────────

def test_no_lanes_verdict_requeues_without_backoff(tmp_path):
    runner = _runner(tmp_path)
    assert record_no_lanes_verdict(runner, "hkA", RuntimeError("x")) is False
    assert record_no_lanes_verdict(runner, "hkA", _NoOperatorLanes("no operator lanes left"))
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)
    q = FundedQueue(tmp_path / "fq.json", entry_ttl_seconds=3600.0)
    q.add("hkA", REF_A, reveal_block=1)
    q.requeue("hkA", error=msg, error_class=cls, burn_attempt=burn)
    e = q.get("hkA")
    assert e.status == "queued" and e.not_before == 0 and e.attempts == 0


def test_lane_b_maps_no_lanes_to_no_capacity():
    from cascade.trainer.lane_b_driver import lane_breaker_failure

    f = lane_breaker_failure(_NoOperatorLanes("no operator lanes left"))["failure"]
    assert (f.miner_fault, f.error_class, f.burn) == (False, "no_capacity", False)


# ── the next try rents ───────────────────────────────────────────────────────

def _drained_pool(runner):
    pool = _FinalLanePool([LANE], lambda: [LANE])
    pool.mark_dead(LANE, "drained")
    runner.__dict__["_rolling_lane_pool_obj"] = pool
    return pool


def test_drained_lanes_on_file_are_not_operator_lanes(tmp_path):
    r, _q = _expired_runner(tmp_path)
    r.remote_hosts = [*r.remote_hosts, LANE]
    assert [h.name for h in r._operator_fallback_lanes()] == ["targon-lane-2"]
    _drained_pool(r)
    assert r._operator_fallback_lanes() == []


def test_operator_billed_next_attempt_rents_an_operator_pod(tmp_path):
    r, q = _expired_runner(tmp_path)
    r.remote_hosts = [*r.remote_hosts, LANE]
    _vault(tmp_path, "hkA")
    q.set_operator_billed("hkA", True)                           # owner make-good
    _drained_pool(r)
    with pytest.raises(_FundedOperatorRent):                     # not the dead lane pool
        r._rent_funded_host("777", _challenger("hkA", ref=REF_A))


def test_miner_funded_next_attempt_polls_its_own_key(tmp_path):
    runner = _runner(tmp_path, funded_operator_fallback=True)
    runner.remote_hosts_path = None
    runner.remote_hosts = [replace(runner.remote_hosts[0], profile_only=True), LANE]
    runner._operator_fallback_lanes = TrainerRunner._operator_fallback_lanes.__get__(runner)
    _arm_wait(runner, deadline_offsets=3600, capacity_seq=[0, 2])
    runner._operator_fallback_eligible = lambda hotkey, now, deadline: True
    # live lane on file: the wait leaves for it …
    assert runner._wait_for_funded_capacity("RTX4090", describe="leg", hotkey="hk") == "operator"
    # … all lanes drained: the marketplace (own key) is polled until capacity
    _drained_pool(runner)
    _arm_wait(runner, deadline_offsets=3600, capacity_seq=[0, 2])
    runner._operator_fallback_eligible = lambda hotkey, now, deadline: True
    assert runner._wait_for_funded_capacity("RTX4090", describe="leg", hotkey="hk") is True
