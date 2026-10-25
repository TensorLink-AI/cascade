"""In-flight legs and pods are reconciled while the trainer runs (T4), not
only at startup; lane-B open legs never hold their domain slot forever."""

from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("torch")

from cascade.trainer import expert_lane as tel  # noqa: E402
from cascade.trainer.expert_lane import submission_key  # noqa: E402
from cascade.trainer.lane_b_driver import FinishedExpertLeg  # noqa: E402

from .test_lane_b_routing_gate import ROUTES, _driver  # noqa: E402
from .test_rolling import (  # noqa: E402
    REF,
    Clock,
    FakeClient,
    _armed,
    _commit,
    _join,
    _sched,
)


@pytest.fixture
def routes(monkeypatch):
    """Synthetic routing probes (the routing-gate tests' table)."""
    monkeypatch.setattr(tel, "probe_routing",
                        lambda king_dir, contexts, cfg: ROUTES[Path(king_dir).name])


def test_periodic_reconcile_keeps_live_legs_and_reattaches_orphans(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    b0 = armed.round.rolling_from_block + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100), _commit("BRAV", REF["BRAV"], b0 - 50)]
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    gate = ops.hold["ALFA"] = threading.Event()        # ALFA's leg runs until released
    sched.tick(client, b0)
    assert "ALFA" in sched._threads and len(ops.sweeps) == 1        # the startup sweep
    # BRAV in flight in the queue with NO thread here (its thread died)
    q.add("BRAV", REF["BRAV"], reveal_block=b0 - 50)
    assert q.mark_in_flight("BRAV", REF["BRAV"], target_boundary=b0 - 5 + 900 * 2,
                            era_index=sched.state.current.index, started_block=b0)
    # ALFA settled while its payer-pod bench still runs: not in flight, thread alive
    q.mark_done("ALFA")
    sched.tick(client, b0 + 1)
    assert len(ops.sweeps) == 1                          # not due yet
    clock.t += sched.RECONCILE_SECONDS + 1
    sched.tick(client, b0 + 2)
    assert len(ops.sweeps) == 2
    keep_ids, keep = ops.sweeps[-1]
    assert {"ALFA", "BRAV"} <= keep                     # a live thread and an in-flight leg
    assert keep_ids == (str(sched.state.current.base_seed),)
    gate.set()
    _join(sched)
    assert "BRAV" in [c[0] for c in ops.leg_calls]      # the orphan was re-attached
    assert any(f.hotkey == "BRAV" for f in sched.state.finished)


def test_lane_b_open_legs_of_a_dead_process_are_requeued_uncharged(tmp_path, routes):
    drv, ops = _driver(tmp_path)
    drv.add_lineage_tokens(100)
    drv.tick(10, 0, "king-a")
    (leg,) = drv.tick(11, 1, "king-b")                  # dispatched; the process then dies
    again, ops2 = _driver(tmp_path)
    assert [o.sub.hotkey for o in again.queue.open] == ["alice"]
    legs = again.tick(12, 1, "king-b")
    assert [x.sub.hotkey for x in legs] == ["alice"] and ops2.dispatched
    assert len(again.queue.open) == 1
    assert submission_key(leg.sub) not in again.queue.attempts      # uncharged
    # only the FIRST tick of a process recovers: a live leg is never requeued
    assert again.tick(13, 1, "king-b") == [] and len(again.queue.open) == 1


def test_lane_b_stale_open_leg_expires_and_a_late_finish_is_discarded(tmp_path, routes):
    drv, ops = _driver(tmp_path)
    drv.add_lineage_tokens(100)
    drv.tick(10, 0, "king-a")
    (old,) = drv.tick(11, 1, "king-b")
    age = drv.cfg.lane_b.max_base_age_eras
    assert drv.tick(12, 1 + age, "king-b") == []        # not stale yet: slot still held
    legs = drv.tick(13, 2 + age, "king-b")              # stale: expired, re-admitted, re-dispatched
    assert [x.base_era for x in legs] == [2 + age]
    assert drv.queue.attempts[submission_key(old.sub)] == 1
    drv._leg_done(old, "late-pointer", "c", 5)          # the hung leg finally returns
    assert drv.st.finished == [] and [o.base_era for o in drv.queue.open] == [2 + age]


def test_lane_b_abandoned_settlement_commits_nothing(tmp_path, routes):
    drv, ops = _driver(tmp_path)
    leg = {"sub": {"hotkey": "alice", "uid": 7, "gen_ref": "g", "domain": "energy",
                   "commit_block": 5},
           "selection": {"0": [1]}, "base_king_ref": "k", "base_era": -10, "exclude": {}}
    drv.st.finished = [asdict(FinishedExpertLeg(leg=leg, leg_pointer="p",
                                                corpus_digest="c", train_block=3))]
    assert drv.abandon("r1") == ""
    assert drv.settle(round_id="r1", block=20, era_index=0, king_pointer="k",
                      work_dir=tmp_path / "w") == []
    assert "r1" not in drv.st.settled and len(drv.st.finished) == 1   # legs kept
    drv.settle(round_id="r2", block=21, era_index=0, king_pointer="k", work_dir=tmp_path / "w")
    assert drv.st.settled["r2"] == {"digest": ""} and drv.abandon("r2") == ""


def test_runner_lane_b_commitments_read_under_the_deadline():
    from types import SimpleNamespace

    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    seen = {}

    def with_deadline(fn, seconds):
        seen["s"] = seconds
        raise TimeoutError

    r = SimpleNamespace(_with_deadline=with_deadline, chain_read_timeout_s=7.0)
    ops = RunnerLaneBOps(r, SimpleNamespace(poll_commitments=lambda: []), None)
    with pytest.raises(TimeoutError):
        ops.commitments()
    assert seen["s"] == 7.0
