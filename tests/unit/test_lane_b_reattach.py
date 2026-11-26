"""Lane B / lane C legs re-attach after a trainer restart (owner 2026-11-26,
parity with lane A detached legs).

Torch-free: the real funding queue, intake and lane B driver over the WB18
fixtures; the pod side is a fake ``reattach_leg`` (the production one drives
the funded path attach-only — its two guards, the rent refusal and the
dispatcher's no-relaunch, are pinned at the bottom)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from cascade.trainer import leg_reattach
from tests.accept.w_s2 import helpers as ws2
from tests.accept.wb18 import helpers as wb18


class _Wall:
    def __init__(self, t: float = 5_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _running_leg(tmp: Path, monkeypatch):
    """A funded energy leg dispatched (not returned) in era 1, with ops that
    can re-attach."""
    wall = _Wall()
    monkeypatch.setattr(leg_reattach, "_wall", wall)
    sc = wb18.Scenario(tmp, suite_targets=False, sentinel_from_block=0)
    for era, suite in ((1, ws2.SENTINEL), (2, "gift")):
        sc.set_sentinel(era, suite)
    sc.ops.reattached = []
    sc.ops.reattach_leg = lambda leg, era, block, done, rec: sc.ops.reattached.append(
        (leg, era, block, done, rec))
    sc.ops.leg_identity = lambda leg: {"base_seed": "77", "pod_name": "cascade-x-0",
                                       "tag": f"expert-{leg.sub.hotkey[:12]}-77"}
    sc.fund(wb18.HK_E, 5, "energy")
    sc.open_gate()
    assert len(sc.ops.dispatched) == 1, "fixture: the funded leg did not dispatch"
    return sc, wall


def _on_disk(tmp: Path) -> dict:
    return json.loads((tmp / "lane_b_driver.json").read_text())


def test_restart_mid_leg_reattaches_without_rerent_or_redispatch(tmp_path, monkeypatch):
    sc, wall = _running_leg(tmp_path, monkeypatch)
    leg0, _ = sc.ops.dispatched[0]
    entry = sc.fq.get(wb18.HK_E)
    attempt = entry.attempt_id
    assert entry.status == "in_flight"
    rec = next(iter(_on_disk(tmp_path)["attached"].values()))
    assert rec["pod_name"] == "cascade-x-0" and rec["attempt_id"] == leg0.funding["attempt_id"]
    assert rec["dispatch_block"] == leg0.funding["started_block"]

    wall.t += 300                               # a restart inside the grace
    sc.restart_all()
    assert sc.driver.live_hotkeys() == {wb18.HK_E}   # kept out of the startup sweep
    sc.tick(1)
    assert len(sc.ops.dispatched) == 1          # never re-dispatched
    assert len(sc.ops.reattached) == 1
    leg, era, block, done, got = sc.ops.reattached[0]
    assert (leg.sub.hotkey, leg.selection, leg.base_king_ref, leg.base_era) == (
        leg0.sub.hotkey, leg0.selection, leg0.base_king_ref, leg0.base_era)
    assert block == leg0.funding["started_block"] and got["tag"].startswith("expert-")
    e = sc.fq.get(wb18.HK_E)
    assert (e.status, e.attempt_id) == ("in_flight", attempt)   # never re-charged
    for _ in range(2):
        sc.tick(1)
    assert len(sc.ops.reattached) == 1          # re-attached once
    done(leg, ws2.pointer("leg-reattached"), "corpus", 1500, "cpu")
    assert [f["leg"]["sub"]["hotkey"] for f in sc.driver.st.finished] == [wb18.HK_E]
    assert sc.driver.st.attached == {}
    assert sc.fq.get(wb18.HK_E).attempt_id == attempt


def test_dead_pod_requeues_like_a_restart_requeue(tmp_path, monkeypatch):
    sc, wall = _running_leg(tmp_path, monkeypatch)
    wall.t += 60
    sc.restart_all()
    sc.tick(1)
    leg, _era, _block, done, _rec = sc.ops.reattached[0]
    # the funded path refused: the ledgered pod is gone (nothing rented)
    done(leg, None, failure=leg_reattach.refusal_failure(
        f"{leg_reattach.REATTACH_REFUSED}: the leg's ledgered pod is gone"))
    e = sc.fq.get(wb18.HK_E)
    assert e.status == "queued" and e.operator_billed
    assert e.last_error_class == "no_capacity"
    assert sc.driver.queue.open == [] and sc.driver.st.finished == []
    assert sc.driver.st.attached == {}


def test_expired_grace_requeues_without_a_reattach(tmp_path, monkeypatch):
    sc, wall = _running_leg(tmp_path, monkeypatch)
    attempt = sc.fq.get(wb18.HK_E).attempt_id
    wall.t += 901                               # [round] dispatch_reattach_grace_seconds
    sc.restart_all()
    assert sc.driver.live_hotkeys() == set()
    sc.tick(1)
    assert sc.ops.reattached == []
    # today's restart requeue, unchanged: requeued operator-billed, then
    # dispatched afresh as a new attempt
    e = sc.fq.get(wb18.HK_E)
    assert e.operator_billed and e.attempt_id != attempt
    assert len(sc.ops.dispatched) == 2


def test_heartbeat_keeps_a_long_leg_reattachable(tmp_path, monkeypatch):
    sc, wall = _running_leg(tmp_path, monkeypatch)
    for _ in range(4):                          # the live process keeps ticking
        wall.t += 600
        sc.tick(1)
    rec = next(iter(_on_disk(tmp_path)["attached"].values()))
    assert rec["seen_at"] == wall.t and rec["dispatched_at"] < wall.t - 2000
    wall.t += 120
    sc.restart_all()
    sc.tick(1)
    assert len(sc.ops.reattached) == 1 and len(sc.ops.dispatched) == 1


def test_ops_without_reattach_keep_the_old_requeue(tmp_path, monkeypatch):
    sc, wall = _running_leg(tmp_path, monkeypatch)
    del sc.ops.reattach_leg
    attempt = sc.fq.get(wb18.HK_E).attempt_id
    sc.restart_all()
    sc.tick(1)
    assert sc.ops.reattached == []
    assert sc.fq.get(wb18.HK_E).attempt_id != attempt and len(sc.ops.dispatched) == 2


# ── lane C: the same pattern ─────────────────────────────────────────────────


def _lane_c(tmp: Path, wall, *, reattach: bool = True):
    from cascade.trainer.lane_c_driver import LaneCDriver

    q = types.SimpleNamespace(requeued=[], in_flight=lambda: [])

    def requeue(hk, **kw):
        q.requeued.append((hk, kw))
        return True

    q.requeue = requeue
    ops = types.SimpleNamespace(queue=lambda: q, spawned=[],
                                spawn=lambda fn, name: ops.spawned.append(name))
    if reattach:
        ops.supports_reattach = True
    cfg = types.SimpleNamespace(round=types.SimpleNamespace(dispatch_reattach_grace_seconds=900))
    drv = LaneCDriver(cfg, ops, tmp / "lane_c_driver.json")
    drv.clock = wall
    return drv, ops, q


def _open_c_leg(drv):
    leg = {"hotkey": "5LaneC", "uid": 3, "target": "mkt:market", "ref": "r", "entry_ref": "e",
           "commit_block": 1, "base_pointer": "p", "base_era": 2, "started_block": 10,
           "attempt_id": "a1", "train_time": 1.0}
    drv.st.open.append(leg)
    drv._record_attach(leg)
    drv.save()
    return leg


def test_lane_c_restart_reattaches_inside_the_grace(tmp_path):
    wall = _Wall()
    drv, ops, q = _lane_c(tmp_path, wall)
    leg = _open_c_leg(drv)
    wall.t += 100
    drv2, ops2, q2 = _lane_c(tmp_path, wall)
    assert drv2.live_hotkeys() == {"5LaneC"}
    with drv2._lock:
        drv2._recovered = True
        drv2.reconcile(q2, startup=True)
    assert drv2.st.open == [leg] and q2.requeued == []
    assert [x["hotkey"] for x in drv2._to_reattach] == ["5LaneC"]


def test_lane_c_expired_grace_or_refusal_requeues(tmp_path):
    wall = _Wall()
    drv, _, _ = _lane_c(tmp_path, wall)
    _open_c_leg(drv)
    wall.t += 1000
    drv2, _, q2 = _lane_c(tmp_path, wall)
    with drv2._lock:
        drv2._recovered = True
        drv2.reconcile(q2, startup=True)
    assert drv2.st.open == [] and q2.requeued[0][1]["operator_billed"] is True

    (tmp_path / "b").mkdir()
    drv3, _, q3 = _lane_c(tmp_path / "b", _Wall())
    leg = _open_c_leg(drv3)
    drv3._leg_done(leg, failure=leg_reattach.refusal_failure("the leg's ledgered pod is gone"))
    hk, kw = q3.requeued[0]
    assert hk == "5LaneC" and kw["operator_billed"] is True and kw["burn_attempt"] is False
    assert kw["error_class"] == "no_capacity" and drv3.st.attached == {}


def test_lane_c_state_is_byte_identical_without_reattach_support(tmp_path):
    wall = _Wall()
    drv, _, _ = _lane_c(tmp_path, wall, reattach=False)
    _open_c_leg(drv)
    assert "attached" not in json.loads((tmp_path / "lane_c_driver.json").read_text())


# ── the attach-only guards in the funded path ───────────────────────────────


def test_rent_refusal_never_allows_a_fresh_rent_when_attach_only():
    assert leg_reattach.rent_refusal(prior_alloc=False, pod_live=None) == ""   # normal leg
    with leg_reattach.reattach_only():
        assert leg_reattach.attach_only()
        assert leg_reattach.rent_refusal(prior_alloc=True, pod_live=True) == ""
        for prior, live in ((False, None), (True, False), (True, None)):
            assert leg_reattach.is_refusal(
                leg_reattach.rent_refusal(prior_alloc=prior, pod_live=live))
    assert not leg_reattach.attach_only()


def test_dispatcher_attaches_to_a_running_run_and_never_relaunches(monkeypatch):
    import cascade.trainer.remote as remote_mod
    from cascade.trainer.remote import (
        DETACHED_STDERR_MARK,
        ReattachMissing,
        RemoteDispatcher,
        RemoteHost,
    )

    monkeypatch.setattr(remote_mod, "_SLEEP", lambda s: None)
    calls: list[str] = []

    def runner_for(steps):
        def run(argv, timeout, stdin=None):
            calls.append(argv[-1])
            step = steps.pop(0)
            return types.SimpleNamespace(returncode=0, stdout=step, stderr="")
        return run

    host = RemoteHost(name="pod", host="10.0.0.1", workdir="/root/cascade")
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                            _runner=runner_for([""]))
    with leg_reattach.reattach_only(), pytest.raises(ReattachMissing) as ei:
        disp.dispatch(host, gen_ref="g", uid=7, hotkey="hk" * 24, role="expert",
                      domain="energy", expert_selection={0: [1]}, base_seed=1, block=1)
    assert leg_reattach.is_refusal(str(ei.value))
    assert len(calls) == 1 and "setsid nohup" not in calls[0]       # find only, no launch

    calls.clear()
    from tests.unit.test_detached_dispatch import _receipt_stdout

    rd = "/root/cascade/_train_work/_dispatch/expert-hkhkhkhkhkhk-1-ab"
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1, _runner=runner_for(
        [f"ATTACH:{rd}\n", "EXIT:0\n",
         _receipt_stdout("expert") + DETACHED_STDERR_MARK + "\n"]))
    with leg_reattach.reattach_only():
        try:
            disp.dispatch(host, gen_ref="g", uid=7, hotkey="hk" * 24, role="expert",
                          domain="energy", expert_selection={0: [1]}, base_seed=1, block=1)
        except ReattachMissing:                 # pragma: no cover — the regression
            raise
        except Exception:  # noqa: BLE001 — the fixture receipt's shape is not the point
            pass
    assert calls[0].startswith("cd ") and "ATTACH:" in calls[0]       # the find
    assert not any("setsid nohup" in c for c in calls)                # attached, no launch
    assert any("EXIT:" in c for c in calls[1:])                       # polled the run


def test_runner_ops_reattach_refuses_another_seed_epoch(tmp_path):
    """The production ``reattach_leg``: a leg whose seed epoch moved (another
    pod name and run tag) is refused before anything rents — requeued as a
    restart requeue (operator-billed, unburned)."""
    import threading

    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    era = types.SimpleNamespace(start_block=100, index=2)
    runner = types.SimpleNamespace(
        cfg=types.SimpleNamespace(lane_b=types.SimpleNamespace(leg_funding="miner")),
        _rolling_sched=types.SimpleNamespace(state=types.SimpleNamespace(current=era)),
        _rolling_seeds=lambda e: types.SimpleNamespace(base_seed=5))
    ops = RunnerLaneBOps(runner, client=None, window_source=None)
    leg = OpenLeg(sub=LaneBSubmission("5Hk", 7, "g", "energy", 5), selection={0: [1]},
                  base_king_ref="k", base_era=2, funding={"entry_ref": "e", "started_block": 9})
    got, fired = [], threading.Event()

    def done(leg, ptr, *a, failure=None, **kw):
        got.append((ptr, failure))
        fired.set()

    ops.reattach_leg(leg, 2, 9, done, {"base_seed": "4"})
    assert fired.wait(10)
    ptr, failure = got[0]
    assert ptr is None and leg_reattach.is_refusal(failure)
    assert failure.operator_billed and failure.error_class == "no_capacity" and not failure.burn
