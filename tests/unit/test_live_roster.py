"""Live funded roster: each in-flight / waiting leg carries a phase + reason,
``funded/latest.json`` refreshes between settlements (rate-limited, change-
driven, heartbeat), and the per-round audit record is never rewritten."""

from __future__ import annotations

import argparse
import threading
from types import SimpleNamespace

from cascade.funding.queue import FundedQueue
from cascade.miner import cli as cli_mod
from cascade.trainer import rolling as R
from cascade.trainer.loop import TrainerRunner

# ── phase rule ──────────────────────────────────────────────────────────────

def test_leg_phase_matrix():
    lp = R.leg_phase
    assert lp("in_flight", "", 0, pod_live=True, finished=False) == "training"
    assert lp("in_flight", "no_capacity", 0, pod_live=False, finished=False) == "waiting_gpu"
    assert lp("in_flight", "", 0, pod_live=False, finished=False) == "starting"
    assert lp("in_flight", "", 0, pod_live=True, finished=True) == "benching"
    assert lp("in_flight", "", 0, pod_live=False, finished=True) == "awaiting_settlement"
    assert lp("queued", "deferred", 0, pod_live=False, finished=False) == "deferred"
    assert lp("queued", "infra", 1, pod_live=False, finished=False) == "requeued"
    assert lp("queued", "no_capacity", 0, pod_live=False, finished=False) == "requeued"
    assert lp("queued", "", 0, pod_live=False, finished=False) == "queued"
    assert set(R.PHASE_DETAIL) >= {"training", "waiting_gpu", "requeued", "deferred",
                                   "benching", "awaiting_settlement", "starting", "queued"}


# ── roster content ──────────────────────────────────────────────────────────

class _Ops:
    def __init__(self, q, phases):
        self.q, self.phases, self.published = q, phases, []

    def queue(self):
        return self.q

    def leg_phases(self):
        return self.phases

    def publish_roster(self, round_id, roster, *, live=False):
        self.published.append((round_id, roster, live))


def _sched(tmp_path, pods, finished=(), round_id="R7", running=()):
    q = FundedQueue(tmp_path / "q.json", entry_ttl_seconds=36 * 3600.0)
    obj = SimpleNamespace(ops=_Ops(q, pods), _lock=threading.Lock(),
                          _threads={hk: object() for hk in running},
                          state=SimpleNamespace(finished=list(finished), rents=[],
                                                last_published_round_id=round_id))
    obj._roster = lambda rid, ready, queue: R.RollingScheduler._roster(obj, rid, ready, queue)
    return obj, q


def test_roster_entries_carry_phase_and_reason(tmp_path):
    obj, q = _sched(tmp_path, pods={})
    q.add("TRAINhotkey01xx", "repo@sha256:a", reveal_block=10)
    q.add("WAITGPUhotkey1x", "repo@sha256:b", reveal_block=11)
    q.add("REQUEUEhotkey1x", "repo@sha256:c", reveal_block=12)
    e_train, e_wait = q.get("TRAINhotkey01xx"), q.get("WAITGPUhotkey1x")
    # phases are RECORDED where they happen (pod ready / capacity wait)
    obj.ops.phases = {"TRAINhotkey01xx": ("training", 1_792_000_000.0, "H100"),
                      "WAITGPUhotkey1x": ("waiting_gpu", 1_792_000_100.0, "")}
    entries = {e.hotkey: e for e in q.entries()}
    entries["TRAINhotkey01xx"] = R.replace(e_train, status="in_flight")
    entries["WAITGPUhotkey1x"] = R.replace(e_wait, status="in_flight",
                                           last_error_class="no_capacity")
    entries["REQUEUEhotkey1x"] = R.replace(q.get("REQUEUEhotkey1x"), attempts=1,
                                           last_error_class="infra", last_error="pod 1.2.3.4 died")
    fake_q = SimpleNamespace(entries=lambda: list(entries.values()))
    roster = R.RollingScheduler._roster(obj, "R7", [], fake_q)
    fly = {e["hotkey"]: e for e in roster["in_flight"]}
    assert fly["TRAINhotkey01xx"]["phase"] == "training" and fly["TRAINhotkey01xx"]["since"]
    assert fly["WAITGPUhotkey1x"]["phase"] == "waiting_gpu"
    assert fly["TRAINhotkey01xx"]["since"] == "2026-10-14T17:46:40Z"
    assert "GPU" in fly["WAITGPUhotkey1x"]["phase_detail"]
    wait = {e["hotkey"]: e for e in roster["waiting"]}
    assert wait["REQUEUEhotkey1x"]["phase"] == "requeued"
    assert wait["REQUEUEhotkey1x"]["attempts"] == 1
    # only the CLASS is public — never the raw error text (IPs, pod names)
    assert "1.2.3.4" not in str(roster)


def test_adopted_pod_after_restart_publishes_no_since(tmp_path):
    obj, q = _sched(tmp_path, pods={})
    q.add("ADOPThotkey01xx", "repo@sha256:a", reveal_block=10)
    e = R.replace(q.get("ADOPThotkey01xx"), status="in_flight", started_block=9334783)
    obj.ops.phases = {"ADOPThotkey01xx": ("training", 0.0, "RTX3090")}   # since unknown
    row = R.RollingScheduler._roster(obj, "R7", [], SimpleNamespace(entries=lambda: [e]))
    row = row["in_flight"][0]
    assert row["phase"] == "training" and row["since"] == ""
    assert row["started_block"] == 9334783


def test_in_flight_without_a_recorded_phase_is_not_called_training(tmp_path):
    obj, q = _sched(tmp_path, pods={})
    q.add("NOPHASEhotkeyxx", "repo@sha256:a", reveal_block=10)
    e = R.replace(q.get("NOPHASEhotkeyxx"), status="in_flight")
    rows = R.RollingScheduler._roster(obj, "R7", [], SimpleNamespace(entries=lambda: [e]))
    assert rows["in_flight"][0]["phase"] == "starting"


def test_finished_leg_is_benching_then_awaiting_settlement(tmp_path):
    fin = SimpleNamespace(hotkey="DONEhotkey012xx")
    obj, q = _sched(tmp_path, pods={}, finished=[fin], running=["DONEhotkey012xx"])
    q.add("DONEhotkey012xx", "repo@sha256:a", reveal_block=10)
    e = R.replace(q.get("DONEhotkey012xx"), status="in_flight")
    fq = SimpleNamespace(entries=lambda: [e])
    assert R.RollingScheduler._roster(obj, "R7", [], fq)["in_flight"][0]["phase"] == "benching"
    obj._threads = {}
    assert (R.RollingScheduler._roster(obj, "R7", [], fq)["in_flight"][0]["phase"]
            == "awaiting_settlement")


# ── live publish cadence ────────────────────────────────────────────────────

def test_live_roster_is_rate_limited_change_driven_with_heartbeat(tmp_path):
    obj, q = _sched(tmp_path, pods={})
    q.add("LIVEhotkey012xx", "repo@sha256:a", reveal_block=10)
    pub = obj.ops.published
    R.RollingScheduler._maybe_publish_live_roster(obj, 1000.0)
    assert len(pub) == 1 and pub[0][2] is True and pub[0][0] == "R7"
    assert pub[0][1]["live"] is True and pub[0][1]["updated_at"].endswith("Z")
    R.RollingScheduler._maybe_publish_live_roster(obj, 1030.0)    # < 60 s: skipped
    R.RollingScheduler._maybe_publish_live_roster(obj, 1100.0)    # unchanged: skipped
    assert len(pub) == 1
    q.add("NEWhotkey0123xx", "repo@sha256:b", reveal_block=11)    # change ⇒ publish
    R.RollingScheduler._maybe_publish_live_roster(obj, 1200.0)
    assert len(pub) == 2
    R.RollingScheduler._maybe_publish_live_roster(obj, 1200.0 + R.ROSTER_LIVE_HEARTBEAT_SECONDS)
    assert len(pub) == 3                                           # heartbeat


def test_no_live_roster_before_the_first_settlement(tmp_path):
    obj, _q = _sched(tmp_path, pods={}, round_id="")
    R.RollingScheduler._maybe_publish_live_roster(obj, 1000.0)
    assert obj.ops.published == []


def test_live_publish_failure_never_raises(tmp_path):
    obj, _q = _sched(tmp_path, pods={})

    def boom(*a, **k):
        raise RuntimeError("s3 down")
    obj.ops.publish_roster = boom
    R.RollingScheduler._maybe_publish_live_roster(obj, 1000.0)    # must not raise


# ── trainer-side publish keys + ledger lookup ───────────────────────────────

def test_live_publish_writes_latest_only(monkeypatch):
    written = []
    monkeypatch.setattr("cascade.shared.heat_status._publish_public_json",
                        lambda store, key, doc: written.append(key))
    runner = SimpleNamespace(cfg=SimpleNamespace(round=SimpleNamespace(funded_pods="rent")),
                             manifest_store=lambda: object())
    TrainerRunner._rolling_publish_roster(runner, "R9", {"mode": "rolling"}, live=True)
    assert written == ["funded/latest.json"]
    written.clear()
    TrainerRunner._rolling_publish_roster(runner, "R9", {"mode": "rolling"})
    assert written == ["funded/round-R9.json", "funded/latest.json"]


# ── miner CLI ───────────────────────────────────────────────────────────────

def test_cascade_queue_prints_phase_and_reason(capsys, monkeypatch):
    doc = {"round_id": "r1", "mode": "rolling", "live": True,
           "updated_at": "2026-10-22T01:00:00Z", "seated": [], "terminal": [],
           "in_flight": [{"hotkey": "hkFly", "reveal_block": 5, "era_index": 2593,
                          "target_boundary": 9335700, "phase": "waiting_gpu",
                          "phase_detail": R.PHASE_DETAIL["waiting_gpu"],
                          "since": "2026-10-22T00:10:00Z", "attempts": 1}],
           "waiting": [{"hotkey": "hkDef", "reveal_block": 6, "phase": "deferred",
                        "phase_detail": R.PHASE_DETAIL["deferred"]}]}
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json",
                        lambda storage, key, **kw: doc)
    monkeypatch.setattr(cli_mod, "load_chain_config",
                        lambda p: type("C", (), {"storage": None})())
    assert cli_mod._cmd_queue(argparse.Namespace(
        chain_toml=None, round=None, intake=None, hotkey="hkFly")) == 0
    out = capsys.readouterr().out
    assert "funded roster (live), updated 2026-10-22T01:00:00Z" in out
    assert "← you  [waiting_gpu, since 2026-10-22T00:10:00Z, attempts 1] — no GPU capacity" in out
    assert "hkDef  reveal=6  [deferred]" in out
