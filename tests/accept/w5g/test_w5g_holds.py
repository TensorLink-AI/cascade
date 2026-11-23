"""W5g: bench holds are bounded by `[lane_d] max_bench_holds` (plan rev6
§10.4: "Crowning held; CRIT after 1; after `max_bench_holds` (4) the
candidate re-queues unjudged. Deliberately not lane A's never-timeout hold";
RV7: "bench holds never expire and aren't capped"). README.md has the
contract.

X clears at its first verdict at boundary 2 (plan rev6.11 (b): a first
clear is the crowning candidate); every later boundary re-uses that
snapshot, so a held X can only be re-gated, never re-duelled. With
`max_bench_holds = N`, N consecutive `bench_hold` outcomes keep the clear in
status `hold`; the next one re-queues X: status `pending`, reason
`bench_hold`, the held clear dropped, not lost, not closed, not crowned.

On main the old code first-clears X at boundary 1 and confirms at boundary
2, so these tests fail on an AssertionError.
"""

from __future__ import annotations

import shutil

import pytest

from .helpers import (
    DAY,
    HOLD_MODES,
    T_X,
    Flow,
    boundary,
    entry,
    entry_of,
    lane_d_cfg,
    read_state,
    report_digest,
    rows_of,
    skips_of,
    verdict_of,
)

# rev6.11 (b): a held first clear stays open as `hold` (no confirm_pending).
HELD_STATUSES = ("hold",)
T_LATE = T_X + 60 * DAY   # revealed after every fixture window: never frozen


def _hold_modes(k: int) -> list[str]:
    """`k` hold modes cycling through every kind of hold (all count)."""
    return [HOLD_MODES[i % len(HOLD_MODES)] for i in range(k)]


def _assert_held(f: Flow, report, n: int):
    rec = f.state_entry()
    assert rec["reason"] == "bench_hold", f"hold {n}: {rec['status']}/{rec['reason']}"
    assert rec["status"] in HELD_STATUSES, \
        f"hold {n} (<= max_bench_holds) must keep the entry held: {rec['status']}"
    e = entry_of(report, f.x["digest"])
    assert (e["status"], e["reason"]) == (rec["status"], rec["reason"])
    assert report["crowned"] is None
    assert f.king() == f.h0["digest"]


def _assert_requeued(f: Flow, report):
    """The hold that exceeds `max_bench_holds`: re-queued unjudged."""
    rec = f.state_entry()
    assert (rec["status"], rec["reason"]) == ("pending", "bench_hold"), \
        (f"after max_bench_holds holds the candidate must re-queue (pending / bench_hold), "
         f"not stay held forever: {rec['status']}/{rec['reason']}")
    assert rec.get("confirm") is None, "a re-queued entry keeps no first clear"
    assert not rec.get("gate_hold"), "a re-queued entry keeps no held clear"
    e = entry_of(report, f.x["digest"])
    assert (e["status"], e["reason"]) == ("pending", "bench_hold")
    assert report["crowned"] is None
    assert report["king_after"] == report["king_before"]
    assert f.king() == f.h0["digest"]
    gate = report.get("gate")
    if gate is not None:
        assert all(r["outcome"] != "pass" for r in gate.get("results", [])), gate


def _assert_requeue_is_judged_afresh(f: Flow, mode: str = "pass"):
    """The settlement after the re-queue, on a fresh snapshot: X is duelled
    again (rows, a verdict at stage `first`), never re-gated on the old clear.
    rev6.11 (b): that first verdict is decided at once, crowned on a passing
    gate (`mode` "pass"), held on a hold mode. Returns the report."""
    snap = max(f.pools) + 1
    report = f.next(mode, snap=snap)
    v = verdict_of(report, f.x["digest"])
    assert v is not None, "a re-queued entry was not judged on the fresh snapshot"
    assert v["stage"] == "first", f"a re-queued entry restarts its duel: {v['stage']}"
    assert v["cleared"] is True, v
    assert rows_of(report, f.x["digest"]), "a re-queued entry must be re-duelled (rows)"
    rec = f.state_entry()
    if mode == "pass":
        assert report["crowned"] == f.x["digest"]
        assert (rec["status"], rec["reason"]) == ("crowned", "crowned")
    else:
        assert report["crowned"] is None
        assert (rec["status"], rec["reason"]) == ("hold", "bench_hold")
    return report


@pytest.mark.parametrize("limit", [0, 2, None], ids=["max0", "max2", "default4"])
def test_holds_are_bounded_by_max_bench_holds(tmp_path, limit):
    cfg = lane_d_cfg() if limit is None else lane_d_cfg(max_bench_holds=limit)
    n_max = int(cfg.max_bench_holds)
    if limit is None:
        assert n_max == 4, "plan §10.4: the default max_bench_holds is 4"
    modes = _hold_modes(n_max + 1)
    f = Flow(tmp_path, mode2=modes[0], cfg=cfg)
    reports = [f.reports[2]]
    for mode in modes[1:]:
        reports.append(f.next(mode))
    for i, report in enumerate(reports[:-1], start=1):
        # (the state file only shows the last one; the report entry is per boundary)
        e = entry_of(report, f.x["digest"])
        assert e["reason"] == "bench_hold" and e["status"] in HELD_STATUSES, (i, e)
        assert report["crowned"] is None
    _assert_requeued(f, reports[-1])
    if n_max > 0:
        # Exhausted on a re-gate: considered and not judged, a typed skip.
        assert skips_of(reports[-1], f.x["digest"]) == ["bench_hold"]
        assert verdict_of(reports[-1], f.x["digest"]) is None
        assert rows_of(reports[-1], f.x["digest"]) == set()
    _assert_requeue_is_judged_afresh(f)


def test_a_hold_within_the_bound_keeps_the_chance(tmp_path):
    """N holds, then a healthy bench: the held clear is re-gated and crowns
    without a new duel (a hold is never charged)."""
    cfg = lane_d_cfg(max_bench_holds=2)
    f = Flow(tmp_path, mode2="host_down", cfg=cfg)
    _assert_held(f, f.reports[2], 1)
    _assert_held(f, f.next("no_bench"), 2)
    report = f.next("pass")
    assert report["crowned"] == f.x["digest"], "a hold within max_bench_holds burned the chance"
    assert rows_of(report, f.x["digest"]) == set(), "a held candidate was re-duelled"
    assert f.king() == f.x["digest"]
    rec = f.state_entry()
    assert (rec["status"], rec["reason"]) == ("crowned", "crowned")


def test_every_kind_of_hold_counts(tmp_path):
    """A missing gate input, a host that is down and a reference fault are
    all `bench_hold` and all count against the one bound."""
    cfg = lane_d_cfg(max_bench_holds=len(HOLD_MODES) - 1)
    f = Flow(tmp_path, mode2=HOLD_MODES[0], cfg=cfg)
    for mode in HOLD_MODES[1:-1]:
        _assert_held(f, f.next(mode), f.n - 1)
    _assert_requeued(f, f.next(HOLD_MODES[-1]))


def test_the_bound_counts_consecutive_holds_of_one_clear(tmp_path):
    """The count belongs to one held clear: a re-queued entry that clears
    again (at its new first verdict) starts a new count."""
    cfg = lane_d_cfg(max_bench_holds=1)
    f = Flow(tmp_path, mode2="no_bench", cfg=cfg)
    _assert_held(f, f.reports[2], 1)
    _assert_requeued(f, f.next("host_down"))
    # snapshot 3: a new first clear, held: hold 1 of the new clear (with
    # max_bench_holds = 1 the old count would re-queue it here).
    report = _assert_requeue_is_judged_afresh(f, mode="reference_fault")
    _assert_held(f, report, 1)
    report = f.next("pass", snap=3)
    assert report["crowned"] == f.x["digest"]
    assert rows_of(report, f.x["digest"]) == set(), "a held candidate was re-duelled"
    assert f.king() == f.x["digest"]


def test_an_exhausted_hold_is_not_charged(tmp_path):
    """Re-queued is not a loss: X keeps no lost/closed record, holds no court
    slot, and other entries are untouched."""
    cfg = lane_d_cfg(max_bench_holds=1)
    late = entry("late", reveal_ts=T_LATE)
    f = Flow(tmp_path, mode2="no_bench", cfg=cfg, extra=[late])
    report = f.next("host_down")
    _assert_requeued(f, report)
    assert all(s["digest"] != f.x["digest"] for s in report["pay"]["court"])
    state = read_state(f.path)
    assert state["entries"][late["digest"]]["reason"] == "pending_floor"
    assert entry_of(report, late["digest"])["reason"] == "pending_floor"



def test_the_hold_count_survives_a_restart(tmp_path):
    """The count is state: a copy of the state directory after the first hold
    (a fresh process) re-queues at the same boundary with the same report."""
    cfg = lane_d_cfg(max_bench_holds=1)
    f = Flow(tmp_path / "a", mode2="no_bench", cfg=cfg)
    shutil.copytree(f.path.parent, tmp_path / "b" / "state")
    g = Flow.__new__(Flow)
    g.__dict__.update(f.__dict__)
    g.path = tmp_path / "b" / "state" / "lane_d_state.json"
    g.reports = dict(f.reports)
    ra = f.next("host_down")
    rb = g.next("host_down")
    _assert_requeued(f, ra)
    _assert_requeued(g, rb)
    assert report_digest(ra) == report_digest(rb)
    assert read_state(f.path)["entries"] == read_state(g.path)["entries"]


def test_exhaustion_is_deterministic_in_entry_order(tmp_path):
    cfg = lane_d_cfg(max_bench_holds=1)
    late = entry("late", reveal_ts=T_LATE)
    fa = Flow(tmp_path / "a", mode2="no_bench", cfg=cfg, extra=[late])
    fb = Flow(tmp_path / "b", mode2="no_bench", cfg=cfg, extra=[late])
    ra = fa.next("host_down")
    rb = fb.next("host_down", entries=list(reversed(fb.entries)))
    _assert_requeued(fa, ra)
    assert report_digest(ra) == report_digest(rb)
    for key in ("entries", "skipped", "judged", "crowned", "verdicts", "rows_sha256"):
        assert ra[key] == rb[key], key
    assert read_state(fa.path)["entries"] == read_state(fb.path)["entries"]
    assert ra["boundary"] == boundary(3)
