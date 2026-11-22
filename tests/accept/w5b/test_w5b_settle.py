"""W5b through `settle_lane_d` (plan rev6 §1.3, T18, T18e, §3.2 invariants 5
and 10, the W1.5 row "every later WP keeps it green and adds >= 1
assertion").

* union: an entry short of the floors stays `pending` with its accumulated
  window ids in lane_d_state.json, and the next settlement judges it on the
  union of both snapshots (`prior_pool_dirs` carries the earlier one);
* T18: readied in an era's last settlement, it carries into the next era;
  at `until_block` every open entry closes unjudged (`lane_closed`) and unpaid;
* T18e: a dominant source is capped in the judged set, the drop is a
  function of the settlement (same settlement, same drop; another, another);
* invariant 5 flavour: replaying a settlement from the same state gives the
  same rows and the same state;
* invariant 10 pair.

On main `prior_pool_dirs` does not exist: those tests fail at their import of
`cascade.validator.lane_d_frozen` (PATHS). The `until_block` test fails on an
assertion: main judges the entry at the closing settlement.
"""

from __future__ import annotations

import json
import shutil

from .helpers import (
    ARM_BLOCK,
    BOUNDARY_1,
    BOUNDARY_2,
    CHALLENGER_UID,
    ERA,
    ERA_LAST,
    NEXT_ERA_FIRST,
    SECOND_UID,
    SETTLEMENT,
    T0,
    capped_counts,
    entry,
    forecasts,
    house,
    lane_d_cfg,
    per_source,
    set_sha,
    settle,
    snapshot_records,
    wids,
    write_pool,
)

FOUR = {"src0": 3, "src1": 3, "src2": 3, "src3": 3}      # 12 windows: below 24


def _two_snapshots(tmp_path):
    a, b = snapshot_records("a", FOUR, 0), snapshot_records("b", FOUR, 2)
    return (a, write_pool(tmp_path / "pool_a", a, "w5b-a"),
            b, write_pool(tmp_path / "pool_b", b, "w5b-b"))


def _member_wids(report, dg) -> set[str]:
    return {r["window_id"] for r in report["rows"] if r["digest"] == dg}


def _state(path) -> dict:
    return json.loads(path.read_text())


def _without_signature(state: dict) -> dict:
    s = json.loads(json.dumps(state))
    s.pop("last_report", None)
    return s


def _union_chain(tmp_path, b1, b2, e):
    """Settle b1 on snapshot A alone (short of the floors), then b2 on B with A
    as a prior snapshot. Returns (state_path, s1, s2, a, b, house)."""
    cfg = lane_d_cfg()
    h = house()
    a, pa, b, pb = _two_snapshots(tmp_path)
    fc = forecasts(a + b, [e["digest"]], h["digest"])
    uid_of = {e["hotkey"]: CHALLENGER_UID}
    state_path = tmp_path / "state" / "lane_d_state.json"
    s1 = settle(cfg, b1, pa, [e], h, fc, uid_of, state_path)
    mid = _state(state_path)
    s2 = settle(cfg, b2, pb, [e], h, fc, uid_of, state_path, prior=[pa])
    return state_path, s1, mid, s2, a, b, h


# ── union across settlements ────────────────────────────────────────────────


def test_e2e_union_freezes_across_two_settlements(tmp_path):
    import cascade.validator.lane_d_frozen  # noqa: F401

    e = entry("challenger")
    state_path, s1, mid, s2, a, b, h = _union_chain(tmp_path, BOUNDARY_1, BOUNDARY_2, e)
    assert s1.report is not None
    assert _member_wids(s1.report, e["digest"]) == set(), "judged short of the floors"
    rec = mid["entries"][e["digest"]]
    assert rec["status"] == "pending" and rec["reason"] == "pending_floor"
    fs = rec["frozen_set"]
    assert fs["frozen"] is False and fs["window_ids"] == sorted(wids(a))
    assert fs["set_sha256"] is None

    got = _member_wids(s2.report, e["digest"])
    assert got == wids(a) | wids(b), "not judged on the union of both snapshots"
    king = {r["window_id"] for r in s2.report["rows"] if r["digest"] == h["digest"]}
    assert king == got
    (cohort,) = s2.report["cohorts"]
    assert cohort["members"] == [e["digest"]]
    assert sorted(cohort["window_ids"]) == sorted(got)
    assert cohort["set_sha256"] == set_sha(got)
    after = _state(state_path)["entries"][e["digest"]]
    assert after["status"] == "confirm_pending", after
    assert after["confirm"]["first_set_sha256"] == set_sha(got)


def test_t18_entry_readied_in_an_eras_last_settlement_carries_across_eras(tmp_path):
    import cascade.validator.lane_d_frozen  # noqa: F401

    assert ERA_LAST % ERA == 0 and NEXT_ERA_FIRST == ERA_LAST + SETTLEMENT
    e = entry("late-ready", ready_block=ERA_LAST - 200, reveal_block=ERA_LAST - 150)
    state_path, s1, mid, s2, a, b, _ = _union_chain(tmp_path, ERA_LAST, NEXT_ERA_FIRST, e)
    assert mid["entries"][e["digest"]]["status"] == "pending"
    assert mid["entries"][e["digest"]]["frozen_set"]["window_ids"] == sorted(wids(a))
    assert _member_wids(s2.report, e["digest"]) == wids(a) | wids(b), \
        "the accumulated union did not carry across the era boundary"
    assert _state(state_path)["entries"][e["digest"]]["status"] == "confirm_pending"


def test_t18_until_block_closes_pending_entries_unjudged_and_unpaid(tmp_path):
    from cascade.validator.lane_d_state import ENTRY_REASONS

    until = ARM_BLOCK + ERA
    cfg = lane_d_cfg(until_block=until)
    assert cfg.active(until) and not cfg.active(until + SETTLEMENT)
    h = house()
    first, second = entry("first"), entry("second", reveal_ts=T0 + 3600)
    small = snapshot_records("a", FOUR, 0)
    big = snapshot_records("b", {"src0": 6, "src1": 6, "src2": 6, "src3": 6}, 2)
    pa = write_pool(tmp_path / "pool_a", small, "w5b-a")
    pb = write_pool(tmp_path / "pool_b", big, "w5b-b")
    fc = forecasts(small + big, [first["digest"], second["digest"]], h["digest"])
    uid_of = {first["hotkey"]: CHALLENGER_UID, second["hotkey"]: SECOND_UID}
    state_path = tmp_path / "state" / "lane_d_state.json"
    s1 = settle(cfg, BOUNDARY_1, pa, [first], h, fc, uid_of, state_path)
    assert s1.report is not None
    assert _state(state_path)["entries"][first["digest"]]["status"] == "pending"

    # snapshot B alone meets the floors: anything judged here would show rows
    s = settle(cfg, until, pb, [first, second], h, fc, uid_of, state_path)
    assert s.report is not None and s.pay is not None
    judged = {r["digest"] for r in s.report["rows"]}
    assert first["digest"] not in judged and second["digest"] not in judged, \
        "an entry was judged at until_block"
    assert s.report["cohorts"] == []
    state = _state(state_path)
    for e in (first, second):
        rec = state["entries"][e["digest"]]
        assert (rec["status"], rec["reason"]) == ("closed", "lane_closed"), rec
        listed = {x["digest"]: x for x in s.report["entries"]}[e["digest"]]
        assert (listed["status"], listed["reason"]) == ("closed", "lane_closed")
    assert state["king"]["digest"] == h["digest"]
    assert CHALLENGER_UID not in s.pay["court_uids"]
    assert SECOND_UID not in s.pay["court_uids"]

    before = state_path.read_bytes()
    after_close = settle(cfg, until + SETTLEMENT, pb, [first, second], h, fc, uid_of,
                         state_path)
    assert after_close.report is None and after_close.pay is None
    assert state_path.read_bytes() == before
    assert "lane_closed" in ENTRY_REASONS, "the closing reason is not a typed reason"


# ── T18e through the settlement ─────────────────────────────────────────────


DOMINANT = {"big": 40, "a": 10, "b": 10, "c": 10, "d": 10}


def _cap_run(tmp_path, name, boundary):
    import cascade.validator.lane_d_frozen  # noqa: F401

    cfg = lane_d_cfg()
    h = house()
    e = entry("capped")
    recs = snapshot_records("cap", DOMINANT, 0)
    pool = write_pool(tmp_path / f"pool_{name}", recs, "w5b-cap")
    fc = forecasts(recs, [e["digest"]], h["digest"])
    s = settle(cfg, boundary, pool, [e], h, fc, {e["hotkey"]: CHALLENGER_UID},
               tmp_path / name / "lane_d_state.json")
    return s, e, {w: r["source"] for r in recs for w in wids([r])}


def test_t18e_settlement_caps_a_dominant_source_deterministically(tmp_path):
    s1, e, src = _cap_run(tmp_path, "one", BOUNDARY_1)
    got = _member_wids(s1.report, e["digest"])
    assert got, "the capped set (53 windows, 5 sources) meets the floors but was not judged"
    assert per_source(got, src) == capped_counts(DOMINANT, 0.25), \
        "the source with more than max_source_share kept its excess"
    s1b, *_ = _cap_run(tmp_path, "one-again", BOUNDARY_1)
    assert _member_wids(s1b.report, e["digest"]) == got, \
        "the same settlement dropped different windows"
    assert s1b.report["rows_sha256"] == s1.report["rows_sha256"]
    s2, *_ = _cap_run(tmp_path, "two", BOUNDARY_2)
    other = _member_wids(s2.report, e["digest"])
    assert per_source(other, src) == capped_counts(DOMINANT, 0.25)
    assert other != got, "the drop is not seeded by the settlement"


# ── invariant 5 flavour: replay from the same state ─────────────────────────


def test_inv5_replaying_a_settlement_from_the_same_state_is_identical(tmp_path):
    import cascade.validator.lane_d_frozen  # noqa: F401

    cfg = lane_d_cfg()
    h = house()
    e = entry("replay")
    a, pa, b, pb = _two_snapshots(tmp_path)
    fc = forecasts(a + b, [e["digest"]], h["digest"])
    uid_of = {e["hotkey"]: CHALLENGER_UID}
    base = tmp_path / "base" / "lane_d_state.json"
    settle(cfg, BOUNDARY_1, pa, [e], h, fc, uid_of, base)
    results = []
    for n in ("r1", "r2"):
        path = tmp_path / n / "lane_d_state.json"
        path.parent.mkdir(parents=True)
        shutil.copyfile(base, path)
        # the restarted validator re-reads snapshot A both as prior and from state
        s = settle(cfg, BOUNDARY_2, pb, [e], h, fc, uid_of, path, prior=[pa, pa])
        results.append((s, _without_signature(_state(path))))
    (s_a, st_a), (s_b, st_b) = results
    assert s_a.report["rows_sha256"] == s_b.report["rows_sha256"]
    assert st_a == st_b, "replaying the settlement from the same state diverged"
    assert len(_member_wids(s_a.report, e["digest"])) == len(wids(a) | wids(b)), \
        "a window carried in state and in a prior snapshot was counted twice"
    n_member_rows = sum(1 for r in s_a.report["rows"] if r["digest"] == e["digest"])
    assert n_member_rows == 24


# ── invariant 10 pair ───────────────────────────────────────────────────────


def test_union_unchanged_when_unarmed(tmp_path):
    import cascade.validator.lane_d_frozen  # noqa: F401

    unarmed = lane_d_cfg(from_block=0)
    assert not unarmed.active(BOUNDARY_2)
    h = house()
    e = entry("unarmed")
    a, pa, b, pb = _two_snapshots(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    s = settle(unarmed, BOUNDARY_2, pb, [e], h, forecasts(a + b, [e["digest"]], h["digest"]),
               {e["hotkey"]: CHALLENGER_UID}, state_dir / "lane_d_state.json", prior=[pa])
    assert s.report is None and s.pay is None
    assert list(state_dir.iterdir()) == [], "unarmed lane D wrote state"


def test_union_armed_nonempty_lane_d_effect(tmp_path):
    import cascade.validator.lane_d_frozen  # noqa: F401

    e = entry("armed")
    _, s1, _, s2, a, b, _ = _union_chain(tmp_path, BOUNDARY_1, BOUNDARY_2, e)
    assert s1.report is not None and s2.report is not None and s2.pay is not None
    assert len(_member_wids(s2.report, e["digest"])) == 24
