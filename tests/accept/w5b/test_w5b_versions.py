"""(RV4) One version per judged window; the signed report names every
snapshot it judged windows from (plan rev6 §1.3 "Evidence accumulates, then
freezes", §1.5, invariant 4 "forward-only cannot be violated").

A window id `<series_id>:<target_start_ts>:h64` is stable across snapshots,
but its clocks, values and its feed's measured lag are not. README "One
version per window" states the contract:

* a judged window is scored, and its report row (`min_first_seen_ts`,
  `mirror_lag_hours`, ...) taken, from ONE snapshot's version: the EARLIEST
  loaded snapshot that admits it for the cohort's max(T). Never admission
  from one snapshot with clocks, values or lag from another;
* a window no loaded snapshot admits is never judged;
* no judged window carries a pre-bound clock or an over-cap lag;
* `report["pool"]["snapshots"]` lists every snapshot a judged window came
  from (`snapshot`, `sha256`, `first_seen_roots`), each `windows[]` row names
  its snapshot (`snapshot_sha256`), and the report's two first-seen root lists
  (`pool.first_seen_roots`, `cohorts[].first_seen_roots`) agree.

On main the newest snapshot's version is scored, so the two-version tests
fail on their first `assert` (a pre-bound `min_first_seen_ts`), and the report
has no `pool.snapshots` / `windows[].snapshot_sha256` (an `assert`). The
"refused in every snapshot" test passes on main; it keeps a fix honest.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import (
    BOUND,
    BOUNDARY_1,
    CHALLENGER_UID,
    HORIZON,
    HOUR,
    HOUSE_T,
    T0,
    digest,
    entry,
    feeds_block,
    fixture_key,
    forecasts,
    house,
    lane_d_cfg,
    record,
    snapshot_records,
    target_values,
    wids,
    window_id,
)

SIX = {"s0": 6, "s1": 6, "s2": 6, "s3": 6}       # 24 windows, 4 sources: the floors
THREE = {"s0": 3, "s1": 3, "s2": 3, "s3": 3}     # 12 windows: half the floors
OVER_LAG = 9.0                                   # above max_mirror_lag_hours = 6


def _write_pool(directory: Path, records: list[dict], snapshot: str, *, roots=(),
                lag: dict[str, float] | None = None,
                shift: dict[str, float] | None = None) -> Path:
    """`helpers.write_pool` plus the knobs a revised snapshot needs: B12
    `first_seen_roots`, a per-feed measured-lag override, and a per-series
    shift of the target values (a source revision)."""
    directory.mkdir(parents=True, exist_ok=True)
    feeds = feeds_block(records)
    for f, hours in (lag or {}).items():
        feeds[f] = {**feeds[f], "mirror_lag_hours": float(hours)}
    meta = {"schema": "pool_d/1", "snapshot": snapshot, "horizon": HORIZON,
            "feeds": feeds, "series": records, "first_seen_roots": sorted(roots)}
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    hist, tgt = {}, {}
    for r in records:
        h, y = target_values(r["series_id"], r["target_ts"][0])
        hist[r["series_id"]] = h
        tgt[r["series_id"]] = y + float((shift or {}).get(r["series_id"], 0.0))
    np.savez(directory / "hist.npz", **hist)
    np.savez(directory / "tgt.npz", **tgt)
    return directory


def _load(directory: Path) -> dict:
    from cascade.validator.lane_d import load_pool

    return load_pool(directory, HORIZON)


def _verdicts(directory: Path, cfg) -> dict[str, str | None]:
    from cascade.validator.lane_d_forward import eligible_windows

    return {v.window_id: (None if v.reason is None else str(v.reason))
            for v in eligible_windows(_load(directory), t_entry=T0, t_king=HOUSE_T, cfg=cfg)}


def _settle(cfg, pool_dir, fc, e, h, state_path, prior=()):
    from cascade.validator.lane_d import settle_lane_d

    kw = {"prior_pool_dirs": [Path(p) for p in prior]} if prior else {}
    return settle_lane_d(boundary_ts=2_000_000_000 + 12 * int(BOUNDARY_1), cfg=cfg, boundary=BOUNDARY_1, pool_dir=pool_dir,
                         entries=[dict(e)], house=dict(h), forecasts=fc,
                         uid_of={e["hotkey"]: CHALLENGER_UID}, state_path=state_path,
                         signer=fixture_key("signer"), **kw)


def _two_versions(tmp_path):
    """Snapshot A admits all 24 windows. The newest snapshot B holds the same
    window ids, revised: x's first-seen moves before the bound (B refuses it
    `publish_ahead`), y's target values are revised (still eligible in B),
    and feed s2's measured lag rises over the cap (B refuses every s2 window
    `over_lag`)."""
    recs = snapshot_records("v", SIX, day=0)
    x = recs[0]
    y = next(r for r in recs if r["source"] == "s1")
    pa = _write_pool(tmp_path / "pool_a", recs, "w5b-v-a", roots=[digest("root a")])
    revised = copy.deepcopy(recs)
    revised[0]["first_seen_ts"] = [BOUND - HOUR] * HORIZON
    pb = _write_pool(tmp_path / "pool_b", revised, "w5b-v-b", roots=[digest("root b")],
                     lag={"s2": OVER_LAG}, shift={y["series_id"]: 5.0})
    return recs, x, y, pa, pb


def _judged(report) -> dict[str, dict]:
    return {w["window_id"]: w for w in report["windows"]}


def _cells(report) -> dict[tuple[str, str], tuple[float, float]]:
    return {(r["window_id"], r["digest"]): (r["crps"], r["mase"]) for r in report["rows"]}


def test_a_two_version_window_is_scored_from_its_earliest_admitting_snapshot(tmp_path):
    cfg = lane_d_cfg()
    recs, x, y, pa, pb = _two_versions(tmp_path)
    xid, yid = window_id(x["series_id"], x["target_ts"]), window_id(y["series_id"], y["target_ts"])
    s2 = {window_id(r["series_id"], r["target_ts"]) for r in recs if r["source"] == "s2"}
    # fixture sanity (W5a, already on main): A admits all, B refuses x and s2
    va, vb = _verdicts(pa, cfg), _verdicts(pb, cfg)
    assert all(r is None for r in va.values()) and len(va) == 24
    assert vb[xid] == "publish_ahead" and vb[yid] is None
    assert {vb[w] for w in s2} == {"over_lag"}

    e, h = entry("c1"), house()
    fc = forecasts(recs, [e["digest"]], h["digest"])
    rep = _settle(cfg, pb, fc, e, h, tmp_path / "state.json", prior=[pa]).report
    ref = _settle(cfg, pa, fc, e, h, tmp_path / "ref.json").report   # A alone
    judged = _judged(rep)

    # invariant 4: no judged window carries a pre-bound clock or an over-cap lag
    for wid, w in sorted(judged.items()):
        assert w["min_first_seen_ts"] >= BOUND, (
            f"{wid} judged with min_first_seen_ts {w['min_first_seen_ts']} < bound {BOUND}: "
            "scored from a version §1.3 refuses")
        assert w["min_target_ts"] >= BOUND, wid
        assert w["mirror_lag_hours"] is None or w["mirror_lag_hours"] <= 6.0, (
            f"{wid} judged with mirror_lag_hours {w['mirror_lag_hours']} over the cap")

    # A is the earliest admitting snapshot of every window: all 24 are judged,
    # and each one exactly as A alone judges it (row values and scores).
    assert set(judged) == wids(recs)
    assert judged == _judged(ref)
    got, want = _cells(rep), _cells(ref)
    assert set(got) == set(want)
    for k in sorted(want):
        assert got[k] == pytest.approx(want[k], rel=1e-12, abs=0.0), (
            f"{k} scored on values no admitting snapshot holds")


def test_a_window_refused_in_every_snapshot_is_never_judged(tmp_path):
    cfg = lane_d_cfg()
    recs = snapshot_records("r", SIX, day=0)
    z = record("r-z", "s0", BOUND + 3 * HOUR + 40 * HOUR)
    z["feed"] = "zfeed"
    zid = window_id(z["series_id"], z["target_ts"])
    za = copy.deepcopy(z)
    za["first_seen_ts"] = [BOUND - HOUR] * HORIZON          # A: publish_ahead
    pa = _write_pool(tmp_path / "pool_a", recs + [za], "w5b-r-a", roots=[digest("ra")])
    pb = _write_pool(tmp_path / "pool_b", recs + [z], "w5b-r-b", roots=[digest("rb")],
                     lag={"zfeed": OVER_LAG})              # B: over_lag
    assert _verdicts(pa, cfg)[zid] == "publish_ahead"
    assert _verdicts(pb, cfg)[zid] == "over_lag"

    e, h = entry("c1"), house()
    fc = forecasts(recs + [z], [e["digest"]], h["digest"])
    state_path = tmp_path / "state.json"
    rep = _settle(cfg, pb, fc, e, h, state_path, prior=[pa]).report
    assert set(_judged(rep)) == wids(recs)
    assert zid not in {r["window_id"] for r in rep["rows"]}
    assert all(zid not in c["window_ids"] for c in rep["cohorts"])
    state = json.loads(state_path.read_text())
    fs = state["entries"][e["digest"]].get("frozen_set") or {}
    assert zid not in fs.get("window_ids", [])


def _assert_pool_names(report, loaded: list[dict], want_from: dict[str, str]):
    """README "the report names every judged snapshot": `pool.snapshots`
    holds every contributing snapshot (only loaded ones, once each), every
    judged window names its snapshot, and the root lists agree."""
    snaps = report["pool"].get("snapshots")
    assert isinstance(snaps, list), (
        "report.pool.snapshots missing: the signed report must list every snapshot "
        "it judged windows from (new lane-d-report/1 field, W7a)")
    by_sha = {p["sha256"]: p for p in loaded}
    listed = [s["sha256"] for s in snaps]
    assert len(listed) == len(set(listed)), "a snapshot listed twice"
    assert set(listed) <= set(by_sha), "pool.snapshots names a snapshot that was not loaded"
    for s in snaps:
        p = by_sha[s["sha256"]]
        assert s["snapshot"] == p["snapshot"]
        assert sorted(s["first_seen_roots"]) == sorted(p["first_seen_roots"])
    for wid, w in sorted(_judged(report).items()):
        assert w.get("snapshot_sha256") == want_from[wid], (
            f"{wid}: windows[].snapshot_sha256 must name the snapshot its version came from")
    assert set(want_from.values()) <= set(listed)
    roots = sorted({r for s in snaps for r in s["first_seen_roots"]})
    assert report["pool"]["first_seen_roots"] == roots, (
        "pool.first_seen_roots must be the union of the listed snapshots' roots")
    for c in report["cohorts"]:
        assert c["first_seen_roots"] == roots, (
            "cohorts[].first_seen_roots disagrees with pool.first_seen_roots")


def test_report_names_every_snapshot_of_a_two_snapshot_union(tmp_path):
    cfg = lane_d_cfg()
    a, b = snapshot_records("a", THREE, 0), snapshot_records("b", THREE, 2)
    pa = _write_pool(tmp_path / "pool_a", a, "w5b-u-a", roots=[digest("ua")])
    pb = _write_pool(tmp_path / "pool_b", b, "w5b-u-b",
                     roots=[digest("ub1"), digest("ub2")])
    la, lb = _load(pa), _load(pb)
    e, h = entry("c1"), house()
    fc = forecasts(a + b, [e["digest"]], h["digest"])
    rep = _settle(cfg, pb, fc, e, h, tmp_path / "state.json", prior=[pa]).report
    want_from = {**{w: la["sha256"] for w in wids(a)}, **{w: lb["sha256"] for w in wids(b)}}
    assert set(_judged(rep)) == set(want_from)           # union of both, judged
    _assert_pool_names(rep, [la, lb], want_from)
    assert {la["sha256"], lb["sha256"]} <= {s["sha256"] for s in rep["pool"]["snapshots"]}


def test_report_names_the_admitting_snapshot_of_a_two_version_window(tmp_path):
    cfg = lane_d_cfg()
    recs, _x, _y, pa, pb = _two_versions(tmp_path)
    la, lb = _load(pa), _load(pb)
    e, h = entry("c1"), house()
    fc = forecasts(recs, [e["digest"]], h["digest"])
    rep = _settle(cfg, pb, fc, e, h, tmp_path / "state.json", prior=[pa]).report
    _assert_pool_names(rep, [la, lb], {w: la["sha256"] for w in wids(recs)})
