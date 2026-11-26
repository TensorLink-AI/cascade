"""W9 (rev6.1): window eligibility re-derived from the judged snapshot itself
(plan rev6.1 §3.0 W9 row: "Refuses any judged snapshot in which a series
lacks `frontier` (absent or `null`) or a feed is not listed in `feeds` (W6b).
It re-derives S0 presence (`absent_from_s0`), the backfill rule and
mirror-lag eligibility (`over_lag`, `forecast_product`) from the snapshot
itself, without importing `cascade/validator/lane_d_forward` (invariant 12;
RV3 BLOCKING 1); its reason labels follow §1.3 (`publish_ahead` = first-seen
fails, `late_mirror` = data time fails)"; §1.3; T8; README "Snapshots").

Two surfaces:

* `cascade.audit.lane_d.window_refusal(series, feeds, bound, cfg)`: the §1.3
  rules over one `pool_d/1` series record, re-implemented in the audit, with
  W5a's labels and precedence;
* `check_lane_d(..., pool_dirs=[...])`: with the snapshots the report names,
  every judged snapshot is valid, every judged window's clock record equals
  the snapshot's and the window is eligible at its cohort's bound.

On main `window_refusal` does not exist (`cannot import name`, in PATHS) and
`check_lane_d` takes no `pool_dirs` (an `assert`).
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h

H = h.w7a.HORIZON
BOUND = 1_790_100_000.0
# rev6.10 / rev6.12: every listed feed pins its effective embargo. These
# cases pin e = 0 (the rev3 rule at bound `b`, byte-identical to the rev6.1
# labels); test_w9_embargo.py covers e > 0.
FEEDS = {"fa": {"mirror_lag_hours": 1.0, "forecast_product": False, "embargo_hours": 0.0},
         "unmeasured": {"mirror_lag_hours": None, "forecast_product": False,
                        "embargo_hours": 0.0},
         "laggy": {"mirror_lag_hours": 6.5, "forecast_product": False, "embargo_hours": 0.0},
         "at-limit": {"mirror_lag_hours": 6.0, "forecast_product": False, "embargo_hours": 0.0},
         "forecast": {"mirror_lag_hours": 1.0, "forecast_product": True, "embargo_hours": 0.0}}


def _series(*, data_start=BOUND + 3600, seen_start=BOUND + 5400, feed="fa", source="src0",
            frontier=((BOUND + 600, BOUND - 3600),), **over) -> dict:
    ts = [int(data_start + i * 3600) for i in range(H)]
    seen = [int(seen_start + i * 3600) for i in range(H)]
    rec = {"series_id": "s-1", "source": source, "freq": "H", "seasonal_period": 24,
           "target_ts": ts, "first_seen_ts": seen, "first_target_ts": ts[0], "end_ts": ts[-1],
           "frontier": None if frontier is None else [list(f) for f in frontier]}
    if feed is not None:
        rec["feed"] = feed
    rec.update(over)
    return rec


def _refusal(rec, feeds=None):
    from cascade.audit.lane_d import window_refusal

    got = window_refusal(copy.deepcopy(rec), copy.deepcopy(FEEDS if feeds is None else feeds),
                         BOUND, h.w7a.lane_d_cfg())
    return None if got is None else str(got)


CASES = [
    ("eligible", {}, None),
    ("unmeasured lag is not a refusal", {"feed": "unmeasured"}, None),
    ("lag exactly at the limit", {"feed": "at-limit"}, None),
    ("the feed key falls back to the source", {"feed": None, "source": "fa"}, None),
    ("short target_ts", {"target_ts": [int(BOUND + 3600)] * (H - 1)}, "missing_clock"),
    ("end_ts absent", {"end_ts": None}, "missing_clock"),
    ("end_ts not the last target", {"end_ts": int(BOUND)}, "missing_clock"),
    ("feed not listed", {"feed": "nowhere"}, "unlisted_feed"),
    ("source not listed (no feed)", {"feed": None, "source": "nowhere"}, "unlisted_feed"),
    ("forecast product", {"feed": "forecast"}, "forecast_product"),
    ("mirror lag over the limit", {"feed": "laggy"}, "over_lag"),
    ("both clocks before the bound", {"data_start": BOUND - 7200, "seen_start": BOUND - 3600,
                                      "frontier": ((BOUND + 600, BOUND - 9000),)},
     "before_margin"),
    ("data time after, first-seen before: publish-ahead",
     {"seen_start": BOUND - 3600}, "publish_ahead"),
    ("first-seen after, data time before: late mirror",
     {"data_start": BOUND - 3600, "frontier": ((BOUND + 600, BOUND - 9000),)}, "late_mirror"),
    ("a clock equal to the bound fails it", {"data_start": BOUND}, "late_mirror"),
    ("frontier absent", {"frontier": None}, "absent_from_s0"),
    ("frontier empty", {"frontier": ()}, "absent_from_s0"),
    ("no frontier record after the bound", {"frontier": ((BOUND - 60, BOUND - 3600),)},
     "absent_from_s0"),
    ("target inside S0: a backfill", {"frontier": ((BOUND + 600, BOUND + 3600),)}, "backfill"),
    ("S0 is the EARLIEST record after the bound",
     {"frontier": ((BOUND + 90_000, BOUND - 3600), (BOUND + 600, BOUND + 7200))}, "backfill"),
    ("precedence: unlisted feed before the clocks", {"feed": "nowhere",
                                                     "data_start": BOUND - 7200}, "unlisted_feed"),
    ("precedence: over_lag before the clocks", {"feed": "laggy", "seen_start": BOUND - 60},
     "over_lag"),
]


@pytest.mark.parametrize("what,over,want", CASES, ids=[c[0] for c in CASES])
def test_window_refusal_rules_and_labels(what, over, want):
    rec = _series(**over)
    assert _refusal(rec) == want, f"{what}: {_refusal(rec)!r} != {want!r}"


def test_window_refusal_frontier_key_absent_is_absent_from_s0():
    rec = _series()
    del rec["frontier"]
    assert _refusal(rec) == "absent_from_s0"


def test_window_refusal_no_feeds_block_fails_closed():
    assert _refusal(_series(), feeds={}) == "unlisted_feed"


def test_window_refusal_labels_are_the_report_enum():
    from cascade.audit.lane_d import window_refusal  # noqa: F401
    from cascade.validator.lane_d_report import WINDOW_REFUSAL_REASONS

    labels = {c[2] for c in CASES if c[2] is not None}
    assert labels <= set(WINDOW_REFUSAL_REASONS), sorted(labels - set(WINDOW_REFUSAL_REASONS))


# ── check_lane_d with the snapshots ─────────────────────────────────────────


@pytest.fixture(scope="module")
def snaps(tmp_path_factory):
    return h.snapshots(tmp_path_factory.mktemp("w9-snapshots"))


def test_honest_report_passes_with_its_snapshot(snaps):
    run = snaps["honest"]
    res = h.audit_snapshots(run["report"], snaps["cfg"], [run["pool"]])
    assert res.status == "PASS", res.detail


def test_honest_report_passes_with_extra_snapshots(snaps):
    run = snaps["honest"]
    res = h.audit_snapshots(run["report"], snaps["cfg"],
                            [snaps["chain"]["pools"][2], run["pool"]])
    assert res.status == "PASS", res.detail


@pytest.mark.parametrize("defect", ["frontier_absent", "frontier_null", "unlisted_feed"])
def test_a_defective_judged_snapshot_is_refused(snaps, defect):
    """The producer failed the defective windows closed and judged the rest:
    the report alone PASSes, but the snapshot it was judged on is invalid."""
    run = snaps[defect]
    h.assert_pass(run["report"], snaps["cfg"], f"fixture: the {defect} report alone")
    res = h.audit_snapshots(run["report"], snaps["cfg"], [run["pool"]])
    assert res.status == "FAIL", (
        f"a judged snapshot with {defect}: check_lane_d {res.status} ({res.detail})")


def test_a_named_snapshot_that_is_not_provided_fails(snaps):
    run = snaps["honest"]
    res = h.audit_snapshots(run["report"], snaps["cfg"], [snaps["chain"]["pools"][2]])
    assert res.status == "FAIL", f"no snapshot matches the report's: {res.status}"


@pytest.mark.parametrize("field,delta", [("min_first_seen_ts", 3600), ("min_target_ts", 3600),
                                         ("mirror_lag_hours", 0.25)])
def test_a_window_record_that_differs_from_the_snapshot_fails(snaps, field, delta):
    """The signed per-window clocks must be the snapshot's: a re-signed report
    restating one (still eligible on its own) is refused once the snapshot is
    read."""
    run = snaps["honest"]
    report = copy.deepcopy(run["report"])
    wid = report["cohorts"][0]["window_ids"][0]
    w = next(x for x in report["windows"] if x["window_id"] == wid)
    assert w[field] is not None, f"fixture: {wid} has no {field}"
    w[field] = w[field] + delta
    report = h.resign(report)
    h.assert_pass(report, snaps["cfg"], f"fixture: {field} restated, report alone")
    res = h.audit_snapshots(report, snaps["cfg"], [run["pool"]])
    assert res.status == "FAIL", f"{field} differs from the snapshot: {res.status}"
