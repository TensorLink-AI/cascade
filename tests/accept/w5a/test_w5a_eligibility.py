"""W5a forward eligibility, unit level (plan rev6 §1.3, Proposed changes 1-2,
T8). `eligible_windows(pool, *, t_entry, t_king, cfg)` over a real `pool_d/1`
snapshot written to disk and read back by `load_pool`.

On main each test fails at its first import of
`cascade.validator.lane_d_forward` (in PATHS).
"""

from __future__ import annotations

import ast
import importlib.util
import inspect

import numpy as np
import pytest

from .helpers import (
    HORIZON,
    HOUR,
    LAG_H,
    MARGIN_H,
    REASONS,
    REPO_SCRIPTS,
    T0,
    bound_of,
    case,
    feeds_block,
    hourly,
    lane_d_cfg,
    spec,
    window_id,
    write_pool,
)

KING_T = T0 - 2 * 86_400
BOUND = int(T0 + MARGIN_H * HOUR)       # max(T_E, T_K) + margin with T_E > T_K


def _verdicts(tmp_path, series, *, feeds="default", t_entry=T0, t_king=KING_T, cfg=None,
              name="pool"):
    from cascade.validator.lane_d import load_pool  # noqa: I001
    from cascade.validator.lane_d_forward import eligible_windows

    cfg = cfg or lane_d_cfg()
    if feeds == "default":
        feeds = feeds_block({s["source"] for s in series})
    pool = load_pool(write_pool(tmp_path / name, series, feeds=feeds), HORIZON)
    out = eligible_windows(pool, t_entry=t_entry, t_king=t_king, cfg=cfg)
    return {v.window_id: v for v in out}


def _one(tmp_path, rec, **kw):
    got = _verdicts(tmp_path, [rec], **kw)
    wid = window_id(rec["series_id"], rec["target_ts"])
    assert list(got) == [wid], f"expected one verdict for {wid}; got {list(got)}"
    return got[wid]


def _assert_refused(v, reason):
    assert v.eligible is False, f"{v.window_id}: expected refusal {reason!r}, got eligible"
    assert v.reason == reason, f"{v.window_id}: reason {v.reason!r}, expected {reason!r}"


def _assert_eligible(v):
    assert v.eligible is True and v.reason is None, \
        f"{v.window_id}: expected eligible, got reason {v.reason!r}"


# ── the reason enum ─────────────────────────────────────────────────────────


def test_refusal_reasons_are_the_typed_enum_in_precedence_order():
    from cascade.validator.lane_d_forward import REFUSAL_REASONS

    assert tuple(str(r) for r in REFUSAL_REASONS) == REASONS


def test_one_verdict_per_window_in_pool_order(tmp_path):
    series = [case(k, BOUND, f"s{i}", f"src{i % 3}")
              for i, k in enumerate(("eligible", "late_mirror", "backfill", "no_frontier"))]
    from cascade.validator.lane_d import load_pool  # noqa: I001
    from cascade.validator.lane_d_forward import eligible_windows

    pool = load_pool(write_pool(tmp_path / "p", series,
                                feeds=feeds_block({"src0", "src1", "src2"})), HORIZON)
    out = list(eligible_windows(pool, t_entry=T0, t_king=KING_T, cfg=lane_d_cfg()))
    assert [v.window_id for v in out] == [window_id(s["series_id"], s["target_ts"])
                                          for s in series]
    assert [v.eligible for v in out] == [True, False, False, False]
    assert [str(v.reason) for v in out[1:]] == ["late_mirror", "backfill", "absent_from_s0"]


# ── T8 cases, each with its exact reason ────────────────────────────────────


def test_t8_baseline_both_clocks_after_the_bound_is_eligible(tmp_path):
    _assert_eligible(_one(tmp_path, case("eligible", BOUND, "base", "srcA")))


def test_t8_late_mirror_first_seen_passes_data_time_fails(tmp_path):
    """§1.3: first-seen alone admits values public before T that reached our
    mirror late (outage, backfill). (T8's parenthetical swaps the clock names;
    the §1.3 semantics are pinned.)"""
    _assert_refused(_one(tmp_path, case("late_mirror", BOUND, "late", "srcA")), "late_mirror")


def test_t8_publish_ahead_data_time_passes_first_seen_fails(tmp_path):
    """§1.3: data time alone admits future-stamped values public before T."""
    _assert_refused(_one(tmp_path, case("publish_ahead", BOUND, "ahead", "srcA")),
                    "publish_ahead")


def test_t8_feed_over_max_mirror_lag_is_never_eligible(tmp_path):
    _assert_refused(_one(tmp_path, case("over_lag", BOUND, "lag", "srcA")), "over_lag")


def test_t8_mirror_lag_equal_to_the_limit_is_within_it(tmp_path):
    rec = case("over_lag", BOUND, "lag", "srcA")
    feeds = feeds_block({"srcA"})
    feeds["laggy"]["mirror_lag_hours"] = LAG_H
    _assert_eligible(_one(tmp_path, rec, feeds=feeds))


def test_t8_forecast_product_feed_is_never_eligible(tmp_path):
    _assert_refused(_one(tmp_path, case("forecast_product", BOUND, "fc", "srcA")),
                    "forecast_product")


def test_t8_backfilled_partition_is_never_eligible(tmp_path):
    """Change 1(c): every target ts > end_ts(S0); points behind S0's frontier
    that arrive later are a backfill, even with both clocks after the bound."""
    _assert_refused(_one(tmp_path, case("backfill", BOUND, "bf", "srcA")), "backfill")


def test_t8_series_absent_from_s0_is_refused(tmp_path):
    """Change 1(a): no snapshot after the bound (before this one) held the series."""
    _assert_refused(_one(tmp_path, case("absent_from_s0", BOUND, "new", "srcA")),
                    "absent_from_s0")
    rec = case("eligible", BOUND, "pre", "srcA")
    rec["frontier"] = [[BOUND - 2 * HOUR, BOUND - 3 * HOUR]]   # only before the bound
    _assert_refused(_one(tmp_path, rec, name="pool2"), "absent_from_s0")


def test_absent_series_starts_its_clock_at_first_appearance(tmp_path):
    """A series first appearing after the bound: its first-appearance snapshot
    is its S0; later windows beyond that frontier are eligible."""
    first_seen_snapshot = BOUND + 20 * HOUR
    ts = hourly(first_seen_snapshot + 2 * HOUR)
    rec = spec("late-new", "srcA", ts, [t + 1800 for t in ts],
               frontier=[[first_seen_snapshot, first_seen_snapshot - HOUR]])
    _assert_eligible(_one(tmp_path, rec))
    inside = spec("late-new", "srcA", ts, [t + 1800 for t in ts],
                  frontier=[[first_seen_snapshot, ts[10]]])
    _assert_refused(_one(tmp_path, inside, name="pool2"), "backfill")


def test_before_margin_both_clocks_fail(tmp_path):
    _assert_refused(_one(tmp_path, case("before_margin", BOUND, "old", "srcA")),
                    "before_margin")


@pytest.mark.parametrize("clock", ["data", "seen"])
def test_margin_is_strict_on_both_clocks(tmp_path, clock):
    ts = hourly(BOUND + 3 * HOUR)
    seen = [t + 1800 for t in ts]
    if clock == "data":
        ts = [BOUND] + ts[1:]
        ts_ok = [BOUND + 1] + ts[1:]
        seen_ok = seen
    else:
        seen = [BOUND] + seen[1:]
        seen_ok = [BOUND + 1] + seen[1:]
        ts_ok = ts
    want = "late_mirror" if clock == "data" else "publish_ahead"
    fr = [[BOUND + 2 * HOUR, BOUND - HOUR]]
    _assert_refused(_one(tmp_path, spec("edge", "srcA", ts, seen, frontier=fr)), want)
    _assert_eligible(_one(tmp_path, spec("edge", "srcA", ts_ok, seen_ok, frontier=fr),
                          name="pool2"))


def test_every_target_point_counts_not_only_the_first(tmp_path):
    ts = hourly(BOUND + 3 * HOUR)
    seen = [t + 1800 for t in ts]
    seen[-1] = BOUND - HOUR              # the LAST point was seen before the bound
    _assert_refused(_one(tmp_path, spec("tail", "srcA", ts, seen,
                                        frontier=[[BOUND + 2 * HOUR, BOUND + HOUR]])),
                    "publish_ahead")


@pytest.mark.parametrize("order", ["entry_later", "king_later"])
def test_t8_entry_and_king_t_in_both_orders_uses_the_max(tmp_path, order):
    early, late = T0, T0 + 4 * 86_400
    t_entry, t_king = (late, early) if order == "entry_later" else (early, late)
    cfg = lane_d_cfg()
    lo, hi = int(bound_of(early, early, cfg)), int(bound_of(early, late, cfg))
    between = spec("between", "srcA", hourly(lo + 3 * HOUR),
                   [t + 1800 for t in hourly(lo + 3 * HOUR)],
                   frontier=[[hi + 2 * HOUR, hi + HOUR]])
    assert between["target_ts"][-1] < hi, "fixture: the window must sit inside (lo, hi]"
    after = case("eligible", hi, "after", "srcA")
    got = _verdicts(tmp_path, [between, after], t_entry=t_entry, t_king=t_king, cfg=cfg)
    _assert_refused(got[window_id("between", between["target_ts"])], "before_margin")
    _assert_eligible(got[window_id("after", after["target_ts"])])
    swapped = _verdicts(tmp_path, [between, after], t_entry=t_king, t_king=t_entry,
                        cfg=cfg, name="pool2")
    assert {k: (v.eligible, v.reason) for k, v in swapped.items()} == \
        {k: (v.eligible, v.reason) for k, v in got.items()}


# ── config keys are read ────────────────────────────────────────────────────


def test_forward_margin_hours_is_read_from_cfg(tmp_path):
    rec = case("eligible", BOUND, "m", "srcA")    # first target at bound + 3 h
    _assert_eligible(_one(tmp_path, rec))
    wide = lane_d_cfg(forward_margin_hours=MARGIN_H + 12.0)
    _assert_refused(_one(tmp_path, rec, cfg=wide, name="pool2"), "before_margin")


def test_max_mirror_lag_hours_is_read_from_cfg(tmp_path):
    rec = case("over_lag", BOUND, "lag", "srcA")   # feed lag = LAG_H + 1
    _assert_refused(_one(tmp_path, rec), "over_lag")
    _assert_eligible(_one(tmp_path, rec, cfg=lane_d_cfg(max_mirror_lag_hours=LAG_H + 2.0),
                          name="pool2"))


# ── feeds metadata ──────────────────────────────────────────────────────────


def test_feed_defaults_to_the_series_source(tmp_path):
    rec = case("eligible", BOUND, "src-feed", "srcLag")
    feeds = {"srcLag": {"mirror_lag_hours": LAG_H + 3.0, "forecast_product": False,
                       "embargo_hours": 0.0}}
    _assert_refused(_one(tmp_path, rec, feeds=feeds), "over_lag")
    feeds = {"srcLag": {"mirror_lag_hours": 1.0, "forecast_product": True,
                       "embargo_hours": 0.0}}
    _assert_refused(_one(tmp_path, rec, feeds=feeds, name="pool2"), "forecast_product")


def test_listed_feed_with_null_lag_is_not_a_refusal(tmp_path):
    """W6fs measures lag "where the feed provides" a publish stamp: a feed the
    snapshot LISTS with `mirror_lag_hours: null` is declared unmeasured; the
    two clocks still govern (RV3: only an explicit null admits)."""
    rec = case("eligible", BOUND, "u", "srcU")
    _assert_eligible(_one(tmp_path, rec, feeds={"srcU": {"mirror_lag_hours": None,
                                                             "embargo_hours": 0.0}}))
    _assert_eligible(_one(tmp_path, rec, name="pool2", feeds={
        "srcU": {"mirror_lag_hours": None, "forecast_product": False, "embargo_hours": 0.0}}))


def test_unlisted_feed_is_refused(tmp_path):
    """RV3 BLOCKING 1: a feed with no `feeds` entry is not "lag unmeasured":
    the window fails closed as `unlisted_feed`, both for a series' own source
    and for an explicit `feed`."""
    rec = case("eligible", BOUND, "u", "srcU")
    _assert_refused(_one(tmp_path, rec, feeds={}), "unlisted_feed")
    _assert_refused(_one(tmp_path, rec, name="pool2", feeds={
        "srcOther": {"mirror_lag_hours": 1.0, "forecast_product": False}}), "unlisted_feed")
    _assert_refused(_one(tmp_path, case("unlisted_feed", BOUND, "uf", "srcA"), name="pool3"),
                    "unlisted_feed")


def test_pool_without_feeds_block_is_refused(tmp_path):
    """RV3 BLOCKING 1: a pool with no top-level `feeds` lists no feed, so
    every window is `unlisted_feed` (over_lag and forecast_product cannot be
    silently switched off by dropping the block)."""
    series = [case("eligible", BOUND, f"nf{i}", f"src{i}") for i in range(3)]
    got = _verdicts(tmp_path, series, feeds=None)
    assert len(got) == 3
    for v in got.values():
        _assert_refused(v, "unlisted_feed")


def test_series_without_frontier_key_is_refused(tmp_path):
    """RV3 BLOCKING 1 (change 1(a): eligible only if the series is present in
    S0 with a recorded end_ts): a series with no `frontier` key has no
    evidence of S0 and fails closed as `absent_from_s0`, like an empty list.
    The clock rules still come first."""
    _assert_refused(_one(tmp_path, case("no_frontier", BOUND, "nofr", "srcA")),
                    "absent_from_s0")
    old = case("before_margin", BOUND, "nofr-old", "srcA")
    del old["frontier"]
    _assert_refused(_one(tmp_path, old, name="pool2"), "before_margin")


def test_series_with_null_frontier_is_refused(tmp_path):
    """RV3 BLOCKING 1: `"frontier": null` (e.g. from a serializer) is the same
    as absent: `absent_from_s0`, never the two-clock-only verdict."""
    rec = case("null_frontier", BOUND, "nullfr", "srcA")
    assert rec["frontier"] is None
    _assert_refused(_one(tmp_path, rec), "absent_from_s0")


# ── precedence and missing clocks ───────────────────────────────────────────


def test_missing_clock_series_is_never_eligible(tmp_path):
    rec = case("eligible", BOUND, "noseen", "srcA")
    rec["first_seen_ts"] = []
    _assert_refused(_one(tmp_path, rec), "missing_clock")
    rec = case("eligible", BOUND, "noend", "srcA")
    rec["end_ts"] = None
    _assert_refused(_one(tmp_path, rec, name="pool2"), "missing_clock")
    rec = case("eligible", BOUND, "short", "srcA")
    rec["first_seen_ts"] = rec["first_seen_ts"][:-1]
    _assert_refused(_one(tmp_path, rec, name="pool3"), "missing_clock")


def test_reason_precedence(tmp_path):
    # A real forecast product: future-stamped, published (seen) before T.
    fc = case("publish_ahead", BOUND, "fc-ahead", "srcA")
    fc["feed"] = "forecasts"
    _assert_refused(_one(tmp_path, fc), "forecast_product")
    # An over-lag feed whose values also reached the mirror late.
    lag = case("late_mirror", BOUND, "lag-late", "srcA")
    lag["feed"] = "laggy"
    _assert_refused(_one(tmp_path, lag, name="pool2"), "over_lag")
    # A late mirror behind S0's frontier: the clock reason wins over backfill.
    late = case("late_mirror", BOUND, "late-bf", "srcA")
    late["frontier"] = [[BOUND + 2 * HOUR, BOUND + HOUR]]
    _assert_refused(_one(tmp_path, late, name="pool3"), "late_mirror")
    # Missing clock beats everything.
    miss = case("forecast_product", BOUND, "miss-fc", "srcA")
    miss["end_ts"] = None
    _assert_refused(_one(tmp_path, miss, name="pool4"), "missing_clock")
    # RV3: missing clock beats an unlisted feed ...
    miss = case("unlisted_feed", BOUND, "miss-uf", "srcA")
    miss["end_ts"] = None
    _assert_refused(_one(tmp_path, miss, name="pool5"), "missing_clock")
    # ... an unlisted feed beats the clock and S0 rules ...
    uf = case("late_mirror", BOUND, "uf-late", "srcA")
    uf["feed"] = "unlisted"
    del uf["frontier"]
    _assert_refused(_one(tmp_path, uf, name="pool6"), "unlisted_feed")
    # ... and a missing frontier ranks with absent_from_s0, after the lag rules.
    lag = case("over_lag", BOUND, "lag-nofr", "srcA")
    del lag["frontier"]
    _assert_refused(_one(tmp_path, lag, name="pool7"), "over_lag")


# ── the v2 snapshot-only rule is refused ────────────────────────────────────


def _names_in(tree: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    return out


def test_v2_snapshot_rule_is_not_on_the_lane_d_code_path():
    import cascade.validator.lane_d_forward as fwd  # noqa: I001
    import cascade.validator.lane_d as lane_d

    banned = ("laned_w1_measure", "snapshot_diff", "BOUND_LABEL", "APPEARED", "CARRIED")
    for mod in (fwd, lane_d):
        names = _names_in(ast.parse(inspect.getsource(mod)))
        hits = sorted(n for n in names for b in banned if b in n)
        assert not hits, f"{mod.__name__} references the v2 snapshot rule: {hits}"
    calls = {node.func.id if isinstance(node.func, ast.Name) else
             getattr(node.func, "attr", "")
             for node in ast.walk(ast.parse(inspect.getsource(lane_d)))
             if isinstance(node, ast.Call)}
    assert "eligible_windows" in calls, \
        "settle_lane_d's module never calls eligible_windows (§1.3 not on the code path)"


def test_v2_snapshot_rule_admits_what_w5a_refuses(tmp_path):
    """The v2 rule ("every target point absent from the first snapshot after
    max(T) + margin") counts a series new to that snapshot as fresh, so its
    pre-T points pass; W5a refuses them."""
    from cascade.validator.lane_d_forward import eligible_windows  # noqa: I001
    from cascade.validator.lane_d import load_pool

    loader = importlib.util.spec_from_file_location("w5a_v2_snapshot_rule", REPO_SCRIPTS)
    v2 = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(v2)
    cfg = lane_d_cfg()
    t_build_a, t_build_b = BOUND - HOUR, BOUND + 5 * HOUR
    values = np.arange(2 * HORIZON, dtype=np.float64) + 1.0
    prev, _ = v2.advance(None, {"other": values.copy()}, t_build_a)
    state, tr = v2.advance(prev, {"other": values.copy(), "newcomer": values.copy()},
                           t_build_b)
    assert tr.get("appeared") == 1
    res = v2.eligibility({"newcomer": state["newcomer"]}, {"newcomer": {"source": "srcA"}},
                         float(T0), margin_s=MARGIN_H * HOUR, horizon=HORIZON,
                         min_context=HORIZON, min_windows=1, min_clusters=1)
    assert res["total"]["windows"] == 1, "premise: the v2 rule admits the newcomer"
    old = hourly(T0 - 80 * HOUR)        # data public long before T
    rec = spec("newcomer", "srcA", old, [t_build_b] * HORIZON,
               frontier=[[t_build_b + HOUR, old[0] - HOUR]])
    pool = load_pool(write_pool(tmp_path / "p", [rec], feeds=feeds_block({"srcA"})),
                     HORIZON)
    (v,) = list(eligible_windows(pool, t_entry=T0, t_king=KING_T, cfg=cfg))
    _assert_refused(v, "late_mirror")
