"""W5i: backtest cells in the W5j draw (hold-out of the last h points, h drawn
in the bin, short-history skip, forward-only caps and floors)."""

from __future__ import annotations

from .helpers import (
    HASH_A,
    HASH_B,
    MIN_HISTORY,
    bt_series,
    cell,
    draw,
    fwd_series,
    lane_d_cfg,
    reference_draw,
    view,
)


def _cells():
    """Hourly forward windows from 6 sources, a daily short forward cell, and
    weekly / monthly backtest cells (two series too short to hold out)."""
    hourly = [fwd_series(f"h-{i:02d}", f"src{i % 6}", 64, step=3600) for i in range(24)]
    daily = [fwd_series(f"d-{i}", f"src{i % 6}", 60) for i in range(6)]
    weekly = [bt_series(f"w-{i}", f"wsrc{i % 3}", 200) for i in range(6)]
    weekly += [bt_series("w-short", "wsrc0", MIN_HISTORY + 3)]
    monthly = [bt_series(f"m-{i}", f"msrc{i}", 120) for i in range(4)]
    monthly += [bt_series("m-short", "msrc0", MIN_HISTORY + 10)]
    return [cell("H", "forward", "forward", hourly),
            cell("D", "short", "forward", daily),
            cell("W", "medium", "backtest", weekly),
            cell("M", "long", "backtest", monthly)]


def test_w5i_draw_with_backtest_cells_equals_the_reference():
    cfg = lane_d_cfg()
    for h in (HASH_A, HASH_B):
        out, ref = draw(_cells(), cfg, h), reference_draw(_cells(), block_hash=h)
        assert out["capacity"] == ref["capacity"]
        assert out["quota"] == ref["quota"]
        assert view(out) == ref["windows"]
        assert out["meets_floors"] is ref["meets_floors"]


def test_w5i_backtest_window_is_the_hold_out_of_the_last_h_points():
    out = draw(_cells(), lane_d_cfg(), HASH_A)
    n_of = {"w": 200, "m": 120}
    bt = [w for w in view(out) if w["kind"] == "backtest"]
    assert bt, "no backtest window drawn"
    for w in bt:
        n = n_of[w["series_id"][0]]
        lo, hi = {"W": (14, 26), "M": (25, 36)}[w["timeframe"]]
        assert lo <= w["h"] <= hi
        i = n - w["h"]
        assert w["origin"] == 1_500_000_000 + i * 86_400
        assert w["embargo_window"] is False
        assert w["freq_class"] == {"W": "weekly", "M": "monthly"}[w["timeframe"]]


def test_w5i_daily_short_is_a_forward_window_at_the_drawn_h():
    out = draw(_cells(), lane_d_cfg(), HASH_A)
    ds = [w for w in view(out) if w["timeframe"] == "D"]
    assert ds and all(w["kind"] == "forward" and 7 <= w["h"] <= 30 for w in ds)
    assert all(w["freq_class"] == "daily" for w in ds)


def test_w5i_short_history_series_are_skipped_with_a_typed_reason():
    out = draw(_cells(), lane_d_cfg(), HASH_A)
    assert out["skipped"] == reference_draw(_cells(), block_hash=HASH_A)["skipped"]
    # w-short: 35 points, weekly medium h >= 14 leaves 21 < 32; m-short: 42
    # points, monthly long h >= 25 leaves 17 < 32.
    assert {(s["series_id"], s["reason"]) for s in out["skipped"]} == {
        ("m-short", "backtest_short_history"), ("w-short", "backtest_short_history")}
    assert not any(w["series_id"] in ("m-short", "w-short") for w in out["windows"])


def test_w5i_floors_count_forward_windows_only():
    """Plenty of backtest windows never meet the floors on their own."""
    cfg = lane_d_cfg(min_windows=8, min_clusters=2)
    cells = [cell("W", "medium", "backtest",
                  [bt_series(f"w-{i}", f"wsrc{i}", 200) for i in range(20)]),
             cell("H", "forward", "forward",
                  [fwd_series(f"h-{i}", "src0", 64, step=3600) for i in range(4)])]
    out = draw(cells, cfg, HASH_A)
    assert sum(1 for w in out["windows"] if w["kind"] == "backtest") >= 8
    assert out["meets_floors"] is False


def test_w5i_source_cap_ignores_backtest_windows():
    """max_source_share caps forward windows only: a source holding many
    backtest windows keeps them all."""
    cfg = lane_d_cfg(max_source_share=0.5)
    cells = [cell("W", "medium", "backtest",
                  [bt_series(f"w-{i}", "solo", 200) for i in range(10)]),
             cell("H", "forward", "forward",
                  [fwd_series(f"h-{i}", f"src{i % 4}", 64, step=3600) for i in range(16)])]
    out = draw(cells, cfg, HASH_A)
    assert view(out) == reference_draw(cells, block_hash=HASH_A, max_source_share=0.5)["windows"]
    assert sum(1 for w in out["windows"] if w["source"] == "solo") == 10
