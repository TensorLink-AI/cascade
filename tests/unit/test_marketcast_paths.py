"""Every suite on every timeframe, as multi-step paths (owner 2026-11-18):
``<bar>x<steps>`` horizons, every path drawn from the embargo before the
dispatch (owner 2026-11-24, DEC-CA-0073), and the calibration slack counted
in windows, not the correlated steps of one path."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.eval.marketcast import window_geometry
from cascade.pool.marketcast import embargo_span
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import (
    draw_problems,
    duration_minutes,
    horizon_path,
    parse_marketcast,
    scorability_problems,
)
from cascade.shared.marketcast_report import (
    MktEntryRows,
    MktReport,
    MktWindowRow,
    calibration_gate,
    load_mkt_report,
)

DAY = 86400


def test_path_horizons_parse_and_set_the_geometry():
    assert horizon_path("1mx60") == ("1m", 60) and horizon_path("1h") is None
    assert duration_minutes("15mx96") == 1440 and duration_minutes("1dx7") == 7 * 1440
    assert window_geometry(("1m", "1h"), "1mx60") == (1, 60)
    assert window_geometry(("1m", "1h"), "1h") == (60, 1)          # duration horizons unchanged
    with pytest.raises(ValueError, match="not a frequency"):
        window_geometry(("1h",), "5mx36")
    with pytest.raises(ValueError, match="steps"):
        window_geometry(("1m",), "1mx2000")


def test_shipped_grid_is_every_suite_on_every_timeframe():
    want = ["1mx60", "5mx36", "15mx96", "1hx48", "4hx42", "1dx7"]
    for toml in ("chain.toml", "chain.testnet.toml"):
        mc = load_chain_config(toml).marketcast
        assert mc.forward_horizons == ("1mx60", "5mx36") and not hasattr(mc, "gap_horizons")
        for s in mc.suites:
            assert list(mc.horizons_map[s]) == want
            assert [window_geometry(mc.frequencies_map[s], h) for h in want] == [
                (1, 60), (5, 36), (15, 96), (60, 48), (240, 42), (1440, 7)]
        assert scorability_problems(mc) == [] and draw_problems(mc) == []


def test_draw_and_path_config_is_checked():
    grid = {"suites": ["returns"], "frequencies": {"returns": ["1m", "1d"]},
            "horizons": {"returns": ["1mx60", "1dx7"]}}
    mc = parse_marketcast(grid)
    assert mc.forward_horizons == ("1mx60",) and draw_problems(mc) == []
    # the week-long path needs two aligned weeks of embargo span
    assert any("embargo eval span" in p for p in draw_problems(replace(mc, embargo_eval_days=10)))
    assert any(">= 2 days" in p for p in draw_problems(replace(mc, embargo_days=1)))
    # a forward horizon must fit the <= 13 h forward span
    assert any("forward horizon" in p
               for p in draw_problems(replace(mc, forward_horizons=("1mx60", "1dx7"))))
    with pytest.raises(ValueError, match="no enabled suite"):
        parse_marketcast({**grid, "forward_horizons": ["4hx42"]})
    with pytest.raises(ValueError, match="embargo_eval_days"):
        parse_marketcast({**grid, "embargo_eval_days": 30})
    with pytest.raises(ValueError, match="backtest_windows_per_cell"):
        parse_marketcast({**grid, "backtest_windows_per_cell": 257})
    with pytest.raises(ValueError, match="paths"):
        parse_marketcast({**grid, "frequencies": {"returns": ["1mx60"]}})
    # owner 2026-11-24: a verdict within 13 h — a longer wait is capped in code
    assert parse_marketcast({**grid, "verdict_wait_blocks": 43200}).wait_blocks == 3900
    # the retired gap_horizons key of an older file loads and does nothing
    assert parse_marketcast({**grid, "gap_horizons": ["1dx7"]}) == mc


def test_the_embargo_is_the_days_no_leg_reads():
    mc = SimpleNamespace(embargo_days=30)
    sub = 1_790_000_000 + 5000
    lo, hi = embargo_span(mc, sub)
    day = sub // DAY * DAY
    # a recipe's data.end is <= the commit's UTC date - 30 d, read whole
    assert lo + 1 == day - 29 * DAY and hi == sub


def test_the_draw_record_round_trips_and_is_omitted_when_empty():
    ent = MktEntryRows("hk", "p", "mkt:market", "judge", 1.0,
                       draw={"rule": "embargo-v1", "block": 7, "lo": 1.0})
    rep = MktReport("r", 1, {}, (ent, replace(ent, phase="confirm", draw={})))
    back = load_mkt_report(rep.to_json()).entries
    assert [x.draw for x in back] == [ent.draw, {}]
    assert b'"draw"' not in MktReport("r", 1, {}, (replace(ent, draw={}),)).canonical_body()


def test_calibration_slack_counts_windows_not_path_steps():
    mc = parse_marketcast({})

    def row(i, steps, inside):
        return MktWindowRow(f"crypto:BTC|returns|1mx60|{1_790_000_000 + 3600 * i}", 1.0, 1.0,
                            (inside, inside, steps), steps, float("nan"), (0, 0))

    # 10 windows, 60 steps each, 40 % inside the 50 % band: 600 correlated steps
    # are 10 draws — the slack (3 SE over 10 windows ≈ 0.47) keeps it passing
    g = calibration_gate([row(i, 60, 24) for i in range(10)], mc, mc.coverage_levels)
    r50 = next(r for r in g.rows if r.nominal == 0.5 and r.kind != "tail")
    assert r50.empirical == pytest.approx(0.4) and r50.slack > 0.4 and not r50.under
