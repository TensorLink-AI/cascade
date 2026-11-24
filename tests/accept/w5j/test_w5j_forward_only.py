"""W5j (plan rev6.18 D24, F4; owner 2026-11-24 DECIDED: the slow-cadence
backtest windows are dropped): the live draw has forward cells only. No
backtest cell is built, drawn or scored; the report schema knows no backtest
kind or bin; every judged window and every draw quota cell is ``forward``.
Contract: README.md "Forward cells only (rev6.18 D24)".
"""

from __future__ import annotations

import json
from pathlib import Path

from .helpers import HASH_A, lane_d_cfg
from .settle_helpers import (
    BOUNDARY_1,
    CHALLENGER_UID,
    entry,
    forecasts,
    house,
    records,
    settle,
    write_pool,
)

REPO = Path(__file__).resolve().parents[3]


def test_the_schema_has_forward_cells_only():
    defs = json.loads((REPO / "cascade/validator/schemas/lane_d_report_1.json")
                      .read_text(encoding="utf-8"))["$defs"]
    assert defs["window_kind"]["enum"] == ["forward"], defs["window_kind"]
    assert defs["window_bin"]["enum"] == ["forward"], defs["window_bin"]


def test_every_judged_window_and_quota_cell_is_forward(tmp_path):
    from cascade.shared.config.lane_d import parse_lane_d

    assert not hasattr(parse_lane_d({}), "backtest_weight_cap"), (
        "rev6.18 D24: the backtest cap is still a [lane_d] key, so backtest cells can be armed")
    cfg = lane_d_cfg(max_windows_per_settlement=40)
    recs = records({f"src{i}": 8 for i in range(8)})
    pool = write_pool(tmp_path / "pool", recs)
    e = entry("alice")
    h = house()
    out = settle(cfg, BOUNDARY_1, pool, [e], h, forecasts(recs, [e["digest"]], h["digest"]),
                 {e["hotkey"]: CHALLENGER_UID}, tmp_path / "state.json", HASH_A)
    rep = out.report
    assert rep["windows"], "fixture: the cohort judged windows"
    assert {(w["kind"], w["bin"]) for w in rep["windows"]} == {("forward", "forward")}
    for c in rep["draw"]["cohorts"]:
        assert all(k.endswith("|forward") for k in (*c["quota"], *c["capacity"])), c


def test_no_backtest_key_or_statistic_is_left():
    from cascade.shared.config.lane_d import parse_lane_d

    ld = parse_lane_d({})
    assert not hasattr(ld, "backtest_weight_cap") and not hasattr(ld, "backtest_ratio_floor"), (
        "the W5i backtest keys are still parsed (rev6.18 D24 deletes them)")
