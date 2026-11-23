"""W5i: the owner's horizon bins, freq classes and the cells a series feeds
(rev6.9 (a) "Horizon bins", "Daily short stays forward-only"; the cap is
ON by default: `backtest_weight_cap` = 0.05, owner 2026-11-23, W2e #781,
plan rev6.15; 0 still switches the backtest off)."""

from __future__ import annotations

import pytest

from .helpers import BINS, CLASS, HORIZON, lane_d_cfg


def _mod():
    import cascade.validator.lane_d_backtest as m

    return m


def test_w5i_horizon_bins_table_is_the_owner_table():
    m = _mod()
    got = {f: {b: tuple(r) for b, r in bins.items()} for f, bins in m.HORIZON_BINS.items()}
    assert got == BINS
    assert max(hi for bins in BINS.values() for _lo, hi in bins.values()) <= 512


@pytest.mark.parametrize("freq", sorted(CLASS))
def test_w5i_freq_class_of_slow_freqs(freq):
    assert _mod().freq_class(freq) == CLASS[freq]


@pytest.mark.parametrize("freq", ["10s", "1min", "5min", "15min", "30min", "H", "h", "d", ""])
def test_w5i_freq_class_subdaily_and_unknown(freq):
    """Exact match on the pool freq string: anything else is sub-daily."""
    assert _mod().freq_class(freq) == "subdaily"


def _headers(cells):
    return [(c["timeframe"], c["bin"], c["kind"], c["h_lo"], c["h_hi"]) for c in cells]


@pytest.mark.parametrize("freq", ["10s", "5min", "H"])
def test_w5i_subdaily_feeds_one_forward_cell_at_h64(freq):
    cells = _mod().cells_for_freq(freq, lane_d_cfg())
    assert _headers(cells) == [(freq, "forward", "forward", HORIZON, HORIZON)]


@pytest.mark.parametrize("freq", sorted(CLASS))
def test_w5i_off_at_cap_0_every_freq_is_forward_h64(freq):
    """backtest_weight_cap = 0 (an explicit off; the default is 0.05): nothing
    built; daily stays h = 64."""
    cells = _mod().cells_for_freq(freq, lane_d_cfg(backtest_weight_cap=0.0))
    assert _headers(cells) == [(freq, "forward", "forward", HORIZON, HORIZON)]


@pytest.mark.parametrize("freq", sorted(CLASS))
def test_w5i_the_default_cap_builds_the_slow_bins(freq):
    """Owner 2026-11-23 (W2e #781): with no `backtest_weight_cap` in the
    table the parsed default is 0.05, so a slow freq feeds its three bins
    (never the cap-0 single forward cell)."""
    from cascade.shared.config.lane_d import parse_lane_d

    from .helpers import ARM_BLOCK, ERA

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a", "remainder": "burn",
           "horizon": HORIZON, "min_windows": 8, "min_clusters": 2, "max_source_share": 1.0,
           "bootstrap_B": 400, "d_margin": 0.01, "max_windows_per_settlement": 60,
           "min_fresh_window_share": 0.5}
    cfg = parse_lane_d(raw, era_blocks_at=lambda _block: ERA)
    assert float(cfg.backtest_weight_cap) == 0.05, cfg.backtest_weight_cap
    cells = _mod().cells_for_freq(freq, cfg)
    assert [c["bin"] for c in cells] == ["short", "medium", "long"], _headers(cells)


@pytest.mark.parametrize("freq", sorted(CLASS))
def test_w5i_armed_slow_freq_feeds_three_bins(freq):
    cells = _mod().cells_for_freq(freq, lane_d_cfg())
    want = []
    for b in ("short", "medium", "long"):
        kind = "forward" if (freq == "D" and b == "short") else "backtest"
        want.append((freq, b, kind, *BINS[freq][b]))
    assert _headers(cells) == want


@pytest.mark.parametrize(("n_fwd", "n_bt", "cap", "c"), [
    (100, 0, 0.05, 0.0),
    (98, 2, 0.05, 0.02),        # a handful of backtest windows keeps its natural share
    (50, 50, 0.05, 0.05),
    (10, 90, 0.05, 0.05),
    (50, 50, 0.0, 0.0),
])
def test_w5i_backtest_weight(n_fwd, n_bt, cap, c):
    assert _mod().backtest_weight(n_fwd, n_bt, cap) == pytest.approx(c, abs=1e-15)


@pytest.mark.parametrize(("r_fwd", "r_bt", "c", "floor"), [
    (0.95, 1.0, 0.05, 0.85),
    (1.0, 1e-9, 0.05, 0.85),     # clamped at the floor
    (1.0, 1e9, 0.05, 0.85),      # clamped at 1 / floor
    (0.9, 0.9, 0.05, 0.85),      # inside the band: unclamped
    (0.97, 0.5, 0.0, 0.85),      # c = 0: forward only
])
def test_w5i_combined_log_ratio(r_fwd, r_bt, c, floor):
    import math

    want = (1 - c) * math.log(r_fwd) + c * math.log(min(max(r_bt, floor), 1 / floor))
    got = _mod().combined_log_ratio(r_fwd, r_bt, c=c, floor=floor)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_w5i_memorisation_bound_at_the_defaults():
    """Perfect memorisation with forward equal gains at most 1 - floor ** c."""
    import math

    lr = _mod().combined_log_ratio(1.0, 1e-30, c=0.05, floor=0.85)
    gain = 1 - math.exp(lr)
    assert gain == pytest.approx(1 - 0.85 ** 0.05, rel=1e-12)
    assert gain < 0.01


def test_w5i_loader_refuses_a_cap_breaking_the_inequality():
    try:
        lane_d_cfg(backtest_weight_cap=0.10)          # 1 - 0.85**0.1 ~ 0.016 >= 0.01
    except ValueError:
        pass
    else:
        raise AssertionError("backtest_weight_cap 0.10 at floor 0.85 / d_margin 0.01 loaded")
    cfg = lane_d_cfg(backtest_weight_cap=0.05)
    assert float(cfg.backtest_weight_cap) == 0.05
