"""The strong simple baselines (owner 2026-11-18): random walk + EWMA vol
with filtered-historical-simulation innovations for returns and price
levels, a seasonal AR(1) for vol / range / excursions / volume — and the
composite gate that a fine-tune must beat them."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cascade.eval.marketcast import crps_quantile
from cascade.eval.marketcast_baselines import baseline_kind, baseline_quantiles
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import TIMESTRAL_KNOTS, parse_marketcast
from cascade.shared.marketcast_report import MktEntryRows, MktWindowRow, verdict_from_rows

K = np.asarray(TIMESTRAL_KNOTS)
T0 = 1_790_000_000 // 86400 * 86400


def test_families_by_suite_label():
    assert baseline_kind("returns") == "rw_fhs" and baseline_kind("ohlcv.c") == "rw_fhs_level"
    assert baseline_kind("ohlcv.v") == "seasonal_ar" and baseline_kind("vol") == "seasonal_ar"
    assert baseline_kind("liquidity") == "seasonal_ar"


def test_returns_baseline_tracks_the_recent_volatility():
    rng = np.random.default_rng(0)
    ctx = np.r_[rng.normal(0, 1, 1500), rng.normal(0, 4, 500)]       # vol just quadrupled
    q = baseline_quantiles("returns", ctx, 5, 1, K)
    assert q.shape == (5, 21) and np.all(np.diff(q, axis=-1) >= 0)
    width = q[0, -2] - q[0, 1]                                        # 5–95 %
    assert 2.5 * 3.29 < width < 5 * 3.29                              # near the recent σ = 4


def test_price_level_path_widens_with_the_horizon():
    rng = np.random.default_rng(1)
    ctx = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 2000)))
    q = baseline_quantiles("ohlcv.c", ctx, 16, 5, K)
    w = q[:, -2] - q[:, 1]
    assert q[0, 10] == pytest.approx(ctx[-1], rel=1e-3)               # median ~ last value
    assert w[15] / w[0] == pytest.approx(4.0, rel=0.05)              # σ √k


def test_seasonal_ar_learns_the_daily_profile():
    rng = np.random.default_rng(2)
    m = 24                                                            # 1h bars
    n = 24 * 30
    prof = np.sin(2 * np.pi * np.arange(m) / m)
    ctx = np.tile(prof, n // m) + rng.normal(0, 0.1, n)
    q = baseline_quantiles("liquidity", ctx, 24, 60, K)
    assert np.corrcoef(q[:, 10], prof)[0, 1] > 0.95
    y = prof + rng.normal(0, 0.1, m)
    flat = np.tile(np.quantile(ctx, K), (m, 1))
    assert crps_quantile(q, y).mean() < 0.5 * crps_quantile(flat, y).mean()


def test_thin_context_is_nan():
    assert np.isnan(baseline_quantiles("returns", np.ones(5), 3, 1, K)).all()


def _rows(*, vs_baseline, returns_vs_baseline=None, n=60, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for s in ("returns", "vol"):
        r = vs_baseline if returns_vs_baseline is None or s != "returns" else returns_vs_baseline
        for a in ("BTC", "ETH"):
            for t in range(n):
                x = float(rng.uniform(1, 2))
                out.append(MktWindowRow(f"crypto:{a}|{s}|1h|{T0 + 21600 * t}", x, x * 0.9,
                                        (1, 1, 1), 2, float("nan"), (0, 0),
                                        bl=x * 0.9 / r))
    return MktEntryRows("hk", "p", "mkt:market", "judge", 0.0, windows=tuple(out))


def test_the_composite_must_beat_the_baselines():
    mc = replace(load_chain_config("chain.toml").marketcast, bootstrap_B=300, min_windows=10,
                 min_clusters=2, min_time_blocks=2, suites=("returns", "vol"), guards=(),
                 coverage_tolerance=0.45, coverage_tolerance_over=0.45, tail_tolerance=0.45)
    assert mc.baseline_gate
    ok = verdict_from_rows(_rows(vs_baseline=0.9), mc, suite="market", k=1, seed="s")
    assert ok.passed, ok.reason
    assert ok.diagnostics["vs_baseline"]["gain"] == pytest.approx(0.1)
    worse = verdict_from_rows(_rows(vs_baseline=1.1), mc, suite="market", k=1, seed="s")
    assert not worse.passed and "simple baselines" in worse.reason
    # a composite win that is CONFIDENTLY worse than a random walk on returns
    one = verdict_from_rows(_rows(vs_baseline=0.8, returns_vs_baseline=1.05), mc,
                            suite="market", k=1, seed="s")
    assert not one.passed and "confidently worse" in one.reason and "returns" in one.reason
    assert verdict_from_rows(_rows(vs_baseline=1.1), replace(mc, baseline_gate=False),
                             suite="market", k=1, seed="s").passed
    with pytest.raises(ValueError, match="baseline_gate"):
        parse_marketcast({"baseline_gate": "yes"})
