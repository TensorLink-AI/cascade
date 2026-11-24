"""Forge reader scrape-gap bridging: a SHORT gap (<= BRIDGE_MAX_SECONDS and
<= BRIDGE_MAX_FRAC of the bridged segment) on a regular grid is bridged with
NaN slots instead of splitting the series; anything larger still splits.
Exercises ``TsbenchForgeSource._extract_values`` directly with synthetic
panel-row frames (no forge mirror needed), like ``test_forge_mv.py``.
"""
from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from cascade.pool.builder import PoolBuildConfig, prepare_series
from cascade.pool.source import HarvestContext, HarvestedSeries
from cascade.pool.sources.tsbench_forge import (
    BRIDGE_MAX_FRAC,
    BRIDGE_MAX_SECONDS,
    TsbenchForgeSource,
)

# as_of well after every frame so the freshness cutoff keeps the whole series
CTX = HarvestContext(as_of=dt.date(2027, 3, 1), horizon=16, max_series=100)
SRC = TsbenchForgeSource(forge_dir="/tmp")


def _frame_from(ts: pd.DatetimeIndex | pd.Series, *, name: str = "v") -> pd.DataFrame:
    r = np.arange(len(ts), dtype=float)
    return pd.DataFrame({"timestamp": pd.Series(ts).astype(str),
                         name: np.sin(r / 37.0) * 5.0 + 50.0 + r * 1e-3})


def _midnight_gapped_10s(days: int = 5, gap_min: int = 25) -> pd.DatetimeIndex:
    """A 10 s date-templated feed that misses ``gap_min`` minutes after every
    UTC midnight (the promlabs shape), ending at the last day's midnight gap."""
    stamps = []
    start = pd.Timestamp("2027-02-01", tz="UTC")
    for d in range(days):
        day0 = start + pd.Timedelta(days=d)
        first = day0 + pd.Timedelta(minutes=gap_min) if d else day0
        stamps.append(pd.date_range(first, day0 + pd.Timedelta(days=1),
                                    freq="10s", inclusive="left"))
    return stamps[0].append(stamps[1:])


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()
                          ).hexdigest()


# ── bridging ────────────────────────────────────────────────────────────────


def test_midnight_gaps_bridge_into_one_segment_with_nans():
    ts = _midnight_gapped_10s(days=5, gap_min=25)
    df = _frame_from(ts)
    vals = SRC._extract_values(pd, df, [], CTX)
    assert vals is not None and vals.ndim == 1
    per_gap = 25 * 6                        # 25 min of 10 s slots
    assert len(vals) == len(df) + 4 * per_gap == 5 * 8640   # full 5-day grid
    nan = np.isnan(vals)
    assert int(nan.sum()) == 4 * per_gap
    # NaNs sit exactly at the midnight slots of days 2..5; real values untouched
    for d in range(1, 5):
        assert nan[d * 8640: d * 8640 + per_gap].all()
    assert np.array_equal(vals[~nan], df["v"].to_numpy())


def test_two_hour_gap_still_splits():
    a = pd.date_range("2027-02-01", periods=5000, freq="10s", tz="UTC")
    b = pd.date_range(a[-1] + pd.Timedelta(hours=2), periods=5000, freq="10s")
    df = _frame_from(a.append(b))
    vals = SRC._extract_values(pd, df, [], CTX)
    assert BRIDGE_MAX_SECONDS < 2 * 3600
    assert len(vals) == 5000 and np.isfinite(vals).all()       # newest segment only
    assert np.array_equal(vals, df["v"].to_numpy()[5000:])


def test_one_hour_gap_above_two_percent_still_splits():
    a = pd.date_range("2027-02-01", periods=5000, freq="10s", tz="UTC")
    b = pd.date_range(a[-1] + pd.Timedelta(hours=1), periods=5000, freq="10s")
    df = _frame_from(a.append(b))
    missing = 3600 // 10 - 1                        # 359 slots
    assert missing > BRIDGE_MAX_FRAC * (10_000 + missing)
    vals = SRC._extract_values(pd, df, [], CTX)
    assert len(vals) == 5000 and np.isfinite(vals).all()
    assert np.array_equal(vals, df["v"].to_numpy()[5000:])


def test_one_hour_gap_within_two_percent_bridges():
    a = pd.date_range("2027-02-01", periods=20_000, freq="10s", tz="UTC")
    b = pd.date_range(a[-1] + pd.Timedelta(hours=1), periods=20_000, freq="10s")
    df = _frame_from(a.append(b))
    vals = SRC._extract_values(pd, df, [], CTX)
    assert len(vals) == 40_000 + 359 and int(np.isnan(vals).sum()) == 359


def test_mv_channels_bridge_on_the_shared_grid():
    ts = _midnight_gapped_10s(days=3, gap_min=10)
    df = _frame_from(ts, name="a")
    df["b"] = df["a"] * 2.0
    src = TsbenchForgeSource(forge_dir="/tmp", mv_pack=True)
    vals = src._extract_values(pd, df, [], CTX, columns=["a", "b"])
    assert vals.shape == (2, 3 * 8640)
    assert np.array_equal(np.isnan(vals[0]), np.isnan(vals[1]))
    assert int(np.isnan(vals[0]).sum()) == 2 * 60


def test_bridged_series_over_max_missing_is_dropped_by_builder():
    # 30 one-hour gaps, each < 2% of the bridged segment, but ~26% missing in
    # total: the reader bridges, the builder's max_missing_frac (0.2) drops it.
    parts, t = [], pd.Timestamp("2027-02-01", tz="UTC")
    for _ in range(31):
        blk = pd.date_range(t, periods=1000, freq="10s")
        parts.append(blk)
        t = blk[-1] + pd.Timedelta(hours=1)
    df = _frame_from(parts[0].append(parts[1:]))
    vals = SRC._extract_values(pd, df, [], CTX)
    assert len(vals) == 31_000 + 30 * 359                       # bridged, not split
    frac = float(np.isnan(vals).mean())
    cfg = PoolBuildConfig(context_length=512, horizon=16)
    assert frac > cfg.max_missing_frac
    hs = HarvestedSeries(series_id="x", values=vals, freq="10S", domain="infra",
                         seasonal_period=360, source="x")
    rec, reason = prepare_series(hs, cfg)
    assert rec is None and reason == "too_much_missing"


def test_bridged_series_under_max_missing_is_kept_and_filled_by_builder():
    vals = SRC._extract_values(pd, _frame_from(_midnight_gapped_10s()), [], CTX)
    cfg = PoolBuildConfig(context_length=512, horizon=16, keep_tail=False)
    hs = HarvestedSeries(series_id="x", values=vals, freq="10S", domain="infra",
                         seasonal_period=360, source="x")
    rec, reason = prepare_series(hs, cfg)
    assert reason is None and np.isfinite(rec.values).all()


def test_irregular_series_is_never_bridged():
    # No regular grid (spacing drawn from a wide distribution): unchanged rule.
    rng = np.random.default_rng(7)
    steps = rng.uniform(1.0, 60.0, size=3000)
    steps[1500] = 1800.0                          # a 30-min gap > 8 x median
    ts = pd.Timestamp("2027-02-01", tz="UTC") + pd.to_timedelta(np.cumsum(steps), unit="s")
    df = _frame_from(pd.DatetimeIndex(ts))
    vals = SRC._extract_values(pd, df, [], CTX)
    assert np.isfinite(vals).all() and len(vals) == 3000 - 1500


def test_bridging_is_deterministic():
    df = _frame_from(_midnight_gapped_10s())
    a = SRC._extract_values(pd, df, [], CTX)
    b = SRC._extract_values(pd, df.copy(), [], CTX)
    assert _sha(a) == _sha(b)
    # shuffled input rows sort back to the identical bridged output
    c = SRC._extract_values(pd, df.sample(frac=1.0, random_state=3), [], CTX)
    assert _sha(a) == _sha(c)


# ── untouched output for fixtures without a bridgeable gap ──────────────────
# Hashes recorded from main BEFORE the bridging change: these must never move.


def _daily_gap_free() -> pd.DataFrame:
    return _frame_from(pd.date_range("2026-01-01", periods=400, freq="D"))


def _five_min_long_gap() -> pd.DataFrame:              # 3-day gap: split as before
    a = pd.date_range("2027-01-01", periods=2000, freq="5min", tz="UTC")
    b = pd.date_range(a[-1] + pd.Timedelta(days=3), periods=1200, freq="5min")
    return _frame_from(a.append(b))


def _ten_s_small_jitter() -> pd.DataFrame:             # 60 s holes < 8 x median
    a = pd.date_range("2027-02-01", periods=6000, freq="10s", tz="UTC")
    return _frame_from(a.delete(np.arange(100, 6000, 700)))


def _irregular() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    steps = rng.uniform(1.0, 60.0, size=2500)
    steps[900] = 4000.0
    ts = pd.Timestamp("2027-02-01", tz="UTC") + pd.to_timedelta(np.cumsum(steps), unit="s")
    return _frame_from(pd.DatetimeIndex(ts))


PINNED = {
    "daily_gap_free": (_daily_gap_free,
                        "e2a48081a706712c64b27741c2a7d6cf03b7b13a13092b63a02648a5be2982cb"),
    "five_min_long_gap": (_five_min_long_gap,
                        "3a7aa40a8f101424ef59251a04e8bdbbbceb43b6f70c9d124b90f9dd007bfb30"),
    "ten_s_small_jitter": (_ten_s_small_jitter,
                        "1712752d9d984cf24b05c09c6deba836233ab3d7232d32f02e0a6950086dde7d"),
    "irregular": (_irregular,
                        "b149daf00cc80a3496da1e68284165edb8cd3e8799decef6e51e6ca2b95cb2aa"),
}


def test_untouched_output_for_unbridgeable_fixtures():
    for name, (make, expected) in PINNED.items():
        vals = SRC._extract_values(pd, make(), [], CTX)
        assert np.isfinite(vals).all(), name
        assert _sha(vals) == expected, name
