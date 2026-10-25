"""DEC-CA-0064 review fixes (2026-10-19), B3: the published quantile
sidecar + the shared loss rule, and ``cascade-audit marketcast --reproduce``
(its pure parts: deterministic sampling, the bar and quantile comparisons,
the orchestration over injected feeds / forecasters)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.stats import norm

from cascade.audit import marketcast_reproduce as R
from cascade.pool import marketcast as P
from cascade.shared.config.marketcast import TIMESTRAL_KNOTS, parse_marketcast
from cascade.shared.lane_c_manifest import LaneCEntry
from cascade.shared.manifest import format_trained_pointer
from cascade.shared.marketcast_report import (
    MktEntryRows,
    MktReport,
    MktWindowRow,
    decode_quantiles,
    encode_quantiles,
    load_mkt_report,
    score_windows,
)
from cascade.validator.marketcast import compute_entry_rows

T0 = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp())
Z = norm.ppf(np.asarray(TIMESTRAL_KNOTS))
MC = parse_marketcast({
    "from_block": 7200, "suites": ["returns"], "assets": {"crypto": ["BTC", "ETH"]},
    "frequencies": {"returns": ["1m", "5m"]}, "horizons": {"returns": ["1m", "5m"]},
    "eval_context_length": 64, "windows_per_cell": 8, "verdict_wait_blocks": 14400,
    "min_time_blocks": 4, "min_dollar_volume_24h": 0.0})


class _Fc:
    quantile_levels = TIMESTRAL_KNOTS

    def __init__(self, factor):
        self.factor = factor

    def forecast_quantiles(self, context, horizon, *, past_covariates=None,
                           future_covariates=None):
        c = np.asarray(context)
        sd = float(np.nanstd(c[np.isfinite(c)])) or 1.0
        return np.tile(Z * sd * self.factor, (int(horizon), 1))


def _snap(tmp_path):
    rng = np.random.default_rng(3)
    bars = {}
    for a in ("BTC", "ETH"):
        c = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, 3000)))
        bars[("crypto", a)] = {"open": c, "high": c, "low": c, "close": c,
                               "volume": np.ones(3000)}
    P.write_snapshot(tmp_path, t0=T0, minutes=3000, bars=bars, sources={"crypto": "hyperliquid"},
                     anchor_block=0, anchor_time=float(T0))
    return P.load_snapshot(tmp_path, verify_signature=False)


def _entry():
    ptr = format_trained_pointer("cascade/ckpt-ft@sha256:" + "c" * 64)
    return LaneCEntry("alice", 7, "mkt:returns", "alice/recipe@sha256:" + "1" * 64, ptr, 1,
                      marketcast={"train_time": float(T0 + 60 * 2000)})


def test_sidecar_round_trip_and_null_losses():
    qb = [np.random.default_rng(1).normal(size=(s, 21)) for s in (1, 5, 1)]
    qf = [q + 0.1 for q in qb]
    data = encode_quantiles(qb, qf)
    assert data == encode_quantiles(qb, qf)                        # deterministic bytes
    back = decode_quantiles(data, [1, 5, 1], 21)
    for (b, f), b0, f0 in zip(back, qb, qf, strict=True):
        np.testing.assert_array_equal(b, b0.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(f, f0.astype(np.float32).astype(np.float64))
    with pytest.raises(ValueError):
        decode_quantiles(data, [1, 5], 21)
    # a non-finite loss is published as null and reads back NaN; no tallies
    ys = [np.array([np.nan]), np.zeros(5), np.zeros(1)]
    out = score_windows(ys, qb, qf, knots=TIMESTRAL_KNOTS, coverage_levels=(0.5,), gated=True)
    assert np.isnan(out[0][0]) and out[0][2:] == ((), 0, ()) and out[1][3] == 5
    rows = MktEntryRows("a", "p", "mkt:returns", "judge", 1.0,
                        windows=(MktWindowRow("crypto:BTC|returns|1m|60", float("nan"), 1.0),))
    text = MktReport("r", 1, {}, (rows,)).to_json()
    assert '"base":null' in text and np.isnan(load_mkt_report(text).entries[0].windows[0].base)


def test_sample_and_compare_helpers():
    ids = [f"w{i}" for i in range(50)]
    s = R.sample_ids(ids, 7, "seed")
    assert len(s) == 7 and s == R.sample_ids(ids, 7, "seed") and s == sorted(s, key=ids.index)
    assert R.sample_ids(ids, 7, "other") != s
    a = np.array([[1.0, 2.0], [np.nan, 4.0]])
    assert R.compare_quantiles(a, a)[1]
    assert R.compare_quantiles(a, a * (1 + 1e-4))[1]
    assert not R.compare_quantiles(a, a + 0.1)[1]
    assert not R.compare_quantiles(a, np.array([[1.0, 2.0], [3.0, 4.0]]))[1]
    smp = [("crypto", "BTC", "1h", 0, 100.0), ("crypto", "ETH", "1m", 60, 50.0)]
    assert R.compare_bars(smp, lambda *a: {"BTC": 100.0, "ETH": 50.0}[a[1]],
                          tol_bp=1.0).status == "PASS"
    bad = R.compare_bars(smp, lambda *a: 100.5 if a[1] == "BTC" else 50.0, tol_bp=1.0)
    assert bad.status == "FAIL" and "BTC" in bad.detail
    assert R.compare_bars(smp, lambda *a: None, tol_bp=1.0).status == "WARN"


def test_reproduce_marketcast_over_injected_feed_and_forecasters(tmp_path):
    snap = _snap(tmp_path / "s")
    e = _entry()
    base, ft = _Fc(3.0), _Fc(1.0)
    after = float(e.marketcast["train_time"]) + MC.start_margin_seconds
    rows, data = compute_entry_rows(e, phase="judge", after=after, snap=snap, mc=MC,
                                    base_fc=base, load_ft=lambda p: (ft, ()), base_cache={},
                                    round_id="r1")
    assert rows.windows and hashlib.sha256(data).hexdigest() == rows.quantiles_sha256
    rep = MktReport("r1", 1, {"digest": snap.digest, "end": snap.end}, (rows,))
    reports = {"marketcast/round-r1.json": rep.to_json()}
    quantiles = {rows.quantiles_key: data}

    def fetch_ok(s):
        def _f(cls, a, interval, t):
            close = s.bars(cls, a)["close"]
            return float(close[(t - s.t0) // 60]) if interval == "1m" else None
        return _f

    fns = {"base": (base, ()), e.trained_pointer: (ft, ())}
    out = R.reproduce_marketcast(None, MC, reports=reports, quantiles=quantiles,
                                 snapshots={snap.digest: snap}, forecaster_fn=fns.get,
                                 sample=4, bar_fetch=fetch_ok)
    assert [r.status for r in out] == ["PASS", "PASS"], [r.detail for r in out]
    # a fine-tune that does not reproduce, a feed that disagrees
    fns2 = {"base": (base, ()), e.trained_pointer: (_Fc(1.05), ())}
    out = R.reproduce_marketcast(None, MC, reports=reports, quantiles=quantiles,
                                 snapshots={snap.digest: snap}, forecaster_fn=fns2.get,
                                 sample=4, bar_fetch=lambda s: (lambda *a: 1e9))
    assert [r.status for r in out] == ["FAIL", "FAIL"]
    # no torch / timestral: forecasts are reported as not re-run, never PASS
    out = R.reproduce_marketcast(None, MC, reports=reports, quantiles=quantiles,
                                 snapshots={snap.digest: snap}, forecaster_fn=None,
                                 sample=4, bar_fetch=fetch_ok)
    assert out[-1].status == "WARN" and "not installed" in out[-1].detail
    assert R.reproduce_marketcast(None, MC, reports=reports, quantiles=quantiles,
                                  snapshots={})[0].status == "WARN"
