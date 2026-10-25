"""MarketCast review fixes, second batch (2026-10-19, DEC-CA-0064 §Review
fixes II): M10 spread windows + class × time-block clusters + equal horizon
weight + min_time_blocks, M11 non-finite forecasts never masked, M12 the
embargo on the trainer and the pod clipped to the dispatch time, M13 stale /
illiquid windows out + equal class weight + the reported covariate baseline,
M14 the per-horizon two-sided coverage gate, M15 the forward bar recorder."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.stats import norm

from cascade.eval import marketcast as M
from cascade.pool import marketcast as P
from cascade.pool.marketcast_store import MktBarStore, record_once
from cascade.shared.config.marketcast import TIMESTRAL_KNOTS, parse_marketcast
from cascade.shared.lane_c_manifest import LaneCEntry
from cascade.shared.manifest import format_trained_pointer
from cascade.shared.marketcast_report import (
    MktEntryRows,
    MktWindowRow,
    calibration_gate,
    verdict_from_rows,
    window_weights,
)

T0 = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp())
Z = norm.ppf(np.asarray(TIMESTRAL_KNOTS))


def _row(cls, asset, h, start, base, ft, cov=(), n=0, tail=()):
    return MktWindowRow(f"{cls}:{asset}|returns|{h}|{int(start)}", base, ft, tuple(cov), n,
                        float("nan"), tuple(tail))


# ── M10 ──────────────────────────────────────────────────────────────────────


def test_horizons_weigh_equally_and_clusters_are_class_by_time_block():
    rng = np.random.default_rng(0)
    b = rng.uniform(1, 2, 300)
    hz = ["1m"] * 290 + ["1d"] * 10
    f = np.where(np.array(hz) == "1m", b * 0.5, b)            # 1m improves, 1d does not
    cl = [f"crypto|{i % 30}" for i in range(300)]
    g, lcb = M.clustered_gain(b, f, clusters=cl, horizons=hz, alpha=0.05, B=500, seed="s")
    assert g == pytest.approx(1 - math.sqrt(0.5))            # not ≈ 0.5 (window-weighted)
    assert lcb <= g
    pooled, _ = M.clustered_gain(b, f, clusters=cl, horizons=["x"] * 300, alpha=0.05, B=200,
                                 seed="s")
    assert pooled > 0.45
    w = _row("crypto", "BTC", "1m", 3600 * 7, 1.0, 1.0)
    assert w.time_cluster(6) == "crypto|" + str((3600 * 7) // (6 * 3600))


def test_min_time_blocks_makes_a_one_regime_sample_inconclusive():
    mc = parse_marketcast({"min_windows": 10, "min_clusters": 1, "min_time_blocks": 3,
                           "bootstrap_B": 200, "verdict_wait_blocks": 43200})
    rows = [_row("crypto", a, "1h", T0 + 60 * i, 2.0, 1.0) for a in ("BTC", "ETH")
            for i in range(30)]                                # all inside one 6 h block
    v = verdict_from_rows(MktEntryRows("h", "p", "mkt:range", "judge", 0.0, windows=tuple(rows)),
                          mc, suite="range", k=1, seed="s")
    assert not v.passed and "time blocks" in v.reason
    spread = [_row("crypto", a, "1h", T0 + 6 * 3600 * i, 2.0, 1.0) for a in ("BTC", "ETH")
              for i in range(30)]
    v = verdict_from_rows(MktEntryRows("h", "p", "mkt:range", "judge", 0.0,
                                       windows=tuple(spread)), mc, suite="range", k=1, seed="s")
    assert v.passed and v.n_clusters == 30


# ── M13 ──────────────────────────────────────────────────────────────────────


def test_each_asset_class_weighs_the_same_inside_a_horizon():
    rows = ([_row("crypto", f"C{i}", "1h", T0, 1.0, 1.0) for i in range(28)]
            + [_row("metals", f"M{i}", "1h", T0, 1.0, 1.0) for i in range(4)]
            + [_row("metals", "M9", "1h", T0, float("nan"), 1.0)])
    w = window_weights(rows)
    assert sum(w[:28]) == pytest.approx(1.0) and sum(w[28:32]) == pytest.approx(1.0)
    assert w[-1] == 0.0


def _snap(tmp_path, *, price=100.0, vol=1.0, flat_after=None, minutes=3000):
    rng = np.random.default_rng(3)
    c = price * np.exp(np.cumsum(rng.normal(0, 1e-3, minutes)))
    if flat_after is not None:
        c[flat_after:] = c[flat_after]
    v = np.full(minutes, float(vol))
    P.write_snapshot(tmp_path, t0=T0, minutes=minutes,
                     bars={("crypto", "BTC"): {"open": c, "high": c, "low": c, "close": c,
                                               "volume": v}},
                     sources={"crypto": "hyperliquid"}, anchor_block=0, anchor_time=float(T0))
    return P.load_snapshot(tmp_path, verify_signature=False)


MC = parse_marketcast({"from_block": 7200, "suites": ["returns"], "assets": {"crypto": ["BTC"]},
                       "frequencies": {"returns": ["1m", "5m"]},
                       "horizons": {"returns": ["1m", "5m"]}, "eval_context_length": 64,
                       "windows_per_cell": 20, "verdict_wait_blocks": 14400,
                       "min_time_blocks": 2})


def test_stale_and_illiquid_windows_are_not_eligible(tmp_path):
    after = T0 + 60 * 1500
    # $100 × 1 unit/minute = $144k a day: below the $1M floor
    thin = _snap(tmp_path / "thin")
    assert P.snapshot_windows(thin, MC, "returns", after=after) == []
    assert P.snapshot_windows(thin, replace(MC, min_dollar_volume_24h=100_000.0), "returns",
                              after=after)
    # a market that stops trading: once its CONTEXT is flat its windows are out
    # (red-team F3: read from the context known at the start, never the target)
    rich = replace(MC, min_dollar_volume_24h=0.0)
    flat = _snap(tmp_path / "flat", flat_after=2200)
    ws = P.snapshot_windows(flat, rich, "returns", after=after)
    assert ws and all(w.start <= T0 + 60 * (2200 + w.freq_minutes * M.MIN_CONTEXT) for w in ws)
    # a feed gap: under min_bar_coverage in the 24 h before ⇒ out
    gap = _snap(tmp_path / "gap", minutes=3000)
    c = gap.bars("crypto", "BTC")["close"]
    c[1000:2400] = np.nan
    ok = P.liquidity_filter(gap, rich, "crypto", "BTC", "returns", 1)
    assert not ok(T0 + 60 * 2400) and ok(T0 + 60 * 900)


# ── M11 + the covariate baseline ─────────────────────────────────────────────


class _Fc:
    quantile_levels = TIMESTRAL_KNOTS

    def __init__(self, factor, nan_at=None):
        self.factor, self.nan_at, self.n = factor, nan_at, 0

    def forecast_quantiles(self, context, horizon, *, past_covariates=None,
                           future_covariates=None):
        self.n += 1
        c = np.asarray(context)
        q = np.tile(Z * (float(np.nanstd(c)) or 1.0) * self.factor, (int(horizon), 1))
        if self.nan_at is not None and self.n == self.nan_at:
            q[0, 3] = np.nan
        return q


def _entry():
    return LaneCEntry("alice", 7, "mkt:returns", "a/r@sha256:" + "1" * 64,
                      format_trained_pointer("cascade/ft@sha256:" + "c" * 64), 1,
                      marketcast={"train_time": float(T0 + 60 * 1500)})


def test_non_finite_finetune_is_a_fault_and_non_finite_base_is_excluded(tmp_path):
    from cascade.validator.marketcast import compute_entry_rows, judge_marketcast_entry

    snap = _snap(tmp_path / "s")
    mc = replace(MC, min_dollar_volume_24h=0.0)
    kw = dict(phase="judge", after=T0 + 60 * 1560.0, snap=snap, mc=mc, round_id="r")
    rows, data = compute_entry_rows(_entry(), base_fc=_Fc(2.0), base_cache={},
                                    load_ft=lambda p: (_Fc(1.0, nan_at=3), ()), **kw)
    assert "non-finite" in rows.fault and data is None
    rows, _ = compute_entry_rows(_entry(), base_fc=_Fc(2.0, nan_at=2), base_cache={},
                                 load_ft=lambda p: (_Fc(1.0), ("calendar",)), **kw)
    assert not rows.fault and sum(1 for w in rows.windows if math.isnan(w.base)) == 1
    assert all(math.isfinite(w.bc) for w in rows.windows)     # the covariate baseline ran
    j = judge_marketcast_entry(_entry(), rows, mc=replace(mc, min_windows=5, min_clusters=1),
                               k=1, phase="judge", seed="s", round_block=1, origin_block=1)
    assert j.mkt["base_excluded"] == 1 and "gain_vs_base_cov" in j.mkt
    assert j.n == len(rows.windows) - 1


# ── M14 ──────────────────────────────────────────────────────────────────────


def test_coverage_gate_is_per_horizon_and_two_sided():
    mc = parse_marketcast({})                       # gate 50/80 %, tol 0.05 under, 0.10 over

    def rows(h, ins, tail=(5, 5)):                  # ins = inside counts for (50, 80, 98)
        return [_row("crypto", "BTC", h, T0 + 60 * i, 1.0, 1.0, ins, 100, tail)
                for i in range(3)]

    good = rows("1h", (50, 80, 98)) + rows("1d", (50, 80, 98))
    assert calibration_gate(good, mc, mc.coverage_levels).passed
    one_bad = rows("1h", (50, 80, 98)) + rows("1d", (50, 70, 98))    # pooled would pass 0.75
    g = calibration_gate(one_bad, mc, mc.coverage_levels)
    assert not g.passed and [(r.horizon, r.nominal) for r in g.rows if r.under and r.gated] == [
        ("1d", 0.8)]
    wide = rows("1h", (65, 80, 98))                                  # 0.65 > 0.5 + 0.10
    assert not calibration_gate(wide, mc, mc.coverage_levels).passed
    tails = rows("1h", (50, 80, 80))                                 # 98 % is report-only
    g = calibration_gate(tails, mc, mc.coverage_levels)
    assert g.passed and any(r.nominal == 0.98 and r.under and not r.gated for r in g.rows)
    with pytest.raises(ValueError, match="coverage_gate_levels"):
        parse_marketcast({"coverage_gate_levels": [0.9]})


# ── M12 ──────────────────────────────────────────────────────────────────────


RECIPE = {"target": "mkt:returns",
          "data": {"assets": {"crypto": ["BTC"]}, "class_weights": {"crypto": 1.0},
                   "frequencies": ["1m", "5m"], "start": "2026-09-01", "end": "2026-09-10"},
          "context_length": 128, "horizon_mix": {"1m": 1.0},
          "covariates": [], "augmentations": [],
          "objective": {"suite_weights": {"returns": 1.0}},
          "optimizer": {"lr": 1e-4, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 5,
                        "batch": 4, "weight_decay": 0.0, "clip": 1.0},
          "trainable": {"scope": "full"}, "synthetic_fraction": 0.0}


def test_trainer_enforces_the_embargo_and_the_pod_is_clipped(tmp_path):
    from cascade.shared.config import load_chain_config
    from cascade.trainer.finetune import RecipeError
    from cascade.trainer.marketcast_leg import build_training_snapshot
    from cascade.trainer.marketcast_runner import read_mkt_recipe

    mc = replace(MC, embargo_days=30)
    cfg = replace(load_chain_config("chain.toml"), marketcast=mc)
    d = tmp_path / "tree"
    d.mkdir()
    (d / "marketcast.json").write_text(json.dumps(RECIPE))
    commit = datetime(2026, 9, 20, tzinfo=UTC).timestamp()           # only 10 days later
    with pytest.raises(RecipeError, match="embargo"):
        read_mkt_recipe(d, cfg, target="mkt:returns", submission_time=commit)
    ok = read_mkt_recipe(d, cfg, target="mkt:returns",
                         submission_time=datetime(2026, 10, 19, tzinfo=UTC).timestamp(),
                         train_time=1.5e9)
    assert json.loads(ok.to_json())["train_time"] == 1.5e9
    # the pod never fetches a bar at or after the dispatch time
    from cascade.shared.marketcast_recipe import parse_marketcast_recipe

    rec = parse_marketcast_recipe(RECIPE, mc)
    asked = []

    def post(url, body):
        asked.append(body.get("req", body))
        return []

    stop = datetime(2026, 9, 5, 12, tzinfo=UTC).timestamp()
    build_training_snapshot(tmp_path / "t", mc, rec, post=post, get=lambda u: [],
                            not_after=stop)
    assert asked and max(int(b["endTime"]) for b in asked) < stop * 1000


# ── M15: the forward recorder ────────────────────────────────────────────────


def _candles(t_from, n, price=100.0):
    return [{"t": (t_from + 60 * i) * 1000, "o": str(price), "c": str(price + i),
             "h": str(price + i + 1), "l": str(price - 1), "v": "2"} for i in range(n)]


def test_bar_store_records_closed_bars_resumes_and_keeps_gaps(tmp_path):
    store = MktBarStore(tmp_path / "bars")
    now = T0 + 86400 + 600                                      # 00:10 on day 2
    from cascade.pool.marketcast_feeds import parse_hl_candles

    rows = parse_hl_candles(_candles(T0 + 86400 - 300, 11))     # 23:55 … 00:05 (open)
    assert store.write_bars("crypto", "BTC", rows, now=T0 + 86400 + 300) == 10
    got = store.read("crypto", "BTC", T0 + 86400 - 300, 11)
    assert np.isfinite(got["close"][:10]).all() and np.isnan(got["close"][10])   # open bar out
    # a resumed pass re-fetches and fills the open minute; nothing duplicated
    store.write_bars("crypto", "BTC", rows, now=now)
    assert np.isfinite(store.read("crypto", "BTC", T0 + 86400 - 300, 11)["close"]).all()
    # a gap (never recorded) stays NaN — never interpolated
    late = parse_hl_candles(_candles(T0 + 86400 + 1200, 5))
    store.write_bars("crypto", "BTC", late, now=T0 + 86400 + 2000)
    g = store.read("crypto", "BTC", T0 + 86400 - 300, 30)["close"]
    assert np.isnan(g[11:25]).all() and np.isfinite(g[25:30]).all()
    store.write_funding("crypto", "BTC", [((T0 + 86400) * 1000, 1e-5, -3e-4)])
    f = store.read("crypto", "BTC", T0 + 86400 - 300, 30)["funding"]
    assert np.isnan(f[0]) and f[10] == 1e-5
    assert 0 < store.coverage("crypto", "BTC", T0 + 86400 - 300, 30) < 1


def test_record_once_and_publish_from_the_store(tmp_path):
    mc = parse_marketcast({"assets": {"crypto": ["BTC", "ETH"]},
                           "data_sources": {"crypto": "hyperliquid"}})
    now = T0 + 3 * 86400 + 30
    seen = []

    def post(url, body):
        if body["type"] == "candleSnapshot":
            seen.append(body["req"]["coin"])
            if body["req"]["coin"] == "ETH":
                raise OSError("429")
            s = body["req"]["startTime"] // 1000
            return [c for c in _candles(now // 60 * 60 - 600, 10) if c["t"] >= s * 1000]
        return []

    out = record_once(MktBarStore(tmp_path / "b"), mc, post=post, now=now,
                      lookback_minutes=60)
    assert out["crypto:BTC"] == 10 and str(out["crypto:ETH"]).startswith("error")
    with pytest.raises(ValueError):
        record_once(MktBarStore(tmp_path / "b"), mc, post=post, now=now, lookback_minutes=9000)
    b, _rep = P.fetch_class_bars("crypto", "BTC", ("hyperliquid",), now // 60 * 60 - 600, 10,
                                 post=lambda *a: pytest.fail("the store, not the API"),
                                 get=lambda u: [], store=MktBarStore(tmp_path / "b"))
    assert np.isfinite(b["close"]).all()
