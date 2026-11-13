"""MarketCast redesign (owner 2026-11-10, DEC-CA-0064 addendum): native
candle banks (no recording lead time), Garman–Klass vol, the ``ohlcv`` /
``high`` / ``low`` suites with scale-normalised level CRPS, and the ONE
composite target ``mkt:market`` with its per-suite no-regression floor."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from cascade.eval import marketcast as M
from cascade.pool import marketcast as P
from cascade.pool import marketcast_feeds as F
from cascade.pool.marketcast_store import HL_DEPTH_CANDLES, MktBarStore, record_once
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import (
    COMPOSITE_LAUNCH_SUITES,
    MarketCastConfig,
    parse_marketcast,
)
from cascade.shared.marketcast_report import (
    MktEntryRows,
    MktWindowRow,
    score_windows,
    verdict_from_rows,
)

T0 = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())          # a UTC day boundary


def _candles(n, seed, *, step_minutes=1):
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3 * math.sqrt(step_minutes), n)))
    o = np.concatenate([[100.0], c[:-1]])
    hi = np.maximum(o, c) * (1 + rng.uniform(1e-4, 1e-3, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(1e-4, 1e-3, n))
    return {"open": o, "high": hi, "low": lo, "close": c, "volume": rng.uniform(1, 2, n),
            "funding": np.full(n, 1e-5), "premium": np.full(n, -3e-4)}


# ── config ───────────────────────────────────────────────────────────────────


def test_shipped_configs_are_the_composite_launch_set():
    for toml in ("chain.toml", "chain.testnet.toml"):
        cfg = load_chain_config(toml)
        mc = cfg.marketcast
        assert mc.from_block == 0 and mc.target_suites == ()          # inert
        assert mc.composite and mc.suites == COMPOSITE_LAUNCH_SUITES
        assert "funding" not in mc.suites
        assert mc.target_names == ("market",) and mc.targets == ("mkt:market",)
        assert cfg.lane_c.targets == ("mkt:market",)
        assert mc.native_intervals == ("5m", "15m")
        assert mc.native_history_map == {"5m": 21, "15m": 50}
        assert mc.ohlcv_open_weight == 0.0 and mc.composite_suite_floor == 0.02
        # every series reads a bank whose history the API depth (or the
        # recorder from arming) covers — no 21-day minute bank
        assert mc.snapshot_days == 10
        assert mc.bank_minutes(5) == 5 and mc.bank_minutes(15) == 15
        assert mc.bank_minutes(60) == mc.bank_minutes(1440) == 60 and mc.bank_minutes(1) == 1
        assert mc.series_labels("ohlcv") == ("ohlcv.h", "ohlcv.l", "ohlcv.c", "ohlcv.v")


def test_new_knobs_round_trip_and_refusals():
    mc = parse_marketcast({"composite": True,
                           "suites": ["returns", "vol", "ohlcv"],
                           "composite_suite_floor": 0.05, "ohlcv_open_weight": 0.5,
                           "native_intervals": ["5m"], "native_history_days": {"5m": 14}})
    assert mc.composite and mc.composite_suite_floor == 0.05 and mc.ohlcv_open_weight == 0.5
    assert mc.native_intervals == ("5m",) and mc.native_history_map == {"5m": 14, "15m": 50}
    assert mc.series_labels("ohlcv")[0] == "ohlcv.o"               # weighted open: scored
    assert mc.suites_of("market") == ("returns", "vol", "ohlcv")
    assert mc.suites_of("vol") == ("vol",)
    assert mc.bank_days(5) == 14 and mc.bank_days(60) == mc.coarse_history_days
    # the defaults: per-suite targets, no native bank (the pre-2026-11-10 layout)
    d = MarketCastConfig()
    assert not d.composite and d.targets == ("mkt:returns", "mkt:vol")
    assert d.native_intervals == () and d.bank_minutes(5) == 1
    for bad, msg in [({"composite": 1}, "composite must be"),
                     ({"composite": True, "suites": ["returns"]}, "at least two"),
                     ({"composite_suite_floor": -0.1}, "composite_suite_floor"),
                     ({"ohlcv_open_weight": 2}, "ohlcv_open_weight"),
                     ({"native_intervals": ["1h"]}, "native_intervals"),
                     ({"native_intervals": ["5m", "5m"]}, "native_intervals"),
                     ({"native_history_days": {"30m": 3}}, "native_history_days"),
                     ({"native_history_days": {"5m": 0}}, "native_history_days")]:
        with pytest.raises(ValueError, match=msg):
            parse_marketcast(bad)


def test_composite_target_grammar_and_recipe():
    from cascade.shared.config.lane_c import check_target
    from cascade.shared.marketcast_recipe import meta_problem, parse_marketcast_recipe

    mc = replace(load_chain_config("chain.toml").marketcast, from_block=7200)
    assert check_target("mkt:market", mkt_suites=mc.target_suites) == "mkt:market"
    with pytest.raises(ValueError):
        check_target("mkt:returns", mkt_suites=mc.target_suites)
    recipe = {
        "target": "mkt:market",
        "data": {"assets": {"crypto": ["BTC"]}, "class_weights": {"crypto": 1.0},
                 "frequencies": ["1h"], "start": "2026-01-01", "end": "2026-02-01"},
        "context_length": 512, "horizon_mix": {"1h": 1.0, "15m": 0.5},
        "covariates": [], "augmentations": [],
        "objective": {"suite_weights": {"returns": 1.0, "ohlcv": 1.0}},
        "optimizer": {"lr": 1e-4, "schedule": "cosine", "warmup_fraction": 0.05, "steps": 10,
                      "batch": 8, "weight_decay": 0.0, "clip": 1.0},
        "trainable": {"scope": "full"}, "synthetic_fraction": 0.0}
    r = parse_marketcast_recipe(recipe, mc)
    assert r.target == "mkt:market" and r.suite == "market"
    assert dict(r.suite_weights) == {"returns": 1.0, "ohlcv": 1.0}
    with pytest.raises(ValueError, match="horizon_mix"):
        parse_marketcast_recipe({**recipe, "horizon_mix": {"5d": 1.0}}, mc)
    meta = {"base_ref": mc.base_ref, "base_sha256": mc.base_sha256, "target": "mkt:market",
            "covariates": []}
    from cascade.shared.marketcast_recipe import META_REQUIRED

    meta.update({k: meta.get(k, 0) for k in META_REQUIRED if k not in meta})
    assert meta_problem(meta, mc) == ""
    assert "not an enabled" in meta_problem({**meta, "target": "mkt:returns"}, mc)


# ── series: one candle rule, whichever bank ──────────────────────────────────


def test_block_candle_is_exact_aggregation_and_bank_independent():
    one = _candles(60 * 24, 3)
    five = {k: v for k, v in M.block_ohlcv(one, 5).items()}
    assert five["open"][1] == one["open"][5] and five["close"][1] == one["close"][9]
    assert five["high"][2] == one["high"][10:15].max()
    assert five["low"][2] == one["low"][10:15].min()
    assert five["volume"][3] == pytest.approx(one["volume"][15:20].sum())
    for suite in ("returns", "vol", "range", "high", "low", "liquidity", "ohlcv.o",
                  "ohlcv.h", "ohlcv.l", "ohlcv.c", "ohlcv.v"):
        a = M.suite_series(suite, one, 15)
        b = M.suite_series(suite, five, 15, bar_minutes=5)
        np.testing.assert_allclose(a, b, rtol=1e-12, equal_nan=True, err_msg=suite)
    with pytest.raises(ValueError, match="ohlcv"):
        M.suite_series("ohlcv.x", one, 5)
    # a missing last bar leaves the close (block-end rule) missing
    gap = {k: v.copy() for k, v in one.items()}
    gap["close"][4] = np.nan
    assert np.isnan(M.block_ohlcv(gap, 5)["close"][0])


def test_garman_klass_vol():
    o, h, lo, c = np.array([100.0]), np.array([102.0]), np.array([99.0]), np.array([101.0])
    want = 1e4 * math.sqrt(0.5 * math.log(102 / 99) ** 2
                           - (2 * math.log(2) - 1) * math.log(101 / 100) ** 2)
    assert M.garman_klass_vol_bp(o, h, lo, c)[0] == pytest.approx(want)
    flat = M.garman_klass_vol_bp(np.ones(1), np.ones(1), np.ones(1), np.ones(1))
    assert np.isnan(flat[0])                                        # σ² = 0: missing
    one = _candles(600, 4)
    k = M.block_ohlcv(one, 60)
    np.testing.assert_allclose(
        M.suite_series("vol", one, 60),
        np.log(M.garman_klass_vol_bp(k["open"], k["high"], k["low"], k["close"])))
    hi = M.suite_series("high", one, 60)
    lo_ = M.suite_series("low", one, 60)
    assert (hi >= 0).all() and (lo_ >= 0).all()
    assert hi[0] == pytest.approx(1e4 * math.log(one["high"][:60].max() / one["open"][0]))


def test_level_scale_and_scale_normalised_crps():
    ctx = np.array([np.nan, 99.0, 101.0, np.nan])
    assert M.level_scale("ohlcv.c", ctx) == 101.0
    assert M.level_scale("ohlcv.v", np.array([1.0, 3.0])) == 2.0
    assert M.level_scale("returns", ctx) == 1.0
    assert math.isnan(M.level_scale("ohlcv.h", np.array([np.nan])))
    knots = (0.1, 0.5, 0.9)
    y = [np.array([110.0])]
    qb = [np.array([[100.0, 105.0, 110.0]])]
    qf = [np.array([[105.0, 110.0, 115.0]])]
    raw = score_windows(y, qb, qf, knots=knots, coverage_levels=(), gated=False)[0]
    sc = score_windows(y, qb, qf, knots=knots, coverage_levels=(), gated=False,
                       scales=[100.0])[0]
    assert sc[0] == pytest.approx(raw[0] / 100) and sc[1] == pytest.approx(raw[1] / 100)
    # BTC-sized and EUR-sized levels with the same relative error score the same
    big = score_windows([np.array([1.1e5])], [np.array([[1e5, 1.05e5, 1.1e5]])],
                        [np.array([[1.05e5, 1.1e5, 1.15e5]])], knots=knots,
                        coverage_levels=(), gated=False, scales=[1e5])[0]
    assert big[0] == pytest.approx(sc[0]) and big[1] == pytest.approx(sc[1])
    with pytest.raises(ValueError, match="one value per window"):
        score_windows(y, qb, qf, knots=knots, coverage_levels=(), gated=[True, False])


# ── the composite statistic and verdict ─────────────────────────────────────


def _losses(rng, suites, n_cls=3, n_blocks=12, per=6, ratio=None):
    ratio = ratio or {}
    b, f, sl, hz, cl = [], [], [], [], []
    for s in suites:
        for c in range(n_cls):
            for t in range(n_blocks):
                for _ in range(per):
                    x = float(rng.uniform(1, 2))
                    b.append(x)
                    f.append(x * ratio.get(s, 1.0) * float(rng.uniform(0.99, 1.01)))
                    sl.append(s)
                    hz.append("1h")
                    cl.append(f"c{c}|{t}")
    return np.array(b), np.array(f), sl, hz, cl


def test_composite_gain_is_the_geomean_of_suite_gains():
    rng = np.random.default_rng(0)
    b, f, sl, hz, cl = _losses(rng, ("returns", "vol", "ohlcv.c", "ohlcv.v"),
                               ratio={"returns": 0.9, "vol": 1.0, "ohlcv.c": 0.8, "ohlcv.v": 1.0})
    d = M.composite_gain_detail(b, f, clusters=cl, suites=sl, horizons=hz, alpha=0.05, B=300,
                                seed="s")
    per = d["suites"]
    assert set(per) == {"returns", "vol", "ohlcv"}                 # fields fold into ohlcv
    # ohlcv = mean of its fields' log ratios (c 0.8, v 1.0)
    assert 1 - per["ohlcv"]["gain"] == pytest.approx(math.sqrt(0.8), rel=5e-3)
    want = 1 - math.exp((math.log(0.9) + math.log(1.0) + 0.5 * math.log(0.8)) / 3)
    assert d["gain"] == pytest.approx(want, rel=2e-2)
    assert d["lcb"] <= d["gain"] and all(v["lcb"] <= v["gain"] for v in per.values())
    # an open field with weight 0 is not scored
    b2, f2, sl2, hz2, cl2 = _losses(rng, ("ohlcv.o", "ohlcv.c"),
                                    ratio={"ohlcv.o": 4.0, "ohlcv.c": 0.8})
    d2 = M.composite_gain_detail(b2, f2, clusters=cl2, suites=sl2, horizons=hz2, alpha=0.05,
                                 B=100, seed="s", label_weights={"ohlcv.o": 0.0})
    assert 1 - d2["suites"]["ohlcv"]["gain"] == pytest.approx(0.8, rel=2e-2)


def test_judge_composite_floor_margin_and_missing_suite():
    rng = np.random.default_rng(1)
    kw = dict(alpha=0.05, B=400, gain_margin=0.01, seed="s", suite_floor=0.02,
              expected_suites=("returns", "vol"))
    b, f, sl, hz, cl = _losses(rng, ("returns", "vol"), ratio={"returns": 0.9, "vol": 0.95})
    ok = M.judge_composite(b, f, suites=sl, horizons=hz, clusters=cl, coverage={
        "returns": M.CoverageGate(True, 0.05, ()), "vol": M.CoverageGate(True, 0.05, ())}, **kw)
    assert ok.passed and ok.reason == "win" and ok.suite == "market"
    assert set(ok.diagnostics["suites"]) == {"returns", "vol"}
    # a large gain on one suite cannot buy a regression on another
    b, f, sl, hz, cl = _losses(rng, ("returns", "vol"), ratio={"returns": 0.6, "vol": 1.1})
    bad = M.judge_composite(b, f, suites=sl, horizons=hz, clusters=cl, coverage={
        "returns": M.CoverageGate(True, 0.05, ()), "vol": M.CoverageGate(True, 0.05, ())}, **kw)
    assert not bad.passed and "suite regression" in bad.reason and "vol" in bad.reason
    # no gain at all
    b, f, sl, hz, cl = _losses(rng, ("returns", "vol"))
    flat = M.judge_composite(b, f, suites=sl, horizons=hz, clusters=cl, coverage={
        "returns": M.CoverageGate(True, 0.05, ()), "vol": M.CoverageGate(True, 0.05, ())}, **kw)
    assert not flat.passed and "margin" in flat.reason
    # an expected suite with no paired windows is inconclusive, never judged
    b, f, sl, hz, cl = _losses(rng, ("returns",), ratio={"returns": 0.8})
    miss = M.judge_composite(b, f, suites=sl, horizons=hz, clusters=cl, coverage={
        "returns": M.CoverageGate(True, 0.05, ())}, **kw)
    assert not miss.passed and miss.reason.startswith("inconclusive") and "vol" in miss.reason
    # a calibration-gated suite without its gate never passes
    b, f, sl, hz, cl = _losses(rng, ("returns", "vol"), ratio={"returns": 0.9, "vol": 0.95})
    nocov = M.judge_composite(b, f, suites=sl, horizons=hz, clusters=cl,
                              coverage={"returns": M.CoverageGate(True, 0.05, ())}, **kw)
    assert not nocov.passed and "coverage gate missing" in nocov.reason
    failed = M.judge_composite(b, f, suites=sl, horizons=hz, clusters=cl, coverage={
        "returns": M.CoverageGate(True, 0.05, ()),
        "vol": M.CoverageGate(False, 0.05, (M.CoverageRow(0.5, 0.2, True, False, "1h"),))},
        **kw)
    assert not failed.passed and "coverage gate failed" in failed.reason


def test_verdict_from_rows_routes_the_composite():
    mc = replace(load_chain_config("chain.toml").marketcast, bootstrap_B=200, min_windows=10,
                 min_clusters=2, min_time_blocks=2, suites=("returns", "ohlcv"))
    rng = np.random.default_rng(2)
    rows = []
    for s, ratio in (("returns", 0.9), ("ohlcv.c", 0.9)):
        for a in ("BTC", "ETH"):
            for t in range(40):
                x = float(rng.uniform(1, 2))
                start = T0 + 3600 * 6 * t
                cin = (1, 1, 1) if s == "returns" else ()
                tail = (0, 0) if s == "returns" else ()
                rows.append(MktWindowRow(f"crypto:{a}|{s}|1h|{start}", x, x * ratio,
                                         cin, 1 if s == "returns" else 0, float("nan"), tail))
    ent = MktEntryRows("hk", "ptr", "mkt:market", "judge", 0.0, windows=tuple(rows))
    v = verdict_from_rows(ent, mc, suite="market", k=1, seed="s")
    assert v.suite == "market" and set(v.diagnostics["suites"]) == {"returns", "ohlcv"}
    assert v.gain == pytest.approx(0.1, rel=5e-2)
    assert rows[0].suite == "returns" and rows[-1].suite == "ohlcv"
    assert rows[-1].suite_label == "ohlcv.c"


# ── banks, snapshots and target windows ─────────────────────────────────────


def _native_snapshot(tmp_path, mc, *, end):
    t0, minutes = P.snapshot_grid(end, mc.snapshot_days)
    ct0, hours = P.coarse_grid(end, mc.coarse_history_days)
    bars, cb, nat = {}, {}, {}
    pairs = sorted({x for s in mc.suites for x in mc.suite_assets(s)})
    for i, (cls, a) in enumerate(pairs):
        bars[(cls, a)] = _candles(minutes, i)
        cb[(cls, a)] = _candles(hours, 50 + i, step_minutes=60)
    for iv in mc.native_intervals:
        bm = F.HL_INTERVAL_MINUTES[iv]
        nt0, steps = P.native_grid(end, mc.native_history_map[iv], bm)
        nat[iv] = (nt0, steps, {p: _candles(steps, 90 + j, step_minutes=bm)
                                for j, p in enumerate(pairs)})
    P.write_snapshot(tmp_path, t0=t0, minutes=minutes, bars=bars, sources={}, anchor_block=0,
                     anchor_time=float(t0), coarse=cb, coarse_t0=ct0, coarse_steps=hours,
                     native=nat)
    return P.load_snapshot(tmp_path, verify_signature=False)


def test_native_banks_snapshot_and_composite_windows(tmp_path):
    mc = replace(load_chain_config("chain.toml").marketcast,
                 assets=(("crypto", ("BTC", "ETH")),), min_dollar_volume_24h=0.0,
                 windows_per_cell=4)
    end = T0 + 200 * 86400
    snap = _native_snapshot(tmp_path / "s", mc, end=end)
    assert set(snap.meta["native"]) == {"5m", "15m"}
    assert snap.meta["native"]["5m"]["bar_minutes"] == 5
    assert snap.bank("crypto", "BTC", "vol", 15)[2] == 15
    assert snap.bank("crypto", "BTC", "ohlcv.c", 5)[2] == 5
    assert snap.bank("crypto", "BTC", "returns", 1)[2] == 1
    assert snap.bank("crypto", "BTC", "vol", 60)[2] == 60
    after = end - 5 * 86400
    ws = P.target_windows(snap, mc, "market", after=after)
    assert [w.wid for w in ws] == sorted(w.wid for w in ws)
    labels = {w.suite for w in ws}
    assert {"returns", "vol", "high", "low", "range", "liquidity", "ohlcv.h", "ohlcv.l",
            "ohlcv.c", "ohlcv.v"} <= labels and "ohlcv.o" not in labels
    lv = [w for w in ws if w.suite == "ohlcv.c"]
    assert lv and all(w.scale == w.context[np.isfinite(w.context)][-1] for w in lv)
    assert all(w.scale == 1.0 for w in ws if w.suite == "returns")
    # a 15m vol window has a long context from the 15m bank, not the 10-day 1m bank
    v15 = [w for w in ws if w.suite == "vol" and w.horizon == "15m"]
    assert v15 and max(w.context.size for w in v15) > 10 * 96
    # one suite target is that suite's windows alone
    only = P.target_windows(snap, mc, "vol", after=after)
    assert {w.suite for w in only} == {"vol"}
    (tmp_path / "s/n5m/crypto/BTC/extra.npy").write_bytes(b"x")
    with pytest.raises(P.SnapshotError, match="unlisted"):
        P.load_snapshot(tmp_path / "s", verify_signature=False)
    # no native bank ⇒ the body carries no "native" record
    meta = P.write_snapshot(tmp_path / "n", t0=T0, minutes=4,
                            bars={("crypto", "BTC"): _candles(4, 1)}, sources={},
                            anchor_block=0, anchor_time=0.0)
    assert "native" not in meta


def _hl_post(seen):
    def post(url, body):
        if body["type"] == "candleSnapshot":
            iv = body["req"]["interval"]
            step = F.HL_INTERVAL_MINUTES[iv] * 60_000
            s, e = body["req"]["startTime"], body["req"]["endTime"]
            seen.append((iv, s, e))
            first = -(-s // step) * step
            return [{"t": t, "o": "1", "c": "1.01", "h": "1.02", "l": "0.99", "v": "3"}
                    for t in range(first, min(e, first + 600 * step), step)]
        return []
    return post


def test_build_snapshot_fetches_native_banks(tmp_path):
    mc = parse_marketcast({"assets": {"crypto": ["BTC"]}, "data_sources": {"crypto": "hyperliquid"},
                           "suites": ["vol", "returns"],
                           "frequencies": {"vol": ["15m"], "returns": ["5m"]},
                           "horizons": {"vol": ["15m"], "returns": ["5m"]},
                           "native_intervals": ["5m", "15m"],
                           "native_history_days": {"5m": 2, "15m": 3}})
    end = T0 + 40 * 86400 + 1234
    seen: list = []
    meta = P.build_snapshot(tmp_path / "s", mc, end_time=end, anchor_block=5, anchor_time=end,
                            post=_hl_post(seen), get=lambda u: [], days=1)
    assert {"5m", "15m"} <= {iv for iv, _s, _e in seen}
    assert meta["native"]["15m"]["t0"] % 86400 == 0
    assert meta["native"]["5m"]["steps"] == P.native_grid(end, 2, 5)[1]
    snap = P.load_snapshot(tmp_path / "s", verify_signature=False)
    assert np.isfinite(snap.bars("crypto", "BTC", bank="n15m")["close"]).any()


def test_store_records_native_intervals_and_backfills(tmp_path):
    mc = parse_marketcast({"assets": {"crypto": ["BTC"]}, "data_sources": {"crypto": "hyperliquid"},
                           "native_intervals": ["5m", "15m"]})
    store = MktBarStore(tmp_path)
    now = T0 + 30 * 86400 + 17
    seen: list = []
    got = record_once(store, mc, post=_hl_post(seen), now=now, lookback_minutes=120)
    assert got["crypto:BTC"] > 0
    first: dict = {}
    for iv, s_, _e in seen:                          # (a backfill pages: first page each)
        first.setdefault(iv, s_)
    assert list(first) == ["1m", "5m", "15m", "1h"]
    end_ms = now // 60 * 60 * 1000
    for iv in ("1m", "5m", "15m", "1h"):                 # empty store: the API's full depth
        assert end_ms - first[iv] == HL_DEPTH_CANDLES * F.HL_INTERVAL_MINUTES[iv] * 60_000
    assert store.has_bars("crypto", "BTC", "15m") and not store.has_bars("crypto", "ETH", "5m")
    seen.clear()
    record_once(store, mc, post=_hl_post(seen), now=now + 600, lookback_minutes=120)
    first = {}
    for iv, s_, _e in seen:
        first.setdefault(iv, s_)
    assert (now + 600) // 60 * 60 * 1000 - first["1m"] == 120 * 60_000      # lookback now
    assert (now + 600) // 60 * 60 * 1000 - first["1h"] == 120 * 60_000
    # read back on the 15m grid: recorded candles, NaN elsewhere
    t0 = (now - 86400) // 900 * 900
    r = store.read("crypto", "BTC", t0, 10, interval="15m")
    assert np.isfinite(r["close"]).any() and r["close"].shape == (10,)
    with pytest.raises(ValueError, match="grid"):
        store.read("crypto", "BTC", t0 + 60, 10, interval="15m")
    # the 1m layout is unchanged (day dirs directly under the asset)
    assert any(p.name[:1].isdigit() for p in (tmp_path / "crypto" / "BTC").iterdir())
