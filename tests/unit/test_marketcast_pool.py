"""MarketCast (DEC-CA-0064) data half: the extra series builders, window
geometry / forward-only cutting, covariates, the feed adapters (hand-written
fixtures in the live endpoints' shapes), the signed snapshot and its window
draw, and the ``cascade-pool --lane mkt`` wiring."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import numpy as np
import pytest

from cascade.eval import marketcast as M
from cascade.pool import marketcast as P
from cascade.pool import marketcast_feeds as F
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import parse_marketcast

# ── hand-written fixtures (shapes read from the public endpoints 2026-10-19) ──

T0 = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())          # a UTC day boundary
HL_CANDLES = [
    {"t": T0 * 1000, "T": T0 * 1000 + 59999, "s": "BTC", "i": "1m", "o": "82565.0",
     "c": "82449.0", "h": "82610.0", "l": "82443.0", "v": "48.69329", "n": 660},
    {"t": T0 * 1000 + 60000, "T": T0 * 1000 + 119999, "s": "BTC", "i": "1m", "o": "82449.0",
     "c": "82478.0", "h": "82484.0", "l": "82415.0", "v": "25.84373", "n": 400},
    # minute 2 missing; minute 3 present
    {"t": T0 * 1000 + 180000, "T": T0 * 1000 + 239999, "s": "BTC", "i": "1m", "o": "82477.0",
     "c": "82467.0", "h": "82486.0", "l": "82452.0", "v": "10.8458", "n": 156},
]
HL_FUNDING = [
    {"coin": "BTC", "fundingRate": "0.0000125", "premium": "-0.0003172973",
     "time": T0 * 1000 + 52},
    {"coin": "BTC", "fundingRate": "0.0000250", "premium": "-0.0003042616",
     "time": T0 * 1000 + 120000 + 55},
]
BINANCE = [
    [T0 * 1000, "82516.00000000", "82518.01000000", "82462.00000000", "82462.01000000",
     "14.69985000", T0 * 1000 + 59999, "1212651.99105000", 1719, "0.75455000",
     "62243.28287130", "0"],
    [T0 * 1000 + 60000, "82462.01000000", "82471.33000000", "82462.01000000",
     "82471.33000000", "1.27173000", T0 * 1000 + 119999, "104871.18667470", 282,
     "1.22040000", "100637.93381910", "0"],
]


def test_request_builders_and_symbols():
    assert F.hl_coin("hyperliquid", "BTC") == "BTC"
    assert F.hl_coin("hyperliquid_xyz", "SP500") == "xyz:SP500"
    with pytest.raises(ValueError):
        F.hl_coin("binance_spot", "BTC")
    assert F.hl_candle_request("xyz:GOLD", 1, 2) == {
        "type": "candleSnapshot",
        "req": {"coin": "xyz:GOLD", "interval": "1m", "startTime": 1, "endTime": 2}}
    assert F.hl_funding_request("BTC", 1, 2) == {"type": "fundingHistory", "coin": "BTC",
                                                 "startTime": 1, "endTime": 2}
    assert F.binance_klines_url("BTCUSDT", 1, 2) == (
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime=1"
        "&endTime=2&limit=1000")
    assert F.binance_symbol("ETH") == "ETHUSDT" and F.alpha_netuid("SN64") == 64
    with pytest.raises(ValueError):
        F.alpha_netuid("64")


def test_parsers_and_shape_drift():
    rows = F.parse_hl_candles(HL_CANDLES)
    assert rows[0] == (T0 * 1000, 82565.0, 82610.0, 82443.0, 82449.0, 48.69329)
    assert F.parse_hl_funding(HL_FUNDING)[1][1] == pytest.approx(0.000025)
    assert F.parse_binance_klines(BINANCE)[1][4] == pytest.approx(82471.33)
    for bad in ({"t": 1}, [{"t": 1, "o": "x"}], [{"t": 1, "o": "nan", "h": "1", "l": "1",
                                                  "c": "1", "v": "1"}]):
        with pytest.raises(F.FeedError):
            F.parse_hl_candles(bad)
    with pytest.raises(F.FeedError):
        F.parse_binance_klines([[1, "2"]])


def test_paginated_fetch_pages_forward():
    calls = []

    def post(url, body):
        assert url == F.HL_INFO_URL
        calls.append(body["req"]["startTime"])
        start = body["req"]["startTime"]
        return [r for r in HL_CANDLES if r["t"] >= start][:2]       # a 2-row page

    got = F.fetch_hl_candles(post, "BTC", T0 * 1000, T0 * 1000 + 4 * 60000)
    assert [r[0] for r in got] == [T0 * 1000, T0 * 1000 + 60000, T0 * 1000 + 180000]
    assert calls == [T0 * 1000, T0 * 1000 + 120000]               # stops at the end

    def get(url):
        start = int(url.split("startTime=")[1].split("&")[0])
        return [r for r in BINANCE if r[0] >= start][:1]

    assert len(F.fetch_binance_klines(get, "BTCUSDT", T0 * 1000, T0 * 1000 + 120000)) == 2


def test_onto_the_grid():
    b = F.bars_on_grid(F.parse_hl_candles(HL_CANDLES), T0, 5)
    assert b["close"][0] == 82449.0 and np.isnan(b["close"][2]) and b["close"][3] == 82467.0
    assert np.isnan(b["close"][4])
    fu, pr = F.funding_on_grid(F.parse_hl_funding(HL_FUNDING), T0, 5)
    # the first sample (t0 + 52 ms) applies from minute 1; the second from minute 3
    assert np.isnan(fu[0]) and fu[1] == fu[2] == pytest.approx(0.0000125)
    assert fu[3] == fu[4] == pytest.approx(0.000025) and pr[4] == pytest.approx(-0.0003042616)
    rows = [F.AlphaRow(T0 + 61, 100, 64, 0.02, 1000.0, 50_000.0),
            F.AlphaRow(T0 + 70, 101, 64, 0.021, 1001.0, 49_000.0)]
    a = F.alpha_on_grid(rows, T0, 3)
    assert np.isnan(a["price"][0]) and a["price"][1] == 0.021 and np.isnan(a["price"][2])
    rep = F.cross_check([100.0, 100.0, np.nan], [100.0, 101.0, 5.0], tol_bp=50)
    assert rep["n"] == 2 and rep["share_over_tol"] == 0.5


def test_alpha_recorder_round_trip(tmp_path):
    p = tmp_path / "alpha.jsonl"
    F.append_alpha_row(p, F.AlphaRow(1, 2, 64, 0.5, 1.0, 2.0))
    F.append_alpha_row(p, F.AlphaRow(3, 4, 8, 0.6, 1.0, 2.0))
    assert [r.netuid for r in F.load_alpha_rows(p)] == [64, 8]
    assert F.load_alpha_rows(tmp_path / "missing") == []


# ── extra series builders ────────────────────────────────────────────────────


def test_spread_adv_alpha_basis():
    hi = np.array([101.0, 101.0, 102.0, 102.0])
    lo = np.array([100.0, 100.0, 101.0, 101.0])
    s = M.corwin_schultz_spread_bp(hi, lo, 2)
    assert s.shape == (2,) and np.all(s >= 0) and np.isfinite(s).all()
    # identical bars ⇒ β = 2 ln²(H/L), γ = ln²(H/L) ⇒ hand formula
    b = 2 * math.log(101 / 100) ** 2
    g = math.log(101 / 100) ** 2
    k = 3 - 2 * math.sqrt(2)
    a = (math.sqrt(2 * b) - math.sqrt(b)) / k - math.sqrt(g / k)
    want = max(0.0, 2 * (math.exp(a) - 1) / (1 + math.exp(a)))
    assert s[0] == pytest.approx(1e4 * want)
    dv = M.dollar_volume([10.0, 10.0, 20.0, 20.0], [1.0, 2.0, 1.0, np.nan], 2)
    np.testing.assert_allclose(dv, [30.0, 20.0])
    np.testing.assert_allclose(M.adv([1.0, 2.0, 3.0, np.nan], 2), [np.nan, 1.5, 2.5, np.nan])
    np.testing.assert_allclose(M.alpha_price([10.0, 0.0], [5.0, 1.0]), [2.0, np.nan])
    np.testing.assert_allclose(M.log_depth([math.e, -1.0]), [1.0, np.nan])
    np.testing.assert_allclose(M.basis_bp([0.0001]), [1.0])
    np.testing.assert_allclose(M.funding_bp([0.0002]), [2.0])


def test_suite_series_at_a_frequency():
    close = np.exp(np.linspace(0, 0.01, 10)) * 100
    bars = {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
            "volume": np.ones(10), "funding": np.arange(10) * 1e-5, "price": close}
    r = M.suite_series("returns", bars, 5)
    assert np.isnan(r[0]) and r[1] == pytest.approx(1e4 * (math.log(close[9]) - math.log(close[4])))
    assert M.suite_series("liquidity", bars, 5)[0] == pytest.approx(math.log(5.0))
    assert M.suite_series("funding", bars, 5)[1] == pytest.approx(1e4 * 5e-5)
    assert np.isfinite(M.suite_series("vol", bars, 5)).all()
    assert np.isfinite(M.suite_series("range", bars, 5)).all()
    np.testing.assert_allclose(M.suite_series("alpha", bars, 5), r)
    with pytest.raises(ValueError):
        M.suite_series("prices", bars, 5)


def test_window_geometry():
    f = ("1m", "5m", "1h", "1d")
    assert M.window_geometry(f, "1m") == (1, 1)
    assert M.window_geometry(f, "4h") == (60, 4)
    assert M.window_geometry(f, "5d") == (1440, 5)
    with pytest.raises(ValueError):
        M.window_geometry(("1h",), "5m")


def test_cut_windows_forward_only_and_stable_ids():
    s = np.arange(200, dtype=np.float64)
    t0 = T0
    ws = M.cut_windows(s, cls="crypto", asset="BTC", suite="returns", horizon="5m",
                       freq_minutes=1, steps=5, t0=t0, context_length=50, after=t0 + 60 * 100,
                       now=t0 + 60 * 150, min_context=10)
    # origins on the absolute 5-minute grid, start strictly after `after`, end <= now
    assert [w.index for w in ws] == [105, 110, 115, 120, 125, 130, 135, 140, 145]
    assert all(w.start > t0 + 6000 and w.end <= t0 + 9000 for w in ws)
    assert ws[0].wid == f"crypto:BTC|returns|5m|{t0 + 6300}"
    np.testing.assert_array_equal(ws[0].target, s[105:110])
    assert ws[0].context.size == 50
    # review M10: `limit` spreads the kept windows EVENLY over the eligible span
    # (first … last), never the most recent hours only
    spread = M.cut_windows(s, cls="crypto", asset="BTC", suite="returns", horizon="5m",
                           freq_minutes=1, steps=5, t0=t0, context_length=50, after=t0,
                           now=t0 + 60 * 150, min_context=10, limit=3)
    assert [w.index for w in spread] == [10, 80, 145]
    assert M.spread_select(5, 2) == [0, 4] and M.spread_select(3, 9) == [0, 1, 2]
    # red-team F3: the realised target never filters a window (a flat one stays;
    # its cell is masked in the verdict if its base loss is ~0); a failed
    # START-time eligibility test is out
    flat = np.zeros(200)
    assert len(M.cut_windows(flat, cls="c", asset="a", suite="returns", horizon="5m",
                             freq_minutes=1, steps=5, t0=t0, context_length=50, after=t0,
                             now=t0 + 60 * 150, min_context=10)) == 28
    assert M.cut_windows(s, cls="c", asset="a", suite="returns", horizon="5m", freq_minutes=1,
                         steps=5, t0=t0, context_length=50, after=t0, now=t0 + 60 * 150,
                         min_context=10, eligible=lambda st: st > t0 + 60 * 120)[0].index == 125
    with pytest.raises(ValueError):
        M.cut_windows(s, cls="c", asset="a", suite="returns", horizon="5m", freq_minutes=5,
                      steps=1, t0=t0 + 60, context_length=5, after=0, now=1e12)


def test_covariates_and_us_eastern_dst():
    # 2026: DST from Sun 8 Mar 07:00 UTC to Sun 1 Nov 06:00 UTC
    assert M._us_eastern_offset_hours(datetime(2026, 3, 8, 6, 59, tzinfo=UTC).timestamp()) == -5
    assert M._us_eastern_offset_hours(datetime(2026, 3, 8, 7, 0, tzinfo=UTC).timestamp()) == -4
    assert M._us_eastern_offset_hours(datetime(2026, 11, 1, 5, 59, tzinfo=UTC).timestamp()) == -4
    assert M._us_eastern_offset_hours(datetime(2026, 11, 1, 6, 0, tzinfo=UTC).timestamp()) == -5
    # Friday 2026-10-16: 13:30 UTC = 09:30 EDT (cash open); 21:00 UTC = 17:00 EDT (rollover)
    t_open = datetime(2026, 10, 16, 13, 30, tzinfo=UTC).timestamp()
    t_roll = datetime(2026, 10, 16, 21, 0, tzinfo=UTC).timestamp()
    t_sat = datetime(2026, 10, 17, 15, 0, tzinfo=UTC).timestamp()
    rows = M.session_rows([t_open, t_roll, t_sat])
    np.testing.assert_array_equal(rows, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    cal = M.calendar_rows([0.0, 6 * 3600.0])
    assert cal.shape == (4, 2) and cal[0, 1] == pytest.approx(1.0)
    past, known = M.build_covariates(["session", "volume", "calendar"],
                                     block_starts=np.arange(5) * 60.0, n_context=3,
                                     volume_series=[1.0, 2.0, 3.0])
    assert past.shape == (1, 3) and known.shape == (7, 5)
    past, known = M.build_covariates(["funding"], block_starts=np.arange(5) * 60.0,
                                     n_context=3)
    assert np.isnan(past).all() and known is None
    with pytest.raises(ValueError):
        M.build_covariates(["macro_events"], block_starts=[0.0], n_context=1)


def test_coverage_counts_reaggregate():
    q = np.tile(np.array([-1.0, 0.0, 1.0]), (4, 1))
    ins, n = M.window_coverage_counts(q, [0.5, -0.5, 2.0, np.nan], (0.1, 0.5, 0.9), (0.8,))
    assert ins == [2] and n == 3
    emp = M.coverage_from_counts([[2], [1]], [3, 1], (0.8,))
    assert emp == {0.8: 0.75}


# ── the snapshot ─────────────────────────────────────────────────────────────


def _mc(**over):
    raw = {"suites": ["returns", "vol"], "assets": {"crypto": ["BTC", "ETH"]},
           "min_dollar_volume_24h": 0.0,
           "frequencies": {"returns": ["1m", "5m"], "vol": ["5m"]},
           "horizons": {"returns": ["5m"], "vol": ["5m"]},
           "eval_context_length": 64, "windows_per_cell": 10}
    raw.update(over)
    return parse_marketcast(raw)


def _bars(minutes, seed):
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, minutes)))
    return {"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c,
            "volume": rng.uniform(1, 2, minutes), "funding": np.full(minutes, 1e-5)}


def test_snapshot_write_load_tamper(tmp_path):
    bars = {("crypto", "BTC"): _bars(600, 1), ("crypto", "ETH"): _bars(600, 2)}
    meta = P.write_snapshot(tmp_path / "s", t0=T0, minutes=600, bars=bars,
                            sources={"crypto": "hyperliquid+binance_spot"},
                            anchor_block=1000, anchor_time=float(T0))
    again = P.write_snapshot(tmp_path / "s2", t0=T0, minutes=600, bars=bars,
                             sources={"crypto": "hyperliquid+binance_spot"},
                             anchor_block=1000, anchor_time=float(T0))
    assert P.snapshot_digest(meta) == P.snapshot_digest(again)      # deterministic bytes
    snap = P.load_snapshot(tmp_path / "s", verify_signature=False)
    assert snap.assets("crypto") == ["BTC", "ETH"] and snap.end == T0 + 36000
    assert snap.block_time(1010) == T0 + 120.0
    np.testing.assert_array_equal(snap.bars("crypto", "BTC")["close"],
                                  bars[("crypto", "BTC")]["close"])
    with pytest.raises(P.SnapshotError, match="signature"):
        P.load_snapshot(tmp_path / "s", signer_hotkey="5Main")       # unsigned is refused
    arr = np.load(tmp_path / "s/bars/crypto/BTC/close.npy")
    arr[0] += 1
    np.save(tmp_path / "s/bars/crypto/BTC/close.npy", arr)
    with pytest.raises(P.SnapshotError, match="sha256"):
        P.load_snapshot(tmp_path / "s", verify_signature=False)
    (tmp_path / "s2/bars/crypto/BTC/extra.npy").write_bytes(b"x")
    with pytest.raises(P.SnapshotError, match="unlisted"):
        P.load_snapshot(tmp_path / "s2", verify_signature=False)


def test_snapshot_signing_with_a_fake_hotkey(tmp_path, monkeypatch):
    class _HK:
        ss58_address = "5Main"

        def sign(self, body):
            return b"sig:" + body[:8]

    meta = P.write_snapshot(tmp_path / "s", t0=T0, minutes=10,
                            bars={("crypto", "BTC"): _bars(10, 1)}, sources={"crypto": "x"},
                            anchor_block=1, anchor_time=0.0, signer="5Main",
                            wallet=type("W", (), {"hotkey": _HK()})())
    assert meta["signature"] == (b"sig:" + P.canonical_body(meta)[:8]).hex()
    import cascade.shared.sr25519 as sr

    monkeypatch.setattr(sr, "verify_hotkey",
                        lambda hk, body, sig: hk == "5Main" and sig == (b"sig:" + body[:8]).hex())
    assert P.load_snapshot(tmp_path / "s", signer_hotkey="5Main").meta["signer"] == "5Main"
    with pytest.raises(P.SnapshotError):
        P.load_snapshot(tmp_path / "s", signer_hotkey="5Other")


def test_build_snapshot_from_feeds(tmp_path):
    mc = _mc()
    end = T0 + 10 * 60

    def post(url, body):
        if body["type"] == "candleSnapshot":
            s = body["req"]["startTime"]
            return [r for r in HL_CANDLES if r["t"] >= s] if body["req"]["coin"] == "BTC" else []
        return [r for r in HL_FUNDING if r["time"] >= body["startTime"]] \
            if body["coin"] == "BTC" else []

    def get(url):
        s = int(url.split("startTime=")[1].split("&")[0])
        return [r for r in BINANCE if r[0] >= s]

    meta = P.build_snapshot(tmp_path / "s", mc, end_time=end, anchor_block=5, anchor_time=end,
                            post=post, get=get, days=1)
    # ETH answered nothing ⇒ left out; BTC carries funding + a Binance cross-check
    assert meta["assets"] == {"crypto": ["BTC"]}
    assert "bars/crypto/BTC/funding.npy" in meta["files"]
    assert meta["cross_check"]["crypto:BTC"]["binance_spot"]["n"] == 2
    assert meta["t0"] % 86400 == 0 and meta["end"] == end


def test_snapshot_windows_and_covariates(tmp_path):
    mc = _mc()
    bars = {("crypto", "BTC"): _bars(2000, 1), ("crypto", "ETH"): _bars(2000, 2)}
    P.write_snapshot(tmp_path / "s", t0=T0, minutes=2000, bars=bars,
                     sources={"crypto": "hyperliquid"}, anchor_block=0, anchor_time=float(T0))
    snap = P.load_snapshot(tmp_path / "s", verify_signature=False)
    after = T0 + 60 * 1500
    ws = P.snapshot_windows(snap, mc, "returns", after=after)
    assert {w.asset for w in ws} == {"BTC", "ETH"}
    assert all(w.start > after and w.end <= snap.end for w in ws)
    assert len(ws) == 2 * 10                                         # windows_per_cell
    assert ws == sorted(ws, key=lambda w: w.wid)
    w = ws[0]
    past, known = P.window_covariates(snap, w, ("cross_asset", "funding", "volume", "calendar",
                                                "session"), context_length=64)
    assert past.shape == (3, w.context.size) and known.shape == (7, w.context.size + w.steps)
    assert np.isfinite(past[1]).all()                                # funding present
    vol = P.snapshot_windows(snap, mc, "vol", after=after)
    assert vol and all(w.suite == "vol" for w in vol)
    assert P.snapshot_windows(snap, mc, "returns", after=snap.end) == []


def test_pool_cli_prefix_and_reveal_lane():
    import argparse

    from cascade.pool.cli import pool_prefix

    cfg = load_chain_config("chain.toml")
    assert pool_prefix(argparse.Namespace(lane="mkt", prefix=None), cfg) == "pool_mkt"
    assert pool_prefix(argparse.Namespace(lane="a", prefix=None), cfg) == "pool"
    from cascade.pool.cli import main

    for lane in ("zzz", "c"):          # MarketCast-only lane C: no pool_c lane
        with pytest.raises(SystemExit):
            main(["publish", "--lane", lane])


def test_publish_mkt_refuses_a_wrong_signer(tmp_path, monkeypatch, capsys):
    from dataclasses import replace

    import cascade.pool.cli as cli

    cfg = load_chain_config("chain.toml")
    cfg = replace(cfg, storage=replace(cfg.storage, pool_bucket="b"),
                  netconfig=replace(cfg.netconfig, owner_hotkey="5Main"))
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(cli, "_resolve_effective_block", lambda *_a, **_k: 7200)

    class _W:
        class hotkey:  # noqa: N801
            ss58_address = "5Other"

    import cascade.shared.chain as chain

    monkeypatch.setattr(chain, "open_wallet", lambda *a, **k: _W())
    rc = cli.main(["publish", "--lane", "mkt", "--out", str(tmp_path / "o"),
                   "--wallet-name", "w", "--wallet-hotkey", "h", "--anchor-block", "1"])
    assert rc == 2 and "--bar-store" in capsys.readouterr().err      # review M15
    rc = cli.main(["publish", "--lane", "mkt", "--out", str(tmp_path / "o"),
                   "--bar-store", str(tmp_path / "bars"),
                   "--wallet-name", "w", "--wallet-hotkey", "h", "--anchor-block", "1"])
    assert rc == 2 and "owner_hotkey" in capsys.readouterr().err


def test_snapshot_meta_is_json_canonical(tmp_path):
    meta = P.write_snapshot(tmp_path / "s", t0=T0, minutes=4,
                            bars={("crypto", "BTC"): _bars(4, 1)}, sources={}, anchor_block=0,
                            anchor_time=0.0)
    on_disk = json.loads((tmp_path / "s" / P.SNAPSHOT_FILE).read_text())
    assert on_disk == meta and P.canonical_body(on_disk) == P.canonical_body(meta)


# ── review fixes 2026-10-19 (B4): the coarse bank, scorability ──────────────


def _perp(n, seed, *, step_minutes=1):
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3 * math.sqrt(step_minutes), n)))
    return {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
            "volume": rng.uniform(1, 2, n), "funding": 1e-5 + 1e-6 * rng.standard_normal(n),
            "premium": np.full(n, -3e-4)}


def _alpha(n, seed):
    rng = np.random.default_rng(seed)
    p = 0.01 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    return {"price": p, "tao_in": np.full(n, 1e4), "alpha_in": 1e4 / p}


def _full_snapshot(tmp_path, mc, *, coarse=True):
    """A synthetic FULL-LENGTH snapshot: ``snapshot_days`` of minute bars and
    ``coarse_history_days`` of hourly bars, every asset present."""
    end = T0 + 200 * 86400
    t0, minutes = P.snapshot_grid(end, mc.snapshot_days)
    ct0, hours = P.coarse_grid(end, mc.coarse_history_days)
    bars, cb = {}, {}
    for i, (cls, a) in enumerate(sorted({x for s in mc.suites for x in mc.suite_assets(s)})):
        mk = _alpha if cls == "alpha" else _perp
        bars[(cls, a)] = mk(minutes, i)
        cb[(cls, a)] = _alpha(hours, 100 + i) if cls == "alpha" else _perp(hours, 100 + i,
                                                                        step_minutes=60)
    P.write_snapshot(tmp_path, t0=t0, minutes=minutes, bars=bars, sources={},
                     anchor_block=0, anchor_time=float(t0),
                     coarse=cb if coarse else None, coarse_t0=ct0, coarse_steps=hours)
    return P.load_snapshot(tmp_path, verify_signature=False)


def test_every_default_horizon_yields_windows_per_asset(tmp_path):
    from cascade.shared.config.marketcast import BLOCK_SECONDS, SUITES

    assets = {"crypto": ["BTC", "ETH"], "fx": ["EUR"], "alpha": ["SN1"]}
    liq = {"min_dollar_volume_24h": 0.0}            # the fixture's volumes are tiny
    for mc in (parse_marketcast({"assets": assets, **liq}),               # the shipped suites
               parse_marketcast({"assets": assets, "suites": list(SUITES), **liq})):
        snap = _full_snapshot(tmp_path / "-".join(mc.suites), mc)
        # the worst case: dispatched a whole verdict wait before the snapshot's end
        after = snap.end - mc.verdict_wait_blocks * BLOCK_SECONDS + mc.start_margin_seconds
        for suite in mc.suites:
            ws = P.snapshot_windows(snap, mc, suite, after=after)
            for h in mc.horizons_map[suite]:
                for cls, a in mc.suite_assets(suite):
                    n = sum(1 for w in ws if w.horizon == h and w.asset == a)
                    assert n > 0, (suite, h, cls, a)
    # the bug the bank fixes: minute bars alone (21 days) hold no daily window
    mc = parse_marketcast({"assets": {"crypto": ["BTC"]}, **liq})
    old = _full_snapshot(tmp_path / "old", mc, coarse=False)
    ws = P.snapshot_windows(old, mc, "returns", after=old.end - 6 * 86400)
    assert not [w for w in ws if w.horizon == "1d"] and [w for w in ws if w.horizon == "4h"]


def test_coarse_bank_snapshot_and_covariates(tmp_path):
    mc = parse_marketcast({"assets": {"crypto": ["BTC", "ETH"]}, "min_dollar_volume_24h": 0.0})
    snap = _full_snapshot(tmp_path / "s", mc)
    c = snap.meta["coarse"]
    assert c["bar_minutes"] == 60 and c["t0"] % 86400 == 0
    assert any(k.startswith("coarse/crypto/BTC/") for k in snap.meta["files"])
    w = next(w for w in P.snapshot_windows(snap, mc, "vol", after=snap.end - 5 * 86400)
             if w.horizon == "1d")
    assert w.freq_minutes == 1440 and w.context.size >= 32       # 180 days of daily RV
    past, known = P.window_covariates(snap, w, ("cross_asset", "funding", "volume", "calendar",
                                                "session"), context_length=4096)
    assert past.shape == (3, w.context.size) and known.shape[1] == w.context.size + w.steps
    assert np.isfinite(past[:, 1:]).all()                 # (block 0's return is NaN)
    # 5m returns still read the minute bars
    w5 = next(w for w in P.snapshot_windows(snap, mc, "returns", after=snap.end - 86400)
              if w.horizon == "5m")
    assert snap.bank("crypto", "BTC", "returns", w5.freq_minutes)[2] == 1
    (tmp_path / "s/coarse/crypto/BTC/extra.npy").write_bytes(b"x")
    with pytest.raises(P.SnapshotError, match="unlisted"):
        P.load_snapshot(tmp_path / "s", verify_signature=False)
    # no bank ⇒ the body is exactly the pre-bank one
    meta = P.write_snapshot(tmp_path / "n", t0=T0, minutes=4,
                            bars={("crypto", "BTC"): _bars(4, 1)}, sources={}, anchor_block=0,
                            anchor_time=0.0)
    assert "coarse" not in meta and not any(k.startswith("coarse/") for k in meta["files"])


def test_build_snapshot_fetches_the_coarse_bank_as_hourly_candles(tmp_path):
    mc = parse_marketcast({"assets": {"crypto": ["BTC"]}, "data_sources": {"crypto": "hyperliquid"},
                           "coarse_history_days": 40})
    end = T0 + 40 * 86400 + 1234
    seen = []

    def post(url, body):
        if body["type"] == "candleSnapshot":
            iv = body["req"]["interval"]
            seen.append(iv)
            step = F.HL_INTERVAL_MINUTES[iv] * 60_000
            s, e = body["req"]["startTime"], body["req"]["endTime"]
            first = -(-s // step) * step
            return [{"t": t, "o": "1", "c": "1.01", "h": "1.02", "l": "0.99", "v": "3"}
                    for t in range(first, min(e, first + 500 * step), step)]
        return []

    meta = P.build_snapshot(tmp_path / "s", mc, end_time=end, anchor_block=5, anchor_time=end,
                            post=post, get=lambda u: [], days=1)
    assert {"1m", "1h"} <= set(seen)
    ct0, hours = P.coarse_grid(end, 40)
    assert meta["coarse"] == {"t0": ct0, "steps": hours, "bar_minutes": 60}
    assert ct0 + 3600 * hours == end // 3600 * 3600       # complete hours only
    snap = P.load_snapshot(tmp_path / "s", verify_signature=False)
    assert np.isfinite(snap.bars("crypto", "BTC", bank="coarse")["close"]).all()


def test_load_check_refuses_an_unscorable_horizon():
    from cascade.shared.config.marketcast import scorability_problems

    with pytest.raises(ValueError, match="unscorable.*returns/5d"):
        parse_marketcast({"horizons": {"returns": ["1h", "1d", "5d"], "vol": ["5m"]}})
    with pytest.raises(ValueError, match="unscorable.*returns/1d.*history"):
        parse_marketcast({"coarse_history_days": 21})
    with pytest.raises(ValueError, match="unscorable.*aligned"):
        parse_marketcast({"verdict_wait_blocks": 3600})              # 12 h wait vs 1d windows
    with pytest.raises(ValueError, match="coarse_history_days >= snapshot_days"):
        parse_marketcast({"coarse_history_days": 7})
    with pytest.raises(ValueError, match="report_timeout_settlements"):
        parse_marketcast({"report_timeout_settlements": 0})
    for path in ("chain.toml", "chain.testnet.toml"):                 # the shipped tomls pass
        mc = load_chain_config(path).marketcast
        assert scorability_problems(mc) == [] and "5d" not in mc.horizons_map["returns"]
        assert (mc.coarse_history_days, mc.report_timeout_settlements,
                mc.max_infra_deferrals) == (180, 8, 8)
