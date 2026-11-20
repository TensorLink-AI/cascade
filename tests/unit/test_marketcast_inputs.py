"""More model inputs (owner 2026-11-18): cross-market returns (ETH, the S&P
500 perp, gold, the euro), order flow the API serves only live (open
interest, the impact spread — sampled by the recorder from now on) and a
pinned scheduled-event calendar."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from cascade.eval import marketcast as M
from cascade.pool import marketcast_feeds as F
from cascade.pool import marketcast_relay as R
from cascade.pool.marketcast_store import MktBarStore, record_once
from cascade.shared.config.marketcast import parse_marketcast

T0 = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())


def _ctx_resp(names, ois, imps, mids):
    return [{"universe": [{"name": n} for n in names]},
            [{"openInterest": str(o), "impactPxs": [str(a), str(b)], "midPx": str(m)}
             for o, (a, b), m in zip(ois, imps, mids, strict=True)]]


def test_asset_contexts_parse_open_interest_and_impact_spread():
    got = F.parse_hl_asset_ctxs(_ctx_resp(["BTC", "ETH"], [100.0, 5.0],
                                          [(99.99, 100.01), (1.0, 1.0)], [100.0, 1.0]))
    assert got["BTC"][0] == 100.0 and got["BTC"][1] == pytest.approx(2.0)   # 0.02 / 100 = 2 bp
    assert got["ETH"][1] == 0.0
    with pytest.raises(F.FeedError):
        F.parse_hl_asset_ctxs({"nope": 1})


def test_ctx_on_grid_forward_fills_but_never_across_an_outage():
    rows = [((T0 + 60) * 1000, 10.0, 1.0), ((T0 + 600) * 1000, 12.0, 2.0)]
    oi, sp = F.ctx_on_grid(rows, T0, 300, step_minutes=1, max_stale_minutes=60)
    # a sample counts from the END of the step it lands in (known at block end)
    assert oi[0] == 10.0 and oi[8] == 10.0 and oi[9] == 12.0
    assert np.isfinite(oi[9 + 60]) and np.isnan(oi[9 + 61])       # > 60 min stale ⇒ missing
    assert sp[9] == 2.0


def test_the_recorder_samples_contexts_and_the_store_serves_them(tmp_path):
    mc = parse_marketcast({"assets": {"crypto": ["BTC"]}, "data_sources": {"crypto": "hyperliquid"},
                           "suites": ["returns"], "frequencies": {"returns": ["1m"]},
                           "horizons": {"returns": ["1m"]}})
    calls = []

    def post(url, body):
        calls.append(body["type"])
        if body["type"] == "metaAndAssetCtxs":
            return _ctx_resp(["BTC"], [42.0], [(99.0, 101.0)], [100.0])
        return []

    store = MktBarStore(tmp_path)
    now = T0 + 3600
    record_once(store, mc, post=post, now=now, lookback_minutes=60, backfill=False)
    assert calls.count("metaAndAssetCtxs") == 1                    # once per dex per pass
    bars = store.read("crypto", "BTC", T0 + 3000, 10)
    assert bars["oi"][-1] == 42.0 and bars["spread"][-1] == pytest.approx(200.0)
    assert np.isnan(bars["oi"][0])                                   # before the sample
    # and the relay carries ctx.jsonl to a training pod
    assert "crypto/BTC/ctx.jsonl" in R.relay_rels([("crypto", ["BTC"])], ("1m",), T0, T0 + 1)
    assert R._REL.fullmatch("crypto/BTC/ctx.jsonl")


def test_event_rows_flag_the_block_and_count_down():
    starts = T0 + 3600.0 * np.arange(6)
    ev = [T0 + 3600 * 2 + 1800]                                       # mid block 2
    r = M.event_rows(starts, ev, horizon_hours=10)
    assert r[0].tolist() == [0, 0, 1, 0, 0, 0]
    assert r[1][0] == pytest.approx(2.5 / 10) and r[1][3] == 1.0      # none ahead ⇒ 1
    assert M.event_rows(starts, []).tolist() == [[0] * 6, [1] * 6]


def test_build_covariates_orders_the_new_rows_by_the_fixed_list():
    n = 8
    past, known = M.build_covariates(
        ["spread", "cross_eth", "volume", "events"], block_starts=T0 + 60.0 * np.arange(n + 2),
        n_context=n, volume_series=np.ones(n),
        series={"cross_eth": np.full(n, 2.0), "spread": np.full(n, 3.0)},
        event_times=[T0 + 60.0 * (n + 1) + 1])
    # PAST_COVARIATES order: volume, cross_eth, …, spread
    assert past.shape == (3, n) and past[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert known.shape == (2, n + 2) and known[0, -1] == 1.0
    p2, _ = M.build_covariates(["open_interest"], block_starts=T0 + 60.0 * np.arange(n),
                               n_context=n)
    assert np.isnan(p2).all()                                         # not recorded ⇒ missing


def test_config_lists_the_inputs_and_parses_the_calendar():
    mc = parse_marketcast({"event_times": ["2026-12-10T19:00:00Z", "2026-11-13T13:30:00+00:00"]})
    assert mc.event_times == (datetime(2026, 11, 13, 13, 30, tzinfo=UTC).timestamp(),
                              datetime(2026, 12, 10, 19, tzinfo=UTC).timestamp())
    assert {"cross_eth", "cross_equity", "cross_gold", "cross_fx", "open_interest", "spread",
            "events"} <= set(mc.covariates)
    for bad in (["2026-12-10T19:00:00"], ["tomorrow"], "2026-12-10T19:00:00Z"):
        with pytest.raises(ValueError, match="event_times"):
            parse_marketcast({"event_times": bad})
