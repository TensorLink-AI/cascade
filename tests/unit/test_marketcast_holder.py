"""The paired challenger-vs-holder slot test (owner 2026-11-18): the slot
holder is forecast on the challenger's own windows, the challenger takes the
slot only by beating it THERE (not by a higher LCB recorded on another
week's windows), and the audit recomputes the holder's losses."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import TIMESTRAL_KNOTS
from cascade.shared.lane_c import TargetWin, win_beats
from cascade.shared.marketcast_report import (
    MktEntryRows,
    MktReport,
    MktWindowRow,
    decode_quantiles,
    encode_quantiles,
    holder_gain,
    load_mkt_report,
    score_windows,
)
from cascade.validator import marketcast as V

T0 = 1_790_000_000 // 86400 * 86400


def _win(lcb, *, block=10, ptr="p-new", holder="", beats=False):
    return TargetWin("mkt:market", "hk", 1, f"r{block}", block, trained_pointer=ptr,
                     base_pointer="base", gain_lcb=lcb, holder_pointer=holder,
                     beats_holder=beats)


def test_a_paired_win_decides_the_slot_not_the_recorded_lcb():
    cur = _win(0.08, block=5, ptr="p-holder")
    # paired against the holder: beating it there takes the slot even with a lower LCB …
    assert win_beats(_win(0.03, holder="p-holder", beats=True), cur)
    # … and losing there keeps it, however high the LCB vs the base
    assert not win_beats(_win(0.20, holder="p-holder", beats=False), cur)
    # paired against someone else (the slot moved): the recorded-LCB rule
    assert win_beats(_win(0.09, holder="p-other", beats=False), cur)
    assert not win_beats(_win(0.07), cur)


def _mc(**kw):
    return replace(load_chain_config("chain.toml").marketcast, bootstrap_B=300,
                   suites=("returns", "vol"), **kw)


def _rows(ratio_vs_holder, n=60, seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for s in ("returns", "vol"):
        for a in ("BTC", "ETH"):
            for t in range(n):
                b, h = float(rng.uniform(1, 2)), float(rng.uniform(1, 2))
                out.append(MktWindowRow(f"crypto:{a}|{s}|1hx48|{T0 + 21600 * t}", b, h *
                                        ratio_vs_holder, hold=h))
    return MktEntryRows("hk", "p-new", "mkt:market", "confirm", 0.0, windows=tuple(out),
                        holder={"hotkey": "hk0", "pointer": "p-holder"})


def test_holder_gain_is_the_composite_statistic_vs_the_holder():
    mc = _mc()
    up = holder_gain(_rows(0.9), mc, k=1, seed="s")
    assert up["pointer"] == "p-holder" and up["beats"] and up["gain"] == pytest.approx(0.1)
    down = holder_gain(_rows(1.0), mc, k=1, seed="s")
    assert not down["beats"] and down["gain"] == pytest.approx(0.0, abs=1e-12)
    assert holder_gain(replace(_rows(0.9), holder={}), mc, k=1, seed="s") is None


def test_holder_losses_round_trip_the_signed_report():
    ent = _rows(0.9, n=1)
    rep = MktReport("r", 1, {}, (ent, replace(ent, holder={}, windows=tuple(
        replace(w, hold=math.nan) for w in ent.windows))))
    back = load_mkt_report(rep.to_json()).entries
    assert back[0].holder == ent.holder and back[0].windows[0].hold == ent.windows[0].hold
    assert back[1].holder == {} and math.isnan(back[1].windows[0].hold)
    assert b'"hold"' not in MktReport("r", 1, {}, (back[1],)).canonical_body()


def test_holder_record_scores_the_holder_on_the_entrys_windows(monkeypatch):
    mc = _mc()
    K = len(TIMESTRAL_KNOTS)
    rng = np.random.default_rng(1)
    wins = [SimpleNamespace(wid=f"crypto:BTC|returns|1mx60|{T0 + 3600 * i}", steps=3,
                            target=rng.normal(size=3), scale=2.0) for i in range(4)]
    qb = [np.sort(rng.normal(size=(3, K)), axis=-1).astype(np.float32) for _ in wins]
    qf = [np.sort(rng.normal(size=(3, K)), axis=-1).astype(np.float32) for _ in wins]
    qh = [np.sort(rng.normal(size=(3, K)), axis=-1).astype(np.float32) for _ in wins]
    monkeypatch.setattr("cascade.pool.marketcast.target_windows", lambda *a, **k: wins)
    snap = SimpleNamespace(digest="d")
    cache = {("p-holder", "d", w.wid): q for w, q in zip(wins, qh, strict=True)}
    rows = MktEntryRows("hk", "p-new", "mkt:market", "confirm", 0.0, windows=tuple(
        MktWindowRow(w.wid, 1.0, 1.0) for w in wins))
    data = encode_quantiles(qb, qf)
    out, blob = V.holder_record(rows, data, ("hk0", "p-holder", 3), snap=snap, mc=mc,
                                load_ft=lambda p: pytest.fail("cached"), ft_cache=cache,
                                round_id="r")
    want = score_windows([w.target for w in wins], qb, qh, knots=TIMESTRAL_KNOTS,
                         coverage_levels=(), gated=False, scales=[2.0] * 4)
    assert [w.hold for w in out.windows] == [x[1] for x in want]
    assert out.holder["pointer"] == "p-holder" and blob is not None
    back = decode_quantiles(blob, [3] * 4, K)
    assert all(np.array_equal(a[1], b) for a, b in zip(back, qh, strict=True))
    # no holder, or the holder is the entry itself: unchanged
    assert V.holder_record(rows, data, None, snap=snap, mc=mc, load_ft=None, ft_cache={},
                           round_id="r") == (rows, None)
    assert V.holder_record(rows, data, ("hk", "p-new", 1), snap=snap, mc=mc, load_ft=None,
                           ft_cache={}, round_id="r")[1] is None
