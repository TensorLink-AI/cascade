"""The composite's balance (owner 2026-11-18): per-suite weights so each
forecast property counts once, and three no-regression guards beside CRPS —
MASE of the median, QLIKE of the vol forecast's mean variance and the tail
quantile score — each judged like a suite and failing the entry when the
fine-tune is confidently worse than the base."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from cascade.eval import marketcast as M
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import (
    COMPOSITE_LAUNCH_SUITES,
    TIMESTRAL_KNOTS,
    parse_marketcast,
)
from cascade.shared.marketcast_report import (
    GUARD_FIELDS,
    MktEntryRows,
    MktReport,
    MktWindowRow,
    guard_scores,
    load_mkt_report,
    verdict_from_rows,
)

K = np.asarray(TIMESTRAL_KNOTS)
Z = np.array([-2.326, -1.645, -1.282, -1.036, -0.842, -0.674, -0.524, -0.385, -0.253, -0.126,
              0.0, 0.126, 0.253, 0.385, 0.524, 0.674, 0.842, 1.036, 1.282, 1.645, 2.326])
T0 = 1_790_000_000 // 86400 * 86400


def _normal_q(mu, sd):
    return np.asarray(mu, dtype=np.float64)[..., None] + np.asarray(sd)[..., None] * Z


# ── the metrics ─────────────────────────────────────────────────────────────


def test_median_ae_is_the_mase_numerator():
    q = _normal_q(np.array([[1.0, 2.0]]), np.array([[0.5, 0.5]]))
    y = np.array([[1.5, np.nan]])
    assert M.window_median_ae(q, y)[0] == pytest.approx(0.5)
    assert np.isnan(M.window_median_ae(q, np.full((1, 2), np.nan))[0])


def test_qlike_prefers_the_calibrated_variance_forecast():
    rng = np.random.default_rng(0)
    mu, sd, n = 3.0, 0.4, 20000
    y = rng.normal(mu, sd, (n, 1))                       # realised log vol
    true = _normal_q(np.full((n, 1), mu), np.full((n, 1), sd))
    narrow = _normal_q(np.full((n, 1), mu), np.full((n, 1), 0.05))   # right median, no spread
    low = _normal_q(np.full((n, 1), mu - 0.3), np.full((n, 1), sd))
    lt, ln_, ll = (float(np.mean(M.window_qlike_logvol(q, y))) for q in (true, narrow, low))
    assert lt < ln_ and lt < ll
    # scale-free, finite at huge levels (computed in logs), zero iff exact
    big = M.window_qlike_logvol(_normal_q(np.full((1, 1), 400.0), np.full((1, 1), 1e-9)),
                                np.full((1, 1), 400.0))
    assert np.isfinite(big[0]) and big[0] == pytest.approx(0.0, abs=1e-6)


def test_tail_score_rewards_honest_tails():
    rng = np.random.default_rng(1)
    n = 20000
    y = rng.standard_normal((n, 1))
    honest = _normal_q(np.zeros((n, 1)), np.ones((n, 1)))
    thin = honest.copy()
    thin[..., [0, 1, -2, -1]] *= 0.6                     # collapsed 1/5/95/99 % knots
    a, b = (float(np.mean(M.window_tail_score(q, y))) for q in (honest, thin))
    assert a < b


def test_guard_scores_scale_and_skip():
    q = [_normal_q(np.array([1.0]), np.array([0.5]))] * 3
    ys = [np.array([2.0])] * 3
    out = guard_scores(ys, q, q, knots=TIMESTRAL_KNOTS,
                       guards=[("mase",), ("mase", "qlike", "tail"), ()], scales=[2.0, 1.0, 1.0])
    assert out[0] == {"mase_b": pytest.approx(0.5), "mase_f": pytest.approx(0.5)}
    assert set(out[1]) == set(GUARD_FIELDS) and out[2] == {}
    assert out[1]["qlike_b"] > 0


# ── config ──────────────────────────────────────────────────────────────────


def test_shipped_balance_and_guards():
    for toml in ("chain.toml", "chain.testnet.toml"):
        mc = load_chain_config(toml).marketcast
        w = mc.suite_weight_map
        assert set(w) == set(COMPOSITE_LAUNCH_SUITES) and sum(w.values()) == pytest.approx(1.0)
        # each property counts once: location / dispersion / extremes / activity
        f = w["ohlcv"] / 4
        assert w["returns"] + f == pytest.approx(0.30)
        assert w["vol"] + w["range"] == pytest.approx(0.30)
        assert w["high"] + w["low"] + 2 * f == pytest.approx(0.25)
        assert w["liquidity"] + f == pytest.approx(0.15)
        assert mc.guards == ("mase", "qlike", "tail") and mc.guard_floor == 0.0
        assert mc.guards_of("returns") == ("tail",)
        assert mc.guards_of("vol") == ("mase", "qlike", "tail")
        assert mc.guards_of("ohlcv.c") == ("mase",) and mc.guards_of("liquidity") == ("mase",)


def test_balance_and_guard_config_bounds():
    assert parse_marketcast({}).guards_of("vol") == ()            # per-suite slots: no guards
    for bad in ({"guards": ["mase", "mase"]}, {"guards": ["sharpe"]}, {"guard_floor": 1.0},
                {"composite_suite_weights": {"nope": 1.0}},
                {"composite_suite_weights": {"vol": 0.0}},
                {"composite_suite_weights": [1.0]}):
        with pytest.raises(ValueError):
            parse_marketcast(bad)
    with pytest.raises(ValueError, match="tail guard"):
        parse_marketcast({"quantile_knots": [0.02, 0.1, 0.5, 0.9, 0.98], "tail_levels": [0.1, 0.9], "coverage_levels": [0.5, 0.8]})
    assert parse_marketcast({"quantile_knots": [0.02, 0.1, 0.5, 0.9, 0.98], "tail_levels": [0.1, 0.9], "coverage_levels": [0.5, 0.8], "guards": []}).guards == ()


# ── the verdict ─────────────────────────────────────────────────────────────


def _mc(**kw):
    return replace(load_chain_config("chain.toml").marketcast, bootstrap_B=300, min_windows=10,
                   min_clusters=2, min_time_blocks=2, suites=("returns", "vol"), **kw)


def _rows(*, crps=0.9, mase=1.0, qlike=1.0, tail=1.0, n=60, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for s in ("returns", "vol"):
        for a in ("BTC", "ETH"):
            for t in range(n):
                x, g = float(rng.uniform(1, 2)), float(rng.uniform(1, 2))
                gv = {"tail_b": g, "tail_f": g * tail}
                if s == "vol":
                    gv |= {"mase_b": g, "mase_f": g * mase, "qlike_b": g, "qlike_f": g * qlike}
                rows.append(MktWindowRow(f"crypto:{a}|{s}|1h|{T0 + 21600 * t}", x, x * crps,
                                         (1, 1, 1), 2, float("nan"), (0, 0), **gv,
                                         bl=x * 1.2))
    return MktEntryRows("hk", "ptr", "mkt:market", "judge", 0.0, windows=tuple(rows))


def test_guards_pass_a_clean_win_and_fail_a_bought_one():
    mc = _mc(coverage_tolerance=0.45, coverage_tolerance_over=0.45, tail_tolerance=0.45)
    v = verdict_from_rows(_rows(), mc, suite="market", k=1, seed="s")
    assert v.passed, v.reason
    assert set(v.diagnostics["guards"]) == {"mase", "qlike", "tail"}
    assert v.diagnostics["guards"]["qlike"]["gain"] == pytest.approx(0.0, abs=1e-9)
    for kw, name in (({"mase": 1.08}, "mase"), ({"qlike": 1.08}, "qlike"),
                     ({"tail": 1.08}, "tail")):
        v = verdict_from_rows(_rows(**kw), mc, suite="market", k=1, seed="s")
        assert not v.passed and "guard regression" in v.reason and name in v.reason
        assert v.gain == pytest.approx(0.1, rel=5e-2)         # CRPS alone would have won
        assert verdict_from_rows(_rows(**kw), replace(mc, guards=()), suite="market", k=1,
                                 seed="s").passed


def test_a_report_without_guard_losses_fails_the_guards():
    mc = _mc(coverage_tolerance=0.45, coverage_tolerance_over=0.45, tail_tolerance=0.45)
    ent = _rows()
    bare = replace(ent, windows=tuple(replace(w, **{k: math.nan for k in GUARD_FIELDS})
                                      for w in ent.windows))
    v = verdict_from_rows(bare, mc, suite="market", k=1, seed="s")
    assert not v.passed and "guard regression" in v.reason


def test_suite_weights_set_the_composite_balance():
    rng = np.random.default_rng(4)
    n = 400
    suites = ["returns"] * n + ["vol"] * n
    b = rng.uniform(1, 2, 2 * n)
    f = b * np.r_[np.full(n, 0.8), np.full(n, 1.0)]
    kw = {"clusters": [f"crypto|{i % 10}" for i in range(2 * n)], "suites": suites,
          "horizons": ["1h"] * (2 * n), "alpha": 0.05, "B": 50, "seed": "x"}
    eq = M.composite_gain_detail(b, f, **kw)["gain"]
    wt = M.composite_gain_detail(b, f, suite_weights={"returns": 0.75, "vol": 0.25}, **kw)["gain"]
    assert eq == pytest.approx(1 - math.exp(0.5 * math.log(0.8)))
    assert wt == pytest.approx(1 - math.exp(0.75 * math.log(0.8)))


def test_guard_losses_round_trip_the_signed_report():
    ent = _rows(n=2)
    rep = MktReport("r", 1, {}, (ent,))
    back = load_mkt_report(rep.to_json()).entries[0].windows
    assert MktReport("r", 1, {}, (replace(ent, windows=back),)).to_json() == rep.to_json()
    ret = [w for w in back if w.suite == "returns"][0]
    assert math.isnan(ret.mase_b) and ret.tail_b == ent.windows[0].tail_b
    assert b'"mase_b"' not in MktReport("r", 1, {}, (replace(
        ent, windows=ent.windows[:1]),)).canonical_body()       # NaN guards are omitted
