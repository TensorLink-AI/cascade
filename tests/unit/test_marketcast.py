"""MarketCast (DEC-CA-0064): inert config, target grammar, recipe, the pure
series / metric / verdict pieces and the base-model adapter."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval import marketcast as M
from cascade.eval import timestral_adapter as TA
from cascade.shared.config import load_chain_config
from cascade.shared.config.lane_c import (
    DEFAULT_TARGETS,
    check_target,
    is_mkt_target,
    parse_lane_c,
    parse_target,
)
from cascade.shared.config.marketcast import (
    TIMESTRAL_KNOTS,
    MarketCastConfig,
    duration_minutes,
    parse_marketcast,
    split_base_ref,
)
from cascade.shared.marketcast_recipe import parse_marketcast_recipe
from cascade.shared.netconfig import NetConfigError, check_overrides

# ── config ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("toml", ["chain.toml", "chain.testnet.toml"])
def test_shipped_tables_parse_inert(toml):
    cfg = load_chain_config(toml)
    mc = cfg.marketcast
    assert mc.from_block == 0 and not mc.active(10**9)
    assert mc.target_suites == ()
    assert mc.suites == ("returns", "vol") and mc.base_arch == "timestral"
    assert mc.pool_prefix == "pool_mkt" and mc.gain_margin == 0.01
    assert mc.quantile_knots == TIMESTRAL_KNOTS and len(TIMESTRAL_KNOTS) == 21
    # MarketCast-only lane C: the shipped targets are exactly its suites
    assert all(is_mkt_target(t) for t in cfg.lane_c.targets)
    assert tuple(t.split(":", 1)[1] for t in cfg.lane_c.targets) == mc.suites


def test_missing_table_is_the_default():
    assert parse_marketcast(None) == MarketCastConfig()
    assert parse_marketcast({}) == MarketCastConfig()


def test_table_local_arming_checks():
    # the shipped values are armable table-locally (pinned base + built sources)
    mc = parse_marketcast({"from_block": 9_000_000})
    assert mc.from_block == 9_000_000 and mc.active(9_000_000) and not mc.active(8_999_999)
    assert mc.target_suites == ("returns", "vol")
    with pytest.raises(ValueError, match="base_sha256 is unset"):
        parse_marketcast({"from_block": 9_000_000, "base_sha256": ""})
    with pytest.raises(ValueError, match="not pinned to a commit"):
        parse_marketcast({"from_block": 9_000_000, "base_ref": "mistralai/Timestral-28M"})
    with pytest.raises(ValueError, match="data_sources.fx is empty"):
        parse_marketcast({"from_block": 9_000_000, "data_sources": {"fx": ""}})
    # an owner choice: empty the class instead
    assert parse_marketcast({"from_block": 9_000_000, "data_sources": {"fx": ""},
                             "assets": {"crypto": ["BTC"], "fx": []}}).armed_classes == (
        "crypto",)
    with pytest.raises(ValueError, match="no adapter"):
        parse_marketcast({"from_block": 9_000_000, "data_sources": {"fx": "pyth_hermes"}})
    # inert: an empty / unbuilt source parses (it only blocks arming)
    assert parse_marketcast({"data_sources": {"fx": "pyth_hermes"}}).from_block == 0
    with pytest.raises(ValueError, match="serves"):
        parse_marketcast({"data_sources": {"crypto": "bittensor_alpha"}})
    with pytest.raises(ValueError, match="unknown source"):
        parse_marketcast({"data_sources": {"crypto": "polygon"}})


def test_cross_table_arming_checks():
    from cascade.shared.config.marketcast import check_marketcast_arming

    mc = parse_marketcast({"from_block": 30 * 7200})
    ok = dict(era_blocks=7200, v2_from_block=10 * 7200, lane_c_from_block=20 * 7200,
              validator_bench_from=10 * 7200, owner_hotkey="5Main")
    check_marketcast_arming(mc, **ok)
    check_marketcast_arming(MarketCastConfig(), **{**ok, "owner_hotkey": ""})   # off: no-op
    for bad, msg in [({"era_blocks": 7000}, "era boundary"),
                     ({"v2_from_block": 0}, "v2"), ({"v2_from_block": 31 * 7200}, "v2"),
                     ({"lane_c_from_block": 0}, "lane_c"),
                     ({"lane_c_from_block": 40 * 7200}, "lane_c"),
                     ({"validator_bench_from": 0}, "bench_source"),
                     ({"validator_bench_from": 31 * 7200}, "bench_source"),
                     ({"owner_hotkey": ""}, "owner_hotkey")]:
        with pytest.raises(ValueError, match=msg):
            check_marketcast_arming(mc, **{**ok, **bad})


def test_loader_refuses_arming_without_lane_c(tmp_path):
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "chain.toml").read_text(encoding="utf-8")
    old = "from_block         = 0                          # off; an era-boundary block arms it"
    assert old in text
    p = tmp_path / "chain.toml"
    p.write_text(text.replace(old, "from_block = 7200000"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[marketcast\] from_block=7200000"):
        load_chain_config(p)


def test_pins_found_on_the_hub():
    from cascade.shared.config.marketcast import (
        TIMESTRAL_BLOCKS,
        TIMESTRAL_REVISION,
        TIMESTRAL_SHA256,
    )

    for toml in ("chain.toml", "chain.testnet.toml"):
        mc = load_chain_config(toml).marketcast
        assert mc.base_ref == f"mistralai/Timestral-28M@{TIMESTRAL_REVISION}"
        assert mc.base_sha256 == TIMESTRAL_SHA256
        assert mc.data_sources_map["fx"] == "hyperliquid_xyz"
        assert mc.assets_map["fx"] == ("EUR", "JPY", "GBP")
    assert TIMESTRAL_BLOCKS == 12 and len(TIMESTRAL_REVISION) == 40


@pytest.mark.parametrize("raw", [
    {"from_block": -1}, {"from_block": True},
    {"base_ref": "Timestral"}, {"base_ref": "mistralai/Timestral-28M@main"},
    {"base_sha256": "abc"}, {"base_arch": "cascade_moe"},
    {"suites": []}, {"suites": ["returns", "returns"]}, {"suites": ["prices"]},
    {"suites": "returns"},
    {"assets": {"bonds": ["TLT"]}}, {"assets": {"crypto": ["BTC", "BTC"]}},
    {"horizons": {"returns": ["1x"]}}, {"horizons": {"vol": ["1h"]}},  # returns missing
    {"frequencies": {"returns": ["1h"]}},                              # 1m horizon < 1h
    {"coverage_levels": [0.995]}, {"coverage_levels": [0.0]},
    {"coverage_tolerance": 0.0}, {"gain_margin": 0.0}, {"gain_margin": 1.0},
    {"alpha": 0.5}, {"bootstrap_B": 10}, {"min_windows": 3, "min_clusters": 5},
    {"pool_prefix": "pool_c"}, {"pool_prefix": "Pool"},
    {"quantile_knots": [0.1, 0.9]}, {"quantile_knots": [0.1, 0.3, 0.2, 0.5]},
    {"quantile_knots": [0.1, 0.2, 0.9]},                               # no median
    {"data_sources": {"bonds": "x"}}, {"max_context_length": 65536},
    {"data_sources": {"crypto": "hyperliquid+hyperliquid"}}, {"eval_context_length": 8},
    {"windows_per_cell": 0}, {"verdict_wait_blocks": -1},
    {"max_synthetic_fraction": 1.5}, {"embargo_days": -1}, {"covariates": ["Bad Label"]},
])
def test_placeholder_keys_are_validated(raw):
    with pytest.raises(ValueError):
        parse_marketcast(raw)


def test_base_ref_pinning_and_durations():
    sha = "a" * 40
    assert split_base_ref(f"mistralai/Timestral-28M@{sha}") == ("mistralai/Timestral-28M", sha)
    assert split_base_ref("mistralai/Timestral-28M") == ("mistralai/Timestral-28M", "")
    assert duration_minutes("5m") == 5 and duration_minutes("4h") == 240
    assert duration_minutes("5d") == 7200
    with pytest.raises(ValueError):
        duration_minutes("0m")


def test_overlay_allows_marketcast_thresholds_but_keeps_the_model_and_data_pinned():
    assert check_overrides({"marketcast": {"gain_margin": 0.02}})
    for key, val in (("base_ref", "other@pin"), ("from_block", 7200),
                     ("data_sources", {"crypto": "other"}), ("suites", ["returns"]),
                     ("pool_prefix", "other"), ("eval_context_length", 512)):
        with pytest.raises(NetConfigError, match="release-only"):
            check_overrides({"marketcast": {key: val}})


# ── target grammar ──────────────────────────────────────────────────────────


def test_default_targets_are_the_default_suites():
    assert DEFAULT_TARGETS == ("mkt:returns", "mkt:vol")
    for t in DEFAULT_TARGETS:
        assert check_target(t, mkt_suites=("returns", "vol")) == t
    with pytest.raises(ValueError, match="MarketCast admits no targets"):
        parse_target("mkt:returns")                                   # no suites admitted
    with pytest.raises(ValueError, match="registered competition"):
        parse_target("weather:returns")
    with pytest.raises(ValueError, match="forward-only eval"):
        parse_target("horizon:720")


def test_mkt_targets_only_with_config():
    assert parse_target("mkt:returns", mkt_suites=("returns", "vol")) == ("mkt", "returns")
    assert check_target("mkt:vol", mkt_suites=("returns", "vol")) == "mkt:vol"
    with pytest.raises(ValueError, match="MarketCast suite"):
        parse_target("mkt:range", mkt_suites=("returns", "vol"))
    with pytest.raises(ValueError, match="MarketCast suite"):
        parse_lane_c({"targets": ["mkt:range"]})                      # not a configured suite
    with pytest.raises(ValueError, match="admits no targets"):
        parse_lane_c({"targets": ["mkt:returns"]}, mkt_suites=())
    lc = parse_lane_c({"targets": ["mkt:returns"]}, mkt_suites=("returns",))
    assert lc.targets == ("mkt:returns",)
    with pytest.raises(ValueError, match="public bench guards are off"):
        parse_lane_c({"targets": ["mkt:returns"], "guard_public": {"mkt:returns": ["gift"]}},
                     mkt_suites=("returns",))


def test_shipped_marketcast_cannot_admit_mkt_targets():
    """Off ⇒ the intake / commit / recipe surfaces admit no mkt: target,
    while the shipped [lane_c] targets still load (they are inactive)."""
    cfg = load_chain_config("chain.toml")
    with pytest.raises(ValueError):
        check_target("mkt:returns", mkt_suites=cfg.marketcast.target_suites)
    assert cfg.lane_c.targets == ("mkt:returns", "mkt:vol")


# ── recipe ──────────────────────────────────────────────────────────────────


def _recipe(**over):
    r = {
        "target": "mkt:returns",
        "data": {"assets": {"crypto": ["BTC", "ETH"], "fx": ["EUR"]},
                 "class_weights": {"crypto": 2.0, "fx": 1.0},
                 "frequencies": ["1m", "1h"], "start": "2024-01-01", "end": "2026-08-31"},
        "context_length": 4096,
        "horizon_mix": {"1h": 1.0, "1d": 0.5},
        "covariates": ["volume", "session"],
        "augmentations": [{"name": "jitter", "p": 0.5, "params": {"sigma": 0.01}}],
        "objective": {"suite_weights": {"returns": 1.0, "vol": 0.2},
                      "coverage_penalty": 0.5},
        "optimizer": {"lr": 1e-4, "schedule": "cosine", "warmup_fraction": 0.05,
                      "steps": 2000, "batch": 64, "weight_decay": 0.01, "clip": 1.0,
                      "ema_decay": 0.999},
        "trainable": {"scope": "lora", "lora_rank": 16},
        "synthetic_fraction": 0.0,
    }
    r.update(over)
    return r


SUB = datetime(2026, 10, 19, tzinfo=UTC).timestamp()


def test_recipe_parses():
    rc = parse_marketcast_recipe(_recipe(), MarketCastConfig(), target="mkt:returns",
                                 submission_time=SUB)
    assert rc.suite == "returns" and rc.lora_rank == 16 and rc.n_blocks == 0
    assert rc.quantile_weights == (1.0,) * 21 and rc.ema_decay == 0.999
    import json
    assert parse_marketcast_recipe(json.dumps(_recipe()), MarketCastConfig()) == \
        parse_marketcast_recipe(_recipe(), MarketCastConfig())


def _set(path, value):
    r = _recipe()
    node = r
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return r


@pytest.mark.parametrize("bad", [
    _recipe(extra=1),
    _set(("data", "bogus"), 1),
    _recipe(target="domain:energy"),
    _recipe(target="mkt:range"),                                      # not an enabled suite
    _set(("data", "assets"), {"crypto": ["PEPE"]}),
    _set(("data", "class_weights"), {"metals": 1.0}),
    _set(("data", "class_weights"), {"crypto": 0.0}),
    _set(("data", "frequencies"), ["2m"]),
    _set(("data", "end"), "2023-01-01"),
    _set(("data", "end"), "2026-10-01"),                              # inside the embargo
    _recipe(context_length=40000),
    _recipe(horizon_mix={"15m": 1.0}),                                # a vol horizon
    _recipe(covariates=["order_book"]),
    _recipe(augmentations=[{"name": "flip", "p": 0.1}]),
    _recipe(augmentations=[{"name": "jitter", "p": 1.5}]),
    _recipe(augmentations=[{"name": "jitter", "p": 0.1, "x": 1}]),
    _set(("objective", "suite_weights"), {"vol": 1.0}),
    _set(("objective", "quantile_weights"), [1.0] * 9),
    _set(("objective", "coverage_penalty"), 11.0),
    _set(("optimizer", "lr"), 0.1),
    _set(("optimizer", "schedule"), "step"),
    _set(("optimizer", "warmup_fraction"), 0.5),
    _set(("optimizer", "steps"), 0),
    _set(("optimizer", "batch"), 10_000),
    _set(("optimizer", "clip"), 0.0),
    _set(("optimizer", "ema_decay"), 0.5),
    _set(("optimizer", "momentum"), 0.9),
    _recipe(trainable={"scope": "adapter"}),
    _recipe(trainable={"scope": "lora", "lora_rank": 1000}),
    _recipe(trainable={"scope": "last_n_blocks", "n_blocks": 13}),
    _recipe(trainable={"scope": "full", "lora_rank": 4}),
    _recipe(synthetic_fraction=0.9),
    _recipe(synthetic_fraction=0.2),                                  # no generator
])
def test_recipe_bounds(bad):
    with pytest.raises(ValueError):
        parse_marketcast_recipe(bad, MarketCastConfig(), submission_time=SUB)


def test_recipe_committed_target_and_generator():
    with pytest.raises(ValueError, match="committed target"):
        parse_marketcast_recipe(_recipe(), MarketCastConfig(), target="mkt:vol")
    rc = parse_marketcast_recipe(_recipe(synthetic_fraction=0.2,
                                         trainable={"scope": "last_n_blocks", "n_blocks": 2}),
                                 MarketCastConfig(), has_generator=True)
    assert rc.synthetic_fraction == 0.2 and rc.n_blocks == 2


# ── series builders ─────────────────────────────────────────────────────────


def test_log_returns_bp():
    c = [100.0, 101.0, 99.0, 100.0, 102.0]
    np.testing.assert_allclose(M.log_returns_bp(c, 1), 1e4 * np.diff(np.log(c)))
    np.testing.assert_allclose(M.log_returns_bp(c, 2),
                               [1e4 * math.log(99 / 100), 1e4 * math.log(102 / 99)])
    with pytest.raises(ValueError):
        M.log_returns_bp([1.0, -1.0], 1)


def test_realized_vol_blocks():
    c = np.exp(np.cumsum([0.0, 0.001, -0.001, 0.002, 0.0, np.nan, 0.001]))
    c[5] = np.nan                      # bar 5 missing ⇒ returns 4→5 and 5→6 missing
    rv = M.realized_vol_bp(c, 3)
    # block 0: returns 10, -10, 20 bp; block 1: 0, NaN, NaN ⇒ < 2 obs
    assert rv.shape == (2,)
    assert rv[0] == pytest.approx(np.std([10.0, -10.0, 20.0], ddof=1), rel=1e-9)
    assert np.isnan(rv[1])


def test_range_excursions():
    o = [100.0, 101.0, 102.0, 100.0]
    h = [102.0, 103.0, 104.0, 101.0]
    lo = [99.0, 100.0, 98.0, 99.5]
    out = M.range_excursions(o, h, lo, 2)
    assert out["range"][0] == pytest.approx(math.log(103 / 99))
    assert out["up"][0] == pytest.approx(math.log(103 / 100))
    assert out["down"][0] == pytest.approx(math.log(100 / 99))
    assert out["range"][1] == pytest.approx(math.log(104 / 98))
    assert out["down"][1] == pytest.approx(math.log(102 / 98))


def test_log_volume_and_funding():
    lv = M.log_volume([1.0, 2.0, 0.0, 0.0, 3.0, np.nan, 5.0], 2)
    assert lv[0] == pytest.approx(math.log(3.0)) and np.isnan(lv[1])
    assert lv[2] == pytest.approx(math.log(3.0)) and lv.size == 3
    np.testing.assert_allclose(M.funding_change_bp([0.0001, 0.0002, 0.0004, 0.0001], 1),
                               [1.0, 2.0, -3.0])


# ── metrics ─────────────────────────────────────────────────────────────────

LV = (0.1, 0.5, 0.9)


def test_crps_quantile_hand_checked():
    # y = 1, quantiles (0, 1, 2) at (0.1, 0.5, 0.9):
    # pinball = 0.1*1, 0, (0.9-1)*(1-2)=0.1 ⇒ mean 0.0667 ⇒ CRPS = 0.1333
    assert M.crps_quantile([0.0, 1.0, 2.0], 1.0, LV) == pytest.approx(2 * 0.2 / 3)
    # y = 3: 0.1*3 + 0.5*2 + 0.9*1 = 2.2 ⇒ 2 * 2.2/3
    assert M.crps_quantile([0.0, 1.0, 2.0], 3.0, LV) == pytest.approx(2 * 2.2 / 3)
    w = M.window_crps(np.array([[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]),
                      np.array([[1.0, np.nan]]), LV)
    assert w[0] == pytest.approx(2 * 0.2 / 3)


def test_qlike():
    assert M.qlike(2.0, 2.0) == pytest.approx(0.0)
    assert M.qlike(2.0, 1.0) == pytest.approx(4.0 - math.log(4.0) - 1.0)
    assert np.isnan(M.qlike(0.0, 1.0))


def test_interval_coverage_and_gate():
    q = np.array([[-1.0, 0.0, 1.0]] * 4)     # 80% interval [-1, 1]
    y = np.array([0.5, -0.5, 2.0, 0.0])
    cov = M.interval_coverage(q, y, LV, (0.8,))
    assert cov == {0.8: 0.75}
    assert M.coverage_gate(cov, tolerance=0.1).passed          # 0.75 >= 0.8 - 0.1
    g = M.coverage_gate(cov, tolerance=0.03)
    assert not g.passed and g.rows[0].under
    g = M.coverage_gate({0.5: 0.5, 0.8: 0.95}, tolerance=0.1)
    assert g.passed and g.rows[1].over and not g.rows[1].under
    assert not M.coverage_gate({0.5: float("nan")}, tolerance=0.1).passed


def test_mase_and_rank_diagnostics():
    assert M.mase_median([1.0, 2.0], [2.0, 2.0], [0.0, 1.0, 2.0]) == pytest.approx(0.5)
    assert M.rank_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert M.rank_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert M.rank_auc([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert M.rank_auc([4, 3, 2, 1], [1, 2, 3, 4]) == pytest.approx(0.0)


def test_knot_mapping_round_trip():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(size=(5, 21)), axis=-1)
    sub = M.to_repo_levels(q)
    np.testing.assert_allclose(sub, q[:, [2, 4, 6, 8, 10, 12, 14, 16, 18]])
    fine = tuple(sorted(set(TIMESTRAL_KNOTS) | {0.025, 0.125, 0.975}))
    back = M.map_quantiles(M.map_quantiles(q, TIMESTRAL_KNOTS, fine), fine, TIMESTRAL_KNOTS)
    np.testing.assert_allclose(back, q)
    # clamped outside the source range, crossing knots sorted first
    out = M.map_quantiles([[2.0, 1.0, 3.0]], LV, (0.05, 0.5, 0.95))
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]])


def test_forward_eligibility():
    assert M.window_eligible(100.0, 101.0, 200.0)
    assert not M.window_eligible(100.0, 100.0, 200.0)          # start == submission
    assert not M.window_eligible(100.0, 150.0, 200.0, now=180.0)
    assert M.window_eligible(100.0, 150.0, 200.0, now=200.0)
    np.testing.assert_array_equal(
        M.eligible_mask(100.0, [50, 101, 150], [60, 110, 300], now=200.0), [False, True, False])
    with pytest.raises(ValueError):
        M.window_eligible(0.0, 10.0, 5.0)


# ── paired verdict ──────────────────────────────────────────────────────────

KW = dict(alpha=0.05, B=2000, gain_margin=0.01, seed="t", min_windows=50, min_clusters=5)


def _losses(n=200, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0.0, 0.3, n)
    return base, [f"a{i % 10}" for i in range(n)]


def test_verdict_clear_win_and_identical_never_passes():
    base, cl = _losses()
    good = M.coverage_gate({0.5: 0.5, 0.8: 0.8, 0.98: 0.97}, tolerance=0.05)
    v = M.judge_suite("range", base, base * 0.8, clusters=cl, **KW)
    assert v.passed and v.gain == pytest.approx(0.2) and v.gain_lcb >= 0.01
    same = M.judge_suite("range", base, base.copy(), clusters=cl, **KW)
    assert not same.passed and same.gain_lcb == pytest.approx(0.0)
    same_r = M.judge_suite("returns", base, base.copy(), clusters=cl, coverage=good, **KW)
    assert not same_r.passed
    with pytest.raises(ValueError):
        M.judge_suite("range", base, base, **{**KW, "gain_margin": 0.0})


def test_verdict_coverage_gate_on_returns():
    base, cl = _losses()
    good = M.coverage_gate({0.5: 0.5, 0.8: 0.8, 0.98: 0.97}, tolerance=0.05)
    bad = M.coverage_gate({0.5: 0.5, 0.8: 0.6, 0.98: 0.97}, tolerance=0.05)
    assert M.judge_suite("returns", base, base * 0.8, clusters=cl, coverage=good, **KW).passed
    v = M.judge_suite("returns", base, base * 0.8, clusters=cl, coverage=bad, **KW)
    assert not v.passed and "coverage" in v.reason
    v = M.judge_suite("returns", base, base * 0.8, clusters=cl, **KW)
    assert not v.passed and v.reason == "coverage gate missing"
    # calibration is not a gate on the suites outside COVERAGE_GATE_SUITES
    assert M.judge_suite("range", base, base * 0.8, clusters=cl, coverage=bad, **KW).passed
    assert M.COVERAGE_GATE_SUITES == ("returns", "vol")


def test_verdict_inconclusive():
    base, _ = _losses(n=40)
    assert "windows" in M.judge_suite("range", base, base * 0.5, **KW).reason
    base, _ = _losses()
    v = M.judge_suite("range", base, base * 0.5, clusters=["a"] * 200, **KW)
    assert not v.passed and "clusters" in v.reason


# ── adapter ─────────────────────────────────────────────────────────────────


class _Stub:
    quantile_levels = TIMESTRAL_KNOTS

    def __init__(self):
        self.seen = []

    def forecast_quantiles(self, context, horizon, *, past_covariates=None,
                           future_covariates=None):
        self.seen.append((context.size, None if future_covariates is None
                          else np.asarray(future_covariates).shape[-1]))
        last = float(context[-1])
        return np.tile(last + np.asarray(TIMESTRAL_KNOTS) - 0.5, (horizon, 1))


def test_adapter_stub_forecaster():
    stub = _Stub()
    assert isinstance(stub, TA.QuantileForecaster)
    out = TA.forecast_windows(stub, [np.arange(10.0), np.arange(5.0)], 3,
                              levels=(0.1, 0.5, 0.9), max_context=8,
                              future_covariates=[np.zeros((1, 13)), None])
    assert out.shape == (2, 3, 3)
    np.testing.assert_allclose(out[0, 0], [9.0 - 0.4, 9.0, 9.0 + 0.4])
    assert stub.seen == [(8, 11), (5, None)]
    with pytest.raises(ValueError):
        TA.forecast_windows(stub, [np.arange(5.0)], 2048)


def test_adapter_missing_library(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "timestral":
            raise ImportError("No module named 'timestral'")
        return real(name, *a, **k)

    monkeypatch.setattr(TA.importlib, "import_module", fake)
    with pytest.raises(TA.TimestralUnavailable, match="not installed"):
        TA.load_timestral()


def test_sha256_check(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"timestral")
    d = hashlib.sha256(b"timestral").hexdigest()
    assert TA.sha256_file(p) == d
    assert TA.verify_checkpoint(p, d.upper()) == d
    with pytest.raises(ValueError, match="!="):
        TA.verify_checkpoint(p, "0" * 64)
    with pytest.raises(ValueError, match="not pinned"):
        TA.check_base_sha256(d, "")


def test_config_dataclass_is_hashable_and_frozen():
    mc = MarketCastConfig()
    hash(mc)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mc.from_block = 1  # type: ignore[misc]
    assert replace(mc, from_block=5).target_suites == ("returns", "vol")


# ── intake + miner CLI pass-through ─────────────────────────────────────────


def test_intake_mkt_targets_inactive_then_active(tmp_path):
    from cascade.funding.intake import LANE_C_TARGET_HEADER, FundingIntake
    from cascade.funding.queue import FundedQueue, LaneCQueue, lane_c_queue_path, split_lane_c_ref
    from cascade.funding.vault import PayerKeyVault

    hk, ref = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "org/r@sha256:" + "a" * 64

    def _intake(mkt):
        lc = LaneCQueue(lane_c_queue_path(tmp_path / f"{mkt}q.json"))
        return FundingIntake(FundedQueue(tmp_path / f"{mkt}q.json"), PayerKeyVault(dir=None),
                             resolve_reveal=lambda h, r: None, require_signature=False,
                             lane_c_queue=lc, resolve_lane_c_reveal=lambda h, t, r: 7,
                             lane_c_targets=("mkt:vol", "mkt:returns"),
                             mkt_suites=mkt), lc

    h = {"X-Miner-Hotkey": hk, "X-Commit-Ref": ref, LANE_C_TARGET_HEADER: "mkt:returns",
         "X-Lium-Api-Key": "sk"}
    off, _ = _intake(())
    status, body = off.lane_c_fund(h)
    assert (status, body["code"]) == (400, "marketcast_inactive")
    on, q = _intake(("returns", "vol"))
    assert on.lane_c_fund(h)[0] == 202
    assert split_lane_c_ref(q.get(hk).ref) == ("mkt:returns", ref)
    assert on.lane_c_fund({**h, LANE_C_TARGET_HEADER: "mkt:range"})[1]["code"] == "bad_target"


def test_reveal_resolver_reads_mkt_commitments():
    from cascade.funding.main import ChainRevealResolver
    from cascade.shared.lane_c_manifest import format_finetune_commit

    ref = "org/r@sha256:" + "a" * 64
    payload = format_finetune_commit("mkt:returns", ref, mkt_suites=("returns",))
    client = SimpleNamespace(poll_commitments=lambda include_history=True: [
        SimpleNamespace(hotkey="hk", payload=payload, commit_block=11, uid=1)])
    r = ChainRevealResolver(client)
    assert r.lane_c("hk", "mkt:returns", ref) is None              # inert
    r2 = ChainRevealResolver(client)
    r2.mkt_suites = ("returns",)
    assert r2.lane_c("hk", "mkt:returns", ref) == 11


def test_miner_cli_refuses_mkt_while_inactive(tmp_path):
    import json as _json

    from cascade.miner.lane_c import mkt_inactive, recipe_error

    cfg = load_chain_config("chain.toml")
    assert "not active" in mkt_inactive(cfg, "mkt:returns")
    assert mkt_inactive(cfg, "domain:energy") is None
    from cascade.shared.config.lane_c import parse_lane_c as _plc

    armed = replace(cfg, marketcast=replace(cfg.marketcast, from_block=7200),
                    lane_c=_plc({"targets": ["mkt:returns"]}, mkt_suites=("returns", "vol")))
    d = tmp_path / "tree"
    d.mkdir()
    assert "marketcast.json" in recipe_error(armed, d, "mkt:returns")
    (d / "marketcast.json").write_text(_json.dumps(_recipe(
        data={"assets": {"crypto": ["BTC"]}, "class_weights": {"crypto": 1.0},
              "frequencies": ["1m"], "start": "2024-01-01", "end": "2024-06-01"})))
    assert recipe_error(armed, d, "mkt:returns") is None
    assert "not active" in recipe_error(replace(cfg, lane_c=armed.lane_c), d, "mkt:returns")
