"""Owner 2026-11-16: MarketCast recipes choose their own leg wall (``leg_hours``,
up to ``[marketcast] max_leg_hours``) and their data mix — pure synthetic
(``synthetic_fraction`` 1, no real assets), pure real (0) or anything between."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from cascade.pool import marketcast as P
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import MAX_LEG_HOURS_CEILING, parse_marketcast
from cascade.shared.marketcast_recipe import parse_marketcast_recipe
from cascade.trainer.marketcast_runner import MktLegRecipe, mkt_leg_contract, mkt_leg_hours

ROOT = Path(__file__).resolve().parents[2]
T0 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
MC = parse_marketcast({
    "suites": ["returns", "vol"], "assets": {"crypto": ["BTC", "ETH"]},
    "frequencies": {"returns": ["1m", "5m"], "vol": ["5m"]},
    "horizons": {"returns": ["1m", "5m"], "vol": ["5m"]}, "max_steps": 1000,
})


def _recipe(**over):
    r = {"target": "mkt:returns",
         "data": {"assets": {"crypto": ["BTC", "ETH"]}, "class_weights": {"crypto": 1.0},
                  "frequencies": ["1m", "5m"], "start": "2025-01-01", "end": "2025-01-02"},
         "context_length": 128, "horizon_mix": {"1m": 1.0, "5m": 1.0},
         "covariates": [], "augmentations": [],
         "objective": {"suite_weights": {"returns": 1.0, "vol": 0.5}},
         "optimizer": {"lr": 1e-3, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 6,
                       "batch": 4, "weight_decay": 0.0, "clip": 1.0},
         "trainable": {"scope": "lora", "lora_rank": 2}, "synthetic_fraction": 0.0}
    r.update(over)
    return r


def _pure(**over):
    return _recipe(**{"data": {"assets": {}, "class_weights": {}, "frequencies": ["1m", "5m"],
                               "start": "2025-01-01", "end": "2025-01-02"},
                      "synthetic_fraction": 1.0, **over})


# ── config ───────────────────────────────────────────────────────────────────


def test_config_defaults_and_bounds():
    mc = parse_marketcast({})
    assert mc.max_leg_hours == 12.0 and mc.max_synthetic_fraction == 1.0
    assert parse_marketcast({"max_leg_hours": 48}).max_leg_hours == 48.0
    for bad in (0, -1, MAX_LEG_HOURS_CEILING + 1):
        with pytest.raises(ValueError, match="max_leg_hours"):
            parse_marketcast({"max_leg_hours": bad})


@pytest.mark.parametrize("toml", ["chain.toml", "chain.testnet.toml"])
def test_shipped_tomls_carry_the_flexible_recipe_bounds(toml):
    mc = load_chain_config(ROOT / toml).marketcast
    # TESTNET-FAST (owner 2026-11-24): a recipe leg is capped at 1 h on testnet
    want_hours = 1.0 if toml == "chain.testnet.toml" else 12.0
    assert mc.max_leg_hours == want_hours and mc.max_synthetic_fraction == 1.0


# ── recipe: leg_hours ────────────────────────────────────────────────────────


def test_leg_hours_is_optional_and_bounded():
    assert parse_marketcast_recipe(_recipe(), MC).leg_hours == 0.0          # absent
    assert parse_marketcast_recipe(_recipe(leg_hours=12), MC).leg_hours == 12.0
    assert parse_marketcast_recipe(_recipe(leg_hours=0.25), MC).leg_hours == 0.25
    for bad in (0, -1, 12.5, "6"):
        with pytest.raises(ValueError):
            parse_marketcast_recipe(_recipe(leg_hours=bad), MC)


def test_leg_hours_resolves_to_the_recipe_else_lane_c_and_is_capped():
    cfg = load_chain_config(ROOT / "chain.toml")
    assert mkt_leg_hours(cfg) == cfg.lane_c.leg_hours
    assert mkt_leg_hours(cfg, 7.5) == 7.5
    assert mkt_leg_hours(cfg, 99.0) == cfg.marketcast.max_leg_hours   # this release's cap
    r = MktLegRecipe("mkt:market", '{"leg_hours": 9}', False)
    assert r.leg_hours == 9.0
    assert MktLegRecipe("mkt:market", "{}", False).leg_hours == 0.0


def test_a_recipe_leg_may_outrun_the_throne_wall():
    """A 12 h leg is not clamped to the throne's 5 h max_train_seconds (nor,
    on testnet, a 1 h leg to the 300 s fast-round throne)."""
    for toml, hours in (("chain.toml", 12.0), ("chain.testnet.toml", 1.0)):
        cfg = load_chain_config(ROOT / toml)
        base = cfg.throne_contracts()[0]
        leg = mkt_leg_contract(cfg, base, hours)
        gf = float(cfg.round.heat_guard_factor or 1.0)
        want = max(int(round(gf * hours * 3600)), int(cfg.round.heat_guard_floor_seconds or 0))
        assert leg.max_train_seconds == want > base.max_train_seconds
        assert leg.target_train_hours == hours and leg.corpus_mode == "stream_cpu"
    cfg = load_chain_config(ROOT / "chain.toml")
    base = cfg.throne_contracts()[0]
    assert (mkt_leg_contract(cfg, base).max_train_seconds
            == mkt_leg_contract(cfg, base, cfg.lane_c.leg_hours).max_train_seconds)


# ── recipe: data mix ─────────────────────────────────────────────────────────


def test_pure_synthetic_may_name_no_assets():
    r = parse_marketcast_recipe(_pure(), MC, has_generator=True)
    assert r.assets == () and r.class_weights == () and r.synthetic_fraction == 1.0
    with pytest.raises(ValueError, match="generator"):
        parse_marketcast_recipe(_pure(), MC)                        # still needs a generator
    nearly = _pure(synthetic_fraction=0.99)
    with pytest.raises(ValueError, match="empty only with synthetic_fraction 1"):
        parse_marketcast_recipe(nearly, MC, has_generator=True)
    # a pure-synthetic recipe may still name assets (they are just never sampled)
    assert parse_marketcast_recipe(_recipe(synthetic_fraction=1.0), MC,
                                   has_generator=True).assets
    # real-only stays the default shape
    assert parse_marketcast_recipe(_recipe(), MC).synthetic_fraction == 0.0


def test_capped_synthetic_fraction_still_refused():
    mc = parse_marketcast({"suites": ["returns"], "assets": {"crypto": ["BTC", "ETH"]},
                           "max_synthetic_fraction": 0.5})
    with pytest.raises(ValueError, match="synthetic_fraction"):
        parse_marketcast_recipe(_pure(), mc, has_generator=True)


# ── trainer: a pure-synthetic leg samples with no real series ────────────────


def test_pure_synthetic_sampler_trains_on_an_empty_snapshot(tmp_path):
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    P.write_snapshot(tmp_path / "t", t0=T0, minutes=60, bars={}, sources={},
                     anchor_block=0, anchor_time=float(T0))
    snap = P.load_snapshot(tmp_path / "t", verify_signature=False)
    r = parse_marketcast_recipe(_pure(), MC, has_generator=True)
    gen = iter([np.cumsum(np.ones(600))[None] for _ in range(1000)])
    s = L.MktSampler(r, MC, snap, seed=3, synthetic=gen)
    assert set(s.suites) == {"returns", "vol"} and not s.series
    b = s.batch(8)
    assert b.synthetic == 8 and b.contexts.shape[0] == 8
    with pytest.raises(L.MktLegError):              # pure synthetic with no generator series
        L.MktSampler(r, MC, snap, seed=3)
    # real-only on the same empty snapshot has nothing to train on
    with pytest.raises(L.MktLegError):
        L.MktSampler(parse_marketcast_recipe(_recipe(), MC), MC, snap, seed=3)


def test_a_horizon_with_no_real_series_is_skipped_not_a_crash(tmp_path):
    """Live 2026-11-17 (RTX 3060 leg): a composite recipe naming 15m over an
    embargoed range got 'could not cut a vol/15m training sample' — the
    training snapshot holds 1m bars only, so 5m/15m series never exist."""
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    n = 600
    bars = {("crypto", a): {"open": np.linspace(100, 110, n), "high": np.linspace(101, 111, n),
                            "low": np.linspace(99, 109, n), "close": np.linspace(100, 110, n),
                            "volume": np.ones(n)} for a in ("BTC", "ETH")}
    P.write_snapshot(tmp_path / "t", t0=T0, minutes=n, bars=bars, sources={},
                     anchor_block=0, anchor_time=float(T0))
    snap = P.load_snapshot(tmp_path / "t", verify_signature=False)
    r = parse_marketcast_recipe(_recipe(), MC)
    s = L.MktSampler(r, MC, snap, seed=3)
    real_f = {(k[0], k[3]) for k in s.series}
    assert s.geom and all((sh[0], g[0]) in real_f for sh, g in s.geom.items())
    for _ in range(50):
        s.batch(4)                                   # never raises
