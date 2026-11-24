"""Shared helpers for the W-CMP acceptance tests (DEC-CA-0075)."""

from __future__ import annotations

import copy
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ("chain.toml", "chain.testnet.toml")
T0 = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())


def shipped_raw(name: str = "chain.toml") -> dict:
    return tomllib.loads((ROOT / name).read_text())


def parse(raw: dict):
    from cascade.shared.config import parse_chain_config

    return parse_chain_config(copy.deepcopy(raw))


def with_specs(raw: dict, specs: dict) -> dict:
    out = copy.deepcopy(raw)
    out.setdefault("lane_c", {})["specs"] = copy.deepcopy(specs)
    return out


def overlay(base, overrides: dict, eff: int = 10**9):
    """``(config, None)`` the overlay builds on ``base``, or ``(None, why)``."""
    from cascade.shared import netconfig as nc

    try:
        body = nc.next_overlay(overrides, eff, None, network=nc.network_identity(base))
        return nc.build_config(base, body), None
    except nc.NetConfigError as e:
        return None, str(e)


def bars(minutes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, minutes)))
    return {"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c,
            "volume": rng.uniform(1, 2, minutes), "funding": np.full(minutes, 1e-5)}


def mkt_snapshot(path: Path, minutes: int = 600) -> dict:
    """A small legacy MarketCast snapshot (two crypto assets, one fx)."""
    from cascade.pool import marketcast as P

    b = {("crypto", "BTC"): bars(minutes, 1), ("crypto", "ETH"): bars(minutes, 2),
         ("fx", "EUR"): bars(minutes, 3)}
    return P.write_snapshot(path, t0=T0, minutes=minutes, bars=b,
                            sources={"crypto": "hyperliquid+binance_spot",
                                     "fx": "hyperliquid_xyz"},
                            anchor_block=1000, anchor_time=float(T0))


def small_mc(**over):
    from cascade.shared.config.marketcast import parse_marketcast

    raw = {"suites": ["returns", "vol"], "assets": {"crypto": ["BTC", "ETH"], "fx": ["EUR"]},
           "min_dollar_volume_24h": 0.0,
           "frequencies": {"returns": ["1m", "5m"], "vol": ["5m"]},
           "horizons": {"returns": ["5m"], "vol": ["5m"]},
           "eval_context_length": 64, "windows_per_cell": 10}
    raw.update(over)
    return parse_marketcast(raw)
