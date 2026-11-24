"""Shared helpers for the W-LAUNCH acceptance tests (Wed 2026-11-25 launch)."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Owner 2026-11-24: lane B / C (MarketCast) / D start together at the first
# mainnet era boundary after Wed 2026-11-25 18:00 UTC.
MAINNET_LAUNCH = 9586800
# The first testnet era boundary (600 blocks) after Wed 18:00 UTC, from head
# 8511976 @ 2026-11-24 02:06:48 UTC at the measured 12.0 s/block.
TESTNET_LAUNCH = 8524200

# (base chain.toml, step-1 overlay, step-2 overlay, launch block)
PAIRS = {
    "mainnet": ("chain.toml", "deploy/overlays/launch-20261125.pre.mainnet.toml",
                "deploy/overlays/launch-20261125.mainnet.toml", MAINNET_LAUNCH),
    "testnet": ("chain.testnet.toml", "deploy/overlays/launch-20261125.pre.testnet.toml",
                "deploy/overlays/launch-20261125.testnet.toml", TESTNET_LAUNCH),
}


def overrides(rel: str) -> dict:
    p = ROOT / rel
    assert p.is_file(), f"missing launch overlay {rel}"
    return tomllib.loads(p.read_text())


def shipped_cfg(name: str):
    from cascade.shared.config import load_chain_config

    return load_chain_config(ROOT / name)


def bodies(net: str):
    """(base cfg, step-1 body, step-2 body, launch block) — the two bodies as
    ``next_overlay`` chains them (step 2's previous is step 1)."""
    from cascade.shared import netconfig as nc

    base_name, pre, full, launch = PAIRS[net]
    base = shipped_cfg(base_name)
    ident = nc.network_identity(base)
    b1 = nc.next_overlay(overrides(pre), launch, None, network=ident)
    b2 = nc.next_overlay(overrides(full), launch, b1, network=ident)
    return base, b1, b2, launch
