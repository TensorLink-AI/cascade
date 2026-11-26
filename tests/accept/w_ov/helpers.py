"""Shared helpers for the W-OV acceptance tests (overlay envelope, rev6.24)."""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ("chain.toml", "chain.testnet.toml")
NETWORK = {"netuid": 91, "chain": "bittensor:finney"}


def shipped_raw(name: str) -> dict:
    return tomllib.loads((ROOT / name).read_text())


def shipped_cfg(name: str = "chain.toml"):
    from cascade.shared.config import load_chain_config

    return load_chain_config(ROOT / name)


def refusal(base, overrides: dict, eff: int | None = None) -> str | None:
    """The NetConfigError text ``build_config`` raises for ``overrides`` on
    ``base`` (None = accepted)."""
    from cascade.shared import netconfig as nc

    try:
        nc.check_overrides(overrides)
        body = nc.next_overlay(overrides, int(eff or 10**9), None,
                               network=nc.network_identity(base))
        nc.build_config(base, body)
    except nc.NetConfigError as e:
        return str(e)
    return None


def fixture_raw(tmp_path) -> dict:
    """The unit suite's ``cfg`` fixture table (tests/conftest.py), rebuilt
    here so the test also runs with ``--noconftest``."""
    from cascade.shared.config import load_chain_config
    from tests.unit.predecision_main import predecision_path

    c = load_chain_config(predecision_path(tmp_path))
    raw = copy.deepcopy(c.raw)
    raw["v2"].update(launch_at_utc="", commit_open_at_utc="", commit_from_block=0,
                     image_decoupled_from_block=0)
    raw["netconfig"]["owner_hotkey"] = ""
    raw["scoring"].update(forfeit_from_block=0, margin_v2_from_block=0)
    return json.loads(json.dumps(raw))
