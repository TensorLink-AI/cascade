"""R7t: the testnet overlay identity (owner 2026-11-24). Contract: README.md."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pre.testnet.toml"
DEPLOY_PRE = ROOT / "deploy" / "overlays" / "launch-20261125.pre.testnet.toml"

VALIDATOR0_HOTKEY = "5F1VmMSaPPi2qm91HE8VyMiHoCz4YxGaexJ3qmbyAPBUcUtR"
OWNER_HOTKEY = "5Cyver53HZaxdn6j9K2W9LyaoGxjz1W3cXHoKcaDyzfjfpHs"
MAINNET_MAIN_VALIDATOR = "5FcCsoB2R5M55RdrNrvPQv7xmLJRFzs7R7C73UUSxgUCyfsw"
# The first testnet era boundary (600 blocks) after Wed 2026-11-25 18:00 UTC (#870).
TESTNET_LAUNCH = 8524200


def _testnet():
    from cascade.shared.config import load_chain_config

    return load_chain_config(ROOT / "chain.testnet.toml")


def _pre_overrides() -> dict:
    src = DEPLOY_PRE if DEPLOY_PRE.is_file() else FIXTURE
    return tomllib.loads(src.read_text())


def _plan():
    from cascade.shared import netconfig as nc

    cfg = _testnet()
    return cfg, nc.plan_overlay(cfg, _pre_overrides(), TESTNET_LAUNCH, None)


def test_testnet_owner_hotkey_is_the_main_validator():
    cfg = _testnet()
    assert cfg.subnet.netuid == 259
    assert cfg.netconfig.network_id == "bittensor:test"
    assert cfg.netconfig.owner_hotkey == VALIDATOR0_HOTKEY
    assert cfg.netconfig.enabled
    assert cfg.netconfig.owner_hotkey == cfg.manifest.validator_hotkey
    assert cfg.netconfig.owner_hotkey != cfg.manifest.trainer_hotkey
    assert cfg.manifest.trainer_hotkey == OWNER_HOTKEY
    # mainnet chain.toml keeps its own main validator, untouched by R7t
    from cascade.shared.config import load_chain_config

    main = load_chain_config(ROOT / "chain.toml")
    assert main.netconfig.owner_hotkey == MAINNET_MAIN_VALIDATOR
    assert main.netconfig.owner_hotkey != cfg.netconfig.owner_hotkey


def test_pre_testnet_overlay_plans_valid():
    from cascade.shared import netconfig as nc

    # the mirrored fixture is #870's step-1 body (identical once it lands)
    fixture = tomllib.loads(FIXTURE.read_text())
    assert fixture == {"lane_b": {"share": 0.35},
                       "lane_c": {"share": 0.15, "takes_from": "lane_a"}}
    if DEPLOY_PRE.is_file():
        assert tomllib.loads(DEPLOY_PRE.read_text()) == fixture
    cfg, plan = _plan()
    assert plan.ok, plan.errors
    assert plan.errors == []
    assert plan.body["seq"] == 1 and plan.body["prev"] is None
    assert plan.body["effective_block"] == TESTNET_LAUNCH
    assert plan.body["network"] == {"chain": "bittensor:test", "netuid": 259}
    nc.check_body(plan.body)
    built = nc.validate_overlay(cfg, plan.body)
    assert built.netconfig.owner_hotkey == VALIDATOR0_HOTKEY


def test_overlay_verifies_only_against_the_configured_key():
    from bittensor.sp_core import Keypair

    from cascade.shared import netconfig as nc

    cfg, plan = _plan()
    owner = cfg.netconfig.owner_hotkey
    assert owner == VALIDATOR0_HOTKEY
    dev = Keypair.create_from_uri("//Alice")
    env = nc.sign_overlay(plan.body, dev)
    # the dev key's own signature is sound, so a refusal below is the identity
    assert nc.verify_overlay(env, owner=dev.ss58_address) == plan.body
    with pytest.raises(nc.NetConfigError):
        nc.verify_overlay(env, owner=owner)
    # an envelope claiming the subnet-owner (trainer) hotkey is not the signer
    forged = dict(env, signer=OWNER_HOTKEY)
    with pytest.raises(nc.NetConfigError):
        nc.verify_overlay(forged, owner=owner)
    forged = dict(env, signer=owner)
    with pytest.raises(nc.NetConfigError):
        nc.verify_overlay(forged, owner=owner)
