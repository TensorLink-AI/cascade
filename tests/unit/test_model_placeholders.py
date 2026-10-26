"""Reserved variants cannot become an executable training contract by accident."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from cascade.shared.config import parse_chain_config
from cascade.shared.config.training import parse_training_contract
from cascade.shared.netconfig import effective_raw

PRESETS = Path(__file__).resolve().parents[2] / "deploy" / "presets" / "pending"


@pytest.mark.parametrize("size", [56])
@pytest.mark.parametrize("location", ["primary", "v2", "additional"])
def test_reserved_variant_is_rejected_even_over_a_complete_contract(cfg, size, location):
    with (PRESETS / f"cascade-moe-{size}m.toml").open("rb") as stream:
        placeholder = tomllib.load(stream)["training"]
    assert placeholder["arch_preset"] == f"cascade-moe-v2-{size}m"
    assert placeholder["target_active_parameters"] == size * 1_000_000
    raw = effective_raw(cfg)
    if location == "primary":
        raw["training"].update(placeholder)
    elif location == "v2":
        raw["v2"]["training"].update(placeholder)
    else:
        raw["training"]["sizes"] = [{**raw["training"], **placeholder}]
    with pytest.raises(ValueError, match="model placeholder"):
        parse_chain_config(raw)


@pytest.mark.parametrize("metadata", ["implementation_status", "target_active_parameters"])
def test_either_placeholder_marker_blocks_an_incomplete_contract(metadata):
    with pytest.raises(ValueError, match="model placeholder"):
        parse_training_contract({"arch_preset": "unpopulated", metadata: "placeholder"})


@pytest.mark.parametrize("size", [2, 7, 28, 56, 120])
def test_reserved_name_cannot_relabel_the_old_model(cfg, size):
    raw = effective_raw(cfg)
    raw["v2"]["training"]["base_arch"] = "cascade_moe"
    raw["v2"]["training"]["training_steps"] = 0
    raw["v2"]["training"]["arch_preset"] = f"cascade-moe-v2-{size}m"
    with pytest.raises(ValueError, match="model placeholder|require.*cascade_pyramid"):
        parse_chain_config(raw)
