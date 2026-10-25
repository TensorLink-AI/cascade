"""[round] funded_* knobs: validators and loader round-trips."""

from __future__ import annotations

import pytest

from cascade.shared.config import RoundConfig, validate_funded_mode


def test_round_config_defaults_are_inert():
    assert RoundConfig().funded_mode == "off"


def test_validate_funded_mode():
    assert validate_funded_mode("shadow") == "shadow"
    with pytest.raises(ValueError):
        validate_funded_mode("on")


def test_funded_config_validators_fail_loud():
    import pytest

    from cascade.shared.config import validate_funded_field_cap, validate_funded_pod_skus

    assert validate_funded_field_cap(0) == 0
    assert validate_funded_field_cap("8") == 8
    with pytest.raises(ValueError):
        validate_funded_field_cap(-1)            # truthy → would seat NOBODY
    assert validate_funded_pod_skus(["RTX4090", "A6000"]) == ("RTX4090", "A6000")
    with pytest.raises(ValueError):
        validate_funded_pod_skus("RTX4090")      # would iterate as characters


def test_funded_cpu_blocklist_parses_from_toml(tmp_path):
    """Knob round-trip: the dataclass field alone would make TOML arming a
    silent no-op (config knobs need loader parsing)."""
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    repo_toml = Path(__file__).resolve().parents[2] / "chain.toml"
    assert load_chain_config(repo_toml).round.funded_cpu_blocklist == ("Xeon(R) CPU E5-2",)
    src = repo_toml.read_text(encoding="utf-8")
    edited = src.replace('funded_cpu_blocklist = ["Xeon(R) CPU E5-2"]',
                         'funded_cpu_blocklist = [" e5-2673 ", "", "Gold 61"]')
    assert edited != src
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    assert load_chain_config(tmp_path / "chain.toml").round.funded_cpu_blocklist == ("e5-2673", "Gold 61")


def test_funded_host_bench_floor_parses_from_toml(tmp_path):
    """Knob round-trip (config knobs need loader parsing): the repo ships the
    RTX4090 floor ARMED at 500M tokens/s (2026-09-16, owner: "add the guard
    for 4090") and nothing else; an edited table lands as sorted (sku,
    tokens/s) pairs and resolves case-insensitively per SKU, 0.0 for a SKU
    without one."""
    from pathlib import Path

    import pytest

    from cascade.shared.config import (
        funded_host_bench_floor_for,
        load_chain_config,
        validate_funded_host_bench_floor,
    )

    repo_toml = Path(__file__).resolve().parents[2] / "chain.toml"
    shipped = load_chain_config(repo_toml).round.funded_host_bench_floor
    assert shipped == (("RTX4090", 500e6),)
    assert funded_host_bench_floor_for(shipped, "rtx4090") == 500e6
    assert funded_host_bench_floor_for(shipped, "A100") == 0.0
    src = repo_toml.read_text(encoding="utf-8")
    edited = src.replace("funded_host_bench_floor = { RTX4090 = 500e6 }",
                         "funded_host_bench_floor = { RTX4090 = 450e6, A6000 = 200_000_000 }")
    assert edited != src
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    floors = load_chain_config(tmp_path / "chain.toml").round.funded_host_bench_floor
    assert floors == (("A6000", 200_000_000.0), ("RTX4090", 450e6))
    assert funded_host_bench_floor_for(floors, "rtx4090") == 450e6
    assert funded_host_bench_floor_for(floors, "L40S") == 0.0
    assert funded_host_bench_floor_for((), "RTX4090") == 0.0
    # Fail-loud on shapes that would otherwise arm nothing.
    with pytest.raises(ValueError):
        validate_funded_host_bench_floor(450e6)
    with pytest.raises(ValueError):
        validate_funded_host_bench_floor({"RTX4090": "450e6"})
    with pytest.raises(ValueError):
        validate_funded_host_bench_floor({"RTX4090": -1})
