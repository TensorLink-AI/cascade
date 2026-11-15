"""2026-11-15: a cgen VM offer that never boots must stay quarantined across
trainer restarts (strikes lived in memory, so every restart re-rented the dead
latitude/dallas-usa-1/cpu_32 offer at "strike 1"), and an operator must be
able to exclude a known-dead offer outright."""

from __future__ import annotations

import json
import time

import pytest

from cascade.generation.stage import OfferQuarantine
from cascade.provision.cpu import CpuShape
from cascade.shared.config import load_chain_config
from cascade.shared.config.generation import parse_generation


def _dallas():
    return CpuShape(vcpus=32, memory_gb=128, price_per_hour=1.02, provider="shadeform",
                    type_id="cpu_32", region="dallas-usa-1", cloud="latitude")


def _ashburn():
    return CpuShape(vcpus=32, memory_gb=128, price_per_hour=1.02, provider="shadeform",
                    type_id="cpu_32", region="ashburn-usa-1", cloud="latitude")


def test_strikes_and_deadline_survive_a_restart(tmp_path):
    path = tmp_path / "generation_offer_quarantine.json"
    q = OfferQuarantine(cooldown_s=600, path=path)
    assert q.strike("shadeform", _dallas()) == (1, 600)
    assert q.blocked("shadeform", _dallas()) and not q.blocked("shadeform", _ashburn())
    doc = json.loads(path.read_text())
    (row,) = doc["offers"]
    assert row["key"] == ["shadeform", "latitude", "dallas-usa-1", "cpu_32"] and row["strikes"] == 1
    assert row["until"] > time.time() + 500          # wall-clock deadline, not monotonic

    restarted = OfferQuarantine(cooldown_s=600, path=path)   # a new trainer process
    assert restarted.blocked("shadeform", _dallas())
    assert restarted.strike("shadeform", _dallas()) == (2, 1200)   # the count continues


def test_an_expired_deadline_unblocks_but_keeps_the_strike_count(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"offers": [{"key": ["shadeform", "latitude", "dallas-usa-1",
                                                    "cpu_32"], "strikes": 3,
                                            "until": time.time() - 1}]}))
    q = OfferQuarantine(cooldown_s=600, path=path)
    assert not q.blocked("shadeform", _dallas())
    assert q.strike("shadeform", _dallas()) == (4, 4800)


def test_clear_is_persisted(tmp_path):
    path = tmp_path / "q.json"
    q = OfferQuarantine(cooldown_s=600, path=path)
    q.strike("shadeform", _dallas())
    q.clear("shadeform", _dallas())
    assert json.loads(path.read_text())["offers"] == []
    assert not OfferQuarantine(cooldown_s=600, path=path).blocked("shadeform", _dallas())


def test_a_torn_file_starts_empty_and_never_raises(tmp_path):
    path = tmp_path / "q.json"
    path.write_text("{not json")
    q = OfferQuarantine(cooldown_s=600, path=path)
    assert not q.blocked("shadeform", _dallas())
    assert q.strike("shadeform", _dallas()) == (1, 600)      # and it rewrites a valid file
    assert json.loads(path.read_text())["offers"][0]["strikes"] == 1


def test_without_a_path_behaviour_is_unchanged(tmp_path):
    t = [100.0]
    q = OfferQuarantine(cooldown_s=10, clock=lambda: t[0])
    assert q.strike("shadeform", _dallas()) == (1, 10)
    assert q.blocked("shadeform", _dallas())
    t[0] += 11
    assert not q.blocked("shadeform", _dallas())
    assert not list(tmp_path.iterdir())                      # nothing written


@pytest.mark.parametrize("spec,dallas,ashburn", [
    ("shadeform/latitude/dallas-usa-1/cpu_32", True, False),
    ("shadeform/latitude/dallas-usa-1", True, False),        # every type in the region
    ("shadeform/latitude", True, True),                       # the whole cloud
    ("shadeform/latitude/dallas-usa-1/cpu_64", False, False),
])
def test_excluded_offers_are_always_blocked(spec, dallas, ashburn):
    q = OfferQuarantine(excluded=(spec,))
    assert q.blocked("shadeform", _dallas()) is dallas
    assert q.blocked("shadeform", _ashburn()) is ashburn


def test_config_parses_defaults_and_values():
    cfg = parse_generation({})
    assert cfg.boot_fail_cooldown_seconds == 10800 and cfg.exclude_offers == ()
    cfg = parse_generation({"boot_fail_cooldown_seconds": 3600,
                            "exclude_offers": ["shadeform/latitude/dallas-usa-1/cpu_32"]})
    assert cfg.boot_fail_cooldown_seconds == 3600
    assert cfg.exclude_offers == ("shadeform/latitude/dallas-usa-1/cpu_32",)
    shipped = load_chain_config()
    assert isinstance(shipped.generation.exclude_offers, tuple)


@pytest.mark.parametrize("raw", [
    {"boot_fail_cooldown_seconds": 10},
    {"boot_fail_cooldown_seconds": 30 * 3600},
    {"boot_fail_cooldown_seconds": True},
    {"boot_fail_cooldown_seconds": 1.5},
    {"exclude_offers": "shadeform/latitude/dallas-usa-1"},
    {"exclude_offers": ["shadeform"]},
    {"exclude_offers": ["shadeform//dallas-usa-1"]},
    {"exclude_offers": ["a/b/c/d/e"]},
])
def test_config_rejects_bad_values(raw):
    with pytest.raises(ValueError):
        parse_generation(raw)


def test_the_runner_wires_the_persisted_quarantine(tmp_path):
    import inspect

    from cascade.trainer import runner_generation

    src = inspect.getsource(runner_generation)
    assert "generation_offer_quarantine.json" in src
    assert "exclude_offers=tuple(self.cfg.generation.exclude_offers)" in src
    assert "boot_fail_cooldown_s=float(self.cfg.generation.boot_fail_cooldown_seconds)" in src
