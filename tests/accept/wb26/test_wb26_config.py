"""WB26: the ``[lane_b]`` keys ``gain_margin_public`` (0.01),
``tau_anchor_public`` (0.0) and ``tau_anchor_domain`` (0.0) — loader parse,
shipped defaults, range checks, round-trip / mutation, ``verdict_params``,
overlay-settable (``ALLOWED["lane_b"] = None``)."""

from __future__ import annotations

import pytest

from tests.accept.wb10 import helpers as wb10

KEYS = {"gain_margin_public": 0.01, "tau_anchor_public": 0.0, "tau_anchor_domain": 0.0}
MUTATED = {"gain_margin_public": 0.02, "tau_anchor_public": 0.005, "tau_anchor_domain": 0.003}
BAD = [("gain_margin_public", 0.0), ("gain_margin_public", 1.0), ("gain_margin_public", -0.1),
       ("gain_margin_public", "x"), ("tau_anchor_public", -0.01), ("tau_anchor_public", 1.0),
       ("tau_anchor_domain", -0.01), ("tau_anchor_domain", True)]


@pytest.mark.parametrize("name", wb10.SHIPPED)
def test_shipped_files_carry_the_defaults(name):
    raw = wb10.shipped_raw(name)["lane_b"]
    lb = wb10.load_ok(wb10.shipped_raw(name)).lane_b
    for key, want in KEYS.items():
        assert raw.get(key) == want, f"{name} [lane_b] {key} = {raw.get(key)!r}, want {want}"
        assert getattr(lb, key, None) == want, f"{name}: LaneBConfig.{key} not parsed"


@pytest.mark.parametrize("name", wb10.SHIPPED)
@pytest.mark.parametrize("key", sorted(KEYS))
def test_keys_round_trip_and_mutate(name, key):
    lb = wb10.load_ok(wb10.raw_with(name, {key: MUTATED[key]})).lane_b
    assert getattr(lb, key, None) == MUTATED[key], f"[lane_b] {key} = {MUTATED[key]} not parsed"
    vp = lb.verdict_params()
    assert getattr(vp, key, None) == MUTATED[key], f"verdict_params() drops {key}"
    dropped = wb10.load_ok(wb10.raw_with(name, drop=(key,))).lane_b
    assert getattr(dropped, key, None) == KEYS[key], f"absent {key} defaults to {KEYS[key]}"


@pytest.mark.parametrize("key,value", BAD)
def test_out_of_range_values_are_refused(key, value):
    assert wb10.refused(wb10.raw_with("chain.toml", {key: value})), (
        f"[lane_b] {key} = {value!r} was accepted")


@pytest.mark.parametrize("key", sorted(KEYS))
def test_keys_are_overlay_settable(key):
    base = wb10.load_ok(wb10.shipped_raw("chain.toml"))
    out = wb10.validate_ok(base, {"lane_b": {key: MUTATED[key]}})
    assert getattr(out.lane_b, key, None) == MUTATED[key]
    assert out.training == base.training and out.round == base.round
