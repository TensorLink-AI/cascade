"""W-S1: the sentinel keys are overlay-settable (§14.2: "Signed overlay once the
capability release adds them to ALLOWED["v2"]"), arming takes effect from
the overlay's era boundary (D45: no shadow), and nothing else in the lane A
config moves."""

from __future__ import annotations

import pytest

from .helpers import (
    EFFECTIVE,
    ERA,
    MUTATIONS,
    SHIPPED,
    WS1_DEFAULTS,
    WS1_KEYS,
    field,
    lane_a_contract_digests,
    load_ok,
    norm,
    overrides_refusal,
    shipped_raw,
    validate_ok,
)


def _value(key: str):
    a, b = MUTATIONS[key]
    return b if norm(a) == norm(WS1_DEFAULTS[key]) else a


def test_ws1_keys_are_on_the_v2_allow_list():
    from cascade.shared import netconfig as nc

    missing = [k for k in WS1_KEYS if k not in (nc.ALLOWED.get("v2") or ())]
    assert not missing, f"ALLOWED['v2'] lacks {missing}"


@pytest.mark.parametrize("key", WS1_KEYS)
def test_ws1_key_passes_check_overrides(key):
    why = overrides_refusal({"v2": {key: _value(key)}})
    assert why is None, f"[v2] {key} refused by check_overrides: {why}"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", WS1_KEYS)
def test_ws1_key_overlay_takes_effect(name, key):
    base = load_ok(shipped_raw(name))
    value = _value(key)
    out = validate_ok(base, {"v2": {key: value}})
    assert norm(field(out.v2, key)) == norm(value), f"overlay [v2] {key}={value!r} built " \
        f"{field(out.v2, key)!r}"
    assert out.training == base.training and out.round == base.round
    assert out.scoring == base.scoring
    assert lane_a_contract_digests(out) == lane_a_contract_digests(base)


@pytest.mark.parametrize("name", SHIPPED)
def test_ws1_arming_overlay_is_live_from_its_era_boundary(name):
    """One overlay arms the sentinel at an era boundary; ``sentinel_armed`` is
    false before it and true from it, with no shadow period."""
    from cascade.shared.sentinel import sentinel_armed

    base = load_ok(shipped_raw(name))
    for b in (0, EFFECTIVE - ERA[name], EFFECTIVE, EFFECTIVE + ERA[name]):
        assert sentinel_armed(base, b) is False, f"unarmed config armed at {b}"
    armed = validate_ok(base, {"v2": {"sentinel_from_block": EFFECTIVE}})
    assert sentinel_armed(armed, EFFECTIVE - ERA[name]) is False
    assert sentinel_armed(armed, EFFECTIVE - 1) is False
    assert sentinel_armed(armed, EFFECTIVE) is True
    assert sentinel_armed(armed, EFFECTIVE + ERA[name]) is True


@pytest.mark.parametrize("name", SHIPPED)
def test_ws1_overlay_off_block_is_refused(name):
    """A sentinel_from_block off the era grid never validates."""
    from cascade.shared import netconfig as nc

    base = load_ok(shipped_raw(name))
    body = {"protocol": nc.PROTOCOL, "network": nc.network_identity(base), "seq": 1,
            "prev": None, "effective_block": EFFECTIVE,
            "overrides": {"v2": {"sentinel_from_block": EFFECTIVE + 1}}}
    accepted = True
    try:
        nc.validate_overlay(base, body)
    except nc.NetConfigError:
        accepted = False
    assert not accepted, "an overlay arming the sentinel off an era boundary validated"
    assert overrides_refusal({"v2": {"sentinel_from_block": EFFECTIVE}}) is None, (
        "[v2] sentinel_from_block is not overlay-settable")


def test_ws1_typo_key_is_refused():
    why = overrides_refusal({"v2": {"sentinel_suite": ["gift"]}})
    assert why is not None, "a typo'd [v2] sentinel key passed check_overrides"
