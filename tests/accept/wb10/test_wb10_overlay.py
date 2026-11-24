"""WB10: ``ALLOWED["lane_b"] = None`` covers every new key (overlay-settable,
§12b.4-§12b.6, §14.2) except ``contamination_hashset``, which joins
``remainder`` in the release-only tuple (rev5, ``netconfig.py:210``)."""

from __future__ import annotations

import pytest

from .helpers import (
    HASHSET_A,
    HASHSET_B,
    MUTATIONS,
    RELEASE_ONLY,
    SHIPPED,
    SINGLE_VALUED,
    WB10_DEFAULTS,
    WB10_KEYS,
    field,
    load_ok,
    overlay_body,
    overrides_refusal,
    raw_with,
    same_value,
)

OVERLAY_KEYS = [k for k in WB10_KEYS if k not in RELEASE_ONLY]


def _value(key: str):
    """A non-default value where the key has one (single-valued keys restate
    their decided value: the base table carries the key, so it is accepted)."""
    if key in SINGLE_VALUED:
        return WB10_DEFAULTS[key]
    a, b, _co = MUTATIONS[key]
    return b if same_value(key, a, WB10_DEFAULTS[key]) else a


def _base(name: str):
    """The shipped config with a hash set pinned by release (a suite-target
    overlay arms on top of it)."""
    return load_ok(raw_with(name, {"contamination_hashset": HASHSET_A}), " (hash set pinned)")


def test_wb10_lane_b_is_wildcard_on_the_allow_list():
    from cascade.shared import netconfig as nc

    assert nc.ALLOWED.get("lane_b", "missing") is None, "ALLOWED['lane_b'] is not None"


@pytest.mark.parametrize("key", OVERLAY_KEYS)
def test_wb10_every_new_key_passes_check_overrides(key):
    why = overrides_refusal({"lane_b": {key: _value(key)}})
    assert why is None, f"[lane_b] {key} refused by check_overrides: {why}"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", OVERLAY_KEYS)
def test_wb10_every_new_key_overlay_takes_effect(name, key):
    """Each key set by a signed overlay validates and lands in the built
    config (lane A tables ride through unchanged)."""
    from cascade.shared import netconfig as nc

    base = _base(name)
    value = _value(key)
    body = overlay_body(base, {"lane_b": {key: value}})
    try:
        out = nc.validate_overlay(base, body)
    except nc.NetConfigError as e:
        raise AssertionError(f"overlay [lane_b] {key}={value!r} refused: {e}") from None
    got = field(out.lane_b, key)
    assert same_value(key, got, value), f"overlay [lane_b] {key}={value!r} built {got!r}"
    assert out.training == base.training and out.round == base.round
    assert out.scoring == base.scoring


def test_wb10_contamination_hashset_overlay_is_a_content_hash():
    """Owner 2026-11-24 (DEC-CA-0075 item 3, tests-correction of the rev5
    release-only pin): an overlay names the hash set by CONTENT HASH (+ its
    storage key) — accepted by the allow-list, every reader verifies the
    fetched bytes against the pin; other lane B keys stay settable."""
    why = overrides_refusal({"lane_b": {"contamination_hashset": HASHSET_B,
                                        "contamination_hashset_key": "contamination/hs.json",
                                        "open_rule": "tokens_king_check"}})
    assert why is None, why


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_contamination_hashset_overlay_validates_only_as_sha256(name):
    from cascade.shared import netconfig as nc

    base = _base(name)

    def body(v):
        return {"protocol": nc.PROTOCOL, "network": nc.network_identity(base), "seq": 1,
                "prev": None, "effective_block": 2698 * 3600,
                "overrides": {"lane_b": {"contamination_hashset": v}}}

    out = nc.validate_overlay(base, body(HASHSET_B))
    assert out.lane_b.contamination_hashset == HASHSET_B
    for bad in ("md5:abc", "sha256:XYZ", "hashset-v2.json"):
        try:
            nc.validate_overlay(base, body(bad))
        except nc.NetConfigError as e:
            assert "contamination_hashset" in str(e), e
        else:
            raise AssertionError(f"a non-content-hash pin {bad!r} was adopted")


def test_wb10_remainder_stays_release_only():
    why = overrides_refusal({"lane_b": {"remainder": "burn"}})
    assert why is not None and "[lane_b] remainder" in why and "release-only" in why, why


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_unknown_lane_b_overlay_key_is_refused(name):
    """A typo'd lane B key reaches no parser: refused as having no effect."""
    from cascade.shared import netconfig as nc

    base = _base(name)
    body = overlay_body(base, {"lane_b": {"open_rul": "tokens"}})
    try:
        nc.validate_overlay(base, body)
    except nc.NetConfigError as e:
        assert "no effect" in str(e), f"refused for another reason: {e}"
    else:
        raise AssertionError("an overlay with unknown key [lane_b] open_rul was adopted")
