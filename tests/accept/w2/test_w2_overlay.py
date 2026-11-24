"""W2: ``ALLOWED["lane_d"]`` in netconfig — every lane D tuning key is
overlay-settable (§14.2), ``remainder`` is release-only through the
``(table, key)`` special case (netconfig.py:210, with ``[lane_c] remainder``
closed in the same release), and the T18 / T21 / T24 config halves."""

from __future__ import annotations

import pytest

from .helpers import (
    ARM,
    ARM_BLOCK,
    MUTATIONS,
    SETTLEMENT,
    SHIPPED,
    UNTIL_BLOCK,
    W2_KEYS,
    field,
    load_ok,
    norm,
    overlay_body,
    overrides_refusal,
    raw_with,
    shipped_raw,
    validate_ok,
    weights,
)

OVERLAY_KEYS = [k for k in W2_KEYS if k not in ("remainder", "dtypes")]
# Overlay values that differ from both the plan default and ARM (an overlay
# restating a default is refused as "no effect"). takes_from moves the armed
# carve (ARM: lane_a) to another lane.
OVERLAY_VALUE = {"takes_from": "lane_b", "confirm_on_king_change": "original",
                 "ready_ttl_blocks": 54321}


def test_w2_allowed_has_lane_d():
    from cascade.shared import netconfig as nc

    assert "lane_d" in nc.ALLOWED, "netconfig.ALLOWED has no lane_d table"
    policy = nc.overlay_policy()
    assert "lane_d" in str(policy), "overlay_policy() does not list lane_d"


@pytest.mark.parametrize("key", OVERLAY_KEYS)
def test_w2_every_tuning_key_passes_check_overrides(key):
    value = MUTATIONS[key][1]
    why = overrides_refusal({"lane_d": {key: value}})
    assert why is None, f"[lane_d] {key} refused by check_overrides: {why}"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", OVERLAY_KEYS)
def test_w2_every_tuning_key_overlay_takes_effect(name, key):
    """Each key set by a signed overlay (with the activation keys) validates
    and lands in the built config."""
    b = OVERLAY_VALUE.get(key, MUTATIONS[key][1])
    ov = {**ARM, key: b}
    if key == "gate_suites":
        # unequal, so the weights are not a no-op restatement of the default
        ov["gate_weights"] = {s: float(i + 1) for i, s in enumerate(b)}
    base = load_ok(shipped_raw(name))
    out = validate_ok(base, {"lane_d": ov})
    got = field(out.lane_d, key)
    if key == "gate_weights":
        assert weights(got) == weights(b)
    else:
        assert norm(got) == norm(b), f"overlay [lane_d] {key}={b!r} built {got!r}"


def test_w2_remainder_overlay_is_refused_as_release_only():
    """B5 / §14.2: the burn destination is release-only, like lane_b's."""
    why = overrides_refusal({"lane_d": {"remainder": "burn", "share": 0.25}})
    assert why is not None, "an overlay setting [lane_d] remainder was accepted"
    assert "[lane_d] remainder" in why and "burn destination is release-only" in why, why
    assert "share" not in why, f"share must stay overlay-settable: {why}"


def test_w2_lane_c_remainder_overlay_is_refused_as_release_only():
    """§14.2 (rev5): the same release closes [lane_c] remainder."""
    why = overrides_refusal({"lane_c": {"remainder": "burn"}})
    assert why is not None, "an overlay setting [lane_c] remainder was accepted"
    assert "[lane_c] remainder" in why and "burn destination is release-only" in why, why
    assert overrides_refusal({"lane_c": {"share": 0.2}}) is None


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_unknown_lane_d_overlay_key_is_refused(name):
    """A typo'd lane D key reaches no parser: refused as having no effect."""
    from cascade.shared import netconfig as nc

    base = load_ok(shipped_raw(name))
    body = overlay_body(base, {"lane_d": {"d_margn": 0.02}})
    try:
        nc.validate_overlay(base, body)
    except nc.NetConfigError as e:
        assert "no effect" in str(e), f"refused for another reason: {e}"
    else:
        raise AssertionError("an overlay with unknown key [lane_d] d_margn was adopted")


@pytest.mark.parametrize("name", SHIPPED)
def test_t18_until_block_overlay_with_entries_pending(name):
    """T18 (config part): on an armed base an overlay sets until_block; it
    validates and the lane stops being active after it."""
    armed = load_ok(raw_with(name, ARM), " armed base")
    out = validate_ok(armed, {"lane_d": {"until_block": UNTIL_BLOCK}})
    assert field(out.lane_d, "until_block") == UNTIL_BLOCK
    assert out.lane_d.active(UNTIL_BLOCK - SETTLEMENT)
    assert not out.lane_d.active(UNTIL_BLOCK + SETTLEMENT)


@pytest.mark.parametrize("name", SHIPPED)
def test_t21_kill_switch_through_the_signed_overlay(name):
    """T21 (config part): share = 0 or until_block by overlay, no release."""
    armed = load_ok(raw_with(name, ARM), " armed base")
    out = validate_ok(armed, {"lane_d": {"share": 0.0}})
    assert field(out.lane_d, "share") == 0.0
    out2 = validate_ok(armed, {"lane_d": {"until_block": UNTIL_BLOCK}})
    assert not out2.lane_d.active(UNTIL_BLOCK + SETTLEMENT)
    # the lane A tables ride through untouched
    assert out.scoring == armed.scoring and out.training == armed.training


@pytest.mark.parametrize("name", SHIPPED)
def test_t24_old_code_refuses_a_lane_d_overlay_whole(name, monkeypatch):
    """T24 (config part): code without ALLOWED["lane_d"] (today's main)
    refuses the whole overlay, lane A keys included; the new code accepts it."""
    from cascade.shared import netconfig as nc

    overrides = {"lane_d": {"from_block": ARM_BLOCK}, "scoring": {"min_windows": 7}}
    assert "lane_d" in nc.ALLOWED, "new code has no ALLOWED['lane_d']"
    assert overrides_refusal(overrides) is None, overrides_refusal(overrides)
    old = {t: v for t, v in nc.ALLOWED.items() if t != "lane_d"}
    monkeypatch.setattr(nc, "ALLOWED", old)
    why = overrides_refusal(overrides)
    assert why is not None and "[lane_d] (not on the DEC-CA-0057 allow-list)" in why, why
    base = load_ok(shipped_raw(name))
    body = {"protocol": nc.PROTOCOL, "network": nc.network_identity(base), "seq": 1,
            "prev": None, "effective_block": 2698 * 3600, "overrides": overrides}
    try:
        nc.validate_overlay(base, body)
    except nc.NetConfigError:
        pass
    else:
        raise AssertionError("old allow-list adopted a [lane_d] overlay")
