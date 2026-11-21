"""WB10: invariant 1 for lane B (unarmed => byte-identical), paired per
invariant 10 with an armed test. The tooling's pairing suffix is
``armed_nonempty_lane_d_effect``; here it names the lane B effect."""

from __future__ import annotations

import pytest

from .helpers import (
    HASHSET_A,
    SHIPPED,
    WB10_DEFAULTS,
    WB10_KEYS,
    comparable,
    consensus_fingerprint,
    default_of,
    field,
    lane_a_contract_digests,
    load_ok,
    norm,
    raw_with,
    same_value,
    shipped_raw,
    validate_ok,
)


def _explicit_defaults(name: str) -> dict:
    out = {}
    for k in WB10_KEYS:
        v = default_of(name, k)
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_b_config_unchanged_when_unarmed(name):
    """Drop-when-default: the shipped config, the config whose [lane_b] has
    every WB10 key removed (main's table) and the config with every WB10
    default written out parse equal, with equal contract / numerics / koth /
    lane B leg-contract / verdict digests, slots and active() probes."""
    shipped = load_ok(shipped_raw(name))
    dropped = load_ok(raw_with(name, drop=WB10_KEYS), " (no WB10 keys)")
    explicit = load_ok(raw_with(name, _explicit_defaults(name)), " (explicit defaults)")
    assert shipped.lane_b == dropped.lane_b == explicit.lane_b
    assert comparable(shipped) == comparable(dropped) == comparable(explicit)
    fp = consensus_fingerprint(dropped)
    assert consensus_fingerprint(shipped) == fp
    assert consensus_fingerprint(explicit) == fp


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_b_config_armed_nonempty_lane_d_effect(name):
    """The armed pair: a release pins the hash set, then one signed overlay
    sets suite targets, the D41 opening rule, the D39 fractions and the D42
    leg keys. Every key takes effect and lane B's config differs; lane A's
    training contract digests, [round] (incl. its GPU list) and [training]
    max_train_seconds don't move."""
    base = load_ok(raw_with(name, {"contamination_hashset": HASHSET_A}), " (release pin)")
    overrides = {
        "suite_targets": True,
        "open_rule": "tokens_king_check",
        "min_experts_frac": 0.0625,
        "max_experts_frac": 0.125,
        "sentinel_max_defer_eras": 2,
        "funded_pod_skus": ["8xH100", "8xA100"],
        "max_leg_hours": 4.5,
        "max_leg_seconds": 21_600,
        "guard_factor": 1.5,
        "suite_target_max_repo_mb": 4,
    }
    armed = validate_ok(base, {"lane_b": overrides})
    lb = armed.lane_b
    for k, v in overrides.items():
        assert same_value(k, field(lb, k), v), f"overlay [lane_b] {k}={v!r} built {field(lb, k)!r}"
    assert field(lb, "contamination_hashset") == HASHSET_A
    assert field(lb, "router_rows") == WB10_DEFAULTS["router_rows"]
    assert lb != base.lane_b
    assert lane_a_contract_digests(armed) == lane_a_contract_digests(base), (
        "arming lane B keys moved a lane A contract digest")
    assert armed.round == base.round
    assert norm(armed.round.funded_pod_skus) == norm(base.round.funded_pod_skus)
    assert armed.training.max_train_seconds == base.training.max_train_seconds
