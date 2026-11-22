"""WB10: type and range validation of every new [lane_b] key at load (the
same loader an overlay-merged table runs through), incl. the cross-key
checks: min_experts_frac <= max_experts_frac and suite_target_max_repo_mb
within [generator] max_repo_mb."""

from __future__ import annotations

import pytest

from .helpers import (
    ACCEPTED,
    MUTATIONS,
    REFUSED,
    SHIPPED,
    SINGLE_VALUED,
    WB10_DEFAULTS,
    field,
    generator_max_repo_mb,
    load_ok,
    raw_with,
    refused,
    same_value,
)

CASES = [(k, v) for k, vals in REFUSED.items() for v in vals]
OK_CASES = [(k, v) for k, vals in ACCEPTED.items() for v in vals]


def _co(key: str) -> dict:
    return dict(MUTATIONS[key][2]) if key in MUTATIONS else {}


def _parsed(name: str, key: str, value):
    """Load with ``key = value``; the key must be parsed (so main fails here
    with an AssertionError rather than passing a refusal test vacuously)."""
    lb = load_ok(raw_with(name, {**_co(key), key: value}), f" with {key}={value!r}").lane_b
    return field(lb, key)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize(("key", "value"), CASES, ids=[f"{k}={v!r}" for k, v in CASES])
def test_wb10_invalid_value_is_refused_at_load(name, key, value):
    raw = raw_with(name, {**_co(key), key: value})
    assert refused(raw), f"[lane_b] {key} = {value!r} loaded; it must be refused"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize(("key", "value"), OK_CASES, ids=[f"{k}={v!r}" for k, v in OK_CASES])
def test_wb10_valid_value_loads(name, key, value):
    assert same_value(key, _parsed(name, key, value), value)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", SINGLE_VALUED)
def test_wb10_single_valued_key_takes_only_its_decided_value(name, key):
    """D40 (a) router rows frozen; D35 (a) a sentinel-target entry defers.
    The decided value loads; every other string is refused."""
    assert _parsed(name, key, WB10_DEFAULTS[key]) == WB10_DEFAULTS[key]
    for bad in ("", "any", "none"):
        assert refused(raw_with(name, {key: bad})), f"[lane_b] {key} = {bad!r} loaded"


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_expert_fracs_need_min_le_max(name):
    """D39: per-layer clamp min = ceil(min_frac E), max = floor(max_frac E)."""
    assert _parsed(name, "min_experts_frac", 0.0625) == 0.0625
    assert refused(raw_with(name, {"min_experts_frac": 0.2, "max_experts_frac": 0.1484375}))
    assert refused(raw_with(name, {"min_experts_frac": 0.03125, "max_experts_frac": 0.02}))
    lb = load_ok(raw_with(name, {"min_experts_frac": 0.125, "max_experts_frac": 0.125})).lane_b
    assert field(lb, "min_experts_frac") == field(lb, "max_experts_frac") == 0.125


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_expert_fracs_give_the_256_era_clamp(name):
    """D39: the default fractions are exactly 8/256 and 38/256."""
    lb = load_ok(raw_with(name, drop=("min_experts_frac", "max_experts_frac"))).lane_b
    assert field(lb, "min_experts_frac") * 256 == 8
    assert field(lb, "max_experts_frac") * 256 == 38


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_suite_target_repo_cap_within_generator_cap(name):
    """D38: a stricter cap for suite-target entries; never above [generator]
    max_repo_mb, which stays unchanged for everyone else."""
    cap = generator_max_repo_mb(name)
    assert _parsed(name, "suite_target_max_repo_mb", cap) == cap
    assert refused(raw_with(name, {"suite_target_max_repo_mb": cap + 1}))
    cfg = load_ok(raw_with(name, {"suite_target_max_repo_mb": 4}))
    assert field(cfg.lane_b, "suite_target_max_repo_mb") == 4
    assert int(cfg.generator.max_repo_mb) == cap


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_max_leg_seconds_never_touches_training_max_train_seconds(name):
    """D42: the lane B cap is new; [training] max_train_seconds (a
    contract-digest field) is never edited for it."""
    base = load_ok(raw_with(name, {}))
    cfg = load_ok(raw_with(name, {"max_leg_seconds": 21_600, "max_leg_hours": 4.5,
                                  "guard_factor": 1.5}))
    assert field(cfg.lane_b, "max_leg_seconds") == 21_600
    assert field(cfg.lane_b, "max_leg_hours") == 4.5 and field(cfg.lane_b, "guard_factor") == 1.5
    assert cfg.training.max_train_seconds == base.training.max_train_seconds
    v2b, v2c = (getattr(c.training, "v2_contract", None) for c in (base, cfg))
    assert (v2b is None) == (v2c is None)
    if v2b is not None:
        assert v2c.max_train_seconds == v2b.max_train_seconds
