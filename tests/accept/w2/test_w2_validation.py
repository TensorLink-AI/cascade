"""W2: type and range validation of every [lane_d] key at load (the same
loader an overlay-merged table runs through), incl. the 131072 context cap,
contexts / dtypes subsets and the era-boundary activation block."""

from __future__ import annotations

import pytest

from .helpers import (
    ACCEPTED,
    ARM_BLOCK,
    REFUSED,
    SHIPPED,
    co_for,
    field,
    load_ok,
    norm,
    raw_with,
    refused,
)

CASES = [(k, v) for k, vals in REFUSED.items() for v in vals]
OK_CASES = [(k, v) for k, vals in ACCEPTED.items() for v in vals]


def _co(key: str, value=None) -> dict:
    return co_for(key, value)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize(("key", "value"), CASES, ids=[f"{k}={v!r}" for k, v in CASES])
def test_w2_invalid_value_is_refused_at_load(name, key, value):
    raw = raw_with(name, {**_co(key, value), key: value})
    assert refused(raw), f"[lane_d] {key} = {value!r} loaded; it must be refused"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize(("key", "value"), OK_CASES, ids=[f"{k}={v!r}" for k, v in OK_CASES])
def test_w2_valid_value_loads(name, key, value):
    ld = load_ok(raw_with(name, {**_co(key, value), key: value}),
                 f" with {key}={value!r}").lane_d
    assert norm(field(ld, key)) == norm(value)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_contexts_cap_is_131072(name):
    """Contexts are a subset of the allowed variants, each <= 131072 (D2)."""
    ld = load_ok(raw_with(name, {"contexts": [131072]})).lane_d
    assert norm(field(ld, "contexts")) == (131072,)
    for over in ([262144], [4096, 131072 * 2], [131072 + 32]):
        assert refused(raw_with(name, {"contexts": over})), f"contexts {over} loaded"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_horizon_cap_is_512(name):
    """§1.1: any lane D horizon <= 512 points (CPM 16 patches x 32); 720 refused."""
    assert norm(field(load_ok(raw_with(name, {"horizon": 512})).lane_d, "horizon")) == 512
    assert refused(raw_with(name, {"horizon": 513}))
    assert refused(raw_with(name, {"horizon": 720}))


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_dtypes_only_bf16(name):
    """D14: BF16 only; F32 refused."""
    ld = load_ok(raw_with(name, {"dtypes": ["BF16"]})).lane_d
    assert norm(field(ld, "dtypes")) == ("BF16",)
    assert refused(raw_with(name, {"dtypes": ["F32"]}))
    assert refused(raw_with(name, {"dtypes": ["BF16", "F32"]}))


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_remainder_only_burn(name):
    """D21 option 1: the unpaid remainder burns; no other destination loads."""
    ld = load_ok(raw_with(name, {"remainder": "burn"})).lane_d
    assert field(ld, "remainder") == "burn"
    for bad in ("lane_a", "king", "house"):
        assert refused(raw_with(name, {"remainder": bad})), f"remainder {bad!r} loaded"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_from_block_arming_loads_at_an_era_boundary(name):
    """Today's loader refuses any from_block > 0 (lane_d.py:78-79). W2 lifts
    that: an era-boundary from_block loads, an off-boundary one is refused."""
    ld = load_ok(raw_with(name, {"from_block": ARM_BLOCK}), " armed at an era boundary").lane_d
    assert field(ld, "from_block") == ARM_BLOCK
    assert refused(raw_with(name, {"from_block": ARM_BLOCK + 1}))
    assert refused(raw_with(name, {"from_block": ARM_BLOCK + 900 + 1}))


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_until_block_must_follow_from_block(name):
    for until in (ARM_BLOCK, ARM_BLOCK - 3600):
        assert refused(raw_with(name, {"from_block": ARM_BLOCK, "until_block": until})), until
