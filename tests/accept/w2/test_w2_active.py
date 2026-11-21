"""W2: ``LaneDConfig.active()`` is real (not the hardcoded False of
lane_d.py:65-68), and invariant 1 (unarmed => byte-identical) for the config
half, paired per invariant 10 with an armed test."""

from __future__ import annotations

import pytest

from .helpers import (
    ARM,
    ARM_BLOCK,
    SETTLEMENT,
    SHIPPED,
    UNTIL_BLOCK,
    W2_DEFAULTS,
    comparable,
    consensus_fingerprint,
    field,
    load_ok,
    norm,
    raw_with,
    shipped_raw,
    validate_ok,
)

PROBES = (None, 0, 1, ARM_BLOCK - SETTLEMENT, ARM_BLOCK, ARM_BLOCK + SETTLEMENT,
          UNTIL_BLOCK + SETTLEMENT, 10**9)


def _explicit_default_table(name: str) -> dict:
    """The shipped [lane_d] with every pinned W2 default written out."""
    table = dict(shipped_raw(name).get("lane_d") or {})
    for k, v in W2_DEFAULTS.items():
        if v is not None:
            table[k] = list(v) if isinstance(v, tuple) else v
    return table


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_active_is_real_when_armed(name):
    ld = load_ok(raw_with(name, ARM), " armed").lane_d
    assert not ld.active(None)
    assert not ld.active(ARM_BLOCK - SETTLEMENT), "active before from_block"
    assert ld.active(ARM_BLOCK + SETTLEMENT), "armed lane D not active after from_block"
    assert ld.active(10**9)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_active_ends_at_until_block(name):
    """T18 (config part): at until_block the lane closes."""
    ld = load_ok(raw_with(name, {**ARM, "until_block": UNTIL_BLOCK}), " with until_block").lane_d
    assert ld.active(ARM_BLOCK + SETTLEMENT)
    assert ld.active(UNTIL_BLOCK - SETTLEMENT)
    assert not ld.active(UNTIL_BLOCK + SETTLEMENT), "active after until_block"
    assert not ld.active(10**9)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_from_block_zero_is_never_active(name):
    """from_block = 0 stays off whatever else the table sets."""
    ld = load_ok(raw_with(name, {"share": 0.25, "takes_from": "lane_a",
                                 "contexts": [4096, 131072]})).lane_d
    assert field(ld, "from_block") == 0
    assert not any(ld.active(b) for b in PROBES)


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_d_config_unchanged_when_unarmed(name):
    """Invariant 1, W2's part: with from_block = 0 the parsed config, every
    contract / consensus digest and active() equal today's; a default
    [lane_d] (absent, as shipped, or every default written out) moves
    nothing (drop-when-default)."""
    shipped = load_ok(shipped_raw(name))
    absent_raw = shipped_raw(name)
    absent_raw.pop("lane_d", None)
    absent = load_ok(absent_raw, " (no [lane_d] table)")
    explicit = load_ok(raw_with(name, table=_explicit_default_table(name)),
                       " (explicit defaults)")
    assert shipped.lane_d.from_block == 0
    assert shipped.lane_d == absent.lane_d == explicit.lane_d
    assert comparable(shipped) == comparable(explicit) == comparable(absent)
    fp = consensus_fingerprint(shipped)
    assert consensus_fingerprint(absent) == fp
    assert consensus_fingerprint(explicit) == fp
    assert not any(shipped.lane_d.active(b) for b in PROBES)


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_d_config_armed_nonempty_lane_d_effect(name):
    """The armed pair: from_block > 0 by signed overlay => active() true
    after the block and the new keys take effect; lane A's training contract
    digests are still untouched."""
    base = load_ok(shipped_raw(name))
    armed = validate_ok(base, {"lane_d": {**ARM, "contexts": [4096, 8192, 16384, 32768],
                                          "d_margin": 0.02}})
    ld = armed.lane_d
    assert field(ld, "from_block") == ARM_BLOCK
    assert ld.active(ARM_BLOCK + SETTLEMENT) and not ld.active(ARM_BLOCK - SETTLEMENT)
    assert field(ld, "share") == 0.25 and field(ld, "takes_from") == "lane_a"
    assert norm(field(ld, "contexts")) == (4096, 8192, 16384, 32768)
    assert field(ld, "d_margin") == 0.02
    assert ld != base.lane_d
    fp_base, fp_armed = consensus_fingerprint(base), consensus_fingerprint(armed)
    for k in ("contract_digest", "numerics_contract_digest"):
        assert fp_armed[k] == fp_base[k], f"arming lane D moved {k}"
