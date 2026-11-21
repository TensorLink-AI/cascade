"""W2b: ``[lane_d_ops]`` is non-consensus. Changing every value (or removing
the table) leaves the parsed consensus config, the lane A / v2 contract
digests, the koth params and a built receipt byte-identical, with lane D
unarmed and armed alike (§14.2: "never in receipts")."""

from __future__ import annotations

import pytest

from .helpers import (
    ARM,
    OPS_KEYS,
    SHIPPED,
    TABLE_A,
    TABLE_B,
    comparable,
    consensus_fingerprint,
    field,
    load_ok,
    ops_of,
    raw_with,
    receipt_bytes,
)

ARMING = {"unarmed": None, "armed": ARM}


def _variants(name: str, lane_d: dict | None):
    """(shipped ops, every key at A, every key at B, no table) configs; the
    A and B configs must really carry their values (so the mutation is legal
    and read, not ignored)."""
    shipped = load_ok(raw_with(name, lane_d=lane_d), " (shipped [lane_d_ops])")
    a = load_ok(raw_with(name, table=TABLE_A, lane_d=lane_d), " (every key at A)")
    b = load_ok(raw_with(name, table=TABLE_B, lane_d=lane_d), " (every key at B)")
    none = load_ok(raw_with(name, absent=True, lane_d=lane_d), " (no [lane_d_ops])")
    for cfg, table in ((a, TABLE_A), (b, TABLE_B)):
        ops = ops_of(cfg)
        for k in OPS_KEYS:
            assert field(ops, k) == table[k], f"[lane_d_ops] {k} not carried"
    assert ops_of(a) != ops_of(b)
    return shipped, a, b, none


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("arming", sorted(ARMING))
def test_w2b_parsed_config_identical_across_ops_values(name, arming):
    cfgs = _variants(name, ARMING[arming])
    base = comparable(cfgs[0])
    for cfg in cfgs[1:]:
        assert comparable(cfg) == base, "[lane_d_ops] moved the parsed chain config"
        assert cfg.lane_d == cfgs[0].lane_d


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("arming", sorted(ARMING))
def test_w2b_contract_digests_and_koth_params_identical(name, arming):
    cfgs = _variants(name, ARMING[arming])
    base = consensus_fingerprint(cfgs[0])
    for cfg in cfgs[1:]:
        assert consensus_fingerprint(cfg) == base, "[lane_d_ops] moved a consensus digest"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_built_receipt_byte_identical(name):
    cfgs = _variants(name, None)
    base = receipt_bytes(cfgs[0])
    for cfg in cfgs[1:]:
        assert receipt_bytes(cfg) == base, "[lane_d_ops] changed a built receipt's bytes"
    assert "lane_d_ops" not in base[0] and "lane_d_ops" not in base[1]


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_receipt_module_never_names_the_table(name):
    """No receipt field and no receipt builder reads the table."""
    from .helpers import string_constants

    for mod in ("cascade.shared.receipt", "cascade.validator.receipts",
                "cascade.shared.manifest"):
        assert "lane_d_ops" not in string_constants(mod), f"{mod} names [lane_d_ops]"
    assert ops_of(load_ok(raw_with(name))) is not None
