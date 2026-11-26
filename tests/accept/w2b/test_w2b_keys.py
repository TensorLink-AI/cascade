"""W2b: every ``[lane_d_ops]`` key parses with its default, is carried by
both shipped files, is validated, survives a TOML round-trip and is
MUTATION-tested; unknown and secret-shaped keys are refused."""

from __future__ import annotations

import pytest

from .helpers import (
    ACCEPTED,
    INT_KEYS,
    MUTATIONS,
    OPS_DEFAULTS,
    OPS_KEYS,
    OPS_MODULE,
    REFUSED,
    SECRET_KEYS,
    SHIPPED,
    STR_KEYS,
    TABLE_A,
    TABLE_B,
    comparable,
    dump_ops,
    field,
    load_ok,
    load_text,
    ops_of,
    ops_ok,
    raw_with,
    refusal,
    shipped_raw,
    splice_ops,
    valid_bucket,
    valid_prefix,
)

PINNED_DEFAULTS = [k for k, v in OPS_DEFAULTS.items() if v is not None]


def test_w2b_module_defines_the_config():
    import importlib

    mod = importlib.import_module(OPS_MODULE)
    assert hasattr(mod, "LaneDOpsConfig"), f"{OPS_MODULE} has no LaneDOpsConfig"
    assert hasattr(mod, "lane_d_ops_config"), f"{OPS_MODULE} has no lane_d_ops_config"
    assert mod.LaneDOpsConfig.__module__ == OPS_MODULE


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", PINNED_DEFAULTS)
def test_w2b_key_default(name, key):
    ops = ops_ok(raw_with(name, drop=(key,)), f" without {key}")
    assert field(ops, key) == OPS_DEFAULTS[key], (
        f"[lane_d_ops] {key} default {field(ops, key)!r}, plan says {OPS_DEFAULTS[key]!r}")


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_absent_table_gives_valid_defaults(name):
    """A deployment without the table still yields a full, valid ops view."""
    ops = ops_ok(raw_with(name, absent=True), " without [lane_d_ops]")
    for k in PINNED_DEFAULTS:
        assert field(ops, k) == OPS_DEFAULTS[k], k
    assert valid_bucket(field(ops, "pool_d_bucket")), field(ops, "pool_d_bucket")
    assert valid_prefix(field(ops, "pool_d_prefix")), field(ops, "pool_d_prefix")
    assert field(ops, "pool_d_bucket") != field(ops, "vault_bucket")
    for k in INT_KEYS:
        v = field(ops, k)
        assert type(v) is int and v > 0, f"[lane_d_ops] {k} default {v!r}"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_shipped_table_carries_every_key(name):
    table = shipped_raw(name).get("lane_d_ops")
    assert isinstance(table, dict), f"{name} has no [lane_d_ops] table"
    missing = [k for k in OPS_KEYS if k not in table]
    assert not missing, f"{name} [lane_d_ops] lacks {missing}"
    extra = sorted(set(table) - set(OPS_KEYS))
    assert not extra, f"{name} [lane_d_ops] carries undocumented keys {extra}"
    ops = ops_ok(shipped_raw(name))
    for k in STR_KEYS:
        assert type(field(ops, k)) is str, k
    for k in INT_KEYS:
        assert type(field(ops, k)) is int and field(ops, k) > 0, k
    assert field(ops, "eval_host") == "", f"{name} names a live eval host in a public file"


def test_w2b_mainnet_vault_is_the_plan_bucket():
    ops = ops_ok(shipped_raw("chain.toml"))
    assert field(ops, "vault_bucket") == "cascade-lane-d-vault"
    assert field(ops, "vault_prefix") == "lane-d"


def test_w2b_testnet_locations_differ_from_mainnet():
    """§10.3: the testnet stack is separate; its pool_d is under a testnet
    prefix, and testnet entries never land in the mainnet vault path."""
    main = ops_ok(shipped_raw("chain.toml"))
    test = ops_ok(shipped_raw("chain.testnet.toml"))
    assert ((field(main, "vault_bucket"), field(main, "vault_prefix"))
            != (field(test, "vault_bucket"), field(test, "vault_prefix")))
    assert ((field(main, "pool_d_bucket"), field(main, "pool_d_prefix"))
            != (field(test, "pool_d_bucket"), field(test, "pool_d_prefix")))


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", sorted(MUTATIONS))
def test_w2b_key_mutation_changes_the_parsed_value(name, key):
    a, b = MUTATIONS[key]
    va = field(ops_ok(raw_with(name, table={**TABLE_A, key: a})), key)
    vb = field(ops_ok(raw_with(name, table={**TABLE_A, key: b})), key)
    assert va == a, f"[lane_d_ops] {key}={a!r} parsed as {va!r}"
    assert vb == b, f"[lane_d_ops] {key}={b!r} parsed as {vb!r}"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("table", [TABLE_A, TABLE_B], ids=["A", "B"])
def test_w2b_round_trip_load_dump_load_equal(name, table, tmp_path):
    ops1 = ops_ok(raw_with(name, table=table))
    for k in OPS_KEYS:
        assert field(ops1, k) == table[k], k
    cfg2 = load_text(splice_ops(name, dump_ops(ops1)), tmp_path, name)
    assert ops_of(cfg2) == ops1, "[lane_d_ops] changed across load -> dump -> load"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_round_trip_of_the_shipped_defaults(name, tmp_path):
    cfg1 = load_ok(shipped_raw(name))
    ops1 = ops_of(cfg1)
    cfg2 = load_text(splice_ops(name, dump_ops(ops1)), tmp_path, name)
    assert ops_of(cfg2) == ops1
    assert comparable(cfg2) == comparable(cfg1)


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key,value", [(k, v) for k, vs in REFUSED.items() for v in vs])
def test_w2b_invalid_value_refused(name, key, value):
    why = refusal(raw_with(name, table={**TABLE_A, key: value}))
    assert why is not None, f"[lane_d_ops] {key}={value!r} was accepted"


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key,value", [(k, v) for k, vs in ACCEPTED.items() for v in vs])
def test_w2b_valid_value_accepted(name, key, value):
    ops = ops_ok(raw_with(name, table={**TABLE_A, key: value}), f" with {key}={value!r}")
    assert field(ops, key) == value


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("storage_key", ["manifest_bucket", "logs_bucket", "pool_bucket"])
@pytest.mark.parametrize("key", ["vault_bucket", "pool_d_bucket"])
def test_w2b_private_buckets_never_shared(name, storage_key, key):
    """§1.2 / §13: the vault (and the private pool_d) is never a shared
    [storage] bucket (private code once leaked through one)."""
    shared = shipped_raw(name)["storage"][storage_key]
    assert valid_bucket(shared), f"{name} [storage] {storage_key} = {shared!r}"
    why = refusal(raw_with(name, table={**TABLE_A, key: shared}))
    assert why is not None, f"[lane_d_ops] {key} = [storage] {storage_key} was accepted"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_vault_and_pool_d_buckets_differ(name):
    """The vault is a dedicated bucket (§1.2)."""
    why = refusal(raw_with(name, table={**TABLE_A, "pool_d_bucket": TABLE_A["vault_bucket"]}))
    assert why is not None, "pool_d_bucket equal to vault_bucket was accepted"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_unknown_key_refused(name):
    why = refusal(raw_with(name, table={**TABLE_A, "vault_buckt": "cascade-lane-d-vault"}))
    assert why is not None, "unknown [lane_d_ops] key vault_buckt was accepted"
    assert "vault_buckt" in why, why


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.parametrize("key", SECRET_KEYS)
def test_w2b_secret_shaped_key_refused(name, key):
    """Names and locations only: a credential-shaped key is refused as such
    (credentials are env-only), whatever its value."""
    for value in ("s3cr3tVALUEw2b", "", 1):
        why = refusal(raw_with(name, table={**TABLE_A, key: value}))
        assert why is not None, f"secret-shaped [lane_d_ops] {key} was accepted"
        assert key in why, why
        low = why.lower()
        assert "secret" in low or "credential" in low, (
            f"[lane_d_ops] {key} refused without naming it a secret/credential: {why}")
        assert "s3cr3tVALUEw2b" not in why, f"the refusal echoes the secret value: {why}"
