"""W2b: ``[lane_d_ops]`` is operator-only, like ``[generation]``: never on
the DEC-CA-0057 allow-list, listed in ``overlay_policy()["operator_only"]``,
and an overlay touching it is refused whole (lane D keys included)."""

from __future__ import annotations

import pytest

from .helpers import ARM_BLOCK, OPS_KEYS, SHIPPED, TABLE_B, load_ok, shipped_raw

EFFECTIVE = 2698 * 3600


def test_w2b_not_on_the_allow_list():
    from cascade.shared import netconfig as nc

    assert "lane_d_ops" not in nc.ALLOWED, "netconfig.ALLOWED lists lane_d_ops"
    assert "lane_d" in nc.ALLOWED


def test_w2b_overlay_policy_lists_it_operator_only():
    from cascade.shared import netconfig as nc

    policy = nc.overlay_policy()
    assert "lane_d_ops" in policy["operator_only"], policy["operator_only"]
    assert "generation" in policy["operator_only"], policy["operator_only"]
    assert "lane_d_ops" not in policy["overlay_keys"]


@pytest.mark.parametrize("key", OPS_KEYS)
def test_w2b_check_overrides_refuses_every_key(key):
    from cascade.shared import netconfig as nc

    with pytest.raises(nc.NetConfigError) as e:
        nc.check_overrides({"lane_d_ops": {key: TABLE_B[key]}})
    assert "[lane_d_ops]" in str(e.value), str(e.value)


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_overlay_with_lane_d_ops_refused_whole(name):
    from cascade.shared import netconfig as nc

    overrides = {"lane_d": {"from_block": ARM_BLOCK},
                 "lane_d_ops": {"eval_host": "laned-eval"}}
    with pytest.raises(nc.NetConfigError):
        nc.check_overrides(overrides)
    base = load_ok(shipped_raw(name))
    body = {"protocol": nc.PROTOCOL, "network": nc.network_identity(base), "seq": 1,
            "prev": None, "effective_block": EFFECTIVE, "overrides": overrides}
    with pytest.raises(nc.NetConfigError):
        nc.validate_overlay(base, body)
    # the lane D half alone is a valid overlay: the refusal is the ops table's
    ok = dict(body, overrides={"lane_d": {"from_block": ARM_BLOCK}})
    assert nc.validate_overlay(base, ok).lane_d.from_block == ARM_BLOCK


@pytest.mark.parametrize("key", OPS_KEYS)
def test_w2b_dedrift_classifies_it_operator_only(key):
    """R6 de-drift: a deployed [lane_d_ops] value is an operator-only knob,
    never an overlay or a main-diff candidate."""
    from cascade.laned import dedrift

    assert dedrift.classify("lane_d_ops", key, TABLE_B[key]) == (
        "operator-only", "operator_only_table")
