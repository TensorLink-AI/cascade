"""TESTNET-FAST profile (owner 2026-11-24): chain.testnet.toml caps a lane B
leg at 1 h (max_leg_hours 6 -> 1, max_leg_seconds 27000 -> 4500 = 1 h x the
same 1.25 guard) so lane B runs end-to-end on testnet in hours. Mainnet
(chain.toml) keeps the release caps; the GPU list is unchanged."""

from __future__ import annotations

import pytest

from .helpers import (
    TESTNET,
    TESTNET_FAST_LANE_B,
    WB10_DEFAULTS,
    field,
    file_raw,
    load_ok,
    round_skus,
    shipped_raw,
)


@pytest.mark.parametrize("key", sorted(TESTNET_FAST_LANE_B))
def test_testnet_file_carries_the_fast_lane_b_cap(key):
    default, fast = TESTNET_FAST_LANE_B[key]
    assert default == WB10_DEFAULTS[key] and default != fast, key
    assert file_raw(TESTNET)["lane_b"][key] == fast
    assert field(load_ok(file_raw(TESTNET)).lane_b, key) == fast
    assert field(load_ok(file_raw("chain.toml")).lane_b, key) == default
    assert field(load_ok(shipped_raw(TESTNET)).lane_b, key) == default


def test_testnet_fast_lane_b_keeps_the_guard_ratio_and_the_gpu_list():
    from cascade.shared.netconfig_bounds import value_refusals

    lb = load_ok(file_raw(TESTNET)).lane_b
    hours, secs = field(lb, "max_leg_hours"), field(lb, "max_leg_seconds")
    assert (hours, secs) == tuple(v for _d, v in (TESTNET_FAST_LANE_B["max_leg_hours"],
                                                   TESTNET_FAST_LANE_B["max_leg_seconds"]))
    assert secs == hours * 3600 * field(lb, "guard_factor")
    assert tuple(field(lb, "funded_pod_skus")) == round_skus(TESTNET)
    fast = {k: v for k, (_d, v) in TESTNET_FAST_LANE_B.items()}
    assert value_refusals({"lane_b": fast}) == []
