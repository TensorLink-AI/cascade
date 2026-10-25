"""Supported policy tuning must reject parameters that break the scorer or scheduler."""

import pytest

from cascade.shared.config import parse_lane_b
from cascade.shared.netconfig import NetConfigError, next_overlay, validate_overlay

from .test_netconfig_core import E2, base_cfg


@pytest.mark.parametrize("key,value", [
    ("alpha", 0), ("alpha", 1.1), ("bootstrap_B", 0),
    ("tau_domain", -1), ("tau_public", float("inf")),
    ("half_life_rounds", float("nan")), ("esft_p", 0),
    ("min_public_items", 0), ("min_guard_windows", 0),
    ("leg_hours", 0), ("leg_hours", float("nan")),
    ("affinity_windows", 0), ("open_tokens", -1),
    ("require_public", ["unknown"]), ("require_public", ["gift", "gift"]),
])
def test_lane_b_refuses_unusable_runtime_policy(key, value):
    with pytest.raises(ValueError):
        parse_lane_b({key: value})


def test_signed_overlay_uses_the_same_bounds():
    with pytest.raises(NetConfigError, match="loader"):
        validate_overlay(base_cfg(), next_overlay({"lane_b": {"alpha": 2.}}, E2, None, network={"netuid": 91, "chain": "bittensor:finney"}))
