"""W-LAUNCH overlay sync + live reward summary (owner 2026-11-25).

The published step 1 overlay (seq 1, cfg-961291bc) carries ``[v2]
reign_warm_start_blocks = 0`` (lane A shuts down: no king checks, uid 164's
checkpoint stays the frozen reference for lanes B and D). The repo copies of
both launch steps, and the next-era fallback step 2, must restate it, or the
R3 ``non-lane-d`` lint refuses step 2 and main would not match what was
signed. The generated ``cascade:rewards:summary`` block states the LIVE launch
split computed from these overlays, not the checked-in template.
"""

from __future__ import annotations

import sys

import pytest

from .helpers import ROOT, bodies, overrides

STEP1_DIGEST = "961291bc"


@pytest.mark.parametrize("rel", [
    "deploy/overlays/launch-20261125.pre.mainnet.toml",
    "deploy/overlays/launch-20261125.mainnet.toml",
    "deploy/overlays/lane-d-step2.mainnet.toml",
])
def test_mainnet_launch_overlays_freeze_lane_a(rel):
    assert overrides(rel)["v2"] == {"reign_warm_start_blocks": 0}


def test_step1_body_is_the_published_one():
    from cascade.shared import netconfig as nc

    _, b1, _, _ = bodies("mainnet")
    assert nc.body_digest(b1).startswith(STEP1_DIGEST)


def test_reward_summary_states_the_live_launch_split():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_docs

    summary = check_docs.reward_reference()["summary"]
    assert f"| Step 1 (overlay cfg-{STEP1_DIGEST}, lanes B and C) | 50% | 35% | 15% | 0% |" in summary
    assert "| Step 2 (second overlay, lane D armed) | 0% | 35% | 15% | 50% |" in summary
    assert "Lane A is closed to new challengers from block 9,586,800" in summary
    assert "each pay 3.5% of total emissions" in summary
    assert "no lane B slot is paid under the launch config" in summary
    assert "The embargo is 24 h at launch" in summary
