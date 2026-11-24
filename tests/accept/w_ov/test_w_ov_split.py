"""W-OV (plan rev6.24 (f)): the owner's A 0 / B 35 / C 15 / D 50 split ships
as a ready-to-sign overlay (deploy/overlays/), never as a repo default; the
emission code handles lane A = 0 exactly (no lane A weight, no NaN, sums to
1)."""

from __future__ import annotations

import math
import tomllib
from types import SimpleNamespace

import pytest

from .helpers import ROOT, shipped_cfg, shipped_raw

FILES = {"chain.toml": "deploy/overlays/split-a0-b35-c15-d50.mainnet.toml",
         "chain.testnet.toml": "deploy/overlays/split-a0-b35-c15-d50.testnet.toml"}


def _overrides(name: str) -> dict:
    return tomllib.loads((ROOT / FILES[name]).read_text())


@pytest.mark.parametrize("name", sorted(FILES))
def test_w_ov_split_overlay_validates_on_its_base(name):
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb

    ov = _overrides(name)
    assert nb.value_refusals(ov) == [] and nb.key_refusals(ov) == []
    base = shipped_cfg(name)
    era = int(base.round.epoch_blocks) * int(base.round.era_settlements)
    eff = (10**8 // era + 1) * era
    plan = nc.plan_overlay(base, ov, eff, None)
    assert plan.ok, plan.render()
    out = nc.build_config(base, plan.body)
    assert (out.lane_b.share, out.lane_c.share, out.lane_d.share) == (0.35, 0.15, 0.5)
    assert out.lane_c.takes_from == "lane_a" and out.lane_d.takes_from == "lane_a"
    assert nb.share_refusals(out) == []


def test_w_ov_split_repo_defaults_are_unchanged():
    """The split must not move by a repo default (lane A is live with six
    external validators; a default change forks on deploy)."""
    from cascade.shared import netconfig_bounds as nb

    raw = shipped_raw("chain.toml")
    assert nb.share_refusals(shipped_cfg()) == []
    assert (raw["lane_b"]["share"], raw["lane_c"]["share"]) == (0.25, 0.25)
    assert raw["lane_d"].get("share", 0.0) == 0.0


def test_w_ov_split_lane_a_zero_exact_vector():
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb
    from cascade.shared.emission import lane_parts
    from cascade.shared.lane_c import LaneCWin

    base = shipped_cfg()
    out = nc.build_config(base, nc.next_overlay(_overrides("chain.toml"), 10**8, None,
                                                network=nc.network_identity(base)))
    assert nb.share_refusals(out) == []
    n, burn = 12, 0
    lb, lc, ld = out.lane_b, out.lane_c, out.lane_d
    # one paid lane B slot, one paid lane C slot, a 2-member lane D court
    b_win = SimpleNamespace(uid=3, multiplier=1.0)          # lane_parts reads uid, multiplier
    c_win = LaneCWin(uid=5, target=lc.targets[0], multiplier=1.0)
    a, rest = lane_parts([1, 2], n, king_decay=0.5, burn_uid=burn,
                         lane_b_cfg=lb, wins=[b_win], lane_c_cfg=lc, c_wins=[c_win],
                         c_slot_frac={lc.targets[0]: 1.0},
                         lane_d_cfg=ld, d_court_uids=[7, 8])
    assert a == [0.0] * n                                      # no lane A weight
    vec = [x + y for x, y in zip(a, rest, strict=True)]
    assert all(math.isfinite(v) and v >= 0.0 for v in vec)
    assert math.isclose(sum(vec), 1.0, abs_tol=1e-12)
    slot_b = 0.35 / len(lb.targets or lb.domains)
    want = [0.0] * n
    want[3] = slot_b
    want[burn] += 0.35 - slot_b                                 # unpaid lane B slots burn
    want[5] = 0.15
    want[7], want[8] = 0.5 * (1 / 1.5), 0.5 * (0.5 / 1.5)        # king_decay 0.5 court
    for u in range(n):
        assert math.isclose(vec[u], want[u], abs_tol=1e-12), (u, vec[u], want[u])
    # the v2 ladder / carry-over cap re-shape lane A's (empty) pool only
    from cascade.shared.emission import EmissionPolicy, emission_vector

    for pol in (EmissionPolicy(king_share=0.8, former_share=0.1),
                EmissionPolicy(lane_a_cap=0.05), EmissionPolicy(king_slash=0.5)):
        got = emission_vector(a, rest, pol, burn_uid=burn, lane_a_uids=[1, 2])
        assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(got, vec, strict=True))
