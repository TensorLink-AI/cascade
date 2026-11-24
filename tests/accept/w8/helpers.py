"""Shared spec and builders for the W8 acceptance tests (lane D emission).

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W8 row, §1.4 (KING_D, the
house, pay), D1/D7/D9/D21/D48 (lane D is 0.25 of the total, taken from lane
A's pot; the court is KING_D plus 2 prior miner kings at decay 0.5; the house
is never in the court and its slot burns), B8 (a deregistered court member's
slot burns), §3.2 invariants 1, 2 and 9, and T1/T13.

No PATHS module is imported at module level: every ``cascade.*`` import is
inside a function a test body calls, so on main a test fails inside its body
(the W0c expect-fail convention). README.md states the contract.
"""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Era grid of the W1.5 fixture (mainnet: 900 x 4).
ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA

# Fixture metagraph: uid 0 burns; lane A's court is [king 3, prior king 4].
N_UIDS = 8
BURN_UID = 0
LANE_A_COURT = (3, 4)
LANE_A_KING_DECAY = 0.5

# Lane D court uids used across the goldens (KING_D first, then priors).
KING_D_UID = 6
PRIOR_1_UID = 5
PRIOR_2_UID = 7

# D1/D48: 0.25 of the total, taken from lane A. D7: 2 prior kings; rev6.23
# (owner 2026-11-24): court_weights 0.80 / 0.15 / 0.05.
ARMED = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
         "remainder": "burn", "reward_prior_kings": 2, "court_weights": [0.80, 0.15, 0.05]}

# Exact float tolerance for armed goldens (the order of float operations is
# the builder's choice; the values are exact fractions).
REL = 1e-12
ABS = 1e-15


# rev6.18 F4 (A3): these fixtures pin the live-only rule: the public half off
# (public_weight = 0, a spec setting) and no per-window ratio clamp in effect
# (a floor far below any fixture ratio), so their pre-F4 numbers stand. F4's
# own tests (tests/accept/w5e/test_w5e_combined.py) pin the combined rule.
LIVE_ONLY = {"public_weight": 0.0, "window_ratio_floor": 1e-9}


def lane_d_cfg(**over):
    """The real W2 parse of ``ARMED`` with ``over`` applied."""
    from cascade.shared.config.lane_d import parse_lane_d

    return parse_lane_d({**ARMED, **LIVE_ONLY, **over}, era_blocks_at=lambda _block: ERA)


def lane_b_cfg(share: float = 0.25, **over):
    """Two lane B slots (``energy``, ``web``) at ``share`` of the total."""
    from cascade.shared.config import parse_lane_b

    return parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "share": share,
                         **over})


def lane_c_cfg(share: float = 0.25, takes_from: str = "lane_a"):
    """Two lane C targets at ``share`` of the total, from ``takes_from``."""
    from cascade.shared.config.lane_c import parse_lane_c

    base = parse_lane_c({"targets": ["mkt:returns", "mkt:vol"]}, mkt_suites=("returns", "vol"))
    return replace(base, from_block=1, share=share, takes_from=takes_from)


def full_b_wins():
    """Both lane B slots paid in full: energy -> uid 1, web -> uid 2."""
    from cascade.eval.expert_lane import LaneBWin

    return [LaneBWin(uid=1, domain="energy", multiplier=1.0),
            LaneBWin(uid=2, domain="web", multiplier=1.0)]


def full_c_wins():
    """Both lane C targets paid in full: returns -> uid 5, vol -> uid 7."""
    from cascade.shared.lane_c import LaneCWin

    return [LaneCWin(uid=5, target="mkt:returns", multiplier=1.0),
            LaneCWin(uid=7, target="mkt:vol", multiplier=1.0)]


def parts(*, d_cfg=None, d_court=(), lanes_bc: bool = False, court=LANE_A_COURT,
          n_uids: int = N_UIDS):
    """``lane_parts`` for the fixture: lane A's court, optionally lanes B and
    C at 25 % each (C from A, both fully paid), optionally lane D."""
    from cascade.shared.emission import lane_parts

    kw = {}
    if lanes_bc:
        kw.update(lane_b_cfg=lane_b_cfg(), wins=full_b_wins(), lane_c_cfg=lane_c_cfg(),
                  c_wins=full_c_wins())
    if d_cfg is not None:
        kw.update(lane_d_cfg=d_cfg, d_court_uids=tuple(d_court))
    return lane_parts(list(court), n_uids, king_decay=LANE_A_KING_DECAY, burn_uid=BURN_UID,
                      **kw)


def vector(a, b, policy=None, *, lane_a_uids=LANE_A_COURT):
    from cascade.shared.emission import EmissionPolicy, emission_vector

    return emission_vector(a, b, policy if policy is not None else EmissionPolicy(),
                           burn_uid=BURN_UID, lane_a_uids=list(lane_a_uids))


def as_bytes(vec) -> bytes:
    """Byte identity of a weight vector: its float64 bytes."""
    values = [float(x) for x in vec]
    return struct.pack(f"<{len(values)}d", *values)


def normalised(vec) -> list[float]:
    total = sum(vec)
    return [float(x) / total for x in vec]


def paid_uids(vec, *, burn_uid: int = BURN_UID) -> list[int]:
    return [u for u, w in enumerate(vec) if w > 0.0 and u != burn_uid]
