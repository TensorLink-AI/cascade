"""W8: exact weight vectors with lane D (plan rev6 §1.4, D1/D7/D9/D21/D48,
B8, invariants 1, 2 and 9, T13).

* Unarmed vectors are byte-identical to today's on recorded receipts and the
  existing emission goldens (values recorded on main a6204c0, copied from
  tests/unit/receipt_fixture.py, test_dec55_emission.py and
  test_lane_c_emission.py).
* Armed: lane D is 0.25 of the total, taken from lane A's pot; its budget
  never varies with the lane A ladder (D48); the court is KING_D plus 2 prior
  miner kings at decay 0.5 (D7); the house slot burns to ``burn_uid`` (D9,
  D21); a deregistered court member's slot burns (B8); an unregistered king
  burns. Lanes B and C off and on (A 25 / B 25 / C 25 / D 25).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from .helpers import (
    ABS,
    BURN_UID,
    KING_D_UID,
    LANE_A_COURT,
    N_UIDS,
    PRIOR_1_UID,
    PRIOR_2_UID,
    REL,
    REPO,
    as_bytes,
    lane_b_cfg,
    lane_c_cfg,
    lane_d_cfg,
    normalised,
    paid_uids,
    parts,
    vector,
)

# ── recorded goldens (unarmed, today's code) ────────────────────────────────

# lane_parts([3, 4], 8, king_decay=0.5, burn_uid=0) under EmissionPolicy()
GOLDEN_PLAIN = [0.0, 0.0, 0.0, 0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0]
# test_dec55_emission::test_ladder_vector_pays_lane_a_only_and_burns_the_rest
GOLDEN_DEC55_LADDER = [0.425, 0.0, 0.0, 0.55, 0.025, 0.0, 0.0, 0.0]
# test_dec55_emission::test_carried_cap_leaves_lane_b_pay_of_a_court_uid_untouched
GOLDEN_DEC55_CAP = [0.45, 0.025, 0.525, 0.0]
# test_lane_c_emission::test_three_lane_split_is_a50_b25_c25 (both sources)
GOLDEN_THREE_LANE = [0.0, 0.3333333333333333, 0.16666666666666666,
                     0.125, 0.125, 0.125, 0.125, 0.0]
# test_lane_c_emission::test_ladder_reshapes_lane_a_only_and_lanes_b_c_ride_through
GOLDEN_LADDER_C = [0.6625, 0.25, 0.025, 0.0, 0.0, 0.0625, 0.0, 0.0]
# tests/unit/receipt_fixture.make_scored_receipt(None).weights
GOLDEN_RECEIPT = (0.5, 0.5, 0.0, 0.0)


def _unarmed_cases(d_kw):
    """Today's golden scenarios, each built with the extra lane D keywords
    ``d_kw`` (``{}`` = today's call; ``lane_d_cfg=None`` = unarmed)."""
    from cascade.eval.expert_lane import LaneBWin
    from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts
    from cascade.shared.lane_c import LaneCWin

    out = {}
    a, _ = lane_parts([3, 4], 8, king_decay=0.5, burn_uid=0, **d_kw)
    out["plain"] = (emission_vector(a, None, EmissionPolicy(), burn_uid=0,
                                    lane_a_uids=[3, 4]), GOLDEN_PLAIN)
    lb = lane_b_cfg(0.5)
    a, b = lane_parts([3, 4], 8, king_decay=0.5, burn_uid=0, lane_b_cfg=lb,
                      wins=[LaneBWin(uid=3, domain="energy", multiplier=1.0)], **d_kw)
    out["dec55_ladder"] = (emission_vector(a, b, EmissionPolicy(king_share=0.6,
                                                                former_share=0.05),
                                           burn_uid=0, lane_a_uids=[3, 4]),
                           GOLDEN_DEC55_LADDER)
    from cascade.shared.config import parse_lane_b

    lb1 = parse_lane_b({"from_block": 1, "domains": ["energy"], "share": 0.5})
    a, b = lane_parts([1, 2], 4, king_decay=1.0, burn_uid=0, lane_b_cfg=lb1,
                      wins=[LaneBWin(uid=2, domain="energy", multiplier=1.0)], **d_kw)
    out["dec55_cap"] = (emission_vector(a, b, EmissionPolicy(lane_a_cap=0.05), burn_uid=0,
                                        lane_a_uids=[1, 2]), GOLDEN_DEC55_CAP)
    for source in ("lane_a", "lane_b"):
        lbs = lane_b_cfg(0.25 if source == "lane_a" else 0.5)
        lc = lane_c_cfg(takes_from=source)
        a, b = lane_parts([1, 2], 8, king_decay=0.5, burn_uid=0, lane_b_cfg=lbs,
                          wins=[LaneBWin(uid=3, domain="energy", multiplier=1.0),
                                LaneBWin(uid=4, domain="web", multiplier=1.0)],
                          lane_c_cfg=lc,
                          c_wins=[LaneCWin(uid=5, target="mkt:returns", multiplier=1.0),
                                  LaneCWin(uid=6, target="mkt:vol", multiplier=1.0)],
                          **d_kw)
        out[f"three_lane_{source}"] = (emission_vector(a, b, EmissionPolicy(), burn_uid=0,
                                                       lane_a_uids=[1, 2]),
                                       GOLDEN_THREE_LANE)
    a, b = lane_parts([1, 2], 8, king_decay=0.5, burn_uid=0, lane_b_cfg=lb, wins=[],
                      lane_c_cfg=lane_c_cfg(takes_from="lane_b"),
                      c_wins=[LaneCWin(uid=5, target="mkt:returns", multiplier=0.5)], **d_kw)
    out["ladder_c"] = (emission_vector(a, b, EmissionPolicy(king_share=0.5, former_share=0.05),
                                       burn_uid=0, lane_a_uids=[1, 2]), GOLDEN_LADDER_C)
    return out


def _receipt_vector(d_kw):
    from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts  # noqa: I001
    from tests.unit.receipt_fixture import make_scored_receipt

    receipt, _, _ = make_scored_receipt(None)
    assert tuple(receipt.weights) == GOLDEN_RECEIPT
    a, _ = lane_parts(list(receipt.reward_uids), len(receipt.weights), king_decay=1.0,
                      burn_uid=0, **d_kw)
    vec = emission_vector(a, None, EmissionPolicy(), burn_uid=0,
                          lane_a_uids=list(receipt.reward_uids))
    return receipt, a, vec


def _close(got, want):
    assert len(got) == len(want)
    for u, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == pytest.approx(w, rel=REL, abs=ABS), f"uid {u}: {g!r} != {w!r} in {got!r}"


# ── invariant 1: unarmed => byte-identical ──────────────────────────────────


def test_inv1_weights_unchanged_when_unarmed():
    for d_kw in ({}, {"lane_d_cfg": None},
                 {"lane_d_cfg": None, "d_court_uids": (KING_D_UID, PRIOR_1_UID)}):
        for name, (vec, golden) in _unarmed_cases(d_kw).items():
            assert as_bytes(vec) == as_bytes(golden), (name, d_kw, vec)
        receipt, a, vec = _receipt_vector(d_kw)
        assert as_bytes(a) == as_bytes(receipt.weights)
        assert as_bytes(vec) == as_bytes(receipt.weights)


def test_inv1_weights_armed_nonempty_lane_d_effect():
    a, b = parts(d_cfg=lane_d_cfg(), d_court=(KING_D_UID,))
    vec = vector(a, b)
    assert as_bytes(vec) != as_bytes(GOLDEN_PLAIN)
    _close(vec, [0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.25, 0.0])
    _, a_r, _ = _receipt_vector({"lane_d_cfg": lane_d_cfg(), "d_court_uids": (2,)})
    assert sum(a_r) == pytest.approx(0.75, rel=REL)


# ── armed goldens: court decay, house burn, B8, unregistered king ───────────

S = 0.25
K, P1, P2 = S * 4 / 7, S * 2 / 7, S * 1 / 7      # decay 0.5 over 3 slots

ARMED_NO_BC = {
    # name: (d_court, expected vector)       lane A: 0.75 x [2/3, 1/3] on [3, 4]
    "sole_king": ((KING_D_UID,), [0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.25, 0.0]),
    "king_and_two_priors": ((KING_D_UID, PRIOR_1_UID, PRIOR_2_UID),
                            [0.0, 0.0, 0.0, 0.5, 0.25, P1, K, P2]),
    "house_holds_king_d": ((None, PRIOR_1_UID, PRIOR_2_UID),          # D9/D21
                           [K, 0.0, 0.0, 0.5, 0.25, P1, 0.0, P2]),
    "house_only": ((None,), [S, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0]),  # T1
    "deregistered_prior": ((KING_D_UID, None, PRIOR_2_UID),           # B8
                           [P1, 0.0, 0.0, 0.5, 0.25, 0.0, K, P2]),
    "deregistered_king": ((None, PRIOR_1_UID),                        # B8
                          [S * 2 / 3, 0.0, 0.0, 0.5, 0.25, S / 3, 0.0, 0.0]),
    "unregistered_king_uid": ((99, PRIOR_1_UID, PRIOR_2_UID),         # no such uid
                              [K, 0.0, 0.0, 0.5, 0.25, P1, 0.0, P2]),
    "empty_court": ((), [S, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0]),
}

# lanes B and C on (25 % each, C from A, fully paid): lane A 0.25 x [2/3, 1/3];
# B: energy -> 1, web -> 2; C: returns -> 5, vol -> 7.
_A3, _A4, _Q = 0.25 * 2 / 3, 0.25 / 3, 0.125
ARMED_BC = {
    "sole_king": ((KING_D_UID,), [0.0, _Q, _Q, _A3, _A4, _Q, 0.25, _Q]),
    "king_and_two_priors": ((KING_D_UID, PRIOR_1_UID, PRIOR_2_UID),
                            [0.0, _Q, _Q, _A3, _A4, _Q + P1, K, _Q + P2]),
    "house_holds_king_d": ((None, PRIOR_1_UID, PRIOR_2_UID),
                           [K, _Q, _Q, _A3, _A4, _Q + P1, 0.0, _Q + P2]),
    "deregistered_prior": ((KING_D_UID, None, PRIOR_2_UID),
                           [P1, _Q, _Q, _A3, _A4, _Q, K, _Q + P2]),
    "house_only": ((None,), [S, _Q, _Q, _A3, _A4, _Q, 0.0, _Q]),
}


@pytest.mark.parametrize("name", sorted(ARMED_NO_BC))
def test_armed_vector_golden_lanes_b_c_off(name):
    d_court, want = ARMED_NO_BC[name]
    a, b = parts(d_cfg=lane_d_cfg(), d_court=d_court)
    assert sum(a) == pytest.approx(0.75, rel=REL)           # D1/D48: from lane A's pot
    assert sum(b) == pytest.approx(0.25, rel=REL)           # lane D's part
    vec = vector(a, b)
    _close(vec, want)
    assert sum(vec) == pytest.approx(1.0, rel=REL)


@pytest.mark.parametrize("name", sorted(ARMED_BC))
def test_armed_vector_golden_lanes_b_c_on(name):
    d_court, want = ARMED_BC[name]
    a, b = parts(d_cfg=lane_d_cfg(), d_court=d_court, lanes_bc=True)
    assert sum(a) == pytest.approx(0.25, rel=REL)           # A 25 / B 25 / C 25 / D 25
    vec = vector(a, b)
    _close(vec, want)
    assert sum(vec) == pytest.approx(1.0, rel=REL)


def test_armed_lane_d_share_from_lane_a_when_lane_b_refunds_lane_a():
    """Lane B's unpaid part under ``remainder = "lane_a"`` goes back to lane A;
    lane D's 0.25 is unchanged."""
    from cascade.shared.emission import lane_parts

    lb = lane_b_cfg(0.25, remainder="lane_a")
    a, b = lane_parts(list(LANE_A_COURT), N_UIDS, king_decay=0.5, burn_uid=BURN_UID,
                      lane_b_cfg=lb, wins=[], lane_d_cfg=lane_d_cfg(),
                      d_court_uids=(KING_D_UID,))
    _close(vector(a, b), [0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.25, 0.0])


def test_house_and_dead_slots_burn_to_burn_uid_only():
    """D21 (decided: burn): every unpaid lane D slot lands on ``burn_uid``,
    never on a lane A, B or C uid."""
    _, b_house = parts(d_cfg=lane_d_cfg(), d_court=(None, PRIOR_1_UID))
    _, b_paid = parts(d_cfg=lane_d_cfg(), d_court=(KING_D_UID, PRIOR_1_UID))
    diff = [x - y for x, y in zip(b_house, b_paid, strict=True)]
    _close(diff, [S * 2 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, -S * 2 / 3, 0.0])


# ── D48: lane D's budget never varies with the lane A ladder ────────────────


@pytest.mark.parametrize("lanes_bc, pool_a", [(False, 0.75), (True, 0.25)])
def test_lane_d_budget_fixed_across_ladder_shares(lanes_bc, pool_a):
    from cascade.shared.emission import EmissionPolicy

    a, b = parts(d_cfg=lane_d_cfg(), d_court=(KING_D_UID, PRIOR_1_UID, PRIOR_2_UID),
                 lanes_bc=lanes_bc)
    d_part = None
    for share in (0.10, 0.80, 0.95):
        vec = vector(a, b, EmissionPolicy(king_share=share, former_share=0.05))
        assert vec[3] == pytest.approx(share * pool_a, rel=REL)
        assert vec[4] == pytest.approx(0.05 * pool_a, rel=REL)
        assert vec[KING_D_UID] == pytest.approx(K, rel=REL)
        part = (vec[KING_D_UID], vec[PRIOR_1_UID], vec[PRIOR_2_UID])
        if d_part is None:
            d_part = part
        assert as_bytes(part) == as_bytes(d_part), f"lane D moved with ladder share {share}"
        assert sum(vec) == pytest.approx(1.0, rel=REL)
    for policy in (EmissionPolicy(lane_a_cap=0.05), EmissionPolicy(king_slash=0.01)):
        vec = vector(a, b, policy)
        assert as_bytes((vec[KING_D_UID], vec[PRIOR_1_UID], vec[PRIOR_2_UID])) == \
            as_bytes(d_part)


# ── invariant 2 / T13: lane A verdicts and court unchanged ──────────────────


def _typed_cfg():
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(REPO / "chain.toml")
    cfg = replace(cfg, v2=replace(cfg.v2, prelaunch_burn_from_block=0))
    era = cfg.round.epoch_blocks * cfg.round.era_settlements
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate),
                  training=replace(cfg.training, v2_from_block=gate))
    return cfg, gate, era


def _policies(cfg, gate, era):
    from cascade.shared.emission import LadderState, emission_policy

    mid = gate + 3 * era
    return [
        emission_policy(cfg, boundary=mid, king_since_block=gate - era),
        emission_policy(cfg, boundary=mid, king_since_block=gate + era),
        emission_policy(cfg, boundary=mid, king_since_block=gate + era,
                        ladder=LadderState(king_share_bp=7000)),
        emission_policy(cfg, boundary=mid, king_since_block=gate + era,
                        ladder=LadderState(king_share_bp=9000, slashed=True)),
        emission_policy(cfg, boundary=gate - 5 * era, king_since_block=None,
                        prelaunch_from_block=gate - 6 * era),
    ]


def test_inv2_t13_lane_a_court_unchanged_when_unarmed():
    cfg, gate, era = _typed_cfg()
    unarmed = replace(cfg, lane_d=lane_d_cfg(from_block=0))
    assert _policies(unarmed, gate, era) == _policies(cfg, gate, era)
    for lanes_bc in (False, True):
        a0, b0 = parts(lanes_bc=lanes_bc)
        a1, b1 = parts(lanes_bc=lanes_bc, d_cfg=None, d_court=(KING_D_UID,))
        assert as_bytes(a1) == as_bytes(a0) and as_bytes(b1) == as_bytes(b0)


def test_inv2_t13_lane_a_court_armed_nonempty_lane_d_effect():
    cfg, gate, era = _typed_cfg()
    armed = replace(cfg, lane_d=lane_d_cfg(from_block=gate - 2 * era))
    # lane A's verdict-side policy (ladder, cap, slash, pre-launch burn) is unchanged
    assert _policies(armed, gate, era) == _policies(cfg, gate, era)
    for lanes_bc, pool_a in ((False, 0.75), (True, 0.25)):
        a0, _ = parts(lanes_bc=lanes_bc)
        a1, b1 = parts(lanes_bc=lanes_bc, d_cfg=lane_d_cfg(), d_court=(KING_D_UID,))
        # the court and its relative split are identical
        assert paid_uids(a1) == paid_uids(a0) == list(LANE_A_COURT)
        _close(normalised(a1), normalised(a0))
        # lane A's share-of-total changes by design (D48), and lane D pays
        assert sum(a1) == pytest.approx(pool_a, rel=REL)
        assert sum(a0) == pytest.approx(pool_a + 0.25, rel=REL)
        assert vector(a1, b1)[KING_D_UID] == pytest.approx(0.25, rel=REL)


# ── invariant 9: armed, share = 0 => identical vector ───────────────────────


def test_inv9_share_zero_vector_unchanged_when_unarmed():
    for lanes_bc in (False, True):
        a0, b0 = parts(lanes_bc=lanes_bc)
        unarmed = vector(a0, b0 if lanes_bc else None)
        for court in ((KING_D_UID,), (None, PRIOR_1_UID, PRIOR_2_UID), ()):
            az, bz = parts(lanes_bc=lanes_bc, d_cfg=lane_d_cfg(share=0.0), d_court=court)
            assert as_bytes(az) == as_bytes(a0)
            assert as_bytes(vector(az, bz)) == as_bytes(unarmed), (lanes_bc, court)
    a0, _ = parts()
    assert as_bytes(vector(a0, None)) == as_bytes(GOLDEN_PLAIN)
