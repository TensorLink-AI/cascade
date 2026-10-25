"""DEC-CA-0055 emission rules, pure: the lane-A ladder, the former-king court,
the key slash, the carried-king cap beside lane B, and the vote-gated
pre-launch burn."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cascade.eval.expert_lane import LaneBWin
from cascade.shared.config import load_chain_config, parse_lane_b
from cascade.shared.emission import (
    COURT_BLOCKS,
    EmissionPolicy,
    LadderState,
    court_active,
    emission_policy,
    emission_vector,
    ladder_step,
    lane_parts,
)
from cascade.shared.lane_b import combined_weights

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def typed():
    cfg = load_chain_config(ROOT / "chain.toml")
    era = cfg.round.epoch_blocks * cfg.round.era_settlements
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate),
                  training=replace(cfg.training, v2_from_block=gate),
                  lane_b=replace(cfg.lane_b, from_block=gate + 1))
    return cfg, gate, era


def test_court_is_five_days_of_blocks():
    assert COURT_BLOCKS == 36000


def test_ladder_transitions():
    assert ladder_step(None, crowned=False, check=None) == 5000      # new v2 king: 50 %
    assert ladder_step(None, crowned=False, check="pass") == 6000    # +10 %
    assert ladder_step(9000, crowned=False, check="pass") == 9500    # capped at 95 %
    assert ladder_step(9500, crowned=False, check="pass") == 9500
    assert ladder_step(8000, crowned=False, check="fail") == 1000    # drops TO 10 %
    assert ladder_step(1000, crowned=False, check="pass") == 2000    # then climbs
    assert ladder_step(7000, crowned=False, check="deferred") == 7000
    assert ladder_step(9500, crowned=True, check=None) == 5000       # dethrone: new king 50 %


def test_policy_regimes(typed):
    cfg, gate, era = typed
    mid = gate + 3 * era
    # carried king (crowned before v2): the 5 % cap, no ladder
    assert emission_policy(cfg, boundary=mid, king_since_block=gate - era) == \
        EmissionPolicy(lane_a_cap=0.05)
    # crowned at a v2 settlement: the ladder (unmoved = 50 %)
    pol = emission_policy(cfg, boundary=mid, king_since_block=gate + era)
    assert pol == EmissionPolicy(king_share=0.5)
    pol = emission_policy(cfg, boundary=mid, king_since_block=gate + era,
                          ladder=LadderState(king_share_bp=7000))
    assert pol.king_share == pytest.approx(0.7) and pol.former_share == 0.0
    # ladder off ⇒ the DEC-CA-0050 court vector once crowned
    off = replace(cfg, v2=replace(cfg.v2, king_ladder=False))
    assert emission_policy(off, boundary=mid, king_since_block=gate + era).inert


def test_court_expires_at_36000_blocks(typed):
    cfg, gate, era = typed
    dethrone = gate + 2 * era
    lad = LadderState(court_hotkey="old", court_until_block=dethrone + COURT_BLOCKS)
    assert court_active(lad, dethrone + COURT_BLOCKS - 1)
    assert not court_active(lad, dethrone + COURT_BLOCKS)
    on = emission_policy(cfg, boundary=dethrone, king_since_block=dethrone, ladder=lad)
    assert on.former_share == pytest.approx(0.05)
    after = emission_policy(cfg, boundary=dethrone + COURT_BLOCKS, king_since_block=dethrone,
                            ladder=lad)
    assert after.former_share == 0.0


def test_key_slash_pays_the_king_one_percent(typed):
    cfg, gate, era = typed
    mid = gate + 3 * era
    lad = LadderState(king_share_bp=9000, slashed=True)
    pol = emission_policy(cfg, boundary=mid, king_since_block=gate + era, ladder=lad)
    assert pol.king_share == pytest.approx(0.01)
    # the carried king: capped AND slashed
    pol = emission_policy(cfg, boundary=mid, king_since_block=gate - era, ladder=lad)
    assert pol.lane_a_cap == 0.05 and pol.king_slash == pytest.approx(0.01)


def test_ladder_vector_pays_lane_a_only_and_burns_the_rest(typed):
    cfg, gate, era = typed
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "share": 0.5})
    # uid 3 is the king AND holds a lane-B domain win; uid 4 is the former king
    wins = [LaneBWin(uid=3, domain="energy", multiplier=1.0)]
    a, b = lane_parts([3, 4], 8, king_decay=0.5, burn_uid=0, lane_b_cfg=lb, wins=wins)
    assert sum(a) == pytest.approx(0.5) and sum(b) == pytest.approx(0.5)
    pol = EmissionPolicy(king_share=0.6, former_share=0.05)
    vec = emission_vector(a, b, pol, burn_uid=0, lane_a_uids=[3, 4])
    assert vec[3] == pytest.approx(0.6 * 0.5 + 0.25)      # ladder share + untouched lane B
    assert vec[4] == pytest.approx(0.05 * 0.5)
    assert vec[0] == pytest.approx(0.5 - 0.325 + 0.25)    # unpaid lane A + empty web slot
    assert sum(vec) == pytest.approx(1.0)
    # the lanes kept apart sum to the DEC-CA-0050 combined vector
    from cascade.shared.lane_b import DomainWin, LaneBState

    st = LaneBState()
    st.record(DomainWin(domain="energy", hotkey="k", uid=3, round_id="r", merge_block=0, target_lcb=0.03, tier=0.02, full_eras=5))
    legacy = combined_weights([3, 4], 8, king_decay=0.5, burn_uid=0, state=st, block=5,
                              era_blocks=era, cfg=lb, uid_of={"k": 3})
    assert [x + y for x, y in zip(a, b, strict=True)] == pytest.approx(legacy)


def test_carried_cap_leaves_lane_b_pay_of_a_court_uid_untouched():
    lb = parse_lane_b({"from_block": 1, "domains": ["energy"], "share": 0.5})
    wins = [LaneBWin(uid=2, domain="energy", multiplier=1.0)]
    a, b = lane_parts([1, 2], 4, king_decay=1.0, burn_uid=0, lane_b_cfg=lb, wins=wins)
    vec = emission_vector(a, b, EmissionPolicy(lane_a_cap=0.05), burn_uid=0, lane_a_uids=[1, 2])
    assert vec[1] == pytest.approx(0.025) and vec[2] == pytest.approx(0.025 + 0.5)
    assert vec[0] == pytest.approx(0.45) and sum(vec) == pytest.approx(1.0)


def test_prelaunch_burn_needs_the_vote(typed):
    cfg, gate, era = typed
    pre = gate - 5 * era
    assert cfg.v2.prelaunch_burn == 0.95 and cfg.v2.prelaunch_burn_from_block == 0
    assert emission_policy(cfg, boundary=pre, king_since_block=None).inert
    voted = emission_policy(cfg, boundary=pre, king_since_block=None,
                            prelaunch_from_block=pre - era)
    assert voted == EmissionPolicy(prelaunch_burn=0.95)
    assert emission_policy(cfg, boundary=pre, king_since_block=None,
                           prelaunch_from_block=pre + era).inert       # not yet
    # it ends at the first v2 settlement
    assert emission_policy(cfg, boundary=gate + era, king_since_block=None,
                           prelaunch_from_block=pre).prelaunch_burn == 0.0
    vec = emission_vector([0.0, 0.7, 0.3], None, voted, burn_uid=0, lane_a_uids=[1, 2])
    assert vec == pytest.approx([0.95, 0.035, 0.015])


def test_record_roundtrip():
    pol = EmissionPolicy(king_share=0.6, former_share=0.05)
    assert EmissionPolicy.from_record(pol.as_record()) == pol
