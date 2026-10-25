"""DEC-CA-0056 emission: the 3-lane split (lane C taken from lane B), inert
before ``[lane_c] from_block`` (bit-identical vectors and records), and the
interplay with the DEC-CA-0055 lane-A ladder / cap."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.eval.expert_lane import LaneBWin
from cascade.shared.config import parse_lane_b
from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts
from cascade.shared.lane_b import DomainWin
from cascade.shared.lane_c import LaneCWin, TargetWin, lane_c_shares, lane_c_vector
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis

LB = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "share": 0.5})
SUITES = ("returns", "vol")
LC = replace(parse_lane_c({"targets": ["mkt:returns", "mkt:vol"]}, mkt_suites=SUITES),
             from_block=7200)
BURN = 0


def test_shares():
    assert lane_c_shares(LB, LC, 7200) == (0.5, 0.0)       # the settlement AT the block
    assert lane_c_shares(LB, LC, 7201) == (0.25, 0.25)
    assert lane_c_shares(LB, replace(LC, from_block=0), 10**9) == (0.5, 0.0)
    assert lane_c_shares(LB, replace(LC, share=0.9), 10**9) == (0.0, 0.5)   # capped at B's


def test_future_confirmed_slot_cannot_be_paid_on_an_earlier_boundary():
    from cascade.shared.lane_c import LaneCState, lane_c_payouts

    win = TargetWin("mkt:returns", "miner", 5, "future", 9000,
                    gain_lcb=.02, tier=.02, full_eras=5)
    state = LaneCState(slots={win.target: win})
    assert lane_c_payouts(state, block=8000, era_blocks=100, cfg=LC, n_uids=8) == []
    assert lane_c_payouts(state, block=9000, era_blocks=100, cfg=LC, n_uids=8)[0]["age"] == 0


def test_audit_rejects_future_dated_lane_c_reward(cfg):
    from types import SimpleNamespace

    from cascade.audit.checks import _lane_c_paid_wins
    from cascade.shared.era import era_length_blocks
    from cascade.shared.lane_c import LaneCState, lane_c_pay_record

    cfg = replace(cfg, lane_c=LC, marketcast=replace(cfg.marketcast, from_block=1))
    block = 8000
    pay = lane_c_pay_record(LaneCState(), block=block,
                           era_blocks=era_length_blocks(cfg.round, block), cfg=LC,
                           n_uids=8, mc=cfg.marketcast)
    pay["wins"] = [{"target": "mkt:returns", "uid": 5, "hotkey": "miner",
                    "win_block": 9000, "age": 0, "multiplier": 1.,
                    "gain_lcb": .02, "tier": .02, "full_eras": 5, "below_floor": False}]
    _, reason = _lane_c_paid_wins(SimpleNamespace(lane_c_pay=pay,
                                                 epoch_start_block=block), cfg, 8)
    assert "after the payout" in reason


def test_lane_c_vector_pays_slots_and_burns_the_rest():
    v = lane_c_vector(8, cfg=LC, wins=[LaneCWin(uid=5, target="mkt:returns", multiplier=1.0)],
                      burn_uid=BURN)
    assert v[5] == pytest.approx(0.125) and v[BURN] == pytest.approx(0.125)
    with pytest.raises(ValueError):
        lane_c_vector(8, cfg=LC, burn_uid=BURN,
                      wins=[LaneCWin(5, "mkt:returns", 1.0), LaneCWin(6, "mkt:returns", 1.0)])


@pytest.mark.parametrize("source", ["lane_a", "lane_b"])
def test_three_lane_split_is_a50_b25_c25(source):
    lb = replace(LB, share=0.25) if source == "lane_a" else LB
    lc = replace(LC, takes_from=source)
    b_wins = [LaneBWin(uid=3, domain="energy", multiplier=1.0),
              LaneBWin(uid=4, domain="web", multiplier=1.0)]
    c_wins = [LaneCWin(uid=5, target="mkt:returns", multiplier=1.0),
              LaneCWin(uid=6, target="mkt:vol", multiplier=1.0)]
    a, b = lane_parts([1, 2], 8, king_decay=0.5, burn_uid=BURN, lane_b_cfg=lb, wins=b_wins,
                      lane_c_cfg=lc, c_wins=c_wins)
    vec = emission_vector(a, b, EmissionPolicy(), burn_uid=BURN, lane_a_uids=[1, 2])
    assert sum(a) == pytest.approx(0.5)                     # lane A's pool untouched
    assert vec[3] == pytest.approx(0.125) and vec[4] == pytest.approx(0.125)
    assert vec[5] == pytest.approx(0.125) and vec[6] == pytest.approx(0.125)
    assert vec[BURN] == pytest.approx(0.0, abs=1e-12) and sum(vec) == pytest.approx(1.0)


def test_c_from_a_preserves_b_allocation():
    lb, lc = replace(LB, share=0.25), replace(LC, takes_from="lane_a")
    assert lane_c_shares(lb, lc, 7200) == (0.25, 0.0)
    assert lane_c_shares(lb, lc, 7201) == (0.25, 0.25)
    before = lane_parts([1], 8, king_decay=0.5, burn_uid=BURN, lane_b_cfg=lb)
    after = lane_parts([1], 8, king_decay=0.5, burn_uid=BURN, lane_b_cfg=lb, lane_c_cfg=lc)
    assert sum(before[0]) == pytest.approx(0.75)
    assert sum(after[0]) == pytest.approx(0.50)


def test_without_lane_c_lane_parts_is_bit_identical():
    wins = [LaneBWin(uid=3, domain="energy", multiplier=0.25)]
    for court in ([], [1], [1, 2, 6]):
        old = lane_parts(court, 8, king_decay=0.5, burn_uid=BURN, lane_b_cfg=LB, wins=wins)
        new = lane_parts(court, 8, king_decay=0.5, burn_uid=BURN, lane_b_cfg=LB, wins=wins,
                         lane_c_cfg=None, c_wins=[LaneCWin(9, "x", 1.0)])
        assert old == new


def test_ladder_reshapes_lane_a_only_and_lanes_b_c_ride_through():
    c_wins = [LaneCWin(uid=5, target="mkt:returns", multiplier=0.5)]
    a, b = lane_parts([1, 2], 8, king_decay=0.5, burn_uid=BURN, lane_b_cfg=LB, wins=[],
                      lane_c_cfg=LC, c_wins=c_wins)
    pol = EmissionPolicy(king_share=0.5, former_share=0.05)
    vec = emission_vector(a, b, pol, burn_uid=BURN, lane_a_uids=[1, 2])
    assert vec[1] == pytest.approx(0.25) and vec[2] == pytest.approx(0.025)   # of A's 0.5
    assert vec[5] == pytest.approx(0.125 * 0.5)
    assert sum(vec) == pytest.approx(1.0)
    capped = emission_vector(a, b, EmissionPolicy(lane_a_cap=0.05), burn_uid=BURN,
                             lane_a_uids=[1, 2])
    assert capped[1] + capped[2] == pytest.approx(0.05) and capped[5] == vec[5]
    burned = emission_vector(a, b, EmissionPolicy(prelaunch_burn=0.95), burn_uid=BURN,
                             lane_a_uids=[1, 2])
    assert burned[5] == pytest.approx(vec[5] * 0.05)


# ── the validator's push ─────────────────────────────────────────────────────

KING = "5King"


class _Client:
    def __init__(self, block, uids):
        self.block, self.uids = block, uids
        self.vectors: list[list[float]] = []
        self.equal: list[list[int]] = []

    def current_block(self):
        return self.block

    def n_uids(self):
        return 16

    def uid_for_hotkey(self, hk):
        return self.uids.get(hk)

    def set_equal_share_weights(self, reward_uids, n_uids, *, decay, burn_uid):
        self.equal.append(list(reward_uids))

    def set_weight_vector(self, vec):
        self.vectors.append(list(vec))


def _runner(cfg, tmp_path, lc, *, mkt_from: int = 0):
    v = replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    mc = replace(cfg.marketcast, from_block=int(mkt_from))
    r = ValidatorRunner(cfg=replace(cfg, lane_b=LB, lane_c=lc, validator=v, marketcast=mc),
                        state=genesis(KING, 3))
    r._lane_b().record(DomainWin(domain="energy", hotkey="alice", uid=9, round_id="r",
                                 merge_block=7200, target_lcb=0.03, tier=0.02, full_eras=5))
    return r


@pytest.mark.parametrize("lc", [parse_lane_c({}), replace(LC, from_block=10**9)])
def test_inert_lane_c_leaves_the_push_bit_identical(cfg, tmp_path, lc):
    base = _runner(cfg, tmp_path / "a", parse_lane_c({}))
    other = _runner(cfg, tmp_path / "b", lc)
    for r in (base, other):
        r._lane_b_block = 14400
    ca, cb = _Client(14400, {KING: 3, "alice": 9}), _Client(14400, {KING: 3, "alice": 9})
    va = base._apply_weights(ca, "r", [3])
    vb = other._apply_weights(cb, "r", [3])
    assert va == vb and ca.vectors == cb.vectors
    assert base._last_lane_b_pay == other._last_lane_b_pay and "share" not in base._last_lane_b_pay
    assert other._last_lane_c_pay is None and other._last_emission is None
    assert other._lane_c_receipt_kw(None) == {"lane_c": None, "lane_c_pay": None}


def test_active_lane_c_push_and_records(cfg, tmp_path):
    r = _runner(cfg, tmp_path, LC, mkt_from=7200)
    r._lane_c().record(TargetWin(target="mkt:vol", hotkey="bob", uid=4, round_id="r",
                                 win_block=14400, gain_lcb=0.03, tier=0.02, full_eras=5))
    r._lane_b_block = r._lane_c_block = 14400
    c = _Client(14400, {KING: 3, "alice": 9, "bob": 4})
    vec = r._apply_weights(c, "r", [3])
    assert c.vectors == [list(vec)] and c.equal == []
    assert vec[3] == pytest.approx(0.5)                 # lane A
    assert vec[9] == pytest.approx(0.125)               # lane B: 0.25 / 2 domains
    assert vec[4] == pytest.approx(0.125)               # lane C: 0.25 / 2 targets
    assert vec[cfg.scoring.burn_uid] == pytest.approx(0.25)
    assert r._last_lane_b_pay["share"] == 0.25 and r._last_lane_b_pay["slots"] == 2
    assert r._last_lane_c_pay["wins"][0]["target"] == "mkt:vol"
    assert r._last_emission is None                     # no DEC-CA-0055 rule shaped it
    kw = r._lane_c_receipt_kw(None)
    assert kw["lane_c"] == [] and kw["lane_c_pay"] == r._last_lane_c_pay


def test_lane_c_on_marketcast_off_burns_lane_c_share(cfg, tmp_path):
    """MarketCast-only lane C (owner 2026-10-20): lane C armed while
    [marketcast] from_block = 0 ⇒ its mkt: targets are listed but not active
    ⇒ zero slots, lane C's whole share burns and the vector sums to 1."""
    r = _runner(cfg, tmp_path, LC, mkt_from=0)
    r._lane_c().record(TargetWin(target="mkt:vol", hotkey="bob", uid=4, round_id="r",
                                 win_block=14400, gain_lcb=0.03, tier=0.02, full_eras=5))
    r._lane_b_block = r._lane_c_block = 14400
    c = _Client(14400, {KING: 3, "alice": 9, "bob": 4})
    vec = r._apply_weights(c, "r", [3])
    assert r._last_lane_c_pay["slots"] == 0 and r._last_lane_c_pay["wins"] == []
    assert vec[4] == 0.0                                # no lane-C holder is paid
    assert vec[3] == pytest.approx(0.5) and vec[9] == pytest.approx(0.125)
    assert vec[cfg.scoring.burn_uid] == pytest.approx(0.125 + 0.25)   # lane B's empty slot + C
    assert sum(vec) == pytest.approx(1.0)
