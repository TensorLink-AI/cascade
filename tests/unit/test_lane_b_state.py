"""DEC-CA-0050: lane-B consensus state, [lane_b] config, combined weights."""

from __future__ import annotations

import pytest

from cascade.shared.chain import decayed_share_vector
from cascade.shared.config import LaneBConfig, parse_lane_b
from cascade.shared.lane_b import DomainWin, LaneBState, combined_weights

CFG = parse_lane_b({"from_block": 1000, "domains": ["energy", "web", "nature"], "share": 0.5})


def _win(domain, uid, block, hk=None):
    return DomainWin(domain=domain, hotkey=hk or f"hk{uid}", uid=uid, round_id="r",
                     merge_block=block, target_lcb=0.03, tier=0.02, full_eras=5)


def test_config_defaults_are_inert():
    c = parse_lane_b({})
    assert c == LaneBConfig() and not c.active(10**9)
    assert CFG.active(1000) and not CFG.active(999)


@pytest.mark.parametrize("bad", [
    {"share": 1.5}, {"floor": 0.0}, {"remainder": "x"}, {"domains": ["a", "a"]},
    {"from_block": 5}, {"min_experts": 40, "max_experts": 38},
])
def test_config_guards(bad):
    with pytest.raises(ValueError):
        parse_lane_b(bad)


def test_inactive_is_exactly_lane_a():
    st = LaneBState()
    st.record(_win("energy", 5, 900))
    v = combined_weights([1, 2], 10, king_decay=0.5, burn_uid=0, state=st, block=999,
                         era_blocks=100, cfg=CFG)
    assert v == decayed_share_vector([1, 2], 10, decay=0.5, burn_uid=0)


def test_decay_schedule_and_burn():
    st = LaneBState()
    st.record(_win("energy", 5, 1000))
    slot = 0.5 / 3
    # ages in ERAS (era_blocks = 100 here); full 5 eras, half-life 1, floor 0.1,
    # paid until win_ttl_eras (28) — then the slot burns
    for eras, mult in [(0, 1.0), (4, 1.0), (5, 0.5), (6, 0.25), (7, 0.125), (8, 0.1),
                       (20, 0.1), (27, 0.1), (28, 0.0), (40, 0.0)]:
        v = combined_weights([1], 10, king_decay=0.5, burn_uid=0, state=st,
                             block=1000 + eras * 100, era_blocks=100, cfg=CFG)
        assert v[5] == pytest.approx(slot * mult)
        assert v[1] == pytest.approx(0.5)
        assert v[0] == pytest.approx(0.5 - slot * mult)
        assert sum(v) == pytest.approx(1.0)


def test_newer_win_replaces_slot_and_older_win_is_history_only():
    st = LaneBState()
    st.record(_win("web", 3, 1000))
    st.record(_win("web", 4, 1500))
    assert st.wins["web"].uid == 4
    # an out-of-order (catch-up / on-demand) older win never takes the slot,
    # and the folded state is order-independent
    st.record(_win("web", 6, 1200))
    assert st.wins["web"].uid == 4
    other = LaneBState()
    for w in (_win("web", 6, 1200), _win("web", 4, 1500), _win("web", 3, 1000)):
        other.record(w)
    assert other.wins == st.wins and other.history == st.history


def test_deregistered_winner_is_not_paid():
    st = LaneBState()
    st.record(_win("energy", 5, 1000, hk="alice"))
    v = combined_weights([1], 10, king_decay=1.0, burn_uid=0, state=st, block=1000,
                         era_blocks=100, cfg=CFG, uid_of={"bob": 5})
    assert v[5] == 0.0 and v[0] == pytest.approx(0.5)


def test_json_round_trip():
    st = LaneBState()
    st.record(_win("energy", 5, 1000))
    st.mark_applied("r1", 1000)
    back = LaneBState.from_json(st.to_json())
    assert back == st and back.settled_block == 1000 and back.last_round == "r1"
    # a torn/empty file is an ERROR (the validator rebuilds), never silent-empty
    for torn in ("", st.to_json()[:20]):
        with pytest.raises(ValueError):
            LaneBState.from_json(torn)


def test_ttl_expiry_drops_the_payout_and_keeps_the_win():
    """win_ttl_eras: payout stops at that age (the record and its merge
    provenance stay — the merged weights are in the king forever); 0 = never."""
    from cascade.shared.lane_b import lane_b_payouts

    st = LaneBState()
    st.record(_win("energy", 5, 1000))
    live = lane_b_payouts(st, block=1000 + 27 * 100, era_blocks=100, cfg=CFG, n_uids=10)
    assert [p["age"] for p in live] == [27]
    assert lane_b_payouts(st, block=1000 + 28 * 100, era_blocks=100, cfg=CFG, n_uids=10) == []
    assert st.wins["energy"].uid == 5 and st.history
    forever = parse_lane_b({"from_block": 1000, "domains": ["energy"], "win_ttl_eras": 0})
    assert lane_b_payouts(st, block=10**9, era_blocks=100, cfg=forever, n_uids=10)


@pytest.mark.parametrize("bad", [{"win_ttl_eras": -1}, {"win_ttl_eras": 5},
                                 {"win_ttl_eras": 3, "full_pay_tiers": [[0.05, 5]]}])
def test_ttl_guard(bad):
    with pytest.raises(ValueError):
        parse_lane_b(bad)


def test_replay_settlements_covers_the_ttl():
    """Unset ⇒ (ttl + 1) × era_settlements, so a rebuilt validator holds every
    still-paying win; an explicit value below the cover is refused."""
    assert parse_lane_b({}, era_settlements=4).replay_settlements == 29 * 4
    assert parse_lane_b({"replay_settlements": 200}, era_settlements=4).replay_settlements == 200
    with pytest.raises(ValueError):
        parse_lane_b({"replay_settlements": 64}, era_settlements=4)
    # no TTL ⇒ no finite cover: the legacy default
    assert parse_lane_b({"win_ttl_eras": 0}, era_settlements=4).replay_settlements == 64
