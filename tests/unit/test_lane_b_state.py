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
    {"share": 1.5}, {"undeclared_gain_margin": 0.005}, {"remainder": "x"},
    {"domains": ["a", "a"]},
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


def test_full_slot_forever_and_burn():
    """LANE_D_PLAN rev6.22 (owner 2026-11-24): a held slot pays its full share
    every settlement, no decay, no TTL; the empty slots burn."""
    st = LaneBState()
    st.record(_win("energy", 5, 1000))
    slot = 0.5 / 3
    for eras in (0, 4, 5, 8, 27, 28, 40, 10_000):
        v = combined_weights([1], 10, king_decay=0.5, burn_uid=0, state=st,
                             block=1000 + eras * 100, era_blocks=100, cfg=CFG)
        assert v[5] == pytest.approx(slot)
        assert v[1] == pytest.approx(0.5)
        assert v[0] == pytest.approx(0.5 - slot)
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


def test_no_ttl_the_holder_is_paid_until_replaced():
    """rev6.22: no expiry; the payout row carries age and since_era, multiplier 1."""
    from cascade.shared.lane_b import lane_b_payouts

    st = LaneBState()
    st.record(_win("energy", 5, 1000))
    for eras in (27, 28, 500):
        (row,) = lane_b_payouts(st, block=1000 + eras * 100, era_blocks=100, cfg=CFG,
                                n_uids=10)
        assert (row["age"], row["multiplier"], row["since_era"]) == (eras, 1.0, 10)
    st.record(_win("energy", 7, 1000 + 600 * 100))
    (row,) = lane_b_payouts(st, block=1000 + 700 * 100, era_blocks=100, cfg=CFG, n_uids=10)
    assert row["uid"] == 7


@pytest.mark.parametrize("key", ["win_ttl_eras", "full_pay_tiers", "half_life_rounds",
                                 "floor"])
def test_retired_payout_keys_are_ignored(key):
    """rev6.22: the decay / TTL keys are retired (loader RETIRED_KEYS); a
    direct parse ignores them and the config has no such field."""
    from cascade.shared.config import RETIRED_KEYS

    assert key in RETIRED_KEYS["lane_b"]
    assert not hasattr(parse_lane_b({key: 5}), key)


def test_replay_settlements_default_and_explicit():
    """No TTL (rev6.22) means no finite cover: the legacy default 64; an
    explicit value is kept (under acceptance recovery walks the full ledger)."""
    assert parse_lane_b({}, era_settlements=4).replay_settlements == 64
    assert parse_lane_b({"replay_settlements": 200}, era_settlements=4).replay_settlements == 200