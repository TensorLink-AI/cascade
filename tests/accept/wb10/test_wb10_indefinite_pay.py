"""WB10 (plan rev6.22, owner 2026-11-24): lane B pay is indefinite.

``[lane_b] indefinite_pay_from_block`` (typed era boundary, default 0 =
never): from the era starting at it a held slot pays its FULL slot share
every settlement (multiplier 1, no decay, no TTL) until a newer accepted
merge takes the slot; empty slots still burn; one hotkey may hold several
slots (a multi-target leg). A newer leg takes a held slot only through the
merge-acceptance path (judged marginally on top of the base that already
carries the holder), so the key is refused unless ``acceptance_from_block``
is typed at or before it. Before the key the decay + TTL rule is unchanged.

Every config check goes through the real loader on the shipped files;
payouts go through ``cascade.shared.lane_b`` (the validator's own code).
Nothing from PATHS is imported at module level."""

from __future__ import annotations

from dataclasses import replace

import pytest

from .helpers import SHIPPED, load_ok, overrides_refusal, raw_with, refused

ERA_MAIN = 3600                 # 900 x 4
V2_MAIN = 9_399_600             # [v2] from_block (mainnet); acceptance "v2" = it
GATE_MAIN = V2_MAIN + 10 * ERA_MAIN
ERA_TEST = 600                  # 150 x 4
GATE_TEST = 8_400_000           # an era boundary after testnet [v2] from_block


def _lb(**kw):
    from cascade.shared.config import parse_lane_b

    base = {"from_block": 1, "domains": ["energy", "web", "sales", "nature"], "share": 0.5,
            "acceptance_from_block": 1000, "win_ttl_eras": 28}
    base.update(kw)
    return parse_lane_b(base)


# ── the key ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_indefinite_pay_ships_off(name):
    lb = load_ok(raw_with(name)).lane_b
    assert getattr(lb, "indefinite_pay_from_block", None) == 0, (
        "[lane_b] indefinite_pay_from_block must ship as 0 (never: today's decay + TTL rule)")
    assert not lb.indefinite_pay_at_settlement(GATE_MAIN + ERA_MAIN)


def test_wb10_indefinite_pay_typed_on_mainnet_loads():
    lb = load_ok(raw_with("chain.toml", {"indefinite_pay_from_block": GATE_MAIN})).lane_b
    assert lb.indefinite_pay_from_block == GATE_MAIN
    # the settlement AT the boundary belongs to the era before it
    assert not lb.indefinite_pay_at_settlement(GATE_MAIN)
    assert lb.indefinite_pay_at_settlement(GATE_MAIN + 900)
    assert not lb.indefinite_pay_at_settlement(GATE_MAIN - 900)


@pytest.mark.parametrize("value", [True, -3600, "v2", 1.5, GATE_MAIN + 900, V2_MAIN - ERA_MAIN])
def test_wb10_indefinite_pay_refuses_bad_values(value):
    assert refused(raw_with("chain.toml", {"indefinite_pay_from_block": value})), (
        f"indefinite_pay_from_block={value!r} must be refused (an integer era boundary at "
        "or after [v2] from_block)")


def test_wb10_indefinite_pay_needs_acceptance_at_or_before_it():
    # testnet ships acceptance_from_block = 0: typing the gate alone is refused
    assert refused(raw_with("chain.testnet.toml", {"indefinite_pay_from_block": GATE_TEST}))
    # acceptance after the gate: refused
    assert refused(raw_with("chain.testnet.toml", {
        "indefinite_pay_from_block": GATE_TEST,
        "acceptance_from_block": GATE_TEST + ERA_TEST}))
    lb = load_ok(raw_with("chain.testnet.toml", {
        "indefinite_pay_from_block": GATE_TEST, "acceptance_from_block": GATE_TEST})).lane_b
    assert lb.indefinite_pay_from_block == GATE_TEST
    with pytest.raises(ValueError, match="acceptance_from_block"):
        _lb(indefinite_pay_from_block=5000, acceptance_from_block=0)


def test_wb10_indefinite_pay_is_overlay_settable():
    assert overrides_refusal({"lane_b": {"indefinite_pay_from_block": GATE_MAIN}}) is None


def test_wb10_legacy_payout_keys_still_load():
    """The decay + TTL keys keep loading (they pay every settlement before the
    gate, and testnet's settled lane B history)."""
    lb = load_ok(raw_with("chain.toml", {"indefinite_pay_from_block": GATE_MAIN})).lane_b
    assert lb.win_ttl_eras == 28 and lb.full_pay_tiers and lb.floor == 0.1


# ── payouts ──────────────────────────────────────────────────────────────────


def _win(domain, hotkey, uid, merge_block, **kw):
    from cascade.shared.lane_b import DomainWin

    return DomainWin(domain=domain, hotkey=hotkey, uid=uid, round_id=str(merge_block),
                     merge_block=merge_block, target_lcb=0.03, tier=0.02, full_eras=5, **kw)


def _state(*wins):
    from cascade.shared.lane_b import LaneBState

    st = LaneBState()
    for w in wins:
        st.record(w)
    return st


ERA = 3600


def test_wb10_held_slot_pays_full_forever_after_the_gate():
    from cascade.shared.lane_b import lane_b_payouts

    gate = 100 * ERA
    lb = _lb(indefinite_pay_from_block=gate, acceptance_from_block=ERA)
    st = _state(_win("energy", "alice", 9, 10 * ERA))
    for block in (gate + 900, gate + 500 * ERA, gate + 10_000 * ERA):
        pay = lane_b_payouts(st, block=block, era_blocks=ERA, cfg=lb, n_uids=16)
        assert [(p["domain"], p["uid"], p["multiplier"]) for p in pay] == [("energy", 9, 1.0)], (
            f"at block {block} the holder must be paid its full slot (no decay, no TTL)")
        assert pay[0]["since_era"] == 10
        assert pay[0]["age"] == (block - 10 * ERA) // ERA


def test_wb10_before_the_gate_the_decay_and_ttl_rule_is_unchanged():
    from cascade.shared.lane_b import lane_b_payouts

    gate = 100 * ERA
    lb = _lb(indefinite_pay_from_block=gate, acceptance_from_block=ERA)
    off = replace(lb, indefinite_pay_from_block=0)
    st = _state(_win("energy", "alice", 9, 10 * ERA))
    for block in (10 * ERA, 16 * ERA, 30 * ERA, 60 * ERA, gate):
        got = lane_b_payouts(st, block=block, era_blocks=ERA, cfg=lb, n_uids=16)
        assert got == lane_b_payouts(st, block=block, era_blocks=ERA, cfg=off, n_uids=16)
        assert all("since_era" not in p for p in got)
    # past the TTL (28 eras) and before the gate the slot burns
    assert lane_b_payouts(st, block=60 * ERA, era_blocks=ERA, cfg=lb, n_uids=16) == []
    # the same old win is paid in full again once the gate is in force
    assert lane_b_payouts(st, block=gate + 900, era_blocks=ERA, cfg=lb,
                          n_uids=16)[0]["multiplier"] == 1.0


def test_wb10_one_leg_holds_several_slots_and_empty_slots_burn():
    from cascade.shared.lane_b import combined_weights

    gate = 100 * ERA
    lb = _lb(indefinite_pay_from_block=gate, acceptance_from_block=ERA)
    # one multi-target leg (alice) won energy and web; sales and nature empty
    st = _state(_win("energy", "alice", 9, 10 * ERA, targets=("energy", "web")),
                _win("web", "alice", 9, 10 * ERA, targets=("energy", "web")))
    vec = combined_weights([3], 16, king_decay=1.0, burn_uid=0, state=st,
                           block=gate + 200 * ERA, era_blocks=ERA, cfg=lb)
    slot = 0.5 / 4
    assert vec[9] == pytest.approx(2 * slot)
    assert vec[3] == pytest.approx(0.5)
    assert vec[0] == pytest.approx(2 * slot), "the two empty slots must burn"
    assert sum(vec) == pytest.approx(1.0)


def test_wb10_a_deregistered_holder_is_not_paid():
    from cascade.shared.lane_b import lane_b_payouts

    lb = _lb(indefinite_pay_from_block=100 * ERA, acceptance_from_block=ERA)
    st = _state(_win("energy", "alice", 9, 10 * ERA))
    assert lane_b_payouts(st, block=101 * ERA, era_blocks=ERA, cfg=lb, n_uids=16,
                          uid_of={}) == []


# ── replacement: only a newer ACCEPTED merge takes a held slot ───────────────


def test_wb10_holder_keeps_slot_until_a_newer_accepted_merge():
    from cascade.shared.lane_b import LaneBState, lane_b_payouts
    from cascade.validator.expert_lane import apply_judgements

    gate = 100 * ERA
    lb = _lb(indefinite_pay_from_block=gate, acceptance_from_block=ERA)
    st = LaneBState()
    holder = _win("energy", "alice", 9, 0)
    merge_a = {"rebased_pointer": "a@1", "king_pointer": "k@1", "selection": {"0": [1]},
               "round_id": "r1"}
    st.accept(merge_a, [holder], 10 * ERA)
    assert st.wins["energy"].hotkey == "alice"
    # a newer leg QUALIFIES (passes its own settlement judge): qualify-only,
    # the slot does not move
    assert apply_judgements(st, [], round_id="r2", settle_block=150 * ERA,
                            qualify_only=True) == []
    newer = _win("energy", "bob", 4, 0, qualified_block=150 * ERA)
    st.qualify(newer)
    pay = lane_b_payouts(st, block=160 * ERA, era_blocks=ERA, cfg=lb, n_uids=16)
    assert [(p["hotkey"], p["multiplier"]) for p in pay] == [("alice", 1.0)], (
        "a qualified (not yet accepted) newer leg must not take the slot")
    # an OLDER accepted merge (out-of-order catch-up) never displaces the holder
    st.accept({**merge_a, "rebased_pointer": "old@1"}, [_win("energy", "carol", 5, 0)],
              5 * ERA)
    assert st.wins["energy"].hotkey == "alice"
    # the newer merge passes acceptance at a later warm start: it takes the slot
    st.accept({**merge_a, "rebased_pointer": "b@1", "round_id": "r2"}, [newer], 170 * ERA)
    pay = lane_b_payouts(st, block=171 * ERA, era_blocks=ERA, cfg=lb, n_uids=16)
    assert [(p["hotkey"], p["multiplier"], p["since_era"]) for p in pay] == [("bob", 1.0, 170)]
