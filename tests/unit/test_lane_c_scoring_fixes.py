"""DEC-CA-0056 scoring hardening: the slot rule (strictly higher LCB on the
same base, any win on a newer base, order-independent). MarketCast-only lane C
(owner 2026-10-20): the generic king-vs-fine-tune flow (Bonferroni over a
target's pool_c entries, the cluster floor, its pending chain) is retired;
MarketCast's own: tests/unit/test_marketcast_validator.py."""

from __future__ import annotations

import itertools

from cascade.shared.lane_c import LaneCState, TargetWin, win_beats
from tests.unit.test_lane_c_validator import EB, KING_PTR, _ptr

T = "mkt:returns"
KING2_PTR = _ptr("king2")


# ── the slot rule (pure) ─────────────────────────────────────────────────────


def _w(hk, block, lcb, base=KING_PTR, rid=None):
    return TargetWin(target=T, hotkey=hk, uid=7, round_id=rid or str(block), win_block=block,
                     base_pointer=base, gain_lcb=lcb)


def test_two_percent_qualifier_cannot_displace_a_twenty_percent_holder():
    st = LaneCState()
    assert st.record(_w("dave", 10 * EB, 0.20))
    assert not st.record(_w("alice", 11 * EB, 0.02))
    assert st.slots[T].hotkey == "dave" and st.slots[T].win_block == 10 * EB


def test_higher_lcb_qualifier_displaces_and_restarts_the_clock():
    st = LaneCState()
    st.record(_w("dave", 10 * EB, 0.20))
    assert st.record(_w("alice", 11 * EB, 0.21))
    assert st.slots[T].hotkey == "alice" and st.slots[T].win_block == 11 * EB


def test_same_recipe_rerun_with_lower_or_equal_lcb_keeps_the_clock():
    st = LaneCState()
    st.record(_w("dave", 10 * EB, 0.20))
    for lcb in (0.19, 0.20):            # the holder's own hotkey: no special case
        assert not st.record(_w("dave", 12 * EB, lcb))
        assert st.slots[T].win_block == 10 * EB
    assert st.record(_w("dave", 13 * EB, 0.2001))
    assert st.slots[T].win_block == 13 * EB


def test_old_king_holder_is_replaced_by_any_qualifier_on_the_new_king():
    st = LaneCState()
    st.record(_w("dave", 10 * EB, 0.20))
    assert st.record(_w("alice", 11 * EB, 0.011, base=KING2_PTR))
    assert st.slots[T].hotkey == "alice"
    # the old-king win arriving late (catch-up) does not come back
    assert not win_beats(_w("dave", 10 * EB, 0.20), st.slots[T])


def test_slot_rule_is_order_independent():
    wins = [_w("a", 10 * EB, 0.05), _w("b", 11 * EB, 0.20), _w("c", 12 * EB, 0.20),
            _w("d", 13 * EB, 0.03, base=KING2_PTR), _w("e", 14 * EB, 0.04, base=KING2_PTR),
            _w("f", 15 * EB, 0.04, base=KING2_PTR)]
    holders = set()
    for perm in itertools.permutations(wins):
        st = LaneCState()
        for w in perm:
            st.record(w)
        holders.add(st.slots[T].hotkey)
    assert holders == {"e"}             # newest king, highest LCB, earliest on the tie
