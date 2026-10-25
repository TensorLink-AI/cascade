"""DEC-CA-0056 shared lane-C rules: the slot state, payouts with the lane-B
decay + TTL, and a second pool prefix (lane A's keys byte-identical).
MarketCast-only lane C (owner 2026-10-20): every target is an ``mkt:`` one."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared import hippius
from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.lane_c import (
    LaneCState,
    TargetWin,
    lane_c_pay_record,
    lane_c_payouts,
    load_state,
    save_state_atomic,
)

LC = parse_lane_c({})
ERA = 7200
MC_LIVE = SimpleNamespace(active=lambda block: True)    # [marketcast] armed


def _win(t, hk, block, rid="r", uid=7):
    return TargetWin(target=t, hotkey=hk, uid=uid, round_id=rid, win_block=block, gain_lcb=0.03, tier=0.02, full_eras=5)


def test_slot_rule_on_one_king_and_history():
    # same base pointer: an equal LCB never displaces (the earliest holds, in
    # any fold order); a strictly higher LCB does; every win enters history
    st = LaneCState()
    st.record(_win("mkt:returns", "alice", 10 * ERA, "a"))
    st.record(_win("mkt:returns", "bob", 12 * ERA, "b"))
    st.record(_win("mkt:returns", "carol", 11 * ERA, "c"))    # out-of-order catch-up
    assert st.slots["mkt:returns"].hotkey == "alice"
    assert [w.hotkey for w in st.history] == ["alice", "carol", "bob"]
    assert st.record(replace(_win("mkt:returns", "dave", 13 * ERA, "d"), gain_lcb=0.05))
    assert st.slots["mkt:returns"].hotkey == "dave"


def test_state_round_trip_and_torn_file(tmp_path):
    st = LaneCState()
    st.record(_win("mkt:vol", "alice", ERA))
    st.mark_applied("r1", ERA)
    p = tmp_path / "s.lane_c.json"
    save_state_atomic(st, p)
    back = load_state(p)
    assert back.to_json() == st.to_json() and back.last_round == "r1"
    assert load_state(tmp_path / "absent.json").slots == {}
    p.write_text("{torn", encoding="utf-8")
    assert load_state(p) is None
    with pytest.raises(ValueError):
        LaneCState.from_json("[]")


def test_payout_schedule_and_ttl():
    st = LaneCState()
    st.record(_win("mkt:returns", "alice", 0, uid=7))
    st.record(_win("mkt:alpha", "x", 0, uid=8))     # not a target: never paid
    mult = []
    for age in (0, 4, 5, 6, 8, 27, 28):
        pay = lane_c_payouts(st, block=age * ERA, era_blocks=ERA, cfg=LC, n_uids=16)
        mult.append(pay[0]["multiplier"] if pay else None)
        assert all(p["target"] == "mkt:returns" for p in pay)
    assert mult == [1.0, 1.0, 0.5, 0.25, 0.1, 0.1, None]       # floor, then TTL burns it
    # deregistered / out-of-range holders are unpaid
    assert lane_c_payouts(st, block=0, era_blocks=ERA, cfg=LC, n_uids=16, uid_of={}) == []
    assert lane_c_payouts(st, block=0, era_blocks=ERA, cfg=LC, n_uids=5) == []
    rec = lane_c_pay_record(st, block=0, era_blocks=ERA, cfg=LC, n_uids=16, mc=MC_LIVE)
    assert rec["share"] == 0.25 and rec["slots"] == 2 and len(rec["wins"]) == 1
    # MarketCast off: the mkt: targets are listed but not active — no slot, no pay
    off = lane_c_pay_record(st, block=0, era_blocks=ERA, cfg=LC, n_uids=16)
    assert off["slots"] == 0 and off["wins"] == []


def test_pool_prefix_keeps_lane_a_keys_and_adds_a_second_pool():
    assert hippius.pool_index_key() == "pool/index.json" == hippius.POOL_INDEX_KEY
    assert hippius.pool_snapshot_key(7200) == "pool/snapshots/block-7200.tar"
    assert hippius.pool_index_key("pool_mkt") == "pool_mkt/index.json"
    assert hippius.pool_snapshot_key(7200, "pool_mkt") == "pool_mkt/snapshots/block-7200.tar"

    class _S:
        def __init__(self):
            self.o = {}

        def put_bytes(self, k, b, content_type=None):
            self.o[k] = b

        def put_text(self, k, t, content_type=None):
            self.o[k] = t

        def get_text(self, k):
            if k not in self.o:
                raise hippius.ObjectNotFound(k)
            return self.o[k]

    s = _S()
    hippius.publish_pool_snapshot(s, b"a" * 10, effective_block=7200, as_of="x", n_series=1,
                                  context_length=1, horizon=1)
    hippius.publish_pool_snapshot(s, b"b" * 10, effective_block=7200, as_of="x", n_series=2,
                                  context_length=1, horizon=1, prefix="pool_mkt")
    assert set(s.o) == {"pool/index.json", "pool/snapshots/block-7200.tar",
                        "pool_mkt/index.json", "pool_mkt/snapshots/block-7200.tar"}
    assert hippius.read_pool_index(s)[0].n_series == 1
    assert hippius.read_pool_index(s, "pool_mkt")[0].n_series == 2
