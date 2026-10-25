"""MarketCast slot eviction after the review of 2026-10-25: the fold replays
evictions in the order judge time applied them (expiry / deregistration
BEFORE offers, identity-dedup AFTER), a failed registration lookup holds the
settlement as ``LaneCTransient`` (never an unclassified exception), a client
without the historical-registration hook is loud, and an expired holder is
evicted on a quiet settlement too."""
from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.chain import ChainError
from cascade.shared.era import era_length_blocks
from cascade.shared.lane_c import LaneCState, TargetWin
from cascade.validator.lane_c import LaneCTransient
from cascade.validator.marketcast import MktPlan, fold_mkt_plan

from . import test_marketcast_validator as T

TARGET = "mkt:returns"


def _win(hk, ptr, lcb, block=1000):
    return TargetWin(TARGET, hk, 1, "r", block, base_pointer="base", gain_lcb=lcb, tier=0.05,
                     full_eras=10, trained_pointer=ptr)


def _holder(block, *, lcb=0.99, hotkey="former", uid=5):
    return TargetWin(TARGET, hotkey, uid, "old", block, base_pointer=T.MC.base_pointer,
                     gain_lcb=lcb, tier=0.05, full_eras=10, trained_pointer=T._ptr(hotkey))


def _judge_time(st, offered, evict_before, evict_after):
    """What ``_mkt_settle`` does on its scratch copy."""
    work = LaneCState(slots=dict(st.slots))
    for t, hk, ptr in evict_before:
        cur = work.slots.get(t)
        if cur is not None and (cur.hotkey, cur.trained_pointer) == (hk, ptr):
            del work.slots[t]
    for w in offered:
        work.record(w)
    for t, hk, ptr in evict_after:
        cur = work.slots.get(t)
        if cur is not None and (cur.hotkey, cur.trained_pointer) == (hk, ptr):
            del work.slots[t]
    return work


# ── fix 2: the fold mirrors judge time ───────────────────────────────────────


def test_fold_evicts_a_same_settlement_confirm_superseded_by_a_senior_commit():
    st = LaneCState()
    junior = _win("junior", "ptr-j", 0.5)                      # confirmed this settlement
    superseded = [(TARGET, "junior", "ptr-j")]                 # ... and an earlier commit of
    plan = MktPlan(offered=[junior], evict_after=superseded)   # the same weights arrived
    work = _judge_time(st, [junior], [], superseded)
    assert TARGET not in work.slots
    took = fold_mkt_plan(st, plan)
    assert TARGET not in st.slots                              # the fold agrees: no holder
    assert [w.hotkey for w in took] == ["junior"]              # the win is history, unpaid
    assert [w.hotkey for w in st.history] == ["junior"]


def test_fold_still_evicts_an_ineligible_holder_before_offers():
    old = _win("former", "ptr-f", 0.99, block=0)
    st = LaneCState(slots={TARGET: old})
    weaker = _win("alice", "ptr-a", 0.3)
    plan = MktPlan(offered=[weaker], evict=[(TARGET, "former", "ptr-f")])
    assert _judge_time(st, [weaker], plan.evict, []).slots[TARGET].hotkey == "alice"
    fold_mkt_plan(st, plan)
    assert st.slots[TARGET].hotkey == "alice"


def test_fold_of_a_plan_without_the_new_field_is_unchanged():
    st = LaneCState()
    plan = MktPlan(offered=[_win("alice", "ptr-a", 0.3)])
    del plan.__dict__["evict_after"]                           # an older in-flight plan shape
    assert [w.hotkey for w in fold_mkt_plan(st, plan)] == ["alice"]


def test_settlement_pays_no_holder_its_rows_say_was_evicted(cfg, tmp_path):
    """End to end: alice's confirm resolves in round 1003 while a SENIOR
    commit of her exact weights (an earlier block, another hotkey) registers
    in the same lane-C manifest. Judge time labels alice evicted; the live
    state must agree (no holder, nobody paid)."""
    s1, s2 = T._snapshot(tmp_path, 5000, "s1"), T._snapshot(tmp_path, 9000, "s2")
    store = T._Store()
    es = [T._entry("alice"), T._entry("carol")]
    senior = T._entry("dave", train_block=T.BLOCK - 50,
                      ckpt=dict(es[0].marketcast)["checkpoint_sha256"])
    r = T._runner(cfg, tmp_path / "v", store, lambda b: s1 if b < T.BLOCK + T.EB else s2,
                  history=T._history(es + [senior], blocks={"dave": T.BLOCK - 900}))
    T._rows(r, T._settle(store, "1001", es))
    T._rows(r, T._settle(store, "1002", [], prev="1001", block=T.BLOCK + T.EB,
                         reports={"1001": T._recorded(store, "1001")}))
    rows3 = T._rows(r, T._settle(store, "1003", [senior], prev="1002", block=T.BLOCK + 2 * T.EB,
                                 reports={"1002": T._recorded(store, "1002")}))
    by = {(x.get("hotkey"), x.get("phase"), (x.get("mkt") or {}).get("at", "")): x
          for x in rows3 if "error" not in x}
    # the rows: alice's confirm passed but she does not hold the slot (the
    # relabel ran on the post-eviction scratch state), and her register row
    # says why — the slot was evicted for dave's senior commit
    confirm = by[("alice", "confirm", "")]
    assert confirm["passed"] and confirm["outcome"] == "below holder LCB" and not confirm.get("holder")
    evicted = by[("alice", "judge", "register")]
    assert evicted["stage"] == "duplicate" and evicted["mkt"].get("evicted") == "slot"
    st = r._lane_c()
    assert TARGET not in st.slots                              # state = the rows
    client = SimpleNamespace(uid_for_hotkey=lambda hk: T.UIDS.get(hk))
    assert r._lane_c_pay(client, 16, T.BLOCK + 2 * T.EB)["wins"] == []


# ── fix 3: the registration lookup ───────────────────────────────────────────


def test_registration_chain_error_holds_as_lane_c_transient(cfg, tmp_path):
    calls = []

    def reg(hotkey, block):
        calls.append((hotkey, block))
        raise ChainError("metagraph_failed: State already discarded")

    def holder(r):
        r.registration_at_block_fn = reg
        r._lane_c().slots[TARGET] = _holder(T.BLOCK)

    with pytest.raises(LaneCTransient, match="registration of slot holder"):
        T._three_rounds(cfg, tmp_path, before_confirm=holder)
    assert calls == [("former", T.BLOCK + 2 * T.EB - 1)]


def test_missing_registration_hook_is_loud_once_per_settlement_and_changes_no_verdict(
        cfg, tmp_path, caplog):
    def holder(r):
        r.registration_at_block_fn = None
        r._lane_c().slots[TARGET] = _holder(T.BLOCK)

    with caplog.at_level(logging.ERROR, logger="cascade.validator"):
        f = T._three_rounds(cfg, tmp_path, before_confirm=holder)
    hits = [m for m in caplog.messages if "uid_for_hotkey_at_block" in m]
    assert sorted(m.split(":")[0] for m in hits) == [
        "marketcast round=1001", "marketcast round=1002", "marketcast round=1003"]
    # the verdict is the pre-fix one: the (deregistered-at-head) holder keeps the slot
    assert f.r._lane_c().slots[TARGET].hotkey == "former"
    rows = [x for x in f.rows[2] if x.get("hotkey") == "alice" and x.get("phase") == "confirm"]
    assert rows[-1]["outcome"] == "below holder LCB"


# ── fix 6: a quiet settlement evicts an expired holder ───────────────────────


def test_expired_holder_is_evicted_on_a_quiet_settlement(cfg, tmp_path):
    f = T._three_rounds(cfg, tmp_path)
    r, store = f.r, f.store
    assert r._lane_c().slots[TARGET].hotkey == "alice"
    r.cfg = replace(r.cfg, lane_c=replace(r.cfg.lane_c, win_ttl_eras=1))
    assert r._lane_c().mkt_backlog == [] and not r._lane_c().pending
    eb = era_length_blocks(r.cfg.round, 0)
    soon = T.BLOCK + 2 * T.EB + eb - r.cfg.round.epoch_blocks  # age 0: still the holder
    m4 = T._settle(store, "1004", [], prev="1003", block=soon)
    assert T._rows(r, m4) == []
    assert r._lane_c().slots[TARGET].hotkey == "alice"
    far = T.BLOCK + 2 * T.EB + 3 * eb                          # long expired
    m5 = T._settle(store, "1005", [], prev="1004", block=far)
    assert T._rows(r, m5) == []                                # no rows: nothing judged
    st = r._lane_c()
    assert TARGET not in st.slots and "1005" in st.applied
    assert [w.hotkey for w in st.history] == ["alice"]        # the win stays history
    client = SimpleNamespace(uid_for_hotkey=lambda hk: T.UIDS.get(hk))
    assert r._lane_c_pay(client, 16, far)["wins"] == []
