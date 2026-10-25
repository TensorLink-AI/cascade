"""Lane C review fixes (2026-10-19, DEC-CA-0056 §Review fixes): L5 one leg
per commitment (``spent``; an operator-fault requeue still retrains), L6
the recipe tree's byte identity, L8 lane C never without lane B (one rule, both functions), L9 slot
count = targets ACTIVE at the settlement. (The L6 identity pass over
non-MarketCast entries went with the generic targets — MarketCast-only lane
C, owner 2026-10-20; MarketCast keeps its own ``mkt_seen`` dedup — and so
did L7's era-fit guard: a MarketCast leg's pinned base never moves.)"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.funding.queue import OPERATOR_FAULT_CLASS
from cascade.shared.chain import Commitment
from cascade.shared.config import parse_lane_b
from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.config.marketcast import parse_marketcast
from cascade.shared.emission import lane_parts
from cascade.shared.lane_c import (
    LaneCState,
    LaneCWin,
    TargetWin,
    active_targets,
    lane_c_pay_record,
    lane_c_shares,
    lane_c_vector,
)
from cascade.shared.lane_c_manifest import format_finetune_commit, tree_digest
from cascade.trainer.lane_c_driver import LaneCDriver

from .test_lane_c_trainer_driver import FROM, KING, REF, UID, Clock, E, FakeOps, _cfg, _fund
from .test_lane_c_trainer_driver import _ptr as ptr

M, V = "mkt:returns", "mkt:vol"

# ── L5: one leg per commitment ───────────────────────────────────────────────


def test_a_trained_commitment_is_spent_but_an_operator_requeue_retrains(tmp_path):
    clock = Clock()
    ops = FakeOps(tmp_path, clock)
    drv = LaneCDriver(_cfg(), ops, tmp_path / "d.json")
    _fund(ops, "ALFA", E)
    assert [leg["hotkey"] for leg in drv.tick(FROM + 10, 10, KING, FROM)] == ["ALFA"]
    assert drv.settle(round_id="1", block=FROM + 900, era_index=10, base_pointer=KING)
    assert drv.st.spent == [f"ALFA|{E}|{REF['ALFA']}|100"]
    # re-funding the SAME commitment would keep its old reveal block at the front
    ops.q.add_lane_c("ALFA", E, REF["ALFA"], 100)
    assert drv.tick(FROM + 920, 10, KING, FROM) == []
    e = ops.q.get("ALFA")
    assert (e.status, e.last_error_class) == ("failed", "spent")
    # an operator-fault requeue (the leg was lost on OUR side) still retrains
    ops.q.add_lane_c("ALFA", E, REF["ALFA"], 100)
    ops.q.requeue("ALFA", error="harvest upload failed", error_class=OPERATOR_FAULT_CLASS,
                  burn_attempt=False)
    clock.t += 3600
    assert [leg["hotkey"] for leg in drv.tick(FROM + 930, 10, KING, FROM)] == ["ALFA"]
    assert ops.q.get("ALFA").attempts == 0                        # uncharged
    # a re-commit (a new reveal) is a new commitment
    drv.settle(round_id="2", block=FROM + 1800, era_index=10, base_pointer=KING)
    ops.q.add_lane_c("ALFA", E, REF["ALFA"], 300)
    ops.commits.append(Commitment(uid=UID["ALFA"], hotkey="ALFA", coldkey=None,
                                  payload=format_finetune_commit(E, REF["ALFA"],
                                                                 mkt_suites=("returns", "vol")),
                                  commit_block=300))
    assert [leg["hotkey"] for leg in drv.tick(FROM + 1810, 10, KING, FROM)] == ["ALFA"]


def test_spent_is_not_persisted_while_empty(tmp_path):
    import json

    drv = LaneCDriver(_cfg(), FakeOps(tmp_path, Clock()), tmp_path / "d.json")
    drv.save()
    assert "spent" not in json.loads((tmp_path / "d.json").read_text())


# ── L6: the recipe tree's byte identity ─────────────────────────────────────


def test_tree_digest_is_byte_identity(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d / "pkg").mkdir(parents=True)
        (tmp_path / d / "recipe.json").write_text('{"lr_scale": 0.5}')
        (tmp_path / d / "pkg" / "gen.py").write_text("x = 1\n")
        (tmp_path / d / ".git").mkdir()
        (tmp_path / d / ".git" / "HEAD").write_text(d)                # ignored
    assert tree_digest(tmp_path / "a") == tree_digest(tmp_path / "b")
    (tmp_path / "b" / "pkg" / "gen.py").write_text("x = 2\n")
    assert tree_digest(tmp_path / "a") != tree_digest(tmp_path / "b")


def test_check_lane_c_accepts_a_duplicate_row(cfg):
    from cascade.audit.checks import check_lane_c

    lc = replace(parse_lane_c({"targets": [M]}), from_block=7200)
    c = replace(cfg, lane_c=lc)
    row = {"hotkey": "BRAV", "uid": 2, "target": M, "stage": "duplicate", "passed": False,
           "reason": "exact duplicate", "gain_lcb": None, "gain": None, "n": 0,
           "public_lcbs": {}, "trained_pointer": ptr("x"), "train_block": 1, "phase": "judge",
           "k": 1, "alpha": lc.alpha, "clusters": 0, "outcome": "", "base_pointer": KING}
    rc = SimpleNamespace(status="scored", epoch_start_block=7201 + 7200, round_id="r",
                         lane_c=[row, {"error": "lane_c_duplicate_evicted"}], lane_c_pay=None)
    assert check_lane_c(rc, c).status == "PASS", check_lane_c(rc, c).detail


# ── L8: lane C never without lane B ─────────────────────────────────────────

LB = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "share": 0.5})
LC = replace(parse_lane_c({"targets": [M, V]}), from_block=7200)


def test_lane_c_without_lane_b_is_refused_by_both_functions():
    with pytest.raises(ValueError, match="without lane B"):
        lane_parts([3], 8, king_decay=0.5, burn_uid=0, lane_c_cfg=LC, c_wins=())
    with pytest.raises(ValueError, match="without lane B"):
        lane_c_shares(None, LC, 10**6)
    assert lane_c_shares(None, LC, 7200) == (0.0, 0.0)            # inactive: no lane C
    a, b = lane_parts([3], 8, king_decay=0.5, burn_uid=0, lane_b_cfg=LB, lane_c_cfg=LC)
    assert sum(a) == pytest.approx(0.5) and sum(b) == pytest.approx(0.5)
    assert lane_c_shares(LB, LC, 7201) == (0.25, 0.25)


# ── L9: slots = targets ACTIVE at the settlement ─────────────────────────────


def test_inactive_mkt_targets_never_hold_a_slot(cfg):
    """MarketCast-only lane C: while [marketcast] is not active at a
    settlement its mkt: targets count no slot (lane C's share burns); once it
    is, each is one slot of share / n."""
    mc = parse_marketcast({"from_block": 3 * 7200})
    lc = replace(parse_lane_c({"targets": [M, V]}), from_block=7200)
    assert active_targets(lc, 2 * 7200 + 1, mc) == ()
    assert active_targets(lc, 3 * 7200, mc) == (M, V)
    assert active_targets(lc, 3 * 7200, None) == ()
    st = LaneCState(slots={M: TargetWin(M, "ALFA", 1, "r", 7200, recipe_ref=REF["ALFA"],
                                        gain_lcb=0.03, tier=0.02, full_eras=5)})
    rec = lane_c_pay_record(st, block=2 * 7200 + 1, era_blocks=7200, cfg=lc, n_uids=8, mc=mc)
    assert rec["slots"] == 0 and rec["wins"] == []
    v = lane_c_vector(8, cfg=lc, burn_uid=0, slot_frac=rec["slot_frac"], wins=[])
    assert v[0] == pytest.approx(lc.share) and sum(v) == pytest.approx(lc.share)
    live = lane_c_pay_record(st, block=3 * 7200, era_blocks=7200, cfg=lc, n_uids=8, mc=mc)
    assert live["slots"] == 2 and [w["target"] for w in live["wins"]] == [M]
    a, b = lane_parts([3], 8, king_decay=0.5, burn_uid=0, lane_b_cfg=LB, lane_c_cfg=lc,
                      c_wins=[LaneCWin(1, M, 1.0)], c_slot_frac=live["slot_frac"])
    assert b[1] == pytest.approx(lc.share / 2)
    # the audit counts the same way
    from cascade.audit.checks import _lane_c_paid_wins
    from cascade.shared.era import era_length_blocks

    c = replace(cfg, lane_c=lc, marketcast=mc)
    for blk in (2 * 7200 + 1, int(live["block"])):
        rec = lane_c_pay_record(st, block=blk, era_blocks=era_length_blocks(c.round, blk),
                                cfg=lc, n_uids=8, mc=mc)
        rc = SimpleNamespace(epoch_start_block=blk, lane_c_pay=rec)
        assert _lane_c_paid_wins(rc, c, 8)[1] is None
    assert "slots" in str(_lane_c_paid_wins(SimpleNamespace(
        epoch_start_block=blk, lane_c_pay={**rec, "slots": 3}), c, 8)[1])
