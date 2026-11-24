"""W9 (plan rev6.22, owner 2026-11-24): the audit re-derives lane B weights
under BOTH pay rules, chosen by the settlement's block.

Before ``[lane_b] indefinite_pay_from_block`` the decay + TTL rebuild is
today's; from it every held slot pays multiplier 1 with no TTL, the signed
``lane_b_pay`` record names ``pay_rule = "indefinite"`` and each win its
``since_era``. The receipt is built by the validator's own
``_lane_b_vector`` on the shipped chain.toml; the audit is
``cascade.audit.checks._check_lane_b_weights`` (the lane B branch of
``check_weights``). Nothing from PATHS is imported at module level."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
ERA = 3600                        # mainnet era: 900 x 4
V2 = 9_399_600                    # [v2] from_block (acceptance "v2" resolves to it)
GATE = V2 + 20 * ERA              # indefinite_pay_from_block (an era boundary)
MERGE = V2 + ERA                  # the holder's merge (acceptance) block
KING_UID, HOLDER_UID, N = 3, 9, 16


class _Client:
    def uid_for_hotkey(self, hk):
        return {"5King": KING_UID, "alice": HOLDER_UID}.get(hk)


def _cfg(tmp_path, gate):
    from cascade.shared.config import load_chain_config, parse_lane_b

    c = load_chain_config(REPO / "chain.toml")
    lb = parse_lane_b({"from_block": V2 + 1, "domains": ["energy", "web"], "share": 0.5,
                       "acceptance_from_block": V2, "indefinite_pay_from_block": gate,
                       "win_ttl_eras": 28})
    return replace(c, lane_b=lb, lane_c=replace(c.lane_c, from_block=0),
                   validator=replace(c.validator, state_db_path=str(tmp_path / "v.json")))


def _receipt(tmp_path, block, *, gate=GATE):
    from cascade.shared.lane_b import DomainWin
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    cfg = _cfg(tmp_path, gate)
    r = ValidatorRunner(cfg=cfg, state=genesis("5King", KING_UID))
    r._lane_b().record(DomainWin(domain="energy", hotkey="alice", uid=HOLDER_UID,
                                 round_id="r1", merge_block=MERGE, target_lcb=0.03,
                                 tier=0.02, full_eras=5))
    vec, pay = r._lane_b_vector(_Client(), [KING_UID], N, block)
    rc = SimpleNamespace(status="scored", weights=tuple(vec), reward_uids=(KING_UID,),
                         verdict=None, epoch_start_block=block, lane_b=[], lane_b_pay=pay,
                         state_post=None)
    return cfg, rc


def _audit(cfg, rc):
    from cascade.audit.checks import _check_lane_b_weights
    from cascade.shared.chain import decayed_share_vector

    lane_a = decayed_share_vector([KING_UID], len(rc.weights),
                                  decay=cfg.scoring.king_decay, burn_uid=cfg.scoring.burn_uid)
    return _check_lane_b_weights("weights", rc, cfg, lane_a)


def _with_pay(rc, pay):
    return SimpleNamespace(**{**rc.__dict__, "lane_b_pay": pay})


def test_w9_pre_gate_receipt_rebuilds_under_decay_and_ttl(tmp_path):
    block = MERGE + 10 * ERA                    # inside the TTL, past full_eras=5
    cfg, rc = _receipt(tmp_path, block)
    assert "pay_rule" not in rc.lane_b_pay
    (w,) = rc.lane_b_pay["wins"]
    assert w["multiplier"] < 1.0 and "since_era" not in w
    assert _audit(cfg, rc).status == "PASS"


def test_w9_post_gate_receipt_pays_full_past_the_ttl(tmp_path):
    block = GATE + 100 * ERA                    # 119 eras after the merge (TTL is 28)
    cfg, rc = _receipt(tmp_path, block)
    assert rc.lane_b_pay["pay_rule"] == "indefinite"
    (w,) = rc.lane_b_pay["wins"]
    assert w["multiplier"] == 1.0 and w["since_era"] == MERGE // ERA
    assert rc.weights[HOLDER_UID] == pytest.approx(0.25)     # 0.5 / 2 slots, in full
    assert rc.weights[cfg.scoring.burn_uid] == pytest.approx(0.25)   # the empty slot
    res = _audit(cfg, rc)
    assert res.status == "PASS", res


def test_w9_settlement_at_the_gate_keeps_the_old_rule(tmp_path):
    cfg, rc = _receipt(tmp_path, GATE)          # 20 eras after merge: decayed, in TTL
    assert "pay_rule" not in rc.lane_b_pay
    assert rc.lane_b_pay["wins"][0]["multiplier"] < 1.0
    assert _audit(cfg, rc).status == "PASS"
    cfg, rc = _receipt(tmp_path, GATE + 900)    # the first settlement of the gate's era
    assert rc.lane_b_pay["pay_rule"] == "indefinite"
    assert _audit(cfg, rc).status == "PASS"


@pytest.mark.parametrize("tamper", ["decayed_multiplier", "no_pay_rule", "since_era",
                                    "bool_since_era"])
def test_w9_post_gate_tampering_fails(tmp_path, tamper):
    cfg, rc = _receipt(tmp_path, GATE + 100 * ERA)
    pay = copy.deepcopy(rc.lane_b_pay)
    if tamper == "decayed_multiplier":
        pay["wins"][0]["multiplier"] = 0.1
    elif tamper == "no_pay_rule":
        del pay["pay_rule"]
    elif tamper == "since_era":
        pay["wins"][0]["since_era"] += 1
    else:
        pay["wins"][0]["since_era"] = True
    assert _audit(cfg, _with_pay(rc, pay)).status == "FAIL", tamper


def test_w9_pre_gate_record_must_not_claim_the_new_rule(tmp_path):
    cfg, rc = _receipt(tmp_path, MERGE + 10 * ERA)
    pay = copy.deepcopy(rc.lane_b_pay)
    pay["pay_rule"] = "indefinite"
    assert _audit(cfg, _with_pay(rc, pay)).status == "FAIL"
    pay = copy.deepcopy(rc.lane_b_pay)
    pay["wins"][0]["since_era"] = MERGE // ERA
    assert _audit(cfg, _with_pay(rc, pay)).status == "FAIL"


def test_w9_post_gate_vector_built_under_the_old_rule_fails(tmp_path):
    """A validator still paying decay + TTL after the gate (an old release, or
    the gate missing from its config) is caught: its expired slot burned."""
    block = GATE + 100 * ERA
    _old_cfg, old_rc = _receipt(tmp_path, block, gate=0)
    assert old_rc.lane_b_pay["wins"] == []      # past the TTL under the old rule
    cfg = _cfg(tmp_path, GATE)
    assert _audit(cfg, old_rc).status == "FAIL"


def test_w9_ungated_config_never_pays_indefinitely(tmp_path):
    cfg, rc = _receipt(tmp_path, GATE + 100 * ERA, gate=0)
    assert "pay_rule" not in rc.lane_b_pay
    assert _audit(cfg, rc).status == "PASS"
