"""W9 (plan rev6.22, owner 2026-11-24): the audit re-derives lane B under
the slot rule.

* **Weights:** one rule — every paid holder has multiplier 1.0 (no decay, no
  TTL), the re-derived ``age`` and ``since_era``, and equals its holder in
  ``state_post``. Receipts are built by the validator's own
  ``_lane_b_vector`` on the shipped chain.toml; the audit is
  ``_check_lane_b_weights`` (the lane B branch of ``check_weights``).
* **Rows:** a judged leg has one row per slot; ``check_lane_b`` re-derives
  each row's verdict for its family (declared at ``gain_margin``, undeclared
  at ``undeclared_gain_margin``) from the recorded CRPS/MASE family LCBs,
  point gains, the other metric's non-inferiority LCB and the guards, and
  binds one critical value per family. Rows come from the validator's
  ``judgements_record`` over a real ``judge_expert_slots`` verdict.

Nothing from PATHS is imported at module level."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
ERA = 3600
V2 = 9_399_600
MERGE = V2 + ERA
KING_UID, HOLDER_UID, N = 3, 9, 16
DOMS = ("econ_fin", "energy", "healthcare", "nature", "sales", "transport", "web_cloudops")


def _cfg(tmp_path):
    from cascade.shared.config import load_chain_config, parse_lane_b

    c = load_chain_config(REPO / "chain.toml")
    lb = parse_lane_b({"from_block": V2 + 1, "domains": list(DOMS), "share": 0.5,
                       "acceptance_from_block": V2})
    return replace(c, lane_b=lb, lane_c=replace(c.lane_c, from_block=0),
                   validator=replace(c.validator, state_db_path=str(tmp_path / "v.json")))


class _Client:
    def uid_for_hotkey(self, hk):
        return {"5King": KING_UID, "alice": HOLDER_UID}.get(hk)


def _receipt(tmp_path, block):
    from cascade.shared.lane_b import DomainWin
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    cfg = _cfg(tmp_path)
    r = ValidatorRunner(cfg=cfg, state=genesis("5King", KING_UID))
    r._lane_b().record(DomainWin(domain="energy", hotkey="alice", uid=HOLDER_UID,
                                 round_id="r1", merge_block=MERGE, target_lcb=0.03))
    vec, pay = r._lane_b_vector(_Client(), [KING_UID], N, block)
    return cfg, SimpleNamespace(status="scored", weights=tuple(vec), reward_uids=(KING_UID,),
                                verdict=None, epoch_start_block=block, lane_b=[],
                                lane_b_pay=pay, state_post=None)


def _audit(cfg, rc):
    from cascade.audit.checks import _check_lane_b_weights
    from cascade.shared.chain import decayed_share_vector

    lane_a = decayed_share_vector([KING_UID], len(rc.weights),
                                  decay=cfg.scoring.king_decay, burn_uid=cfg.scoring.burn_uid)
    return _check_lane_b_weights("weights", rc, cfg, lane_a)


@pytest.mark.parametrize("eras", [0, 10, 28, 500])
def test_w9_holder_paid_in_full_at_any_age(tmp_path, eras):
    block = MERGE + eras * ERA
    cfg, rc = _receipt(tmp_path, block)
    (w,) = rc.lane_b_pay["wins"]
    assert (w["multiplier"], w["age"], w["since_era"]) == (1.0, eras, MERGE // ERA)
    assert rc.weights[HOLDER_UID] == pytest.approx(0.5 / len(cfg.lane_b.targets))
    res = _audit(cfg, rc)
    assert res.status == "PASS", res


@pytest.mark.parametrize("tamper", ["decayed", "since_era", "bool_since_era", "age", "no_lcb"])
def test_w9_tampered_payout_row_fails(tmp_path, tamper):
    cfg, rc = _receipt(tmp_path, MERGE + 40 * ERA)
    pay = copy.deepcopy(rc.lane_b_pay)
    w = pay["wins"][0]
    if tamper == "decayed":
        w["multiplier"] = 0.1
    elif tamper == "since_era":
        w["since_era"] += 1
    elif tamper == "bool_since_era":
        w["since_era"] = True
    elif tamper == "age":
        w["age"] += 1
    else:
        w["gain_lcb"] = None
    bad = SimpleNamespace(**{**rc.__dict__, "lane_b_pay": pay})
    assert _audit(cfg, bad).status == "FAIL", tamper


def test_w9_payout_binds_to_the_settlement_state(tmp_path):
    cfg, rc = _receipt(tmp_path, MERGE + 5 * ERA)
    ok = {"lane_b": {"energy": ["alice", HOLDER_UID, MERGE, "r1", 0.03, None, None]}}
    assert _audit(cfg, SimpleNamespace(**{**rc.__dict__, "state_post": ok})).status == "PASS"
    bad = {"lane_b": {"energy": ["mallory", HOLDER_UID, MERGE, "r1", 0.03, None, None]}}
    assert _audit(cfg, SimpleNamespace(**{**rc.__dict__, "state_post": bad})).status == "FAIL"


# ── slot rows ────────────────────────────────────────────────────────────────


def _ptr(tag):
    from cascade.shared.manifest import format_trained_pointer

    return format_trained_pointer("cascade/ckpt@sha256:" + hashlib.sha256(tag.encode()).hexdigest())


def _rows(cfg, declared, effects):
    from cascade.eval.expert_lane import judge_expert_slots
    from cascade.shared.expert_manifest import ExpertEntry
    from cascade.validator.expert_lane import ExpertJudgement, judgements_record

    rng = np.random.default_rng(1)
    dom, cl = [], []
    for d in DOMS:
        for i in range(120):
            dom.append(d)
            cl.append(f"{d}-{i % 8}")
    n = len(dom)
    kq = rng.uniform(0.5, 1.5, size=(n, 3))
    ka = rng.uniform(5.0, 10.0, size=n)
    km = rng.uniform(0.8, 1.2, size=n)
    cq = kq * np.exp(rng.normal(0.0, 0.01, size=(n, 1)))
    cm = km * np.exp(rng.normal(0.0, 0.01, size=n))
    for d, e in effects.items():
        m = np.array([x == d for x in dom])
        cq[m] *= e
        cm[m] *= e
    lb = cfg.lane_b
    tv = judge_expert_slots(slots=lb.targets, declared=declared, domains=dom, clusters=cl,
                            king_qloss=kq, king_abs_target=ka, king_mase=km, chal_qloss=cq,
                            chal_abs_target=ka, chal_mase=cm,
                            params=replace(lb.verdict_params(), B=500), seed="w9")
    e = ExpertEntry(miner_hotkey="alice", miner_uid=HOLDER_UID, gen_ref="g",
                    domain=declared[0], selection={0: [1]}, exclude={},
                    base_king_pointer=_ptr("k"), rebased_pointer=_ptr("r"), corpus_digest="c",
                    train_block=5, commit_block=4,
                    targets=tuple(declared) if len(declared) > 1 else ())
    j = ExpertJudgement(e, bool(tv.won), "verdict", "won", tv.verdicts[declared[0]],
                        king_pointer=_ptr("k"), targets=tv)
    return tv, judgements_record([j])


def _check_rows(cfg, rows):
    from cascade.audit.checks import check_lane_b

    rc = SimpleNamespace(epoch_start_block=V2 + 10 * ERA, lane_b=rows, manifest={},
                         lane_b_pay={"block": V2 + 10 * ERA, "wins": []})
    # the row checks only (no warm-start acceptance evidence in this receipt)
    return check_lane_b(rc, replace(cfg, lane_b=replace(cfg.lane_b, acceptance_from_block=0)))


def test_w9_judged_leg_has_one_row_per_slot_in_two_families(tmp_path):
    cfg = _cfg(tmp_path)
    tv, rows = _rows(cfg, ["energy"], {"energy": 0.9, "sales": 0.94})
    assert sorted(r["domain"] for r in rows) == sorted(cfg.lane_b.targets)
    fam = {r["domain"]: (r["family"], r["gain_bar"], r["passed"]) for r in rows}
    assert fam["energy"] == ("declared", cfg.lane_b.gain_margin, True)
    assert fam["sales"] == ("undeclared", cfg.lane_b.undeclared_gain_margin, True)
    assert not fam["nature"][2]
    res = _check_rows(cfg, rows)
    assert res.status == "PASS", res


@pytest.mark.parametrize("tamper", ["gain_bar", "family", "passed", "crit", "missing_slot",
                                    "declared_flag"])
def test_w9_tampered_slot_rows_fail(tmp_path, tamper):
    cfg = _cfg(tmp_path)
    _tv, rows = _rows(cfg, ["energy", "web_cloudops"], {"energy": 0.9, "sales": 0.94})
    rows = copy.deepcopy(rows)
    by = {r["domain"]: r for r in rows}
    if tamper == "gain_bar":
        by["sales"]["gain_bar"] = cfg.lane_b.gain_margin            # an undeclared slot at 1%
    elif tamper == "family":
        by["sales"]["family"] = "declared"
    elif tamper == "passed":
        by["nature"]["passed"] = True
    elif tamper == "crit":
        und = [r for r in rows if r["family"] == "undeclared" and not r["inconclusive"]]
        und[0]["maxt_crit"] = (und[0]["maxt_crit"] or 0.0) + 0.5
    elif tamper == "missing_slot":
        rows = [r for r in rows if r["domain"] != "nature"]
    else:
        by["sales"]["declared"] = True
    assert _check_rows(cfg, _rows(cfg, ["energy", "web_cloudops"],
                                  {"energy": 0.9, "sales": 0.94})[1]).status == "PASS"
    res = _check_rows(cfg, rows)
    assert res.status == "FAIL" and "acceptance" not in res.detail, (tamper, res)
