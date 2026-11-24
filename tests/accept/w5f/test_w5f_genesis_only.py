"""DEC-CA-0076 (owner 2026-11-24): "I just want [the house] to be for genesis
of the comp until a king is crowned … remove the house-holds-king mechanism
after that."

- Genesis (no miner has ever held KING_D): the house holds KING_D and is the
  bar; lane D's share does NOT burn, it stays with its ``takes_from`` lane
  (the vector is exactly the lane-D-off one) — on the main validator
  (``LaneDStep.genesis_for``) and in the audit / satellite replay
  (``lane_d_pay(...).genesis``).
- After the first miner crown: no house second bar, no house challenger, no
  retake (a report claiming one is refused, ``ld_crown``); the court is
  80/15/5 among miner kings, renormalised over the kings that exist; a
  deregistered king's slot burns (unchanged).

Every settlement runs the real ``settle_lane_d`` (W8b's rig: W7a's synthetic
chain, numpy only)."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from tests.accept.w8b import helpers as h

pytest.importorskip("bittensor.sp_core")


def _off(cfg):
    return replace(cfg, lane_d=replace(cfg.lane_d, from_block=0))


def _genesis(step, boundary):
    """``step.genesis_for(boundary)``; ``None`` when the step has none."""
    fn = getattr(step, "genesis_for", None)
    return None if fn is None else fn(boundary)


def _rig(tmp_path, *, lanes_bc=False):
    cfg = h.chain_cfg(tmp_path / "v", lanes_bc=lanes_bc)
    return cfg, h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))


@pytest.mark.parametrize("lanes_bc", [False, True], ids=["lane_a_only", "lanes_bc_on"])
def test_genesis_pays_lane_a_and_burns_nothing(tmp_path, lanes_bc):
    cfg, rig = _rig(tmp_path, lanes_bc=lanes_bc)
    rig.step(h.B1)
    rep = rig.published_report(h.B1)
    assert rep["king_after"]["house"] is True, "fixture: the house holds KING_D at B1"
    assert _genesis(rig.step, h.B1) is True, "the step does not report lane D's genesis"
    at = h.B1 + 5
    got, _ = h.vector_at(cfg, at, step=rig.step)
    want, _ = h.vector_at(_off(cfg), at)
    assert list(got) == list(want), "genesis is not byte-identical to lane D off"
    assert got[int(cfg.scoring.burn_uid)] == want[int(cfg.scoring.burn_uid)], \
        "lane D's share burned at genesis"


def test_first_crown_beats_the_house_and_after_it_the_house_is_gone(tmp_path):
    cfg, rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.step(h.B2)
    rep = rig.published_report(h.B2)
    house = rep["house"]["digest"]
    assert rep["crowned"] == rig.fx["x"], "fixture: X is crowned at B2"
    assert [c["king_digest"] for c in rep["cohorts"]] == [house], \
        "the first crown was not judged against the house (the genesis KING_D)"
    assert _genesis(rig.step, h.B2) is False
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]
    rig.step(h.B3)
    rep3 = rig.published_report(h.B3)
    assert rep3["king_after"]["digest"] == rig.fx["x"]
    assert all("house_digest" not in c for c in rep3["cohorts"]), "the house is a second bar"
    assert all("house" not in v for v in rep3["verdicts"])


def test_a_new_house_after_a_miner_crown_opens_no_challenger():
    from cascade.validator.lane_d_state import HOUSE_ENTRY_KEY, genesis, refresh_house

    old = {"digest": "a" * 64, "reveal_ts": 10}
    new = {"digest": "b" * 64, "reveal_ts": 20}
    state = genesis(old)
    refresh_house(state, new)
    assert state["king"] == {**new, "hotkey": None, "house": True}, \
        "at genesis the throne moves with the house in place"
    miner = {"digest": "c" * 64, "hotkey": "m_hk", "house": False, "reveal_ts": 15}
    state = dict(genesis(old), king=dict(miner))
    refresh_house(state, new)
    assert state["king"] == miner
    assert HOUSE_ENTRY_KEY not in state, "a new house challenges a miner KING_D"
    assert state["house"] == new


def _signed(report):
    from tests.accept.w9 import helpers as w9

    return w9.resign(report)


def test_a_house_retake_after_a_miner_crown_is_refused(tmp_path):
    from cascade.audit.lane_d import lane_d_findings

    cfg, rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.step(h.B2)
    rig.step(h.B3)
    honest = rig.published_report(h.B3)
    assert lane_d_findings(honest, cfg.lane_d, signer_hotkey=h.signer().ss58_address).burn \
        is None
    forged = copy.deepcopy(honest)
    house = forged["house"]
    x = forged["king_after"]
    forged["king_after"] = {"digest": house["digest"], "hotkey": None, "house": True,
                            "reveal_ts": house["reveal_ts"]}
    forged["prior_kings"] = [x, *forged.get("prior_kings", [])]
    forged["pay"]["court"] = [
        {"digest": house["digest"], "hotkey": None, "uid": None, "burn_reason": "house_king"},
        {"digest": x["digest"], "hotkey": x["hotkey"], "uid": h.CHALLENGER_UID,
         "burn_reason": None}]
    f = lane_d_findings(_signed(forged), cfg.lane_d, signer_hotkey=h.signer().ss58_address)
    assert f.burn is not None and f.burn[0] == "ld_crown" and "DEC-CA-0076" in f.burn[1], \
        f"the retake is not refused as a house KING_D after a miner crown: {f.burn}"


def test_the_audit_derives_the_genesis_vector_from_the_signed_record(tmp_path):
    from cascade.audit.checks import check_lane_d_pay, derive_weights
    from tests.accept.w9 import helpers as w9

    cfg, rig = _rig(tmp_path)
    rig.step(h.B1)
    ccfg = w9.chain_config(cfg.lane_d)
    receipt0 = w9.lane_a_receipt(h.B1, [0.0] * w9.N_UIDS)
    want, why = derive_weights(receipt0, _off(ccfg))
    assert why == "" and want is not None
    receipt = replace(receipt0, weights=tuple(want))
    record = {"report": rig.published_report(h.B1), "receipt": rig.published_receipt(h.B1)}
    got, why = derive_weights(receipt, ccfg, lane_d=record)
    assert why == "" and list(got) == list(want), (got, why)
    assert check_lane_d_pay(receipt, ccfg, record).status == "PASS"
    # the court-paying vector is not the genesis one
    burned = [float(x) for x in derive_weights(receipt, ccfg)[0]]
    assert burned != list(want), "fixture: lane D burned differs from lane D off"
    # after genesis the court is 80/15/5 over the miner kings that exist; a
    # deregistered king's slot burns (unchanged)
    from cascade.audit.checks import _lane_d_part

    w = (0.80, 0.15, 0.05)
    two = _lane_d_part(8, share=0.5, court=[3, 2], court_weights=w, burn_uid=0)
    assert two[3] == pytest.approx(0.5 * 0.80 / 0.95) and two[2] == pytest.approx(
        0.5 * 0.15 / 0.95) and two[0] == 0.0
    dereg = _lane_d_part(8, share=0.5, court=[3, None, 4], court_weights=w, burn_uid=0)
    assert dereg[0] == pytest.approx(0.5 * 0.15), "a deregistered king's slot must burn"
    assert dereg[3] == pytest.approx(0.5 * 0.80) and dereg[4] == pytest.approx(0.5 * 0.05)
