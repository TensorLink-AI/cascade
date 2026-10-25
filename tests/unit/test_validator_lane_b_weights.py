"""DEC-CA-0050: the validator's weight push once lane B is active."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.config import parse_lane_b
from cascade.shared.lane_b import DomainWin
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis

KING = "5King"


class _Client:
    def __init__(self, block, uids):
        self.block, self.uids = block, uids
        self.vectors: list[list[float]] = []
        self.equal: list[list[int]] = []

    def current_block(self):
        return self.block

    def n_uids(self):
        return 16

    def uid_for_hotkey(self, hk):
        return self.uids.get(hk)

    def set_equal_share_weights(self, reward_uids, n_uids, *, decay, burn_uid):
        self.equal.append(list(reward_uids))

    def set_weight_vector(self, vec):
        self.vectors.append(list(vec))


def _runner(cfg, tmp_path, from_block):
    lb = parse_lane_b({"from_block": from_block, "domains": ["energy", "web"], "share": 0.5})
    v = replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    return ValidatorRunner(cfg=replace(cfg, lane_b=lb, validator=v), state=genesis(KING, 3))


def test_inactive_lane_b_keeps_legacy_push(cfg, tmp_path):
    r = _runner(cfg, tmp_path, 0)
    c = _Client(10_000, {KING: 3})
    r._apply_weights(c, "r", [3])
    assert c.equal == [[3]] and c.vectors == []


def test_active_lane_b_pays_domain_slot_and_burns_rest(cfg, tmp_path):
    r = _runner(cfg, tmp_path, 1)
    r._lane_b().record(DomainWin(domain="energy", hotkey="alice", uid=9, round_id="r",
                                 merge_block=7200, target_lcb=0.03, tier=0.02, full_eras=5))
    r._persist_lane_b()
    r._lane_b_state = None                      # reload from disk
    r._lane_b_block = 7200
    c = _Client(7200, {KING: 3, "alice": 9})
    vec = r._apply_weights(c, "r", [3])
    assert c.vectors and c.vectors[0] == list(vec)
    assert vec[9] == pytest.approx(0.25)        # 0.5 share / 2 domains × 100%
    assert vec[3] == pytest.approx(0.5)
    assert vec[cfg.scoring.burn_uid] == pytest.approx(0.25)
    assert sum(vec) == pytest.approx(1.0)


def test_reassert_without_settlement_uses_current_boundary(cfg, tmp_path):
    r = _runner(cfg, tmp_path, 1)
    r._lane_b().record(DomainWin(domain="web", hotkey="bob", uid=4, round_id="r",
                                 merge_block=0, target_lcb=0.03, tier=0.02, full_eras=5))
    era = r._lane_b_era_blocks(0)
    c = _Client(20 * era, {KING: 3, "bob": 4})
    vec = r._apply_weights(c, "r", [3])
    assert vec[4] == pytest.approx(0.25 * 0.1)  # past the full window, inside the TTL → floor
    # past win_ttl_eras the payout stops: the slot burns
    c = _Client(40 * era, {KING: 3, "bob": 4})
    vec = r._apply_weights(c, "r", [3])
    assert vec[4] == 0.0 and vec[cfg.scoring.burn_uid] == pytest.approx(0.5)


def test_audit_accepts_lane_b_vector_and_flags_tampering(cfg, tmp_path):
    from types import SimpleNamespace

    from cascade.audit.checks import check_weights

    r = _runner(cfg, tmp_path, 1)
    r._lane_b().record(DomainWin(domain="energy", hotkey="alice", uid=9, round_id="r",
                                 merge_block=7200, target_lcb=0.03, tier=0.02, full_eras=5))
    r._lane_b_block = 7200
    c = _Client(7200, {KING: 3, "alice": 9})
    vec = r._apply_weights(c, "r", [3])
    rc = SimpleNamespace(status="scored", weights=vec, reward_uids=(3,), verdict=None,
                         epoch_start_block=7200, lane_b=({"passed": True},),
                         lane_b_pay=r._last_lane_b_pay)
    acfg = r.cfg
    assert check_weights(rc, acfg).status == "PASS"
    bad = list(vec)
    bad[9] += 0.1
    bad[3] -= 0.1
    rc2 = SimpleNamespace(**{**rc.__dict__, "weights": tuple(bad)})
    assert check_weights(rc2, acfg).status == "FAIL"


# ── audit finding 7: resync receipts + the strengthened rebuild check ────────


def _lane_b_receipt(cfg, tmp_path, *, merge_block=7200, block=7200, rows=None):
    from types import SimpleNamespace

    r = _runner(cfg, tmp_path, 1)
    r._lane_b().record(DomainWin(domain="energy", hotkey="alice", uid=9, round_id="r",
                                 merge_block=merge_block, target_lcb=0.03, tier=0.02, full_eras=5))
    r._lane_b_block = block
    vec = r._apply_weights(_Client(block, {KING: 3, "alice": 9}), "r", [3])
    kw = r._lane_b_receipt_kw(rows)
    return r, SimpleNamespace(status="scored", weights=vec, reward_uids=(3,), verdict=None,
                              epoch_start_block=block, **kw)


def test_resync_receipt_carries_lane_b_and_passes_audit(cfg, tmp_path):
    """The resync path judged nothing (rows None) but set the lane-B vector:
    the receipt still carries rows ([]) + the payout record, and the audit
    selects the lane-B check by config — no strict lane-A false FAIL."""
    from cascade.audit.checks import check_weights

    r, rc = _lane_b_receipt(cfg, tmp_path, rows=None)
    assert rc.lane_b == [] and rc.lane_b_pay is not None
    assert check_weights(rc, r.cfg).status == "PASS"
    # even a receipt with NO lane-B fields at all is judged as lane B (never
    # strict lane-A equality) once lane B is active at its block
    from types import SimpleNamespace

    bare = SimpleNamespace(**{**rc.__dict__, "lane_b": None, "lane_b_pay": None})
    assert check_weights(bare, r.cfg).status == "WARN"


def test_audit_rebuild_catches_inflated_multiplier_and_slot_theft(cfg, tmp_path):
    from types import SimpleNamespace

    from cascade.audit.checks import check_weights

    # a win 6 ERAS old pays 0.25 of its slot (full 5, half-life 1)
    eb = _runner(cfg, tmp_path, 1)._lane_b_era_blocks(0)
    r, rc = _lane_b_receipt(cfg, tmp_path, merge_block=0, block=6 * eb)
    assert rc.lane_b_pay["wins"][0]["age"] == 6
    assert check_weights(rc, r.cfg).status == "PASS"
    # the validator claims the win is fresh (multiplier 1.0) and pays the full slot
    w = dict(rc.lane_b_pay["wins"][0], multiplier=1.0)
    pay = dict(rc.lane_b_pay, wins=[w])
    bad = list(rc.weights)
    extra = 0.25 * (1.0 - 0.25)
    bad[9] += extra
    bad[cfg.scoring.burn_uid] -= extra
    forged = SimpleNamespace(**{**rc.__dict__, "weights": tuple(bad), "lane_b_pay": pay})
    assert check_weights(forged, r.cfg).status == "FAIL"
    # an honest record but the burned remainder diverted to the winner
    stolen = SimpleNamespace(**{**rc.__dict__, "weights": tuple(bad)})
    assert check_weights(stolen, r.cfg).status == "FAIL"
    # a wrong age (the decay clock rewound)
    young = SimpleNamespace(**{**rc.__dict__, "lane_b_pay": dict(
        rc.lane_b_pay, wins=[dict(rc.lane_b_pay["wins"][0], age=0)])})
    assert check_weights(young, r.cfg).status == "FAIL"
    # a clock that is not the settlement's block
    late = SimpleNamespace(**{**rc.__dict__, "epoch_start_block": rc.epoch_start_block - eb})
    assert check_weights(late, r.cfg).status == "FAIL"


def test_restart_uses_the_persisted_settlement_block(cfg, tmp_path):
    """After a restart (no in-memory settlement block) the decay clock is the
    persisted last settled lane-B boundary, not the head floored."""
    r = _runner(cfg, tmp_path, 1)
    st = r._lane_b()
    eb = r._lane_b_epoch_blocks(0)
    st.record(DomainWin(domain="web", hotkey="bob", uid=4, round_id="r", merge_block=eb, target_lcb=0.03, tier=0.02, full_eras=5))
    st.mark_applied("r", eb)
    r._persist_lane_b()
    fresh = _runner(cfg, tmp_path, 1)              # restart: reload from disk
    c = _Client(40 * eb + 17, {KING: 3, "bob": 4})  # the head is far past the settlement
    vec = fresh._apply_weights(c, "reassert", [3])
    assert vec[4] == pytest.approx(0.25)            # age 0 at the persisted block: full slot
    assert fresh._last_lane_b_pay["block"] == eb
