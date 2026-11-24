"""W9: `derive_weights` re-derives lane D's pay from the signed report and its
W7c receipt record (plan rev6 §1.5 "`checks.derive_weights` and
`check_lane_d` re-derive the lane D weights from the receipt chain", §1.4
pay, §14.3 rule 1 "derive weights", invariant 2; RV3 note 10, the lane D
pay-wiring trap; README "derive_weights").

Honest chains: the vector equals W8's emission (`lane_parts(...,
lane_d_cfg=..., d_court_uids=...)` + `emission_vector`), uid for uid.
Doctored ones: rev6.18 B1 (owner 2026-11-23, "A failed lane D check must
burn only lane D's share"; tests-correction): `(vector, why)` where the
vector is the honest one with lane D's share burned (lanes A/B/C
unchanged) and `why` carries a typed code from
`cascade.audit.checks.LANE_D_WEIGHT_ERRORS` (before rev6.18: `(None, why)`,
the whole vector refused). rev6.18 B3: a chain-provenance failure is
alert-class, so its signed court stays paid.

On main `derive_weights` takes no `lane_d` (an `assert`), returns the lane
A-only vector for an armed settlement, and `LANE_D_WEIGHT_ERRORS` does not
exist (`cannot import name`, in PATHS).
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = dict(h.crown_chain(tmp_path_factory.mktemp("w9-weights")))
    out["ccfg"] = h.chain_config(out["cfg"])
    return out


def _honest(run, name: str):
    """(lane A receipt, lane_d record, producer vector) for settlement ``name``."""
    s = run[name]
    ccfg = run["ccfg"]
    want = h.producer_vector(ccfg, s.pay["court_uids"])
    receipt = h.lane_a_receipt(int(s.report["boundary"]), want)
    return receipt, h.lane_d_record(s), want


def _burned(run) -> list[float]:
    """The producer's vector with an empty lane D court: lane D's share on
    ``burn_uid``, lanes A/B/C exactly as when lane D pays."""
    return h.producer_vector(run["ccfg"], [])


def _refused(got, code: str, want, sub: str | None = None) -> None:
    """rev6.18 B1: a lane D refusal burns lane D's share only — the vector
    is ``want`` (lanes A/B/C unchanged), never ``None``."""
    vec, why = got
    assert vec is not None and h.close_vectors(vec, want), \
        f"expected lane D's share burned ({code}), got the vector {vec} ({why!r})"
    assert h.code_of(why) == code, f"refusal {why!r}; expected code {code!r}"
    if sub is not None:
        assert why.split(":", 2)[1].strip() == sub, f"refusal {why!r}; expected B3 code {sub!r}"


# ── the typed codes ─────────────────────────────────────────────────────────


def test_weight_error_codes_are_exported():
    from cascade.audit.checks import LANE_D_WEIGHT_ERRORS

    assert isinstance(LANE_D_WEIGHT_ERRORS, tuple)
    assert all(isinstance(c, str) and c for c in LANE_D_WEIGHT_ERRORS)
    assert len(set(LANE_D_WEIGHT_ERRORS)) == len(LANE_D_WEIGHT_ERRORS)
    assert set(h.WEIGHT_ERRORS) <= set(LANE_D_WEIGHT_ERRORS), \
        sorted(set(h.WEIGHT_ERRORS) - set(LANE_D_WEIGHT_ERRORS))


# ── honest chains agree with the producer ───────────────────────────────────


@pytest.mark.parametrize("name,lane_d_uid", [
    ("s1", h.CHALLENGER_UID),        # X crowned at its first verdict: pays X (rev6.11)
    ("s2", h.CHALLENGER_UID),        # nothing judged: X keeps the slot
    ("s3", h.CHALLENGER_UID),        # nothing judged: X keeps the slot
    ("s3d", None),                   # X deregistered: its slot burns (B8)
])
def test_honest_settlement_agrees_with_the_producer(run, name, lane_d_uid):
    receipt, rec, want = _honest(run, name)
    ccfg = run["ccfg"]
    burn = ccfg.scoring.burn_uid
    share = float(ccfg.lane_d.share)
    assert want[lane_d_uid if lane_d_uid is not None else burn] >= share - 1e-12, \
        f"fixture: W8's vector does not pay lane D's {share} to {lane_d_uid}"
    vec, why = h.derive(receipt, ccfg, lane_d=rec)
    assert why == "" and h.close_vectors(vec, want), \
        f"{name}: derive_weights {vec} ({why!r}) != the producer's {want}"


def test_honest_settlement_agrees_in_chain_mode(run):
    receipt, _, want = _honest(run, "s3")
    rec = h.lane_d_record(run["s3"], prior=[run["s1"].report, run["s2"].report])
    vec, why = h.derive(receipt, run["ccfg"], lane_d=rec)
    assert why == "" and h.close_vectors(vec, want), f"{vec} ({why!r}) != {want}"


def test_inv1_derive_weights_unchanged_when_unarmed(run):
    """Before lane D arms (the settlement AT `from_block` is not lane D's),
    no lane D record: the vector is today's lane A vector, exactly."""
    ccfg = run["ccfg"]
    assert not ccfg.lane_d.active(h.ARM_BLOCK)
    want = h.lane_a_only_vector(ccfg)
    receipt = h.lane_a_receipt(h.ARM_BLOCK, want)
    from cascade.audit.checks import derive_weights

    vec, why = derive_weights(receipt, ccfg)
    assert (vec, why) == (want, ""), (vec, why)


def test_inv1_derive_weights_armed_nonempty_lane_d_effect(run):
    """Armed and paying: lane A's pool shrinks to 1 - share (its split is
    unchanged, invariant 2) and lane D's share reaches its court."""
    receipt, rec, want = _honest(run, "s2")
    ccfg = run["ccfg"]
    vec, why = h.derive(receipt, ccfg, lane_d=rec)
    assert why == "" and vec is not None, why
    share = float(ccfg.lane_d.share)
    lane_a = h.lane_a_only_vector(ccfg)
    assert vec != lane_a
    assert vec[h.CHALLENGER_UID] == pytest.approx(share, rel=1e-9)
    assert vec[h.LANE_A_UID] == pytest.approx((1.0 - share) * lane_a[h.LANE_A_UID], rel=1e-9)
    assert sum(vec) == pytest.approx(1.0, rel=1e-9)


def test_lane_a_only_vector_is_not_the_derivation(run):
    """RV3 note 10 (the wiring trap): a validator that drops lane D's part
    (lane D paying, B and C off) records the lane A-only vector, and
    `set_weights` hands lane D's share to lane A. The derivation still
    rebuilds the true vector, so it disagrees with that receipt."""
    _, rec, want = _honest(run, "s2")
    ccfg = run["ccfg"]
    trap = h.lane_a_only_vector(ccfg)
    receipt = h.lane_a_receipt(int(run["s2"].report["boundary"]), trap)
    vec, why = h.derive(receipt, ccfg, lane_d=rec)
    assert why == "" and h.close_vectors(vec, want), (vec, why)
    assert not h.close_vectors(vec, trap)


def test_lane_d_survives_a_dec55_policy_with_b_and_c_off(run):
    """The wiring trap at `audit/checks.py` `_lane_vector_want` (rev6.1 W9
    row): with an emission record and no lane B / C pay record, lane D's
    part must still reach `emission_vector`'s second argument. The ladder
    re-shapes lane A's pool only; lane D's share is untouched (D48)."""
    from dataclasses import replace

    s = run["s2"]
    ccfg = run["ccfg"]
    pol = h.ladder_policy()
    want = h.producer_vector(ccfg, s.pay["court_uids"], pol)
    b = int(s.report["boundary"])
    receipt = replace(h.lane_a_receipt(b, want), emission={"boundary": b,
                                                           "policy": pol.as_record()})
    vec, why = h.derive(receipt, ccfg, lane_d=h.lane_d_record(s))
    assert why == "" and h.close_vectors(vec, want), f"{vec} ({why!r}) != {want}"
    assert vec[h.CHALLENGER_UID] == pytest.approx(float(ccfg.lane_d.share), rel=1e-9)


# ── doctored inputs: typed refusals, lane D's share only (rev6.18 B1) ──────


def test_armed_without_a_lane_d_record_is_refused(run):
    """Never a silent lane A-only vector when lane D pays at the boundary:
    lane D's share burns, lanes A/B/C unchanged."""
    receipt, _, _ = _honest(run, "s2")
    _refused(h.derive(receipt, run["ccfg"], lane_d=None), "lane_d_record_missing", _burned(run))


def test_a_lane_d_record_before_lane_d_arms_is_refused(run):
    """Lane D pays nothing at an unarmed settlement: the lane A vector."""
    _, rec, _ = _honest(run, "s1")
    lane_a = h.lane_a_only_vector(run["ccfg"])
    receipt = h.lane_a_receipt(h.ARM_BLOCK, lane_a)
    _refused(h.derive(receipt, run["ccfg"], lane_d=rec), "lane_d_not_armed", lane_a)


def test_another_settlements_report_is_refused(run):
    """b1's report and receipt offered for the b2 settlement."""
    receipt, _, _ = _honest(run, "s2")
    _refused(h.derive(receipt, run["ccfg"], lane_d=h.lane_d_record(run["s1"])),
             "lane_d_boundary_mismatch", _burned(run))


def test_a_receipt_that_binds_another_body_is_refused(run):
    """The W7c record's `report_sha256` is b1's, the report is b2's."""
    receipt, _, _ = _honest(run, "s2")
    bad = dict(run["s2"].receipt, report_sha256=run["s1"].receipt["report_sha256"])
    rec = h.lane_d_record(run["s2"], receipt=bad)
    _refused(h.derive(receipt, run["ccfg"], lane_d=rec), "lane_d_report_mismatch", _burned(run))


def test_a_doctored_court_uid_is_refused(run):
    """The paid uid moved and the receipt re-bound to the doctored body, but
    the signature no longer verifies (B3 `ld_signature`, burn)."""
    receipt, _, _ = _honest(run, "s2")
    report = copy.deepcopy(run["s2"].report)
    report["pay"]["court"][0]["uid"] = h.UID_B
    rebound = dict(run["s2"].receipt, report_sha256=h.body_sha256(report))
    rec = h.lane_d_record(run["s2"], report=report, receipt=rebound)
    _refused(h.derive(receipt, run["ccfg"], lane_d=rec), "lane_d_report_unverified",
             _burned(run), sub="ld_signature")


def test_a_report_signed_by_another_key_is_refused(run):
    receipt, _, _ = _honest(run, "s2")
    report = h.resign(run["s2"].report, h.w7a.fixture_key("w9-not-the-validator"))
    rebound = dict(run["s2"].receipt, report_sha256=h.body_sha256(report))
    rec = h.lane_d_record(run["s2"], report=report, receipt=rebound)
    _refused(h.derive(receipt, run["ccfg"], lane_d=rec), "lane_d_report_unverified",
             _burned(run), sub="ld_signature")


def test_a_re_signed_report_the_audit_refutes_is_refused(run):
    """Validator key, receipt re-bound, but `pay.share` doubled: `check_lane_d`
    FAILs it (B3 `ld_params`, the config echo: burn), so its pay is never
    derived and lane D's share burns."""
    receipt, _, _ = _honest(run, "s2")
    report = copy.deepcopy(run["s2"].report)
    report["pay"]["share"] = 0.5
    report = h.resign(report)
    rebound = dict(run["s2"].receipt, report_sha256=h.body_sha256(report))
    rec = h.lane_d_record(run["s2"], report=report, receipt=rebound)
    vec, why = h.derive(receipt, run["ccfg"], lane_d=rec)
    _refused((vec, why), "lane_d_report_unverified", _burned(run))
    from cascade.audit.lane_d import REFUSAL_CODES

    sub = why.split(":", 2)[1].strip()
    assert REFUSAL_CODES.get(sub) == "burn", why


def test_a_report_the_chain_refutes_alerts_and_keeps_paying(run):
    """b3 forged to undo X's crown (test_w9_chain), re-signed and re-bound.
    Its chain provenance fails (B3 `ld_chain`: alert-class), so it is
    alerted and its signed record stays in force (rev6.18 B3, owner
    2026-11-23: "Every other failed check logs, alerts and keeps paying").
    Its KING_D is the house, so it reads as lane D's genesis (DEC-CA-0076,
    laned:tests-correction; was: the house slot burns): lane D carves
    nothing and the vector is lane A's alone."""
    from cascade.audit.checks import lane_d_pay

    receipt, _, _ = _honest(run, "s3")
    forged = h.resign(h.undo_crown(run["s3"].report, run["s1"].report))
    rebound = dict(run["s3"].receipt, report_sha256=h.body_sha256(forged))
    rec = h.lane_d_record(run["s3"], report=forged, receipt=rebound,
                          prior=[run["s1"].report, run["s2"].report])
    vec, why = h.derive(receipt, run["ccfg"], lane_d=rec)
    assert why == "" and h.close_vectors(vec, h.lane_a_only_vector(run["ccfg"])), \
        (vec, why)
    pay = lane_d_pay(receipt, run["ccfg"], rec)
    assert pay.paid and pay.genesis and any(a.startswith("ld_chain:") for a in pay.alerts), pay
