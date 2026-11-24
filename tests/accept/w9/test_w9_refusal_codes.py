"""W9 (rev6.18 B1/B3, owner 2026-11-23): per-check refusal codes and burn
scoping.

Plan rev6.18 B3: "Only four checks stop pay (BURN, lane D share only):
report signature; config echo = `config_at(boundary)`; scores reproduce the
named winner (`rows_sha256` and bootstrap); weights match the paid court.
Everything else logs, alerts (`cascade-ops`), and keeps paying the signed
court. `check_lane_d` returns a code per check (`code: detail`), and the
audit row keeps the code." Tests: "one burn case (`ld_rows_sha256`: lane D
burns, A/B/C unchanged) and one alert case (`ld_clock`: alert row, pay
unchanged)". B1: "A failed lane D check burns only lane D's share."

On main there is no code table (`cannot import name 'REFUSAL_CODES'`) and a
failed check refuses the whole vector (`derive_weights` -> `(None, why)`).
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h

# The plan's table (rev6.18 B3), restated: code -> class.
PLAN_TABLE = {
    "lane_d_record_missing": "burn", "lane_d_record_malformed": "burn",
    "lane_d_boundary_mismatch": "burn", "lane_d_report_mismatch": "burn",
    "ld_schema": "burn", "ld_signature": "burn", "ld_not_armed": "burn",
    "lane_d_not_armed": "burn", "ld_params": "burn", "ld_pay_share": "burn",
    "ld_seed": "burn", "ld_rows_sha256": "burn", "ld_rows_count": "burn",
    "ld_rows_duplicate": "burn", "ld_cohort": "burn", "ld_crown": "burn",
    "ld_court": "burn", "lane_d_carve_unsupported": "burn", "ld_malformed": "burn",
    "ld_scoring_method": "alert", "ld_sentinel": "alert", "ld_row_order": "alert",
    "ld_clock": "alert", "ld_embargo": "alert", "ld_draw": "alert",
    "ld_judged_set": "alert", "ld_chain": "alert", "ld_snapshot": "alert",
    "ld_public_index": "alert", "ld_bench": "alert",
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = dict(h.crown_chain(tmp_path_factory.mktemp("w9-codes")))
    out["ccfg"] = h.chain_config(out["cfg"])
    return out


def _sign(report: dict) -> dict:
    """Signed by the validator's key WITHOUT re-sealing ``rows_sha256``."""
    from cascade.validator.lane_d_report import sign_report

    return sign_report(report, h.signer())


def _record(s, report: dict) -> dict:
    return h.lane_d_record(s, report=report,
                           receipt=dict(s.receipt, report_sha256=h.body_sha256(report)))


def _receipt(run, name: str):
    s = run[name]
    want = h.producer_vector(run["ccfg"], s.pay["court_uids"])
    return h.lane_a_receipt(int(s.report["boundary"]), want), want


def test_the_code_table_is_the_plans():
    from cascade.audit.lane_d import REFUSAL_CODES, refusal_class

    assert dict(REFUSAL_CODES) == PLAN_TABLE
    assert refusal_class("not-a-code") == "burn", "an unknown code must fail closed"


def test_check_lane_d_details_lead_with_a_table_code(run):
    from cascade.audit.lane_d import REFUSAL_CODES, check_lane_d

    report = copy.deepcopy(run["s1"].report)
    report["rows_sha256"] = "0" * 64
    res = check_lane_d(_sign(report), run["cfg"], signer_hotkey=h.signer().ss58_address)
    assert res.status == "FAIL"
    assert res.detail.split(":", 1)[0] == "ld_rows_sha256", res.detail
    assert res.detail.split(":", 1)[0] in REFUSAL_CODES


def test_burn_case_rows_sha256_burns_lane_d_only(run):
    """`ld_rows_sha256`: lane D's share burns; every other uid is exactly the
    honest vector's, and the whole vector is never refused."""
    receipt, honest = _receipt(run, "s1")
    burn = run["ccfg"].scoring.burn_uid
    share = float(run["ccfg"].lane_d.share)
    report = copy.deepcopy(run["s1"].report)
    report["rows_sha256"] = "0" * 64
    vec, why = h.derive(receipt, run["ccfg"], lane_d=_record(run["s1"], _sign(report)))
    assert vec is not None, why
    assert h.code_of(why) == "lane_d_report_unverified"
    assert why.split(":", 2)[1].strip() == "ld_rows_sha256", why
    for u, (got, want) in enumerate(zip(vec, honest, strict=True)):
        if u == h.CHALLENGER_UID:
            assert got == pytest.approx(want - share, abs=1e-12)
        elif u == burn:
            assert got == pytest.approx(want + share, abs=1e-12)
        else:
            assert got == pytest.approx(want, abs=1e-12), f"uid {u} moved: {got} != {want}"
    assert vec[h.LANE_A_UID] == pytest.approx(honest[h.LANE_A_UID], rel=1e-12)


def test_alert_case_clock_alerts_and_pay_is_unchanged(run):
    """`ld_clock` (a judged window's mirror lag over the limit): the
    finding is an alert, the signed court stays paid, the `lane_d` row
    WARNs with the cascade-ops tag."""
    from cascade.audit.checks import check_lane_d_pay, lane_d_pay
    from cascade.audit.lane_d import ALERT_TAG, check_lane_d, lane_d_findings

    receipt, honest = _receipt(run, "s1")
    report = copy.deepcopy(run["s1"].report)
    wid = report["cohorts"][0]["window_ids"][0]
    lim = float(run["cfg"].max_mirror_lag_hours)
    for w in report["windows"]:
        if w["window_id"] == wid:
            w["mirror_lag_hours"] = lim + 1.0
    report = _sign(report)
    f = lane_d_findings(report, run["cfg"], signer_hotkey=h.signer().ss58_address)
    assert f.burn is None, f.findings
    assert [c for c, _ in f.alerts] == ["ld_clock"], f.findings
    res = check_lane_d(report, run["cfg"], signer_hotkey=h.signer().ss58_address)
    assert res.status == "FAIL" and res.detail.startswith("ld_clock:"), res.detail
    rec = _record(run["s1"], report)
    vec, why = h.derive(receipt, run["ccfg"], lane_d=rec)
    assert why == "" and h.close_vectors(vec, honest), (vec, why)
    pay = lane_d_pay(receipt, run["ccfg"], rec)
    assert pay.paid and [a.split(":", 1)[0] for a in pay.alerts] == ["ld_clock"], pay
    row = check_lane_d_pay(receipt, run["ccfg"], rec)
    assert row.status == "WARN" and row.detail.startswith(ALERT_TAG), row


def test_a_burn_behind_an_alert_still_burns(run):
    """Alerts never stop the checks: a report with an alert-class finding
    AND a later burn-class one (`ld_pay_share`) burns."""
    from cascade.audit.lane_d import lane_d_findings

    report = copy.deepcopy(run["s1"].report)
    wid = report["cohorts"][0]["window_ids"][0]
    for w in report["windows"]:
        if w["window_id"] == wid:
            w["mirror_lag_hours"] = float(run["cfg"].max_mirror_lag_hours) + 1.0
    report["pay"]["share"] = 0.5
    f = lane_d_findings(_sign(report), run["cfg"], signer_hotkey=h.signer().ss58_address)
    assert [c for c, _ in f.alerts] == ["ld_clock"], f.findings
    assert f.burn is not None and f.burn[0] == "ld_pay_share", f.findings


def test_the_lane_d_row_fails_on_a_burn_and_passes_an_honest_settlement(run):
    from cascade.audit.checks import check_lane_d_pay

    receipt, _ = _receipt(run, "s1")
    assert check_lane_d_pay(receipt, run["ccfg"], h.lane_d_record(run["s1"])).status == "PASS"
    row = check_lane_d_pay(receipt, run["ccfg"], None)
    assert row.status == "FAIL" and row.detail.startswith("lane_d_record_missing"), row


def test_weights_off_the_signed_court_burn_lane_d_only(run):
    """B3 "weights match the paid court": a recorded vector that pays lane D's
    share to lane A (RV3 note 10) FAILs the `lane_d` row (`ld_court`), while
    the derivation still pays the signed court (the true vector)."""
    from dataclasses import replace

    from cascade.audit.checks import check_lane_d_pay

    _, honest = _receipt(run, "s1")
    trap = h.lane_a_only_vector(run["ccfg"])
    receipt = replace(h.lane_a_receipt(int(run["s1"].report["boundary"]), trap), verdict=None)
    row = check_lane_d_pay(receipt, run["ccfg"], h.lane_d_record(run["s1"]))
    assert row.status == "FAIL" and "ld_court" in row.detail, row
    vec, why = h.derive(receipt, run["ccfg"], lane_d=h.lane_d_record(run["s1"]))
    assert why == "" and h.close_vectors(vec, honest), (vec, why)
