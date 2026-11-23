"""W14: the absent lane D receipt record (`cascade/validator/lane_d_state.py`).

Plan W14 row: "a `lane_d: {status: "absent"}` receipt state with a lane
D-only burn"; change 4: "reusing MarketCast's `REPORT_ABSENT` /
`INFRA_DEFERRED`". README.md "The absent record" has the contract.
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    ABSENT_KEYS,
    NEW_RECEIPT_CODES,
    PRESENT_KEYS,
    REQUIRED_REASONS,
    Chain,
    boundary,
    receipt_error_code,
)

S = "ab" * 32
T = "cd" * 32


def _record(**over):
    from cascade.validator.lane_d_state import make_absent_receipt

    kw = {"boundary": boundary(2), "state_sha256": S, "reason": "report_absent"}
    kw.update(over)
    return make_absent_receipt(**kw)


def test_absent_status_and_reasons_reuse_marketcast():
    from cascade.shared.marketcast_report import INFRA_DEFERRED, REPORT_ABSENT
    from cascade.validator import lane_d_state as m

    assert hasattr(m, "ABSENT_STATUS"), "lane_d_state exports no ABSENT_STATUS"
    assert m.ABSENT_STATUS == REPORT_ABSENT == "absent"
    assert hasattr(m, "ABSENT_REASONS"), "lane_d_state exports no ABSENT_REASONS"
    reasons = tuple(m.ABSENT_REASONS)
    assert set(REQUIRED_REASONS) <= set(reasons)
    assert INFRA_DEFERRED in reasons
    assert len(set(reasons)) == len(reasons)
    assert all(isinstance(r, str) and r and r == r.strip().lower() for r in reasons)


def test_absent_record_is_exactly_the_pinned_keys():
    from cascade.validator.lane_d_state import ABSENT_STATUS, RECEIPT_SCHEMA

    rec = _record()
    assert sorted(rec) == sorted(ABSENT_KEYS)
    assert rec["schema"] == RECEIPT_SCHEMA
    assert rec["status"] == ABSENT_STATUS
    assert rec["boundary"] == boundary(2) and type(rec["boundary"]) is int
    assert rec["reason"] == "report_absent"
    # One state digest: an absent settlement moves no lane D state.
    assert rec["prior_state_sha256"] == rec["posterior_state_sha256"] == S
    assert "report_sha256" not in rec


@pytest.mark.parametrize("reason", REQUIRED_REASONS)
def test_every_required_reason_builds_a_record(reason):
    assert _record(reason=reason)["reason"] == reason


@pytest.mark.parametrize("over", [
    {"reason": "operator_said_so"}, {"reason": ""}, {"reason": None},
    {"boundary": True}, {"boundary": "8100"}, {"boundary": 1.0}, {"boundary": -1},
    {"state_sha256": "A" * 64}, {"state_sha256": "0x" + "ab" * 32}, {"state_sha256": ""}])
def test_make_absent_receipt_refuses_bad_values(over):
    with pytest.raises(ValueError) as e:
        _record(**over)
    assert receipt_error_code(e) == "bad_value"


def test_error_codes_are_exported():
    from cascade.validator.lane_d_state import RECEIPT_ERROR_CODES

    for code in NEW_RECEIPT_CODES:
        assert code in RECEIPT_ERROR_CODES, code
    # W7c's codes are kept.
    for code in ("not_object", "unknown_schema", "missing_field", "unknown_field",
                 "bad_value", "boundary_mismatch", "report_mismatch", "state_mismatch",
                 "boundary_order", "chain_break"):
        assert code in RECEIPT_ERROR_CODES, code


@pytest.mark.parametrize("field", ["status", "reason", "prior_state_sha256",
                                   "posterior_state_sha256", "boundary", "schema"])
def test_an_absent_record_missing_a_field_is_refused(field):
    from cascade.validator.lane_d_state import verify_receipt, verify_receipt_chain

    rec = _record()
    cut = {k: v for k, v in rec.items() if k != field}
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([cut])
    assert receipt_error_code(e) == "missing_field"
    with pytest.raises(ValueError) as e:
        verify_receipt(cut, report=None)
    assert receipt_error_code(e) == "missing_field"


@pytest.mark.parametrize("extra", [{"report_sha256": S}, {"w14_extra": 1}])
def test_an_absent_record_with_an_extra_field_is_refused(extra):
    from cascade.validator.lane_d_state import verify_receipt_chain

    with pytest.raises(ValueError) as e:
        verify_receipt_chain([{**_record(), **extra}])
    assert receipt_error_code(e) == "unknown_field"


@pytest.mark.parametrize("over", [{"status": "deferred"}, {"status": "ABSENT"},
                                  {"status": None}, {"reason": "operator_said_so"},
                                  {"reason": 3}, {"boundary": "8100"}])
def test_an_absent_record_with_a_bad_value_is_refused(over):
    from cascade.validator.lane_d_state import verify_receipt_chain

    with pytest.raises(ValueError) as e:
        verify_receipt_chain([{**_record(), **over}])
    assert receipt_error_code(e) == "bad_value"


def test_an_absent_record_that_moves_the_state_is_refused():
    """An absent settlement can't change KING_D, the court or any entry: a
    record whose posterior differs from its prior is refused."""
    from cascade.validator.lane_d_state import verify_receipt, verify_receipt_chain

    moved = {**_record(), "posterior_state_sha256": T}
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([moved])
    assert receipt_error_code(e) == "absent_state_change"
    with pytest.raises(ValueError) as e:
        verify_receipt(moved, report=None)
    assert receipt_error_code(e) == "absent_state_change"


def test_verify_receipt_on_an_absent_record(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt

    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    s1 = chain.settle(1, sp)
    state = json.loads(sp.read_bytes())
    rec = _record(state_sha256=s1.receipt["posterior_state_sha256"])
    verify_receipt(rec, report=None)
    verify_receipt(rec, report=None, state=state)
    with pytest.raises(ValueError) as e:
        verify_receipt(rec, report=None, state={**state, "boundary": 0})
    assert receipt_error_code(e) == "state_mismatch"
    # A report for an absent settlement contradicts it: the boundary had one.
    with pytest.raises(ValueError) as e:
        verify_receipt(rec, report={**s1.report, "boundary": boundary(2)})
    assert receipt_error_code(e) == "absent_with_report"
    with pytest.raises(ValueError) as e:
        verify_receipt(rec, report=s1.report)
    assert receipt_error_code(e) == "absent_with_report"


def test_present_records_keep_the_w7c_shape(tmp_path):
    """W7c's receipt record is unchanged: exactly five keys, no `status`,
    and it still verifies against its report."""
    from cascade.validator.lane_d_state import verify_receipt

    chain = Chain(tmp_path)
    s1 = chain.settle(1, chain.state_path("a"))
    assert sorted(s1.receipt) == sorted(PRESENT_KEYS)
    verify_receipt(s1.receipt, report=s1.report)
