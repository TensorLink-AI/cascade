"""W9 (plan rev6.11 (b), W9 row; rev6.9 W5i): the audit expects no
confirmation stage, re-derives crowning as first clear ∧ A10 ∧ gate, and
re-derives the backtest clamp independently of producer code.

README.md "rev6.11" has the contract. Every forgery is re-signed by the
validator's key with `rows_sha256` recomputed (`helpers.resign`), so the
defect a test introduces is the only one.

On main the audit still knows the confirmation duel: it PASSes or FAILs
these forgeries without the typed code, and `combined_log_ratio` doesn't
exist (`ImportError` from `cascade.audit.lane_d`, in PATHS).
"""

from __future__ import annotations

import copy
import math

import pytest

from . import helpers as h

CODE = "confirmation_stage_present"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-rev611"))


def _s1(run) -> dict:
    return copy.deepcopy(run["s1"].report)


def _fails_with(report: dict, cfg, code: str, what: str) -> None:
    res = h.audit(report, cfg)
    assert res.status == "FAIL", f"{what}: check_lane_d {res.status} ({res.detail})"
    assert code in str(res.detail), f"{what}: FAIL without the code {code!r}: {res.detail}"


# ── no confirmation stage (rev6.11 (b)) ─────────────────────────────────────


def _stage_confirm(rep: dict) -> None:
    assert rep["cohorts"], "fixture: b1 judges X"
    rep["cohorts"][0]["stage"] = "confirm"
    for v in rep["verdicts"]:
        v["stage"] = "confirm"


def _confirmation_block(rep: dict) -> None:
    assert rep["verdicts"], "fixture: b1 has X's verdict"
    co = rep["cohorts"][0]
    rep["verdicts"][0]["confirmation"] = {"stage": "confirm", "rule": "restart",
                                          "first_set_sha256": co["set_sha256"],
                                          "first_king_digest": co["king_digest"]}


def _confirm_pending_entry(rep: dict) -> None:
    rep["entries"][0]["status"] = "confirm_pending"
    rep["entries"][0]["reason"] = "confirm_pending"


@pytest.mark.parametrize("forge", [_stage_confirm, _confirmation_block, _confirm_pending_entry],
                         ids=["cohort_stage_confirm", "confirmation_block",
                              "confirm_pending_entry"])
def test_rev611_a_report_with_a_confirmation_stage_fails(run, forge):
    report = _s1(run)
    forge(report)
    _fails_with(h.resign(report), run["cfg"], CODE, forge.__name__)


def test_rev611_the_crowning_verdict_is_a_first_verdict(run):
    """The honest crowning report: X's only verdict is at stage `first` (or
    carries no stage), it is `crowned`, and the audit PASSes it."""
    report = _s1(run)
    x = run["entry"]["digest"]
    assert report["crowned"] == x, "fixture: X is crowned at b1 (its first verdict)"
    (v,) = [v for v in report["verdicts"] if v["digest"] == x]
    assert v.get("stage", "first") == "first" and "confirmation" not in v
    assert v["outcome"] == "crowned"
    h.assert_pass(report, run["cfg"], "honest b1 crown at the first verdict")


# ── crowned = first clear ∧ A10 ∧ gate ──────────────────────────────────────


def _crowned_verdict(rep: dict, x: str) -> dict:
    assert rep["crowned"] == x, "fixture: X is crowned at b1"
    (v,) = [v for v in rep["verdicts"] if v["digest"] == x]
    return v


def test_rev611_a_crown_whose_a10_check_failed_fails(run):
    """A10 `fail` is a judged, final loss: never gated, never crowned."""
    report = _s1(run)
    v = _crowned_verdict(report, run["entry"]["digest"])
    v["lowfreq"] = {**v["lowfreq"], "result": "fail", "n_windows": 8}
    h.assert_fail(h.resign(report), run["cfg"], "crowned although A10 failed")


def test_rev611_a_crown_without_a_gate_fails(run):
    """No §1.3b gate result, no crown (the first clear alone never crowns)."""
    report = _s1(run)
    _crowned_verdict(report, run["entry"]["digest"])
    report.pop("gate", None)
    h.assert_fail(h.resign(report), run["cfg"], "crowned with no gate")


def test_rev611_a_crown_without_a_clear_fails(run):
    report = _s1(run)
    v = _crowned_verdict(report, run["entry"]["digest"])
    v["cleared"] = False
    h.assert_fail(h.resign(report), run["cfg"], "crowned although the first verdict did not clear")


# ── the backtest clamp (rev6.9 W5i), re-derived in the audit ────────────────


def _ref(r_fwd, r_bt, n_fwd, n_bt, cap, floor) -> float:
    """The rev6.9 combination rule, re-derived here."""
    c = 0.0 if n_bt == 0 else min(cap, n_bt / (n_bt + n_fwd))
    clamped = min(max(r_bt, floor), 1.0 / floor)
    return (1.0 - c) * math.log(r_fwd) + c * math.log(clamped)


CLAMP_CASES = [
    # (r_fwd, r_bt, n_fwd, n_bt, cap, floor)
    (0.97, 0.5, 95, 5, 0.05, 0.85),         # cap binds, lower clamp binds
    (0.97, 0.9, 95, 5, 0.05, 0.85),         # inside the clamp
    (1.02, 3.0, 95, 5, 0.05, 0.85),         # upper clamp binds
    (0.97, 0.5, 99, 1, 0.05, 0.85),         # the natural share binds: c = 0.01
    (0.99, 1e-12, 900, 100, 0.05, 0.9),     # perfect memorisation, another floor
]


@pytest.mark.parametrize("case", CLAMP_CASES)
def test_rev611_the_audit_rederives_the_clamped_combined_statistic(case):
    from cascade.audit.lane_d import combined_log_ratio

    r_fwd, r_bt, n_fwd, n_bt, cap, floor = case
    got = combined_log_ratio(r_fwd, r_bt, n_fwd=n_fwd, n_bt=n_bt, cap=cap, floor=floor)
    assert math.isclose(got, _ref(*case), rel_tol=1e-12, abs_tol=1e-15), (case, got)


@pytest.mark.parametrize("n_bt,cap", [(0, 0.05), (5, 0.0)], ids=["no_backtest", "cap_zero"])
def test_rev611_off_the_combined_statistic_is_the_forward_ratio(n_bt, cap):
    """`backtest_weight_cap = 0` (the default) or no backtest window: exactly
    the forward log ratio, whatever the backtest ratio."""
    from cascade.audit.lane_d import combined_log_ratio

    for r_bt in (1e-12, 0.5, 1.0, 7.0):
        got = combined_log_ratio(0.97, r_bt, n_fwd=100, n_bt=n_bt, cap=cap, floor=0.85)
        assert got == math.log(0.97), (n_bt, cap, r_bt, got)


def test_rev611_memorisation_gains_at_most_one_minus_floor_to_the_cap():
    """Forward equal (R_fwd = 1) and perfect backtest memorisation: the gain
    1 − R is at most 1 − floor^cap, below `d_margin` 0.01 at the defaults
    (cap 0.05, floor 0.85)."""
    from cascade.audit.lane_d import combined_log_ratio

    got = combined_log_ratio(1.0, 1e-12, n_fwd=10, n_bt=10_000, cap=0.05, floor=0.85)
    gain = 1.0 - math.exp(got)
    assert math.isclose(gain, 1.0 - 0.85 ** 0.05, rel_tol=1e-12)
    assert gain < 0.01
