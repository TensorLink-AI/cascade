"""W9 (plan rev6.11 (b), W9 row; rev6.18 F4): the audit expects no
confirmation stage and re-derives crowning as first clear ∧ gate (rev6.18
A7: no A10; D24: no backtest clamp).

README.md "rev6.11" has the contract. Every forgery is re-signed by the
validator's key with `rows_sha256` recomputed (`helpers.resign`), so the
defect a test introduces is the only one.

On main the audit still knows the confirmation duel: it PASSes or FAILs
these forgeries without the typed code, and `combined_log_ratio` doesn't
exist (`ImportError` from `cascade.audit.lane_d`, in PATHS).
"""

from __future__ import annotations

import copy

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


# ── rev6.18 D24 / A7 (F4): no backtest statistic, no A10 ─────────────────────
#
# Correction (owner 2026-11-24: backtest windows dropped; owner 2026-11-23:
# A10 removed): the W5i combined_log_ratio tests and the A10 crown test are
# deleted with the rules they pinned; the audit no longer has either.


def test_rev618_the_audit_has_no_backtest_or_a10_rule():
    from cascade.audit import lane_d

    assert not hasattr(lane_d, "combined_log_ratio"), (
        "cascade.audit.lane_d still re-derives the W5i backtest combination")
    assert not hasattr(lane_d, "_lowfreq_problem"), "the audit still re-derives A10"


def test_rev618_a_crown_whose_public_half_regressed_fails(run):
    """rev6.18 A3: a clear needs a positive point gain on every half; a
    verdict whose stated live gain is made negative no longer re-derives."""
    report = _s1(run)
    v = _crowned_verdict(report, run["entry"]["digest"])
    assert "halves" in v, "F4: verdicts carry their per-half point gains"
    v["halves"]["live"]["gain"] = -abs(v["halves"]["live"]["gain"])
    v["sources"]["live"] = dict(v["halves"]["live"])
    h.assert_fail(h.resign(report), run["cfg"], "crowned with a negative live half")


