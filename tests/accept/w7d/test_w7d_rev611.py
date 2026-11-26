"""W7d (plan rev6.11 (b), W7d row; rev6.9 W5i): the replay expects no
confirmation stage, re-derives crowning as first clear ∧ A10 ∧ gate, and
re-derives the backtest clamp, all independently of producer code
(invariant 12).

README.md "rev6.11" has the contract. Every forgery is re-signed by the
validator's key and its receipt re-issued (`helpers.doctored`), so the lie
is the only defect.

On main the replay still runs the D20 confirmation rule: a forged
confirmation stage is a MATCH or a MISMATCH, never UNVERIFIABLE with the
typed reason, and `combined_log_ratio` doesn't exist (`AttributeError` on
`cascade.audit.lane_d_chain`, in PATHS).
"""

from __future__ import annotations

import pytest

from .helpers import build_all, describe, doctored, has_path, replay, replay_module

REASON = "confirmation_stage_present"


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-rev611"))


# ── no confirmation stage (rev6.11 (b)) ─────────────────────────────────────


def _stage_confirm(rep: dict) -> None:
    assert rep["cohorts"] and rep["verdicts"], "fixture: b2 judges X"
    rep["cohorts"][0]["stage"] = "confirm"
    for v in rep["verdicts"]:
        v["stage"] = "confirm"


def _confirmation_block(rep: dict) -> None:
    assert rep["verdicts"], "fixture: b2 has X's verdict"
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
def test_rev611_a_report_with_a_confirmation_stage_is_refused(chains, forge):
    """Refused, never a MATCH: UNVERIFIABLE `confirmation_stage_present` at
    that settlement, and every later one `prior_unverified`."""
    mod = replay_module()
    c = chains["gated"]
    reports, receipts = doctored(c.pub, 2, forge)
    res = replay(c.pub, reports=reports, receipts=receipts)
    assert REASON in mod.UNVERIFIABLE_REASONS
    assert res[0].status == mod.MATCH, describe(res)
    assert (res[1].status, res[1].reason) == (mod.UNVERIFIABLE, REASON), describe(res)
    for r in res[2:]:
        assert (r.status, r.reason) == (mod.UNVERIFIABLE, "prior_unverified"), describe(res)


# ── crowned = first clear ∧ A10 ∧ gate ──────────────────────────────────────


def _crowned_verdict(rep: dict) -> dict:
    assert rep["crowned"], "fixture: X is crowned at b2, its first verdict"
    (v,) = [v for v in rep["verdicts"] if v["digest"] == rep["crowned"]]
    return v


def _lie(c, n, mutate, paths):
    mod = replay_module()
    reports, receipts = doctored(c.pub, n, mutate)
    res = replay(c.pub, reports=reports, receipts=receipts)
    for r in res[:n - 1]:
        assert r.status == mod.MATCH, describe(res)
    r = res[n - 1]
    assert r.status == mod.MISMATCH, describe(res)
    assert any(has_path(r, p) for p in paths), f"no mismatch at {paths}; got {list(r.mismatches)}"


def test_rev611_a_crown_without_a_gate_is_a_mismatch(chains):
    def m(rep):
        _crowned_verdict(rep)
        rep.pop("gate", None)

    _lie(chains["gated"], 2, m, ["crowned", "gate"])


def test_rev611_the_replay_crowns_at_the_first_verdict(chains):
    """The honest crowns (gated: X at b2; house_bar: X at b2, Y at b3) are
    re-derived from first verdicts alone."""
    mod = replay_module()
    for name, want in (("gated", ["x"]), ("house_bar", ["x", "y"])):
        c = chains[name]
        res = replay(c.pub)
        assert all(r.status == mod.MATCH for r in res), describe(res)
        crowns = [r.crowned for r in res if r.crowned is not None]
        assert crowns == [getattr(c, w)["digest"] for w in want], name
        for rep in (c.pub.reports()[b] for b in c.pub.boundaries):
            for v in rep["verdicts"]:
                assert v.get("stage", "first") == "first" and "confirmation" not in v


# ── rev6.18 D24 / A7 (F4): no backtest statistic, no A10 ─────────────────────
#
# Correction (owner 2026-11-24: backtest windows dropped; owner 2026-11-23:
# A10 removed): the W5i combined_log_ratio tests and the A10 crown test are
# deleted with the rules they pinned.


def test_rev618_the_replay_has_no_backtest_or_a10_rule():
    mod = replay_module()
    assert not hasattr(mod, "combined_log_ratio"), (
        "the chain replay still re-derives the W5i backtest combination")


def test_rev618_a_verdict_with_doctored_point_gains_is_a_mismatch(chains):
    """rev6.18 A8: each verdict's per-half point gains are re-derived."""
    def m(rep):
        v = _crowned_verdict(rep)
        assert "halves" in v, "F4: verdicts carry their per-half point gains"
        v["halves"]["live"]["gain"] = float(v["halves"]["live"]["gain"]) + 0.25

    _lie(chains["gated"], 2, m, ["verdicts[0]"])


