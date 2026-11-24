"""W7d: a doctored report, re-signed by the validator's own key (and its
receipt re-issued to bind the doctored body), is a MISMATCH at the field
path of the lie (plan rev6 §14.3 items 1-2: satellites re-derive every
verdict from signed fields; §1.5).

Each case changes one class of field in one settlement of an honest chain
(README "Doctored classes"): crown, verdict, rows, gate outcome, pay, plus
params, signature, receipts, the king/prior-king chain, the house bar, the
first-verdict king (rev6.11: no D20, no confirmation duel) and the
held-candidate rule. Every earlier settlement still MATCHes. The
module-scoped honest replay is checked to MATCH first, so a case can only
fail on its own lie. rev6.11: the chains crown at a first verdict (b2 for
X), so on main (which first-clears X at b2 and never confirms) most cases
fail on their assertions.

"""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    GATE_SUITES,
    body_sha,
    boundary,
    build_all,
    describe,
    digest,
    doctored,
    fixture_key,
    has_path,
    replay,
    replay_module,
)


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-mismatch"))


_HONEST: set[int] = set()


def _honest_first(c):
    """The honest chain a case doctors MATCHes (checked once per chain)."""
    mod = replay_module()
    if id(c) not in _HONEST:
        res = replay(c.pub, uid_of=dict(c.pub.uid_of))
        assert all(r.status == mod.MATCH for r in res), ("the honest chain", describe(res))
        _HONEST.add(id(c))


def _check(c, n, results, paths):
    mod = replay_module()
    _honest_first(c)
    assert len(results) == len(c.pub.boundaries), describe(results)
    for r in results[:n - 1]:
        assert r.status == mod.MATCH, ("an earlier settlement stopped matching", describe(results))
    r = results[n - 1]
    assert r.boundary == c.pub.boundaries[n - 1]
    assert r.status == mod.MISMATCH, describe(results)
    assert r.reason is None
    assert any(has_path(r, p) for p in paths), \
        f"no mismatch at {paths}; got {list(r.mismatches)}"


def _has_gate(c, n) -> None:
    """The fixture fact a gate case doctors: settlement n ran the gate."""
    assert "gate" in c.pub.report(n), f"fixture: settlement {n} ran no gate"


def _has_no_house_bar(c, n) -> None:
    """DEC-CA-0076: a miner holds KING_D at ``n``, and the house is no bar."""
    rep = c.pub.report(n)
    assert rep["cohorts"] and "house_digest" not in rep["cohorts"][0], \
        f"fixture: settlement {n} judges against the house as a second bar"
    assert rep["verdicts"] and "house" not in rep["verdicts"][0]
    assert not rep["king_before"]["house"], f"fixture: a miner must hold KING_D at {n}"


def _lie(c, n, mutate, paths, *, reseal=True, uid_of=False, key=None):
    reports, receipts = doctored(c.pub, n, mutate, reseal=reseal, key=key)
    kw = {"uid_of": dict(c.pub.uid_of)} if uid_of else {}
    res = replay(c.pub, reports=reports, receipts=receipts, **kw)
    _check(c, n, res, paths)
    return res


# ── crown ───────────────────────────────────────────────────────────────────


def test_crown_withheld_is_a_mismatch_at_crowned(chains):
    def m(rep):
        rep["crowned"] = None

    _lie(chains["gated"], 2, m, ["crowned"])


def test_a_self_consistent_withheld_crown_is_still_a_mismatch(chains):
    """The whole report says X was not crowned (king, verdict, court), but
    the rows re-derive a clear and the gate passed: the crown is X."""
    c = chains["gated"]

    def m(rep):
        rep["crowned"] = None
        rep["king_after"] = copy.deepcopy(rep["king_before"])
        rep["verdicts"][0]["outcome"] = "lost"
        rep["pay"]["court"] = [{"digest": rep["king_before"]["digest"], "hotkey": None,
                                "uid": None, "burn_reason": "house_king"}]

    _lie(c, 2, m, ["crowned"])


def test_a_double_crown_is_a_mismatch_at_crowned(chains):
    c = chains["gated"]

    def m(rep):
        rep["crowned"] = c.x["digest"]

    _lie(c, 3, m, ["crowned"])


def test_a_king_after_that_is_not_the_crown_is_a_mismatch(chains):
    def m(rep):
        rep["king_after"] = copy.deepcopy(rep["king_before"])

    _lie(chains["gated"], 2, m, ["king_after"])


# ── verdicts ────────────────────────────────────────────────────────────────


def test_a_verdict_outcome_lie_is_a_mismatch_at_its_outcome(chains):
    def m(rep):
        rep["verdicts"][0]["outcome"] = "lost"

    _lie(chains["gated"], 2, m, ["verdicts[0].outcome"])


def test_a_flipped_clear_is_a_mismatch_at_cleared(chains):
    def m(rep):
        rep["verdicts"][0]["cleared"] = False
        rep["verdicts"][0]["outcome"] = "lost"

    _lie(chains["gated"], 2, m, ["verdicts[0].cleared"])


def test_a_stated_lcb_off_the_rows_is_a_mismatch_at_lcb(chains):
    def m(rep):
        rep["verdicts"][0]["lcb"] = float(rep["verdicts"][0]["lcb"]) + 0.05

    _lie(chains["gated"], 2, m, ["verdicts[0].lcb"])


# ── rows ────────────────────────────────────────────────────────────────────


def test_resealed_doctored_rows_mismatch_the_stated_verdict(chains):
    c = chains["gated"]

    def m(rep):
        for r in rep["rows"]:
            if r["digest"] == c.x["digest"]:
                r["crps"] = float(r["crps"]) * 1.5

    _lie(c, 2, m, ["verdicts[0].lcb", "verdicts[0].gain"])


def test_unsealed_doctored_rows_are_a_mismatch_at_rows_sha256(chains):
    def m(rep):
        rep["rows"][0]["mase"] = float(rep["rows"][0]["mase"]) * 1.01

    _lie(chains["gated"], 2, m, ["rows_sha256"], reseal=False)


def test_a_dropped_row_is_a_mismatch_under_rows(chains):
    def m(rep):
        rep["rows"].pop()

    _lie(chains["gated"], 2, m, ["rows", "cohorts[0]"])


# ── gate outcome ────────────────────────────────────────────────────────────


def test_a_gate_outcome_against_its_lcbs_is_a_mismatch(chains):
    _has_gate(chains["gated"], 2)

    def m(rep):
        res = rep["gate"]["results"][0]
        res["outcome"] = "gate_regression"
        res["blocked_by"] = ["house"]

    _lie(chains["gated"], 2, m, ["gate.results[0]"])


def test_a_gate_pass_with_an_lcb_below_tau_is_a_mismatch(chains):
    c = chains["gated"]
    _has_gate(c, 2)

    def m(rep):
        rep["gate"]["results"][0]["lcb"]["house"]["crps"] = -10.0 * float(c.cfg.tau_public)

    _lie(c, 2, m, ["gate.results[0]"])


def test_a_gate_suite_that_is_not_the_draw_is_a_mismatch(chains):
    _has_gate(chains["gated"], 2)

    def m(rep):
        g = rep["gate"]
        g["suite"] = [s for s in GATE_SUITES if s != g["suite"] and s in g["eligible"]][0]

    _lie(chains["gated"], 2, m, ["gate.suite"])


def test_a_held_regate_stated_as_a_regression_is_a_mismatch(chains):
    c = chains["held"]
    _has_gate(c, 3)

    def m(rep):
        res = rep["gate"]["results"][0]
        res["outcome"] = "gate_regression"
        res["blocked_by"] = ["house"]
        rep["crowned"] = None
        rep["king_after"] = copy.deepcopy(rep["king_before"])
        rep["pay"]["court"] = [{"digest": rep["king_before"]["digest"], "hotkey": None,
                                "uid": None, "burn_reason": "house_king"}]

    _lie(c, 3, m, ["gate.results[0]"])


def test_a_gate_result_for_an_entry_that_holds_no_cleared_first_verdict_is_a_mismatch(chains):
    """Chain rule C6: a gate result without a verdict this settlement is a
    held candidate only if an earlier settlement left that entry holding a
    cleared, ungated (`bench_hold`) first verdict. X is already the king at
    b3 of the gated chain."""
    c = chains["gated"]
    _has_gate(c, 2)
    b2_gate = copy.deepcopy(c.pub.report(2)["gate"])

    def m(rep):
        rep["gate"] = {**b2_gate, "block": rep["boundary"]}

    _lie(c, 3, m, ["gate"])


# ── pay ─────────────────────────────────────────────────────────────────────


def test_a_pay_share_off_config_is_a_mismatch_at_pay_share(chains):
    def m(rep):
        rep["pay"]["share"] = 0.5

    _lie(chains["gated"], 2, m, ["pay.share"])


def test_a_court_uid_off_the_metagraph_is_a_mismatch_at_its_uid(chains):
    def m(rep):
        rep["pay"]["court"][0]["uid"] = 99

    _lie(chains["gated"], 2, m, ["pay.court[0].uid"], uid_of=True)


def test_a_burn_reason_off_the_slot_is_a_mismatch_at_burn_reason(chains):
    def m(rep):
        rep["pay"]["court"][0]["burn_reason"] = "deregistered"

    _lie(chains["gated"], 1, m, ["pay.court[0].burn_reason"])


def test_an_extra_court_slot_is_a_mismatch_under_pay_court(chains):
    def m(rep):
        rep["pay"]["court"].append({"digest": rep["house"]["digest"], "hotkey": None,
                                    "uid": None, "burn_reason": "house_king"})

    _lie(chains["gated"], 2, m, ["pay.court"])


def test_a_court_out_of_recency_order_is_a_mismatch_under_pay_court(chains):
    def m(rep):
        rep["pay"]["court"].reverse()

    _lie(chains["house_bar"], 3, m, ["pay.court"], uid_of=True)


# ── params and signature ────────────────────────────────────────────────────


def test_a_stated_param_off_config_is_a_mismatch_at_that_param(chains):
    def m(rep):
        rep["params"]["alpha"] = 0.1

    _lie(chains["gated"], 1, m, ["params.alpha"])


def test_a_report_signed_by_another_key_is_a_mismatch_at_signature(chains):
    def m(rep):
        rep["seed"] = str(rep["seed"])

    _lie(chains["gated"], 2, m, ["signature", "signer"], key=fixture_key("not-the-validator"))


# ── the king chain ──────────────────────────────────────────────────────────


def test_a_king_before_off_the_previous_king_after_is_a_mismatch(chains):
    def m(rep):
        rep["king_before"] = {**rep["house"]}
        rep["king_after"] = {**rep["house"]}
        rep["pay"]["court"] = [{"digest": rep["house"]["digest"], "hotkey": None,
                                "uid": None, "burn_reason": "house_king"}]

    _lie(chains["gated"], 3, m, ["king_before"])


def test_prior_kings_off_the_chain_are_a_mismatch(chains):
    def m(rep):
        rep["prior_kings"] = [copy.deepcopy(rep["king_after"])]

    _lie(chains["gated"], 3, m, ["prior_kings"])


# ── the house bar ───────────────────────────────────────────────────────────


def test_a_house_bar_named_after_a_miner_crown_is_a_mismatch(chains):
    """DEC-CA-0076 (owner 2026-11-24, laned:tests-correction; was: a house
    judgement off the rows / a dropped house bar are lies): a miner holds
    KING_D, so the house is no bar; a report naming it as one is a lie."""
    c = chains["house_bar"]
    _has_no_house_bar(c, 3)

    def m(rep):
        rep["cohorts"][0]["house_digest"] = rep["house"]["digest"]

    _lie(c, 3, m, ["cohorts[0].house_digest"])


# ── the first verdict against the refreshed house (rev6.11: no D20) ─────────


def test_a_first_verdict_against_the_pre_refresh_house_is_a_mismatch(chains):
    """rev6.11 (b) drops D20: every verdict is judged against `king_before`.
    After the refresh to H1 a cohort naming the old house H0 as its king is a
    lie (C4)."""
    c = chains["refresh"]

    def m(rep):
        rep["cohorts"][0]["king_digest"] = c.h0["digest"]
        rep["cohorts"][0]["king_reveal_ts"] = c.h0["reveal_ts"]
        rep["verdicts"][0]["king_digest"] = c.h0["digest"]

    _lie(c, 2, m, ["cohorts[0]", "verdicts[0]"])


# ── receipts ────────────────────────────────────────────────────────────────


def test_a_receipt_that_does_not_bind_the_report_body_is_a_mismatch(chains):
    c = chains["gated"]
    reports, _ = doctored(c.pub, 2, lambda rep: rep.update(seed=str(rep["seed"]) + "x"))
    res = replay(c.pub, reports=reports, receipts=copy.deepcopy(c.pub.receipts))
    _check(c, 2, res, ["receipt.report_sha256"])
    assert c.pub.receipts[1]["report_sha256"] != body_sha(reports[boundary(2)])


def test_a_broken_receipt_link_is_a_mismatch_at_its_prior(chains):
    c = chains["gated"]
    receipts = copy.deepcopy(c.pub.receipts)
    receipts[2]["prior_state_sha256"] = digest("w7d a state nobody published")
    res = replay(c.pub, receipts=receipts)
    _check(c, 3, res, ["receipt.prior_state_sha256"])


def test_a_first_receipt_not_from_genesis_is_a_mismatch_at_its_prior(chains):
    c = chains["gated"]
    receipts = copy.deepcopy(c.pub.receipts)
    receipts[0]["prior_state_sha256"] = digest("w7d not the genesis state")
    res = replay(c.pub, receipts=receipts)
    _check(c, 1, res, ["receipt.prior_state_sha256"])


# ── downstream of a mismatch ────────────────────────────────────────────────


def test_every_settlement_after_a_mismatch_is_unverifiable_prior_unverified(chains):
    mod = replay_module()
    c = chains["house_bar"]

    def m(rep):
        rep["verdicts"][0]["outcome"] = "lost"

    res = _lie(c, 2, m, ["verdicts[0].outcome"])
    for r in res[2:]:
        assert r.status == mod.UNVERIFIABLE and r.reason == "prior_unverified", describe(res)
