"""Fault attribution as data: unknown => operator, miner faults cite a rule
(retro 2026-11-28, item #7 / plan P7; owner 2026-11-28).

This week's failures, each with the class it should have got:

* 2026-11-28 00:50: lane C 5Cu2sx "non-finite training loss at step 1050".
  It was OUR flat-context NaN and was first blamed on the recipe. Operator.
* 2026-11-27 22:10: a CUDA OOM of the miner's own recipe on 1xRTX4090. It is
  the miner's ONLY when the fund-time fit check vouched for the recipe on that
  GPU ("fit", or an own-risk warning). Otherwise we rented a GPU nobody
  checked, and the fault is ours.
* 2026-11-27 23:30: "could not cut a high/15mx96 training sample". The
  recipe's data range is the miner's choice, so it is a miner fault (recipe).
* 2026-11-26 11:42 (lemon pod: never ready in 600 s) and 19:20 (OUR
  ProgressWatchdog THROUGHPUT ABORT on a healthy leg). Operator / host.
* Any string no rule recognises is the operator's: requeued, unburned,
  make-good eligible.

Torch-free: no worker, no pod, no network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

FIX = Path(__file__).resolve().parent / "fixtures"
OOM_STDERR = (FIX / "mkt_oom_rtx4090.stderr").read_text()
HK = "5Cu2sxndK8Dw"


def _fc():
    import importlib

    try:
        return importlib.import_module("cascade.trainer.fault_class")
    except ImportError as e:
        raise AssertionError(f"the attribution table (cascade.trainer.fault_class) is "
                             f"missing: {e}") from e


def _oom_final() -> str:
    from cascade.trainer.remote import final_exception_line

    return final_exception_line(OOM_STDERR)


def _att(rc, text, **kw):
    fc = _fc()
    return fc.attribute(fc.FailureFacts(rc=rc, text=text, **kw))


# ── tonight's cases ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("rc,text", [
    (3, "remote mkt_finetune on funded-x: miner submission rejected: marketcast leg: "
        "non-finite training loss at step 1050"),
    (1, "remote mkt_finetune on funded-x failed (rc=1) marketcast leg failed (operator "
        "fault): non-finite training gradient at step 7"),
])
def test_nonfinite_training_is_the_operators(rc, text):
    a = _att(rc, text, role="mkt_finetune")
    assert (a.party, a.error_class, a.burn) == ("operator", "operator", False)
    assert a.rule_id == "O-NONFINITE" and a.make_good_eligible


def test_oom_is_the_miners_only_when_the_fit_check_vouched():
    final = _oom_final()
    assert final.startswith("torch.OutOfMemoryError: CUDA out of memory")
    text = f"remote mkt_finetune on funded-x failed (rc=1) {OOM_STDERR[-200:]}"
    for ev in ("fit", "own_risk"):
        a = _att(1, text, final_exception=final, role="mkt_finetune", fit_evidence=ev)
        assert (a.miner_fault, a.error_class, a.rule_id) == (True, "gpu_oom", "M-OOM-FIT"), ev
        assert not a.make_good_eligible
    # no fund-time evidence (an older entry, an unreadable recipe): OURS
    for ev in ("", "bogus"):
        a = _att(1, text, final_exception=final, role="mkt_finetune", fit_evidence=ev)
        assert (a.party, a.error_class, a.burn, a.rule_id) == (
            "operator", "operator", False, "O-OOM-UNVOUCHED"), ev


def test_uncuttable_range_is_the_miners_recipe():
    text = ("remote mkt_finetune on funded-x: miner submission rejected: marketcast leg: "
            "could not cut a high/15mx96 training sample")
    a = _att(3, text, role="mkt_finetune")
    assert (a.miner_fault, a.error_class, a.burn, a.rule_id) == (
        True, "recipe", False, "M-GEOMETRY")
    from cascade.funding.public_reason import public_reason

    entry = SimpleNamespace(status="failed", last_error_class=a.error_class,
                            last_error=f"{text} [rule {a.rule_id}]")
    code, _ = public_reason(entry)
    assert code == "recipe_geometry", code


@pytest.mark.parametrize("text", [
    "pod funded-5gmzsp-0 (executor cd0b4a66) not ready within 600s",
    "THROUGHPUT ABORT: step 1/23841 after 613 s, rate 0.000 steps/s",
    "remote expert on funded-x failed: pod unreachable (ssh rc 255)",
])
def test_lemon_pod_and_our_watchdog_are_never_the_miners(text):
    a = _att(1, text, role="expert")
    assert not a.miner_fault and a.make_good_eligible and not a.burn
    assert a.rule_id == "O-HOST"


@pytest.mark.parametrize("rc,text", [
    (1, "Traceback (most recent call last): RuntimeError: something new"),
    (None, "ssh died"),
    (137, ""),
])
def test_unknown_resolves_to_operator_unburned_make_good_eligible(rc, text):
    a = _att(rc, text)
    assert a.rule_id == _fc().UNKNOWN_RULE.rule_id and a.unknown
    assert a.verdict() == (False, "operator", False)
    assert a.make_good_eligible


def test_the_legacy_classifier_routes_through_the_table():
    from cascade.trainer.faults import classify_funded_worker_failure

    assert classify_funded_worker_failure(1, "Traceback ...", stalled_before=False) == (
        False, "operator", False)
    assert classify_funded_worker_failure(
        3, "miner submission rejected: marketcast leg: could not cut a high/15mx96 "
           "training sample", stalled_before=False) == (True, "recipe", False)
    assert classify_funded_worker_failure(
        1, "oom", stalled_before=False, final_exception=_oom_final(),
        role="mkt_finetune") == (False, "operator", False)
    assert classify_funded_worker_failure(
        1, "oom", stalled_before=False, final_exception=_oom_final(),
        role="mkt_finetune", fit_evidence="fit") == (True, "gpu_oom", False)


# ── every miner verdict cites a rule ────────────────────────────────────────


def test_table_is_well_formed():
    fc = _fc()
    rows = fc.table_rows()
    ids = [r["rule_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate rule ids"
    for r in rows:
        assert r["party"] in fc.PARTIES, r
        assert r["rule_id"] and r["evidence"] and r["reason"], r
        # the rule id never trips a public_reason substring check
        assert not any(w in r["rule_id"].lower() for w in (
            "stalled", "diverged", "non-finite", "embargo", "expired", "could not cut"))
    assert fc.UNKNOWN_RULE.party == fc.UNKNOWN and not fc.UNKNOWN_RULE.burn
    worker = [r for r in fc.RULES if r.match is not None]
    assert len(worker) == len(fc.RULES), "every worker-exit row has a matcher"
    for r in [*fc.RULES, *fc.MINER_CITATIONS]:
        if r.party == fc.MINER:
            assert fc.miner_rule_for_class(r.error_class), r.rule_id


def test_every_miner_class_has_a_public_reason():
    from cascade.funding.public_reason import public_reason

    fc = _fc()
    for r in (*fc.RULES, *fc.MINER_CITATIONS, fc.UNKNOWN_RULE):
        cls = fc.operator_class() if r.error_class == "operator" else r.error_class
        entry = SimpleNamespace(status="failed", last_error_class=cls,
                                last_error=f"x [rule {r.rule_id}]")
        code, _ = public_reason(entry)
        assert code != "unclassified", (r.rule_id, cls)


def test_an_uncited_miner_verdict_is_reattributed_to_the_operator():
    fc = _fc()
    msg, miner, cls, burn = fc.enforce_miner_citation("weird", True, "made_up_class", True)
    assert (miner, cls, burn) == (False, "operator", False)
    assert "[rule U-DEFAULT" in msg
    msg, miner, cls, burn = fc.enforce_miner_citation("pod identity: x", True, "tamper", False)
    assert (miner, cls) == (True, "tamper") and msg.endswith("[rule M-TAMPER]")
    # an already-tagged message is not tagged twice
    m2, *_ = fc.enforce_miner_citation(msg, True, "tamper", False)
    assert m2 == msg


def test_a_pre_rent_feasibility_refusal_keeps_its_miner_class():
    """#1007's miner-side pre-rent refusal (class ``infeasible``, nothing
    rented) cites M-INFEASIBLE, so the citation gate does not move it to the
    operator."""
    fc = _fc()
    msg, miner, cls, burn = fc.enforce_miner_citation(
        "feasibility gpu_arch_unsupported: RTX5090 is sm_120", True, "infeasible", False)
    assert (miner, cls, burn) == (True, "infeasible", False)
    assert msg.endswith("[rule M-INFEASIBLE]")


def test_the_recorders_enforce_the_citation():
    from cascade.trainer.lane_c_driver import LegFailure
    from cascade.trainer.runner_funded_pods import _FundedPodMixin

    f = LegFailure("odd", miner_fault=True, error_class="made_up_class", burn=True)
    assert (f.miner_fault, f.error_class, f.burn) == (False, "operator", False)
    g = LegFailure("recipe bad", miner_fault=True, error_class="recipe", burn=True)
    assert (g.miner_fault, g.error_class, g.burn) == (True, "recipe", True)
    assert "[rule M-RECIPE]" in g.msg

    r = SimpleNamespace(_funded_leg_failures={})
    _FundedPodMixin._record_funded_failure(r, HK, "x", miner_fault=True,
                                           error_class="made_up_class", burn=True)
    assert r._funded_leg_failures[HK][1:] == (False, "operator", False)
    _FundedPodMixin._record_funded_failure(r, HK, "gen", miner_fault=True,
                                           error_class="generator", burn=False)
    assert r._funded_leg_failures[HK][1:] == (True, "generator", False)
    assert r._funded_leg_failures[HK][0].endswith("[rule M-GENERATOR]")


# ── the fund-time evidence ──────────────────────────────────────────────────


def test_fit_evidence_round_trips_on_the_lane_c_queue(tmp_path):
    from cascade.funding.queue import LaneCQueue

    fc = _fc()
    q = LaneCQueue(tmp_path / "lane_c_queue.json")
    q.add_lane_c(HK, "mkt:market", "Alemaj-ft-ai-7/mkt-8a4e0122@hf:ba4d", 1)
    assert fc.fit_evidence_of(q.get(HK), "mkt_finetune") == ""
    assert q.set_fit_evidence(HK, "fit")
    again = LaneCQueue(tmp_path / "lane_c_queue.json")
    assert fc.fit_evidence_of(again.get(HK), "mkt_finetune") == "fit"
    assert q.set_fit_evidence(HK, "")
    assert fc.fit_evidence_of(LaneCQueue(tmp_path / "lane_c_queue.json").get(HK),
                              "mkt_finetune") == ""


def test_lane_b_fit_evidence_is_derived_from_the_fund():
    fc = _fc()
    own_risk = SimpleNamespace(gpu_measured=False, levers=None)
    lever = SimpleNamespace(gpu_measured=True, levers={"context_length": 8192})
    default = SimpleNamespace(gpu_measured=True, levers=None)
    assert fc.fit_evidence_of(own_risk, "expert") == "own_risk"
    assert fc.fit_evidence_of(lever, "expert") == "fit"
    assert fc.fit_evidence_of(default, "expert") == ""        # our sizing: an OOM is ours
    assert fc.fit_evidence_of(None, "expert") == ""


def test_intake_fit_check_reports_its_evidence():
    from cascade.funding.intake import FundingIntake

    sig = FundingIntake._mkt_fit_check.__annotations__.get("return", "")
    assert "tuple[dict | None, str, str]" in str(sig), sig
