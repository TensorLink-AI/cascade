"""WB26: the suite-target gain test (DEC-CA-0071 Amendment rev6.19; plan
rev6.19 (G)) and the guards of a suite-target leg (DEC-CA-0071 §3), on the
real ``judge_expert_targets``. See helpers.py for the rule."""

from __future__ import annotations

import math

import pytest

from . import helpers as h

GIFT = "suite:gift"
BETTER = 0.95          # 5% better on a metric: clears LCB >= 0.01 on 40 items
WORSE = 1.03           # 3% worse: below the -tau_public = -0.01 bound


def _v(tv, t=GIFT):
    return tv.verdicts[t]


# ── the either-metric gain test ──────────────────────────────────────────────


def test_crps_only_gain_wins():
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER})
    v = _v(tv)
    rec = h.suite_rec(v)
    assert v.passed, v.reason
    assert rec["cleared_metric"] == "crps"
    assert rec["noninf_metric"] == "mase" and rec["noninf_lcb"] >= -h.TAU_PUBLIC
    assert math.isclose(v.target_lcb, rec["crps_lcb"])


def test_mase_only_gain_wins():
    """The owner's 2026-11-24 change: a MASE-only improvement now wins (the
    old rule demanded CRPS superiority)."""
    tv = h.judge([GIFT], factors={("gift", "mase"): BETTER})
    v = _v(tv)
    rec = h.suite_rec(v)
    assert v.passed, v.reason
    assert rec["cleared_metric"] == "mase"
    assert rec["noninf_metric"] == "crps"
    assert math.isclose(v.target_lcb, rec["mase_lcb"])


def test_both_metrics_clear_and_full_pay_uses_the_smaller_lcb():
    tv = h.judge([GIFT], factors={("gift", "crps"): 0.95, ("gift", "mase"): 0.90})
    v = _v(tv)
    rec = h.suite_rec(v)
    assert v.passed, v.reason
    assert rec["cleared_metric"] == "both" and rec["noninf_metric"] == ""
    assert rec["crps_lcb"] < rec["mase_lcb"]
    assert math.isclose(v.target_lcb, min(rec["crps_lcb"], rec["mase_lcb"])), (
        "the full-pay tier input is the SMALLER cleared LCB (orchestrator 2026-11-24)")


@pytest.mark.parametrize("gain_m,bad_m", [("crps", "mase"), ("mase", "crps")])
def test_the_other_metric_must_be_non_inferior(gain_m, bad_m):
    tv = h.judge([GIFT], factors={("gift", gain_m): BETTER, ("gift", bad_m): WORSE})
    v = _v(tv)
    rec = h.suite_rec(v)
    assert rec["cleared_metric"] == gain_m
    assert rec["noninf_metric"] == bad_m and rec["noninf_lcb"] < -h.TAU_PUBLIC
    assert not v.passed and bad_m in v.reason, v.reason


def test_no_gain_on_either_metric_fails():
    tv = h.judge([GIFT], factors={("gift", "crps"): 0.999, ("gift", "mase"): 1.0})
    v = _v(tv)
    rec = h.suite_rec(v)
    assert rec["cleared_metric"] == "" and not v.passed
    assert v.reason.startswith("no gain"), v.reason


def test_gain_is_tested_at_alpha_over_2k_and_rederives():
    """Bonferroni alpha/2 per metric on top of the leg's alpha/k: a k = 2 leg
    (a suite and a domain) tests each suite metric at alpha/4, on the seed
    ``<seed>|gain|<target>|<metric>``; the non-inferiority LCB is at alpha."""
    from cascade.eval.expert_lane import paired_geomean_lcb

    tv = h.judge([GIFT, "energy"], factors={("gift", "crps"): BETTER},
                 pool_effect={"energy": 0.85})
    rec = h.suite_rec(_v(tv))
    assert math.isclose(rec["gain_alpha"], h.ALPHA / 4)
    pub = h.public({("gift", "crps"): BETTER})
    for m in h.METRICS:
        k, c = pub[f"gift_{m}"]
        want = paired_geomean_lcb(k, c, alpha=h.ALPHA / 4, B=h.B, seed=f"{h.SEED}|gain|{GIFT}|{m}")
        assert math.isclose(rec[f"{m}_lcb"], want), (m, rec[f"{m}_lcb"], want)
    k, c = pub["gift_mase"]
    want = paired_geomean_lcb(k, c, alpha=h.ALPHA, B=h.B, seed=f"{h.SEED}|noninf|{GIFT}|mase")
    assert math.isclose(rec["noninf_lcb"], want)
    assert _v(tv).passed and _v(tv, "energy").passed, (_v(tv).reason, _v(tv, "energy").reason)


def test_too_few_items_is_inconclusive():
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER}, n={"gift": 5})
    v = _v(tv)
    assert v.inconclusive == "items" and not v.passed, (v.inconclusive, v.reason)


def test_margin_is_gain_margin_public():
    """A 5% gain fails a 0.2 bar: the bar is ``gain_margin_public``."""
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER}, gain_margin_public=0.2)
    v = _v(tv)
    assert h.suite_rec(v)["cleared_metric"] == "" and not v.passed


# ── guards of a suite-target leg ─────────────────────────────────────────────


def test_own_suite_metrics_are_not_guards_other_suites_are():
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER})
    v = _v(tv)
    h.suite_rec(v)
    assert not {"gift_crps", "gift_mase"} & set(v.public_lcbs), v.public_lcbs
    assert set(v.public_lcbs) == {f"{s}_{m}" for s in h.SUITES if s != "gift" for m in h.METRICS}


def test_other_suite_regression_blocks():
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER, ("boom", "mase"): WORSE})
    v = _v(tv)
    h.suite_rec(v)
    assert not v.passed and "boom_mase" in v.reason, v.reason


def test_pool_domain_regression_blocks():
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER}, pool_effect={"web": 1.05})
    v = _v(tv)
    h.suite_rec(v)
    assert not v.passed and "web" in v.reason, v.reason


def test_guard4_whole_pool_point_gain_must_be_non_negative():
    """A 0.1% pool-wide loss clears every per-domain non-inferiority guard
    (LCB -0.001 >= -0.01) but fails guard 4 (point gain < 0)."""
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER},
                 pool_effect={"energy": 1.001, "web": 1.001},
                 anchor_kw={"pool_effect": {"energy": 1.001, "web": 1.001}})
    v = _v(tv)
    h.suite_rec(v)
    assert all(x >= -0.01 for x in v.guard_lcbs.values()), v.guard_lcbs
    sg = v.suite_guards
    assert sg["pool_gain"] < 0 and not v.passed, (sg, v.reason)
    assert "whole pool" in v.reason, v.reason


def test_anchor_missing_fails_closed():
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER}, anchor=None)
    v = _v(tv)
    h.suite_rec(v)
    assert not v.passed and "anchor_missing" in v.reason, v.reason


def test_anchor_public_rows_missing_fails_closed():
    tv = h.judge([GIFT], factors={("gift", "crps"): BETTER}, anchor_kw={"drop": ("boom_crps",)})
    v = _v(tv)
    assert not v.passed and "anchor_missing" in v.reason, v.reason


def test_anchor_domain_drift_blocks_and_tau_anchor_domain_relaxes():
    """The anchor is 1% better than the king on web: the candidate (= the
    king on web) passes every king guard but is worse than the anchor."""
    kw = {"pool_effect": {"web": 0.99}}
    v = _v(h.judge([GIFT], factors={("gift", "crps"): BETTER}, anchor_kw=kw))
    h.suite_rec(v)
    assert v.suite_guards["anchor_guard_lcbs"]["web"] < 0
    assert not v.passed and "anchor" in v.reason and "web" in v.reason, v.reason
    v2 = _v(h.judge([GIFT], factors={("gift", "crps"): BETTER}, anchor_kw=kw,
                    tau_anchor_domain=0.02))
    assert v2.passed, v2.reason


def test_anchor_public_drift_blocks_and_tau_anchor_public_relaxes():
    kw = {"public_factors": {("time", "crps"): 0.99}}
    v = _v(h.judge([GIFT], factors={("gift", "crps"): BETTER}, anchor_kw=kw))
    h.suite_rec(v)
    assert v.suite_guards["anchor_public_lcbs"]["time_crps"] < 0
    assert not v.passed and "time_crps" in v.reason, v.reason
    v2 = _v(h.judge([GIFT], factors={("gift", "crps"): BETTER}, anchor_kw=kw,
                    tau_anchor_public=0.02))
    assert v2.passed, v2.reason


def test_suite_guards_run_once_per_leg():
    """guard 4 and the anchor guard are leg guards: every conclusive target
    of a mixed leg carries the same record, and a failure blocks them all."""
    tv = h.judge([GIFT, "energy"], factors={("gift", "crps"): BETTER},
                 pool_effect={"energy": 0.85}, anchor=None)
    a, b = _v(tv), _v(tv, "energy")
    assert a.suite_guards and a.suite_guards == b.suite_guards
    assert not a.passed and not b.passed
    assert "anchor_missing" in tv.guard_reason, tv.guard_reason


def test_domain_only_leg_is_judged_as_before():
    """A leg with no suite target never builds suite guards and never reads
    the anchor (the pre-WB26 judgement)."""
    from cascade.eval.expert_lane import judge_expert_leg

    tv = h.judge(["energy"], pool_effect={"energy": 0.85}, anchor=None)
    v = _v(tv, "energy")
    assert v.passed and v.suite_guards == {} and getattr(v, "suite", None) == {}
    doms, clusters, (kq, ka, km), (cq, ca, cm) = h.pool({"energy": 0.85})
    old = judge_expert_leg(target_domain="energy", domains=doms, king_qloss=kq,
                           king_abs_target=ka, king_mase=km, chal_qloss=cq,
                           chal_abs_target=ca, chal_mase=cm, clusters=clusters,
                           public=h.public(), params=h.params(), seed=h.SEED)
    assert (old.passed, old.reason, old.target_lcb, old.guard_lcbs, old.public_lcbs) == (
        v.passed, v.reason, v.target_lcb, v.guard_lcbs, v.public_lcbs)
