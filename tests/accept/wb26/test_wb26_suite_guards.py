"""WB26: guard 4 and the anchor guard (DEC-CA-0071 §3) on suite slots of the
real rev6.22 ``judge_expert_slots``. The gain side (CRPS or MASE, bars, max-T
families) is #854's and is used as-is; see helpers.py."""

from __future__ import annotations

from . import helpers as h

GIFT = "suite:gift"


def _v(tv, t=GIFT):
    return tv.verdicts[t]


def test_cleared_suite_slot_passes_with_both_guards_recorded():
    calls: list = []
    tv = h.judge(factors=h.GIFT_GAIN, calls=calls)
    v = _v(tv)
    assert v.passed, v.reason
    sg = getattr(v, "suite_guards", None)
    assert isinstance(sg, dict) and sg, f"a claimed suite slot records `suite_guards`; got {sg!r}"
    assert sg["pool_gain"] >= 0 and sg["anchor_pointer"] == h.ANCHOR_PTR
    assert set(sg["anchor_guard_lcbs"]) == set(v.guard_lcbs)
    assert set(sg["anchor_public_lcbs"]) == set(v.public_lcbs)
    assert calls == ["anchor"], f"the anchor is resolved once per leg; got {calls}"


def test_guard4_whole_pool_point_gain_must_be_non_negative():
    """A 0.1% pool-wide loss clears every per-domain non-inferiority guard
    (LCB -0.001 >= -0.01) but fails guard 4 (point gain < 0)."""
    tv = h.judge(factors=h.GIFT_GAIN, pool_effect={"energy": 1.001, "web": 1.001},
                 anchor_kw={"pool_effect": {"energy": 1.001, "web": 1.001}})
    v = _v(tv)
    h.need_anchor_param()
    assert all(x >= -0.01 for x in v.guard_lcbs.values()), v.guard_lcbs
    assert v.metric_lcbs["crps"] >= v.bar, "fixture: the GIFT slot clears its bar"
    assert v.suite_guards["pool_gain"] < 0 and not v.passed, (v.suite_guards, v.reason)
    assert "whole pool" in v.reason, v.reason


def test_anchor_missing_fails_closed():
    tv = h.judge(factors=h.GIFT_GAIN, anchor=None)
    v = _v(tv)
    assert not v.passed and "anchor_missing" in v.reason, v.reason


def test_anchor_lacking_a_guarded_public_metric_fails_closed():
    tv = h.judge(factors=h.GIFT_GAIN, anchor_kw={"drop": ("boom_crps",)})
    v = _v(tv)
    assert not v.passed and "anchor_missing" in v.reason, v.reason


def test_anchor_domain_drift_blocks_and_tau_anchor_domain_relaxes():
    """The anchor is 1% better than the king on web: the candidate (= the
    king on web) passes every king guard but is worse than the anchor."""
    kw = {"pool_effect": {"web": 0.99}}
    v = _v(h.judge(factors=h.GIFT_GAIN, anchor_kw=kw))
    h.need_anchor_param()
    assert v.suite_guards["anchor_guard_lcbs"]["web"] < 0
    assert not v.passed and "anchor" in v.reason and "web" in v.reason, v.reason
    v2 = _v(h.judge(factors=h.GIFT_GAIN, anchor_kw=kw, tau_anchor_domain=0.02))
    assert v2.passed, v2.reason


def test_anchor_public_drift_blocks_and_tau_anchor_public_relaxes():
    kw = {"public_factors": {("time", "crps"): 0.99}}
    v = _v(h.judge(factors=h.GIFT_GAIN, anchor_kw=kw))
    h.need_anchor_param()
    assert v.suite_guards["anchor_public_lcbs"]["time_crps"] < 0
    assert not v.passed and "time_crps" in v.reason, v.reason
    v2 = _v(h.judge(factors=h.GIFT_GAIN, anchor_kw=kw, tau_anchor_public=0.02))
    assert v2.passed, v2.reason


def test_domain_slots_are_never_blocked_by_the_suite_guards():
    """A leg that clears a domain slot AND a suite slot but has no anchor:
    the suite slot is refused, the domain slot is still claimed."""
    tv = h.judge(declared=("energy", GIFT), factors=h.GIFT_GAIN,
                 pool_effect={"energy": 0.85}, anchor=None)
    assert _v(tv, "energy").passed, _v(tv, "energy").reason
    assert not _v(tv).passed and "anchor_missing" in _v(tv).reason
    assert _v(tv, "energy").suite_guards == {}


def test_anchor_not_resolved_when_no_suite_slot_clears():
    """Guards 4-5 are lazy: a leg that clears no suite slot never scores the
    anchor, and its verdicts are #854's unchanged."""
    from cascade.eval.expert_lane import judge_expert_slots

    calls: list = []
    tv = h.judge(declared=("energy",), pool_effect={"energy": 0.85}, calls=calls)
    assert calls == [], calls
    doms, clusters, (kq, ka, km), (cq, ca, cm) = h.pool({"energy": 0.85})
    old = judge_expert_slots(
        slots=list(h.DOMAINS) + list(h.SUITE_SLOTS), declared=["energy"], domains=doms,
        king_qloss=kq, king_abs_target=ka, king_mase=km, chal_qloss=cq, chal_abs_target=ca,
        chal_mase=cm, clusters=clusters, public=h.public(), params=h.params(), seed=h.SEED)
    for t in tv.targets:
        a, b = tv.verdicts[t], old.verdicts[t]
        assert (a.passed, a.reason, a.target_lcb, a.metric_lcbs, a.suite_guards) == (
            b.passed, b.reason, b.target_lcb, b.metric_lcbs, b.suite_guards), t


def test_suite_guards_run_once_per_leg():
    """Two suite slots cleared by one leg share one guard-4 / anchor record."""
    calls: list = []
    tv = h.judge(declared=(GIFT, "suite:boom"),
                 factors={**h.GIFT_GAIN, ("boom", "crps"): 0.90}, calls=calls)
    a, b = _v(tv), _v(tv, "suite:boom")
    assert a.passed and b.passed, (a.reason, b.reason)
    assert a.suite_guards == b.suite_guards and calls == ["anchor"], calls
