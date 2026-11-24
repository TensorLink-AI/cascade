"""W5e (plan rev6.18 A3, F4): the combined winning rule.

Live half = the cohort's forward windows; public half = the settlement's
GiftEvalPretrain + RMISC windows; per-window ratio clamped to
``[window_ratio_floor, 1 / window_ratio_floor]``; ``L = (1 − w) L_live +
w L_pub``; ONE two-part max-T cluster bootstrap (resampled within each half,
recombined per replicate); crown iff combined LCB ≥ d_margin AND live point
gain > 0 AND public point gain > 0 (then the unchanged §1.3b gate).
Contract: README.md "Combined rule (rev6.18 F4)".
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest


def _pub():
    """tests/accept/w7a/pub.py's F4 fixture (index-only public sources)."""
    from tests.accept.w7a import pub

    assert hasattr(pub, "index_only"), (
        "F4 (rev6.18 A3): tests/accept/w7a/pub.py has no index-only public fixture")
    return pub


def _judge():
    from cascade.validator.lane_d import judge_cohort

    assert "public" in inspect.signature(judge_cohort).parameters, (
        "F4 (rev6.18 A3): judge_cohort(..., public=) judges on the combined rule")
    return judge_cohort


def _cfg(**over):
    base = dict(alpha=0.05, bootstrap_B=400, d_margin=0.01, public_weight=0.5,
                window_ratio_floor=0.5)
    return SimpleNamespace(**{**base, **over})


def _scores(n, factor, seed, *, spread=0.03):
    rng = np.random.default_rng(seed)
    base = 0.5 + rng.random(n)
    j = 1.0 + spread * (2.0 * rng.random(n) - 1.0)
    return base * factor * j, (1.0 + base) * factor * j


def _bar(n, seed):
    rng = np.random.default_rng(seed)
    base = 0.5 + rng.random(n)
    return base, 1.0 + base


def _case(live_f, pub_f, *, n_live=60, n_pub=80):
    """Bar and one member whose scores are ``factor`` x the bar's per window."""
    lb, pb = _bar(n_live, 1), _bar(n_pub, 2)
    rng = np.random.default_rng(7)
    jl = 1.0 + 0.03 * (2.0 * rng.random(n_live) - 1.0)
    jp = 1.0 + 0.03 * (2.0 * rng.random(n_pub) - 1.0)
    live_m = (lb[0] * live_f * jl, lb[1] * live_f * jl)
    pub_m = (pb[0] * pub_f * jp, pb[1] * pub_f * jp)
    clusters = [f"src{i % 8}" for i in range(n_live)]
    pclusters = [("gift_eval_pretrain" if i % 2 else "rmisc") + f"|fam{i % 5}"
                 for i in range(n_pub)]
    public = {"bar": pb, "members": [pub_m], "clusters": pclusters,
              "sources": [c.split("|")[0] for c in pclusters]}
    return lb, [live_m], clusters, public


# ── the statistic ───────────────────────────────────────────────────────────


def test_one_part_without_clamp_is_the_cohort_max_t_lcb():
    from cascade.eval.bootstrap import clamped_log_ratios, cohort_maxt_lcbs, parts_maxt_lcbs

    n = 90
    king = _bar(n, 3)
    members = [_scores(n, f, s) for f, s in ((0.95, 4), (0.97, 5), (1.02, 6))]
    clusters = [f"s{i % 9}" for i in range(n)]

    def comp(c, m):
        return ((np.asarray(c) / 2.0)[:, None], np.ones(n), np.asarray(m))

    old = cohort_maxt_lcbs(comp(*king), [comp(*m) for m in members], alpha=0.05, B=300,
                           seed="s", clusters=clusters)
    x = [clamped_log_ratios(*king, *m, floor=1e-12) for m in members]
    new = parts_maxt_lcbs([(x, clusters, 1.0)], alpha=0.05, B=300, seed="s")
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(old, new))


def test_the_per_window_ratio_is_clamped():
    from cascade.eval.bootstrap import clamped_log_ratios

    bar = (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    mem = (np.array([1e-6, 1.0, 1e6]), np.array([1e-6, 1.0, 1e6]))
    x = clamped_log_ratios(*bar, *mem, floor=0.5)
    assert np.allclose(x, [math.log(2.0), 0.0, -math.log(2.0)])
    with pytest.raises(ValueError):
        clamped_log_ratios(*bar, *mem, floor=1.0)


def test_both_halves_enter_one_two_part_bootstrap():
    """Each half is resampled within itself: the public half's cluster
    labels move the LCB; the same inputs give the same LCB."""
    from cascade.eval.bootstrap import parts_maxt_lcbs

    rng = np.random.default_rng(11)
    xl = [rng.normal(0.05, 0.02, 60)]
    xp = [rng.normal(0.05, 0.10, 80)]
    lc = [f"s{i % 6}" for i in range(60)]
    per_window = parts_maxt_lcbs([(xl, lc, 0.5), (xp, None, 0.5)], B=500, seed="x")
    one_family = parts_maxt_lcbs([(xl, lc, 0.5), (xp, [f"f{i % 2}" for i in range(80)], 0.5)],
                                 B=500, seed="x")
    assert per_window != one_family, "the public half's clusters do not enter the bootstrap"
    assert per_window == parts_maxt_lcbs([(xl, lc, 0.5), (xp, None, 0.5)], B=500, seed="x")
    live_only = parts_maxt_lcbs([(xl, lc, 1.0)], B=500, seed="x")
    assert per_window != live_only


# ── judge_cohort: the crown conditions ──────────────────────────────────────


@pytest.mark.parametrize(("live_f", "pub_f", "want"), [
    (0.90, 0.90, True),     # better on both halves: clears
    (0.80, 1.05, False),    # better live, worse public: never clears (A3 condition 3)
    (1.05, 0.80, False),    # worse live, better public: never clears (A3 condition 2)
    (0.999, 0.999, False),  # both halves barely better: the combined LCB < d_margin
])
def test_a_clear_needs_the_lcb_and_a_gain_on_each_half(live_f, pub_f, want):
    judge = _judge()
    king, members, clusters, public = _case(live_f, pub_f)
    (r,) = judge(king, members, clusters=clusters, cfg=_cfg(), seed="s", public=public)
    assert r["cleared"] is want, r
    halves = r["halves"]
    assert set(halves) == {"live", "public"}
    assert (halves["live"]["gain"] > 0) is (live_f < 1.0)
    assert (halves["public"]["gain"] > 0) is (pub_f < 1.0)
    if want:
        assert r["lcb"] >= 0.01


def test_better_live_worse_public_can_clear_the_lcb_and_still_lose():
    """The combined LCB alone is not enough (A3: AND point gain > 0 on each half)."""
    judge = _judge()
    king, members, clusters, public = _case(0.6, 1.03)
    (r,) = judge(king, members, clusters=clusters, cfg=_cfg(), seed="s", public=public)
    assert r["lcb"] >= 0.01 and r["halves"]["public"]["gain"] < 0 and not r["cleared"], r


def test_the_combined_gain_weights_the_halves():
    judge = _judge()
    king, members, clusters, public = _case(0.9, 0.95)
    for w in (0.25, 0.5, 0.75):
        (r,) = judge(king, members, clusters=clusters, cfg=_cfg(public_weight=w), seed="s",
                     public=public)
        ll = -math.log1p(-r["halves"]["live"]["gain"])
        lp = -math.log1p(-r["halves"]["public"]["gain"])
        assert math.isclose(r["gain"], -math.expm1(-((1 - w) * ll + w * lp)), rel_tol=1e-9)


def test_point_gains_per_source_and_domain():
    judge = _judge()
    king, members, clusters, public = _case(0.9, 0.9)
    (r,) = judge(king, members, clusters=clusters, cfg=_cfg(), seed="s", public=public)
    assert set(r["sources"]) == {"live", "gift_eval_pretrain", "rmisc"}
    assert r["sources"]["live"] == r["halves"]["live"]
    assert sum(r["sources"][s]["n"] for s in ("gift_eval_pretrain", "rmisc")) \
        == r["halves"]["public"]["n"] == len(public["clusters"])
    assert set(r["domains"]) == {f"live|{c}" for c in clusters} | set(public["clusters"])
    assert sum(d["n"] for d in r["domains"].values()) == len(clusters) + len(public["clusters"])


def test_public_weight_zero_is_the_live_only_rule():
    judge = _judge()
    king, members, clusters, _ = _case(0.9, 1.5)
    (r,) = judge(king, members, clusters=clusters, cfg=_cfg(public_weight=0.0), seed="s",
                 public=None)
    assert set(r["halves"]) == {"live"} and r["cleared"]
    assert set(r["sources"]) == {"live"}


# ── settle_lane_d end to end (index-only public sources) ────────────────────


def test_a_challenger_better_on_both_halves_is_crowned(tmp_path):
    pub = _pub()
    chal = pub.settle_pub(tmp_path / "a", mirrors=False)["chain"]["digests"][0]
    run = pub.settle_pub(tmp_path / "b", mirrors=False, quality={chal: 0.9})
    rep = run["s1"].report
    (v,) = [v for v in rep["verdicts"] if v["digest"] == chal]
    assert rep["crowned"] == chal and v["cleared"] and v["outcome"] == "crowned"
    assert v["halves"]["public"]["gain"] > 0 and v["halves"]["live"]["gain"] > 0
    assert rep["params"]["public_weight"] == 0.5 and rep["params"]["window_ratio_floor"] == 0.5


def test_a_challenger_worse_on_the_public_half_loses(tmp_path):
    pub = _pub()
    probe = pub.settle_pub(tmp_path / "p", mirrors=False)
    chal = probe["chain"]["digests"][0]
    run = pub.settle_pub(tmp_path / "w", mirrors=False, quality={chal: 1.1})
    rep = run["s1"].report
    (v,) = [v for v in rep["verdicts"] if v["digest"] == chal]
    assert rep["crowned"] is None and v["outcome"] == "lost"
    assert v["halves"]["live"]["gain"] > 0 > v["halves"]["public"]["gain"]
    e = [e for e in rep["entries"] if e["digest"] == chal][0]
    assert (e["status"], e["reason"]) == ("lost", "lost_duel")


def test_public_weight_above_zero_without_the_public_half_fails_closed(tmp_path):
    from cascade.validator import lane_d

    assert hasattr(lane_d, "LaneDPublicError"), "F4: no typed public-half error"
    pub = _pub()

    def drop(scores):
        scores.clear()

    with pytest.raises(lane_d.LaneDPublicError) as e:
        pub.settle_pub(tmp_path, mirrors=False, scores_hook=drop)
    assert e.value.code in ("public_missing", "public_mismatch")
    assert not (tmp_path / "state" / "lane_d_state.json").exists(), "nothing is written"


def test_scores_that_miss_a_kept_window_are_the_hosts_fault(tmp_path):
    from cascade.validator import lane_d

    assert hasattr(lane_d, "LaneDPublicError"), "F4: no typed public-half error"
    pub = _pub()

    def cut(scores):
        for per in scores["scores"].values():
            per["rmisc"] = per["rmisc"][1:]

    with pytest.raises(lane_d.LaneDPublicError) as e:
        pub.settle_pub(tmp_path, mirrors=False, scores_hook=cut)
    assert e.value.code == "public_mismatch"


def test_a_challenger_with_non_finite_public_scores_is_a_checkpoint_fault(tmp_path):
    pub = _pub()
    probe = pub.settle_pub(tmp_path / "p", mirrors=False)
    chal = probe["chain"]["digests"][0]

    def nan(scores):
        scores["scores"][chal]["gift_eval_pretrain"][0]["crps"] = float("nan")

    run = pub.settle_pub(tmp_path / "n", mirrors=False, quality={chal: 0.9}, scores_hook=nan)
    rep = run["s1"].report
    e = [e for e in rep["entries"] if e["digest"] == chal][0]
    assert (e["status"], e["reason"]) == ("closed", "checkpoint_fault")
    assert rep["crowned"] is None and rep["verdicts"] == []
