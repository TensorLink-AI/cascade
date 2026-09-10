"""Shared-resample max-T correction for the cohort duel (DEC-CA-0012 revisit).

Bonferroni (alpha/k) over-protects the king because the k challengers share
the king's scores and one window draw — strongly positively correlated tests,
where the union bound is loose. :func:`cohort_maxt_lcbs` reads the critical
value off the ACTUAL joint spread instead. These tests pin the three things
that make it a legitimate replacement: it reduces EXACTLY to the deployed
single-challenger LCB at k=1, it sits between Bonferroni and no-correction for
k>1 IN EXPECTATION (an ensemble property — on any single draw the
studentised-basic bound can fall on either side of the percentile-alpha/k
Bonferroni bound; see ``test_maxt_between_is_an_expectation_not_a_per_draw_law``),
and its false-dethrone rate lands at alpha (not far below, like Bonferroni)
under the cohort's real correlation.
"""
from __future__ import annotations

import numpy as np

from cascade.eval.bootstrap import (
    cohort_maxt_lcbs,
    paired_bootstrap_lcb_aggregated,
)


def _components(n, num_q, scale, seed):
    rng = np.random.default_rng(seed)
    qloss = rng.uniform(0.1, 1.0, size=(n, num_q)) * scale
    abs_t = rng.uniform(5.0, 10.0, size=n)
    mase_a = rng.uniform(0.5, 1.5, size=n) * scale
    return qloss, abs_t, mase_a


def _full_geo(comp):
    """The full-sample bag metric — each cluster counted exactly once (the
    identity resample), i.e. the plug-in geomean RoundResult records."""
    from cascade.eval.bootstrap import _bag_geomeans, _cluster_sums, cluster_codes

    qloss, abs_t, mase = comp
    n = qloss.shape[0]
    codes = cluster_codes(None, n)
    g = int(codes.max()) + 1
    idx = np.arange(g).reshape(1, g)
    return float(_bag_geomeans(*_cluster_sums(qloss, abs_t, mase, codes, g), idx)[0])


def _point_estimate(king, chal):
    """Full-sample relative improvement (king_geo - chal_geo)/king_geo."""
    kg, cg = _full_geo(king), _full_geo(chal)
    return (kg - cg) / (kg if abs(kg) >= 1e-9 else 1e-9)


def test_maxt_reduces_to_the_single_lcb_at_k1():
    """k=1 must be bit-identical to the deployed percentile LCB — the centred
    max-T's critical value cancels the point estimate exactly there."""
    king = _components(60, 9, 1.0, 0)
    chal = _components(60, 9, 0.85, 1)
    king = (king[0], king[1], king[2])
    chal = (chal[0], king[1], chal[2])   # paired windows (shared abs_target)

    single = paired_bootstrap_lcb_aggregated(
        king[0], king[1], king[2], chal[0], chal[1], chal[2],
        alpha=0.05, B=4000, seed="r", clusters=None)
    (maxt,) = cohort_maxt_lcbs(king, [chal], alpha=0.05, B=4000, seed="r")
    assert abs(maxt - single) < 1e-12


def test_maxt_sits_between_bonferroni_and_uncorrected():
    """In the TIGHT/correlated regime (a homogeneous cohort — the common case):
    Bonferroni (alpha/k) is the lowest bound, no-correction (alpha) the highest,
    and the max-T between — it spends the correlation Bonferroni throws away.
    This ordering is the TYPICAL case, not a per-draw law: on a wide/skewed
    heterogeneous cohort it can invert (see
    ``test_maxt_between_is_an_expectation_not_a_per_draw_law``)."""
    n, k = 80, 6
    king = _components(n, 9, 1.0, 100)
    king = (king[0], king[1], king[2])
    chals = []
    for j in range(k):
        cq, _, cm = _components(n, 9, 0.88 + 0.02 * j, 200 + j)
        chals.append((cq, king[1], cm))
    ts = [_point_estimate(king, c) for c in chals]
    j_star = int(np.argmax(ts))   # best observed
    alpha, B, seed = 0.05, 6000, "cohort"
    uncorrected = paired_bootstrap_lcb_aggregated(
        *king, *chals[j_star], alpha=alpha, B=B, seed=seed)
    bonferroni = paired_bootstrap_lcb_aggregated(
        *king, *chals[j_star], alpha=alpha / k, B=B, seed=seed)
    maxt = cohort_maxt_lcbs(king, chals, alpha=alpha, B=B, seed=seed)[j_star]

    assert bonferroni < maxt < uncorrected


def test_maxt_fwer_is_calibrated_under_the_real_correlation():
    """Null cohort (every challenger equals the king in distribution): the
    max-T false-dethrone rate lands near alpha, while Bonferroni comes in far
    below it — the over-protection DEC-CA-0012 flagged. Correlation is REAL:
    all challengers are paired to one king on one window draw per trial."""
    n, k, alpha = 60, 8, 0.10
    margin = 0.0                          # a bare "any improvement" dethrone
    trials = 240
    maxt_dethrones = bonf_dethrones = 0
    for tr in range(trials):
        king = _components(n, 9, 1.0, 5000 + tr)
        king = (king[0], king[1], king[2])
        chals = []
        for j in range(k):
            # same generating scale as the king ⇒ null; independent draw ⇒
            # the challengers are correlated only through the shared king.
            cq, _, cm = _components(n, 9, 1.0, 9000 + tr * k + j)
            chals.append((cq, king[1], cm))
        ts = [_point_estimate(king, c) for c in chals]
        j_star = int(np.argmax(ts))
        seed = f"t{tr}"

        maxt = cohort_maxt_lcbs(king, chals, alpha=alpha, B=1500, seed=seed)[j_star]
        if maxt >= margin:
            maxt_dethrones += 1
        bonf = paired_bootstrap_lcb_aggregated(
            *king, *chals[j_star], alpha=alpha / k, B=1500, seed=seed)
        if bonf >= margin:
            bonf_dethrones += 1

    maxt_rate = maxt_dethrones / trials
    bonf_rate = bonf_dethrones / trials
    # max-T controls FWER at ~alpha (allow Monte-Carlo slack); crucially it is
    # NOT crushed to Bonferroni's level — it recovers the power Bonferroni loses.
    assert maxt_rate <= alpha + 0.06, f"max-T FWER {maxt_rate} exceeds alpha"
    assert maxt_rate >= bonf_rate, f"max-T {maxt_rate} tighter than Bonferroni {bonf_rate}"


def test_maxt_empty_and_order_preserved():
    assert cohort_maxt_lcbs((np.zeros((3, 9)), np.ones(3), np.ones(3)), []) == []
    king = _components(40, 9, 1.0, 1)
    king = (king[0], king[1], king[2])
    a = (_components(40, 9, 0.8, 2)[0], king[1], _components(40, 9, 0.8, 2)[2])
    b = (_components(40, 9, 0.95, 3)[0], king[1], _components(40, 9, 0.95, 3)[2])
    ls = cohort_maxt_lcbs(king, [a, b], B=1000, seed="o")
    assert len(ls) == 2 and ls[0] > ls[1]   # the clearly-better challenger has the higher LCB


def _hetero_cohort(seed, sigma_dom, sigma_weak, n=80, num_q=9):
    """A cohort sharing one per-window difficulty (the real positive
    correlation) with independent idiosyncratic noise: one strong low/high-var
    challenger + two weak. Returns (king, [dom, w1, w2]) as bootstrap
    components."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.5, 1.5, n)                     # shared window difficulty
    def comp(factor, sigma, s2):
        v = d * factor * np.exp(np.random.default_rng(s2).normal(0, sigma, n))
        return np.repeat(v[:, None], num_q, axis=1), np.ones(n), v.copy()
    king = comp(1.0, 0.3, seed * 7 + 1)
    dom = comp(0.85, sigma_dom, seed * 7 + 2)
    w1 = comp(1.00, sigma_weak, seed * 7 + 3)
    w2 = comp(1.02, sigma_weak, seed * 7 + 4)
    return king, [dom, w1, w2]


def test_maxt_between_is_an_expectation_not_a_per_draw_law():
    """max-T is between Bonferroni and uncorrected IN EXPECTATION, not on every
    draw. Over many wide/heterogeneous cohorts: (1) on average its bound is
    >= Bonferroni's (it frees the king — the whole point), but (2) on individual
    draws it lands on BOTH sides of Bonferroni. Guards against re-introducing the
    false 'always sits between' claim (a live testnet cohort, round 17856…, had
    max-T below Bonferroni — a normal minority draw)."""
    alpha, k, B, N = 0.05, 3, 2000, 30
    gaps = []
    for s in range(N):
        king, chals = _hetero_cohort(s, sigma_dom=0.5, sigma_weak=0.5)
        j = int(np.argmax([_point_estimate(king, c) for c in chals]))
        bonf = paired_bootstrap_lcb_aggregated(*king, *chals[j], alpha=alpha / k,
                                               B=B, seed="b")
        maxt = cohort_maxt_lcbs(king, chals, alpha=alpha, B=B, seed="b")[j]
        gaps.append(maxt - bonf)
    gaps = np.array(gaps)
    assert gaps.mean() >= 0.0, f"max-T should be >= Bonferroni on average, mean gap {gaps.mean():.5f}"
    assert (gaps < 0).any(), "expected some draws with max-T below Bonferroni (not a per-draw law)"
    assert (gaps >= 0).any(), "expected some draws with max-T at/above Bonferroni"
