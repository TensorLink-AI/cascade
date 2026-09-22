"""KOTH per-round decision and margin schedule."""

from __future__ import annotations

import numpy as np

from cascade.eval.koth import KothParams, evaluate_round, margin_for_tenure
from cascade.eval.scoring import WindowScore

PARAMS = KothParams(
    win_margin_start=0.02,
    win_margin_end=0.10,
    margin_warmup_rounds=5,
    min_windows=20,
    bootstrap_B=1000,
    bootstrap_alpha=0.05,
    dethrone_cp=3,
)


def _scores(n, scale, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        qloss = rng.uniform(0.1, 1.0, size=9) * scale
        out.append(
            WindowScore(
                series_id=str(i),
                mase=float(rng.uniform(0.5, 1.5) * scale),
                qloss_per_q=qloss,
                abs_target=float(rng.uniform(5.0, 10.0)),
            )
        )
    return out


def test_margin_schedule_ramps_and_clamps():
    assert margin_for_tenure(PARAMS, 0) == 0.02
    assert margin_for_tenure(PARAMS, 5) == 0.10
    assert margin_for_tenure(PARAMS, 100) == 0.10
    mid = margin_for_tenure(PARAMS, 2)
    assert 0.02 < mid < 0.10


def test_inconclusive_below_min_windows():
    king = _scores(10, 1.0, 0)
    # Same windows (abs_target paired): rebuild challenger sharing king abs_target.
    chal = [
        WindowScore(s.series_id, s.mase * 0.5, s.qloss_per_q * 0.5, s.abs_target)
        for s in king
    ]
    res = evaluate_round(king, chal, PARAMS, seed="s")
    assert res.inconclusive
    assert not res.challenger_wins_round


def test_clear_winner_wins_round():
    king = _scores(100, 1.0, 0)
    chal = [
        WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target)
        for s in king
    ]
    res = evaluate_round(king, chal, PARAMS, seed="s", king_tenure_rounds=0)
    assert not res.inconclusive
    assert res.challenger_wins_round
    assert res.lcb >= res.margin


def test_tie_does_not_win():
    king = _scores(100, 1.0, 0)
    chal = [WindowScore(s.series_id, s.mase, s.qloss_per_q, s.abs_target) for s in king]
    res = evaluate_round(king, chal, PARAMS, seed="s")
    assert not res.challenger_wins_round


def test_entrenched_king_needs_bigger_win():
    # A challenger that beats the king by ~5% clears the start margin (0.02) at
    # tenure 0 but not the end margin (0.10) at full tenure.
    king = _scores(200, 1.0, 0)
    chal = [
        WindowScore(s.series_id, s.mase * 0.95, s.qloss_per_q * 0.95, s.abs_target)
        for s in king
    ]
    fresh = evaluate_round(king, chal, PARAMS, seed="s", king_tenure_rounds=0)
    entrenched = evaluate_round(king, chal, PARAMS, seed="s", king_tenure_rounds=10)
    assert fresh.challenger_wins_round
    assert not entrenched.challenger_wins_round


def _paired_chal(king, factor):
    return [
        WindowScore(s.series_id, s.mase * factor, s.qloss_per_q * factor,
                    s.abs_target, domain=s.domain, source=s.source)
        for s in king
    ]


def _scores_with_sources(n, n_sources, seed, domain="energy"):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(
            WindowScore(
                series_id=str(i),
                mase=float(rng.uniform(0.5, 1.5)),
                qloss_per_q=rng.uniform(0.1, 1.0, size=9),
                abs_target=float(rng.uniform(5.0, 10.0)),
                domain=domain,
                source=f"feed{i % n_sources}",
            )
        )
    return out


def test_min_clusters_floor_makes_round_inconclusive():
    params = KothParams(
        win_margin_start=0.02, win_margin_end=0.02, margin_warmup_rounds=0,
        min_windows=20, bootstrap_B=500, bootstrap_alpha=0.05, dethrone_cp=1,
        min_clusters=10,
    )
    king = _scores_with_sources(50, n_sources=3, seed=0)  # 3 feeds < 10
    chal = _paired_chal(king, 0.5)
    res = evaluate_round(king, chal, params, seed="s")
    assert res.inconclusive and not res.challenger_wins_round
    assert res.n_clusters == 3

    enough = _scores_with_sources(50, n_sources=25, seed=0)
    res2 = evaluate_round(enough, _paired_chal(enough, 0.5), params, seed="s")
    assert not res2.inconclusive and res2.n_clusters == 25


def test_shadow_diagnostics_populated_on_clear_win():
    king = _scores_with_sources(100, n_sources=20, seed=1)
    chal = _paired_chal(king, 0.6)
    res = evaluate_round(king, chal, PARAMS, seed="s")
    assert res.challenger_wins_round
    assert res.win_rate == 1.0                      # challenger wins every window
    assert res.wilcoxon_p is not None and res.wilcoxon_p < 0.01
    assert res.per_domain_win_rate == {"energy": (1.0, 100)}
    assert res.n_clusters == 20


def test_unlabeled_scores_are_singleton_clusters():
    king = _scores(100, 1.0, 0)  # no source labels
    chal = [
        WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target)
        for s in king
    ]
    res = evaluate_round(king, chal, PARAMS, seed="s")
    assert res.n_clusters == 100  # every window its own cluster (legacy pools)


def test_nan_bound_never_wins_and_is_inconclusive(monkeypatch):
    """An uncomputable LCB is an inconclusive round, not a compare that
    happens to be False: NaN reaching the verdict path must be explicit."""
    from cascade.eval import koth as koth_mod
    from cascade.eval.koth import with_cohort_lcb

    king = _scores(100, 1.0, 0)
    chal = [
        WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target)
        for s in king
    ]
    clear = evaluate_round(king, chal, PARAMS, seed="s", king_tenure_rounds=0)
    assert clear.challenger_wins_round
    # the family-wise max-T swap with a NaN bound: never a win
    swapped = with_cohort_lcb(clear, float("nan"), PARAMS)
    assert not swapped.challenger_wins_round
    # the paired bootstrap itself yielding NaN ⇒ inconclusive, king holds
    monkeypatch.setattr(koth_mod, "paired_bootstrap_lcb_aggregated",
                        lambda *a, **k: float("nan"))
    res = evaluate_round(king, chal, PARAMS, seed="s", king_tenure_rounds=0)
    assert not res.challenger_wins_round
    assert res.inconclusive
    assert res.n_windows == 100                  # not the min-windows path
