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
