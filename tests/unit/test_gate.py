"""Trained-checkpoint validity gate: the cheap pass/fail pre-duel screen."""

from __future__ import annotations

import numpy as np

from metronome.eval.gate import check_scores_valid
from metronome.eval.scoring import WindowScore


def _ok(n=3):
    return [
        WindowScore(series_id=str(i), mase=1.0, qloss_per_q=np.full(9, 0.5), abs_target=7.0)
        for i in range(n)
    ]


def test_valid_scores_pass():
    assert check_scores_valid(_ok()) is None


def test_empty_scores_fail():
    assert check_scores_valid([]) == "no_scores"


def test_nan_mase_fails():
    s = _ok()
    s[1] = WindowScore("1", mase=float("nan"), qloss_per_q=np.full(9, 0.5), abs_target=7.0)
    assert check_scores_valid(s) == "non_finite"


def test_inf_qloss_fails():
    s = _ok()
    q = np.full(9, 0.5)
    q[3] = np.inf
    s[0] = WindowScore("0", mase=1.0, qloss_per_q=q, abs_target=7.0)
    assert check_scores_valid(s) == "non_finite"


def test_nan_abs_target_fails():
    s = _ok()
    s[2] = WindowScore("2", mase=1.0, qloss_per_q=np.full(9, 0.5), abs_target=float("nan"))
    assert check_scores_valid(s) == "non_finite"


def test_negative_metric_fails():
    s = _ok()
    s[0] = WindowScore("0", mase=-0.1, qloss_per_q=np.full(9, 0.5), abs_target=7.0)
    assert check_scores_valid(s) == "negative_metric"


def test_negative_qloss_fails():
    s = _ok()
    q = np.full(9, 0.5)
    q[0] = -1e-9
    s[0] = WindowScore("0", mase=1.0, qloss_per_q=q, abs_target=7.0)
    assert check_scores_valid(s) == "negative_metric"
