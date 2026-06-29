"""score_forecaster_on_windows end-to-end + static guard."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.scoring import global_geomean, score_forecaster_on_windows
from cascade.eval.window import EvalWindow
from cascade.interface.static_guard import scan_source

BLOCKED = ("socket", "subprocess", "cascade.shared.chain")


def _windows(n=10, h=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        hist = np.cumsum(rng.standard_normal(80)).astype(np.float64)
        tgt = hist[-1] + np.cumsum(rng.standard_normal(h)).astype(np.float64)
        out.append(EvalWindow(series_id=str(i), history=hist, target=tgt, metadata={"freq": "D"}))
    return out


def test_score_forecaster_runs_and_better_model_scores_lower():
    windows = _windows()

    def good(history, horizon, num_samples):
        last = history[-1]
        base = np.full((1, num_samples, horizon), last)
        return base + 0.1 * np.random.default_rng(0).standard_normal((1, num_samples, horizon))

    def bad(history, horizon, num_samples):
        return 1000.0 + np.random.default_rng(1).standard_normal((1, num_samples, horizon))

    good_scores = score_forecaster_on_windows(good, windows, num_samples=50)
    bad_scores = score_forecaster_on_windows(bad, windows, num_samples=50)
    assert len(good_scores) == len(windows)
    assert global_geomean(good_scores) < global_geomean(bad_scores)


def test_score_forecaster_rejects_wrong_shape():
    windows = _windows(n=1)

    def wrong(history, horizon, num_samples):
        return np.zeros((1, num_samples, horizon + 1))

    with pytest.raises(ValueError):
        score_forecaster_on_windows(wrong, windows, num_samples=10)


def test_score_forecaster_rejects_nonfinite():
    windows = _windows(n=1)

    def naughty(history, horizon, num_samples):
        out = np.zeros((1, num_samples, horizon))
        out[0, 0, 0] = np.nan
        return out

    with pytest.raises(ValueError):
        score_forecaster_on_windows(naughty, windows, num_samples=10)


def test_static_guard_flags_blocked_imports():
    assert scan_source("import numpy\n", BLOCKED).ok
    assert not scan_source("import socket\n", BLOCKED).ok
    assert not scan_source("from subprocess import run\n", BLOCKED).ok
    assert not scan_source("import cascade.shared.chain as c\n", BLOCKED).ok
    assert not scan_source("__import__('socket')\n", BLOCKED).ok


def test_static_guard_allows_relative_imports():
    assert scan_source("from . import helpers\n", BLOCKED).ok
