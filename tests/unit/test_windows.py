"""EvalWindow channel promotion + the rotating private window source."""

from __future__ import annotations

import numpy as np
import pytest

from metronome.eval.window import EvalWindow
from metronome.validator.windows import (
    RotatingWindowSource,
    build_windows_from_series,
)


def test_eval_window_promotes_1d_to_single_channel():
    w = EvalWindow(series_id="s", history=np.zeros(80), target=np.zeros(12))
    assert w.history.shape == (1, 80)
    assert w.target.shape == (1, 12)
    assert w.n_channels == 1
    assert w.horizon == 12


def test_eval_window_keeps_multivariate_and_checks_channels():
    w = EvalWindow(series_id="s", history=np.zeros((3, 80)), target=np.zeros((3, 12)))
    assert w.n_channels == 3
    with pytest.raises(ValueError):
        EvalWindow(series_id="s", history=np.zeros((2, 80)), target=np.zeros((1, 12)))


def _pool(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return [
        EvalWindow(series_id=f"p{i}", history=rng.standard_normal(60), target=rng.standard_normal(10))
        for i in range(n)
    ]


def test_rotating_source_is_deterministic_in_round_seed():
    pool = _pool()
    src_a = RotatingWindowSource(pool=tuple(pool))
    src_b = RotatingWindowSource(pool=tuple(pool))  # a second validator
    a = src_a.windows_for_round("block-hash-7", n_windows=10)
    b = src_b.windows_for_round("block-hash-7", n_windows=10)
    # Same round seed → byte-identical slice (validators agree; king/chal pair).
    assert [w.series_id for w in a] == [w.series_id for w in b]
    assert len(a) == 10


def test_rotating_source_rotates_across_rounds():
    src = RotatingWindowSource(pool=tuple(_pool()))
    r1 = {w.series_id for w in src.windows_for_round("round-1", n_windows=10)}
    r2 = {w.series_id for w in src.windows_for_round("round-2", n_windows=10)}
    # A new round permutes differently → the scored slice rotates.
    assert r1 != r2


def test_rotating_source_caps_at_pool_size():
    src = RotatingWindowSource(pool=tuple(_pool(n=8)))
    got = src.windows_for_round("r", n_windows=100)
    assert len(got) == 8  # whole pool; KOTH min_windows decides conclusiveness
    assert len({w.series_id for w in got}) == 8  # distinct, no double-counting


def test_rotating_source_rejects_empty_pool():
    with pytest.raises(ValueError):
        RotatingWindowSource(pool=())


def test_build_windows_from_series_cuts_and_skips_short():
    series = [
        np.arange(100, dtype=np.float64),          # univariate → (1, ·)
        np.zeros((2, 100)),                         # multivariate preserved
        np.arange(5, dtype=np.float64),             # too short → skipped
    ]
    windows = build_windows_from_series(series, context_length=40, horizon=10)
    assert [w.series_id for w in windows] == ["w0", "w1"]
    assert windows[0].history.shape == (1, 40)
    assert windows[0].target.shape == (1, 10)
    assert windows[1].n_channels == 2
    # Target is the last `horizon` steps; history is the `context_length` before.
    np.testing.assert_array_equal(windows[0].target[0], np.arange(90, 100, dtype=np.float64))
    np.testing.assert_array_equal(windows[0].history[0], np.arange(50, 90, dtype=np.float64))
