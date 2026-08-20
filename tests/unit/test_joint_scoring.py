"""Joint (C, L) forecaster contract + the series_id cluster fallback
(DEC-CA-0020 G6 / DEC-CA-0026 Stage 0)."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.koth import _window_clusters
from cascade.eval.scoring import (
    WindowScore,
    adapt_per_channel,
    score_forecaster_on_windows,
    score_joint_forecaster_on_windows,
)
from cascade.eval.window import EvalWindow


def _uni_windows(n=6, h=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        hist = np.cumsum(rng.standard_normal(80)).astype(np.float64)
        tgt = hist[-1] + np.cumsum(rng.standard_normal(h)).astype(np.float64)
        out.append(
            EvalWindow(series_id=f"u{i}", history=hist, target=tgt, metadata={"freq": "D"})
        )
    return out


def _mv_window(c=3, h=12, seed=1, series_id="mv0", metadata=None):
    rng = np.random.default_rng(seed)
    hist = np.cumsum(rng.standard_normal((c, 80)), axis=-1)
    tgt = hist[:, -1:] + np.cumsum(rng.standard_normal((c, h)), axis=-1)
    return EvalWindow(
        series_id=series_id, history=hist, target=tgt, metadata=metadata or {}
    )


def _uni_forecaster(history, horizon, num_samples):
    rng = np.random.default_rng(int(abs(history[-1] * 1e6)) % (2**32))
    return history[-1] + 0.1 * rng.standard_normal((1, num_samples, horizon))


# ── equivalence: the adapter reproduces the historical 1-D path exactly ──────


def test_adapter_path_is_byte_identical_at_c1():
    windows = _uni_windows()
    direct = score_forecaster_on_windows(_uni_forecaster, windows, num_samples=20)
    joint = score_joint_forecaster_on_windows(
        adapt_per_channel(_uni_forecaster), windows, num_samples=20
    )
    assert len(direct) == len(joint) == len(windows)
    for a, b in zip(direct, joint, strict=True):
        assert a.series_id == b.series_id and a.channel == b.channel
        assert a.mase == b.mase
        assert np.array_equal(a.qloss_per_q, b.qloss_per_q)
        assert a.abs_target == b.abs_target


# ── joint contract on a multivariate window ──────────────────────────────────


def test_joint_forecaster_called_once_per_window_with_all_channels():
    w = _mv_window(c=3)
    calls = []

    def joint(history, horizon, num_samples):
        calls.append(np.asarray(history).shape)
        c = history.shape[0]
        return np.repeat(history[:, None, -1:], num_samples, axis=1).repeat(
            horizon, axis=2
        ) + 0.01 * np.random.default_rng(0).standard_normal((c, num_samples, horizon))

    scores = score_joint_forecaster_on_windows(joint, [w], num_samples=10)
    assert calls == [(3, 80)]
    assert [s.channel for s in scores] == [0, 1, 2]
    assert all(s.series_id == "mv0" for s in scores)


def test_joint_forecaster_wrong_channel_count_rejected():
    w = _mv_window(c=3)

    def wrong(history, horizon, num_samples):
        return np.zeros((1, num_samples, horizon))

    with pytest.raises(ValueError, match="expected \\(3,"):
        score_joint_forecaster_on_windows(wrong, [w], num_samples=10)


def test_adapter_scores_mv_window_per_channel():
    # An archived 1-D wrapper still scores a multivariate window — one call
    # per channel, three rows out.
    w = _mv_window(c=3)
    scores = score_joint_forecaster_on_windows(
        adapt_per_channel(_uni_forecaster), [w], num_samples=10
    )
    assert [s.channel for s in scores] == [0, 1, 2]


# ── the cluster fallback: series_id, never per-row ───────────────────────────


def _row(series_id, channel=0, source=None):
    return WindowScore(
        series_id=series_id,
        mase=1.0,
        qloss_per_q=np.ones(9),
        abs_target=1.0,
        channel=channel,
        source=source,
    )


def test_sourceless_univariate_rows_stay_one_cluster_per_window():
    rows = [_row(f"w{i}") for i in range(5)]
    labels, n = _window_clusters(rows)
    assert n == 5 and len(set(labels)) == 5


def test_sourceless_multichannel_window_is_one_cluster_not_c():
    # The DEC-CA-0026 item-3 defence: 4 channels of one window must bootstrap
    # as ONE observation, not four.
    rows = [_row("mv0", channel=c) for c in range(4)] + [_row("w1")]
    labels, n = _window_clusters(rows)
    assert n == 2
    assert labels[0] == labels[1] == labels[2] == labels[3] != labels[4]


def test_source_still_wins_over_series_id():
    rows = [_row("w0", source="feedA"), _row("w1", source="feedA"), _row("w2")]
    labels, n = _window_clusters(rows)
    assert n == 2 and labels[0] == labels[1]


# ── evaluator probe: forecast_joint used when present, adapter otherwise ─────


def _write_wrapper(tmp_path, body: str):
    (tmp_path / "forecast_wrapper.py").write_text(body, encoding="utf-8")
    return tmp_path


def test_evaluator_lifts_archived_1d_wrapper(tmp_path):
    from cascade.validator.evaluator import load_forecaster

    _write_wrapper(
        tmp_path,
        "import numpy as np\n"
        "class Wrapper:\n"
        "    def __init__(self, d, device='cpu'):\n"
        "        pass\n"
        "    def forecast(self, history, horizon, num_samples):\n"
        "        return np.zeros((1, num_samples, horizon)) + history[-1]\n",
    )
    fn = load_forecaster(tmp_path)
    out = fn(np.ones((2, 30)), 5, 7)
    assert out.shape == (2, 7, 5)


def test_evaluator_prefers_forecast_joint(tmp_path):
    from cascade.validator.evaluator import load_forecaster

    _write_wrapper(
        tmp_path,
        "import numpy as np\n"
        "class Wrapper:\n"
        "    def __init__(self, d, device='cpu'):\n"
        "        pass\n"
        "    def forecast(self, history, horizon, num_samples):\n"
        "        raise AssertionError('joint path must win')\n"
        "    def forecast_joint(self, history, horizon, num_samples):\n"
        "        c = history.shape[0]\n"
        "        return np.full((c, num_samples, horizon), 42.0)\n",
    )
    fn = load_forecaster(tmp_path)
    out = fn(np.ones((3, 30)), 5, 7)
    assert out.shape == (3, 7, 5) and float(out[0, 0, 0]) == 42.0
