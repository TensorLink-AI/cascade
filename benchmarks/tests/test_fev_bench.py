"""FEV suite plumbing with a fake fev task + fake checkpoint wrapper (no network,
no fev install): task naming, the univariate scope rule, and per-task scoring
of model vs in-suite Seasonal-Naive."""

from __future__ import annotations

import numpy as np
from cascade_benchmark.suites.fev_bench import score_task, target_column, task_id


class _Window:
    def __init__(self, past, truth):
        self._past, self._truth = past, truth

    def get_input_data(self):
        return {"target": self._past}, {}

    def get_ground_truth(self):
        return {"target": self._truth}


class _Task:
    task_name = "toy/D/h4"
    target = "target"
    horizon = 4
    seasonality = 4

    def __init__(self, windows):
        self._windows = windows

    def iter_windows(self):
        return iter(self._windows)


class _Oracle:
    """Forecasts the true continuation of a period-4 ramp at every quantile."""
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def forecast_quantiles_batch(self, histories, horizon):
        out = []
        for h in histories:
            nxt = np.array([h[-4 + (i % 4)] for i in range(horizon)]) + 1.0
            out.append(np.repeat(nxt[:, None], 9, axis=1))
        return np.stack(out)


def test_naming_and_scope():
    assert task_id(_Task([])) == "toy/D/h4"
    t = _Task([])
    assert target_column(t) == "target"
    t.target = ["a", "b"]
    assert target_column(t) is None


def test_score_task_model_vs_naive():
    hist = [np.arange(40.0)]                      # trend: naive lags by one season
    truth = [np.arange(40.0, 44.0)]
    r = score_task(_Task([_Window(hist, truth)]), _Oracle())
    assert r["n_windows"] == 1 and r["n_series"] == 1
    assert r["CRPS"] < r["snaive_CRPS"]
    assert r["MASE"] < r["snaive_MASE"]
