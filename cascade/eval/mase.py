"""Mean Absolute Scaled Error.

MASE scales forecast errors by the in-sample naive seasonal MAE:

    MASE = mean(|y_t - y_hat_t|) / in_sample_seasonal_naive_MAE

Lower is better. MASE is dimensionless across series, which keeps the
paired-bootstrap-on-geomean primitive well-conditioned. Inputs are point
forecasts (the per-step median of the trained model's sample forecast,
computed by the evaluator before calling).
"""

from __future__ import annotations

import numpy as np


def in_sample_naive_mae(history: np.ndarray, seasonal_period: int) -> float:
    """In-sample MAE of the seasonal-naive forecast on ``history``.

    ``y_hat_t = y_{t - period}``. Returns a small positive floor when
    ``history`` is too short or constant so callers never divide by zero.
    """
    if seasonal_period < 1:
        raise ValueError(f"seasonal_period must be >= 1; got {seasonal_period}")
    if history.ndim != 1:
        raise ValueError(f"history must be 1-D; got {history.shape}")
    if len(history) <= seasonal_period:
        mad = float(np.mean(np.abs(history - history.mean())))
        return max(mad, 1e-9)
    diffs = np.abs(history[seasonal_period:] - history[:-seasonal_period])
    scale = float(diffs.mean())
    return max(scale, 1e-9)


def mase(
    point_forecast: np.ndarray,
    obs: np.ndarray,
    history: np.ndarray,
    seasonal_period: int,
) -> float:
    """MASE for a single (forecast, obs, history) triple."""
    if point_forecast.shape != obs.shape:
        raise ValueError(f"point_forecast {point_forecast.shape} != obs {obs.shape}")
    scale = in_sample_naive_mae(history, seasonal_period)
    return float(np.mean(np.abs(obs - point_forecast)) / scale)
