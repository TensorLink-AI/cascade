"""Pure-numpy metrics for the FEV suite (``suites/fev_bench.py``).

Kept out of the suite module so they import without gluonts / fev / torch (the
suite registry imports the gluonts-backed predictor at package import), and so
the per-task numbers can be pinned by unit tests in any environment.

Conventions match the rest of the sidecar:

* **CRPS** — weighted quantile loss over the 9-level grid, pooled over every
  series and window of a task: ``mean_q 2·Σ pinball_q / Σ|y|`` (gluonts'
  ``mean_weighted_sum_quantile_loss`` with ``axis=None``).
* **MASE** — per series and window, ``mean|y − median| / mean|y_t − y_{t−m}|``
  over the history (seasonal naive in-sample scale), then the arithmetic mean
  over series/windows. A series with a zero or undefined scale contributes no
  MASE term (it has no seasonal-naive error to scale by).
* The **seasonal-naive** baseline is scored with the same two functions, its
  point forecast repeated across every quantile, so the suite headline is the
  Seasonal-Naive-normalised geometric mean like GIFT-Eval / BOOM / TIME.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def seasonal_scale(history: np.ndarray, m: int) -> float | None:
    """In-sample seasonal-naive MAE of ``history`` at lag ``m`` (falls back to
    lag 1 when the history is shorter than ``m + 1``); ``None`` when undefined
    or zero."""
    h = np.asarray(history, dtype=np.float64)
    h = h[np.isfinite(h)]
    m = max(1, int(m))
    if h.size <= m:
        m = 1
    if h.size <= m:
        return None
    s = float(np.mean(np.abs(h[m:] - h[:-m])))
    return s if s > 0 and math.isfinite(s) else None


def seasonal_naive_forecast(history: np.ndarray, horizon: int, m: int) -> np.ndarray:
    """Repeat the last observed season (lag 1 when the history is shorter)."""
    h = np.asarray(history, dtype=np.float64)
    m = max(1, int(m))
    if h.size < m:
        m = 1
    if h.size == 0:
        return np.zeros(int(horizon))
    season = h[-m:]
    reps = -(-int(horizon) // m)
    return np.tile(season, reps)[: int(horizon)]


class TaskAccumulator:
    """Pools one task's windows/series into a ``(CRPS, MASE)`` pair."""

    def __init__(self, levels: Sequence[float] = QUANTILE_LEVELS):
        self.levels = np.asarray(levels, dtype=np.float64)
        self._ql = np.zeros(len(levels))      # Σ pinball per level
        self._abs = 0.0                        # Σ|y|
        self._mase: list[float] = []
        self.n_series = 0

    def add(self, history: np.ndarray, y_true: np.ndarray, q: np.ndarray, m: int) -> None:
        """``q``: ``(horizon, num_q)`` quantile forecasts; non-finite targets are
        masked out of every term."""
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (y.size, len(self.levels)):
            raise ValueError(f"forecast shape {q.shape} != {(y.size, len(self.levels))}")
        ok = np.isfinite(y)
        if not ok.any():
            return
        y, q = y[ok], q[ok]
        if not np.isfinite(q).all():
            raise ValueError("non-finite quantile forecast on an observed target")
        err = y[:, None] - q
        self._ql += np.maximum(self.levels * err, (self.levels - 1.0) * err).sum(axis=0)
        self._abs += float(np.abs(y).sum())
        med = q[:, int(np.argmin(np.abs(self.levels - 0.5)))]
        scale = seasonal_scale(history, m)
        if scale is not None:
            self._mase.append(float(np.mean(np.abs(y - med))) / scale)
        self.n_series += 1

    def result(self) -> tuple[float | None, float | None]:
        crps = (float(np.mean(2.0 * self._ql / self._abs))
                if self._abs > 0 else None)
        mase = float(np.mean(self._mase)) if self._mase else None
        return crps, mase


def point_as_quantiles(point: np.ndarray, num_q: int) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64).reshape(-1, 1)
    return np.repeat(p, num_q, axis=1)


def normalised_geomean(model: Iterable[float | None], naive: Iterable[float | None]) -> float:
    """Geometric mean of ``model / naive`` over tasks where both are finite and
    positive (NaN when none are) — the headline convention of the other suites."""
    logs = []
    for a, b in zip(model, naive, strict=True):
        if a is None or b is None:
            continue
        if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
            continue
        logs.append(math.log(a / b))
    return math.exp(sum(logs) / len(logs)) if logs else float("nan")
