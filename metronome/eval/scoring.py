"""Per-window scoring: turn a forecaster into bootstrap-ready components.

Given a list of :class:`EvalWindow` and a numpy-I/O forecaster, produce
per-window :class:`WindowScore` records carrying MASE and the MWSQL components
``(qloss_per_q, abs_target)``. The KOTH decision (:mod:`.koth`) feeds the
king's and challenger's components into the paired bootstrap.

Why components, not a scalar CRPS: the gluonts ``MeanWeightedSumQuantileLoss``
is a GLOBAL ratio; dividing per window blows up on near-zero-mean windows. The
bootstrap resamples the components and divides once per bag — the GIFT-Eval
aligned value without the per-window pathology.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .crps import DEFAULT_QUANTILE_LEVELS, mwsql_components, mwsql_from_components
from .mase import mase as mase_one
from .seasonality import get_seasonality
from .window import EvalWindow

# A forecaster: ``f(history_1d, horizon, num_samples) -> (1, num_samples, H)``.
ForecastFn = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True)
class WindowScore:
    """One (window, channel) contribution to a model's aggregated metrics.

    Univariate windows produce one score each (``channel == 0``); a multivariate
    window produces one per channel. The scores are the unit the paired
    bootstrap resamples, so a multivariate window's channels are independent
    rows — king and challenger stay paired as long as they emit the same
    (window, channel) order, which they do (same eval set, same scorer).

    Attributes:
        series_id: from the originating EvalWindow.
        channel: variate index within the window (0 for univariate).
        mase: per-window Hyndman MASE (already a ratio; safe to average).
        qloss_per_q: shape ``(num_q,)`` — sum-over-horizon pinball loss per
            quantile, NOT divided yet; the bootstrap sums then divides once.
        abs_target: ``sum_t |y_t|`` over the horizon — the denominator
            companion to ``qloss_per_q``.
        quantile_levels: grid used to produce ``qloss_per_q``.
    """

    series_id: str
    mase: float
    qloss_per_q: np.ndarray
    abs_target: float
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
    channel: int = 0


def _resolve_seasonal_period(metadata: dict) -> int:
    if "seasonal_period" in metadata:
        return int(metadata["seasonal_period"])
    freq = metadata.get("freq")
    if isinstance(freq, str) and freq:
        return get_seasonality(freq)
    return 1


def score_forecaster_on_windows(
    forecast_fn: ForecastFn,
    windows: list[EvalWindow],
    num_samples: int,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> list[WindowScore]:
    """Score a numpy-I/O forecaster on each window.

    ``forecast_fn(history_1d, horizon, num_samples)`` must return samples of
    shape ``(1, num_samples, horizon)``. Non-finite samples raise — a single
    numerical hiccup must not silently corrupt the comparison.
    """
    out: list[WindowScore] = []
    q_tuple = tuple(float(q) for q in quantile_levels)
    for w in windows:
        history = np.asarray(w.history, dtype=np.float64)   # (C, L)
        target = np.asarray(w.target, dtype=np.float64)     # (C, H)
        horizon = int(target.shape[-1])
        period = _resolve_seasonal_period(w.metadata)
        # Score each channel independently with the univariate forecaster
        # contract. Univariate windows (C == 1) emit exactly one score.
        for c in range(target.shape[0]):
            hist_c = history[c]                              # (L,)
            tgt_c = target[c]                                # (H,)
            samples = forecast_fn(hist_c, horizon, num_samples)
            if samples.shape != (1, num_samples, horizon):
                raise ValueError(
                    f"forecaster returned shape {samples.shape}; "
                    f"expected (1, {num_samples}, {horizon})"
                )
            if not np.isfinite(samples).all():
                raise ValueError(
                    f"forecaster produced non-finite samples on window "
                    f"{w.series_id!r} channel {c}"
                )

            qloss_per_q, abs_target = mwsql_components(
                samples, tgt_c[None, :], quantile_levels=q_tuple
            )
            point = np.median(samples[0], axis=0)           # (H,)
            m = mase_one(point, tgt_c, hist_c, period)
            out.append(
                WindowScore(
                    series_id=w.series_id,
                    mase=m,
                    qloss_per_q=qloss_per_q[0],
                    abs_target=float(abs_target[0]),
                    quantile_levels=q_tuple,
                    channel=c,
                )
            )
    return out


def stack_components(
    scores: list[WindowScore],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack WindowScores into ``(qloss_per_q, abs_target, mase)`` arrays of
    shapes ``(N, num_q)``, ``(N,)``, ``(N,)``."""
    if not scores:
        nq = len(DEFAULT_QUANTILE_LEVELS)
        return (
            np.zeros((0, nq), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    qloss = np.stack([s.qloss_per_q for s in scores], axis=0).astype(np.float64)
    abs_t = np.asarray([s.abs_target for s in scores], dtype=np.float64)
    mase_a = np.asarray([s.mase for s in scores], dtype=np.float64)
    return qloss, abs_t, mase_a


def global_geomean(scores: list[WindowScore]) -> float:
    """Round-level geomean(MWSQL, mean MASE) on the observed (non-resampled)
    windows. Reported for diagnostics; the bootstrap LCB gates dethroning."""
    if not scores:
        return float("nan")
    qloss, abs_t, mase_a = stack_components(scores)
    mwsql = mwsql_from_components(qloss, abs_t)
    mase_mean = float(mase_a.mean())
    return float(np.sqrt(max(mwsql, 1e-12) * max(mase_mean, 1e-12)))
