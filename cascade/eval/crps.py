"""CRPS estimators from sample forecasts.

``mwsql_components`` is the gluonts ``MeanWeightedSumQuantileLoss``
decomposition: per-window ``(qloss_per_q, abs_target)`` tensors that aggregate
across windows by summing numerator and denominator separately and dividing
once at the end. This is what GIFT-Eval reports as "CRPS" and is the
cascade-canonical default — it is robust to near-zero-mean windows in a way
that a per-window divide is not. The paired bootstrap in :mod:`.bootstrap`
resamples the components and applies the divide once per bag.

``crps_ensemble_numpy`` is the Gneiting & Raftery sorted-samples identity,
``E|X−y| − ½E|X−X'|``, kept for tests and as a per-cell ground-truth CRPS.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# 9-level grid, matching GIFT-Eval, so live-bucket numbers stay comparable.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
)


def mwsql_components(
    samples: np.ndarray,
    obs: np.ndarray,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window numerator and denominator for ``MeanWeightedSumQuantileLoss``.

    Returns ``(qloss_per_q, abs_target)`` of shapes ``(B, num_q)`` and ``(B,)``.
    The global statistic is
    ``mean_q [ 2 * sum_window qloss_per_q[q] / sum_window abs_target ]`` —
    summed across windows before dividing, which is the only form robust to
    near-zero-mean windows.

    Args:
        samples: shape ``(B, m, H)`` — ``m`` sample forecasts per window.
        obs: shape ``(B, H)`` — ground-truth target per window.
        quantile_levels: floats in (0, 1). Defaults to the 9-level grid.
    """
    if samples.ndim != 3:
        raise ValueError(f"samples must be (B, m, H); got {samples.shape}")
    if obs.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError(
            f"obs shape {obs.shape} incompatible with samples {samples.shape}; "
            f"expected ({samples.shape[0]}, {samples.shape[2]})"
        )
    qs = np.asarray(quantile_levels, dtype=np.float64)
    if qs.ndim != 1 or qs.size == 0:
        raise ValueError(f"quantile_levels must be a non-empty 1-D sequence; got {qs.shape}")
    if not np.all((qs > 0.0) & (qs < 1.0)):
        raise ValueError("quantile_levels must lie in the open interval (0, 1)")

    q_preds = np.quantile(samples, qs, axis=1)            # (num_q, B, H)
    diff = obs[None, :, :] - q_preds                       # (num_q, B, H)
    q_b = qs[:, None, None]
    pinball = np.maximum(q_b * diff, (q_b - 1.0) * diff)   # (num_q, B, H)

    qloss_per_q = pinball.sum(axis=-1).T                   # (B, num_q)
    abs_target = np.abs(obs).sum(axis=-1)                  # (B,)
    return qloss_per_q.astype(np.float64), abs_target.astype(np.float64)


def mwsql_from_components(
    qloss_per_q: np.ndarray,
    abs_target: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """Reduce per-window components to the global gluonts MWSQL scalar."""
    if qloss_per_q.ndim != 2 or abs_target.ndim != 1:
        raise ValueError(
            f"expected qloss_per_q (N, num_q) and abs_target (N,); "
            f"got {qloss_per_q.shape}, {abs_target.shape}"
        )
    if qloss_per_q.shape[0] != abs_target.shape[0]:
        raise ValueError(
            f"window count mismatch: qloss_per_q {qloss_per_q.shape[0]} vs "
            f"abs_target {abs_target.shape[0]}"
        )
    denom = max(float(abs_target.sum()), eps)
    per_q = 2.0 * qloss_per_q.sum(axis=0) / denom          # (num_q,)
    return float(per_q.mean())


def wql_per_window(
    qloss_per_q: np.ndarray,
    abs_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window weighted quantile loss — each window normalised by its OWN |y|.

    Returns ``(wql, valid)``, both ``(N,)``: the per-window ratio, and a boolean
    mask of the windows it is defined on.

    Why this exists alongside :func:`mwsql_from_components`: that function sums
    numerators and denominators across all windows and divides once. It is
    immune to a zero denominator, but it implicitly weights every window by its
    own ``|y|``. On a pool spanning ~15 orders of magnitude in scale that is not
    a mild bias — three windows (BTC difficulty, two US-debt series) were
    measured at 100% of the denominator on a 2000-window round, so half the
    decision statistic had an effective sample size of 3. Normalising per window
    first makes the aggregate scale-invariant, which is what a cross-domain pool
    needs.

    ``abs_target == 0`` (an all-zero target across the horizon — ~3% of a live
    pool: idle bike-share docks, intermittent counts) carries no scale to
    normalise by, so WQL is genuinely undefined there and the window is MASKED
    rather than floored. Flooring the denominator at eps instead injects values
    of order 1e11 into a log-space aggregate and swamps it.
    """
    if qloss_per_q.ndim != 2 or abs_target.ndim != 1:
        raise ValueError(
            f"expected qloss_per_q (N, num_q) and abs_target (N,); "
            f"got {qloss_per_q.shape}, {abs_target.shape}"
        )
    if qloss_per_q.shape[0] != abs_target.shape[0]:
        raise ValueError(
            f"window count mismatch: qloss_per_q {qloss_per_q.shape[0]} vs "
            f"abs_target {abs_target.shape[0]}"
        )
    valid = np.abs(abs_target) > 0.0
    wql = np.zeros_like(abs_target, dtype=np.float64)
    if valid.any():
        wql[valid] = 2.0 * qloss_per_q[valid].mean(axis=1) / np.abs(abs_target[valid])
    return wql, valid


def geomean_wql_from_components(
    qloss_per_q: np.ndarray,
    abs_target: np.ndarray,
    floor: float = 1e-12,
) -> float:
    """Scale-invariant CRPS-family aggregate: geometric mean of per-window WQL.

    The geometric mean mirrors the MASE half of the round metric, and for the
    same reason: per-window ratios are heavy-tailed and an arithmetic mean lets
    one window dominate. NaN when no window has a defined WQL.
    """
    wql, valid = wql_per_window(qloss_per_q, abs_target)
    if not valid.any():
        return float("nan")
    return float(np.exp(np.log(np.maximum(wql[valid], floor)).mean()))


def crps_ensemble_numpy(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-cell CRPS via the Gneiting–Raftery sorted-samples identity.

    For m sorted samples, ``CRPS = (1/m) sum_i |x_i - y|
    - (1/m^2) sum_i (2i - 1 - m) x_(i)``. Returns shape ``(B, H)``.
    """
    if samples.ndim != 3:
        raise ValueError(f"samples must be (B, m, H); got {samples.shape}")
    if obs.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError(
            f"obs shape {obs.shape} incompatible with samples {samples.shape}"
        )
    m = samples.shape[1]
    abs_dev = np.abs(samples - obs[:, None, :])      # (B, m, H)
    t1 = abs_dev.mean(axis=1)                        # (B, H)
    sorted_samples = np.sort(samples, axis=1)
    i = np.arange(1, m + 1, dtype=np.float64)[None, :, None]
    weights = (2.0 * i - 1.0 - m) / (m * m)
    t2 = (weights * sorted_samples).sum(axis=1)      # (B, H)
    return t1 - t2
