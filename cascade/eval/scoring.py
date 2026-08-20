"""Per-window scoring: turn a forecaster into bootstrap-ready components.

Given a list of :class:`EvalWindow` and a numpy-I/O forecaster, produce
per-window :class:`WindowScore` records carrying MASE and the MWSQL components
``(qloss_per_q, abs_target)``. The KOTH decision (:mod:`.koth`) feeds the
king's and challenger's components into the paired bootstrap.

Why components, not a scalar CRPS: the bootstrap needs the numerator and
denominator separately so a bag can aggregate before dividing, and so the
aggregation rule stays a choice made at scoring time rather than baked into
the per-window record.

On GIFT-Eval alignment (this used to say the opposite — see DEC-CA-0009).
The gluonts ``MeanWeightedSumQuantileLoss`` is a pooled ratio
``sum QL_q / sum |y|``, and GIFT-Eval applies it WITHIN one dataset/freq/term
config, where the scale is homogeneous. It then aggregates ACROSS its 97
configs with a *geometric mean* of Seasonal-Naive-normalised values. We had
collapsed those two layers into one: the pooled ratio applied to the whole
heterogeneous pool. Over ~15 orders of magnitude that made three feeds 100%
of the denominator. The current rule — a geometric mean of per-window WQL —
is the analogue of GIFT-Eval's *outer* layer, which is the one that carries
the cross-scale aggregation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .crps import (
    DEFAULT_QUANTILE_LEVELS,
    geomean_wql_from_components,
    mwsql_components,
    mwsql_from_components,
)
from .mase import mase as mase_one
from .seasonality import get_seasonality
from .window import EvalWindow

# CRPS-family aggregation for the round metric. "geomean" (scale-invariant,
# per-window normalised) is the live rule; "pooled" is the pre-2026-07-28 MWSQL,
# kept only to replay archived receipts. See :func:`global_geomean`.
WQL_MODES = ("geomean", "pooled")

# A forecaster: ``f(history_1d, horizon, num_samples) -> (1, num_samples, H)``.
# The archived-checkpoint contract — every 1-D wrapper ever shipped speaks it,
# so it is kept forever (adapted below, never removed).
ForecastFn = Callable[[np.ndarray, int, int], np.ndarray]

# A JOINT forecaster: ``f(history_2d, horizon, num_samples)`` with history
# ``(C, L)`` returning ``(C, num_samples, H)`` — one call per window, all
# channels together, so a multivariate-capable checkpoint can condition its
# forecasts across the variate axis (DEC-CA-0026). At ``C = 1`` this is the
# univariate contract with an extra leading axis.
JointForecastFn = Callable[[np.ndarray, int, int], np.ndarray]


def adapt_per_channel(forecast_fn: ForecastFn) -> JointForecastFn:
    """Lift a 1-D :data:`ForecastFn` to the joint ``(C, L)`` contract.

    The permanent adapter for archived univariate wrappers: each channel is
    forecast independently with the wrapped function and the results stacked.
    At ``C = 1`` (every deployed checkpoint and eval window today) this calls
    the wrapped forecaster exactly once with the identical 1-D history the old
    scorer passed, so scores are byte-identical.
    """

    def joint(history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        history = np.asarray(history, dtype=np.float64)
        parts = []
        for c in range(history.shape[0]):
            samples = np.asarray(forecast_fn(history[c], horizon, num_samples))
            if samples.shape != (1, num_samples, horizon):
                raise ValueError(
                    f"forecaster returned shape {samples.shape}; "
                    f"expected (1, {num_samples}, {horizon})"
                )
            parts.append(samples[0])
        return np.stack(parts, axis=0)

    return joint


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
        domain: coarse bucket from pool metadata (``""`` when absent); drives
            the per-domain diagnostics on the round verdict.
        source: upstream feed id from pool metadata (``None`` when absent);
            the cluster key for the KOTH cluster bootstrap — windows from one
            feed are correlated, so they resample together.
    """

    series_id: str
    mase: float
    qloss_per_q: np.ndarray
    abs_target: float
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
    channel: int = 0
    domain: str = ""
    source: str | None = None


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
    """Score a numpy-I/O univariate forecaster on each window.

    ``forecast_fn(history_1d, horizon, num_samples)`` must return samples of
    shape ``(1, num_samples, horizon)``. Non-finite samples raise — a single
    numerical hiccup must not silently corrupt the comparison.

    This is the archived-wrapper entry point: it lifts the 1-D forecaster
    through :func:`adapt_per_channel` and scores it on the joint path, which
    at ``C = 1`` reproduces the historical behaviour exactly.
    """
    return score_joint_forecaster_on_windows(
        adapt_per_channel(forecast_fn), windows, num_samples, quantile_levels
    )


def score_joint_forecaster_on_windows(
    forecast_fn: JointForecastFn,
    windows: list[EvalWindow],
    num_samples: int,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> list[WindowScore]:
    """Score a joint ``(C, L)`` forecaster on each window — ONE call per window.

    ``forecast_fn(history_2d, horizon, num_samples)`` returns
    ``(C, num_samples, horizon)``. Scoring stays per (window, channel): each
    channel's samples produce one :class:`WindowScore` row, exactly as before —
    only the forecast call is joint, so a multivariate checkpoint can condition
    across channels while the round statistic and the bootstrap see the same
    row shape they always did.
    """
    out: list[WindowScore] = []
    q_tuple = tuple(float(q) for q in quantile_levels)
    for w in windows:
        history = np.asarray(w.history, dtype=np.float64)   # (C, L)
        target = np.asarray(w.target, dtype=np.float64)     # (C, H)
        n_channels = int(target.shape[0])
        horizon = int(target.shape[-1])
        period = _resolve_seasonal_period(w.metadata)
        samples_all = np.asarray(forecast_fn(history, horizon, num_samples))
        if samples_all.shape != (n_channels, num_samples, horizon):
            raise ValueError(
                f"forecaster returned shape {samples_all.shape}; "
                f"expected ({n_channels}, {num_samples}, {horizon})"
            )
        for c in range(n_channels):
            hist_c = history[c]                              # (L,)
            tgt_c = target[c]                                # (H,)
            samples = samples_all[c : c + 1]                 # (1, ns, H)
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
            src = w.metadata.get("source")
            out.append(
                WindowScore(
                    series_id=w.series_id,
                    mase=m,
                    qloss_per_q=qloss_per_q[0],
                    abs_target=float(abs_target[0]),
                    quantile_levels=q_tuple,
                    channel=c,
                    domain=str(w.metadata.get("domain", "") or ""),
                    source=str(src) if src else None,
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


def global_geomean(scores: list[WindowScore], wql_mode: str = "geomean") -> float:
    """Round-level geomean(WQL, geomean MASE) on the observed (non-resampled)
    windows.

    Deliberately the SAME functional form as one bootstrap bag
    (:func:`cascade.eval.bootstrap._bag_geomeans`) evaluated on the identity
    resample, so this is exactly the quantity the LCB is a bound on, and the
    heat screen (which ranks on it) selects on the same metric the duel judges.
    Both halves are geometric means over windows.

    Both halves aggregate in log space for the same reason: per-window ratios
    are heavy-tailed, and an arithmetic mean lets a single exploding window
    dominate the aggregate. That bites hardest in the heat, which screens on a
    fraction of the windows and samples the final does.

    ``wql_mode`` selects the CRPS-family half and exists ONLY so that archived
    receipts stay verifiable:

    * ``"geomean"`` (default, live) — geometric mean of per-window WQL, each
      window normalised by its own ``|y|``. Scale-invariant.
    * ``"pooled"`` — the historical MWSQL: numerators and denominators summed
      across windows, divided once. Retained solely to replay receipts written
      before 2026-07-28; it is scale-dominated on a cross-domain pool (see
      :func:`cascade.eval.crps.wql_per_window`) and must not be used for new
      decisions.
    """
    if not scores:
        return float("nan")
    crps, mase_geo = global_components(scores, wql_mode=wql_mode)
    if crps != crps:  # NaN — no window had a defined WQL
        return float("nan")
    return float(np.sqrt(max(crps, 1e-12) * max(mase_geo, 1e-12)))


def global_components(
    scores: list[WindowScore], wql_mode: str = "geomean"
) -> tuple[float, float]:
    """The two components behind :func:`global_geomean`, ``(crps, mase)``, before
    they are combined into the geomean: ``crps`` is the round-level CRPS-family
    loss and ``mase`` is the *geometric* mean MASE — the same aggregation
    :func:`global_geomean` uses, so ``sqrt(crps * mase)`` reproduces the geomean
    the heat ranks on. NaN-safe — an empty list returns ``(nan, nan)``. Used by
    the heat screener to report a challenger's raw error components, not just
    the geomean. See :func:`global_geomean` for ``wql_mode``."""
    if not scores:
        return (float("nan"), float("nan"))
    if wql_mode not in WQL_MODES:
        raise ValueError(f"wql_mode must be one of {WQL_MODES}; got {wql_mode!r}")
    qloss, abs_t, mase_a = stack_components(scores)
    mase_geo = float(np.exp(np.log(np.maximum(mase_a, 1e-9)).mean()))
    crps = (
        geomean_wql_from_components(qloss, abs_t)
        if wql_mode == "geomean"
        else float(mwsql_from_components(qloss, abs_t))
    )
    return (crps, mase_geo)
