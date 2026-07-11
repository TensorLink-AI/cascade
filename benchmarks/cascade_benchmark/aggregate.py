"""Official GIFT-Eval / BOOM aggregation — a faithful port of DataDog's
``boom/utils/leaderboard.py`` (the same methodology GIFT-Eval's leaderboard
uses).

The headline benchmark number is NOT a plain mean of per-dataset metrics. It is
the **shifted geometric mean, across datasets, of each metric normalized by the
Seasonal-Naive baseline**:

    shifted_gmean_i( model_metric_i / seasonal_naive_metric_i )

with two refinements taken verbatim from the upstream code:

* **Zero-inflated split.** Datasets where the Seasonal-Naive MASE is 0 (so the
  MASE ratio is undefined) are scored in a separate pool using MAE instead of
  MASE. BOOM additionally excludes a fixed ``LOW_VARIANCE_DATASETS`` set from the
  main pool.
* **Invalid-value handling.** ``±inf`` → ``nan`` → filled with the column mean,
  before the ratio (``replace_invalid_values``), matching upstream order.

We normalize against the *vendored official Seasonal-Naive results* (the same
baseline CSVs the leaderboards ship), so the numbers are leaderboard-comparable.
"""

from __future__ import annotations

import numpy as np


def shifted_gmean(x: np.ndarray, epsilon: float = 1e-5) -> float:
    """``exp(mean(log(x + eps))) - eps`` — upstream ``leaderboard.shifted_gmean``."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.exp(np.sum(np.log(x + epsilon)) / x.shape[0]) - epsilon)


def _clean(values: list) -> np.ndarray | None:
    """``replace_invalid_values``: None/±inf → nan → filled with the finite mean."""
    a = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    a[np.isinf(a)] = np.nan
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    a[np.isnan(a)] = finite.mean()
    return a


def _scaled_gmean(pool: list[tuple[dict, dict]], metric: str) -> float:
    """Clean model + naive columns, divide, clean again, shifted-gmean — the
    ``replace_invalid_values`` → ``scale_by_naive`` → ``shifted_gmean`` pipeline."""
    model = _clean([r.get(metric) for r, _ in pool])
    naive = _clean([b.get(metric) for _, b in pool])
    if model is None or naive is None:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = _clean(list(model / naive))  # ÷0 → inf → re-cleaned to column mean
    return shifted_gmean(ratio) if ratio is not None else float("nan")


# Metric-name aliases (case-insensitive) → the canonical Cascade key. TIME's
# timebench writes its own metric names into metrics.npz; GIFT-Eval/BOOM already
# emit lowercase crps/mase. Matching case-insensitively over these aliases makes
# the three suites agree on one pair of numbers regardless of upstream casing.
_CRPS_ALIASES = ("crps", "wql", "mean_weighted_sum_quantile_loss")
_MASE_ALIASES = ("mase",)


def _ci_get(d: dict, aliases: tuple[str, ...]):
    """Case-insensitive lookup of the first matching alias in ``d`` (else None)."""
    low = {str(k).lower(): v for k, v in d.items()}
    for a in aliases:
        if a in low:
            return low[a]
    return None


def seasonal_naive_quantiles(
    target, horizon: int, season: int, n_quantiles: int
) -> np.ndarray:
    """Seasonal-Naive forecast as a degenerate ``(n_quantiles, V, H)`` quantile
    array: the point forecast ``y[L - s + (h mod s)]`` repeated across every
    quantile level. This is the standard Seasonal-Naive reference the leaderboards
    (GIFT-Eval, BOOM, and TIME) normalize against, expressed on the caller's
    quantile grid — a point forecast, so all quantile levels coincide (its
    weighted quantile loss reduces to weighted MAE). ``target`` is one instance's
    context, ``(V, L)`` or ``(L,)``; ``season`` the seasonal period."""
    t = np.atleast_2d(np.asarray(target, dtype=np.float64))  # (V, L)
    length = t.shape[1]
    s = max(1, int(season))
    idx = np.array(
        [min(max((length - s) + (h % s), 0), length - 1) for h in range(int(horizon))],
        dtype=int,
    )
    point = t[:, idx]  # (V, H)
    return np.repeat(point[np.newaxis, :, :], int(n_quantiles), axis=0)  # (num_q, V, H)


def normalize_time(model_rows: list[dict], baseline_rows: list[dict]) -> dict:
    """TIME aggregation, made to match GIFT-Eval/BOOM: the per-task ratio of the
    model metric to the Seasonal-Naive metric, then the **shifted geometric mean**
    across tasks — instead of a plain mean of raw metrics. ``model_rows[i]`` and
    ``baseline_rows[i]`` are the per-task metric dicts (timebench's own keys) for
    the same task, in the same order. Returns ``{"crps": .., "mase": ..}`` in the
    canonical lowercase keys the other suites use, skipping a metric whose baseline
    is unusable. Reuses ``_scaled_gmean`` so the invalid-value / div-0 handling is
    byte-identical to the official aggregate."""
    out: dict = {}
    for canon, aliases in (("crps", _CRPS_ALIASES), ("mase", _MASE_ALIASES)):
        pool = [
            ({canon: _ci_get(mr, aliases)}, {canon: _ci_get(br, aliases)})
            for mr, br in zip(model_rows, baseline_rows, strict=False)
        ]
        val = _scaled_gmean(pool, canon)
        if np.isfinite(val):
            out[canon] = val
    return out


def official_aggregate(
    rows: list[dict],
    baseline: dict[str, dict],
    *,
    low_variance: frozenset[str] = frozenset(),
) -> dict:
    """Aggregate per-config model metrics into the official leaderboard numbers.

    ``rows`` are ``{"full": "<name>/<freq>/<term>", "MASE", "MAE", "CRPS"}`` for
    the model; ``baseline`` maps the same ``full`` key to the Seasonal-Naive
    metrics. ``low_variance`` is the set of base dataset names to exclude from the
    main pool (BOOM only). Returns the scaled non-zero pool (``mase``, ``crps``)
    plus the zero-inflated pool (``mae_zero``, ``crps_zero``) and counts.
    """
    matched = [(r, baseline[r["full"]]) for r in rows if r["full"] in baseline]

    non_zero: list[tuple[dict, dict]] = []
    zero: list[tuple[dict, dict]] = []
    for r, b in matched:
        bmase = b.get("MASE")
        is_zero = bmase is None or bmase == 0 or r["full"].split("/")[0] in low_variance
        (zero if is_zero else non_zero).append((r, b))

    out: dict = {
        "n_scored": len(matched),
        "n_skipped": len(rows) - len(matched),
        "n_nonzero": len(non_zero),
        "n_zero": len(zero),
    }
    if non_zero:
        out["mase"] = _scaled_gmean(non_zero, "MASE")
        out["crps"] = _scaled_gmean(non_zero, "CRPS")
    if zero:
        out["mae_zero"] = _scaled_gmean(zero, "MAE")
        out["crps_zero"] = _scaled_gmean(zero, "CRPS")
    return out
