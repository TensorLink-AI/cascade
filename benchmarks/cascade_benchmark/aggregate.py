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
