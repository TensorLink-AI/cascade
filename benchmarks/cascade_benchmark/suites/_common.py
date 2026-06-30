"""Shared gluonts evaluation loop used by the GIFT-Eval and BOOM runners.

Both benchmarks expose their datasets through gift-eval's ``Dataset`` class
(gluonts-interface), so scoring is the same: build a
:class:`~cascade_benchmark.predictor.CheckpointPredictor` sized to each
dataset's prediction length, run gluonts' evaluator, and average the canonical
metrics across datasets. We report CRPS (= MeanWeightedSumQuantileLoss, the
GIFT-Eval headline metric) and MASE — the same two cascade scores in-protocol,
so the numbers are directly relatable.
"""

from __future__ import annotations

import numpy as np

from ..predictor import CheckpointPredictor

# 9-level grid — matches GIFT-Eval and cascade's own DEFAULT_QUANTILE_LEVELS.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def evaluate_datasets(
    datasets,
    checkpoint_dir: str,
    *,
    num_samples: int,
    device: str,
) -> tuple[dict, int]:
    """Score the checkpoint on an iterable of gift-eval ``Dataset`` objects.

    Returns ``(metrics, n_datasets)`` where ``metrics`` holds the cross-dataset
    mean CRPS and MASE. A dataset that errors is skipped (logged to the per-suite
    detail by the caller via the returned count) rather than aborting the sweep.
    """
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model

    metrics = [
        MASE(),
        MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILE_LEVELS)),
    ]

    crps_vals: list[float] = []
    mase_vals: list[float] = []
    used = 0
    for ds in datasets:
        predictor = CheckpointPredictor(
            checkpoint_dir,
            prediction_length=ds.prediction_length,
            num_samples=num_samples,
            device=device,
        )
        res = evaluate_model(
            predictor,
            test_data=ds.test_data,
            metrics=metrics,
            axis=None,
            batch_size=1024,
            seasonality=getattr(ds, "seasonality", 1),
        )
        # evaluate_model returns a one-row frame; pull the scalar columns.
        crps = float(np.asarray(res["mean_weighted_sum_quantile_loss"]).reshape(-1)[0])
        mase = float(np.asarray(res["MASE[0.5]"] if "MASE[0.5]" in res else res["MASE"]).reshape(-1)[0])
        if np.isfinite(crps):
            crps_vals.append(crps)
        if np.isfinite(mase):
            mase_vals.append(mase)
        used += 1

    out = {
        "crps": float(np.mean(crps_vals)) if crps_vals else float("nan"),
        "mase": float(np.mean(mase_vals)) if mase_vals else float("nan"),
    }
    return out, used
