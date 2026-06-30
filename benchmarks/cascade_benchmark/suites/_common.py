"""Shared gluonts scoring for the GIFT-Eval and BOOM runners.

Both expose datasets through gift-eval's ``Dataset`` class, so scoring one
config is identical and matches gift-eval's reference runner
(``notebooks/naive.ipynb``): the same ``evaluate_model`` call with seasonality
from the dataset frequency. We return the per-config MASE / MAE / CRPS; the
cross-dataset aggregation (Seasonal-Naive normalized shifted-geomean) lives in
``cascade_benchmark.aggregate`` so it matches the official leaderboards.
"""

from __future__ import annotations

# 9-level grid — matches GIFT-Eval and cascade's DEFAULT_QUANTILE_LEVELS.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def build_dataset(name: str, term: str, *, storage_env_var: str = "GIFT_EVAL"):
    """Construct a gift-eval ``Dataset`` following the reference runner's
    ``to_univariate`` rule (multivariate series split into univariate, matching
    the leaderboard). Returns ``None`` if it fails to load so the caller can
    skip it rather than abort the sweep.
    """
    from gift_eval.data import Dataset

    try:
        probe = Dataset(name=name, term=term, to_univariate=False, storage_env_var=storage_env_var)
        to_univariate = probe.target_dim != 1
        return Dataset(
            name=name, term=term, to_univariate=to_univariate, storage_env_var=storage_env_var
        )
    except Exception:  # noqa: BLE001 — unknown/invalid (name, term) combo → skip
        return None


def score_dataset(ds, checkpoint_dir: str, *, num_samples: int, device: str) -> dict:
    """Score the checkpoint on one ``Dataset`` and return ``{MASE, MAE, CRPS}``.

    The ``evaluate_model`` call mirrors gift-eval's reference runner exactly
    (kwargs + seasonality), so the per-config numbers are leaderboard-faithful.
    Column names ``MASE[0.5]`` / ``MAE[0.5]`` / ``mean_weighted_sum_quantile_loss``
    are verified against ``naive.ipynb``.
    """
    from gluonts.ev.metrics import MAE, MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    from ..predictor import CheckpointPredictor

    season_length = get_seasonality(ds.freq)
    predictor = CheckpointPredictor(
        checkpoint_dir,
        prediction_length=ds.prediction_length,
        num_samples=num_samples,
        device=device,
    )
    res = evaluate_model(
        predictor,
        test_data=ds.test_data,
        metrics=[MASE(), MAE(), MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILE_LEVELS))],
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    return {
        "MASE": float(res["MASE[0.5]"][0]),
        "MAE": float(res["MAE[0.5]"][0]),
        "CRPS": float(res["mean_weighted_sum_quantile_loss"][0]),
    }
