"""TIME runner — the contamination-resistant "It's TIME" benchmark.

  paper:   "It's TIME: Towards the Next Generation of Time Series Forecasting
            Benchmarks" (arXiv:2602.12147)
  data:    https://huggingface.co/datasets/Real-TSF/TIME  (50 datasets, 98 tasks)
  code:    https://github.com/zqiao11/TIME  (``timebench`` package)

Unlike GIFT-Eval / BOOM, TIME is *not* gluonts-driven: a model supplies quantile
forecasts directly as an array of shape
``(num_instances, num_quantiles, num_variates, prediction_length)``, and
``timebench`` computes the per-window metrics itself. We mirror TIME's own
``experiments/chronos2.py`` flow exactly, swapping their Chronos pipeline for the
checkpoint's ``forecast`` wrapper: draw sample paths, reduce them to the TIME
quantile grid, hand the array to ``save_window_predictions`` (which writes
``metrics.npz`` using TIME's own metric code — so the numbers match the
leaderboard), then read that back and average.

Enable by pointing ``CASCADE_BENCH_TIME_DATASET`` (or ``TIME_DATASET``) at the
Real-TSF/TIME data. ``CASCADE_BENCH_TIME_DATASETS`` optionally restricts the
``name/freq`` configs (default: all from TIME's bundled config).
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from ..predictor import _load_wrapper
from ..results import SuiteResult

# TIME's default quantile grid (experiments/chronos2.py) — identical to cascade's.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _wrapper_quantile_grid_matches(wrapper) -> bool:
    """True when the wrapper exposes the quantile head on TIME's exact grid."""
    levels = getattr(wrapper, "quantile_levels", None)
    return (
        hasattr(wrapper, "forecast_quantiles_batch")
        and levels is not None
        and len(levels) == len(QUANTILE_LEVELS)
        and np.allclose([float(v) for v in levels], QUANTILE_LEVELS)
    )


def _instance_quantiles(wrapper, target, horizon: int, num_samples: int) -> np.ndarray:
    """One TIME eval instance's forecast, shaped ``(num_q, num_variates, H)``.

    ``target`` is the instance's context: ``(variates, time)`` or ``(time,)``.
    Each variate is forecast independently (cascade's per-channel convention).
    CPM checkpoints hand back the quantile head directly on TIME's grid; older
    checkpoints draw sample paths and collapse them to the grid.
    """
    t = np.atleast_2d(np.asarray(target, dtype=np.float64))  # (V, L)
    if _wrapper_quantile_grid_matches(wrapper):
        q = np.asarray(wrapper.forecast_quantiles_batch(list(t), horizon))  # (V, H, num_q)
        return np.transpose(q, (2, 0, 1))  # (num_q, V, H)
    per_variate = []
    for v in range(t.shape[0]):
        samples = np.asarray(wrapper.forecast(t[v], horizon, num_samples))[0]  # (S, H)
        per_variate.append(np.quantile(samples, QUANTILE_LEVELS, axis=0))      # (num_q, H)
    return np.stack(per_variate, axis=1)  # (num_q, V, H)


def _tasks(config, max_tasks: int | None):
    """Yield ``(dataset_name, term)`` pairs, optionally capped for a smoke run."""
    from timebench.evaluation.utils import get_available_terms

    override = os.environ.get("CASCADE_BENCH_TIME_DATASETS", "").strip()
    if override:
        # Each entry is a full dataset key (e.g. "WUI_Global/Q") — the config keys
        # themselves contain a "/", so the spec must NOT be split on it. Terms are
        # then resolved from the config (its defined short/medium/long).
        names_terms = [(spec.strip(), None) for spec in override.split(",") if spec.strip()]
    else:
        names_terms = [(name, None) for name in config.get("datasets", {})]

    n = 0
    for name, terms in names_terms:
        for term in terms or get_available_terms(name, config):
            yield name, term
            n += 1
            if max_tasks and n >= max_tasks:
                return


def _metrics_from_quantiles(dataset, fc_quantiles, ds_config, output_base_dir, season, model):
    """Write a quantile forecast through TIME's own saver and read back the
    per-window metrics it computes, averaged over windows. ``output_base_dir`` is
    unique per role (model vs Seasonal-Naive baseline) so their metrics.npz files
    never collide under a shared task dir."""
    from timebench.evaluation.saver import save_window_predictions

    save_window_predictions(
        dataset=dataset,
        fc_quantiles=fc_quantiles,
        ds_config=ds_config,
        output_base_dir=str(output_base_dir),
        seasonality=season,
        model_hyperparams={"model": model},
        quantile_levels=QUANTILE_LEVELS,
    )
    metrics_npz = Path(output_base_dir) / ds_config / "metrics.npz"
    if not metrics_npz.is_file():
        # Tolerate a deeper timebench layout, but never search outside this
        # role's own subtree — a wider glob could misattribute another task's
        # (or the other role's) metrics.
        hits = sorted((Path(output_base_dir) / ds_config).rglob("metrics.npz"))
        if not hits:
            raise FileNotFoundError(f"metrics.npz not written for {ds_config}")
        metrics_npz = hits[0]
    with np.load(metrics_npz) as data:
        return {k: float(np.nanmean(data[k])) for k in data.files}


def _score_one(
    wrapper, name: str, term: str, config, out_dir: str, num_samples: int,
    batch_size: int = 64,
) -> tuple[dict, dict]:
    """Run one TIME task and return ``(model_metrics, seasonal_naive_metrics)``,
    each ``{metric_name: mean_value}`` computed by TIME's own metric code. The
    Seasonal-Naive baseline is scored through the identical saver+metric path, so
    the caller can normalize the model metric by the baseline metric per task (the
    ratio-then-geomean the GIFT-Eval / BOOM leaderboards — and TIME's own — use)."""
    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.data import Dataset, get_dataset_settings

    from ..aggregate import seasonal_naive_quantiles

    settings = get_dataset_settings(name, term, config)
    pred_len = settings.get("prediction_length")
    dataset = Dataset(
        name=name,
        term=term,
        to_univariate=False,
        prediction_length=pred_len,
        test_length=settings.get("test_length"),
        val_length=settings.get("val_length"),
    )
    season = get_seasonality(dataset.freq)

    eval_inputs = list(dataset.test_data.input)
    if _wrapper_quantile_grid_matches(wrapper):
        # Quantile head, batched across instances *and* variates: flatten every
        # (instance, variate) context into one job list, chunk it through
        # forecast_quantiles_batch, and reassemble TIME's (N, num_q, V, H).
        jobs: list[np.ndarray] = []
        counts: list[int] = []
        for d in eval_inputs:
            t = np.atleast_2d(np.asarray(d["target"], dtype=np.float64))
            jobs.extend(t)
            counts.append(t.shape[0])
        chunks = [
            np.asarray(wrapper.forecast_quantiles_batch(jobs[i : i + batch_size], pred_len))
            for i in range(0, len(jobs), batch_size)
        ]
        q = np.concatenate(chunks, axis=0)  # (sum_V, H, num_q)
        fc, i = [], 0
        for n_var in counts:
            per = q[i : i + n_var]
            i += n_var
            fc.append(np.transpose(per, (2, 0, 1))[np.newaxis, ...])  # (1, num_q, V, H)
    else:
        fc = [
            _instance_quantiles(wrapper, d["target"], pred_len, num_samples)[np.newaxis, ...]
            for d in eval_inputs
        ]
    fc_quantiles = np.concatenate(fc, axis=0)  # (N, num_q, V, H)

    # Seasonal-Naive baseline on the SAME instances/grid, scored the SAME way — so
    # dividing model by baseline per task is apples-to-apples.
    snaive = np.concatenate(
        [
            seasonal_naive_quantiles(d["target"], pred_len, season, len(QUANTILE_LEVELS))[np.newaxis, ...]
            for d in eval_inputs
        ],
        axis=0,
    )

    ds_config = f"{name}/{term}"
    model_metrics = _metrics_from_quantiles(
        dataset, fc_quantiles, ds_config, Path(out_dir) / "model", season, "cascade",
    )
    snaive_metrics = _metrics_from_quantiles(
        dataset, snaive, ds_config, Path(out_dir) / "snaive", season, "seasonal_naive",
    )
    return model_metrics, snaive_metrics


def run(
    checkpoint_dir: str,
    *,
    num_samples: int = 100,
    max_series: int | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> SuiteResult:
    # TIME needs the dataset location; mirror TIME's TIME_DATASET env var.
    ds_path = os.environ.get("CASCADE_BENCH_TIME_DATASET") or os.environ.get("TIME_DATASET")
    if ds_path:
        os.environ.setdefault("TIME_DATASET", ds_path)
    elif not os.environ.get("CASCADE_BENCH_TIME_DATASETS"):
        return SuiteResult(
            suite="time",
            status="skipped",
            detail=(
                "TIME dataset not configured; set CASCADE_BENCH_TIME_DATASET (or "
                "TIME_DATASET) to the Real-TSF/TIME data path."
            ),
        )
    try:
        from timebench.evaluation.data import load_dataset_config

        from ..aggregate import normalize_time

        config = load_dataset_config(None)
        wrapper = _load_wrapper(Path(checkpoint_dir), device)

        model_rows: list[dict] = []
        snaive_rows: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="cascade-time-") as out_dir:
            for j, (name, term) in enumerate(_tasks(config, max_series)):
                try:
                    # Per-task subdir keeps every task's (and role's) metrics.npz
                    # isolated under the shared temp root.
                    model_m, snaive_m = _score_one(
                        wrapper, name, term, config, str(Path(out_dir) / str(j)),
                        num_samples, batch_size,
                    )
                except Exception:  # noqa: BLE001 — one task must not abort the sweep
                    continue
                model_rows.append(model_m)
                snaive_rows.append(snaive_m)

        if not model_rows:
            return SuiteResult(suite="time", status="error", detail="no TIME tasks scored")

        # Parity with GIFT-Eval/BOOM (and TIME's own leaderboard): per-task ratio to
        # the Seasonal-Naive baseline, aggregated by the shifted geometric mean.
        # ``CASCADE_BENCH_TIME_RAW=1`` forces the legacy raw arithmetic mean.
        raw_mode = os.environ.get("CASCADE_BENCH_TIME_RAW", "").strip() in ("1", "true", "yes")
        metrics: dict = {}
        if not raw_mode:
            metrics = normalize_time(model_rows, snaive_rows)
        if not metrics:
            # Normalization unavailable (no usable baseline, or forced raw): fall
            # back to the raw mean so the suite still yields crps/mase — but this is
            # NOT Seasonal-Naive-normalized, so it is not comparable to the others.
            per_metric: dict[str, list[float]] = {}
            for row in model_rows:
                for k, v in row.items():
                    if np.isfinite(v):
                        per_metric.setdefault(k.lower(), []).append(v)
            metrics = {k: float(np.mean(vs)) for k, vs in per_metric.items() if vs}
            if not raw_mode:
                print("time: Seasonal-Naive normalization produced nothing; reporting raw "
                      "means (NOT comparable to gift-eval/boom)", file=sys.stderr)
        return SuiteResult(suite="time", status="ok", metrics=metrics, n_series=len(model_rows))
    except ImportError as e:
        return SuiteResult(suite="time", status="skipped", detail=f"timebench not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="time",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
