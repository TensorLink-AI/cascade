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


def _score_one(
    wrapper, name: str, term: str, config, out_dir: str, num_samples: int,
    batch_size: int = 64,
) -> dict:
    """Run one TIME task and return ``{metric_name: mean_value}`` from metrics.npz."""
    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.data import Dataset, get_dataset_settings
    from timebench.evaluation.saver import save_window_predictions

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

    ds_config = f"{name}/{term}"
    save_window_predictions(
        dataset=dataset,
        fc_quantiles=fc_quantiles,
        ds_config=ds_config,
        output_base_dir=out_dir,
        seasonality=season,
        model_hyperparams={"model": "cascade"},
        quantile_levels=QUANTILE_LEVELS,
    )

    # TIME writes per-window metric arrays to {out_dir}/{ds_config}/metrics.npz;
    # average each over windows (TIME's own leaderboard aggregation).
    metrics_npz = Path(out_dir) / ds_config / "metrics.npz"
    if not metrics_npz.is_file():
        hits = list(Path(out_dir).rglob("metrics.npz"))
        if not hits:
            raise FileNotFoundError(f"metrics.npz not written for {ds_config}")
        metrics_npz = hits[-1]
    with np.load(metrics_npz) as data:
        return {k: float(np.nanmean(data[k])) for k in data.files}


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

        config = load_dataset_config(None)
        wrapper = _load_wrapper(Path(checkpoint_dir), device)

        per_metric: dict[str, list[float]] = {}
        n_tasks = 0
        with tempfile.TemporaryDirectory(prefix="cascade-time-") as out_dir:
            for name, term in _tasks(config, max_series):
                try:
                    task_metrics = _score_one(
                        wrapper, name, term, config, out_dir, num_samples, batch_size
                    )
                except Exception:  # noqa: BLE001 — one task must not abort the sweep
                    continue
                for k, v in task_metrics.items():
                    if np.isfinite(v):
                        per_metric.setdefault(k, []).append(v)
                n_tasks += 1

        if not n_tasks:
            return SuiteResult(suite="time", status="error", detail="no TIME tasks scored")
        metrics = {k: float(np.mean(vs)) for k, vs in per_metric.items() if vs}
        return SuiteResult(suite="time", status="ok", metrics=metrics, n_series=n_tasks)
    except ImportError as e:
        return SuiteResult(suite="time", status="skipped", detail=f"timebench not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="time",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
