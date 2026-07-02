# ============================================================================
# TIME  ·  the contamination-resistant "It's TIME" benchmark
#   paper: "It's TIME: Towards the Next Generation of Time Series Forecasting
#           Benchmarks" (arXiv:2602.12147); also a Toto 2.0 headline benchmark.
#   data:  https://huggingface.co/datasets/Real-TSF/TIME  (50 datasets, 98 tasks)
#   code:  https://github.com/zqiao11/TIME  (the `timebench` package)
# ============================================================================
"""TIME benchmark harness for MiniTSFM2.

Unlike GIFT-Eval, TIME is not gluonts-driven: the model hands quantile forecasts
directly as an array ``(N_windows, n_quantiles, n_variates, horizon)`` and
``timebench`` computes the leaderboard metrics itself (writing ``metrics.npz``
with its own metric code). We mirror TIME's reference flow (its
``experiments/chronos2.py``) exactly, swapping their sampled-path pipeline for
MiniTSFM2's native quantile head: each variate is forecast independently via the
shared ``infer.batched_quantiles`` core, reshaped to TIME's quantile grid, saved,
and read back.

Point ``TSFM_TIME_DIR`` (or ``TIME_DATASET``) at the Real-TSF/TIME data. The
``timebench`` package is an optional extra — install ``.[time]``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .infer import batched_quantiles

# TIME's default quantile grid — identical to MiniTSFM2's head.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _instance_quantiles(model, target, horizon):
    """One TIME eval instance's context -> ``(n_quantiles, n_variates, horizon)``.

    ``target`` is ``(variates, time)`` or ``(time,)``; each variate is forecast
    independently (matching the ablation's independent-variate training)."""
    t = np.atleast_2d(np.asarray(target, dtype=np.float64))       # (V, L)
    q = batched_quantiles(model, [t[v] for v in range(t.shape[0])], horizon)  # (V, H, Q)
    return np.transpose(q, (2, 0, 1))                              # (Q, V, H)


def _resolve_time_dir(time_dir=None):
    """Set TIME_DATASET so timebench's loader finds the data; return the path."""
    ds_path = time_dir or os.environ.get("TSFM_TIME_DIR") or os.environ.get("TIME_DATASET")
    if ds_path:
        os.environ.setdefault("TIME_DATASET", ds_path)
    return ds_path


def _tasks(config, datasets, max_tasks):
    """Yield ``(name, term)`` pairs — all bundled tasks, or a restricted set."""
    from timebench.evaluation.utils import get_available_terms

    if datasets:
        names_terms = []
        for spec in datasets:
            name, _, term = spec.partition("/")
            names_terms.append((name, [term] if term else None))
    else:
        names_terms = [(name, None) for name in config.get("datasets", {})]
    n = 0
    for name, terms in names_terms:
        for term in terms or get_available_terms(name, config):
            yield name, term
            n += 1
            if max_tasks and n >= max_tasks:
                return


def _score_one(model, name, term, config, out_dir):
    """Run one TIME task; return ``{metric: mean_over_windows}`` from metrics.npz."""
    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.data import Dataset, get_dataset_settings
    from timebench.evaluation.saver import save_window_predictions

    settings = get_dataset_settings(name, term, config)
    pred_len = settings.get("prediction_length")
    dataset = Dataset(
        name=name, term=term, to_univariate=False,
        prediction_length=pred_len,
        test_length=settings.get("test_length"),
        val_length=settings.get("val_length"),
    )
    season = get_seasonality(dataset.freq)

    fc = [
        _instance_quantiles(model, d["target"], pred_len)[np.newaxis, ...]
        for d in dataset.test_data.input
    ]
    fc_quantiles = np.concatenate(fc, axis=0)                     # (N, Q, V, H)

    ds_config = f"{name}/{term}"
    save_window_predictions(
        dataset=dataset, fc_quantiles=fc_quantiles, ds_config=ds_config,
        output_base_dir=out_dir, seasonality=season,
        model_hyperparams={"model": "tsfm-ablation"},
        quantile_levels=QUANTILE_LEVELS,
    )
    metrics_npz = Path(out_dir) / ds_config / "metrics.npz"
    if not metrics_npz.is_file():
        hits = list(Path(out_dir).rglob("metrics.npz"))
        if not hits:
            raise FileNotFoundError(f"metrics.npz not written for {ds_config}")
        metrics_npz = hits[-1]
    with np.load(metrics_npz) as data:
        return {k: float(np.nanmean(data[k])) for k in data.files}


def eval_time(model, time_dir=None, datasets=None, max_tasks=None, verbose=True):
    """Score ``model`` on TIME. Returns ``{per_task, mean, n_tasks}``.

    ``datasets``: optional list of ``name`` or ``name/term`` to restrict the run
    (default: every bundled task). ``max_tasks``: cap for a quick smoke.
    """
    if _resolve_time_dir(time_dir) is None and not datasets:
        raise RuntimeError(
            "TIME data not configured — set TSFM_TIME_DIR (or TIME_DATASET) to the "
            "Real-TSF/TIME dataset path, or pass datasets=[...]."
        )
    from timebench.evaluation.data import load_dataset_config

    config = load_dataset_config(None)
    model.eval()
    per_task, per_metric = {}, {}
    n_tasks = 0
    with tempfile.TemporaryDirectory(prefix="tsfm-time-") as out_dir:
        for name, term in _tasks(config, datasets, max_tasks):
            try:
                m = _score_one(model, name, term, config, out_dir)
            except Exception as ex:  # one task must not abort the sweep
                if verbose:
                    print(f"  TIME {name}/{term:8s} SKIPPED ({type(ex).__name__}: {ex})")
                continue
            per_task[f"{name}/{term}"] = m
            for k, v in m.items():
                if np.isfinite(v):
                    per_metric.setdefault(k, []).append(v)
            n_tasks += 1
            if verbose:
                shown = " ".join(f"{k}={v:.4f}" for k, v in m.items())
                print(f"  TIME {name}/{term:8s} {shown}")
    mean = {k: float(np.mean(vs)) for k, vs in per_metric.items() if vs}
    out = {"per_task": per_task, "mean": mean, "n_tasks": n_tasks}
    if verbose:
        summary = " ".join(f"{k}={v:.4f}" for k, v in mean.items())
        print(f"TIME mean over {n_tasks} tasks: {summary}")
    return out
