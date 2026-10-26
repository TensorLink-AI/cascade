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

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .. import cache
from ..aggregate import _CRPS_ALIASES, _MASE_ALIASES, _ci_get
from ..predictor import impute_history
from ..predictor import load_wrapper_cached as _load_wrapper
from ..results import SuiteResult
from ._common import (
    SuitePlan,
    abort_from_exc,
    describe_exception,
    run_sequential,
    until_abort,
)

log = logging.getLogger("cascade_benchmark.time")

# TIME's default quantile grid (experiments/chronos2.py) — identical to cascade's.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _wrapper_quantile_grid_matches(wrapper) -> bool:
    """The direct head must contain every TIME level exactly."""
    from ..quantiles import quantile_indices

    return (hasattr(wrapper, "forecast_quantiles_batch") and
            quantile_indices(getattr(wrapper, "quantile_levels", None), QUANTILE_LEVELS) is not None)


def _contexts(wrapper, target) -> np.ndarray:
    """``(V, L)`` float64 contexts for the wrapper. Missing values (NaN/±inf in
    TIME's data — 20 of 98 tasks carry them) are imputed exactly as the gluonts
    suites do (``impute_history``: ffill, bfill, all-missing → zeros) unless the
    wrapper advertises ``handles_missing`` and takes the raw gaps itself."""
    t = np.atleast_2d(np.asarray(target, dtype=np.float64))  # (V, L)
    if getattr(wrapper, "handles_missing", False):
        return t
    return np.stack([impute_history(row) for row in t]) if t.size else t


def _instance_quantiles(wrapper, target, horizon: int, num_samples: int) -> np.ndarray:
    """One TIME eval instance's forecast, shaped ``(num_q, num_variates, H)``.

    ``target`` is the instance's context: ``(variates, time)`` or ``(time,)``.
    Each variate is forecast independently (cascade's per-channel convention).
    CPM checkpoints hand back the quantile head directly on TIME's grid; older
    checkpoints draw sample paths and collapse them to the grid.
    """
    t = _contexts(wrapper, target)  # (V, L)
    if _wrapper_quantile_grid_matches(wrapper):
        from ..quantiles import forecast_on_grid

        q = forecast_on_grid(wrapper, list(t), horizon, QUANTILE_LEVELS)
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


def _prepare_task(name: str, term: str, config) -> dict:
    """Build one TIME task's checkpoint-independent inputs ONCE: the timebench
    ``Dataset``, its seasonality, the eval instances and the per-task row info
    (``num_variates`` / ``prediction_length``)."""
    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.data import Dataset, get_dataset_settings

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
    eval_inputs = list(dataset.test_data.input)
    info = {"num_variates": int(np.atleast_2d(np.asarray(eval_inputs[0]["target"])).shape[0])
            if eval_inputs else 0, "prediction_length": int(pred_len or 0)}
    return {"name": name, "term": term, "dataset": dataset, "pred_len": pred_len,
            "season": get_seasonality(dataset.freq), "eval_inputs": eval_inputs, "info": info}


def _score_prepared(
    wrapper, prep: dict, out_dir: str, num_samples: int, batch_size: int = 64, *,
    normalize: bool = True, snaive_memo: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Score one prepared TIME task with ``wrapper`` → ``(model_metrics,
    seasonal_naive_metrics, info)``, each ``{metric_name: mean_value}`` computed
    by TIME's own metric code. The Seasonal-Naive baseline is scored through the
    identical saver+metric path, so the caller can normalize the model metric by
    the baseline metric per task (the ratio-then-geomean the GIFT-Eval / BOOM
    leaderboards — and TIME's own — use).

    The baseline is checkpoint-independent: it is cached on disk (computed at
    most once per task across rounds/checkpoints) and, within one paired unit,
    shared through ``snaive_memo``. ``normalize=False`` skips it entirely (the
    raw fallback path doesn't need it)."""
    from ..aggregate import seasonal_naive_quantiles

    name, term, dataset = prep["name"], prep["term"], prep["dataset"]
    pred_len, season, eval_inputs = prep["pred_len"], prep["season"], prep["eval_inputs"]
    if _wrapper_quantile_grid_matches(wrapper):
        # Quantile head, batched across instances *and* variates: flatten every
        # (instance, variate) context into one job list, chunk it through
        # forecast_quantiles_batch, and reassemble TIME's (N, num_q, V, H).
        jobs: list[np.ndarray] = []
        counts: list[int] = []
        for d in eval_inputs:
            t = _contexts(wrapper, d["target"])
            jobs.extend(t)
            counts.append(t.shape[0])
        from ..quantiles import forecast_on_grid

        chunks = [
            forecast_on_grid(wrapper, jobs[i : i + batch_size], pred_len, QUANTILE_LEVELS)
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
    model_metrics = _metrics_from_quantiles(
        dataset, fc_quantiles, ds_config, Path(out_dir) / "model", season, "cascade",
    )
    if not normalize:
        return model_metrics, {}, prep["info"]

    if snaive_memo is not None and "snaive" in snaive_memo:
        return model_metrics, dict(snaive_memo["snaive"]), prep["info"]
    # Seasonal-Naive baseline: checkpoint-independent, so serve it from cache and
    # only compute (once) on a miss. Scored on the SAME instances/grid through the
    # SAME saver+metric path, so model÷baseline per task is apples-to-apples.
    n_q = len(QUANTILE_LEVELS)
    cache_dir = cache.scoped_baseline_cache_dir({
        "task": name, "term": term, "season": season,
        "prediction_length": pred_len, "quantiles": list(QUANTILE_LEVELS),
    })
    snaive_metrics = cache.load_baseline(cache_dir, name, term, pred_len, n_q)
    if snaive_metrics is None:
        snaive = np.concatenate(
            [
                seasonal_naive_quantiles(_contexts(None, d["target"]), pred_len, season, n_q)[np.newaxis, ...]
                for d in eval_inputs
            ],
            axis=0,
        )
        snaive_metrics = _metrics_from_quantiles(
            dataset, snaive, ds_config, Path(out_dir) / "snaive", season, "seasonal_naive",
        )
        cache.store_baseline(cache_dir, name, term, pred_len, n_q, snaive_metrics)
    if snaive_memo is not None:
        snaive_memo["snaive"] = dict(snaive_metrics)
    return model_metrics, snaive_metrics, prep["info"]


def _score_one(
    wrapper, name: str, term: str, config, out_dir: str, num_samples: int,
    batch_size: int = 64, *, normalize: bool = True,
) -> tuple[dict, dict, dict]:
    """Run one TIME task end to end with one wrapper (``_prepare_task`` +
    ``_score_prepared``)."""
    return _score_prepared(wrapper, _prepare_task(name, term, config), out_dir, num_samples,
                           batch_size, normalize=normalize)


def plan(*, max_series: int | None = None):
    """The TIME task list (one unit per ``(name, term)``) — or the suite's
    final ``SuiteResult`` when it cannot run at all. Checkpoint-free: each
    worker loads the wrapper itself (once, ``load_wrapper_cached``)."""
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

        from ..aggregate import normalize_time  # noqa: F401 — import check (assemble uses it)

        config = load_dataset_config(None)
        # ``CASCADE_BENCH_TIME_RAW=1`` forces the legacy raw arithmetic mean (no
        # Seasonal-Naive normalization) — then the per-task baseline is not even
        # computed.
        raw_mode = os.environ.get("CASCADE_BENCH_TIME_RAW", "").strip() in ("1", "true", "yes")
        tasks = list(_tasks(config, max_series))
    except Exception as e:  # noqa: BLE001
        return SuiteResult(suite="time", **abort_from_exc(
            e, import_detail="timebench not importable")["abort"])
    return SuitePlan(suite="time", units=[(j, name, term) for j, (name, term) in enumerate(tasks)],
                     ctx={"config": config, "raw_mode": raw_mode})


def run_unit(ctx: dict, unit, checkpoints: list[str], *, num_samples: int = 100,
             device: str = "cpu", batch_size: int = 64) -> list[dict]:
    """Prepare one TIME task ONCE and score it with every checkpoint. A
    checkpoint whose wrapper does not load aborts its suite (the old run()
    loaded it before the task loop)."""
    _j, name, term = unit
    full = f"{name}/{term}"
    wrappers: list = []
    out: list[dict | None] = []
    for ckpt in checkpoints:
        try:
            wrappers.append(_load_wrapper(ckpt, device))
            out.append(None)
        except Exception as e:  # noqa: BLE001 — the checkpoint, not the task
            wrappers.append(None)
            out.append(abort_from_exc(e, import_detail="timebench not importable"))
    live = [i for i, o in enumerate(out) if o is None]
    if not live:
        return out
    try:
        prep = _prepare_task(name, term, ctx["config"])
    except Exception as e:  # noqa: BLE001 — one task must not abort the sweep
        reason = describe_exception(e)
        log.warning("time: %s not scored: %s", full, reason)
        for i in live:
            out[i] = {"skip": {"full": full, "stage": "score", "reason": reason}}
        return out
    memo: dict = {}
    with tempfile.TemporaryDirectory(prefix="cascade-time-") as out_dir:
        for i in live:
            try:
                # per-checkpoint subdir keeps every role's metrics.npz isolated
                model_m, snaive_m, info = _score_prepared(
                    wrappers[i], prep, str(Path(out_dir) / str(i)), num_samples, batch_size,
                    normalize=not ctx["raw_mode"], snaive_memo=memo,
                )
            except Exception as e:  # noqa: BLE001 — one task must not abort the sweep
                reason = describe_exception(e)
                log.warning("time: %s not scored: %s", full, reason)
                out[i] = {"skip": {"full": full, "stage": "score", "reason": reason}}
                continue
            out[i] = {"model": model_m, "snaive": snaive_m, "info": info}
    return out


def assemble(plan: SuitePlan, outcomes: list[dict]) -> SuiteResult:
    done, aborted = until_abort(outcomes)
    if aborted is not None:
        return SuiteResult(suite="time", **aborted)
    try:
        from ..aggregate import normalize_time

        raw_mode = plan.ctx["raw_mode"]
        n_tasks = len(plan.units)
        model_rows: list[dict] = []
        snaive_rows: list[dict] = []
        skipped: list[dict] = []  # a task that fails is RECORDED, never dropped silently
        rows: list[dict] = []     # per-task model + Seasonal-Naive metrics (uni/multi splits)
        for (_j, name, term), o in zip(plan.units, done, strict=False):
            if "skip" in o:
                skipped.append(o["skip"])
                continue
            model_m, snaive_m, info = o["model"], o["snaive"], o["info"]
            model_rows.append(model_m)
            snaive_rows.append(snaive_m)
            if not all(np.isfinite(float(_ci_get(model_m, a) or np.nan))
                       for a in (_CRPS_ALIASES, _MASE_ALIASES)):
                # The leaderboard aggregate mean-replaces an invalid task
                # (``_clean``), so the headline still includes it — but the
                # report must say so instead of hiding it in a mean.
                reason = "non-finite CRPS/MASE (mean-replaced in the aggregate)"
                log.warning("time: %s/%s %s", name, term, reason)
                skipped.append({"full": f"{name}/{term}", "stage": "metric", "reason": reason})
            rows.append({"full": f"{name}/{term}", **info,
                         "CRPS": _ci_get(model_m, _CRPS_ALIASES), "MASE": _ci_get(model_m, _MASE_ALIASES),
                         "snaive_CRPS": _ci_get(snaive_m, _CRPS_ALIASES),
                         "snaive_MASE": _ci_get(snaive_m, _MASE_ALIASES)})

        if not model_rows:
            return SuiteResult(suite="time", status="error", detail="no TIME tasks scored",
                               skipped=skipped, n_expected=n_tasks)

        # Parity with GIFT-Eval/BOOM (and TIME's own leaderboard): per-task ratio to
        # the Seasonal-Naive baseline, aggregated by the shifted geometric mean.
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
        detail = ""
        dropped = [x for x in skipped if x["stage"] != "metric"]
        invalid = [x for x in skipped if x["stage"] == "metric"]
        if dropped:
            detail = f"partial: {len(dropped)} of {n_tasks} tasks skipped: " + ", ".join(
                x["full"] for x in dropped[:8]) + (" …" if len(dropped) > 8 else "")
        if invalid:
            detail += ("; " if detail else "") + (
                f"{len(invalid)} of {n_tasks} tasks non-finite (mean-replaced): "
                + ", ".join(x["full"] for x in invalid[:8]) + (" …" if len(invalid) > 8 else ""))
        return SuiteResult(suite="time", status="ok", metrics=metrics, n_series=len(model_rows),
                           detail=detail, skipped=skipped, n_expected=n_tasks, rows=rows)
    except Exception as e:  # noqa: BLE001
        return SuiteResult(suite="time", **abort_from_exc(
            e, import_detail="timebench not importable")["abort"])


def run(
    checkpoint_dir: str,
    *,
    num_samples: int = 100,
    max_series: int | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> SuiteResult:
    """The whole suite for one checkpoint, in this process."""
    return run_sequential(sys.modules[__name__], [checkpoint_dir], num_samples=num_samples,
                          max_series=max_series, device=device, batch_size=batch_size)[0]
