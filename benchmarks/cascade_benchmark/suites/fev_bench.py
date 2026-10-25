"""FEV runner — autogluon's fev-bench (https://github.com/autogluon/fev).

fev supplies the TASKS (dataset, horizon, rolling evaluation windows,
seasonality); the scoring is ours (``cascade_benchmark.fev_metrics``) so every
per-task number is computed exactly like the other suites: pooled 9-level WQL
("CRPS") and seasonal-scaled MASE, each also scored for an in-suite
Seasonal-Naive baseline, headline = Seasonal-Naive-normalised geometric mean.

Only UNIVARIATE-target tasks are scored: the checkpoint forecasts one channel
at a time here (as in the gluonts suites), and fev's multivariate tasks score
the joint target. Multivariate-target tasks are recorded in ``skipped``
(stage ``"scope"``), never dropped silently. Covariates are ignored (the
wrapper is calendar/covariate-free).

Setup:
* install the optional extra: ``uv sync --project benchmarks --extra fev``
  (the suite reports ``skipped`` when ``fev`` is not importable);
* ``CASCADE_BENCH_FEV_TASKS`` → a fev benchmark YAML (path or URL); default the
  upstream fev-bench task list. fev downloads its datasets from the HF hub
  (``autogluon/fev_datasets``) on first use — set ``HF_HOME`` to cache them.

Rows (``SuiteResult.rows``) are per TASK: ``{"full": <task id>, "CRPS",
"MASE", "snaive_CRPS", "snaive_MASE", "n_windows", "n_series"}`` — the items
the lane-B public guard pairs (DEC-CA-0050).
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from ..fev_metrics import (
    QUANTILE_LEVELS,
    TaskAccumulator,
    normalised_geomean,
    point_as_quantiles,
    seasonal_naive_forecast,
)
from ..provenance import task_id, tree_snapshot, verified_snapshot
from ..results import SuiteResult
from ._common import SuitePlan, abort, describe_exception, run_sequential, until_abort

log = logging.getLogger("cascade_benchmark.fev")

DEFAULT_TASKS = (
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/"
    "benchmarks/fev_bench/tasks.yaml"
)


def target_column(task) -> str | None:
    """The single target column, or ``None`` for a multivariate-target task."""
    cols = getattr(task, "target_columns", None)
    if isinstance(cols, (list, tuple)):
        return str(cols[0]) if len(cols) == 1 else None
    t = getattr(task, "target", None)
    if isinstance(t, str):
        return t
    if isinstance(t, (list, tuple)) and len(t) == 1:
        return str(t[0])
    return None


def _windows(task):
    """Yield ``(past, ground_truth)`` datasets per evaluation window across fev
    API generations (``iter_windows`` from fev 0.6; single-window before)."""
    if hasattr(task, "iter_windows"):
        for w in task.iter_windows():
            past, _future = w.get_input_data()
            yield past, w.get_ground_truth()
        return
    past, _future = task.get_input_data()
    gt = getattr(task, "get_ground_truth", None)
    if gt is None:
        raise RuntimeError("fev task exposes no ground-truth accessor (fev too old?)")
    yield past, gt()


def score_task(task, wrapper, *, batch_size: int = 64) -> dict:
    """Score one univariate task: model + seasonal-naive accumulators over every
    window and series."""
    (res,) = score_task_multi(task, [wrapper], batch_size=batch_size)
    if isinstance(res, BaseException):
        raise res
    return res


def score_task_multi(task, wrappers: list, *, batch_size: int = 64) -> list:
    """:func:`score_task` for several checkpoints over ONE pass of the task's
    windows (each window is loaded once). Per wrapper: its result dict, or the
    exception that stopped it. Each wrapper sees exactly the calls a solo
    ``score_task`` makes (same chunks, same order), and a wrapper that raises
    drops out without touching the others; a failure outside the wrappers
    (window load, a target-count mismatch, the seasonal-naive pass) is
    raised to every wrapper still running."""
    from ..predictor import impute_history

    col = target_column(task)
    horizon = int(task.horizon)
    m = int(getattr(task, "seasonality", 1) or 1)
    models = [TaskAccumulator(QUANTILE_LEVELS) for _ in wrappers]
    naive = TaskAccumulator(QUANTILE_LEVELS)
    failed: list[BaseException | None] = [None] * len(wrappers)
    n_windows = 0

    def live() -> list[int]:
        return [k for k, f in enumerate(failed) if f is None]

    try:
        for past, gt in _windows(task):
            n_windows += 1
            hist = [np.asarray(h, dtype=np.float64) for h in past[col]]
            truth = [np.asarray(t, dtype=np.float64) for t in gt[col]]
            if len(hist) != len(truth):
                raise RuntimeError(f"window has {len(hist)} histories but {len(truth)} targets")
            for i in range(0, len(hist), batch_size):
                chunk = [impute_history(h) for h in hist[i:i + batch_size]]
                qs: dict[int, object] = {}
                for k in live():
                    try:
                        qs[k] = wrappers[k].forecast_quantiles_batch(chunk, horizon)  # (B, h, nq)
                    except Exception as e:  # noqa: BLE001 — this checkpoint only
                        failed[k] = e
                for j, h in enumerate(chunk):
                    y = truth[i + j][:horizon]
                    for k in live():
                        try:
                            models[k].add(h, y, qs[k][j][: y.size], m)
                        except Exception as e:  # noqa: BLE001 — this checkpoint only
                            failed[k] = e
                    if not live():
                        break
                    naive.add(h, y, point_as_quantiles(
                        seasonal_naive_forecast(h, y.size, m), len(QUANTILE_LEVELS)), m)
                if not live():
                    break
            if not live():
                break
    except Exception as e:  # noqa: BLE001 — shared by every checkpoint still running
        for k in live():
            failed[k] = e
    out: list = []
    s_crps = s_mase = None
    if live():
        s_crps, s_mase = naive.result()
    for k in range(len(wrappers)):
        if failed[k] is not None:
            out.append(failed[k])
            continue
        crps, mase = models[k].result()
        out.append({"CRPS": crps, "MASE": mase, "snaive_CRPS": s_crps, "snaive_MASE": s_mase,
                    "n_windows": n_windows, "n_series": models[k].n_series})
    return out


# Per-process memo of the task list a worker re-reads when the parent's tasks
# could not be pickled into the plan (``plan``).
_TASKS_MEMO: dict[tuple[str, int, str, str], list] = {}


def _load_tasks(src: str, max_series: int | None) -> list:
    digest = hashlib.sha256(Path(src).read_bytes()).hexdigest() if Path(src).is_file() else ""
    snapshot = verified_snapshot(src)
    key = (src, int(max_series or 0), digest, snapshot["manifest_sha256"] if snapshot else "")
    if key not in _TASKS_MEMO:
        import fev

        tasks = list(fev.Benchmark.from_yaml(src).tasks)
        _TASKS_MEMO[key] = tasks[:max_series] if max_series else tasks
    return _TASKS_MEMO[key]


def plan(*, max_series: int | None = None):
    """The fev task list (one unit per task) — or the suite's final
    ``SuiteResult``. The wrapper is loaded by each unit (once per process)."""
    try:
        import fev  # noqa: F401
    except ImportError as e:
        return SuiteResult(suite="fev", status="skipped",
                           detail=f"fev not importable (uv sync --extra fev): {e}")
    try:
        src = os.environ.get("CASCADE_BENCH_FEV_TASKS", "").strip() or DEFAULT_TASKS
        snapshot = verified_snapshot(src)
        if snapshot and tree_snapshot(src) != snapshot["tasks"]:
            raise ValueError("FEV task file changed after hash verification")
        tasks = _load_tasks(src, max_series)
        digest = hashlib.sha256(Path(src).read_bytes()).hexdigest() if Path(src).is_file() else ""
        if snapshot and (digest != snapshot["task_sha256"] or tree_snapshot(src) != snapshot["tasks"]):
            raise ValueError("FEV task file changed during planning")
        ctx: dict = {"src": src, "source_sha256": digest, "snapshot": snapshot,
                     "max_series": max_series, "tasks": None}
        try:
            pickle.dumps(tasks)
            ctx["tasks"] = tasks
        except Exception:  # noqa: BLE001 — workers re-read the YAML instead
            pass
        units = [(i, task_id(t)) for i, t in enumerate(tasks)]
    except Exception as e:  # noqa: BLE001
        return SuiteResult(suite="fev", status="error",
                           detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    return SuitePlan(suite="fev", units=units, ctx=ctx)


def run_unit(ctx: dict, unit, checkpoints: list[str], *, num_samples: int = 100,
             device: str = "cpu", batch_size: int = 64) -> list[dict]:
    """Score one fev task with every checkpoint over one pass of its windows.
    A wrapper that does not load, or is off the metric grid, aborts its suite
    (the old run() checked both before the task loop)."""
    from ..predictor import load_wrapper_cached

    i, tid = unit
    out: list[dict | None] = []
    wrappers: list = []
    for ckpt in checkpoints:
        try:
            w = load_wrapper_cached(ckpt, device)
        except Exception as e:  # noqa: BLE001
            out.append(abort("error", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
            wrappers.append(None)
            continue
        levels = [round(float(v), 6) for v in getattr(w, "quantile_levels", [])]
        if levels != [round(v, 6) for v in QUANTILE_LEVELS]:
            out.append(abort("skipped",
                             f"wrapper quantile grid {levels} != {list(QUANTILE_LEVELS)}"))
            wrappers.append(None)
            continue
        out.append(None)
        wrappers.append(w)
    live = [k for k, o in enumerate(out) if o is None]
    if not live:
        return out
    try:
        if ctx.get("source_sha256"):
            actual = hashlib.sha256(Path(ctx["src"]).read_bytes()).hexdigest()
            if actual != ctx["source_sha256"]:
                raise ValueError("FEV task file changed after planning")
        tasks = ctx["tasks"] if ctx.get("tasks") is not None else _load_tasks(
            ctx["src"], ctx["max_series"])
        task = tasks[i]
        if task_id(task) != tid:
            raise ValueError("FEV task ordering changed after planning")
        snapshot = ctx.get("snapshot")
        if snapshot:
            if tree_snapshot(ctx["src"]) != snapshot["tasks"]:
                raise ValueError("FEV task file changed after hash verification")
            root = str(Path(task.dataset_path).resolve())
            if tree_snapshot(root) != snapshot["datasets"].get(root):
                raise ValueError("FEV dataset changed after hash verification")
    except Exception as e:  # noqa: BLE001
        for k in live:
            out[k] = abort("error", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        return out
    if target_column(task) is None:
        for k in live:
            out[k] = {"skip": {"full": tid, "stage": "scope",
                               "reason": "multivariate target (scored univariate only)"}}
        return out
    try:
        results = score_task_multi(task, [wrappers[k] for k in live], batch_size=batch_size)
    except Exception as e:  # noqa: BLE001 — defensive: score_task_multi collects per wrapper
        results = [e] * len(live)
    try:
        if snapshot and (tree_snapshot(root) != snapshot["datasets"].get(root)
                         or tree_snapshot(ctx["src"]) != snapshot["tasks"]):
            raise ValueError("FEV dataset/task file changed during scoring")
    except Exception as e:
        for k in live:
            out[k] = abort("error", f"FEV dataset verification failed: {describe_exception(e)}")
        return out
    for k, res in zip(live, results, strict=True):
        if isinstance(res, BaseException):
            reason = describe_exception(res)
            log.warning("fev: %s not scored: %s", tid, reason)
            out[k] = {"skip": {"full": tid, "stage": "score", "reason": reason}}
        else:
            out[k] = {"row": {"full": tid, **res}}
    return out


def assemble(plan: SuitePlan, outcomes: list[dict]) -> SuiteResult:
    done, aborted = until_abort(outcomes)
    if aborted is not None:
        return SuiteResult(suite="fev", **aborted)
    try:
        rows = [o["row"] for o in done if "row" in o]
        skipped = [o["skip"] for o in done if "skip" in o]
        n_expected = len(plan.units)
        if not rows:
            return SuiteResult(suite="fev", status="error", detail="no fev tasks scored",
                               skipped=skipped, n_expected=n_expected)
        metrics = {
            "crps": normalised_geomean((r["CRPS"] for r in rows), (r["snaive_CRPS"] for r in rows)),
            "mase": normalised_geomean((r["MASE"] for r in rows), (r["snaive_MASE"] for r in rows)),
        }
        detail = f"partial: {len(skipped)} of {n_expected} tasks skipped" if skipped else ""
        return SuiteResult(suite="fev", status="ok", metrics=metrics, n_series=len(rows),
                           rows=rows, detail=detail, skipped=skipped, n_expected=n_expected)
    except Exception as e:  # noqa: BLE001
        return SuiteResult(suite="fev", status="error",
                           detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")


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
