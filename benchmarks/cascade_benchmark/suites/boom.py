"""BOOM runner — DataDog's observability benchmark (2,807 series, real prod).

BOOM is scored through gift-eval's ``Dataset`` (DataDog adapted gift-eval) and
aggregated with the *same* official method as GIFT-Eval — the Seasonal-Naive
normalized shifted geometric mean (``cascade_benchmark.aggregate``), including
BOOM's zero-inflated split and ``LOW_VARIANCE_DATASETS`` exclusion, all ported
from DataDog's ``boom/utils/leaderboard.py``.

We drive the run off the vendored official Seasonal-Naive results keys
(``<config>/<freq>/<term>``): each key gives the ``(name, term)`` to evaluate and
the baseline to normalize against, so alignment is exact by construction. The
baseline (``boom_seasonal_naive.json``) and the ``LOW_VARIANCE_DATASETS`` set
(``boom_low_variance.json``) are vendored under ``data/`` (from DataDog/toto,
Apache-2.0).

Setup:
* ``BOOM`` (or ``CASCADE_BENCH_BOOM_PATH``) → the downloaded BOOM data dir. Required.
* ``CASCADE_BENCH_BOOM_DATASETS`` → comma-separated config names to restrict to.

BOOM is large (350M obs); use ``--max-series`` for anything but a full run.
"""

from __future__ import annotations

import logging
import os
import sys

from ..aggregate import official_aggregate
from ..resources import load_json
from ..results import SuiteResult
from ._common import (
    SuitePlan,
    abort,
    abort_from_exc,
    build_dataset_or_reason,
    describe_exception,
    run_sequential,
    score_dataset,
    until_abort,
)

log = logging.getLogger("cascade_benchmark.boom")


def _baseline_items(max_tasks: int | None):
    """Yield ``(full_key, name, term)`` from the official baseline, optionally
    restricted by ``CASCADE_BENCH_BOOM_DATASETS`` and capped for a smoke run."""
    baseline = load_json("boom_seasonal_naive.json")
    restrict = {
        c.strip()
        for c in os.environ.get("CASCADE_BENCH_BOOM_DATASETS", "").split(",")
        if c.strip()
    }
    n = 0
    for full in baseline:
        parts = full.split("/")
        name, term = parts[0], parts[-1]
        if restrict and name not in restrict:
            continue
        yield full, name, term
        n += 1
        if max_tasks and n >= max_tasks:
            return


def plan(*, max_series: int | None = None):
    """The BOOM config list (one unit per official baseline key) — or the
    suite's final ``SuiteResult`` when it cannot run at all."""
    boom_path = os.environ.get("CASCADE_BENCH_BOOM_PATH") or os.environ.get("BOOM")
    if not boom_path:
        return SuiteResult(
            suite="boom",
            status="skipped",
            detail="BOOM (or CASCADE_BENCH_BOOM_PATH) not set; point it at the BOOM data dir.",
        )
    # set before any worker starts: the scheduler snapshots os.environ after planning
    os.environ.setdefault("BOOM", boom_path)
    try:
        baseline = load_json("boom_seasonal_naive.json")
        low_variance = frozenset(load_json("boom_low_variance.json"))
        units = list(_baseline_items(max_series))
    except Exception as e:  # noqa: BLE001
        return SuiteResult(suite="boom", **_abort_of(e)["abort"])
    return SuitePlan(suite="boom", units=units,
                     local={"baseline": baseline, "low_variance": low_variance})


def _abort_of(e: BaseException) -> dict:
    """The old suite-level ``except`` ladder (call inside the handler)."""
    if isinstance(e, FileNotFoundError):
        return abort("skipped", f"BOOM data file missing: {e}")
    return abort_from_exc(e, import_detail="gift-eval not importable")


def run_unit(ctx: dict, unit, checkpoints: list[str], *, num_samples: int = 100,
             device: str = "cpu", batch_size: int = 64) -> list[dict]:
    """Build one BOOM config's dataset ONCE and score it with every checkpoint."""
    full, name, term = unit
    try:
        ds, reason = build_dataset_or_reason(name, term, storage_env_var="BOOM")
    except Exception as e:  # noqa: BLE001 — gift-eval missing ends the suite
        return [_abort_of(e)] * len(checkpoints)
    if ds is None:
        log.warning("boom: %s not loaded: %s", full, reason)
        return [{"skip": {"full": full, "stage": "load", "reason": reason}}] * len(checkpoints)
    out: list[dict] = []
    for ckpt in checkpoints:
        try:
            m = score_dataset(
                ds, ckpt, num_samples=num_samples, device=device, batch_size=batch_size,
            )
        except Exception as e:  # noqa: BLE001 — one config must not abort the sweep
            reason = describe_exception(e)
            log.warning("boom: %s not scored: %s", full, reason)
            out.append({"skip": {"full": full, "stage": "score", "reason": reason}})
            continue
        out.append({"row": {"full": full, **m}})
    return out


def assemble(plan: SuitePlan, outcomes: list[dict]) -> SuiteResult:
    done, aborted = until_abort(outcomes)
    if aborted is not None:
        return SuiteResult(suite="boom", **aborted)
    try:
        rows = [o["row"] for o in done if "row" in o]
        skipped = [o["skip"] for o in done if "skip" in o]
        n_expected = len(plan.units)
        if not rows:
            return SuiteResult(suite="boom", status="error", detail="no BOOM configs scored",
                               skipped=skipped, n_expected=n_expected)
        agg = official_aggregate(rows, plan.local["baseline"],
                                 low_variance=plan.local["low_variance"])
        metrics = {k: agg[k] for k in ("crps", "mase", "crps_zero", "mae_zero") if k in agg}
        detail = ""
        if skipped:
            detail = f"partial: {len(skipped)} of {n_expected} configs skipped"
            log.warning("boom: %s", detail)
        return SuiteResult(suite="boom", status="ok", metrics=metrics, n_series=agg["n_scored"], rows=rows,
                           detail=detail, skipped=skipped, n_expected=n_expected)
    except Exception as e:  # noqa: BLE001
        return SuiteResult(suite="boom", **_abort_of(e)["abort"])


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
