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

import os
import traceback

from ..aggregate import official_aggregate
from ..resources import load_json
from ..results import SuiteResult
from ._common import build_dataset, score_dataset


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


def run(
    checkpoint_dir: str,
    *,
    num_samples: int = 100,
    max_series: int | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> SuiteResult:
    boom_path = os.environ.get("CASCADE_BENCH_BOOM_PATH") or os.environ.get("BOOM")
    if not boom_path:
        return SuiteResult(
            suite="boom",
            status="skipped",
            detail="BOOM (or CASCADE_BENCH_BOOM_PATH) not set; point it at the BOOM data dir.",
        )
    os.environ.setdefault("BOOM", boom_path)
    try:
        baseline = load_json("boom_seasonal_naive.json")
        low_variance = frozenset(load_json("boom_low_variance.json"))
        rows = []
        for full, name, term in _baseline_items(max_series):
            ds = build_dataset(name, term, storage_env_var="BOOM")
            if ds is None:
                continue
            try:
                m = score_dataset(
                    ds, checkpoint_dir,
                    num_samples=num_samples, device=device, batch_size=batch_size,
                )
            except Exception:  # noqa: BLE001 — one config must not abort the sweep
                continue
            rows.append({"full": full, **m})

        if not rows:
            return SuiteResult(suite="boom", status="error", detail="no BOOM configs scored")
        agg = official_aggregate(rows, baseline, low_variance=low_variance)
        metrics = {k: agg[k] for k in ("crps", "mase", "crps_zero", "mae_zero") if k in agg}
        return SuiteResult(suite="boom", status="ok", metrics=metrics, n_series=agg["n_scored"])
    except FileNotFoundError as e:
        return SuiteResult(suite="boom", status="skipped", detail=f"BOOM data file missing: {e}")
    except ImportError as e:
        return SuiteResult(suite="boom", status="skipped", detail=f"gift-eval not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="boom",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
