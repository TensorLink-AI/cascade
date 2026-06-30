"""GIFT-Eval runner — Salesforce's 97-config general TS benchmark.

Enumerates the GIFT-Eval dataset configs via the installed ``gift_eval`` package
and scores the checkpoint on each. The dataset list can be overridden with
``CASCADE_BENCH_GIFTEVAL_DATASETS`` (comma-separated ``name`` or ``name/term``)
to run a fast subset; otherwise the full published config list is used.
"""

from __future__ import annotations

import os
import traceback

from ..results import SuiteResult
from ._common import evaluate_datasets


def _iter_datasets(max_series: int | None):
    """Yield gift-eval ``Dataset`` objects for the configured configs."""
    from gift_eval.data import Dataset

    override = os.environ.get("CASCADE_BENCH_GIFTEVAL_DATASETS", "").strip()
    if override:
        specs = [s.strip() for s in override.split(",") if s.strip()]
    else:
        # gift-eval ships the canonical config list; import lazily so a missing
        # symbol surfaces as a clear skip rather than an import-time crash.
        from gift_eval.data import ALL_DATASETS  # type: ignore

        specs = list(ALL_DATASETS)

    if max_series:
        specs = specs[:max_series]  # smoke-run cap on number of configs
    for spec in specs:
        name, _, term = spec.partition("/")
        kwargs = {"name": name, "to_univariate": False}
        if term:
            kwargs["term"] = term
        yield Dataset(**kwargs)


def run(
    checkpoint_dir: str,
    *,
    num_samples: int = 100,
    max_series: int | None = None,
    device: str = "cpu",
) -> SuiteResult:
    try:
        metrics, n = evaluate_datasets(
            _iter_datasets(max_series),
            checkpoint_dir,
            num_samples=num_samples,
            device=device,
        )
        return SuiteResult(suite="gift-eval", status="ok", metrics=metrics, n_series=n)
    except ImportError as e:
        return SuiteResult(
            suite="gift-eval", status="skipped", detail=f"gift-eval not importable: {e}"
        )
    except Exception as e:  # noqa: BLE001 — one suite must not abort the others
        return SuiteResult(
            suite="gift-eval",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
