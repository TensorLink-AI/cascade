"""BOOM runner — DataDog's observability benchmark (2,807 real prod series).

BOOM is distributed as the ``Datadog/BOOM`` HF dataset and evaluated through the
*same* gift-eval machinery as GIFT-Eval (DataDog's BOOM loader is built on it),
so once the configs are enumerated the scoring loop is shared with
``_common.evaluate_datasets``.

The BOOM config list comes from the installed gift-eval/BOOM integration; it can
be overridden with ``CASCADE_BENCH_BOOM_DATASETS`` (comma-separated
``name`` or ``name/term``). The HF dataset path is taken from
``BOOM`` / ``CASCADE_BENCH_BOOM_PATH`` env (DataDog's runner uses the ``BOOM``
env var), defaulting to the public ``Datadog/BOOM`` repo.
"""

from __future__ import annotations

import os
import traceback

from ..results import SuiteResult
from ._common import evaluate_datasets


def _iter_datasets(max_series: int | None):
    from gift_eval.data import Dataset

    # DataDog's BOOM runner keys the dataset storage off the ``BOOM`` env var;
    # mirror that, allowing our own alias too.
    boom_path = os.environ.get("CASCADE_BENCH_BOOM_PATH") or os.environ.get("BOOM")
    if boom_path:
        os.environ.setdefault("BOOM", boom_path)

    override = os.environ.get("CASCADE_BENCH_BOOM_DATASETS", "").strip()
    if override:
        specs = [s.strip() for s in override.split(",") if s.strip()]
    else:
        # The BOOM integration exposes its config list; import lazily.
        from gift_eval.data import BOOM_DATASETS  # type: ignore

        specs = list(BOOM_DATASETS)

    if max_series:
        specs = specs[:max_series]  # smoke-run cap on number of configs
    for spec in specs:
        name, _, term = spec.partition("/")
        kwargs = {"name": name, "to_univariate": False, "storage_env_var": "BOOM"}
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
        return SuiteResult(suite="boom", status="ok", metrics=metrics, n_series=n)
    except ImportError as e:
        return SuiteResult(suite="boom", status="skipped", detail=f"BOOM/gift-eval not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="boom",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
