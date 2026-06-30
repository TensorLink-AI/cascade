"""TIME runner — the contamination-resistant "It's TIME" benchmark.

  paper: "It's TIME: Towards the Next Generation of Time Series Forecasting
          Benchmarks" (arXiv:2602.12147)

TIME is built from fresh, held-out datasets specifically to resist test-set
contamination, which is philosophically aligned with cascade's own rotating
private pool. Its public evaluation harness API is not yet pinned in this
sidecar, so this runner is a deliberate, clearly-marked stub:

* If TIME ultimately exposes a gluonts-interface dataset list (as GIFT-Eval and
  BOOM do), filling this in is a few lines — enumerate its datasets and hand
  them to ``_common.evaluate_datasets`` exactly like the other two suites.
* Until then it returns ``status="skipped"`` with a reason, so the validator
  logs "time: skipped" rather than a fabricated score. We never emit a number we
  can't stand behind against the published leaderboard.

To wire it up, set ``CASCADE_BENCH_TIME_DATASETS`` (comma-separated specs) once
the TIME loader is added to the env, and replace the stub body below.
"""

from __future__ import annotations

import os
import traceback

from ..results import SuiteResult
from ._common import evaluate_datasets


def _iter_datasets(max_series: int | None):
    # Reuse the gluonts-interface path *iff* TIME datasets are addressable
    # through gift-eval's Dataset class and an explicit spec list is provided.
    from gift_eval.data import Dataset

    override = os.environ.get("CASCADE_BENCH_TIME_DATASETS", "").strip()
    specs = [s.strip() for s in override.split(",") if s.strip()]
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
    if not os.environ.get("CASCADE_BENCH_TIME_DATASETS", "").strip():
        return SuiteResult(
            suite="time",
            status="skipped",
            detail=(
                "TIME loader not configured; set CASCADE_BENCH_TIME_DATASETS once the "
                "TIME harness (arXiv:2602.12147) is added to the sidecar env."
            ),
        )
    try:
        metrics, n = evaluate_datasets(
            _iter_datasets(max_series),
            checkpoint_dir,
            num_samples=num_samples,
            device=device,
        )
        return SuiteResult(suite="time", status="ok", metrics=metrics, n_series=n)
    except ImportError as e:
        return SuiteResult(suite="time", status="skipped", detail=f"TIME deps not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="time",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
