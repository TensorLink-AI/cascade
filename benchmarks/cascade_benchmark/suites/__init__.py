"""Suite registry — name → runner callable.

Each runner has the signature::

    run(checkpoint_dir: str, *, num_samples: int, max_series: int | None,
        device: str, batch_size: int) -> SuiteResult

and must never raise: on failure it returns a ``SuiteResult`` with
``status="error"`` (or ``"skipped"``) so one broken suite cannot abort the rest.
"""

from __future__ import annotations

from collections.abc import Callable

from ..results import SuiteResult
from .boom import run as run_boom
from .fev_bench import run as run_fev
from .gifteval import run as run_gifteval
from .time_bench import run as run_time

# Suite name → the module implementing the work-unit contract (``plan`` /
# ``run_unit`` / ``assemble``, ``suites/_common.py``) the parallel scheduler
# imports by NAME in each worker process.
SUITE_MODULES: dict[str, str] = {
    "gift-eval": "cascade_benchmark.suites.gifteval",
    "boom": "cascade_benchmark.suites.boom",
    "time": "cascade_benchmark.suites.time_bench",
    "fev": "cascade_benchmark.suites.fev_bench",
}

SUITES: dict[str, Callable[..., SuiteResult]] = {
    "gift-eval": run_gifteval,
    "boom": run_boom,
    "time": run_time,
    # FEV (DEC-CA-0050 P5): optional extra; reports `skipped` without it.
    "fev": run_fev,
}
