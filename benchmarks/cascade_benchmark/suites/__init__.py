"""Suite registry — name → runner callable.

Each runner has the signature::

    run(checkpoint_dir: str, *, num_samples: int, max_series: int | None,
        device: str) -> SuiteResult

and must never raise: on failure it returns a ``SuiteResult`` with
``status="error"`` (or ``"skipped"``) so one broken suite cannot abort the rest.
"""

from __future__ import annotations

from collections.abc import Callable

from ..results import SuiteResult
from .boom import run as run_boom
from .gifteval import run as run_gifteval
from .time_bench import run as run_time

SUITES: dict[str, Callable[..., SuiteResult]] = {
    "gift-eval": run_gifteval,
    "boom": run_boom,
    "time": run_time,
}
