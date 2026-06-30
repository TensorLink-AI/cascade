"""Scoring math: CRPS (MWSQL), MASE, paired bootstrap, and the KOTH decision.

All numpy — no torch — so the statistics are unit-testable in a minimal env.
"""

from __future__ import annotations

from .bootstrap import paired_bootstrap_lcb, paired_bootstrap_lcb_aggregated
from .crps import DEFAULT_QUANTILE_LEVELS, mwsql_components, mwsql_from_components
from .koth import KothParams, RoundResult, evaluate_round, margin_for_tenure
from .mase import mase
from .scoring import (
    WindowScore,
    global_geomean,
    score_forecaster_on_windows,
    stack_components,
)
from .window import EvalWindow

__all__ = [
    "paired_bootstrap_lcb",
    "paired_bootstrap_lcb_aggregated",
    "DEFAULT_QUANTILE_LEVELS",
    "mwsql_components",
    "mwsql_from_components",
    "KothParams",
    "RoundResult",
    "evaluate_round",
    "margin_for_tenure",
    "mase",
    "WindowScore",
    "global_geomean",
    "score_forecaster_on_windows",
    "stack_components",
    "EvalWindow",
]
