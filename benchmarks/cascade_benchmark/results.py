"""Stable JSON shape the validator reads back.

Keeping this in one place means the cascade-side bridge
(``cascade.eval.benchmarks``) and this sidecar agree on the contract without
sharing code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Reporting metadata only: these describe the EXISTING scoring contracts.
# Parallel execution must not replace them with a faster approximation.
METHODOLOGY = {
    "gift-eval": {"metrics": "reference GIFT-Eval metrics",
                  "aggregate": "official Seasonal-Naive-normalized shifted geometric mean"},
    "boom": {"metrics": "reference GIFT-Eval metrics over BOOM tasks",
             "aggregate": "official BOOM baseline and low-variance exclusions"},
    "time": {"metrics": "upstream timebench metric engine",
             "aggregate": "Cascade Seasonal-Naive-normalized shifted geometric mean"},
    "fev": {"metrics": "cascade-fev/1: pooled nine-quantile WQL and seasonal MASE",
            "aggregate": "Seasonal-Naive-normalized geometric mean",
            "scope": "univariate targets; covariates unused; multivariate targets reported as scope skips",
            "official_fev_leaderboard_equivalent": False},
}


@dataclass
class SuiteResult:
    """Aggregated metrics for one benchmark suite."""

    suite: str
    status: str  # "ok" | "skipped" | "error"
    metrics: dict = field(default_factory=dict)  # e.g. {"crps": .., "mase": ..}
    n_series: int = 0
    detail: str = ""  # error message or skip reason
    # Per-config rows the consensus gate consumes: one dict per scored config,
    # ``{"full": name/freq/term, "MASE", "MAE", "CRPS", "crps_ratio",
    # "mase_ratio"}`` where the ratios are model ÷ vendored Seasonal-Naive
    # baseline. ``boom`` rows carry ``{"full", "MASE", "MAE", "CRPS"}``; ``time``
    # rows ``{"full": name/term, "num_variates", "prediction_length", "CRPS",
    # "MASE", "snaive_CRPS", "snaive_MASE"}`` — enough for uni/multi splits.
    rows: list = field(default_factory=list)
    # Skip accounting, so a partial sweep can never read as
    # leaderboard-comparable. One dict per config that was
    # requested but produced no row: ``{"full": name/freq/term, "stage":
    # "load" | "score", "reason": "<ExcType>: <message>"}``. ``n_expected`` is
    # the number of configs the sweep set out to score; a consumer compares it
    # with ``n_series`` to see whether the headline covers the whole benchmark.
    skipped: list = field(default_factory=list)
    n_expected: int = 0
    max_series: int = 0

    @property
    def partial(self) -> bool:
        return bool(self.max_series) or bool(self.n_expected) and self.n_series < self.n_expected


@dataclass
class BenchmarkReport:
    checkpoint: str
    suites: list[SuiteResult] = field(default_factory=list)
    # suite → pinned HF dataset revision the numbers were computed against
    # (see ``datasets.DATASETS``) — keeps historical reports traceable.
    data_revisions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "suites": [{**asdict(s), "partial": s.partial} for s in self.suites],
            "data_revisions": dict(self.data_revisions),
            "methodology": {s.suite: METHODOLOGY[s.suite] for s in self.suites
                            if s.suite in METHODOLOGY},
        }
