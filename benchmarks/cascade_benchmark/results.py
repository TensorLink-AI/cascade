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
    def n_scope_skipped(self) -> int:
        """FEV's declared univariate-only scope: multivariate-target tasks are
        planned, reported (``stage == "scope"``) and never scored. They are
        outside the suite's scope, not missing coverage."""
        return sum(1 for k in self.skipped
                   if self.suite == "fev" and isinstance(k, dict) and k.get("stage") == "scope")

    @property
    def partial(self) -> bool:
        expected = self.n_expected - self.n_scope_skipped
        return bool(self.max_series) or bool(self.n_expected) and self.n_series < expected


# Per-row metric floor (operator decision 2026-11-02). The validator's public
# bench gate (cascade.eval.public_guards._usable) treats a row metric that is
# not strictly > 0 as UNUSABLE, so an exact 0 — a perfect forecast of a
# constant series, which a confident model's float32 median hits exactly —
# read as an omitted task and failed the challenger. Both gate metrics are
# scale-free (CRPS = weighted quantile loss ÷ Σ|y|; MASE ÷ in-sample naive
# error), so a fixed floor is unit-consistent: any finite value in
# [0, ROW_METRIC_FLOOR) is published as ROW_METRIC_FLOOR. Two sub-floor values
# (0 vs 1.2e-9) then pair as an exact tie instead of a fake 1000x ratio.
# NaN / inf / negative / missing values are left untouched, so the gate's
# protection against omitted or broken tasks is unchanged. Only the per-row
# fields the gate reads are floored; headline aggregates are computed from the
# unfloored values. Interim: the in-gate fix rides the next consensus release.
ROW_METRIC_FLOOR = 1e-6
FLOORED_ROW_METRICS = ("CRPS", "MASE")


def floor_row_metrics(row: dict) -> dict:
    """``row`` with every finite ``CRPS`` / ``MASE`` in ``[0, ROW_METRIC_FLOOR)``
    raised to the floor. Anything else (NaN, inf, negative, absent, non-numeric)
    is returned as-is."""
    if not isinstance(row, dict):
        return row
    out = dict(row)
    for k in FLOORED_ROW_METRICS:
        v = out.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        f = float(v)
        if f == f and 0.0 <= f < ROW_METRIC_FLOOR:
            out[k] = ROW_METRIC_FLOOR
    return out


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
            "suites": [{**asdict(s), "partial": s.partial,
                        "rows": [floor_row_metrics(r) for r in s.rows]}
                       for s in self.suites],
            "row_metric_floor": ROW_METRIC_FLOOR,
            "data_revisions": dict(self.data_revisions),
            "methodology": {s.suite: METHODOLOGY[s.suite] for s in self.suites
                            if s.suite in METHODOLOGY},
        }
