"""Stable JSON shape the validator reads back.

Keeping this in one place means the cascade-side bridge
(``cascade.eval.benchmarks``) and this sidecar agree on the contract without
sharing code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
    # baseline. Only ``gift-eval`` populates this; other suites leave it empty.
    rows: list = field(default_factory=list)


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
            "suites": [asdict(s) for s in self.suites],
            "data_revisions": dict(self.data_revisions),
        }
