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


@dataclass
class BenchmarkReport:
    checkpoint: str
    suites: list[SuiteResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"checkpoint": self.checkpoint, "suites": [asdict(s) for s in self.suites]}
