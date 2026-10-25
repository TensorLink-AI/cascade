"""Fixed paired-case, source-cluster public guard with deterministic receipts.

Version 2 stratifies source clusters by suite/metric membership, coupling shared
sources without losing suite coverage. Version 1 remains available for historical
receipt replay. Metric ratios are dimensionless before suite aggregation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..shared.direct_moe import SUITES, canonical, digest, finite
from .bootstrap import _seed_to_int


@dataclass(frozen=True)
class Case:
    suite: str
    case: str
    metric: str
    cluster: str
    weight: float = 1.0

    @property
    def key(self):
        return f"{self.suite}/{self.case}/{self.metric}"


@dataclass(frozen=True)
class GuardContract:
    cases: tuple[Case, ...]
    suite_weights: dict[str, float]
    metric_weights: dict[str, dict[str, float]]
    tau_overall: float
    tau_suite: dict[str, float]
    min_clusters: int
    alpha: float = .05
    draws: int = 10_000
    seed: str = ""
    tau_metric: dict[str, float] | None = None
    version: int = 2

    def __post_init__(self):
        if self.version not in (1, 2) or not 0 < self.alpha < .5 or self.draws < 2 or self.min_clusters < 2:
            raise ValueError("invalid bootstrap contract")
        if not self.seed or not self.cases or len({c.key for c in self.cases}) != len(self.cases):
            raise ValueError("pin seed and unique cases")
        if set(self.suite_weights) != set(SUITES) or set(self.tau_suite) != set(SUITES):
            raise ValueError("all four suites required")
        if set(self.metric_weights) != set(SUITES):
            raise ValueError("pin all metric mappings")
        finite(self.tau_overall)
        for suite in SUITES:
            finite(self.tau_suite[suite])
            finite(self.suite_weights[suite])
            weights = self.metric_weights[suite]
            if not weights or not np.isclose(sum(weights.values()), 1, rtol=0, atol=1e-12):
                raise ValueError("metric weights must sum to one")
            for w in weights.values():
                finite(w)
            if {c.metric for c in self.cases if c.suite == suite} != set(weights):
                raise ValueError("case/metric mapping mismatch")
        if not np.isclose(sum(self.suite_weights.values()), 1, rtol=0, atol=1e-12):
            raise ValueError("suite weights must sum to one")
        for c in self.cases:
            if c.suite not in SUITES or not c.case or not c.cluster or finite(c.weight) <= 0:
                raise ValueError("invalid paired case")
        for key, tau in (self.tau_metric or {}).items():
            if key not in {f"{c.suite}/{c.metric}" for c in self.cases}:
                raise ValueError("unknown metric guard")
            finite(tau)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        return cls(**{**obj, "cases": tuple(Case(**c) for c in obj["cases"])})


def judge_public(contract: GuardContract, reference: dict, candidate: dict, *,
                 reference_id: str, candidate_id: str, private_pass: bool):
    p = contract
    result = {"version": 1, "contract": digest(p.to_dict()), "reference": reference_id,
              "candidate": candidate_id, "passed": False, "status": "inconclusive",
              "reason": "", "coverage": {}, "suite": {}, "metric": {},
              "tau_overall": p.tau_overall, "alpha": p.alpha, "draws": p.draws,
              "seed": p.seed, "private_pass": private_pass}

    def refuse(reason):
        result["reason"] = reason
        return result

    if type(private_pass) is not bool:
        return refuse("private evaluation incomplete")
    cases = sorted(p.cases, key=lambda c: c.key)
    keys = {c.key for c in cases}
    if not reference_id or not candidate_id or set(reference) != keys or set(candidate) != keys:
        return refuse("missing or mismatched paired cases")
    try:
        k = np.array([reference[c.key] for c in cases], dtype=float)
        v = np.array([candidate[c.key] for c in cases], dtype=float)
    except (TypeError, ValueError):
        return refuse("invalid metric value")
    if not np.isfinite(k).all() or not np.isfinite(v).all() or (k <= 0).any() or (v < 0).any():
        return refuse("nonfinite/undefined metric normalisation")
    clusters = sorted({c.cluster for c in cases})
    code = np.array([clusters.index(c.cluster) for c in cases])
    for s in SUITES:
        n = len({c.cluster for c in cases if c.suite == s})
        result["coverage"][s] = {"clusters": n, "cases": sum(c.suite == s for c in cases)}
        if n < p.min_clusters:
            return refuse("insufficient independent source clusters")
    # Fixed cluster draws are shared by all metrics/suites and repeated seeds.
    rng = np.random.default_rng(_seed_to_int(p.seed))
    counts = np.zeros((p.draws + 1, len(clusters)), dtype=float)
    counts[0] = 1
    if p.version == 1:
        # Frozen historical algorithm: old receipts must replay unchanged.
        for draw in range(1, p.draws + 1):
            counts[draw] = np.bincount(rng.integers(len(clusters), size=len(clusters)), minlength=len(clusters))
    else:
        # Stratify by source membership across suite/metric cells. This retains
        # each cell's source count while coupling a shared source everywhere it
        # occurs. No rejected draws, missing suites, or conditional resampling.
        strata = {}
        for i, cluster in enumerate(clusters):
            membership = tuple(sorted({(c.suite, c.metric) for c in cases if c.cluster == cluster}))
            strata.setdefault(membership, []).append(i)
        for membership in sorted(strata):
            indices = np.array(strata[membership])
            for draw in range(1, p.draws + 1):
                counts[draw, indices] = np.bincount(rng.integers(len(indices), size=len(indices)), minlength=len(indices))
    combined = np.zeros(p.draws + 1)
    all_pass = True
    for s in SUITES:
        suite = np.zeros(p.draws + 1)
        for m, metric_weight in sorted(p.metric_weights[s].items()):
            idx = np.array([i for i, c in enumerate(cases) if c.suite == s and c.metric == m])
            if len({cases[i].cluster for i in idx}) < p.min_clusters:
                return refuse("insufficient metric source coverage")
            weights = counts[:, code[idx]] * np.array([cases[i].weight for i in idx])
            denom = weights @ k[idx]
            if (denom <= 0).any():
                return refuse("bootstrap draw lacks required suite/metric coverage")
            rel = 1 - (weights @ v[idx]) / denom
            key = f"{s}/{m}"
            lcb = float(np.quantile(rel[1:], p.alpha, method="linear"))
            tau = (p.tau_metric or {}).get(key)
            result["metric"][key] = {"point": float(rel[0]), "lcb": lcb, "tau": tau}
            if tau is not None:
                all_pass &= lcb >= -tau
            suite += metric_weight * rel
        lcb = float(np.quantile(suite[1:], p.alpha, method="linear"))
        result["suite"][s] = {"point": float(suite[0]), "lcb": lcb, "tau": p.tau_suite[s]}
        all_pass &= lcb >= -p.tau_suite[s]
        combined += p.suite_weights[s] * suite
    result.update(point=float(combined[0]), lcb=float(np.quantile(combined[1:], p.alpha, method="linear")))
    result["passed"] = bool(private_pass is True and all_pass and result["point"] > 0
                            and result["lcb"] >= -p.tau_overall)
    result["status"] = "accepted" if result["passed"] else "rejected"
    result["reason"] = "guard passed" if result["passed"] else "private/point/non-inferiority gate failed"
    return result


def make_receipt(contract, reference, candidate, **kw):
    return {"protocol": "direct-moe/1", "guard": contract.to_dict(),
            "reference_rows": reference, "candidate_rows": candidate,
            "verdict": judge_public(contract, reference, candidate, **kw)}


def replay_receipt(receipt):
    if receipt.get("protocol") != "direct-moe/1":
        raise ValueError("unsupported direct receipt")
    v = receipt["verdict"]
    actual = judge_public(GuardContract.from_dict(receipt["guard"]),
                          receipt["reference_rows"], receipt["candidate_rows"],
                          reference_id=v["reference"], candidate_id=v["candidate"],
                          private_pass=v["private_pass"])
    if canonical(actual) != canonical(v):
        raise ValueError("direct-MoE verdict replay mismatch")
    return actual
