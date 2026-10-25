"""Compare recorded statistics within a narrow numerical error budget.

Identities, selected targets, tier boundaries and payment durations stay exact.
This does not make different inference backends consensus-compatible.
"""
from __future__ import annotations

import math

STATISTICS = frozenset({"target_lcb", "gain_lcb", "gain"})


def same_verdict(left, right, *, field=""):
    if field in STATISTICS and isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (not isinstance(left, bool) and not isinstance(right, bool)
                and math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_verdict(left[k], right[k], field=k) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_verdict(a, b) for a, b in zip(left, right, strict=True))
    return left == right
