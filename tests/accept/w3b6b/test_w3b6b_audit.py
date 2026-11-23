"""W3b6b, audit: the W9 audit accepts a gate result's optional ``rows``
(three sha256 digests) and still refuses any other extra key. Contract:
README.md."""

from __future__ import annotations

from . import helpers as h


def _result(**extra) -> dict:
    return {"digest": "a" * 64, "outcome": "pass",
            "lcb": {"king_d": {"crps": 0.02, "mase": 0.01}, "house": {"crps": 0.03, "mase": 0.0}},
            "blocked_by": [], "session": "lane-d-bench/1", **extra}


def test_audit_accepts_bound_rows():
    from cascade.audit.lane_d import _result_problem

    rows = {r: f"{i:x}" * 64 for i, r in enumerate(h.ROLES, start=1)}
    assert _result_problem(0, _result(rows=rows), 0.01) is None, \
        "W3b6b: an optional rows object of three sha256s is a valid gate result"


def test_audit_refuses_malformed_rows():
    from cascade.audit.lane_d import _result_problem

    assert _result_problem(0, _result(rows={"candidate": "a" * 64}), 0.01) is not None
    assert _result_problem(0, _result(rows={r: "nothex" for r in h.ROLES}), 0.01) is not None
    assert _result_problem(0, _result(extra=1), 0.01) is not None
