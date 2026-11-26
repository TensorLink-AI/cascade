"""W14: the audit re-derives an absent lane D settlement
(`cascade/audit/lane_d.py`, `check_lane_d_absent`).

Plan §1.5: "`checks.derive_weights` and `check_lane_d` re-derive the lane D
weights from the receipt chain"; DEC-CA-0058 step d: a satellite re-derives
everything with the code `cascade-audit` runs. Invariant 12: "Audit and
satellite-replay code never import the producer's eligibility, verdict or
weight functions." README.md "Audit and satellites".
"""

from __future__ import annotations

import ast

import pytest

from .helpers import REPO, Chain, absent, boundary, genesis_sha, lane_d_cfg, unarmed_cfg


def _check():
    from cascade.audit.lane_d import check_lane_d_absent

    return check_lane_d_absent


def _status(res) -> str:
    return str(getattr(res, "status", res))


def _run(tmp_path, n_present: int):
    """W7c chain settled through boundary(n_present), then an absent record
    at the next boundary (built by the producer)."""
    chain = Chain(tmp_path)
    sp = chain.state_path("a")
    runs = chain.run(sp, n_present)
    return chain, runs, absent(chain, n_present + 1, sp).receipt


def test_a_genuine_absent_record_passes(tmp_path):
    _, runs, rec = _run(tmp_path, 1)
    res = _check()(rec, lane_d_cfg(), boundary=boundary(2),
                   previous_posterior_sha256=runs[0].receipt["posterior_state_sha256"])
    assert _status(res) == "PASS", res


def test_a_genuine_absent_after_a_crown_passes(tmp_path):
    _, runs, rec = _run(tmp_path, 2)
    res = _check()(rec, lane_d_cfg(), boundary=boundary(3),
                   previous_posterior_sha256=runs[1].receipt["posterior_state_sha256"])
    assert _status(res) == "PASS", res


def test_a_genuine_absent_from_genesis_passes(tmp_path):
    chain = Chain(tmp_path)
    rec = absent(chain, 1, chain.state_path("g")).receipt
    res = _check()(rec, lane_d_cfg(), boundary=boundary(1),
                   previous_posterior_sha256=genesis_sha(chain))
    assert _status(res) == "PASS", res


def test_an_absent_binding_a_state_before_the_crown_fails(tmp_path):
    """The suppression attempt: an absent record at boundary(3) that binds
    the state before X's crown at boundary(2)."""
    from cascade.validator.lane_d_state import make_absent_receipt

    _, runs, _ = _run(tmp_path, 2)
    forged = make_absent_receipt(boundary=boundary(3), reason="report_absent",
                                 state_sha256=runs[0].receipt["posterior_state_sha256"])
    res = _check()(forged, lane_d_cfg(), boundary=boundary(3),
                   previous_posterior_sha256=runs[1].receipt["posterior_state_sha256"])
    assert _status(res) == "FAIL", res


def test_an_absent_contradicted_by_a_published_report_fails(tmp_path):
    """A signed report exists for the boundary the record calls absent."""
    from cascade.validator.lane_d_state import make_absent_receipt

    _, runs, _ = _run(tmp_path, 2)
    forged = make_absent_receipt(boundary=boundary(2), reason="report_absent",
                                 state_sha256=runs[0].receipt["posterior_state_sha256"])
    prev = runs[0].receipt["posterior_state_sha256"]
    assert _status(_check()(forged, lane_d_cfg(), boundary=boundary(2),
                            previous_posterior_sha256=prev)) == "PASS"
    res = _check()(forged, lane_d_cfg(), boundary=boundary(2),
                   previous_posterior_sha256=prev, report=runs[1].report)
    assert _status(res) == "FAIL", res


@pytest.mark.parametrize("defect", ["moved_state", "wrong_boundary", "unarmed", "present",
                                    "bad_reason", "not_object"])
def test_a_defective_absent_record_fails(tmp_path, defect):
    _, runs, rec = _run(tmp_path, 1)
    prev = runs[0].receipt["posterior_state_sha256"]
    cfg, at = lane_d_cfg(), boundary(2)
    if defect == "moved_state":
        rec = {**rec, "posterior_state_sha256": runs[0].receipt["prior_state_sha256"]}
    elif defect == "wrong_boundary":
        at = boundary(3)
    elif defect == "unarmed":
        cfg = unarmed_cfg()
    elif defect == "present":
        rec = runs[0].receipt
    elif defect == "bad_reason":
        rec = {**rec, "reason": "operator_said_so"}
    else:
        rec = [rec]
    res = _check()(rec, cfg, boundary=at, previous_posterior_sha256=prev)
    assert _status(res) == "FAIL", (defect, res)


def test_the_audit_is_deterministic(tmp_path):
    _, runs, rec = _run(tmp_path, 1)
    prev = runs[0].receipt["posterior_state_sha256"]
    a = _check()(rec, lane_d_cfg(), boundary=boundary(2), previous_posterior_sha256=prev)
    b = _check()(dict(rec), lane_d_cfg(), boundary=boundary(2), previous_posterior_sha256=prev)
    assert a == b


def test_inv12_the_audit_never_imports_producer_code():
    """Static: `cascade/audit/lane_d.py` imports no `cascade.validator`
    module (the absent record's rules are re-implemented, not borrowed)."""
    path = REPO / "cascade" / "audit" / "lane_d.py"
    tree = ast.parse(path.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            if "validator" in mod.split(".") or mod.startswith("cascade.validator"):
                bad.append(mod)
        elif isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.startswith("cascade.validator")]
    assert bad == []
    assert "check_lane_d_absent" in {n.name for n in tree.body
                                     if isinstance(n, ast.FunctionDef)}
