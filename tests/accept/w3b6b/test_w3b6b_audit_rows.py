"""W3b6b through the one lane D checker (rev6.18 B2, owner 2026-11-23): the
bound-bench-record re-derivation (``check_gate_rows``, formerly only in the
satellite replay) runs inside ``check_lane_d(..., bench_rows=)``. rev6.18 B3
classes it ``ld_bench``: alert, so a bad record is reported and the signed
court stays paid. On main ``check_lane_d`` takes no ``bench_rows``."""

from __future__ import annotations

import json

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def gated(tmp_path_factory):
    r, rep = h.gated(tmp_path_factory.mktemp("w3b6b_audit_rows"))
    return r, rep, h.persisted_rows(r)


def _findings(r, rep, rows):
    from cascade.audit.lane_d import lane_d_findings

    return lane_d_findings(rep, r.cfg.lane_d, signer_hotkey=h.w8b.signer().ss58_address,
                           bench_rows=rows)


def test_honest_rows_raise_no_bench_finding(gated):
    r, rep, rows = gated
    f = _findings(r, rep, rows)
    assert not [c for c, _ in f.findings if c == "ld_bench"], f.findings
    assert f.burn is None, f.findings


def test_a_tampered_record_is_an_ld_bench_alert(gated):
    r, rep, rows = gated
    rows = dict(rows)
    (res,) = h.gate_results(rep)
    cand = res["rows"]["candidate"]
    rec = json.loads(rows[cand])
    rec["suites"][rep["gate"]["suite"]]["rows"][0]["crps"] *= 0.5
    rows[cand] = h.json_text(rec).encode("ascii")
    f = _findings(r, rep, rows)
    assert f.burn is None, f.findings
    assert any(c == "ld_bench" and "gate.results[0].rows" in d for c, d in f.alerts), \
        f.findings
