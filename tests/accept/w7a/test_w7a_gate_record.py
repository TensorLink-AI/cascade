"""W7a (RV4 correction): `lane-d-report/1` carries the §1.3b gate record, so
W5e can sign it without a schema bump (plan rev6 §1.3b, §1.5 "the drawn
suite", §14.3 items 2-4, §16 rule 1; review evidence/rv4.md BLOCKING W7a).

The record is the optional top-level `gate`: W3b1's `gate_record` (`suite`,
`block`, `block_hash`, `sentinel`, `eligible`) plus `results`, one per
candidate swept at the boundary, projected from W3b2's `paired_gate`
(`digest`, `outcome`, the four LCBs, `blocked_by`, `session`). The fixture
gate is built from the real W3b1 draw and the real W3b2 gate and spliced into
a real `settle_lane_d` report (synthetic chain, no torch), since W5e's wiring
does not exist yet.

On main every test that inserts a gate fails on an `assert`: the closed
schema refuses `gate` itself (`unknown_field at gate`), not the defect the
test introduced.
"""

from __future__ import annotations

import copy
import json

import pytest

from .helpers import (
    BOUNDARY_2,
    GATE_FIELDS,
    GATE_METRICS,
    GATE_OUTCOMES,
    GATE_REFS,
    GATE_RESULT_FIELDS,
    GATE_SENTINEL,
    SCHEMA_FILE,
    build_gate,
    text_digest,
    two_settlements,
)

FAULTY = text_digest("w7a gate faulty candidate")
HELD = text_digest("w7a gate held candidate")
REGRESSED = text_digest("w7a gate regressed candidate")


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7a_gate")


_GATE: dict[str, dict] = {}


def _gated(scratch) -> dict:
    """The crowning settlement's report (BOUNDARY_2) carrying a gate record
    with one result per outcome: the crowned digest passes, the others
    regress, fault and are held."""
    built = two_settlements(scratch)
    key = str(scratch)
    if key not in _GATE:
        crowned = built["entry"]["digest"]
        _GATE[key] = build_gate(built["cfg"], BOUNDARY_2, {
            crowned: {"cand_factor": 0.95},
            REGRESSED: {"cand_factor": 1.2},
            FAULTY: {"faulty": "candidate"},
            HELD: {"faulty": "king_d"},
        })
    r = copy.deepcopy(built["s2"].report)
    r["gate"] = copy.deepcopy(_GATE[key])
    return r


def _validate(report):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    try:
        validate_report(report)
    except ReportSchemaError as exc:
        raise AssertionError(f"a valid gated report is refused: {exc.code} at {exc.field}: "
                             f"{exc}") from exc


def _refused(report, code, field):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    with pytest.raises(ReportSchemaError) as info:
        validate_report(report)
    got = (info.value.code, info.value.field)
    assert got == (code, field), f"refused {got}, want {(code, field)}"


def _result(r, outcome) -> dict:
    return next(x for x in r["gate"]["results"] if x["outcome"] == outcome)


def _index(r, outcome) -> int:
    return [x["outcome"] for x in r["gate"]["results"]].index(outcome)


# ── the fixture is the real W3b1 draw and W3b2 gate ─────────────────────────


def test_fixture_gate_is_the_w3b1_draw_and_the_w3b2_outcomes(scratch):
    from cascade.validator.lane_d_gate import gate_suite_for

    r = _gated(scratch)
    g = r["gate"]
    assert g["block"] == r["boundary"] == BOUNDARY_2
    assert g["sentinel"] == GATE_SENTINEL and GATE_SENTINEL not in g["eligible"]
    cfg = two_settlements(scratch)["cfg"]
    assert g["suite"] == gate_suite_for(g["block_hash"], gate_suites=cfg.gate_suites,
                                        gate_weights=cfg.gate_weights, sentinel=g["sentinel"])
    assert sorted(x["outcome"] for x in g["results"]) == sorted(GATE_OUTCOMES)
    assert _result(r, "pass")["digest"] == r["crowned"]


# ── a gated report validates; a report without one still does ──────────────


def test_report_carrying_a_gate_record_validates(scratch):
    r = _gated(scratch)
    _validate(r)
    # strict JSON round trip: what is signed is what validates
    _validate(json.loads(json.dumps(r, allow_nan=False)))


def test_gate_is_optional_reports_without_it_validate(scratch):
    from cascade.validator.lane_d_report import validate_report

    built = two_settlements(scratch)
    for s in (built["s1"], built["s2"]):
        r = copy.deepcopy(s.report)
        r.pop("gate", None)
        assert validate_report(r) is None


def test_no_gated_candidate_means_no_gate_key(scratch):
    """A W1.5-format pool judges nothing (every window `unlisted_feed`), so no
    candidate reaches the gate and the report carries no `gate` key. `null`
    is not a second way of saying so: one canonical form for a signed body."""
    legacy = two_settlements(scratch / "legacy", rich=False)
    for s in (legacy["s1"], legacy["s2"]):
        assert s.report["judged"] == [] and "gate" not in s.report
    r = _gated(scratch)
    r["gate"] = None
    _refused(r, "bad_type", "gate")


def test_schema_file_describes_gate_as_a_closed_object():
    doc = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert "gate" in (doc.get("properties") or {}), "the schema file has no `gate` property"
    assert "gate" not in (doc.get("required") or [])


def test_module_gate_enums_equal_the_w3b2_ones():
    from cascade.validator import lane_d_bench, lane_d_report

    assert sorted(map(str, lane_d_report.GATE_OUTCOMES)) == sorted(lane_d_bench.OUTCOMES)
    assert sorted(map(str, lane_d_report.GATE_REFS)) == sorted(lane_d_bench.REFS)
    assert set(GATE_OUTCOMES) == set(lane_d_bench.OUTCOMES)
    assert set(GATE_REFS) == set(lane_d_bench.REFS)


# ── closed: missing and unknown fields ──────────────────────────────────────


@pytest.mark.parametrize("name", GATE_FIELDS)
def test_every_gate_field_is_required(scratch, name):
    r = _gated(scratch)
    del r["gate"][name]
    _refused(r, "missing_field", f"gate.{name}")


@pytest.mark.parametrize("name", GATE_RESULT_FIELDS)
def test_every_gate_result_field_is_required(scratch, name):
    r = _gated(scratch)
    del r["gate"]["results"][0][name]
    _refused(r, "missing_field", f"gate.results[0].{name}")


@pytest.mark.parametrize("ref,metric", [(r, m) for r in GATE_REFS for m in GATE_METRICS])
def test_all_four_lcbs_are_required(scratch, ref, metric):
    r = _gated(scratch)
    i = _index(r, "pass")
    del r["gate"]["results"][i]["lcb"][ref][metric]
    _refused(r, "missing_field", f"gate.results[{i}].lcb.{ref}.{metric}")


def test_unknown_gate_fields_are_refused(scratch):
    r = _gated(scratch)
    r["gate"]["note"] = "drawn fairly"
    _refused(r, "unknown_field", "gate.note")
    r = _gated(scratch)
    r["gate"]["results"][0]["reason"] = "candidate: suite 'time' item has a NaN"
    _refused(r, "unknown_field", "gate.results[0].reason")
    r = _gated(scratch)
    i = _index(r, "pass")
    r["gate"]["results"][i]["lcb"]["best_ever"] = {"crps": 0.0, "mase": 0.0}
    _refused(r, "unknown_field", f"gate.results[{i}].lcb.best_ever")
    r = _gated(scratch)
    r["gate"]["results"][i]["lcb"]["house"]["mae"] = 0.0
    _refused(r, "unknown_field", f"gate.results[{i}].lcb.house.mae")


# ── values ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("outcome", ["won", "PASS", "fail", "lowfreq_regression", "hold", ""])
def test_gate_result_with_a_bad_outcome_is_refused(scratch, outcome):
    r = _gated(scratch)
    r["gate"]["results"][0]["outcome"] = outcome
    _refused(r, "bad_value", "gate.results[0].outcome")


def test_gate_numbers_are_finite_numbers(scratch):
    r = _gated(scratch)
    i = _index(r, "pass")
    r["gate"]["results"][i]["lcb"]["king_d"]["crps"] = float("nan")
    _refused(r, "bad_value", f"gate.results[{i}].lcb.king_d.crps")
    r = _gated(scratch)
    r["gate"]["results"][i]["lcb"]["house"]["mase"] = float("-inf")
    _refused(r, "bad_value", f"gate.results[{i}].lcb.house.mase")
    r = _gated(scratch)
    r["gate"]["results"][i]["lcb"]["house"]["mase"] = "0.01"
    _refused(r, "bad_type", f"gate.results[{i}].lcb.house.mase")
    r = _gated(scratch)
    r["gate"]["results"][i]["lcb"]["house"]["mase"] = True
    _refused(r, "bad_type", f"gate.results[{i}].lcb.house.mase")


@pytest.mark.parametrize("path,value,code", [
    (("block",), -1, "bad_value"),
    (("block",), True, "bad_type"),
    (("block",), "9721800", "bad_type"),
    (("block_hash",), "ab" * 32, "bad_value"),
    (("block_hash",), "0x" + "ab" * 31, "bad_value"),
    (("block_hash",), "0x" + "AB" * 32, "bad_value"),
    (("block_hash",), None, "bad_type"),
    (("suite",), "", "bad_value"),
    (("suite",), None, "bad_type"),
    (("sentinel",), "", "bad_value"),
    (("sentinel",), 3, "bad_type"),
    (("eligible",), "boom", "bad_type"),
    (("eligible", 0), "", "bad_value"),
    (("results",), {}, "bad_type"),
    (("results", 0, "digest"), "not-a-digest", "bad_value"),
    (("results", 0, "session"), "", "bad_value"),
    (("results", 0, "session"), None, "bad_type"),
    (("results", 0, "blocked_by"), "house", "bad_type"),
])
def test_gate_types_and_formats(scratch, path, value, code):
    r = _gated(scratch)
    obj = r["gate"]
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value
    field = "gate" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path)
    _refused(r, code, field)


def test_unarmed_sentinel_is_null(scratch):
    """W-S1 unarmed: `gate_record(..., sentinel=None)` gives `sentinel: null`."""
    r = _gated(scratch)
    r["gate"]["sentinel"] = None
    _validate(r)


def test_blocked_by_names_only_the_two_references(scratch):
    r = _gated(scratch)
    i = _index(r, "gate_regression")
    r["gate"]["results"][i]["blocked_by"] = ["king_d", "best_ever"]
    _refused(r, "bad_value", f"gate.results[{i}].blocked_by[1]")


def test_gate_block_is_the_report_boundary(scratch):
    """§1.3b: the suite is drawn from the settlement boundary's block hash."""
    r = _gated(scratch)
    r["gate"]["block"] = r["boundary"] - 900
    _refused(r, "bad_value", "gate.block")


def test_gate_suite_is_one_of_the_eligible_suites(scratch):
    r = _gated(scratch)
    r["gate"]["suite"] = GATE_SENTINEL
    _refused(r, "bad_value", "gate.suite")


def test_gate_results_are_non_empty_and_one_per_digest(scratch):
    r = _gated(scratch)
    r["gate"]["results"] = []
    _refused(r, "bad_value", "gate.results")
    r = _gated(scratch)
    r["gate"]["results"][1]["digest"] = r["gate"]["results"][0]["digest"]
    _refused(r, "bad_value", "gate.results[1].digest")


# ── outcome-consistency (structured, like the court slot's burn_reason) ─────


def test_lcbs_present_iff_the_gate_computed_a_verdict(scratch):
    """`pass` / `gate_regression` carry the four LCBs; `checkpoint_fault` and
    `bench_hold` carry `lcb: null` (W3b2 computes none)."""
    r = _gated(scratch)
    i = _index(r, "pass")
    r["gate"]["results"][i]["lcb"] = None
    _refused(r, "bad_value", f"gate.results[{i}].lcb")
    for outcome in ("checkpoint_fault", "bench_hold"):
        r = _gated(scratch)
        i = _index(r, outcome)
        r["gate"]["results"][i]["lcb"] = copy.deepcopy(_result(r, "pass")["lcb"])
        _refused(r, "bad_value", f"gate.results[{i}].lcb")


def test_blocked_by_is_non_empty_iff_gate_regression(scratch):
    r = _gated(scratch)
    i = _index(r, "gate_regression")
    r["gate"]["results"][i]["blocked_by"] = []
    _refused(r, "bad_value", f"gate.results[{i}].blocked_by")
    for outcome in ("pass", "checkpoint_fault", "bench_hold"):
        r = _gated(scratch)
        i = _index(r, outcome)
        r["gate"]["results"][i]["blocked_by"] = ["house"]
        _refused(r, "bad_value", f"gate.results[{i}].blocked_by")
