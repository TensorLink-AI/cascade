"""W7a (plan rev6.14 (a)): the report signs the boundary's on-chain timestamp
and echoes `eval_delay_hours`; `validate_report` refuses a cohort frozen
before its judging start `t_max + 3600 * eval_delay_hours`, a cohort without
the boundary timestamp, a `boundary_ts_missing` skip on a stamped settlement,
and `forward_margin_hours > eval_delay_hours` (b > j)."""

from __future__ import annotations

import copy

import pytest

from .helpers import assert_valid, loser_then_crown, reseal

HOUR = 3600


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return loser_then_crown(tmp_path_factory.mktemp("w7a_rev614"))


def _refused(report, code, field):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    try:
        validate_report(report)
    except ReportSchemaError as exc:
        assert (exc.code, exc.field) == (code, field), \
            f"refused {exc.code} at {exc.field}; want {code} at {field}"
        return
    raise AssertionError(f"validate_report accepted a report it must refuse ({code} at {field})")


def test_the_report_signs_boundary_ts_and_echoes_eval_delay(built):
    r = built["s2"].report
    assert_valid(r)
    assert r["params"]["eval_delay_hours"] == float(built["cfg"].eval_delay_hours)
    assert isinstance(r.get("boundary_ts"), int), "rev6.14: the boundary timestamp is signed"
    for c in r["cohorts"]:
        assert r["boundary_ts"] >= c["t_max"] + HOUR * r["params"]["eval_delay_hours"]


def test_a_cohort_frozen_before_its_judging_start_is_refused(built):
    r = copy.deepcopy(built["s2"].report)
    assert r["cohorts"], "fixture: s2 judges a cohort"
    c = r["cohorts"][0]
    r["boundary_ts"] = int(c["t_max"] + HOUR * r["params"]["eval_delay_hours"]) - 1
    _refused(r, "bad_value", "cohorts[0]")


def test_a_cohort_without_the_boundary_timestamp_is_refused(built):
    r = copy.deepcopy(built["s2"].report)
    del r["boundary_ts"]
    _refused(r, "missing_field", "boundary_ts")


def test_boundary_ts_is_never_null(built):
    r = copy.deepcopy(built["s2"].report)
    r["boundary_ts"] = None
    _refused(r, "bad_type", "boundary_ts")


def test_a_boundary_ts_missing_skip_on_a_stamped_settlement_is_refused(built):
    r = copy.deepcopy(built["s2"].report)
    assert r["entries"], "fixture: s2 lists an entry"
    r["skipped"].append({"digest": r["entries"][0]["digest"], "reason": "boundary_ts_missing"})
    _refused(r, "bad_value", f"skipped[{len(r['skipped']) - 1}].reason")


def test_forward_margin_above_eval_delay_is_refused(built):
    r = copy.deepcopy(built["s2"].report)
    r["params"]["forward_margin_hours"] = float(r["params"]["eval_delay_hours"]) + 1.0
    _refused(reseal(r), "bad_value", "params.forward_margin_hours")
