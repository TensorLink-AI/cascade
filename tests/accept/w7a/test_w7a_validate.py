"""W7a: `validate_report` refuses everything outside `lane-d-report/1`
(plan rev6 §3.0 W7a row: "JSON-schema; unknown version refused", §14.3
items 3 and 4, B7, §3.2 invariant 11).

Every case starts from a real report produced by `settle_lane_d` on the
synthetic chain (no torch) and changes one thing. Row mutations are
`reseal`ed (rows_sha256 recomputed) so the only defect left is the one the
test introduced. On main the tests fail at `from cascade.validator.lane_d_report
import validate_report` (an `ImportError` for a path in PATHS).
"""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    PARAMS_FIELDS,
    SCHEMA_ERROR_CODES,
    TOP_LEVEL_REQUIRED,
    reseal,
    two_settlements,
)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7a_validate")


def _reports(scratch):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    built = two_settlements(scratch)
    return built, validate_report, ReportSchemaError


def _refused(scratch, report, code, field=None, *, prefix=None):
    _, validate_report, err = _reports(scratch)
    with pytest.raises(err) as info:
        validate_report(report)
    assert info.value.code == code, f"code {info.value.code!r} (field {info.value.field!r})"
    assert info.value.code in SCHEMA_ERROR_CODES
    if field is not None:
        assert info.value.field == field, f"field {info.value.field!r}, want {field!r}"
    if prefix is not None:
        assert str(info.value.field).startswith(prefix), info.value.field


def _s2(scratch) -> dict:
    built, _, _ = _reports(scratch)
    return copy.deepcopy(built["s2"].report)


# ── the genuine reports pass ────────────────────────────────────────────────


def test_settle_reports_validate(scratch):
    built, validate_report, _ = _reports(scratch)
    for s in (built["s1"], built["s2"]):
        assert s.report is not None
        assert validate_report(s.report) is None


def test_extensions_map_is_optional_and_open(scratch):
    _, validate_report, _ = _reports(scratch)
    r = _s2(scratch)
    r["extensions"] = {"operator_note_id": 7, "nested": {"anything": [1, 2]}}
    assert validate_report(r) is None
    r["extensions"] = ["not", "a", "map"]
    _refused(scratch, r, "bad_type", "extensions")


# ── version and scoring method ──────────────────────────────────────────────


def test_not_an_object_is_refused(scratch):
    _refused(scratch, ["lane-d-report/1"], "not_object")


@pytest.mark.parametrize("schema", ["lane-d-report/2", "lane-d-report/0", "lane-d-report",
                                    "mkt-report/3", 1])
def test_unknown_schema_version_is_refused(scratch, schema):
    r = _s2(scratch)
    r["schema"] = schema
    _refused(scratch, r, "unknown_schema", "schema")


def test_unknown_schema_is_checked_before_anything_else(scratch):
    _refused(scratch, {"schema": "lane-d-report/2", "surprise": True}, "unknown_schema", "schema")
    r = _s2(scratch)
    del r["schema"]
    _refused(scratch, r, "unknown_schema", "schema")


def test_scoring_method_absent_is_refused(scratch):
    r = _s2(scratch)
    del r["scoring_method"]
    _refused(scratch, r, "missing_field", "scoring_method")


@pytest.mark.parametrize("method", ["sample_path", "copula", "Quantile_Head", "quantile_head ",
                                    ""])
def test_scoring_method_other_than_quantile_head_is_refused(scratch, method):
    r = _s2(scratch)
    r["scoring_method"] = method
    _refused(scratch, r, "bad_value", "scoring_method")


# ── closed objects: missing and unknown fields ──────────────────────────────


@pytest.mark.parametrize("name", TOP_LEVEL_REQUIRED[1:])
def test_every_required_top_level_field_is_required(scratch, name):
    r = _s2(scratch)
    del r[name]
    _refused(scratch, r, "missing_field", name)


def test_unknown_top_level_field_is_refused(scratch):
    r = _s2(scratch)
    r["verdict_note"] = "challenger looked fine"
    _refused(scratch, r, "unknown_field", "verdict_note")


@pytest.mark.parametrize("name", PARAMS_FIELDS)
def test_every_params_field_is_required(scratch, name):
    r = _s2(scratch)
    del r["params"][name]
    _refused(scratch, r, "missing_field", f"params.{name}")


@pytest.mark.parametrize("path", [
    ("windows", 0, "min_target_ts"), ("windows", 0, "min_first_seen_ts"),
    ("windows", 0, "mirror_lag_hours"), ("windows", 0, "feed"),
    ("entries", 0, "coldkey"), ("entries", 0, "reason"),
    ("cohorts", 0, "first_seen_roots"), ("cohorts", 0, "refused"),
    ("verdicts", 0, "confirmation"), ("verdicts", 0, "lowfreq"),
    ("pool", None, "first_seen_roots"),
])
def test_nested_fields_are_required(scratch, path):
    r = _s2(scratch)
    top, idx, key = path
    obj = r[top] if idx is None else r[top][idx]
    del obj[key]
    field = f"{top}.{key}" if idx is None else f"{top}[{idx}].{key}"
    _refused(scratch, r, "missing_field", field)


def test_nested_confirmation_lowfreq_and_court_fields_are_required(scratch):
    for key in ("stage", "rule", "first_set_sha256", "first_king_digest"):
        r = _s2(scratch)
        del r["verdicts"][0]["confirmation"][key]
        _refused(scratch, r, "missing_field", f"verdicts[0].confirmation.{key}")
    for key in ("result", "freqs", "n_windows", "tau_public"):
        r = _s2(scratch)
        del r["verdicts"][0]["lowfreq"][key]
        _refused(scratch, r, "missing_field", f"verdicts[0].lowfreq.{key}")
    r = _s2(scratch)
    del r["pay"]["court"][0]["burn_reason"]
    _refused(scratch, r, "missing_field", "pay.court[0].burn_reason")


def test_unknown_nested_fields_are_refused(scratch):
    r = _s2(scratch)
    r["windows"][0]["publisher"] = "someone"
    _refused(scratch, r, "unknown_field", "windows[0].publisher")
    r = _s2(scratch)
    r["rows"][0]["note"] = "x"
    _refused(scratch, reseal(r), "unknown_field", "rows[0].note")
    r = _s2(scratch)
    r["params"]["new_knob"] = 1
    _refused(scratch, r, "unknown_field", "params.new_knob")


# ── types (a JSON type mismatch, null included) vs values ──────────────────


def test_wrong_types_are_refused(scratch):
    r = _s2(scratch)
    r["boundary"] = True
    _refused(scratch, r, "bad_type", "boundary")
    r = _s2(scratch)
    r["boundary"] = str(r["boundary"])
    _refused(scratch, r, "bad_type", "boundary")
    r = _s2(scratch)
    r["rows"][0]["crps"] = str(r["rows"][0]["crps"])
    _refused(scratch, reseal(r), "bad_type", "rows[0].crps")
    r = _s2(scratch)
    r["windows"][0]["mirror_lag_hours"] = "1.5"
    _refused(scratch, r, "bad_type", "windows[0].mirror_lag_hours")
    r = _s2(scratch)
    r["windows"][0]["min_first_seen_ts"] = None
    _refused(scratch, r, "bad_type", "windows[0].min_first_seen_ts")
    r = _s2(scratch)
    r["cohorts"][0]["refused"] = {"before_margin": "1"}
    _refused(scratch, r, "bad_type", "cohorts[0].refused.before_margin")


def test_inv11_null_entry_reason_is_refused(scratch):
    r = _s2(scratch)
    r["entries"][0]["reason"] = None
    _refused(scratch, r, "bad_type", "entries[0].reason")


@pytest.mark.parametrize("path,value", [
    (("entries", 0, "reason"), "lost the duel"),
    (("entries", 0, "reason"), "LOST_DUEL"),
    (("entries", 0, "status"), "frozen"),
    (("verdicts", 0, "outcome"), "won"),
    (("verdicts", 0, "confirmation", "stage"), "second"),
    (("verdicts", 0, "confirmation", "rule"), "latest"),
    (("verdicts", 0, "lowfreq", "result"), "skipped"),
    (("pool", "sha256"), "abc"),
    (("pool", "first_seen_roots", 0), "not-a-root"),
    (("rows_sha256",), "Z" * 64),
])
def test_values_outside_the_enum_or_format_are_refused(scratch, path, value):
    r = _s2(scratch)
    obj = r
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value
    field = path[0] + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path[1:])
    _refused(scratch, r, "bad_value", field)


def test_free_text_skip_reason_is_refused(scratch):
    r = _s2(scratch)
    r["skipped"] = [{"digest": r["entries"][0]["digest"], "reason": "waiting for more data"}]
    _refused(scratch, r, "bad_value", prefix="skipped[0]")


def test_unknown_window_refusal_reason_is_refused(scratch):
    r = _s2(scratch)
    r["cohorts"][0]["refused"] = {"too_old": 1}
    _refused(scratch, r, "bad_value", prefix="cohorts[0].refused")


def test_inv11_burn_slot_needs_exactly_one_typed_reason(scratch):
    r = _s2(scratch)
    slot = r["pay"]["court"][0]
    assert slot["uid"] is not None and slot["burn_reason"] is None
    slot["burn_reason"] = "house_king"
    _refused(scratch, r, "bad_value", "pay.court[0].burn_reason")
    r = _s2(scratch)
    r["pay"]["court"][0]["uid"] = None
    _refused(scratch, r, "bad_value", "pay.court[0].burn_reason")
    r = _s2(scratch)
    r["pay"]["court"][0].update(uid=None, burn_reason="ran_out")
    _refused(scratch, r, "bad_value", "pay.court[0].burn_reason")


# ── rows_sha256 ─────────────────────────────────────────────────────────────


def test_non_finite_row_value_is_refused(scratch):
    r = _s2(scratch)
    r["rows"][0]["crps"] = float("nan")
    _refused(scratch, r, "bad_value", "rows[0].crps")


def test_rows_sha256_mismatch_is_refused(scratch):
    r = _s2(scratch)
    r["rows"][0]["crps"] = float(r["rows"][0]["crps"]) + 1.0
    _refused(scratch, r, "rows_sha256_mismatch", "rows_sha256")
    _, validate_report, _ = _reports(scratch)
    assert validate_report(reseal(r)) is None, "a resealed row set is structurally valid"
