"""W7a (rev6.10, rev6.11 (a), rev6.12 correction): the report carries the
embargo, the fresh / embargo split and the W5j draw (plan §3.0 W7a row;
rev6.10 (a)/(b); rev6.12 (a)/(b); README "Correction (rev6.11, rev6.12)").

Per window: `freq`, the effective `embargo_hours`, `min_release_ts`,
`rule` (`embargo` | `embargo0`), `embargo_window` (rev6.12: `false` = fresh,
passes at e = 0 on every point; `true` = embargo, a mixed target included),
and the draw cell `timeframe`, `bin`, `kind`, `freq_class`, `h`, `origin`
(names as W5j / W5i pin them). Per cohort: `fresh_windows`,
`embargo_windows`. Per verdict: the `fresh` and `embargo` sub-scores
`{n, geomean}`. Params: `embargo_hours`, `max_windows_per_settlement`,
`min_fresh_window_share`. Optional top-level `draw` (present iff a cohort was
drawn): `boundary_block_hash`, `seed`, `max_windows_per_settlement`,
`cohorts[]` `{index, quota, capacity, candidates_sha256}`.

The validator tests start from a real `settle_lane_d` report with the fields
spliced in (`helpers.with_embargo_fields`; W5a / W5j / W2e are not built), so
each refusal is the one defect a test introduces. On main every spliced
report is refused (`unknown_field`) and the real ones lack the fields: an
`AssertionError` everywhere.
"""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    DRAW_COHORT_FIELDS,
    DRAW_FIELDS,
    FREQ_CLASSES,
    SCHEMA_FILE,
    SUBSCORE_FIELDS,
    W2E_DEFAULTS,
    WINDOW_BINS,
    WINDOW_KINDS,
    WINDOW_RULES,
    assert_valid,
    loser_then_crown,
    reseal,
    with_embargo_fields,
)

NEW_WINDOW_FIELDS = ("freq", "embargo_hours", "min_release_ts", "rule", "embargo_window",
                     "timeframe", "bin", "kind", "freq_class", "h", "origin")


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    return tmp_path_factory.mktemp("w7a_embargo")


def _base(scratch, **kw) -> dict:
    built = loser_then_crown(scratch)
    return with_embargo_fields(built["s2"].report, **kw)


def _refused(report, code, field):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    try:
        validate_report(report)
    except ReportSchemaError as exc:
        assert (exc.code, exc.field) == (code, field), \
            f"refused {exc.code} at {exc.field}; want {code} at {field}"
        return
    raise AssertionError(f"validate_report accepted a report it must refuse ({code} at {field})")


# ── the spliced fixture validates ───────────────────────────────────────────


@pytest.mark.parametrize("k", [0, 4, 24])
def test_a_report_with_the_new_fields_validates(scratch, k):
    """All fresh, a few embargo windows, and exactly half (the 0.5 bound)."""
    r = _base(scratch, embargo_windows=k)
    c = r["cohorts"][0]
    assert (c["fresh_windows"], c["embargo_windows"]) == (48 - k, k)
    assert_valid(r)


def test_the_real_reports_carry_the_new_fields(scratch):
    """The producer writes every rev6.10-rev6.12 field itself (W7a's emission;
    the e2e fixture pins e = 0, so every window is fresh, `embargo0`)."""
    built = loser_then_crown(scratch)
    r = built["s2"].report
    assert_valid(r)
    for k, v in W2E_DEFAULTS.items():
        assert r["params"].get(k) == getattr(built["cfg"], k, v), f"params.{k}"
    for w in r["windows"]:
        missing = [f for f in NEW_WINDOW_FIELDS if f not in w]
        assert not missing, f"window {w['window_id']} lacks {missing}"
        assert (w["embargo_hours"], w["rule"], w["embargo_window"]) == (0.0, "embargo0", False)
        assert (w["timeframe"], w["bin"], w["kind"], w["h"]) == ("H", "forward", "forward", 64)
        assert w["origin"] == int(w["window_id"].rsplit(":", 2)[1])
        assert w["min_release_ts"] == w["min_first_seen_ts"]
    (c,) = r["cohorts"]
    assert (c["fresh_windows"], c["embargo_windows"]) == (len(c["window_ids"]), 0)
    want = with_embargo_fields(r)
    for v, wv in zip(r["verdicts"], want["verdicts"], strict=True):
        assert v["fresh"]["n"] == wv["fresh"]["n"] and v["embargo"] == wv["embargo"]
        assert abs(v["fresh"]["geomean"] - wv["fresh"]["geomean"]) \
            <= 1e-9 * wv["fresh"]["geomean"]


# ── the schema file ─────────────────────────────────────────────────────────


def test_schema_file_describes_the_new_fields():
    import json

    doc = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    props = doc["properties"]
    assert "draw" in props and "draw" not in doc["required"], "draw is optional"
    defs = doc.get("$defs", {})

    def resolve(node):
        while isinstance(node, dict) and "$ref" in node:
            node = defs[node["$ref"].rsplit("/", 1)[-1]]
        return node

    draw = resolve(props["draw"])
    assert set(draw["required"]) == set(DRAW_FIELDS) and draw["additionalProperties"] is False
    win = resolve(resolve(props["windows"])["items"])
    assert set(NEW_WINDOW_FIELDS) <= set(win["required"]), win["required"]
    for name, values in (("window_rule", WINDOW_RULES), ("window_kind", WINDOW_KINDS),
                         ("window_bin", WINDOW_BINS), ("freq_class", FREQ_CLASSES)):
        assert sorted((defs.get(name) or {}).get("enum") or []) == sorted(values), name


# ── missing, unknown and badly typed fields ─────────────────────────────────


@pytest.mark.parametrize("name", NEW_WINDOW_FIELDS)
def test_every_new_window_field_is_required(scratch, name):
    r = _base(scratch)
    del r["windows"][0][name]
    _refused(r, "missing_field", f"windows[0].{name}")


@pytest.mark.parametrize("name", ["fresh_windows", "embargo_windows"])
def test_cohort_counts_are_required(scratch, name):
    r = _base(scratch)
    del r["cohorts"][0][name]
    _refused(r, "missing_field", f"cohorts[0].{name}")


@pytest.mark.parametrize("which", ["fresh", "embargo"])
@pytest.mark.parametrize("name", SUBSCORE_FIELDS)
def test_verdict_subscores_are_required_and_closed(scratch, which, name):
    r = _base(scratch)
    del r["verdicts"][0][which][name]
    _refused(r, "missing_field", f"verdicts[0].{which}.{name}")
    r = _base(scratch)
    r["verdicts"][0][which]["note"] = 1
    _refused(r, "unknown_field", f"verdicts[0].{which}.note")


@pytest.mark.parametrize("name", DRAW_FIELDS)
def test_every_draw_field_is_required(scratch, name):
    r = _base(scratch)
    del r["draw"][name]
    _refused(r, "missing_field", f"draw.{name}")


@pytest.mark.parametrize("name", DRAW_COHORT_FIELDS)
def test_every_draw_cohort_field_is_required(scratch, name):
    r = _base(scratch)
    del r["draw"]["cohorts"][0][name]
    _refused(r, "missing_field", f"draw.cohorts[0].{name}")


def test_draw_is_optional_but_never_null(scratch):
    r = _base(scratch)
    r.pop("draw")
    assert_valid(r)
    r["draw"] = None
    _refused(r, "bad_type", "draw")


@pytest.mark.parametrize("path,value", [
    (("windows", 0, "rule"), "embargo48"),
    (("windows", 0, "kind"), "replay"),
    (("windows", 0, "bin"), "huge"),
    (("windows", 0, "freq_class"), "hourly"),
    (("windows", 0, "embargo_hours"), -1.0),
    (("windows", 0, "h"), 0),
    (("draw", "boundary_block_hash"), "0x" + "G" * 64),
    (("draw", "cohorts", 0, "candidates_sha256"), "abc"),
])
def test_values_outside_the_enum_or_range_are_refused(scratch, path, value):
    r = _base(scratch)
    obj = r
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value
    field = path[0] + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path[1:])
    _refused(r, "bad_value", field)


@pytest.mark.parametrize("path,value", [
    (("windows", 0, "embargo_window"), 0),
    (("windows", 0, "embargo_hours"), "48"),
    (("windows", 0, "h"), True),
    (("cohorts", 0, "fresh_windows"), 4.5),
    (("params", "min_fresh_window_share"), "0.5"),
])
def test_wrong_types_are_refused(scratch, path, value):
    r = _base(scratch)
    obj = r
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value
    field = path[0] + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path[1:])
    _refused(r, "bad_type", field)


# ── cross-field checks ──────────────────────────────────────────────────────


def test_a_window_embargo_above_the_config_echo_is_refused(scratch):
    r = _base(scratch, embargo_windows=4)
    w = next(w for w in r["windows"] if w["embargo_window"])
    i = r["windows"].index(w)
    w["embargo_hours"] = float(r["params"]["embargo_hours"]) + 1.0
    w["min_release_ts"] = int(w["min_first_seen_ts"]) + int(round(3600 * w["embargo_hours"]))
    _refused(r, "bad_value", f"windows[{i}].embargo_hours")


def test_the_rule_names_the_effective_embargo(scratch):
    r = _base(scratch)
    r["windows"][0]["rule"] = "embargo"
    _refused(r, "bad_value", "windows[0].rule")
    r = _base(scratch, embargo_windows=4)
    i = next(i for i, w in enumerate(r["windows"]) if w["embargo_window"])
    r["windows"][i]["rule"] = "embargo0"
    _refused(r, "bad_value", f"windows[{i}].rule")


def test_min_release_ts_is_first_seen_plus_the_embargo(scratch):
    r = _base(scratch, embargo_windows=4)
    i = next(i for i, w in enumerate(r["windows"]) if w["embargo_window"])
    r["windows"][i]["min_release_ts"] = int(r["windows"][i]["min_first_seen_ts"])
    _refused(r, "bad_value", f"windows[{i}].min_release_ts")


def test_an_embargo0_window_is_never_an_embargo_window(scratch):
    """`rule = embargo0` means every point passes at e = 0: the window is
    fresh."""
    r = _base(scratch)
    r["windows"][0]["embargo_window"] = True
    _refused(r, "bad_value", "windows[0].embargo_window")


def test_h_and_origin_agree_with_the_window_id(scratch):
    r = _base(scratch)
    r["windows"][0]["h"] = 32
    _refused(r, "bad_value", "windows[0].h")
    r = _base(scratch)
    r["windows"][0]["origin"] += 1
    _refused(r, "bad_value", "windows[0].origin")


def test_cohort_counts_agree_with_the_window_flags(scratch):
    r = _base(scratch, embargo_windows=4)
    r["cohorts"][0]["fresh_windows"] += 1
    _refused(r, "bad_value", "cohorts[0].fresh_windows")
    r = _base(scratch, embargo_windows=4)
    r["cohorts"][0]["embargo_windows"] -= 1
    _refused(r, "bad_value", "cohorts[0].embargo_windows")


def test_a_frozen_set_below_the_minimum_fresh_share_is_refused(scratch):
    """rev6.12 (b): F / (F + E) >= min_fresh_window_share on every frozen set
    (0.5: 24 of 48 passes, 25 embargo windows don't)."""
    r = _base(scratch, embargo_windows=25)
    _refused(r, "bad_value", "cohorts[0].embargo_windows")
    r = _base(scratch, embargo_windows=25)
    r["params"]["min_fresh_window_share"] = 0.25
    assert_valid(r)


def test_verdict_subscores_agree_with_the_rows(scratch):
    r = _base(scratch, embargo_windows=4)
    r["verdicts"][0]["fresh"]["n"] += 1
    _refused(r, "bad_value", "verdicts[0].fresh.n")
    r = _base(scratch, embargo_windows=4)
    r["verdicts"][0]["embargo"]["geomean"] *= 1.01
    _refused(r, "bad_value", "verdicts[0].embargo.geomean")
    r = _base(scratch)
    r["verdicts"][0]["embargo"]["geomean"] = 1.0
    _refused(r, "bad_value", "verdicts[0].embargo.geomean")


def test_the_draw_is_bound_to_its_hash_budget_and_cohorts(scratch):
    r = _base(scratch)
    r["draw"]["seed"] = r["draw"]["seed"] + "x"
    _refused(r, "bad_value", "draw.seed")
    r = _base(scratch)
    r["draw"]["max_windows_per_settlement"] = 4999
    _refused(r, "bad_value", "draw.max_windows_per_settlement")
    r = _base(scratch)
    r["draw"]["cohorts"][0]["index"] = 7
    _refused(r, "bad_value", "draw.cohorts[0].index")
    r = _base(scratch)
    key = next(iter(r["draw"]["cohorts"][0]["quota"]))
    r["draw"]["cohorts"][0]["quota"][key] = r["draw"]["cohorts"][0]["capacity"][key] + 1
    _refused(r, "bad_value", f"draw.cohorts[0].quota.{key}")
    r = _base(scratch)
    r["draw"]["cohorts"][0]["quota"] = {"H-forward": 48}
    r["draw"]["cohorts"][0]["capacity"] = {"H-forward": 48}
    _refused(r, "bad_value", "draw.cohorts[0].quota.H-forward")


def test_the_judged_windows_never_exceed_the_budget(scratch):
    r = _base(scratch)
    r["params"]["max_windows_per_settlement"] = 47
    r["draw"]["max_windows_per_settlement"] = 47
    for c in r["draw"]["cohorts"]:
        for k in c["quota"]:
            c["quota"][k] = min(c["quota"][k], 47)
    _refused(reseal(copy.deepcopy(r)), "bad_value", "cohorts")
