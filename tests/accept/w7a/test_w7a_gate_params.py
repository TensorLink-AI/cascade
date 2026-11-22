"""W7a (RV4 correction): `params` carries the three knobs satellites need to
re-derive the gate draw, the source cap and the context floor (plan rev6
§1.3b "The draw" (`gate_weights`), §3.0 W5b row / T18e / B14
(`max_source_share`), §1.7 (`min_long_series`), §14.3 item 2; review
evidence/rv4.md BLOCKING W7a).

All three are **required**, like every other `params` field: §14.3 item 2
says every value a satellite needs is in the signed report and cross-checked
against `config_at(boundary)`, and an optional field would let a report omit
a value the cross-check needs. Ranges are the W2 loader's
(`cascade/shared/config/lane_d.py`): `gate_weights` a map `{suite: weight}`
with keys in `gate_suites` and finite weights ≥ 0 (absent suites weigh 0,
as parsed); `max_source_share` in (0, 1]; `min_long_series` an int ≥ 0.

On main every test fails on an `assert`: the report has no such params and
the closed schema refuses them as `unknown_field`.
"""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    BOUNDARY_1,
    CHALLENGER_UID,
    build_chain,
    coldkey,
    entry,
    fixture_key,
    lane_d_cfg,
    settle,
    two_settlements,
)

OVERLAY = {"gate_weights": {"gift": 2.0, "boom": 1.0}, "max_source_share": 0.5,
           "min_long_series": 120}


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7a_gate_params")


_MISSING = object()


def _want(cfg) -> dict:
    return {"gate_weights": {str(s): float(w) for s, w in cfg.gate_weights},
            "max_source_share": float(cfg.max_source_share),
            "min_long_series": int(cfg.min_long_series)}


def _check(params, cfg):
    for k, v in _want(cfg).items():
        got = params.get(k, _MISSING)
        assert got is not _MISSING, f"params.{k} is missing"
        assert got == v and type(got) is type(v), f"params.{k} = {got!r}, want {v!r}"


def test_params_carry_the_parsed_gate_and_source_knobs(scratch):
    built = two_settlements(scratch)
    for s in (built["s1"], built["s2"]):
        _check(s.report["params"], built["cfg"])
    assert built["s2"].report["params"]["gate_weights"] == {
        "gift": 1.0, "boom": 1.0, "time": 1.0, "fev": 1.0}


def test_params_follow_an_overlay_that_changes_them(scratch):
    """A non-default overlay is stated as parsed: unequal weights, two suites
    left out (weight 0, absent from the map), a looser cap, a lower floor."""
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    cfg = lane_d_cfg(**OVERLAY)
    chain = build_chain(scratch / "overlay")
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    state = scratch / "overlay" / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    s = settle(chain, cfg, state, fixture_key("signer"), boundary=BOUNDARY_1, snap=1,
               entries=[e], uid_of={e["hotkey"]: CHALLENGER_UID})
    assert s.report is not None
    _check(s.report["params"], cfg)
    assert s.report["params"]["gate_weights"] == {"gift": 2.0, "boom": 1.0}
    try:
        validate_report(s.report)
    except ReportSchemaError as exc:
        raise AssertionError(f"{exc.code} at {exc.field}: {exc}") from exc


def _s2(scratch) -> dict:
    return copy.deepcopy(two_settlements(scratch)["s2"].report)


def _refused(report, code, field):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    with pytest.raises(ReportSchemaError) as info:
        validate_report(report)
    got = (info.value.code, info.value.field)
    assert got == (code, field), f"refused {got}, want {(code, field)}"


@pytest.mark.parametrize("key,value,code,field", [
    ("max_source_share", 0.0, "bad_value", "params.max_source_share"),
    ("max_source_share", 1.5, "bad_value", "params.max_source_share"),
    ("max_source_share", -0.25, "bad_value", "params.max_source_share"),
    ("max_source_share", float("nan"), "bad_value", "params.max_source_share"),
    ("max_source_share", "0.25", "bad_type", "params.max_source_share"),
    ("min_long_series", -1, "bad_value", "params.min_long_series"),
    ("min_long_series", 300.5, "bad_type", "params.min_long_series"),
    ("min_long_series", True, "bad_type", "params.min_long_series"),
    ("gate_weights", [["gift", 1.0]], "bad_type", "params.gate_weights"),
    ("gate_weights", None, "bad_type", "params.gate_weights"),
    ("gate_weights", {"gift": -1.0}, "bad_value", "params.gate_weights.gift"),
    ("gate_weights", {"gift": float("inf")}, "bad_value", "params.gate_weights.gift"),
    ("gate_weights", {"gift": "1"}, "bad_type", "params.gate_weights.gift"),
    ("gate_weights", {"gift": False}, "bad_type", "params.gate_weights.gift"),
    ("gate_weights", {"gift": 1.0, "nope": 1.0}, "bad_value", "params.gate_weights.nope"),
])
def test_gate_param_values_are_checked(scratch, key, value, code, field):
    r = _s2(scratch)
    r["params"][key] = value
    _refused(r, code, field)


def test_gate_param_edge_values_validate(scratch):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    for key, value in (("max_source_share", 1.0), ("min_long_series", 0),
                       ("gate_weights", {"gift": 0.0, "boom": 3})):
        r = _s2(scratch)
        r["params"][key] = value
        try:
            validate_report(r)
        except ReportSchemaError as exc:
            raise AssertionError(f"params.{key} = {value!r} refused: {exc.code} at "
                                 f"{exc.field}") from exc
