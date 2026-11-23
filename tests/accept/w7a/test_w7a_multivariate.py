"""F2 (plan rev6.18 A4): the signed report records each multichannel
window's channel pick; univariate windows carry none of it. Contract:
README.md "Multichannel windows (rev6.18 F2)".

On a main without F2 the schema has no channel fields and `settle_mv`
(`mv.py`) fails on its `max_channels` assertion.
"""

from __future__ import annotations

import copy
import json

import pytest

from . import helpers as h
from . import mv


@pytest.fixture
def mv_report(tmp_path):
    r = mv.crowned_mv(tmp_path, channels={"src1": 3, "src3": 40},
                      empty={"src3": (5, 17, 22, 30, 33, 38, 39)})
    return r


def test_schema_declares_the_channel_fields_on_a_window():
    schema = json.loads(h.SCHEMA_FILE.read_text(encoding="utf-8"))
    props = schema["$defs"]["window"]["properties"]
    for k in mv.CHANNEL_FIELDS:
        assert k in props, f"$defs.window.properties.{k}"
    assert k not in schema["$defs"]["window"]["required"]
    then = schema["$defs"]["window"]["then"]["required"]
    assert sorted(then) == sorted(mv.CHANNEL_FIELDS)


def test_every_multichannel_window_records_its_pick_and_univariate_ones_none(mv_report):
    rep, chain = mv_report["s1"].report, mv_report["chain"]
    h.assert_valid(rep)
    picks = chain["picks"][1]
    seen = 0
    for w in rep["windows"]:
        if w["window_id"] not in picks:
            assert not any(k in w for k in mv.CHANNEL_FIELDS), w["window_id"]
            continue
        seen += 1
        want = picks[w["window_id"]]
        assert w["channels"] == want and w["n_channels"] == len(want)
        assert w["channels_sha256"] == mv.channels_sha256(want)
        assert w["series_id"] == w["window_id"].rsplit(":", 2)[0]
        assert w["n_channels_raw"] == (3 if w["source"] == "src1" else 40)
    assert seen


def _mutated(rep: dict, fn) -> dict:
    out = copy.deepcopy(rep)
    w = next(x for x in out["windows"] if "n_channels" in x and x["n_channels_raw"] == 40)
    fn(w)
    return h.reseal(out)


def _refused(report, *codes: str, field: str | None = None) -> None:
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    with pytest.raises(ReportSchemaError) as e:
        validate_report(report)
    assert e.value.code in codes
    if field is not None:
        assert str(e.value.field).endswith(field)


def test_a_partial_channel_record_is_refused(mv_report):
    rep = mv_report["s1"].report
    h.assert_valid(rep)
    # The schema's if/then states the all-or-none rule (a then-branch defect is
    # a bad_value); the cross-field check names the missing field.
    for k in ("channels_sha256", "series_id"):
        _refused(_mutated(rep, lambda w, k=k: w.pop(k)), "missing_field", "bad_value", field=k)


def test_an_inconsistent_channel_record_is_refused(mv_report):
    rep = mv_report["s1"].report
    h.assert_valid(rep)

    def wrong_sha(w):
        w["channels_sha256"] = "0" * 64

    def empty_scored(w):
        w["channels"] = sorted([*w["channels"][1:], w["empty_channels"][0]])
        w["channels_sha256"] = mv.channels_sha256(w["channels"])

    def too_many(w):
        w["n_channels"] = 34

    def unsorted(w):
        w["empty_channels"] = list(reversed(w["empty_channels"]))

    for fn in (wrong_sha, empty_scored, too_many, unsorted):
        _refused(_mutated(rep, fn), "bad_value")
