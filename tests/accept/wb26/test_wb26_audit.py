"""WB26: ``check_lane_b`` re-derives every suite-target verdict row from its
recorded fields (plan WB26 row item 6): the either-metric gain test at
``alpha/(2k)``, the non-inferiority of the other metric, ``target_lcb`` = the
smaller cleared LCB, guard 4 and the anchor guard. Honest rows come from the
real validator (``judgements_record``); each tampered copy must be caught."""

from __future__ import annotations

import copy

import pytest

from tests.accept.w_s2 import helpers as ws2

from . import helpers as h

GIFT = "suite:gift"


def _rows(tmp_path, cand_factors, **kw):
    cfg = h.lane_b_cfg()
    em = ws2.expert_manifest(h.entry(GIFT, h.bench_rows(cand_factors),
                                     anchor=h.bench_rows({})))
    return cfg, h.record(h.judge_manifest(em, cfg, tmp_path, **kw))


def _why(row, cfg):
    from cascade.audit.checks import _lane_b_row_verdict
    from cascade.shared.lane_b import lane_b_slots

    return _lane_b_row_verdict(row, cfg, lane_b_slots(cfg))


def _honest(tmp_path):
    cases = {
        "crps": {("gift", "crps"): 0.95},
        "mase": {("gift", "mase"): 0.95},
        "both": {("gift", "crps"): 0.95, ("gift", "mase"): 0.90},
        "noninf_fail": {("gift", "crps"): 0.95, ("gift", "mase"): 1.03},
        "none": {},
    }
    out = {}
    for name, f in cases.items():
        cfg, (row,) = _rows(tmp_path, f)
        assert "suite_gain" in row, f"{name}: the suite row records suite_gain: {sorted(row)}"
        out[name] = row
    return cfg, out


def test_honest_suite_rows_rederive(tmp_path):
    cfg, rows = _honest(tmp_path)
    for name, row in rows.items():
        assert _why(row, cfg) is None, (name, _why(row, cfg))
    assert rows["crps"]["passed"] and rows["mase"]["passed"] and rows["both"]["passed"]
    assert not rows["noninf_fail"]["passed"] and not rows["none"]["passed"]


def _tamper(row, path, value):
    r = copy.deepcopy(row)
    cur = r
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return r


@pytest.mark.parametrize("case,path,value", [
    ("crps", ("passed",), False),
    ("noninf_fail", ("passed",), True),
    ("none", ("passed",), True),
    ("crps", ("suite_gain", "cleared_metric"), "both"),
    ("mase", ("suite_gain", "noninf_metric"), "mase"),
    ("crps", ("suite_gain", "gain_alpha"), 0.05),
    ("crps", ("suite_gain", "n_items"), 3),
    ("noninf_fail", ("suite_gain", "noninf_lcb"), 0.0),
    ("crps", ("suite_guards", "pool_gain"), -0.01),
    ("crps", ("suite_guards", "anchor_pointer"), ""),
    ("crps", ("suite_guards", "anchor_guard_lcbs", "web"), -0.001),
    ("crps", ("suite_guards", "anchor_public_lcbs", "boom_crps"), -0.001),
    ("crps", ("public_lcbs", "gift_crps"), 0.0),
])
def test_tampered_suite_rows_are_caught(tmp_path, case, path, value):
    cfg, rows = _honest(tmp_path)
    bad = _tamper(rows[case], path, value)
    assert _why(bad, cfg) is not None, f"audit accepted a tampered {case} row: {path} = {value!r}"


def test_full_pay_lcb_must_be_the_smaller_cleared_lcb(tmp_path):
    cfg, rows = _honest(tmp_path)
    row = rows["both"]
    g = row["suite_gain"]
    assert row["target_lcb"] == min(g["crps_lcb"], g["mase_lcb"])
    bad = _tamper(row, ("target_lcb",), max(g["crps_lcb"], g["mase_lcb"]))
    assert _why(bad, cfg) is not None, "audit accepted target_lcb = the LARGER cleared LCB"


def test_suite_row_without_suite_guards_is_refused(tmp_path):
    cfg, rows = _honest(tmp_path)
    bad = copy.deepcopy(rows["crps"])
    bad.pop("suite_guards")
    assert _why(bad, cfg) is not None
