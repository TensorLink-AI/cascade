"""WB26: ``check_lane_b`` re-derives guard 4 and the anchor guard of every
suite slot row from its ``suite_guards`` record (plan WB26 row). Honest rows
come from the real validator (``judgements_record``); each tampered copy
must be caught by the audit's lane B row check."""

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


def _gift(rows):
    (r,) = [r for r in rows if r["domain"] == GIFT]
    return r


def _tamper(row, path, value):
    r = copy.deepcopy(row)
    cur = r
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return r


def test_honest_rows_rederive_and_suite_guards_are_recorded(tmp_path):
    cfg, rows = _rows(tmp_path, h.GIFT_GAIN)
    for r in rows:
        assert _why(r, cfg) is None, (r["domain"], _why(r, cfg))
    g = _gift(rows)
    assert g["passed"] and "suite_guards" in g, sorted(g)


def test_honest_anchor_missing_row_rederives(tmp_path):
    cfg, rows = _rows(tmp_path, h.GIFT_GAIN, anchor_pointer="")
    g = _gift(rows)
    assert not g["passed"] and "suite_guards" in g, g["reason"]
    assert _why(g, cfg) is None, _why(g, cfg)


@pytest.mark.parametrize("path,value", [
    (("suite_guards", "pool_gain"), -0.01),
    (("suite_guards", "anchor_pointer"), ""),
    (("suite_guards", "anchor_guard_lcbs", "web"), -0.001),
    (("suite_guards", "anchor_public_lcbs", "boom_crps"), -0.001),
])
def test_tampered_suite_guards_on_a_passing_row_are_caught(tmp_path, path, value):
    cfg, rows = _rows(tmp_path, h.GIFT_GAIN)
    bad = _tamper(_gift(rows), path, value)
    assert _why(bad, cfg) is not None, f"audit accepted a passing row with {path} = {value!r}"


def test_passing_suite_row_without_suite_guards_is_refused(tmp_path):
    cfg, rows = _rows(tmp_path, h.GIFT_GAIN)
    bad = copy.deepcopy(_gift(rows))
    bad.pop("suite_guards")
    assert _why(bad, cfg) is not None, "audit accepted a claimed suite slot with no suite_guards"


def test_refused_row_flipped_to_passed_is_caught(tmp_path):
    cfg, rows = _rows(tmp_path, h.GIFT_GAIN, anchor_pointer="")
    bad = _tamper(_gift(rows), ("passed",), True)
    bad["gain_metric"] = "crps"
    assert _why(bad, cfg) is not None, "audit accepted an anchor_missing suite row as passed"


def test_suite_guards_on_a_domain_row_are_refused(tmp_path):
    cfg, rows = _rows(tmp_path, h.GIFT_GAIN)
    (dom,) = [r for r in rows if r["domain"] == "energy"]
    bad = copy.deepcopy(dom)
    bad["suite_guards"] = copy.deepcopy(_gift(rows)["suite_guards"])
    assert _why(bad, cfg) is not None, "audit accepted suite_guards on a domain slot row"
