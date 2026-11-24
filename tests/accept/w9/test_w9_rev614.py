"""W9 (plan rev6.14 (a)): the audit re-derives the judging start from the
signed `boundary_ts` (a cohort frozen before `t_max + 3600 * eval_delay_hours`
FAILs, `frozen_before_eval_delay`), refuses a judged cohort without it, and
the eligibility boundary is `b = T + forward_margin_hours` (default 0): a
target whose data time and first-seen time are both just after T is fresh
and eligible."""

from __future__ import annotations

import copy

import pytest

from . import helpers as h

HOUR = 3600


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-rev614"))


def test_rev614_the_honest_crown_signs_boundary_ts_and_passes(run):
    r = run["s1"].report
    assert isinstance(r.get("boundary_ts"), int)
    assert r["params"]["eval_delay_hours"] == float(run["cfg"].eval_delay_hours)
    h.assert_pass(r, run["cfg"], "honest b1 with its boundary_ts")


def test_rev614_a_cohort_frozen_before_the_judging_start_fails(run):
    r = copy.deepcopy(run["s1"].report)
    assert "eval_delay_hours" in r["params"], "rev6.14: params echo eval_delay_hours"
    c = r["cohorts"][0]
    r["boundary_ts"] = int(c["t_max"] + HOUR * r["params"]["eval_delay_hours"]) - 1
    res = h.audit(h.resign(r), run["cfg"])
    assert res.status == "FAIL" and "frozen_before_eval_delay" in str(res.detail), res.detail


def test_rev614_a_judged_cohort_without_boundary_ts_fails(run):
    r = copy.deepcopy(run["s1"].report)
    assert "boundary_ts" in r, "rev6.14: the report signs boundary_ts"
    del r["boundary_ts"]
    res = h.audit(h.resign(r), run["cfg"])
    assert res.status == "FAIL", res.detail

