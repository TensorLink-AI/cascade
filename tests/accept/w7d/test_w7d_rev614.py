"""W7d (plan rev6.14 (a)): the replay re-derives the judging start from the
signed `boundary_ts` and the T values: a set frozen before `j = t_max +
3600 * eval_delay_hours` is a MISMATCH (`frozen_before_eval_delay`), a judged
cohort without `boundary_ts` is a MISMATCH at `boundary_ts`, and the params
echo carries `eval_delay_hours`. Every forgery is re-signed (`doctored`)."""

from __future__ import annotations

import pytest

from .helpers import build_all, describe, doctored, has_path, replay, replay_module

HOUR = 3600


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-rev614"))


def _lie(c, n, mutate, paths):
    mod = replay_module()
    reports, receipts = doctored(c.pub, n, mutate)
    res = replay(c.pub, reports=reports, receipts=receipts)
    for r in res[:n - 1]:
        assert r.status == mod.MATCH, describe(res)
    r = res[n - 1]
    assert r.status == mod.MISMATCH, describe(res)
    assert any(has_path(r, p) for p in paths), f"no mismatch at {paths}; got {list(r.mismatches)}"


def test_rev614_the_honest_chain_signs_boundary_ts_and_matches(chains):
    mod = replay_module()
    c = chains["gated"]
    rep = c.pub.report(2)
    assert "eval_delay_hours" in rep["params"], "rev6.14: params echo eval_delay_hours"
    assert rep["params"]["eval_delay_hours"] == float(c.cfg.eval_delay_hours)
    assert isinstance(rep.get("boundary_ts"), int)
    assert rep["boundary_ts"] >= rep["cohorts"][0]["t_max"] + HOUR * c.cfg.eval_delay_hours
    res = replay(c.pub)
    assert all(r.status == mod.MATCH for r in res), describe(res)


def test_rev614_a_set_frozen_before_the_judging_start_is_a_mismatch(chains):
    def m(rep):
        assert rep["cohorts"], "fixture: b2 judges X"
        assert "eval_delay_hours" in rep["params"], "rev6.14: params echo eval_delay_hours"
        c = rep["cohorts"][0]
        rep["boundary_ts"] = int(c["t_max"] + HOUR * rep["params"]["eval_delay_hours"]) - 1

    _lie(chains["gated"], 2, m, ["cohorts[0].frozen_before_eval_delay"])


def test_rev614_a_judged_cohort_without_boundary_ts_is_a_mismatch(chains):
    def m(rep):
        assert rep["cohorts"] and "boundary_ts" in rep
        del rep["boundary_ts"]

    _lie(chains["gated"], 2, m, ["boundary_ts"])


def test_rev614_a_forged_eval_delay_echo_is_a_mismatch(chains):
    def m(rep):
        assert "eval_delay_hours" in rep["params"], "rev6.14: params echo eval_delay_hours"
        rep["params"]["eval_delay_hours"] = 2.0

    _lie(chains["gated"], 2, m, ["params.eval_delay_hours"])
