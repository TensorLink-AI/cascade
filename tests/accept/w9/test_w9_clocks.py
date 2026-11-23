"""W9: both-clock forward eligibility, re-derived (plan rev6 §1.3 "(rev3) A
window is eligible only if, for every target point, both its data timestamp
and its first-seen time are after wall(max(T_E, T_K)) + forward_margin. Its
feed's measured mirror lag must also be within `max_mirror_lag_hours`";
invariant 4, "A scored window failing §1.3 for max(T_E, T_K) is an audit
failure"; README "Clocks").

Main's audit already checks both clocks against the stated `t_max` (the two
`*_at_the_bound_fails` cases pass on main and stay minimal). It takes the
mirror lag and the king's T on trust: those cases FAIL on main (an `assert`).
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-clocks"))


def _first_window(report: dict) -> dict:
    wid = report["cohorts"][0]["window_ids"][0]
    for w in report["windows"]:
        if w["window_id"] == wid:
            return w
    raise AssertionError(f"fixture: no clock record for {wid}")


def _bound(report: dict, cfg) -> int:
    return int(report["cohorts"][0]["t_max"]) + int(3600 * float(cfg.forward_margin_hours))


@pytest.mark.parametrize("which", ["s1", "s2"])
def test_honest_reports_pass(run, which):
    h.assert_pass(run[which].report, run["cfg"], f"honest {which}")


@pytest.mark.parametrize("clock", ["min_target_ts", "min_first_seen_ts"])
def test_a_window_at_the_bound_fails(run, clock):
    report, cfg = copy.deepcopy(run["s1"].report), run["cfg"]
    _first_window(report)[clock] = _bound(report, cfg)
    h.assert_fail(h.resign(report), cfg, f"{clock} equal to max(T) + margin")


def test_a_window_over_the_mirror_lag_limit_fails(run):
    report, cfg = copy.deepcopy(run["s1"].report), run["cfg"]
    _first_window(report)["mirror_lag_hours"] = float(cfg.max_mirror_lag_hours) + 0.5
    h.assert_fail(h.resign(report), cfg, "a judged window whose feed lags over the limit")


def test_a_window_at_the_mirror_lag_limit_passes(run):
    report, cfg = copy.deepcopy(run["s1"].report), run["cfg"]
    _first_window(report)["mirror_lag_hours"] = float(cfg.max_mirror_lag_hours)
    h.assert_pass(h.resign(report), cfg, "a feed exactly at max_mirror_lag_hours")


@pytest.mark.parametrize("which", ["s1", "s2"])
def test_the_kings_t_is_the_named_kings_reveal(run, which):
    """T_K is not the report's word: the cohort's king is `king_before` (the
    house H0, T = HOUSE_T), so a lower `king_reveal_ts` (which would admit
    windows published before T_K if the king were the later of the two) is
    refused even where the entry's T still dominates `t_max`."""
    report, cfg = copy.deepcopy(run[which].report), run["cfg"]
    c = report["cohorts"][0]
    assert c["king_digest"] == report["king_before"]["digest"]
    assert c["king_reveal_ts"] == report["king_before"]["reveal_ts"] == h.HOUSE_T
    c["king_reveal_ts"] = h.HOUSE_T - 86_400
    h.assert_fail(h.resign(report), cfg, f"{which}: king_reveal_ts below the king's reveal")
