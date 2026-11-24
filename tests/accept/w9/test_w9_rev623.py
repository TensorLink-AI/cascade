"""W9, plan rev6.23 (owner 2026-11-24): the audit re-derives

- the crown bar: ``crown_bar.bar`` is the config's bar at the stated
  ``crown_bar.tenure`` (``d_margin_start`` falling linearly to
  ``d_margin`` over ``d_margin_warmup_settlements``), and each verdict
  clears against it; in chain mode the tenure continues from the previous
  signed report (1 after a crown, else + 1);
- the paid court: lane D's slots are worth ``[lane_d] court_weights``
  (0.80 / 0.15 / 0.05), normalised over the court's slots, in the audit's
  own lane D part (``lane_d_pay`` / ``derive_weights``).

On main the report carries no ``crown_bar`` and the court decays by the
retired ``king_decay`` (57 / 29 / 14).
"""

from __future__ import annotations

import copy
import math

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-rev623"))


def _reports(run) -> dict:
    return {n: copy.deepcopy(run[n].report) for n in ("s1", "s2", "s3")}


def test_the_signed_reports_carry_tenure_and_bar(run):
    cfg = run["cfg"]
    r = _reports(run)
    assert r["s1"]["crown_bar"] == {"tenure": 0, "bar": cfg.d_margin_start}
    # X is crowned at b1: b2 is its first settlement after the crown.
    assert r["s2"]["crown_bar"]["tenure"] == 1
    assert r["s3"]["crown_bar"]["tenure"] == 2
    for name in ("s1", "s2", "s3"):
        p = r[name]["params"]
        assert p["d_margin_start"] == cfg.d_margin_start
        assert p["d_margin_warmup_settlements"] == cfg.d_margin_warmup_settlements
        assert p["court_weights"] == [float(w) for w in cfg.court_weights]
        assert "king_decay" not in p


def test_the_honest_chain_passes_with_its_bars(run):
    r = _reports(run)
    h.assert_pass(r["s1"], run["cfg"], "b1 (genesis)", prior=[])
    h.assert_pass(r["s2"], run["cfg"], "b2 after b1", prior=[r["s1"]])
    h.assert_pass(r["s3"], run["cfg"], "b3 after b1, b2", prior=[r["s1"], r["s2"]])


def test_a_stated_bar_off_the_config_fails(run):
    r = _reports(run)
    bad = copy.deepcopy(r["s1"])
    bad["crown_bar"]["bar"] = float(run["cfg"].d_margin)        # the floor at tenure 0
    h.assert_fail(h.resign(bad), run["cfg"], "a bar below d_margin_start at tenure 0")
    res = h.audit(h.resign(bad), run["cfg"])
    assert "ld_params" in res.detail and "crown_bar" in res.detail, res.detail


def test_a_missing_crown_bar_fails(run):
    bad = copy.deepcopy(_reports(run)["s1"])
    del bad["crown_bar"]
    h.assert_fail(h.resign(bad), run["cfg"], "no crown_bar")


def test_a_verdict_cleared_below_the_bar_fails(run):
    """X's first clear restated under a bar above its LCB (tenure and bar
    consistent): the re-derived verdict does not clear, so nothing may crown."""
    cfg = run["cfg"]
    r = _reports(run)
    lcb = r["s1"]["verdicts"][0]["lcb"]
    from dataclasses import replace

    strict = replace(cfg, d_margin_start=min(0.999, (lcb + 1.0) / 2))
    assert strict.crown_bar(0) > lcb
    bad = copy.deepcopy(r["s1"])
    bad["params"]["d_margin_start"] = strict.d_margin_start
    bad["crown_bar"]["bar"] = strict.crown_bar(0)
    h.assert_fail(h.resign(bad), strict, "a crown whose LCB is below the bar in force")


def test_the_tenure_must_continue_the_chain(run):
    """b2 restated at tenure 0 with tenure 0's bar: self-consistent alone,
    refuted by b1 (a crown at b1 makes b2's tenure 1)."""
    cfg = run["cfg"]
    r = _reports(run)
    bad = copy.deepcopy(r["s2"])
    bad["crown_bar"] = {"tenure": 0, "bar": cfg.crown_bar(0)}
    bad = h.resign(bad)
    h.assert_pass(bad, cfg, "the doctored b2 alone (tenure is a chain rule)")
    h.assert_fail(bad, cfg, "b2 at tenure 0 after a crown at b1", prior=[r["s1"]])
    bad3 = copy.deepcopy(r["s3"])
    bad3["crown_bar"] = {"tenure": 5, "bar": cfg.crown_bar(5)}
    h.assert_fail(h.resign(bad3), cfg, "b3 skipping tenure", prior=[r["s1"], r["s2"]])


def test_the_audit_court_part_pays_court_weights():
    """The audit's own lane D part (never the producer's emission)."""
    from cascade.audit.checks import _lane_d_part

    s, burn = 0.25, 0
    cw = (0.80, 0.15, 0.05)
    full = _lane_d_part(8, share=s, court=[6, 5, 7], court_weights=cw, burn_uid=burn)
    assert [full[6], full[5], full[7]] == pytest.approx([s * 0.80, s * 0.15, s * 0.05])
    two = _lane_d_part(8, share=s, court=[6, 5], court_weights=cw, burn_uid=burn)
    assert math.isclose(two[6], s * 0.80 / 0.95) and math.isclose(two[5], s * 0.15 / 0.95)
    house = _lane_d_part(8, share=s, court=[None, 5, 7], court_weights=cw, burn_uid=burn)
    assert math.isclose(house[burn], s * 0.80) and math.isclose(house[5], s * 0.15)
    dead = _lane_d_part(8, share=s, court=[6, None, 7], court_weights=cw, burn_uid=burn)
    assert math.isclose(dead[burn], s * 0.15) and math.isclose(dead[7], s * 0.05)
    assert math.isclose(sum(full), s) and math.isclose(sum(house), s)
