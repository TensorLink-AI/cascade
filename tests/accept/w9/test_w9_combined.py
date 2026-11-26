"""W9 (plan rev6.18 A3 / A8, F4): `check_lane_d` re-derives the combined
rule from the signed rows: the live rows of each cohort plus the public rows
(`public.rows`, bound by `public.rows_sha256`). Burn class (rev6.18 B3
table): a public-row digest mismatch is `ld_rows_sha256`; a verdict whose
LCB / gain / clear / per-half, per-source or per-domain gains do not
re-derive is `ld_cohort`. Every forgery is re-signed by the validator's own
key (`helpers.resign`), so the defect it introduces is the only one.
Contract: README.md "Combined rule (rev6.18 F4)".
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h


def _pub():
    from tests.accept.w7a import pub

    assert hasattr(pub, "index_only"), (
        "F4 (rev6.18 A3): tests/accept/w7a/pub.py has no index-only public fixture")
    return pub


def _findings(report, cfg):
    from cascade.audit.lane_d import lane_d_findings

    return lane_d_findings(copy.deepcopy(report), cfg, signer_hotkey=h.signer().ss58_address)


def _reseal_public(report: dict) -> dict:
    from cascade.validator.lane_d_report import rows_sha256

    report["public"]["rows_sha256"] = rows_sha256(report["public"]["rows"])
    return report


_RUNS: dict = {}


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("w9-f4")


@pytest.fixture
def runs(runs_dir):
    """Built lazily inside each test body (a missing F4 fixture is then an
    assertion in the test, never a fixture error)."""

    class _Lazy:
        def __getitem__(self, key):
            if not _RUNS:
                pub = _pub()
                chal = pub.settle_pub(runs_dir / "probe", mirrors=False)["chain"]["digests"][0]
                _RUNS.update({
                    "crowned": pub.settle_pub(runs_dir / "c", mirrors=False,
                                              quality={chal: 0.9}),
                    "lost": pub.settle_pub(runs_dir / "l", mirrors=False, quality={chal: 1.1}),
                    "chal": chal})
            return _RUNS[key]

    return _Lazy()


def test_honest_combined_reports_pass(runs):
    for name in ("crowned", "lost"):
        run = runs[name]
        f = _findings(run["s1"].report, run["cfg"])
        assert f.findings == [] and not f.transient, (name, f.findings)
    assert runs["crowned"]["s1"].report["crowned"] == runs["chal"]
    assert runs["lost"]["s1"].report["crowned"] is None


def test_a_public_row_unbound_by_its_digest_burns_as_ld_rows_sha256(runs):
    rep = copy.deepcopy(runs["crowned"]["s1"].report)
    rep["public"]["rows"][0]["crps"] *= 0.5
    f = _findings(h.resign(rep), runs["crowned"]["cfg"])
    assert f.burn is not None and f.burn[0] == "ld_rows_sha256", f.findings


def test_doctored_public_rows_that_flip_the_verdict_burn_as_ld_cohort(runs):
    """The lost challenger's public rows rewritten to look better, digest
    recomputed: the stated verdict no longer re-derives from the rows."""
    rep = copy.deepcopy(runs["lost"]["s1"].report)
    for r in rep["public"]["rows"]:
        if r["digest"] == runs["chal"]:
            r["crps"] *= 0.7
            r["mase"] *= 0.7
    f = _findings(h.resign(_reseal_public(rep)), runs["lost"]["cfg"])
    assert f.burn is not None and f.burn[0] == "ld_cohort", f.findings


@pytest.mark.parametrize("field", ["halves", "sources", "domains"])
def test_doctored_point_gains_burn_as_ld_cohort(runs, field):
    rep = copy.deepcopy(runs["crowned"]["s1"].report)
    v = rep["verdicts"][0]
    k = next(iter(v[field]))
    v[field][k]["gain"] = float(v[field][k]["gain"]) + 0.05
    f = _findings(h.resign(rep), runs["crowned"]["cfg"])
    assert f.burn is not None and f.burn[0] == "ld_cohort", f.findings


def test_a_crown_with_a_regressing_public_half_burns(runs):
    """The lost report doctored into a crown (verdict cleared, crowned set,
    the gate untouched): the clear does not re-derive (public gain < 0)."""
    rep = copy.deepcopy(runs["lost"]["s1"].report)
    v = rep["verdicts"][0]
    assert v["halves"]["public"]["gain"] < 0 < v["halves"]["live"]["gain"]
    v["cleared"] = True
    v["outcome"] = "crowned"
    f = _findings(h.resign(rep), runs["lost"]["cfg"])
    assert f.burn is not None and f.burn[0] in ("ld_cohort", "ld_crown"), f.findings


def test_a_report_without_its_public_half_burns(runs):
    rep = copy.deepcopy(runs["crowned"]["s1"].report)
    del rep["public"]
    f = _findings(h.resign(rep), runs["crowned"]["cfg"])
    assert f.burn is not None and f.burn[0] == "ld_cohort", f.findings


def test_public_rows_missing_a_model_burn_as_ld_rows_count(runs):
    rep = copy.deepcopy(runs["crowned"]["s1"].report)
    rep["public"]["rows"] = [r for r in rep["public"]["rows"] if r["digest"] != runs["chal"]]
    f = _findings(h.resign(_reseal_public(rep)), runs["crowned"]["cfg"])
    assert f.burn is not None and f.burn[0] in ("ld_rows_count", "ld_cohort"), f.findings


def test_the_backtest_combination_is_gone_from_the_audit():
    from cascade.audit import lane_d, lane_d_chain

    assert not hasattr(lane_d, "combined_log_ratio")
    assert not hasattr(lane_d_chain, "combined_log_ratio")
