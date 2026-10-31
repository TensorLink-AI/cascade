"""FEV scope exclusions are not partial coverage (miner report 2026-10-30):
a correct 100-task FEV sweep — 65 univariate scored, 35 multivariate
reported as ``stage == "scope"`` — read as partial and held every v2 gate."""

from __future__ import annotations

from cascade.eval.benchmarks import extract_bench_scores, suite_incomplete

SCOPE = [{"full": f"mv{i}", "stage": "scope", "reason": "multivariate target"} for i in range(35)]


def _fev(**kw):
    s = {"suite": "fev", "status": "ok", "n_series": 65, "n_expected": 100,
         "skipped": list(SCOPE), "partial": True,            # an old sidecar's own flag
         "metrics": {"crps": 0.63, "mase": 0.80}}
    s.update(kw)
    return s


def test_fev_scope_exclusions_are_not_incomplete():
    assert not suite_incomplete(_fev())


def test_real_gaps_stay_incomplete():
    assert suite_incomplete(_fev(skipped=SCOPE + [{"full": "x", "stage": "score"}], n_series=64))
    assert suite_incomplete(_fev(n_series=60))                       # rows short of scope
    assert suite_incomplete(_fev(max_series=8))
    # "scope" is FEV's declaration only
    assert suite_incomplete({"suite": "boom", "status": "ok", "n_series": 65,
                             "n_expected": 100, "skipped": list(SCOPE)})
    assert not suite_incomplete({"suite": "boom", "status": "ok", "n_series": 3,
                                 "n_expected": 3, "skipped": []})


def test_extract_bench_scores_accepts_a_scoped_fev_suite():
    suites = [{"suite": s, "status": "ok", "n_series": 5, "n_expected": 5, "skipped": [],
               "metrics": {"crps": 0.5, "mase": 0.9}} for s in ("gift-eval", "boom", "time")]
    got = extract_bench_scores({"suites": suites + [_fev()]}, require_fev=True)
    assert got is not None and got["fev_crps"] == 0.63
