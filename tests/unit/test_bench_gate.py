"""DEC-CA-0054: the three-suite public-benchmark no-regression gate."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from cascade.eval.bench_gate import evaluate_bench_gate, uncomputable_bench_gate
from cascade.eval.koth import RoundResult, apply_bench_gate
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    bench_rows_key,
    dump_bench_rows,
    load_bench_report,
    load_bench_rows,
)
from cascade.shared.manifest import BenchScores

SIX = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase", "time_crps", "time_mase")


def _rows(scale: dict[str, float] | None = None, n: int = 40, seed: int = 0) -> dict:
    """Item rows for the three suites; ``scale[suite]`` multiplies every value."""
    rng = np.random.default_rng(seed)
    out = {}
    for suite in ("gift", "boom", "time"):
        k = (scale or {}).get(suite, 1.0)
        base = rng.uniform(0.5, 1.5, size=(n, 2))
        out[suite] = {"status": "ok", "rows": [
            {"item": f"{suite}/{i}", "crps": float(c * k), "mase": float(m * k)}
            for i, (c, m) in enumerate(base)]}
    return out


def _six(v: float = 1.0) -> dict:
    return dict.fromkeys(SIX, v)


def _gate(king_rows, chal_rows, king_six, chal_six, **kw):
    return evaluate_bench_gate(king_rows, chal_rows, king_six, chal_six,
                               alpha=0.05, B=500, seed="blockhash", **kw)


def test_better_everywhere_passes():
    g = _gate(_rows(), _rows({"gift": 0.95, "boom": 0.95, "time": 0.95}), _six(1.0), _six(0.95))
    assert g.computed and g.passed, g.reason
    assert set(g.suite_ucb) == {"gift", "boom", "time"}
    assert all(v > 0 for v in g.suite_ucb.values())


def test_one_suite_significantly_worse_blocks():
    g = _gate(_rows(), _rows({"gift": 0.9, "boom": 1.10, "time": 0.9}), _six(1.0), _six(0.9))
    assert g.computed and g.passed is False
    assert "boom" in g.reason and g.suite_ucb["boom"] < 0


def test_noise_level_tie_not_significant_but_mean_must_improve():
    # identical rows: no suite significantly worse, but the six-number geomean
    # is not strictly better → blocked (no threshold).
    g = _gate(_rows(), _rows(), _six(1.0), _six(1.0))
    assert g.computed and g.passed is False
    assert "not better" in g.reason
    g2 = _gate(_rows(), _rows(), _six(1.0), _six(0.999))
    assert g2.passed is True


def test_missing_suite_is_uncomputable():
    chal = _rows()
    chal["time"]["status"] = "error"
    g = _gate(_rows(), chal, _six(1.0), _six(0.9))
    assert not g.computed and g.passed is None and "time" in g.reason


def test_too_few_items_is_uncomputable():
    g = _gate(_rows(n=5), _rows(n=5), _six(1.0), _six(0.9), min_items=10)
    assert not g.computed and "paired items" in g.reason


def test_missing_six_numbers_is_uncomputable():
    bad = _six(1.0)
    bad["time_mase"] = math.nan
    assert not _gate(_rows(), _rows(), _six(1.0), bad).computed


def test_deterministic_for_one_seed():
    a = _gate(_rows(), _rows({"boom": 1.02}, seed=0), _six(1.0), _six(0.99))
    b = _gate(_rows(), _rows({"boom": 1.02}, seed=0), _six(1.0), _six(0.99))
    assert a.suite_ucb == b.suite_ucb


def _win() -> RoundResult:
    return RoundResult(lcb=0.05, margin=0.01, challenger_wins_round=True, n_windows=100,
                       king_geomean=1.0, chal_geomean=0.9, inconclusive=False)


@pytest.mark.parametrize("mode", ["enforce", "shadow"])
def test_apply_truth_table(mode):
    ok = evaluate_bench_gate(_rows(), _rows({"gift": 0.9, "boom": 0.9, "time": 0.9}),
                             _six(1.0), _six(0.9), alpha=0.05, B=200, seed=1)
    fail = evaluate_bench_gate(_rows(), _rows({"boom": 1.2}), _six(1.0), _six(0.9),
                               alpha=0.05, B=200, seed=1)
    unc = uncomputable_bench_gate("rows missing")
    assert apply_bench_gate(_win(), ok, mode=mode).challenger_wins_round
    r = apply_bench_gate(_win(), fail, mode=mode)
    assert r.bench_gate_passed is False
    assert r.challenger_wins_round is (mode == "shadow")
    r = apply_bench_gate(_win(), unc, mode=mode)
    assert r.bench_gate_passed is None
    assert r.inconclusive is (mode == "enforce")


def test_loss_is_never_touched():
    loss = RoundResult(lcb=-0.1, margin=0.01, challenger_wins_round=False, n_windows=100,
                       king_geomean=1.0, chal_geomean=1.1, inconclusive=False)
    fail = uncomputable_bench_gate("x")
    assert apply_bench_gate(loss, fail, mode="enforce").inconclusive is False


def _entry(**kw) -> BenchEntry:
    return BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="hk", miner_uid=3,
                      trained_pointer="metro-v1:trained:hippius:r@sha256:" + "0" * 64,
                      scores=BenchScores(**_six(0.9)), **kw)


def test_report_without_rows_serialises_as_before():
    body = BenchReport(round_id="1", created_block=9, entries=(_entry(),)).canonical_body()
    assert b"rows_key" not in body and b"rows_sha256" not in body


def test_report_rows_pointer_round_trips_and_is_signed():
    rows = dump_bench_rows(_rows())
    sha = hashlib.sha256(rows.encode()).hexdigest()
    e = _entry(rows_key=bench_rows_key("p"), rows_sha256=sha)
    rep = BenchReport(round_id="1", created_block=9, entries=(e,))
    assert b"rows_sha256" in rep.canonical_body()
    from cascade.shared.bench_report import dump_bench_report

    back = load_bench_report(dump_bench_report(rep)).entries[0]
    assert (back.rows_key, back.rows_sha256) == (e.rows_key, sha)
    assert load_bench_rows(rows, sha)["gift"]["status"] == "ok"
    with pytest.raises(ValueError):
        load_bench_rows(rows + " ", sha)
