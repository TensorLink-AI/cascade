"""Sidecar per-row metric floor (operator decision 2026-11-02).

The v2 public bench gate treats a row metric that is not strictly > 0 as
unusable, so a challenger's PERFECT forecast of a constant series (an exact
0 after the float32 de-normalisation) read as an omitted task and failed the
gate ("challenger failed required task"), while a king off by 1e-9 stayed
usable. The sidecar now publishes any finite row CRPS / MASE in
[0, ROW_METRIC_FLOOR) as the floor: a perfect task becomes an exact tie with a
sub-floor king, NaN / inf / missing still fail. No gate code changes — every
validator replays the main validator's signed rows — so this is not a
consensus change.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from cascade.eval.bench_gate import RULE_V2, SUITES_V2, evaluate_bench_gate
from cascade.eval.benchmarks import item_rows_from_report

_RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "cascade_benchmark" / "results.py"
_spec = importlib.util.spec_from_file_location("_sidecar_results", _RESULTS)
R = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(R)

SUITE_NAMES = {"gift": "gift-eval", "boom": "boom", "time": "time", "fev": "fev"}
EIGHT = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase",
         "time_crps", "time_mase", "fev_crps", "fev_mase")


# ── the floor itself ─────────────────────────────────────────────────────────


def test_floor_raises_exact_zero_and_sub_floor_values():
    out = R.floor_row_metrics({"full": "ds/T/short", "CRPS": 0.0, "MASE": 1.2e-9, "MAE": 0.0})
    assert out["CRPS"] == R.ROW_METRIC_FLOOR
    assert out["MASE"] == R.ROW_METRIC_FLOOR
    assert out["MAE"] == 0.0  # not a gate metric: untouched


def test_floor_leaves_normal_and_broken_values_alone():
    for v in (0.5, R.ROW_METRIC_FLOOR, 3.0):
        assert R.floor_row_metrics({"CRPS": v})["CRPS"] == v
    nan = R.floor_row_metrics({"CRPS": float("nan")})["CRPS"]
    assert math.isnan(nan)
    assert R.floor_row_metrics({"CRPS": float("inf")})["CRPS"] == float("inf")
    assert R.floor_row_metrics({"CRPS": -0.1})["CRPS"] == -0.1
    assert R.floor_row_metrics({"CRPS": None})["CRPS"] is None
    assert R.floor_row_metrics({"CRPS": True})["CRPS"] is True
    assert "MASE" not in R.floor_row_metrics({"CRPS": 1.0})


def test_floor_does_not_mutate_the_suite_rows():
    row = {"full": "a/T/short", "CRPS": 0.0, "MASE": 0.0}
    s = R.SuiteResult(suite="boom", status="ok", rows=[row], n_series=1)
    d = R.BenchmarkReport(checkpoint="x", suites=[s]).to_dict()
    assert d["suites"][0]["rows"][0]["CRPS"] == R.ROW_METRIC_FLOOR
    assert row["CRPS"] == 0.0  # the in-memory result keeps the raw value
    assert d["row_metric_floor"] == R.ROW_METRIC_FLOOR


# ── end to end: sidecar report → validator item rows → v2 gate ──────────────


def _report(per_suite_rows: dict[str, list[dict]]) -> dict:
    suites = [R.SuiteResult(suite=SUITE_NAMES[k], status="ok", rows=rows, n_series=len(rows))
              for k, rows in per_suite_rows.items()]
    return R.BenchmarkReport(checkpoint="ckpt", suites=suites).to_dict()


def _raw_rows(seed: int, factor: float, zero_boom: int) -> dict[str, list[dict]]:
    """Per suite, 40 tasks of (CRPS, MASE). ``factor`` scales every task; the
    first ``zero_boom`` BOOM tasks are constant series: the king lands at
    ~1e-9, a challenger with ``factor < 1`` exactly on 0."""
    rng = np.random.default_rng(seed)
    out = {}
    for key in SUITE_NAMES:
        base = rng.uniform(0.5, 1.5, size=(40, 2))
        rows = []
        for i, (c, m) in enumerate(base):
            if key == "boom" and i < zero_boom:
                v = 0.0 if factor < 1 else 1.2e-9
                rows.append({"full": f"{key}/{i}/short", "CRPS": v, "MASE": v, "MAE": v})
            else:
                rows.append({"full": f"{key}/{i}/short", "CRPS": float(c * factor),
                             "MASE": float(m * factor)})
        out[key] = rows
    return out


def _gate(king_report: dict, chal_report: dict):
    scores = dict.fromkeys(EIGHT, 1.0)
    chal_scores = dict.fromkeys(EIGHT, 0.9)
    return evaluate_bench_gate(item_rows_from_report(king_report),
                               item_rows_from_report(chal_report), scores, chal_scores,
                               alpha=0.05, B=400, seed="blockhash", rule=RULE_V2)


def test_perfect_constant_series_no_longer_fails_the_gate():
    # 3 of 40 BOOM tasks constant (< 10% masked either way) so the comparison
    # isolates the required-task check, the miner's reported failure mode.
    king = _report(_raw_rows(seed=1, factor=1.0, zero_boom=3))
    chal = _report(_raw_rows(seed=1, factor=0.9, zero_boom=3))
    res = _gate(king, chal)
    assert res.computed and res.passed, res.reason
    assert res.suites["boom"]["n"] == 40  # every task counted, none masked


def test_floored_constant_tasks_pair_as_exact_ties():
    from cascade.eval.bench_gate import task_log_ratios
    from cascade.eval.public_guards import pair_public_rows

    king = item_rows_from_report(_report(_raw_rows(seed=2, factor=1.0, zero_boom=3)))
    chal = item_rows_from_report(_report(_raw_rows(seed=2, factor=0.9, zero_boom=3)))
    paired = pair_public_rows(king, chal, benchmarks=SUITES_V2, require=SUITES_V2)
    logs = dict(zip(sorted(paired.items["boom_crps"]), task_log_ratios(paired, "boom"),
                    strict=True))
    for i in range(3):
        assert logs[f"boom/{i}/short"] == 0.0  # tie, not a 1000x ratio
    assert logs["boom/3/short"] < 0  # ordinary tasks still compare normally


def test_without_the_floor_the_same_case_fails_documenting_the_bug():
    king = _report(_raw_rows(seed=1, factor=1.0, zero_boom=3))
    chal = _report(_raw_rows(seed=1, factor=0.9, zero_boom=3))
    for suite in chal["suites"]:  # undo the floor on the challenger side only
        for row in suite["rows"]:
            for k in ("CRPS", "MASE"):
                if row.get(k) == R.ROW_METRIC_FLOOR:
                    row[k] = 0.0
    res = _gate(king, chal)
    assert res.computed and res.passed is False
    assert "challenger failed required task" in res.reason


def test_nan_on_a_required_task_still_fails():
    king = _report(_raw_rows(seed=3, factor=1.0, zero_boom=0))
    chal = _report(_raw_rows(seed=3, factor=0.9, zero_boom=0))
    chal["suites"][1]["rows"][0]["CRPS"] = float("nan")  # boom task 0
    res = _gate(king, chal)
    assert res.passed is False
    assert "challenger failed required task" in res.reason


def test_omitted_required_task_still_fails():
    king = _report(_raw_rows(seed=4, factor=1.0, zero_boom=0))
    chal = _report(_raw_rows(seed=4, factor=0.9, zero_boom=0))
    del chal["suites"][1]["rows"][0]
    res = _gate(king, chal)
    assert res.passed is False
    assert "challenger failed required task" in res.reason
