"""A config that fails to load or score is RECORDED, never silently dropped.

2026-09-18: every cascade GIFT-Eval report for two months covered 74 of the 97
official configs (all of electricity, bitbrains, kdd_cup_2018 and four daily
sets) and nothing in the report said so — the per-config ``except: continue``
left no trace, and the headline read as leaderboard-comparable when it was
not. The sweep now carries ``skipped`` (config, stage, reason) and
``n_expected`` next to ``n_series`` so a consumer can tell a partial number
from the whole benchmark.
"""

from __future__ import annotations

from cascade_benchmark.results import SuiteResult
from cascade_benchmark.suites import gifteval
from cascade_benchmark.suites._common import describe_exception


class _DS:
    def __init__(self, name: str, term: str) -> None:
        self.name, self.term = name, term


def _fake_load(name: str, term: str, **_):
    if name.startswith("electricity"):  # ds names carry the freq: "electricity/W"
        return None, "FileNotFoundError: electricity/W/data.arrow"
    return _DS(name, term), ""


def _fake_score(ds, ckpt, **_):
    if ds.name == "m4_daily":
        raise MemoryError("Unable to allocate 4.2 GiB")
    return {"MASE": 1.0, "MAE": 2.0, "CRPS": 0.5}


def test_skipped_configs_are_recorded_with_stage_and_reason(monkeypatch):
    monkeypatch.setenv("CASCADE_BENCH_GIFTEVAL_DATASETS", "m4_yearly m4_daily electricity/W")
    monkeypatch.setattr(gifteval, "build_dataset_or_reason", _fake_load)
    monkeypatch.setattr(gifteval, "score_dataset", _fake_score)
    res = gifteval.run("/nonexistent/ckpt")
    assert res.status == "ok"
    assert res.n_expected == 3
    assert res.n_series == 1
    assert res.partial is True
    assert [r["full"] for r in res.rows] == ["m4_yearly/A/short"]
    assert res.skipped == [
        {"full": "m4_daily/D/short", "stage": "score",
         "reason": "MemoryError: Unable to allocate 4.2 GiB"},
        {"full": "electricity/W/short", "stage": "load",
         "reason": "FileNotFoundError: electricity/W/data.arrow"},
    ]
    assert res.detail.startswith("partial: 2 of 3 configs skipped: ")
    assert "electricity/W/short" in res.detail


def test_full_sweep_has_no_detail_and_is_not_partial(monkeypatch):
    monkeypatch.setenv("CASCADE_BENCH_GIFTEVAL_DATASETS", "m4_yearly m4_weekly")
    monkeypatch.setattr(gifteval, "build_dataset_or_reason", lambda n, t, **_: (_DS(n, t), ""))
    monkeypatch.setattr(gifteval, "score_dataset",
                        lambda ds, c, **_: {"MASE": 1.0, "MAE": 2.0, "CRPS": 0.5})
    res = gifteval.run("/nonexistent/ckpt")
    assert res.status == "ok" and res.detail == ""
    assert res.n_expected == res.n_series == 2
    assert res.partial is False and res.skipped == []


def test_nothing_scored_keeps_the_skip_list(monkeypatch):
    monkeypatch.setenv("CASCADE_BENCH_GIFTEVAL_DATASETS", "m4_yearly")
    monkeypatch.setattr(gifteval, "build_dataset_or_reason",
                        lambda n, t, **_: (None, "ValueError: unknown dataset"))
    res = gifteval.run("/nonexistent/ckpt")
    assert res.status == "error" and res.detail == "no datasets scored"
    assert res.n_expected == 1 and res.skipped[0]["stage"] == "load"


def test_report_json_carries_skip_accounting():
    r = SuiteResult(suite="gift-eval", status="ok", n_series=74, n_expected=97,
                    skipped=[{"full": "electricity/W/short", "stage": "load", "reason": "x"}])
    assert r.partial is True
    d = __import__("dataclasses").asdict(r)
    assert d["n_expected"] == 97 and d["skipped"][0]["full"] == "electricity/W/short"
    # Older reports (no accounting) are not flagged partial — nothing to compare.
    assert SuiteResult(suite="gift-eval", status="ok", n_series=74).partial is False


def test_describe_exception_is_one_line_and_bounded():
    e = ValueError("first line\nsecond line")
    assert describe_exception(e) == "ValueError: first line"
    assert len(describe_exception(RuntimeError("x" * 1000))) == 300


# ── TIME: same class of silent drop, same accounting ─────────────────────────

import sys as _sys
import types as _types

from cascade_benchmark.suites import time_bench


def test_time_tasks_that_fail_to_score_are_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("CASCADE_BENCH_TIME_DATASET", str(tmp_path))
    # stub the timebench package the suite imports lazily
    data = _types.ModuleType("timebench.evaluation.data")
    data.load_dataset_config = lambda _p: {"datasets": {}}
    for name, mod in (("timebench", _types.ModuleType("timebench")),
                      ("timebench.evaluation", _types.ModuleType("timebench.evaluation")),
                      ("timebench.evaluation.data", data)):
        monkeypatch.setitem(_sys.modules, name, mod)
    monkeypatch.setattr(time_bench, "_load_wrapper", lambda *_a, **_k: object())
    monkeypatch.setattr(time_bench, "_tasks", lambda *_a, **_k: [("a/H", "short"), ("b/D", "short"), ("c/D", "long")])

    def fake_score(wrapper, name, term, *_a, **_k):
        if name == "b/D":
            raise ValueError("Forecast contains NaN values")
        return {"CRPS": 0.5, "MASE": 1.0}, {"CRPS": 1.0, "MASE": 2.0}

    monkeypatch.setattr(time_bench, "_score_one", fake_score)
    res = time_bench.run("/nonexistent/ckpt")
    assert res.status == "ok"
    assert res.n_expected == 3 and res.n_series == 2 and res.partial is True
    assert res.skipped == [{"full": "b/D/short", "stage": "score",
                            "reason": "ValueError: Forecast contains NaN values"}]
    assert res.detail.startswith("partial: 1 of 3 tasks skipped: b/D/short")
