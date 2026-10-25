"""scripts/bench_seed_noise.py — the report-only DEC-CA-0061 rule-v2
calibration tool, on synthetic seed replicas."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cascade.shared.bench_report import BenchEntry, BenchReport, dump_bench_report, dump_bench_rows
from cascade.shared.manifest import BenchScores

_SPEC = importlib.util.spec_from_file_location(
    "bench_seed_noise", Path(__file__).resolve().parents[2] / "scripts" / "bench_seed_noise.py")
noise = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = noise
_SPEC.loader.exec_module(noise)

SUITES = ("gift", "boom", "time", "fev")


def _replica(seed: int, *, jitter: float = 0.02, n: int = 30, bias: dict | None = None) -> dict:
    """Same tasks, same base values (the shared recipe), per-task
    multiplicative seed noise of scale ``jitter``."""
    base = np.random.default_rng(0).uniform(0.5, 1.5, size=(len(SUITES), n, 2))
    rng = np.random.default_rng(seed)
    out = {}
    for si, s in enumerate(SUITES):
        f = (bias or {}).get(s, 1.0)
        eps = np.exp(rng.normal(0.0, jitter, size=(n, 2)))
        out[s] = {"status": "ok", "rows": [
            {"item": f"{s}/{i}", "crps": float(base[si, i, 0] * eps[i, 0] * f),
             "mase": float(base[si, i, 1] * eps[i, 1] * f)} for i in range(n)]}
    return out


def _ck(name, rows):
    return {"name": name, "rows": rows, "scores": noise.rows_suite_scores(rows)}


def test_rows_suite_scores_are_task_geomeans_and_skip_unusable():
    rows = {"gift": {"status": "ok", "rows": [{"item": "a", "crps": 1.0, "mase": 4.0},
                                              {"item": "b", "crps": 4.0, "mase": 0.0}]},
            "boom": {"status": "skipped", "rows": []}}
    got = noise.rows_suite_scores(rows)
    assert got["gifteval_crps"] == pytest.approx(2.0) and got["gifteval_mase"] == 4.0
    assert "boom_crps" not in got


def test_identical_replicas_tie_everywhere():
    a = _ck("a", _replica(1, jitter=0.0))
    r = noise.pair_stats(a, dict(a, name="b"), B=200)
    assert r["computed"] and r["avg_ratio"] == 1.0 and r["win_rate"] == 0.5
    assert r["fail_average"] and not r["fail_regression"] and not r["fail_win_rate"]
    assert r["strict_passed"] is False                     # ties fail every strict number
    assert all(v is False for v in r["strict"].values())


def test_seed_noise_flips_strict_comparisons_but_rarely_the_regression_check():
    cks = [_ck(f"s{i}", _replica(i)) for i in range(1, 5)]
    pairs = noise.ordered_pairs(cks, B=300)
    assert len(pairs) == 12 and all(p["computed"] for p in pairs)
    s = noise.summarize(pairs)
    # the strict rule blocks nearly every pure-noise pair; the new rule's
    # regression check (tolerance 1 %) flags none at 2 % per-task noise
    assert s["strict"]["blocked_share"] >= 0.9
    assert set(s["strict"]["flip_rate"]) == {
        f"{p}_{m}" for p in ("gifteval", "boom", "time", "fev") for m in ("crps", "mase")}
    assert s["new_rule"]["fail_regression"] == 0.0
    # the average fails one direction of every pair (by construction)
    assert s["new_rule"]["fail_average"] == pytest.approx(0.5)
    assert set(s["ratio_lcb"]) == set(SUITES)
    assert all(q["max"] <= 1.01 for q in s["ratio_lcb"].values())
    assert 0.0 <= s["min_clean_tolerance"] <= 0.01


def test_a_real_regression_is_flagged_and_sets_the_clean_tolerance():
    good, bad = _ck("good", _replica(1)), _ck("bad", _replica(2, bias={"time": 1.1}))
    s = noise.summarize(noise.ordered_pairs([good, bad], B=300))
    assert s["new_rule"]["fail_regression"] == 0.5         # bad as challenger only
    assert s["min_clean_tolerance"] > 0.05
    out = io.StringIO()
    noise.print_report(s, tolerance=0.01, min_win_rate=0.5, out=out)
    text = out.getvalue()
    assert "(a) old strict rule" in text and "(c) per-suite ratio LCB" in text


def test_uncomputable_pairs_are_listed():
    a = _ck("a", _replica(1))
    b = json.loads(json.dumps(a))
    b["rows"]["fev"]["status"] = "error"
    s = noise.summarize([noise.pair_stats(a, b, B=100)])
    assert s["computed"] == 0 and "fev" in s["uncomputable"][0]


def test_load_checkpoint_from_rows_and_signed_report(tmp_path):
    rows = _replica(3)
    text = dump_bench_rows(rows)
    (tmp_path / "rows").mkdir()
    (tmp_path / "rows" / "x.json").write_text(text)
    scores = BenchScores(*[1.0] * 6, fev_crps=0.5, fev_mase=0.7)
    rep = BenchReport(round_id="7", created_block=1, entries=(BenchEntry(
        role="challenger", size="m", miner_hotkey="hk", miner_uid=1, trained_pointer="ptr",
        scores=scores, rows_key="rows/x.json",
        rows_sha256=hashlib.sha256(text.encode()).hexdigest()),))
    (tmp_path / "report.json").write_text(dump_bench_report(rep))
    read = noise._reader(None)
    ck = noise.load_checkpoint(str(tmp_path / "report.json") + "::ptr", read)
    assert ck["rows"] == rows and ck["scores"]["fev_crps"] == 0.5
    plain = noise.load_checkpoint(str(tmp_path / "rows" / "x.json"), read)
    assert plain["rows"] == rows and "fev_mase" in plain["scores"]
    with pytest.raises(ValueError):
        noise.load_checkpoint(str(tmp_path / "report.json") + "::nope", read)
    # CLI end to end on two replicas (report-only; prints, exits 0)
    (tmp_path / "b.json").write_text(dump_bench_rows(_replica(4)))
    assert noise.main([str(tmp_path / "rows" / "x.json"), str(tmp_path / "b.json"),
                       "--B", "100"]) == 0
