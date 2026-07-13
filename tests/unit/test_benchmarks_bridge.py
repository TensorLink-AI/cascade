"""The benchmark bridge is best-effort and pure stdlib: every failure path must
return None (never raise), so a missing/broken sidecar can't disturb a round.

These cover the cascade-side bridge only; the sidecar itself
(``benchmarks/``) runs in its own isolated env and is not imported here.
"""

from __future__ import annotations

from pathlib import Path

from cascade.eval.benchmarks import extract_bench_scores, format_report, run_benchmarks


def _report(*suites):
    return {"checkpoint": "/x", "suites": list(suites)}


def _ok(suite, metrics):
    return {"suite": suite, "status": "ok", "metrics": metrics, "n_series": 10}


def test_extract_bench_scores_all_three_suites():
    report = _report(
        _ok("gift-eval", {"crps": 0.42, "mase": 0.81}),
        _ok("boom", {"crps": 0.55, "mase": 0.93}),
        _ok("time", {"crps": 0.38, "mase": 0.77}),
    )
    assert extract_bench_scores(report) == {
        "gifteval_crps": 0.42, "gifteval_mase": 0.81,
        "boom_crps": 0.55, "boom_mase": 0.93,
        "time_crps": 0.38, "time_mase": 0.77,
    }


def test_extract_bench_scores_is_case_insensitive_and_alias_aware():
    # TIME may surface upper-case keys, and CRPS may arrive as a WQL name.
    report = _report(
        _ok("gift-eval", {"crps": 0.4, "mase": 0.8}),
        _ok("boom", {"crps": 0.5, "mase": 0.9}),
        _ok("time", {"MASE": 0.7, "mean_weighted_sum_quantile_loss": 0.3}),
    )
    got = extract_bench_scores(report)
    assert got["time_crps"] == 0.3 and got["time_mase"] == 0.7


def test_extract_bench_scores_none_when_a_suite_is_missing_or_skipped():
    # TIME skipped ⇒ incomplete set ⇒ None (and a loud warning, not a silent drop).
    assert extract_bench_scores(_report(
        _ok("gift-eval", {"crps": 0.4, "mase": 0.8}),
        _ok("boom", {"crps": 0.5, "mase": 0.9}),
        {"suite": "time", "status": "skipped", "metrics": {}},
    )) is None
    # A suite present but lacking a recognizable crps key ⇒ None.
    assert extract_bench_scores(_report(
        _ok("gift-eval", {"crps": 0.4, "mase": 0.8}),
        _ok("boom", {"crps": 0.5, "mase": 0.9}),
        _ok("time", {"mase": 0.7, "nd": 0.2}),
    )) is None
    assert extract_bench_scores(None) is None


def test_run_benchmarks_missing_wrapper_returns_none(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    # no forecast_wrapper.py
    assert run_benchmarks(ckpt, project_dir=tmp_path) is None


def test_run_benchmarks_missing_project_returns_none(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "forecast_wrapper.py").write_text("class Wrapper: ...\n", encoding="utf-8")
    # project dir has no pyproject.toml
    assert run_benchmarks(ckpt, project_dir=tmp_path / "nope") is None


def test_run_benchmarks_no_uv_returns_none(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "forecast_wrapper.py").write_text("class Wrapper: ...\n", encoding="utf-8")
    project = tmp_path / "benchmarks"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    # Force the "uv not on PATH" branch deterministically.
    monkeypatch.setattr("cascade.eval.benchmarks.shutil.which", lambda _: None)
    assert run_benchmarks(ckpt, project_dir=project) is None


def test_run_benchmarks_nonzero_exit_returns_none(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "forecast_wrapper.py").write_text("class Wrapper: ...\n", encoding="utf-8")
    project = tmp_path / "benchmarks"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    # `false` exits non-zero and writes no JSON → bridge must return None.
    assert run_benchmarks(ckpt, project_dir=project, uv_bin="/bin/false") is None


def test_format_report_renders_ok_and_skipped():
    report = {
        "checkpoint": "/x",
        "suites": [
            {"suite": "gift-eval", "status": "ok", "metrics": {"crps": 0.42, "mase": 0.81}, "n_series": 97},
            {"suite": "time", "status": "skipped", "metrics": {}, "n_series": 0},
        ],
    }
    out = format_report(report)
    assert "gift-eval ok crps=0.4200 mase=0.8100 n=97" in out
    assert "time skipped" in out
