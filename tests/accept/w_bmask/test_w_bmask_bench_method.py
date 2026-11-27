"""W-BMASK: the bench method is stamped on every report and every published
rows suite, and the audit reproduces a report with its OWN method (reports
swept before #972 use the legacy ``impute/0`` behaviour). Torch-free.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys

import pytest

from .helpers import SIDECAR, load_observed

PTR = "hippius://ckpt@sha256:" + "ab" * 32


def _results():
    path = SIDECAR / "results.py"
    spec = importlib.util.spec_from_file_location("w_bmask_results", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _report(method: str):
    R = _results()
    suite = R.SuiteResult("gift-eval", "ok", rows=[{"full": "m4/D/short", "CRPS": 0.4,
                                                    "MASE": 1.2}], n_series=1)
    return R.BenchmarkReport("ckpt", suites=[suite], data_revisions={"gift-eval": "rev1"},
                             bench_method=method).to_dict()


# ── sidecar: the stamp and the legacy switch ─────────────────────────────────


def test_sidecar_report_carries_its_bench_method_and_legacy_context():
    obs = load_observed()
    assert (obs.METHOD, obs.LEGACY_METHOD) == ("observed_mask/1", "impute/0")
    new, old = _report(obs.METHOD), _report(obs.LEGACY_METHOD)
    assert new["bench_method"] == "observed_mask/1"
    assert old["bench_method"] == "impute/0"
    assert new["methodology"]["gift-eval"]["context"].startswith("gluonts-observed/1")
    assert old["methodology"]["gift-eval"]["context"].startswith("impute/0")
    cli = (SIDECAR / "cli.py").read_text()
    assert '"--bench-method"' in cli and 'os.environ["CASCADE_BENCH_METHOD"]' in cli
    assert "bench_method=bench_method or current_method()" in cli


def test_legacy_method_returns_the_pinned_wrapper_unchanged(monkeypatch):
    obs = load_observed()

    from types import SimpleNamespace

    # looks like the pinned pyramid wrapper to is_pyramid_wrapper
    m = SimpleNamespace(__name__="cascade_pyramid_checkpoint", causal_standardize=lambda *a: a)
    w = SimpleNamespace(m=m, model=object(), cfg=object(), device="cpu", quantile_levels=(0.5,))
    assert obs.is_pyramid_wrapper(w)
    assert obs.adapt(w, "impute/0") is w
    assert isinstance(obs.adapt(w, "observed_mask/1"), obs.ObservedMaskWrapper)
    monkeypatch.setenv("CASCADE_BENCH_METHOD", "impute/0")
    assert obs.current_method() == "impute/0" and obs.adapt(w) is w
    monkeypatch.setenv("CASCADE_BENCH_METHOD", "bogus/9")
    with pytest.raises(ValueError, match="bogus/9"):
        obs.current_method()
    monkeypatch.delenv("CASCADE_BENCH_METHOD")
    assert obs.current_method() == "observed_mask/1"


# ── cascade bridge: rows stamp, method of published rows, the CLI arg ────────


def test_item_rows_carry_the_method_and_unstamped_rows_are_legacy():
    from cascade.eval import benchmarks as B

    rows = B.item_rows_from_report(_report("observed_mask/1"))
    assert rows["gift"]["bench_method"] == "observed_mask/1"
    assert B.rows_bench_method(rows) == "observed_mask/1"
    legacy = B.item_rows_from_report({k: v for k, v in _report("x").items()
                                      if k != "bench_method"})
    assert "bench_method" not in legacy["gift"]
    assert B.rows_bench_method(legacy) == B.LEGACY_BENCH_METHOD == "impute/0"
    mixed = {"gift": {"bench_method": "observed_mask/1"}, "boom": {"bench_method": "impute/0"}}
    assert B.rows_bench_method(mixed) is None
    assert B.bench_method_arg(None) == []
    assert B.bench_method_arg("impute/0") == ["--bench-method", "impute/0"]
    with pytest.raises(ValueError):
        B.bench_method_arg("bogus/9")


def test_run_bench_report_passes_the_method_to_the_sidecar(monkeypatch):
    from cascade.eval import benchmarks as B

    seen = []
    monkeypatch.setattr(B, "_invoke_sidecar",
                        lambda ckpt, project, tail, **_: seen.append(list(tail)) or {})
    B.run_bench_report("/x", project_dir="/p", suites=("gift-eval",), bench_method="impute/0")
    B.run_bench_report("/x", project_dir="/p", suites=("gift-eval",))
    assert seen[0][-2:] == ["--bench-method", "impute/0"]
    assert "--bench-method" not in seen[1]


# ── the audit: reproduce with the published method ───────────────────────────


class _Store:
    def __init__(self):
        self.d = {}

    def put(self, key, rows):
        text = json.dumps(rows, sort_keys=True)
        self.d[key] = text
        return hashlib.sha256(text.encode()).hexdigest()

    def get_text(self, key):
        return self.d[key]


def _audit_report(store, rows):
    from cascade.shared.bench_report import BenchEntry, BenchReport
    from cascade.shared.manifest import BenchScores

    sha = store.put("bench_v/rows/1/x.json", rows)
    e = BenchEntry(role="king", size="", miner_hotkey="k", miner_uid=0, trained_pointer=PTR,
                   scores=BenchScores(*([1.0] * 6)), rows_key="bench_v/rows/1/x.json",
                   rows_sha256=sha, data_revisions={"gift-eval": "rev1"})
    return BenchReport(round_id="1", created_block=5, entries=(e,))


def _rows(method: str | None, crps: float = 0.4):
    suite = {"status": "ok", "rows": [{"item": "m4/D/short", "crps": crps, "mase": 1.2}]}
    if method:
        suite["bench_method"] = method
    return {"gift": suite}


@pytest.mark.parametrize("published", [None, "impute/0", "observed_mask/1"])
def test_audit_reproduces_each_report_with_its_own_method(published):
    from cascade.audit.bench_reproduce import reproduce_report

    store = _Store()
    report = _audit_report(store, _rows(published))
    calls = []

    def sweep(entry, bench_method=None):
        calls.append(bench_method)
        # each method reproduces only its own numbers
        good = bench_method == (published or "impute/0")
        return _rows(bench_method, 0.4 if good else 0.9), {"gift-eval": "rev1"}

    (res,) = reproduce_report(report, store, sweep_fn=sweep)
    assert calls == [published or "impute/0"]
    assert res.status == "PASS", res.detail
    assert (published or "impute/0") in res.detail


def test_audit_warns_instead_of_failing_when_it_cannot_run_the_method():
    from cascade.audit.bench_reproduce import reproduce_report

    store = _Store()

    def method_blind(entry):
        return _rows("observed_mask/1", 0.9), {"gift-eval": "rev1"}

    (old,) = reproduce_report(_audit_report(store, _rows(None)), store, sweep_fn=method_blind)
    assert old.status == "WARN" and "impute/0" in old.detail
    (unknown,) = reproduce_report(_audit_report(store, _rows("future/7")), store,
                                  sweep_fn=lambda e, bench_method=None: None)
    assert unknown.status == "WARN" and "future/7" in unknown.detail


def test_default_sweep_fn_forwards_the_method():
    import inspect

    from cascade.audit import bench_reproduce

    src = inspect.getsource(bench_reproduce.default_sweep_fn)
    assert "def sweep(entry, bench_method=None):" in src
    assert "bench_method=bench_method)" in src
