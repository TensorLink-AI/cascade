from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from cascade.eval.benchmarks import bench_uv_args
from cascade.validator.bench_identity import current_revisions
from cascade.validator.eval_offload import make_eval_host_fn
from cascade.validator.validator_bench import ValidatorBenchMixin
from scripts.bench_timing import build_command, validate_report

SUITES = ["gift-eval", "boom", "time", "fev"]


def _report():
    return {"data_revisions": {s: "a" * (64 if s == "fev" else 40) for s in SUITES},
            "suites": [{"suite": s, "status": "ok", "metrics": {"crps": .8, "mase": .9},
                        "rows": [{"full": "task"}], "n_series": 1, "n_expected": 1,
                        "skipped": []} for s in SUITES]}


def test_four_suite_invocations_activate_both_optional_runners():
    assert bench_uv_args(SUITES) == ["--frozen", "--extra", "time", "--extra", "fev"]
    argv = build_command("uv", "benchmarks", "ckpt", "out", workers="auto", timings_json="t",
                         suites=",".join(SUITES), device="cuda", batch_size=512, max_series=0,
                         num_samples=100, data_dir="data")
    assert argv[argv.index("--suites") + 1] == ",".join(SUITES)
    assert "--frozen" in argv and argv.count("--extra") == 2


@pytest.mark.parametrize("failure", ["missing_suite", "skipped_suite", "missing_row",
                                    "duplicate_row", "unknown_pin", "partial_pin", "nan_metric"])
def test_qualification_refuses_fast_but_incomplete_reports(failure):
    doc = _report()
    if failure == "missing_suite":
        doc["suites"].pop()
    elif failure == "skipped_suite":
        doc["suites"][2]["status"] = "skipped"
    elif failure == "missing_row":
        doc["suites"][1]["rows"] = []
    elif failure == "duplicate_row":
        row = doc["suites"][1]
        row.update(rows=[{"full": "task"}, {"full": "task"}], n_series=2, n_expected=2)
    elif failure == "unknown_pin":
        doc["data_revisions"]["fev"] = "unknown"
    elif failure == "partial_pin":
        doc["data_revisions"]["time"] = "rev (partial)"
    else:
        doc["suites"][0]["metrics"]["crps"] = float("nan")
    with pytest.raises(ValueError):
        validate_report(doc, SUITES)


def test_fev_declared_scope_is_distinct_from_failed_task():
    doc = _report()
    fev = doc["suites"][-1]
    fev.update(n_expected=2, skipped=[{"full": "multivariate", "stage": "scope"}])
    validate_report(doc, SUITES)
    failed = copy.deepcopy(doc)
    failed["suites"][-1]["skipped"][0]["stage"] = "score"
    with pytest.raises(ValueError):
        validate_report(failed, SUITES)


def test_eval_host_exposes_all_gpus_of_one_vm_but_never_combines_machines(tmp_path):
    path = tmp_path / "hosts.toml"
    path.write_text("\n".join(
        f'[[host]]\nname="gpu-{i}"\nhost="{host}"\nuser="ubuntu"\nport=40123\n'
        f'workdir="/home/ubuntu/cascade"\ncuda_device="{device}"\nstage="final"\n'
        for i, host, device in [(0, "same-vm", "0"), (1, "same-vm", "1"), (2, "other-vm", "0")]))
    host = make_eval_host_fn(path)()
    assert host.host == "same-vm" and host.cuda_device == "0,1" and host.user == "ubuntu"


def test_both_cached_sides_are_invalidated_when_data_pins_change(tmp_path):
    for name in SUITES[:3]:
        root = tmp_path / name
        root.mkdir()
        (root / "_cascade_revision.json").write_text(json.dumps({"revision": name + "-one"}))
    fev = tmp_path / "fev"
    fev.mkdir()
    manifest = fev / "manifest.json"
    manifest.write_text(json.dumps({"tasks": {"sha256": "one"}, "datasets": []}))
    original = current_revisions(tmp_path)
    runner = SimpleNamespace(_vbench_cache={"king": {"scores": {"fev_crps": .8},
                                                   "revisions": original}},
                             bench_sweep_fn=None, bench_sweep_many_fn=None,
                             _vbench_current_data_revisions=lambda: current_revisions(tmp_path))
    assert ValidatorBenchMixin._vbench_cache_hit(runner, "king", {"require_fev": True})
    manifest.write_text(json.dumps({"tasks": {"sha256": "two"}, "datasets": []}))
    assert ValidatorBenchMixin._vbench_cache_hit(runner, "king", {"require_fev": True}) is None
