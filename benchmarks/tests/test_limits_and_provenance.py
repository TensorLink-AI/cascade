from __future__ import annotations

import hashlib
import json
import os
import sys
import weakref
from types import SimpleNamespace

import pytest
from cascade_benchmark import cli
from cascade_benchmark.cache import cache_key, scoped_baseline_cache_dir
from cascade_benchmark.limits import GIB, host_limits, spawn_environment
from cascade_benchmark.parallel import choose_plan, interleaved_jobs
from cascade_benchmark.provenance import verify_fev_manifest
from cascade_benchmark.results import SuiteResult
from cascade_benchmark.suites._common import SuitePlan


def _put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_nested_cpu_quota_and_memory_limit_override_host_resources(tmp_path, monkeypatch):
    root, proc = tmp_path / "cgroup", tmp_path / "proc"
    _put(proc / "self/cgroup", "0::/tenant/job\n")
    _put(proc / "meminfo", "MemAvailable: 100000000 kB\n")
    _put(root / "cpu.max", "max 100000")
    _put(root / "tenant/cpu.max", "350000 100000")
    _put(root / "tenant/job/cpu.max", "800000 100000")
    _put(root / "tenant/memory.max", str(10 * GIB))
    _put(root / "tenant/memory.current", str(3 * GIB))
    _put(root / "tenant/job/memory.max", "max")
    monkeypatch.setattr(os, "cpu_count", lambda: 192)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(12)))
    assert host_limits(cgroup_root=root, proc_root=proc) == (3, 7 * GIB)


def test_v1_limits_and_fractional_cpu(tmp_path, monkeypatch):
    root, proc = tmp_path / "cgroup", tmp_path / "proc"
    _put(proc / "self/cgroup", "1:cpu,cpuacct:/job\n2:memory:/job\n")
    _put(root / "cpu,cpuacct/job/cpu.cfs_quota_us", "50000")
    _put(root / "cpu,cpuacct/job/cpu.cfs_period_us", "100000")
    _put(root / "memory/job/memory.limit_in_bytes", str(4 * GIB))
    _put(root / "memory/job/memory.usage_in_bytes", str(GIB))
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert host_limits(cgroup_root=root, proc_root=proc) == (1, 3 * GIB)


def test_auto_worker_ram_cap_does_not_oversubscribe_small_vm():
    plan = choose_plan("auto", checkpoints=[], device="cpu", batch_size=64, n_units=100,
                       cpu_count=64, available_ram_bytes=6 * GIB, per_worker_ram_bytes=2 * GIB)
    assert plan.workers == 2 and len(plan.devices) == 2
    assert "RAM cap=2" in plan.reason


def test_all_four_suites_start_early_without_reordering_their_units():
    names = ["gift-eval", "boom", "time", "fev"]
    plans = {name: SuitePlan(suite=name, units=list(range(n)))
             for name, n in zip(names, [3, 20, 2, 4], strict=True)}
    jobs = interleaved_jobs(names, plans)
    assert jobs[:4] == [(name, 0) for name in names]
    for name in names:
        assert [u for s, u in jobs if s == name] == plans[name].units
    plans["fev"] = SuiteResult(suite="fev", status="skipped")
    assert all(name != "fev" for name, _ in interleaved_jobs(names, plans))


def test_spawn_masks_and_threads_are_restored_even_after_failure(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-original")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with pytest.raises(RuntimeError), spawn_environment("GPU-other"):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-other"
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        raise RuntimeError("spawn failed")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-original"
    assert os.environ["OMP_NUM_THREADS"] == "12"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_vram_probe_retains_every_model_in_a_paired_pass(monkeypatch):
    import torch
    from cascade_benchmark.parallel import _probe_main

    live = weakref.WeakSet()

    class Wrapper:
        def forecast_quantiles_batch(self, *_):
            pass

    def load(*_):
        wrapper = Wrapper()
        live.add(wrapper)
        return wrapper

    for name, value in {
        "is_available": lambda: True, "device_count": lambda: 1,
        "mem_get_info": lambda _: (16 * GIB, 24 * GIB), "get_device_name": lambda _: "fixture",
        "reset_peak_memory_stats": lambda _: None, "synchronize": lambda _: None,
        "max_memory_reserved": lambda _: len(live) * 100,
        "max_memory_allocated": lambda _: len(live) * 100,
    }.items():
        monkeypatch.setattr(torch.cuda, name, value)
    monkeypatch.setattr("cascade_benchmark.predictor._load_wrapper", load)
    values = []
    _probe_main(["candidate", "king"], "cuda", 1, True, SimpleNamespace(put=values.append))
    assert values[0]["peak"] == 200  # two live fixture models; not a GPU measurement


def _manifest(tmp_path, monkeypatch):
    data, tasks = tmp_path / "data.parquet", tmp_path / "tasks.yaml"
    data.write_bytes(b"fixture bytes, not performance evidence")
    tasks.write_text("local tasks")
    task = SimpleNamespace(dataset_path=str(data), dataset_config=None)
    monkeypatch.setitem(sys.modules, "fev", SimpleNamespace(
        Benchmark=SimpleNamespace(from_yaml=lambda _: SimpleNamespace(tasks=[task]))))
    content = {"tasks": {"path": tasks.name, "sha256": hashlib.sha256(tasks.read_bytes()).hexdigest()},
               "datasets": [{"path": data.name, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(content))
    return path, data, task


def test_fev_manifest_rejects_changed_data_and_remote_task(tmp_path, monkeypatch):
    path, data, task = _manifest(tmp_path, monkeypatch)
    identity = verify_fev_manifest(path)
    assert len(identity) == 64
    task.dataset_path = "autogluon/fev_datasets"
    with pytest.raises(ValueError, match="local datasets"):
        verify_fev_manifest(path)
    task.dataset_path = str(data)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dataset hash"):
        verify_fev_manifest(path)


def test_verified_tasks_cannot_change_before_planning(tmp_path, monkeypatch):
    from cascade_benchmark.suites.fev_bench import plan

    path, _data, _task = _manifest(tmp_path, monkeypatch)
    verify_fev_manifest(path)
    (tmp_path / "tasks.yaml").write_text("different tasks")
    result = plan()
    assert result.status == "error" and "changed after hash verification" in result.detail


def test_repinning_data_invalidates_loaded_fev_task_cache(tmp_path, monkeypatch):
    from cascade_benchmark.suites.fev_bench import _load_tasks

    path, data, _task = _manifest(tmp_path, monkeypatch)
    def from_yaml(_):
        return SimpleNamespace(tasks=[SimpleNamespace(dataset_path=str(data), dataset_config=None,
                                                      data_marker=data.read_bytes())])
    monkeypatch.setattr(sys.modules["fev"].Benchmark, "from_yaml", from_yaml)
    verify_fev_manifest(path)
    first = _load_tasks(str(tmp_path / "tasks.yaml"), None)
    data.write_bytes(b"new pinned data")
    manifest = json.loads(path.read_text())
    manifest["datasets"][0]["sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    verify_fev_manifest(path)
    second = _load_tasks(str(tmp_path / "tasks.yaml"), None)
    assert first[0].data_marker != second[0].data_marker == b"new pinned data"


def test_data_changed_while_scoring_aborts_the_whole_fev_suite(tmp_path, monkeypatch):
    from cascade_benchmark.suites import fev_bench

    path, data, task = _manifest(tmp_path, monkeypatch)
    task.target = "target"
    verify_fev_manifest(path)
    plan = fev_bench.plan()
    monkeypatch.setattr("cascade_benchmark.predictor.load_wrapper_cached", lambda *_:
                        SimpleNamespace(quantile_levels=fev_bench.QUANTILE_LEVELS))
    def score(_task, wrappers, **_):
        data.write_bytes(b"changed mid-score")
        return [{"CRPS": .5, "MASE": .5}] * len(wrappers)
    monkeypatch.setattr(fev_bench, "score_task_multi", score)
    outcomes = fev_bench.run_unit(plan.ctx, plan.units[0], ["candidate", "king"])
    assert all(o["abort"]["status"] == "error" for o in outcomes)
    assert fev_bench.assemble(plan, [outcomes[0]]).status == "error"


def test_pin_command_creates_a_verified_manifest_without_overwriting_inputs(tmp_path, monkeypatch):
    from cascade_benchmark.pin_fev import pin_tasks

    _path, data, _task = _manifest(tmp_path, monkeypatch)
    tasks = tmp_path / "tasks.yaml"
    output = tmp_path / "new-manifest.json"
    identity = pin_tasks(tasks, output)
    assert verify_fev_manifest(output) == identity
    with pytest.raises(ValueError, match="overwrite"):
        pin_tasks(tasks, data)


def test_cli_refuses_unpinned_fev_before_starting_sweep(tmp_path, monkeypatch):
    (tmp_path / "forecast_wrapper.py").write_text("unused")
    monkeypatch.delenv("CASCADE_BENCH_FEV_MANIFEST", raising=False)
    assert cli.main([str(tmp_path), str(tmp_path / "report.json")]) == 3
    assert not (tmp_path / "report.json").exists()
    assert cli.build_parser().parse_args(["c", "r"]).suites == "gift-eval,boom,time,fev"


def test_four_suite_cli_records_verified_pins_and_rejects_missing_data(tmp_path, monkeypatch):
    from cascade_benchmark.datasets import DATASETS

    manifest, _data, _task = _manifest(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "forecast_wrapper.py").write_text("not executed by this plumbing test")
    data_root = tmp_path / "bench"
    for name, spec in DATASETS.items():
        root = data_root / name
        root.mkdir(parents=True)
        (root / "fixture.arrow").write_bytes(b"fixture")
        (root / "_cascade_revision.json").write_text(json.dumps({"revision": spec.revision}))
    suites = ["gift-eval", "boom", "time", "fev"]
    def sweep(requested, checkpoints, **_):
        assert requested == suites and len(checkpoints) == 1
        return [[SuiteResult(suite=s, status="ok") for s in suites]]
    monkeypatch.setattr("cascade_benchmark.parallel.run_sweep", sweep)
    output = tmp_path / "out.json"
    argv = [str(checkpoint), str(output), "--data-dir", str(data_root),
            "--fev-manifest", str(manifest)]
    assert cli.main(argv) == 0
    report = json.loads(output.read_text())
    assert set(report["data_revisions"]) == set(suites)
    assert report["data_revisions"]["fev"] == verify_fev_manifest(manifest)
    assert report["methodology"]["fev"]["official_fev_leaderboard_equivalent"] is False
    (data_root / "time/_cascade_revision.json").unlink()
    assert cli.main(argv) == 3


def test_time_baseline_cache_changes_with_data_and_task_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda package: "fixture-timebench-1")
    data = tmp_path / "data"
    data.mkdir()
    marker = data / "_cascade_revision.json"
    monkeypatch.setenv("CASCADE_BENCH_TIME_DATASET", str(data))
    monkeypatch.setenv("CASCADE_BENCH_TIME_BASELINE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CASCADE_BENCH_NO_CACHE", raising=False)
    assert scoped_baseline_cache_dir({"season": 1}) is None
    marker.write_text(json.dumps({"revision": "one"}))
    one = scoped_baseline_cache_dir({"season": 1})
    assert one is not None
    assert one != scoped_baseline_cache_dir({"season": 7})
    marker.write_text(json.dumps({"revision": "two"}))
    assert one != scoped_baseline_cache_dir({"season": 1})
    two = scoped_baseline_cache_dir({"season": 1})
    monkeypatch.setattr("importlib.metadata.version", lambda package: "fixture-timebench-2")
    assert two != scoped_baseline_cache_dir({"season": 1})
    marker.write_text(json.dumps({"revision": "two", "patterns": ["subset"]}))
    assert scoped_baseline_cache_dir({"season": 1}) is None
    assert cache_key(tmp_path, "a/b", "x", 4, 9) != cache_key(tmp_path, "a_b", "x", 4, 9)


def test_a_capped_sweep_is_explicitly_partial_even_when_every_requested_task_finishes():
    from cascade_benchmark.results import BenchmarkReport

    suite = SuiteResult("gift-eval", status="ok", n_series=3, n_expected=3, max_series=3)
    row = BenchmarkReport("fixture", suites=[suite]).to_dict()["suites"][0]
    assert row["partial"] is True and row["max_series"] == 3
