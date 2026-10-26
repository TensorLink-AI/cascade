from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.scoring import score_joint_forecaster_on_windows
from cascade.eval.window import EvalWindow
from cascade.trainer.remote import RemoteHost
from cascade.validator.eval_offload import PrivateEvalUnavailable, forecasts_via_host
from cascade.validator.evaluator import evaluate_checkpoint
from cascade.validator.private_worker import read_forecasts, run_request, write_request


@pytest.fixture
def checkpoint(cfg, tmp_path):
    import torch

    from cascade.trainer.moe_model import QUANTILE_LEVELS, CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.moe_trainer import _save_moe_checkpoint

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    contract = replace(cfg.training.v2_contract, base_arch="cascade_moe", training_steps=0, num_quantiles=9, d_model=16, num_heads=2, head_dim=8, num_layers=4,
                       patch_size=2, context_length=16, horizon=4, d_ff=32)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model_cfg = CascadeMoEConfig.from_contract(contract)
        model = CascadeMoEModel(model_cfg)
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    _save_moe_checkpoint(directory, model.state_dict(), model_cfg, QUANTILE_LEVELS, contract)
    yield directory, contract
    torch.set_num_threads(old_threads)


def _windows():
    rng = np.random.default_rng(14)
    return [EvalWindow(str(i), rng.normal(size=(c, 20 + i)), rng.normal(size=(c, 4 + i % 3)),
                       {"source": "private-feed", "seasonal_period": 2})
            for i, c in enumerate((1, 2, 1, 3, 1, 2))]


def _equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.series_id == y.series_id and x.channel == y.channel
        assert x.mase == y.mase and x.abs_target == y.abs_target
        assert x.source == y.source and x.domain == y.domain
        np.testing.assert_array_equal(x.qloss_per_q, y.qloss_per_q)


def test_real_moe_parallel_windows_preserve_scores_and_order(checkpoint):
    directory, contract = checkpoint
    windows = _windows()
    solo = evaluate_checkpoint(directory, windows, num_samples=8, contract=contract)
    parallel = evaluate_checkpoint(directory, windows, num_samples=8, contract=contract, workers=2)
    _equal(solo, parallel)


def test_fresh_worker_returns_same_forecasts_without_receiving_targets(checkpoint, tmp_path):
    directory, contract = checkpoint
    windows = _windows()
    request, output = tmp_path / "request.npz", tmp_path / "samples.npz"
    identity = write_request(request, directory, windows, contract=contract, num_samples=8)
    with np.load(request, allow_pickle=False) as data:
        assert set(data.files) == {"metadata"} | {f"h_{i}" for i in range(len(windows))}
        assert "private-feed" not in data["metadata"].tobytes().decode()
    proc = subprocess.run([sys.executable, "-m", "cascade.validator.private_worker", str(directory),
                    str(request), str(output), "--device", "cpu", "--workers", "2"],
                   check=True, timeout=60, capture_output=True, text=True)
    assert json.loads(proc.stdout)["windows"] == len(windows)
    samples = iter(read_forecasts(output, identity, windows, 8))
    remote = score_joint_forecaster_on_windows(lambda *_: next(samples), windows, 8)
    _equal(evaluate_checkpoint(directory, windows, num_samples=8, contract=contract), remote)
    with pytest.raises(ValueError, match="another request"):
        read_forecasts(output, "wrong", windows, 8)


def test_private_qualification_checks_exact_forecasts(checkpoint, tmp_path):
    directory, contract = checkpoint
    request, output = tmp_path / "request.npz", tmp_path / "samples.npz"
    write_request(request, directory, _windows(), contract=contract, num_samples=8)
    report = run_request(directory, request, output, device="cpu", workers=2, qualify=True)
    assert report["qualified"] and report["workers"] == 2
    assert report["sequential_seconds"] > 0 and report["seconds"] > 0


def test_worker_refuses_checkpoint_changed_after_request(checkpoint, tmp_path):
    directory, contract = checkpoint
    request = tmp_path / "request.npz"
    write_request(request, directory, _windows(), contract=contract, num_samples=8)
    config = directory / "config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(ValueError, match="checkpoint hash"):
        run_request(directory, request, tmp_path / "samples.npz", device="cpu")


def test_private_transport_scores_locally_and_cleans_up(checkpoint, tmp_path):
    directory, contract = checkpoint
    host = RemoteHost("targon-fixture", "fixture", user="ubuntu", port=40123,
                      workdir=str(tmp_path / "remote"))
    host_root = Path(host.workdir)
    host_root.mkdir()
    bases = []

    def run(argv, _timeout):
        if argv[0] == "scp":
            source, dest = argv[-2:]
            source = source.split(":", 1)[-1] if source.startswith("ubuntu@fixture:") else source
            dest = dest.split(":", 1)[-1] if dest.startswith("ubuntu@fixture:") else dest
            shutil.copyfile(source, dest)
        else:
            command = shlex.split(argv[-1])
            if "mkdir" in command:
                path = Path(command[-1])
                bases.append(path.parent)
                path.mkdir(parents=True)
            elif "cascade.validator.private_worker" in command:
                i = command.index("cascade.validator.private_worker")
                # SSH runs the worker after cd into host.workdir; model the
                # working directory without changing the test process cwd.
                paths = [str(host_root / name) for name in command[i + 1:i + 4]]
                run_request(*paths, device="cpu", workers=2)
            elif command[0] == "stat":
                return SimpleNamespace(returncode=0, stdout=str(Path(command[-1]).stat().st_size))
            elif command[0] == "rm":
                shutil.rmtree(command[-1], ignore_errors=True)
        return SimpleNamespace(returncode=0, stdout="")

    windows = _windows()
    samples = iter(forecasts_via_host(host, directory, windows, contract=contract,
                                     num_samples=8, workers=2, runner=run))
    got = score_joint_forecaster_on_windows(lambda *_: next(samples), windows, 8)
    _equal(evaluate_checkpoint(directory, windows, num_samples=8, contract=contract), got)
    assert bases and all(not p.exists() for p in bases)


def test_transport_failure_is_infrastructure_and_always_cleans_up(checkpoint, tmp_path):
    directory, contract = checkpoint
    calls = []
    host = RemoteHost("targon-fixture", "fixture", workdir=str(tmp_path))

    def fail(argv, _timeout):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stdout="")

    with pytest.raises(PrivateEvalUnavailable):
        forecasts_via_host(host, directory, _windows(), contract=contract,
                           num_samples=8, runner=fail)
    assert calls[-1][-1].startswith("rm -rf ")


def test_missing_private_gpu_holds_instead_of_switching_to_local(checkpoint, cfg):
    from cascade.validator.evaluation import EvaluationMixin

    directory, contract = checkpoint
    runner = SimpleNamespace(
        evaluate_fn=None, _fetch_checkpoint_dir=lambda _: directory,
        cfg=cfg, private_eval_host_fn=lambda: None, private_eval_workers=2)
    with pytest.raises(PrivateEvalUnavailable):
        EvaluationMixin._evaluate(runner, SimpleNamespace(size=contract.arch_preset),
                                  _windows(), contract)
    assert EvaluationMixin._entry_fault_reason(PrivateEvalUnavailable("GPU disappeared")) is None


def test_private_runtime_refuses_incompatible_library_release():
    from cascade.eval.torch_runtime import apply_runtime, capture_runtime

    runtime = capture_runtime()
    runtime["torch"] = "0.0.0"
    with pytest.raises(ValueError, match="runtime differs"):
        apply_runtime(runtime)


@pytest.mark.parametrize("kind", ["missing", "shape", "nan", "duplicate"])
def test_invalid_or_partial_forecasts_cannot_be_scored(tmp_path, kind):
    window = _windows()[0]
    values = {"request_sha256": np.frombuffer(b"identity", dtype=np.uint8),
              "s_0": np.ones((1, 8, window.horizon))}
    if kind == "missing":
        del values["s_0"]
    elif kind == "shape":
        values["s_0"] = np.ones((1, 1, window.horizon))
    elif kind == "nan":
        values["s_0"][0, 0, 0] = np.nan
    else:
        values["s_1"] = values["s_0"]
    path = tmp_path / "samples.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        read_forecasts(path, "identity", [window], 8)
