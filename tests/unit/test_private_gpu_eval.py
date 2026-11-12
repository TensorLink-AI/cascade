from __future__ import annotations

import dataclasses
import json
import logging
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


def _shell_runner(root, worker, calls=None, fail=None):
    """A local stand-in for the SSH/SCP bridge: scp copies files, ssh commands
    run in a real shell (the validator's remote commands exercised as
    written), and the worker command runs ``worker`` in-process on CPU with
    its paths resolved against the host workdir ``root``. ``fail(argv)`` true
    ⇒ that call returns exit 1."""
    root = Path(root)

    def run(argv, _timeout):
        if calls is not None:
            calls.append(argv)
        if fail is not None and fail(argv):
            return SimpleNamespace(returncode=1, stdout="", stderr="injected failure")
        if argv[0] == "scp":
            source, dest = (a.split(":", 1)[-1] if "@fixture:" in a else a for a in argv[-2:])
            shutil.copyfile(source, dest)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        tokens = shlex.split(argv[-1])
        if "cascade.validator.private_worker" in tokens:
            i = tokens.index("cascade.validator.private_worker")
            worker(*[str(root / name) for name in tokens[i + 1:i + 4]])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        proc = subprocess.run(["bash", "-c", argv[-1]], capture_output=True, text=True, timeout=120)
        return SimpleNamespace(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

    return run


def test_private_transport_scores_locally_and_cleans_up(checkpoint, tmp_path):
    directory, contract = checkpoint
    host = RemoteHost("targon-fixture", "fixture", user="ubuntu", port=40123,
                      workdir=str(tmp_path / "remote"))
    host_root = Path(host.workdir)
    host_root.mkdir()
    run = _shell_runner(host_root, lambda *p: run_request(*p, device="cpu", workers=2))

    windows = _windows()
    samples = iter(forecasts_via_host(host, directory, windows, contract=contract,
                                     num_samples=8, workers=2, runner=run))
    got = score_joint_forecaster_on_windows(lambda *_: next(samples), windows, 8)
    _equal(evaluate_checkpoint(directory, windows, num_samples=8, contract=contract), got)
    # the run dir is gone; only the verified checkpoint cache stays
    assert sorted(p.name for p in (host_root / "_private_eval_offload").iterdir()) == ["cache"]


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


# ── streamed private forecasts (2026-11-03: the validator box OOM'd while a
# duel's full sample list sat in RAM next to a trainer restart) ─────────────

def _fake_transport(tmp_path, worker):
    host = RemoteHost("targon-fixture", "fixture", user="ubuntu", port=40123,
                      workdir=str(tmp_path / "remote"))
    root = Path(host.workdir)
    root.mkdir()
    return host, _shell_runner(root, worker)


def test_streamed_forecasts_are_the_same_arrays_and_release_the_local_file(checkpoint, tmp_path):
    from cascade.validator.eval_offload import StreamedForecasts

    directory, contract = checkpoint
    host, run = _fake_transport(
        tmp_path, lambda *p: run_request(*p, device="cpu", workers=1))
    windows = _windows()
    got = forecasts_via_host(host, directory, windows, contract=contract,
                             num_samples=8, runner=run)
    assert isinstance(got, StreamedForecasts) and len(got) == len(windows)
    local = got.path
    assert local.exists()
    streamed = list(got)                       # exhausting it releases the temp dir
    assert not local.exists()
    (tmp_path / "again").mkdir()
    host2, run2 = _fake_transport(
        tmp_path / "again", lambda *p: run_request(*p, device="cpu", workers=1))
    again = forecasts_via_host(host2, directory, windows, contract=contract,
                               num_samples=8, runner=run2)
    reference = read_forecasts(again.path, _identity(again), windows, 8)
    again.close()
    assert not again.path.exists()
    assert len(streamed) == len(reference)
    assert all(np.array_equal(a, b) for a, b in zip(streamed, reference, strict=True))


def _identity(streamed):
    with np.load(streamed.path, allow_pickle=False) as data:
        return data["request_sha256"].tobytes().decode()


def test_a_corrupt_reply_holds_as_infrastructure_before_any_score(checkpoint, tmp_path,
                                                                  monkeypatch):
    import tempfile

    directory, contract = checkpoint
    local_tmp = tmp_path / "local-tmp"
    local_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(local_tmp))

    def nan_worker(ckpt, request, output):
        run_request(ckpt, request, output, device="cpu", workers=1)
        with np.load(output, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays["s_1"][0, 0, 0] = np.nan       # a LATE window: the whole reply is checked first
        np.savez(output, **arrays)

    host, run = _fake_transport(tmp_path, nan_worker)
    with pytest.raises(PrivateEvalUnavailable):
        forecasts_via_host(host, directory, _windows(), contract=contract,
                           num_samples=8, runner=run)
    assert not list(local_tmp.iterdir())       # the local copy is released on failure


def test_evaluate_closes_the_stream_after_scoring(checkpoint, cfg, monkeypatch):
    from cascade.validator import eval_offload
    from cascade.validator.evaluation import EvaluationMixin

    directory, contract = checkpoint
    windows = _windows()
    reference = [np.full((w.n_channels, 8, w.horizon), 1.0) for w in windows]
    closed = []

    class Fake:
        def __iter__(self):
            return iter(reference)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(eval_offload, "forecasts_via_host", lambda *a, **k: Fake())
    runner = SimpleNamespace(
        evaluate_fn=None, _fetch_checkpoint_dir=lambda _: directory, cfg=replace(
            cfg, eval=replace(cfg.eval, num_samples=8)),
        private_eval_host_fn=lambda: object(), private_eval_workers=1,
        private_eval_timeout_s=60)
    rows = EvaluationMixin._evaluate(runner, SimpleNamespace(size=contract.arch_preset),
                                     windows, contract)
    assert rows and closed == [True]


# ── pre-upload compatibility probe + the worker's own error in the log ──────


def _probe_host():
    return RemoteHost("gpu-fixture", "fixture", user="ubuntu", port=40123, workdir="/w")


def _probe_reply(fields, schemas=("cascade-private-eval/1", "cascade-private-eval/2-quantiles")):
    def run(_argv, _timeout):
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            "motd noise\n" + json.dumps({"contract_fields": list(fields), "schemas": list(schemas)})))
    return run


@dataclasses.dataclass
class _Contract:
    alpha: int = 1
    beta: int = 2


def test_probe_refuses_a_host_missing_a_contract_field_and_names_it(caplog):
    from cascade.validator.eval_offload import check_private_worker

    with caplog.at_level(logging.ERROR, "cascade.validator.eval_offload"), \
            pytest.raises(PrivateEvalUnavailable, match="out of date"):
        check_private_worker(_probe_host(), _Contract(), run=_probe_reply(["alpha"]))
    assert "beta" in caplog.text


def test_probe_passes_a_matching_host_and_only_warns_on_a_newer_one(caplog):
    from cascade.validator.eval_offload import check_private_worker

    check_private_worker(_probe_host(), _Contract(), run=_probe_reply(["alpha", "beta"]))
    assert not caplog.records
    with caplog.at_level(logging.WARNING, "cascade.validator.eval_offload"):
        check_private_worker(_probe_host(), _Contract(), run=_probe_reply(["alpha", "beta", "gamma"]))
    assert "gamma" in caplog.text


def test_probe_requires_the_quantile_schema_only_for_quantile_requests():
    from cascade.validator.eval_offload import check_private_worker

    old = _probe_reply(["alpha", "beta"], schemas=("cascade-private-eval/1", ""))
    check_private_worker(_probe_host(), _Contract(), run=old)
    with pytest.raises(PrivateEvalUnavailable):
        check_private_worker(_probe_host(), _Contract(), quantile_levels=(0.5,), run=old)


@pytest.mark.parametrize("reply", [
    SimpleNamespace(returncode=1, stdout="", stderr="ssh: refused"),
    SimpleNamespace(returncode=0, stdout="", stderr=""),
    SimpleNamespace(returncode=0, stdout="not json", stderr=""),
    SimpleNamespace(returncode=0, stdout=json.dumps({"schemas": []}), stderr=""),
])
def test_a_failed_probe_never_decides_the_round(reply):
    from cascade.validator.eval_offload import check_private_worker

    check_private_worker(_probe_host(), _Contract(), run=lambda *_: reply)


def test_a_raising_probe_never_decides_the_round():
    from cascade.validator.eval_offload import check_private_worker

    def boom(*_):
        raise TimeoutError("ssh hung")

    check_private_worker(_probe_host(), _Contract(), run=boom)


def test_the_probe_script_reports_this_release(tmp_path):
    from cascade.shared.config.training import TrainingContractConfig
    from cascade.validator.eval_offload import WORKER_PROBE
    from cascade.validator.private_worker import (
        SCHEMA,
        SCHEMA_QUANTILES,
        SCHEMA_QUANTILES_BATCHED,
    )

    proc = subprocess.run([sys.executable, "-c", WORKER_PROBE], capture_output=True, text=True,
                          check=True, timeout=300)
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(info["contract_fields"]) == {f.name for f in dataclasses.fields(TrainingContractConfig)}
    assert set(info["schemas"]) == {SCHEMA, SCHEMA_QUANTILES, SCHEMA_QUANTILES_BATCHED}


def test_an_out_of_date_host_gets_no_upload(checkpoint, tmp_path):
    directory, contract = checkpoint
    host = RemoteHost("gpu-fixture", "fixture", workdir=str(tmp_path))
    calls = []

    def run(argv, _timeout):
        calls.append(argv)
        if argv[0] == "ssh" and " -c " in argv[-1]:
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
                {"contract_fields": ["only_this"], "schemas": ["cascade-private-eval/1"]}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(PrivateEvalUnavailable, match="out of date"):
        forecasts_via_host(host, directory, _windows(), contract=contract, num_samples=8, runner=run)
    assert not [a for a in calls if a[0] == "scp"]
    assert calls[-1][-1].startswith("rm -rf ")


def test_the_workers_own_error_reaches_the_validator_log(checkpoint, tmp_path, caplog):
    directory, contract = checkpoint
    host = RemoteHost("gpu-fixture", "fixture", workdir=str(tmp_path))

    def run(argv, _timeout):
        if "cascade.validator.private_worker" in " ".join(argv) and " -c " not in argv[-1]:
            return SimpleNamespace(returncode=1, stdout="", stderr="TypeError: unexpected keyword")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with caplog.at_level(logging.ERROR, "cascade.validator.eval_offload"), \
            pytest.raises(PrivateEvalUnavailable):
        forecasts_via_host(host, directory, _windows(), contract=contract, num_samples=8, runner=run)
    assert "TypeError: unexpected keyword" in caplog.text


# ── checkpoint cache on the private host (2026-11-10: a lossy link made every
# held re-run re-upload two 4.6 GB checkpoints, ~75 min per duel) ───────────


def _weight_uploads(calls):
    return [a for a in calls if a[0] == "scp" and a[-2].endswith("weights.safetensors")]


def _cache_entries(root):
    cache = Path(root) / "_private_eval_offload" / "cache"
    return sorted(p.name for p in cache.iterdir()) if cache.is_dir() else []


def test_a_rerun_reuses_the_verified_cached_checkpoint(checkpoint, tmp_path):
    directory, contract = checkpoint
    root = tmp_path / "remote"
    root.mkdir()
    host = RemoteHost("gpu-fixture", "fixture", user="ubuntu", workdir=str(root))
    calls = []
    run = _shell_runner(root, lambda *p: run_request(*p, device="cpu", workers=1), calls=calls)
    windows = _windows()
    first = list(forecasts_via_host(host, directory, windows, contract=contract,
                                    num_samples=8, runner=run))
    second = list(forecasts_via_host(host, directory, windows, contract=contract,
                                     num_samples=8, runner=run))
    assert len(_weight_uploads(calls)) == 1          # the re-run uploaded nothing
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert len(_cache_entries(root)) == 1


def test_a_tampered_cache_entry_is_uploaded_again(checkpoint, tmp_path):
    directory, contract = checkpoint
    root = tmp_path / "remote"
    root.mkdir()
    host = RemoteHost("gpu-fixture", "fixture", user="ubuntu", workdir=str(root))
    calls = []
    run = _shell_runner(root, lambda *p: run_request(*p, device="cpu", workers=1), calls=calls)
    windows = _windows()
    list(forecasts_via_host(host, directory, windows, contract=contract, num_samples=8, runner=run))
    (entry,) = _cache_entries(root)
    weights = root / "_private_eval_offload" / "cache" / entry / "weights.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    list(forecasts_via_host(host, directory, windows, contract=contract, num_samples=8, runner=run))
    assert len(_weight_uploads(calls)) == 2          # mismatch ⇒ fresh upload, never reuse
    assert _cache_entries(root) == [entry]


def test_a_failed_upload_leaves_no_partial_entry(checkpoint, tmp_path):
    directory, contract = checkpoint
    root = tmp_path / "remote"
    root.mkdir()
    host = RemoteHost("gpu-fixture", "fixture", user="ubuntu", workdir=str(root))
    run = _shell_runner(root, lambda *p: run_request(*p, device="cpu", workers=1),
                        fail=lambda a: a[0] == "scp" and a[-2].endswith("weights.safetensors"))
    with pytest.raises(PrivateEvalUnavailable):
        forecasts_via_host(host, directory, _windows(), contract=contract, num_samples=8,
                           runner=run)
    assert _cache_entries(root) == []
    assert sorted(p.name for p in (root / "_private_eval_offload").iterdir()) == ["cache"]


def test_the_cache_keeps_only_the_most_recent_entries(tmp_path):
    import os
    import time as _time

    from cascade.validator.eval_offload import (
        PRIVATE_CACHE_KEEP,
        private_cache_setup_command,
    )

    cache = tmp_path / "cache"
    cache.mkdir()
    now = _time.time()
    names = [f"entry{i}" for i in range(PRIVATE_CACHE_KEEP + 3)]
    for i, name in enumerate(names):                  # entry0 newest … oldest last
        (cache / name).mkdir()
        os.utime(cache / name, (now - 60 * i, now - 60 * i))
    os.utime(cache / names[-1], (now - 3600 * 24, now - 3600 * 24))
    (cache / "fresh.part-aaaa").mkdir()               # an upload in flight: kept
    stale = cache / "old.part-bbbb"
    stale.mkdir()
    os.utime(stale, (now - 3600 * 24, now - 3600 * 24))
    key = names[-1]                                   # the oldest entry is the one in use
    proc = subprocess.run(["bash", "-c", private_cache_setup_command(
        str(tmp_path / "run"), str(cache), key)], capture_output=True, text=True, check=True)
    left = sorted(p.name for p in cache.iterdir())
    assert key in left and "fresh.part-aaaa" in left and "old.part-bbbb" not in left
    assert sorted(n for n in left if n.startswith("entry")) == sorted(
        names[:PRIVATE_CACHE_KEEP - 1] + [key])
    assert (tmp_path / "run").is_dir()
    assert proc.stdout == ""                          # no files in the entry ⇒ nothing to reuse


def test_parse_sha256sum_reads_text_and_binary_mode_lines():
    from cascade.validator.eval_offload import parse_sha256sum

    h = "a" * 64
    assert parse_sha256sum(f"{h}  config.json\n{h.upper()} *model.py\nnoise\n") == {
        "config.json": h, "model.py": h}
    assert parse_sha256sum("") == {} and parse_sha256sum(None) == {}


def test_write_request_with_precomputed_hashes_is_byte_identical(checkpoint, tmp_path):
    from cascade.validator.private_worker import checkpoint_hashes

    directory, contract = checkpoint
    a = write_request(tmp_path / "a.npz", directory, _windows(), contract=contract, num_samples=8)
    b = write_request(tmp_path / "b.npz", directory, _windows(), contract=contract, num_samples=8,
                      hashes=checkpoint_hashes(directory))
    assert a == b
