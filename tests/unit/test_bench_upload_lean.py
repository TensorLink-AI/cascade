"""The validator bench offload uploads only what a bench reads, re-uses a
completed upload of the same content-addressed checkpoint, and gives the
upload its own budget so it cannot starve the benchmark (2026-11-03: two
14 GB checkpoint dirs took ~3.5 h of a 4 h sweep budget, every retry
uploading again)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cascade.validator.eval_offload as eo
from cascade.trainer.remote import RemoteHost
from cascade.validator.eval_offload import (
    bench_upload_files,
    bench_upload_manifest,
    run_bench_via_host_many,
)

DIGEST = "sha256-" + "ab" * 32


def _host():
    return RemoteHost(name="gpu1", host="1.2.3.4", user="root", port=22, workdir="/w")


def _ckpt(tmp_path, name=DIGEST):
    d = tmp_path / name
    d.mkdir()
    for f, size in (("config.json", 10), ("forecast_wrapper.py", 20), ("model.py", 30),
                    ("weights.safetensors", 400), ("weights_stable.safetensors", 400),
                    ("optimizer.safetensors", 800), ("training_measurement.json", 5)):
        (d / f).write_bytes(b"x" * size)
    return d


class _Runner:
    def __init__(self, *, cached_manifest: str = "", on_scp=None):
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.cached_manifest = cached_manifest
        self.on_scp = on_scp

    def __call__(self, argv, timeout):
        self.calls.append(argv)
        self.timeouts.append(timeout)
        text = " ".join(argv)
        if argv[0] == "scp":
            if self.on_scp:
                self.on_scp()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "head -c 65536" in text:
            ok = bool(self.cached_manifest)
            return SimpleNamespace(returncode=0 if ok else 1, stdout=self.cached_manifest,
                                   stderr="")
        if "cat " in text:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": 1}), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def scps(self):
        return [c for c in self.calls if c[0] == "scp"]

    def bench_cmd(self):
        return next(c[-1] for c in self.calls if "cascade-benchmark" in c[-1])


def test_upload_sends_only_the_files_a_bench_reads(tmp_path):
    d = _ckpt(tmp_path)
    r = _Runner()
    assert run_bench_via_host_many(_host(), [d], suites="time", runner=r) == [{"ok": 1}]
    (scp,) = r.scps()
    sent = " ".join(scp)
    for f in ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors"):
        assert str(d / f) in sent
    for f in ("optimizer.safetensors", "weights_stable.safetensors", "training_measurement.json"):
        assert f not in sent
    # uploaded into a partial dir, then published into the per-digest cache
    assert scp[-1].endswith("/") and ".part-" in scp[-1]
    fin = next(" ".join(c) for c in r.calls if "mv " in " ".join(c))
    assert eo._UPLOAD_MANIFEST in fin and f"/w/_eval_offload/cache/{DIGEST}" in fin
    # the bench runs on the cached copy, which cleanup leaves in place
    assert f"/w/_eval_offload/cache/{DIGEST}" in r.bench_cmd()
    cleanup = " ".join(r.calls[-1])
    assert "rm -rf" in cleanup and f"cache/{DIGEST} " not in cleanup + " "


def test_a_completed_upload_is_reused_not_sent_again(tmp_path):
    d = _ckpt(tmp_path)
    r = _Runner(cached_manifest=bench_upload_manifest(bench_upload_files(d)))
    assert run_bench_via_host_many(_host(), [d], suites="time", runner=r) == [{"ok": 1}]
    assert r.scps() == []
    assert f"/w/_eval_offload/cache/{DIGEST}" in r.bench_cmd()


def test_a_stale_or_partial_cached_copy_is_uploaded_again(tmp_path):
    d = _ckpt(tmp_path)
    r = _Runner(cached_manifest="weights.safetensors 1\n")   # sizes differ
    run_bench_via_host_many(_host(), [d], suites="time", runner=r)
    assert len(r.scps()) == 1


def test_non_content_addressed_dirs_are_never_cached(tmp_path):
    d = _ckpt(tmp_path, name="ckpt")
    r = _Runner(cached_manifest=bench_upload_manifest(bench_upload_files(d)))
    run_bench_via_host_many(_host(), [d], suites="time", runner=r)
    assert len(r.scps()) == 1                       # no probe short-circuit
    assert "/w/_eval_offload/cache/ckpt" not in r.bench_cmd()


def test_the_upload_does_not_eat_the_benchmark_budget(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(eo.time, "monotonic", lambda: clock[0])
    a, b = _ckpt(tmp_path, DIGEST), _ckpt(tmp_path, "sha256-" + "cd" * 32)

    def slow_scp():
        clock[0] += 1.5 * 3600        # 3 h of uploads in all — most of the old shared budget

    r = _Runner(on_scp=slow_scp)
    got = run_bench_via_host_many(_host(), [a, b], suites="time", timeout_s=4 * 3600,
                                  runner=r)
    assert got == [{"ok": 1}, {"ok": 1}]
    # the benchmark still gets its whole 4 h after 3 h of uploads
    assert "timeout --signal=TERM --kill-after=15s 14385s" in r.bench_cmd()
    bench_i = next(i for i, c in enumerate(r.calls) if "cascade-benchmark" in c[-1])
    assert r.timeouts[bench_i] == 4 * 3600


def test_the_upload_budget_still_bounds_a_stuck_upload(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(eo.time, "monotonic", lambda: clock[0])
    a, b = _ckpt(tmp_path, DIGEST), _ckpt(tmp_path, "sha256-" + "cd" * 32)

    def stuck():
        clock[0] += 10 * 3600

    r = _Runner(on_scp=stuck)
    got = run_bench_via_host_many(_host(), [a, b], suites="time", timeout_s=60,
                                  upload_timeout_s=3600, runner=r)
    assert got == [None, None]
    assert not any("cascade-benchmark" in c[-1] for c in r.calls)
    assert len(r.scps()) == 1
