"""r44 2026-08-27 incident fixes: dynamic final lane pool, retry-without-
retrain, patient checkpoint uploads.

The incident: a ~10-min pod-side network flap killed both wave-1 checkpoint
uploads (4×2s retry window), the trainer's retry re-dispatched a FULL 3h
retrain of bytes already sitting complete on the pod, and the final fleet —
snapshotted once at final start after a boot-failure slot drop — could not
pick up a replacement pod, serializing 4 duel jobs over 2 lanes.
"""

from __future__ import annotations

import pytest

from cascade.shared import hippius as hippius_mod
from cascade.trainer.loop import _FinalLanePool, ResolvedGenerator, TrainerRunner


class _Host:
    def __init__(self, name, stage="final"):
        self.name = name
        self.stage = stage


# ── the dynamic lane pool ────────────────────────────────────────────────────


def test_lane_pool_serves_initial_lanes_and_discovers_new_ones():
    calls = {"n": 0}
    joined = _Host("final-1-g0")

    def refresh():
        calls["n"] += 1
        return [_Host("final-0-g0"), joined]     # one known, one new

    pool = _FinalLanePool([_Host("final-0-g0")], refresh)
    pool.REFRESH_INTERVAL_S = 0.05
    first = pool.get()
    assert first.name == "final-0-g0"
    # the refresh on this get() absorbed the new lane — served next, no put-back
    second = pool.get()
    assert second.name == "final-1-g0"
    assert calls["n"] >= 1
    assert {h.name for h in pool.known_hosts()} == {"final-0-g0", "final-1-g0"}


def test_lane_pool_never_duplicates_known_names():
    def refresh():
        return [_Host("a"), _Host("a"), _Host("b")]

    pool = _FinalLanePool([_Host("a")], refresh)
    pool.REFRESH_INTERVAL_S = 0.05
    got = {pool.get().name, pool.get().name}
    assert got == {"a", "b"}
    assert pool.qsize() == 0                      # nothing double-enqueued


def test_lane_pool_tolerates_refresh_failure_and_keeps_serving():
    def refresh():
        raise OSError("torn hosts.toml mid-write")

    pool = _FinalLanePool([_Host("a")], refresh)
    pool.REFRESH_INTERVAL_S = 0.05
    assert pool.get().name == "a"                 # failure keeps the last set


def test_lane_pool_timeout_form_passes_through():
    pool = _FinalLanePool([], lambda: [])
    import queue as _q

    with pytest.raises(_q.Empty):
        pool.get(timeout=0.01)                    # bounded form: no refresh loop


# ── retry-without-retrain marker ─────────────────────────────────────────────


def _runner(cfg, tmp_path):
    return TrainerRunner(cfg=cfg, base_trainer=None, work_root=tmp_path)


def _gen(ref="hippius:cascade/gen-a@sha256:" + "b" * 64):
    return ResolvedGenerator(hotkey="hkA", uid=7, ref=ref, reveal_block=1)


def test_marker_round_trip_reuses_complete_checkpoint(cfg, tmp_path):
    r = _runner(cfg, tmp_path)
    contract = cfg.training.primary_size
    out = tmp_path / "checkpoint"
    out.mkdir()
    (out / "weights.safetensors").write_bytes(b"w")
    gen = _gen()
    r._write_train_complete_marker(out, contract, gen,
                                   corpus_digest="c" * 12, gpu_name="L40S")
    payload = r._reusable_checkpoint(out, contract, gen)
    assert payload is not None
    assert payload["corpus_digest"] == "c" * 12
    assert payload["gpu_name"] == "L40S"


def test_marker_absent_or_weightless_means_retrain(cfg, tmp_path):
    r = _runner(cfg, tmp_path)
    contract = cfg.training.primary_size
    out = tmp_path / "checkpoint"
    out.mkdir()
    gen = _gen()
    assert r._reusable_checkpoint(out, contract, gen) is None      # no marker
    r._write_train_complete_marker(out, contract, gen,
                                   corpus_digest="c" * 12, gpu_name="")
    assert r._reusable_checkpoint(out, contract, gen) is None      # no weights


def test_marker_mismatch_means_retrain(cfg, tmp_path):
    from dataclasses import replace as dc_replace

    r = _runner(cfg, tmp_path)
    contract = cfg.training.primary_size
    out = tmp_path / "checkpoint"
    out.mkdir()
    (out / "weights.safetensors").write_bytes(b"w")
    gen = _gen()
    r._write_train_complete_marker(out, contract, gen,
                                   corpus_digest="c" * 12, gpu_name="")
    # different contract → stale product of some other terms
    other = dc_replace(contract, base_lr=contract.base_lr * 0.5)
    assert r._reusable_checkpoint(out, other, gen) is None
    # different generator → not this job's checkpoint
    other_gen = _gen(ref="hippius:cascade/gen-b@sha256:" + "d" * 64)
    assert r._reusable_checkpoint(out, contract, other_gen) is None


def test_corrupt_marker_means_retrain(cfg, tmp_path):
    r = _runner(cfg, tmp_path)
    contract = cfg.training.primary_size
    out = tmp_path / "checkpoint"
    out.mkdir()
    (out / "weights.safetensors").write_bytes(b"w")
    (out / TrainerRunner.TRAIN_COMPLETE_MARKER).write_text("{not json")
    assert r._reusable_checkpoint(out, contract, _gen()) is None


# ── patient uploads ──────────────────────────────────────────────────────────


def test_upload_retry_window_outlasts_a_ten_minute_flap():
    attempts = {"n": 0}
    slept: list[float] = []

    def op():
        attempts["n"] += 1
        raise TimeoutError("operation timed out")

    with pytest.raises(hippius_mod.StorageError):
        hippius_mod._retry_hub_op(
            op, "upload of x", attempts=hippius_mod.UPLOAD_MAX_ATTEMPTS,
            base_delay=hippius_mod.UPLOAD_BACKOFF_BASE_S, sleep=slept.append)
    assert attempts["n"] == hippius_mod.UPLOAD_MAX_ATTEMPTS
    assert sum(slept) >= 600.0        # the window must span a ~10-min flap


def test_upload_dir_to_hub_uses_the_patient_window(monkeypatch, tmp_path):
    seen = {}

    def spy(op, what, **kw):
        seen.update(kw)
        raise hippius_mod.StorageError("stop here")

    monkeypatch.setattr(hippius_mod, "_retry_hub_op", spy)
    monkeypatch.setattr(hippius_mod, "_resolve_hub_token", lambda *_a, **_k: "t")
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "weights.safetensors").write_bytes(b"w")
    with pytest.raises(hippius_mod.StorageError):
        hippius_mod.upload_dir_to_hub(d, "cascade/x")
    assert seen.get("attempts") == hippius_mod.UPLOAD_MAX_ATTEMPTS
    assert seen.get("base_delay") == hippius_mod.UPLOAD_BACKOFF_BASE_S


# ── deployed-config push to image-boot pods ──────────────────────────────────


def test_config_push_scps_box_chain_toml_to_pod(monkeypatch):
    from types import SimpleNamespace

    from cascade.provision import main as pmain

    calls = {}

    def fake_run(argv, **kw):
        calls["argv"] = argv
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(pmain.subprocess, "run", fake_run)
    render = SimpleNamespace(
        chain_toml="/root/cascade/chain.toml", key_path="~/.ssh/k",
        profile_for=lambda p: SimpleNamespace(user="root", workdir="/root/cascade"),
    )
    push = pmain.make_config_push(render, box_chain_toml="/box/chain.toml",
                                  pod_user="root")
    push(SimpleNamespace(ip="1.2.3.4", ssh_port=2222), "heat", "shadeform")
    argv = calls["argv"]
    assert argv[0] == "scp" and "/box/chain.toml" in argv
    assert argv[-1] == "root@1.2.3.4:/root/cascade/chain.toml"


def test_config_push_failure_is_swallowed(monkeypatch):
    from types import SimpleNamespace

    from cascade.provision import main as pmain

    def fake_run(argv, **kw):
        raise OSError("scp missing")

    monkeypatch.setattr(pmain.subprocess, "run", fake_run)
    render = SimpleNamespace(
        chain_toml=None, key_path="~/.ssh/k",
        profile_for=lambda p: SimpleNamespace(user="root", workdir="/wd"),
    )
    push = pmain.make_config_push(render, box_chain_toml="/box/chain.toml",
                                  pod_user="root")
    hooks = pmain._compose_pod_hooks(push, None)
    # composed runner isolates the failure — no raise
    hooks(SimpleNamespace(ip="1.2.3.4", ssh_port=22), "final", "")


def test_compose_pod_hooks_runs_all_and_isolates_failures():
    from cascade.provision.main import _compose_pod_hooks

    seen = []

    def a(addr, stage, provider=""):
        seen.append("a")
        raise RuntimeError("boom")

    def b(addr, stage, provider=""):
        seen.append("b")

    run = _compose_pod_hooks(a, b)
    run(type("A", (), {"ip": "x"})(), "heat")
    assert seen == ["a", "b"]              # a's failure never eats b
    assert _compose_pod_hooks(None, None) is None
    assert _compose_pod_hooks(b, None) is b
