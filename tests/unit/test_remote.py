"""Remote two-device training dispatch — command construction, receipt parsing,
host assignment, and the parallel run_round path (all without real SSH/GPU)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from cascade.shared.chain import Commitment
from cascade.shared.manifest import format_trained_pointer
from cascade.trainer import remote as remote_mod
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import (
    RECEIPT_SENTINEL,
    RemoteDispatcher,
    RemoteDispatchError,
    RemoteHost,
    build_remote_command,
    build_ssh_argv,
    load_hosts,
    parse_receipt,
    worker_argv,
)

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64
REF_T = "cascade/ckpt-r1-king@sha256:" + "c" * 64


def _host(name="king-box", **kw):
    return RemoteHost(name=name, host="1.2.3.4", **kw)


def _receipt_dict(role="king", uid=0, hotkey="hk"):
    # distinct generators produce distinct corpora — a shared digest would (rightly)
    # trip the trainer's content-clone drop, which is not what these tests probe
    return {
        "miner_hotkey": hotkey, "miner_uid": uid, "role": role, "gen_ref": REF_A,
        "trained_pointer": format_trained_pointer(REF_T), "corpus_digest": f"d-{hotkey}",
        "train_block": 10,
    }


# ── pure command construction ────────────────────────────────────────────────


def test_worker_argv_has_required_flags():
    argv = worker_argv(
        _host(remote_python="/venv/python", chain_toml="/r/chain.toml"),
        gen_ref=REF_A, uid=3, hotkey="hkX", role="challenger",
        base_seed=99, block=12, trainer_spec="m:C",
    )
    assert argv[0] == "/venv/python" and argv[1:3] == ["-m", "cascade.trainer.worker"]
    for flag, val in [("--gen-ref", REF_A), ("--uid", "3"), ("--hotkey", "hkX"),
                      ("--role", "challenger"), ("--base-seed", "99"), ("--block", "12"),
                      ("--trainer", "m:C"), ("--chain-toml", "/r/chain.toml")]:
        assert val == argv[argv.index(flag) + 1]
    # no heat overrides by default (final training uses the full config budget/repo)
    assert "--train-hours" not in argv and "--repo-suffix" not in argv


def test_worker_argv_heat_overrides_budget_and_repo():
    # A heat dispatch pins the cheap budget + a per-challenger repo suffix so
    # concurrent same-size heat runs don't overwrite each other's checkpoint.
    argv = worker_argv(
        _host(remote_python="/venv/python"),
        gen_ref=REF_A, uid=5, hotkey="hkX", role="challenger",
        base_seed=99, block=12, trainer_spec="m:C",
        arch_preset="toto2-4m", train_hours=0.02, repo_suffix="-heat-u5",
    )
    assert float(argv[argv.index("--train-hours") + 1]) == 0.02
    # `=` form so the leading-dash suffix isn't parsed as a flag
    assert "--repo-suffix=-heat-u5" in argv
    assert argv[argv.index("--arch-preset") + 1] == "toto2-4m"


def test_worker_maps_train_hours_to_heat_label(cfg, tmp_path, monkeypatch):
    # The pod worker is where a remote heat actually trains + logs. --train-hours
    # (the cheap screen budget) is the sole heat/final discriminator, so it must
    # flag heat=True into train_one — that's what routes the run's S3/wandb
    # telemetry to heat-<hotkey> instead of the final's <role>-<size> key.
    from cascade.shared.manifest import TrainedEntry
    from cascade.trainer import worker as worker_mod

    monkeypatch.setattr(worker_mod, "load_chain_config", lambda p: cfg)
    monkeypatch.setattr(worker_mod, "_load_trainer", lambda spec: object())
    captured: dict = {}

    def _fake_train_one(self, gen, role, seeds, block, *, contract=None,
                        token_budget=None, repo_suffix="", heat=False):
        captured["heat"] = heat
        return TrainedEntry(
            miner_hotkey=gen.hotkey, miner_uid=gen.uid, role=role, gen_ref=gen.ref,
            trained_pointer=format_trained_pointer(REF_T), corpus_digest="d",
            train_block=block, gpu_name="", size=(contract.arch_preset if contract else ""),
        )

    monkeypatch.setattr(TrainerRunner, "train_one", _fake_train_one)
    base = ["--gen-ref", REF_A, "--uid", "5", "--hotkey", "hkB", "--role", "challenger",
            "--base-seed", "1", "--block", "10", "--trainer", "m:C",
            "--arch-preset", "toto2-4m", "--work-root", str(tmp_path)]

    assert worker_mod.main(base + ["--train-hours", "0.02"]) == 0
    assert captured["heat"] is True           # heat dispatch ⇒ heat-<hotkey> label
    assert worker_mod.main(base) == 0         # no --train-hours ⇒ final
    assert captured["heat"] is False


def test_build_remote_command_sets_cd_cuda_and_env():
    host = _host(workdir="/root/metro", cuda_device="1")
    cmd = build_remote_command(host, ["python", "-m", "x"], {"HIPPIUS_S3_ACCESS_KEY": "ak"})
    # training preempts any straggling log-only benchmark sweep before cd'ing in
    from cascade.trainer.remote import PREEMPT_BENCHMARKS

    assert cmd.startswith(PREEMPT_BENCHMARKS)
    assert "cd /root/metro && " in cmd
    assert "CUDA_VISIBLE_DEVICES=1" in cmd
    assert "HIPPIUS_S3_ACCESS_KEY=ak" in cmd
    assert cmd.rstrip().endswith("python -m x")


def _capturing_runner(seen: dict):
    """A dispatcher _runner that records the remote command and returns a receipt."""
    def _run(argv, timeout):
        seen["cmd"] = argv[-1]
        return _fake_proc(stdout=f"{RECEIPT_SENTINEL}{json.dumps(_receipt_dict(role='king'))}")
    return _run


def test_dispatch_forwards_extra_env_on_top_of_host(monkeypatch):
    # extra_forward_env (e.g. WANDB_API_KEY when [wandb] is on) reaches the pod
    # without the operator listing it per-host — so pod-side wandb actually logs.
    monkeypatch.setenv("WANDB_API_KEY", "wbkey")
    monkeypatch.setenv("HIPPIUS_HUB_TOKEN", "tok")
    seen: dict = {}
    disp = RemoteDispatcher(trainer_spec="m:C", extra_forward_env=("WANDB_API_KEY",),
                            _runner=_capturing_runner(seen))
    disp.dispatch(_host(forward_env=("HIPPIUS_HUB_TOKEN",)), gen_ref=REF_A, uid=0,
                  hotkey="hk", role="king", base_seed=1, block=10)
    assert "WANDB_API_KEY=wbkey" in seen["cmd"]       # auto-forwarded extra
    assert "HIPPIUS_HUB_TOKEN=tok" in seen["cmd"]      # host's own forward_env kept


def test_dispatch_omits_extra_env_when_absent_from_orchestrator(monkeypatch):
    # Only forwarded if actually present — absent ⇒ no stray/empty assignment.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    seen: dict = {}
    disp = RemoteDispatcher(trainer_spec="m:C", extra_forward_env=("WANDB_API_KEY",),
                            _runner=_capturing_runner(seen))
    disp.dispatch(_host(), gen_ref=REF_A, uid=0, hotkey="hk", role="king",
                  base_seed=1, block=10)
    assert "WANDB_API_KEY" not in seen["cmd"]


def test_trainer_forwards_wandb_key_only_when_enabled(cfg):
    from dataclasses import replace

    from cascade.shared.config import WandbConfig

    on = TrainerRunner(cfg=replace(cfg, wandb=WandbConfig(enabled=True)),
                       base_trainer=None, work_root=Path("."))
    off = TrainerRunner(cfg=replace(cfg, wandb=WandbConfig(enabled=False)),
                        base_trainer=None, work_root=Path("."))
    assert on._pod_extra_forward_env() == ("WANDB_API_KEY",)
    assert off._pod_extra_forward_env() == ()


def test_preempt_pattern_matches_only_the_live_scorer():
    """The pkill regex must hit a running cascade-benchmark and nothing else —
    in particular NOT cascade-benchmark-download (a multi-hour data pull that
    every dispatch would otherwise kill), the launch shell, or itself."""
    import re

    from cascade.trainer.remote import PREEMPT_BENCHMARKS

    pattern = PREEMPT_BENCHMARKS.split("'")[1]
    scorer = "/root/x/.venv/bin/cascade-benchmark /ckpt /out.json --suites gift-eval"
    scorer_py = "/root/x/.venv/bin/python /root/x/.venv/bin/cascade-benchmark /ckpt /out"
    downloader = "/root/x/.venv/bin/cascade-benchmark-download --data-dir /root/bench_data"
    launch_shell = "uv run --project /root/x/benchmarks cascade-benchmark /ckpt /out"
    assert re.search(pattern, scorer)
    assert re.search(pattern, scorer_py)
    assert not re.search(pattern, downloader)
    assert not re.search(pattern, launch_shell)
    assert not re.search(pattern, PREEMPT_BENCHMARKS)  # bracket trick: no self-match


def test_build_ssh_argv_includes_key_port_dest():
    host = _host(port=2222, user="me", key_path="/k", ssh_options=("ServerAliveInterval=30",))
    argv = build_ssh_argv(host, "cd x && run")
    assert argv[0] == "ssh"
    assert "-p" in argv and "2222" in argv
    assert "-i" in argv
    assert "me@1.2.3.4" in argv
    assert argv[-1] == "cd x && run"
    assert "ServerAliveInterval=30" in argv


# ── receipt parsing ──────────────────────────────────────────────────────────


def test_parse_receipt_extracts_json_after_sentinel():
    stdout = f"loading cuda...\nbanner\n{RECEIPT_SENTINEL}{json.dumps(_receipt_dict())}\n"
    got = parse_receipt(stdout)
    assert got["role"] == "king" and got["gen_ref"] == REF_A


def test_parse_receipt_raises_without_sentinel():
    with pytest.raises(RemoteDispatchError):
        parse_receipt("no receipt here\n")


# ── dispatcher (injected runner, no real ssh) ────────────────────────────────


def _fake_proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def test_dispatch_returns_entry_on_success():
    out = f"{RECEIPT_SENTINEL}{json.dumps(_receipt_dict(role='king'))}"
    disp = RemoteDispatcher(trainer_spec="m:C", _runner=lambda argv, t: _fake_proc(stdout=out))
    entry = disp.dispatch(_host(), gen_ref=REF_A, uid=0, hotkey="hk", role="king",
                          base_seed=1, block=10)
    assert entry.role == "king" and entry.trained_pointer == format_trained_pointer(REF_T)


def test_dispatch_raises_on_nonzero_rc():
    disp = RemoteDispatcher(trainer_spec="m:C",
                            _runner=lambda argv, t: _fake_proc(rc=1, stderr="boom"))
    with pytest.raises(RemoteDispatchError):
        disp.dispatch(_host(), gen_ref=REF_A, uid=0, hotkey="hk", role="king",
                      base_seed=1, block=10)


def test_dispatch_transport_failure_carries_returncode_255():
    # rc=255 is an SSH transport failure — the heat fan-out counts it to refuse
    # caching a fleet-wide dead-pod wipeout, so the error must carry the code.
    disp = RemoteDispatcher(trainer_spec="m:C",
                            _runner=lambda argv, t: _fake_proc(rc=255, stderr="ssh: connect"))
    with pytest.raises(RemoteDispatchError) as ei:
        disp.dispatch(_host(), gen_ref=REF_A, uid=0, hotkey="hk", role="king",
                      base_seed=1, block=10)
    assert ei.value.returncode == 255


def test_probe_host_true_only_on_ssh_echo_roundtrip(monkeypatch):
    # TCP reachability is not enough — only a full SSH echo round-trip proves a
    # pod can run a worker (a booting pod accepts TCP but sshd dies rc=255).
    monkeypatch.setattr(remote_mod, "run_ssh",
                        lambda argv, t: _fake_proc(rc=0, stdout=remote_mod._HEAT_PROBE_TOKEN + "\n"))
    assert remote_mod.probe_host(_host()) is True
    # rc=255 transport failure ⇒ dead
    monkeypatch.setattr(remote_mod, "run_ssh",
                        lambda argv, t: _fake_proc(rc=255, stderr="connection refused"))
    assert remote_mod.probe_host(_host()) is False
    # rc=0 but no token (sshd banner / MOTD only) ⇒ not proven alive
    monkeypatch.setattr(remote_mod, "run_ssh", lambda argv, t: _fake_proc(rc=0, stdout="motd"))
    assert remote_mod.probe_host(_host()) is False
    # a transport exception (no route) ⇒ dead
    def _boom(argv, t):
        raise OSError("no route to host")

    monkeypatch.setattr(remote_mod, "run_ssh", _boom)
    assert remote_mod.probe_host(_host()) is False
    # an un-addressable host can't be probed ⇒ kept (never strand a fleet)
    assert remote_mod.probe_host(types.SimpleNamespace(name="no-addr")) is True


def test_dispatch_rc3_relays_one_line_miner_rejection():
    """Incident 2026-07-14: miners with PRIVATE Hippius repos surfaced as
    15-line infra tracebacks. Worker rc=3 = miner fault; only the final stderr
    line (the one-line reason) travels to the orchestrator log."""
    stderr = ("Traceback (most recent call last):\n  File ...\n"
              "ERROR cascade.trainer.worker: miner submission rejected: "
              "generator_artifact_unreachable: HTTP 401 for ns/repo@sha256:aa\n")
    disp = RemoteDispatcher(trainer_spec="m:C",
                            _runner=lambda argv, t: _fake_proc(rc=3, stderr=stderr))
    with pytest.raises(RemoteDispatchError) as ei:
        disp.dispatch(_host(), gen_ref=REF_A, uid=0, hotkey="hk", role="challenger",
                      base_seed=1, block=10)
    msg = str(ei.value)
    assert "miner submission rejected" in msg
    assert "generator_artifact_unreachable: HTTP 401" in msg
    assert "Traceback" not in msg


def test_dispatch_rejects_role_mismatch():
    out = f"{RECEIPT_SENTINEL}{json.dumps(_receipt_dict(role='challenger'))}"
    disp = RemoteDispatcher(trainer_spec="m:C", _runner=lambda argv, t: _fake_proc(stdout=out))
    with pytest.raises(RemoteDispatchError):
        disp.dispatch(_host(), gen_ref=REF_A, uid=0, hotkey="hk", role="king",
                      base_seed=1, block=10)


# ── load_hosts ───────────────────────────────────────────────────────────────


def test_load_hosts_parses_toml(tmp_path):
    p = tmp_path / "hosts.toml"
    p.write_text(
        '[[host]]\nname="king"\nhost="10.0.0.1"\ncuda_device="0"\n'
        'forward_env=["HIPPIUS_HUB_TOKEN"]\n\n'
        '[[host]]\nname="chal"\nhost="10.0.0.2"\nport=2200\n',
        encoding="utf-8",
    )
    hosts = load_hosts(p)
    assert [h.name for h in hosts] == ["king", "chal"]
    assert hosts[0].cuda_device == "0" and hosts[0].forward_env == ("HIPPIUS_HUB_TOKEN",)
    assert hosts[1].port == 2200


# ── parallel run_round (fake dispatcher) ─────────────────────────────────────


class _RecordingDispatcher:
    calls: list = []

    def __init__(self, *a, **k):
        pass

    def dispatch(self, host, *, gen_ref, uid, hotkey, role, base_seed, block,
                 arch_preset=None, warm_start_ref=None, lane_count=None):
        _RecordingDispatcher.calls.append((host.name, role, gen_ref))
        return remote_mod.receipt_to_entry(_receipt_dict(role=role, uid=uid, hotkey=hotkey))


def test_run_round_remote_assigns_king_and_challenger_to_separate_hosts(cfg, tmp_path, monkeypatch):
    _RecordingDispatcher.calls = []
    monkeypatch.setattr(remote_mod, "RemoteDispatcher", _RecordingDispatcher)
    # contract_digest/base_arch_digest come from cfg; entries just need to assemble.

    hosts = [_host(name="host0"), _host(name="host1")]
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                           remote_hosts=hosts, trainer_spec="m:C")

    commits = [
        Commitment(uid=0, hotkey="a", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_A}", commit_block=5),
        Commitment(uid=1, hotkey="b", coldkey=None, payload=f"metro-v1:gen:hippius:{REF_B}", commit_block=6),
    ]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    assert manifest.entry_for_role("king") is not None
    assert manifest.entry_for_role("challenger") is not None
    by_role = {role: name for (name, role, _ref) in _RecordingDispatcher.calls}
    assert by_role["king"] == "host0" and by_role["challenger"] == "host1"


def test_run_round_remote_aborts_when_king_fails(cfg, tmp_path, monkeypatch):
    class _KingFails(_RecordingDispatcher):
        def dispatch(self, host, *, role, **kw):
            if role == "king":
                raise RemoteDispatchError("king pod died")
            return remote_mod.receipt_to_entry(_receipt_dict(role=role, **{k: kw[k] for k in ("uid", "hotkey")}))

    monkeypatch.setattr(remote_mod, "RemoteDispatcher", _KingFails)
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                           remote_hosts=[_host(name="h0")], trainer_spec="m:C")
    commits = [Commitment(uid=0, hotkey="a", coldkey=None,
                          payload=f"metro-v1:gen:hippius:{REF_A}", commit_block=5)]
    with pytest.raises(RuntimeError):
        runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)


def test_load_hosts_parses_and_validates_stage(tmp_path):
    p = tmp_path / "hosts.toml"
    p.write_text(
        '[[host]]\nname="cheap"\nhost="10.0.0.1"\nstage="heat"\n\n'
        '[[host]]\nname="big"\nhost="10.0.0.2"\nstage="final"\n\n'
        '[[host]]\nname="both"\nhost="10.0.0.3"\n',
        encoding="utf-8",
    )
    hosts = load_hosts(p)
    assert [h.stage for h in hosts] == ["heat", "final", "any"]

    p.write_text('[[host]]\nname="x"\nhost="10.0.0.9"\nstage="warmup"\n', encoding="utf-8")
    with pytest.raises(RemoteDispatchError, match="stage"):
        load_hosts(p)
