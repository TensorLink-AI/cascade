"""Remote two-device training dispatch — command construction, receipt parsing,
host assignment, and the parallel run_round path (all without real SSH/GPU)."""

from __future__ import annotations

import json
import types

import pytest

from metronome.shared.chain import Commitment
from metronome.shared.manifest import format_trained_pointer
from metronome.trainer import remote as remote_mod
from metronome.trainer.loop import TrainerRunner
from metronome.trainer.remote import (
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
REF_T = "metronome/ckpt-r1-king@sha256:" + "c" * 64


def _host(name="king-box", **kw):
    return RemoteHost(name=name, host="1.2.3.4", **kw)


def _receipt_dict(role="king", uid=0, hotkey="hk"):
    return {
        "miner_hotkey": hotkey, "miner_uid": uid, "role": role, "gen_ref": REF_A,
        "trained_pointer": format_trained_pointer(REF_T), "corpus_digest": "d",
        "train_block": 10,
    }


# ── pure command construction ────────────────────────────────────────────────


def test_worker_argv_has_required_flags():
    argv = worker_argv(
        _host(remote_python="/venv/python", chain_toml="/r/chain.toml"),
        gen_ref=REF_A, uid=3, hotkey="hkX", role="challenger",
        base_seed=99, block=12, trainer_spec="m:C",
    )
    assert argv[0] == "/venv/python" and argv[1:3] == ["-m", "metronome.trainer.worker"]
    for flag, val in [("--gen-ref", REF_A), ("--uid", "3"), ("--hotkey", "hkX"),
                      ("--role", "challenger"), ("--base-seed", "99"), ("--block", "12"),
                      ("--trainer", "m:C"), ("--chain-toml", "/r/chain.toml")]:
        assert val == argv[argv.index(flag) + 1]


def test_build_remote_command_sets_cd_cuda_and_env():
    host = _host(workdir="/root/metro", cuda_device="1")
    cmd = build_remote_command(host, ["python", "-m", "x"], {"HIPPIUS_S3_ACCESS_KEY": "ak"})
    assert cmd.startswith("cd /root/metro && ")
    assert "CUDA_VISIBLE_DEVICES=1" in cmd
    assert "HIPPIUS_S3_ACCESS_KEY=ak" in cmd
    assert cmd.rstrip().endswith("python -m x")


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

    def dispatch(self, host, *, gen_ref, uid, hotkey, role, base_seed, block):
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
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10, max_challengers=1)

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
