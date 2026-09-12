"""JIT-rented pods get the DEPLOYED chain.toml before dispatch (2026-09-12 08:58).

The worker image bakes a build-time chain.toml whose [training] train_image_digest
is the PRE-release pin; a pod the trainer rents itself (JIT king, per-payer funded
leg) never sees the provisioner's post-gate config push, so it ran on the baked copy
and refused the final ("runtime image 9d5bc70d… != pinned train_image_digest
3c373d5b…"). The trainer now pushes its deployed chain.toml to <workdir>/
chain.deployed.toml on every rented pod and dispatches with --chain-toml on it.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteHost


def _fake(tmp_path, deployed):
    fake = SimpleNamespace(deployed_chain_toml=deployed, _scp_calls=[])
    fake._scp_runner = lambda argv: fake._scp_calls.append(list(argv))
    fake._push_deployed_chain_toml = TrainerRunner._push_deployed_chain_toml.__get__(fake)
    return fake


def test_push_copies_the_deployed_file_and_pins_the_host_to_it(tmp_path):
    src = tmp_path / "chain.toml"
    src.write_text("[training]\n", encoding="utf-8")
    fake = _fake(tmp_path, src)
    host = RemoteHost(name="funded-king", host="10.0.0.7", port=20016, user="root",
                      key_path="~/.ssh/k", workdir="/root/cascade", chain_toml=None)
    out = fake._push_deployed_chain_toml(host)
    assert out.chain_toml == "/root/cascade/chain.deployed.toml"
    assert out.host == host.host and out.port == host.port          # same pod
    mkdir, scp = fake._scp_calls
    assert mkdir[0] == "ssh" and "mkdir -p /root/cascade" in mkdir[-1]
    assert scp[0] == "scp" and scp[-2] == str(src)
    assert scp[-1] == "root@10.0.0.7:/root/cascade/chain.deployed.toml"
    assert "-P" in scp and scp[scp.index("-P") + 1] == "20016"      # scp port form


def test_push_is_a_no_op_without_a_deployed_path(tmp_path):
    fake = _fake(tmp_path, None)
    host = RemoteHost(name="funded-king", host="10.0.0.7", chain_toml="/x/baked.toml")
    assert fake._push_deployed_chain_toml(host) is host
    assert fake._scp_calls == []


def test_push_failure_raises_so_the_leg_is_infra_not_a_silent_baked_run(tmp_path):
    src = tmp_path / "chain.toml"
    src.write_text("[training]\n", encoding="utf-8")
    fake = _fake(tmp_path, src)

    def boom(argv):
        raise RuntimeError("scp: connection refused")

    fake._scp_runner = boom
    host = RemoteHost(name="funded-x", host="10.0.0.8", workdir="/root/cascade")
    with pytest.raises(RuntimeError, match="connection refused"):
        fake._push_deployed_chain_toml(host)


def test_main_wires_the_deployed_chain_toml(monkeypatch):
    # The service passes its --chain-toml (default chain.toml) through, so the
    # push is armed on every live trainer, never just in tests.
    import inspect

    from cascade.trainer import main as main_mod

    src = inspect.getsource(main_mod)
    assert "deployed_chain_toml=(args.chain_toml or Path(\"chain.toml\"))" in src
