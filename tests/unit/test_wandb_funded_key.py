"""Funded (isolated) pods get a SEPARATE, project-scoped wandb key ([wandb]
funded_key_env) as WANDB_API_KEY — never the operator's own key or any other
orchestrator env (2026-09-13: owner wants live wandb for miner-paid legs)."""
from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

from cascade.shared.config import WandbConfig, load_chain_config
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteDispatcher, RemoteHost

REF = "vault/direct@sha256:" + "a" * 40


def _dispatch(monkeypatch, *, isolated: bool, isolated_forward=(("WANDB_API_KEY", "WANDB_API_KEY_FUNDED"),)):
    monkeypatch.setenv("HIPPIUS_S3_ACCESS_KEY", "op-s3")
    monkeypatch.setenv("WANDB_API_KEY", "op-wandb")
    monkeypatch.setenv("WANDB_API_KEY_FUNDED", "funded-wandb")
    seen = {}

    def runner(argv, timeout, stdin_env=None):
        seen["stdin"] = stdin_env or ""
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "miner_hotkey": "hkA", "role": "challenger"}), stderr="")

    disp = RemoteDispatcher(trainer_spec="m:C", extra_forward_env=("WANDB_API_KEY",),
                            isolated_forward_env=isolated_forward, _runner=runner)
    host = RemoteHost(name="x", host="10.0.0.9", port=22, user="root", key_path=None,
                      remote_python="python", workdir="/w", cuda_device="0",
                      chain_toml="chain.toml", forward_env=("HIPPIUS_S3_ACCESS_KEY",),
                      static_env=(("HIPPIUS_HUB_USERNAME", "robot$x"),), isolated=isolated)
    with contextlib.suppress(Exception):  # the parsed entry shape is not under test
        # an isolated host is dispatched --local-only (the payer path asks for
        # it; the dispatcher forces it otherwise) — the env it receives is the
        # same either way, which is what these tests pin down.
        disp.dispatch(host, lane_count=1, gen_ref=REF, uid=1, hotkey="hkA", role="challenger",
                      base_seed=1, block=1, arch_preset="toto2-4m", warm_start_ref=None,
                      local_checkpoint=isolated)
    return seen["stdin"]


def test_isolated_host_gets_the_funded_key_as_wandb_api_key_and_nothing_else(monkeypatch):
    stdin = _dispatch(monkeypatch, isolated=True)
    assert "WANDB_API_KEY=funded-wandb" in stdin or "WANDB_API_KEY='funded-wandb'" in stdin
    assert "op-wandb" not in stdin and "op-s3" not in stdin
    assert "WANDB_API_KEY_FUNDED" not in stdin            # delivered under the wandb name
    assert "robot$x" in stdin                             # static_env still travels


def test_operator_host_never_receives_the_funded_key(monkeypatch):
    stdin = _dispatch(monkeypatch, isolated=False)
    assert "op-wandb" in stdin and "op-s3" in stdin and "funded-wandb" not in stdin


def test_isolated_host_without_a_funded_key_keeps_no_wandb(monkeypatch):
    stdin = _dispatch(monkeypatch, isolated=True, isolated_forward=())
    assert "wandb" not in stdin.lower()


def _runner_with(wandb: WandbConfig):
    fake = SimpleNamespace(cfg=SimpleNamespace(wandb=wandb))
    fake._pod_isolated_forward_env = TrainerRunner._pod_isolated_forward_env.__get__(fake)
    fake._pod_extra_forward_env = TrainerRunner._pod_extra_forward_env.__get__(fake)
    return fake


def test_trainer_forwards_the_named_key_only_when_enabled_and_never_the_operators():
    r = _runner_with(WandbConfig(enabled=True, funded_key_env="WANDB_API_KEY_FUNDED"))
    assert r._pod_isolated_forward_env() == (("WANDB_API_KEY", "WANDB_API_KEY_FUNDED"),)
    assert r._pod_extra_forward_env() == ("WANDB_API_KEY",)
    assert _runner_with(WandbConfig(enabled=False, funded_key_env="WANDB_API_KEY_FUNDED")) \
        ._pod_isolated_forward_env() == ()
    assert _runner_with(WandbConfig(enabled=True, funded_key_env="")) \
        ._pod_isolated_forward_env() == ()
    # The operator's own key name is refused, not forwarded.
    assert _runner_with(WandbConfig(enabled=True, funded_key_env="WANDB_API_KEY")) \
        ._pod_isolated_forward_env() == ()


def test_funded_key_env_round_trips_through_the_loader(tmp_path):
    from pathlib import Path

    base = Path("chain.toml").read_text(encoding="utf-8")
    assert 'funded_key_env = "WANDB_API_KEY_FUNDED"' in base
    assert load_chain_config("chain.toml").wandb.funded_key_env == "WANDB_API_KEY_FUNDED"
    src = tmp_path / "chain.toml"
    src.write_text(base.replace('funded_key_env = "WANDB_API_KEY_FUNDED"', 'funded_key_env = " X "'),
                   encoding="utf-8")
    assert load_chain_config(str(src)).wandb.funded_key_env == "X"
    src.write_text(base.replace('funded_key_env = "WANDB_API_KEY_FUNDED"\n', ""), encoding="utf-8")
    assert load_chain_config(str(src)).wandb.funded_key_env == ""     # section key optional
