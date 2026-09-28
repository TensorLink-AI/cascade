"""Pods carry no credentials (2026-09-28 hardening, after the 2026-09-24 leak route).

A miner is root on a payer pod; an operator lane that loses its network
namespace runs the generator as a plain child. Either can read whatever the
orchestrator forwards into the worker environment. Pinned:

* the provisioner forwards nothing by default and marks rendered lanes
  ``isolated = true`` (the orchestrator harvests, PR #320);
* a hosts.toml entry that forwards a secret WITHOUT ``isolated`` logs a WARNING
  naming the secret;
* the deploy templates forward nothing; testnet runs ``sandbox_strict = true``;
* the trainer entry point reads the strict flag from ``[generator]``.
"""
from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from cascade.provision.core import DEFAULT_FORWARD_ENV, render_hosts_toml
from cascade.shared.config import load_chain_config
from cascade.trainer.remote import load_hosts

ROOT = Path(__file__).resolve().parents[2]


def test_default_forward_env_is_empty():
    assert DEFAULT_FORWARD_ENV == ()


def _render(forward_env):
    from cascade.provision.core import PodAddress

    return render_hosts_toml(
        [PodAddress("10.0.0.1", 22)], gpus_per_pod=1, user="root", key_path="/k",
        remote_python="/p", workdir="/w", stage="final", chain_toml="",
        forward_env=forward_env, ssh_options=("StrictHostKeyChecking=accept-new",),
        name_prefix="lane",
    )


def test_rendered_lane_is_isolated_unless_it_forwards_something():
    iso = tomllib.loads(_render(()))["host"][0]
    assert iso["forward_env"] == [] and iso["isolated"] is True
    att = tomllib.loads(_render(("HIPPIUS_S3_ACCESS_KEY",)))["host"][0]
    assert att["isolated"] is False


def test_load_hosts_warns_when_a_secret_is_forwarded_unisolated(tmp_path, caplog):
    text = (
        '[[host]]\nname = "a"\nhost = "10.0.0.1"\nremote_python = "p"\nworkdir = "/w"\n'
        'forward_env = ["HIPPIUS_S3_SECRET_KEY", "WANDB_API_KEY", "SOME_FLAG"]\n'
        '[[host]]\nname = "b"\nhost = "10.0.0.2"\nremote_python = "p"\nworkdir = "/w"\n'
        'forward_env = ["HIPPIUS_S3_SECRET_KEY"]\nisolated = true\n'
        '[[host]]\nname = "c"\nhost = "10.0.0.3"\nremote_python = "p"\nworkdir = "/w"\n'
        'forward_env = ["SOME_FLAG"]\n'
    )
    p = tmp_path / "hosts.toml"
    p.write_text(text)
    with caplog.at_level(logging.WARNING, logger="cascade.trainer.remote"):
        hosts = load_hosts(p)
    assert [h.name for h in hosts] == ["a", "b", "c"]
    warns = [r.getMessage() for r in caplog.records if "forwards credential" in r.getMessage()]
    assert len(warns) == 1 and "host a" in warns[0]
    assert "HIPPIUS_S3_SECRET_KEY" in warns[0] and "WANDB_API_KEY" in warns[0] and "SOME_FLAG" not in warns[0]


def test_deploy_templates_forward_nothing():
    for name in ("deploy/provision.mainnet.toml", "deploy/provision.testnet.toml"):
        text = (ROOT / name).read_text()
        assert "HIPPIUS_S3_ACCESS_KEY" not in text, name
        assert "forward_env   = []" in text, name


def test_testnet_runs_strict_sandbox_and_flag_lives_in_generator():
    cfg = load_chain_config(ROOT / "chain.testnet.toml")
    assert cfg.generator.sandbox_strict is True
    assert hasattr(load_chain_config(ROOT / "chain.toml").generator, "sandbox_strict")
