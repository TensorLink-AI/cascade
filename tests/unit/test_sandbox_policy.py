"""Generator code never runs on the orchestrator / a validator without a
hardened sandbox: the load check and the runtime guard."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.sandbox_policy import (
    SandboxPolicyError,
    check_generator_sandbox_config,
    clear_holds_secrets,
    holds_secrets,
    mark_holds_secrets,
    require_generator_sandbox,
    sandbox_hardened,
)

ROOT = Path(__file__).resolve().parents[2]


def test_load_check_refuses_an_armed_probe_on_a_weak_sandbox(tmp_path):
    weak = {"sandbox_mode": "subprocess", "sandbox_strict": False}
    check_generator_sandbox_config(weak, {"dedup_probe_series": 0})
    with pytest.raises(ValueError, match="dedup_probe_series"):
        check_generator_sandbox_config(weak, {"dedup_probe_series": 8})
    check_generator_sandbox_config({**weak, "sandbox_strict": True}, {"dedup_probe_series": 8})
    check_generator_sandbox_config({"sandbox_mode": "container"}, {"dedup_probe_series": 8})
    # A weak deployment with the probe armed is refused by the loader too.
    assert load_chain_config(ROOT / "chain.toml").generator.sandbox_strict
    text = (ROOT / "chain.toml").read_text().replace(
        "dedup_probe_series = 0", "dedup_probe_series = 4", 1).replace(
        "sandbox_strict = true", "sandbox_strict = false", 1)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    with pytest.raises(ValueError, match="dedup_probe_series"):
        load_chain_config(p)
    load_chain_config(ROOT / "chain.toml")
    load_chain_config(ROOT / "chain.testnet.toml")


def test_runtime_guard_is_inert_in_an_unmarked_process(cfg):
    assert holds_secrets() == ""
    require_generator_sandbox(cfg.generator, "x")
    require_generator_sandbox(None, "x", in_process=True)


def test_runtime_guard_refuses_in_a_process_holding_secrets(cfg):
    weak = replace(cfg.generator, sandbox_mode="subprocess", sandbox_strict=False)
    mark_holds_secrets("the trainer wallet")
    try:
        with pytest.raises(SandboxPolicyError, match="not hardened"):
            require_generator_sandbox(weak, "stream_series")
        require_generator_sandbox(replace(weak, sandbox_strict=True), "stream_series")
        require_generator_sandbox(replace(weak, sandbox_mode="container"), "stream_series")
        with pytest.raises(SandboxPolicyError, match="in-process"):
            require_generator_sandbox(replace(weak, sandbox_strict=True), "x", in_process=True)
    finally:
        clear_holds_secrets()
    assert sandbox_hardened({"sandbox_strict": True}) and not sandbox_hardened({})


def test_every_generator_entry_point_is_guarded(cfg, tmp_path):
    from cascade.trainer import sandbox
    from cascade.trainer.corpus import CorpusError, _load_generator

    weak = replace(cfg.generator, sandbox_mode="subprocess", sandbox_strict=False)
    mark_holds_secrets("the private eval pool")
    try:
        with pytest.raises(SandboxPolicyError):
            sandbox.run_in_sandbox(tmp_path, 1, weak)
        with pytest.raises(SandboxPolicyError), sandbox.stream_series(tmp_path, 1, weak, 1000):
            pass
        with pytest.raises(SandboxPolicyError):
            _load_generator(tmp_path, 1)
    finally:
        clear_holds_secrets()
    # unmarked (a pod, a miner tool): the same call reaches its own checks
    with pytest.raises(CorpusError, match="missing generator.py"):
        _load_generator(tmp_path, 1)
    # the refusal is never a CorpusError (never booked as the miner's fault)
    assert not issubclass(SandboxPolicyError, CorpusError)


def test_mains_mark_the_process():
    import inspect

    from cascade.trainer import main as tmain
    from cascade.validator import main as vmain

    assert "mark_holds_secrets(" in inspect.getsource(tmain.main)
    assert "mark_holds_secrets(" in inspect.getsource(vmain.main)
    worker = (ROOT / "cascade" / "trainer" / "worker.py").read_text()
    assert "mark_holds_secrets" not in worker            # pods are never marked


def test_pod_without_isolation_is_an_infra_fault_not_a_miner_rejection():
    from cascade.trainer import worker
    from cascade.trainer.corpus import CorpusError

    refused = CorpusError("sandbox_isolation_unavailable: network namespaces unavailable")
    assert worker._pod_fault(refused)
    assert not worker._pod_fault(CorpusError("generator_output_rejected: 4 channels"))
