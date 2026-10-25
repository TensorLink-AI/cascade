"""Reproductions of failures across process, payment and policy boundaries."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.funding.vault import PayerKeyVault
from cascade.miner.output import NETWORK, Reporter
from cascade.miner.training_context import with_contract
from cascade.miner.verify import verify_repo
from cascade.provision.core import executor_price_per_hour
from cascade.provision.funded import funded_pod_name, reconcile_funded, teardown_funded
from cascade.provision.state import PodInstance
from cascade.shared.config import load_chain_config
from cascade.shared.verdict_numbers import same_verdict
from cascade.trainer.corpus import CorpusError
from cascade.trainer.sandbox import _frame_iter
from cascade.validator.weights import WeightsMixin

from .test_payer_providers import HK, pool  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


def _repo(path, shape):
    path.mkdir()
    (path / "requirements.txt").write_text("")
    (path / "config.json").write_text('{"name":"boundary-fixture"}')
    (path / "generator.py").write_text(
        'import numpy as np\nfrom cascade.interface.generator import DataGenerator\n'
        'class Generator(DataGenerator):\n'
        '    def __init__(self, config_dir, *, seed): self.seed = seed\n'
        '    @property\n    def name(self): return "boundary-fixture"\n'
        '    def generate(self, n_series):\n'
        '        rng = np.random.default_rng(self.seed)\n'
        f'        for _ in range(n_series): yield rng.normal(size={shape})\n')
    return path


def test_cli_cross_thread_arrow_write_and_main_thread_read(tmp_path):
    # Run the real CLI in a fresh interpreter; an allocator segfault must be
    # observed as a failed process rather than taking down pytest.
    repo = _repo(tmp_path / "repo", "(8, 2048)")
    text = (ROOT / "chain.toml").read_text()
    start = text.index("[generation]")
    end = text.index("\n[", start + 1)
    section = text[start:end]
    import re

    for name, value in {"vcpus": 4, "children": 4, "target_points": 524288,
                        "shard_points": 32768, "window_seconds": 30}.items():
        section, n = re.subn(rf"(?m)^{name}\s*=.*$", f"{name} = {value}", section)
        if not n:
            section += f"\n{name} = {value}\n"
    config = tmp_path / "chain.toml"
    config.write_text(text[:start] + section + text[end:])
    env = dict(os.environ)
    env.pop("ARROW_DEFAULT_MEMORY_POOL", None)
    for i in range(3):
        run = subprocess.run([sys.executable, "-m", "cascade.generation.cli", str(repo),
                              str(tmp_path / f"out-{i}"), "--seed", str(i), "--no-sandbox",
                              "--chain-toml", str(config), "--json"],
                             capture_output=True, text=True, timeout=60, env=env)
        assert run.returncode == 0, run.stderr
        assert json.loads(run.stdout)["ok"]


def test_stream_eof_waits_and_reports_the_child_failure():
    proc = subprocess.Popen([sys.executable, "-c",
                             "import sys; sys.stderr.write('MemoryError fixture'); sys.exit(7)"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with pytest.raises(CorpusError, match="rc=7.*MemoryError fixture"):
            list(_frame_iter(proc, 5))
    finally:
        proc.wait(timeout=5)
        proc.stdout.close()
        proc.stderr.close()


def test_lost_shadeform_create_survives_empty_sweep_and_retry(pool):  # noqa: F811
    accounts, vault, rent, _ = pool
    accounts["lium"].has_capacity = False
    shade = accounts["shadeform"]
    shade.fail_create = True
    actual_list = shade.list_tagged
    actual_named = shade.list_named
    shade.list_tagged = lambda _: []  # delayed provider visibility
    shade.list_named = lambda _: {}
    result = rent()
    assert not result.ok and result.leaked_pod and not result.burn_attempt
    from cascade.funding.providers import create_intents, lease_records

    token = next(token for token, _, _ in lease_records(vault, HK))
    scope = funded_pod_name("777", HK, 91)
    intent = PodInstance(provider="shadeform", instance_id=scope + "-0", stage="funded",
                         rented_at_iso="fixture", sku="H100", gpus=1,
                         payer_hotkey=HK, payer_credential_id=token)
    assert teardown_funded([intent], vault) == [intent]
    reconcile_funded([], vault, netuid=91)
    assert create_intents(vault, scope)
    accounts["lium"].has_capacity = True
    for _ in range(2):
        retried = rent()
        assert not retried.ok and not retried.burn_attempt
    assert sum(c[0] == "create" for a in accounts.values() for c in a.calls) == 1
    shade.list_tagged = actual_list
    shade.list_named = actual_named
    assert rent().adopted


def test_pending_create_intent_outlives_key_ttl_and_restart(pool):  # noqa: F811
    accounts, vault, rent, _ = pool
    accounts["lium"].fail_create = True
    accounts["lium"].list_tagged = lambda _: []
    assert not rent().ok
    from cascade.funding.providers import create_intents

    reopened = PayerKeyVault(dir=vault.dir, seal_key=vault.seal_key)
    assert create_intents(reopened, funded_pod_name("777", HK, 91))


def test_verification_resolves_legacy_floor_and_v2_shard_length(tmp_path):
    cfg = load_chain_config()
    cfg = replace(cfg, generator=replace(cfg.generator, sandbox_strict=False))
    legacy = _repo(tmp_path / "legacy", "(64,)")
    assert verify_repo(legacy, cfg, quick=True).ok
    v2 = with_contract(cfg, cfg.training.v2_contract)
    assert not verify_repo(legacy, v2, quick=True).ok
    long = _repo(tmp_path / "long", "(8, 8192)")
    assert not verify_repo(long, cfg, quick=True).ok
    assert verify_repo(long, v2, quick=True).ok
    assert v2.generator.max_memory_mb == cfg.generation.child_memory_mb


def test_verification_enforces_the_cpu_child_memory_cap(tmp_path):
    cfg = load_chain_config()
    cfg = replace(cfg, generator=replace(cfg.generator, sandbox_strict=False))
    cfg = with_contract(cfg, cfg.training.v2_contract)
    repo = _repo(tmp_path / "memory", "(8, 64)")
    source = (repo / "generator.py").read_text().replace(
        "        rng =", "        allocation = np.empty(3_000_000_000, dtype=np.uint8)\n        rng =")
    (repo / "generator.py").write_text(source)
    assert not verify_repo(repo, cfg, quick=True).ok


def test_deregistered_king_cannot_pay_a_recycled_uid():
    runner = SimpleNamespace(state=SimpleNamespace(king_hotkey="old", king_uid=7))
    client = SimpleNamespace(uid_for_hotkey=lambda _: None)
    assert WeightsMixin._king_uid_to_vote(runner, None, client=client) is None


@pytest.mark.parametrize("price", ["nan", "inf", "-inf", -1, 0, True])
def test_invalid_market_prices_are_not_quotes(price):
    assert executor_price_per_hour({"price_per_hour": price}) is None


def test_cli_rejects_nonfinite_results_as_one_strict_json(capsys):
    out = Reporter("status", json_mode=True)
    out.result(nested={"price": float("nan")})
    assert out.finish(0) == NETWORK
    payload = json.loads(capsys.readouterr().out,
                         parse_constant=lambda _: pytest.fail("non-standard JSON constant"))
    assert not payload["ok"] and payload["error"]["code"] == "non_finite_result"


def test_small_statistical_drift_never_changes_payment_terms():
    expected = {"target_lcb": 0.04, "tier": 0.02, "full_eras": 5}
    assert same_verdict(expected, {**expected, "target_lcb": 0.04 * (1 + 1e-7)})
    assert not same_verdict(expected, {**expected, "target_lcb": 0.041})
    assert not same_verdict(expected, {**expected, "full_eras": 10})


def test_heartbeat_cannot_restore_credentials_rotated_by_intake(tmp_path):
    directory = tmp_path / "vault"
    trainer = PayerKeyVault(dir=directory, seal_key=b"x" * 32)
    intake = PayerKeyVault(dir=directory, seal_key=b"x" * 32)
    trainer.insert("H", "fixture-old")
    intake.insert("H", "fixture-new")
    assert trainer.refresh("H")
    assert intake.get_shared("H") == "fixture-new"
    assert trainer.get_shared("H") == "fixture-new"


def test_unreadable_ledger_cannot_be_overwritten_by_an_allocation(tmp_path):
    from .test_funded_pod_wiring import _runner

    runner = _runner(tmp_path)
    path = runner._funded_ledger_path()
    path.write_text('{"torn":')
    pod = PodInstance("lium", "other-account-pod", "funded", "fixture", "H100", 1,
                      payer_hotkey="other")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner._ledger_add(pod)
    assert path.read_text() == '{"torn":'


def test_new_ledger_fields_do_not_lose_other_payers_after_rollback(tmp_path):
    from .test_funded_pod_wiring import _runner

    runner = _runner(tmp_path)
    row = {"provider": "shadeform", "instance_id": "paid-uuid", "stage": "funded",
           "rented_at_iso": "fixture", "sku": "H100", "gpus": 1, "payer_hotkey": "other",
           "new_release_field": "ignored"}
    runner._funded_ledger_path().write_text(json.dumps([row]))
    assert runner._load_funded_ledger()[0].payer_hotkey == "other"


def test_recovery_never_replaces_a_good_backup_with_a_corrupt_primary(tmp_path, monkeypatch):
    from .test_funded_pod_wiring import _runner

    runner = _runner(tmp_path)
    row = PodInstance("lium", "paid-pod", "funded", "fixture", "H100", 1, payer_hotkey="other")
    runner._save_funded_ledger([row])
    runner._save_funded_ledger([row])
    runner._funded_ledger_path().write_text("{torn")

    def fail_rename(*args):
        raise OSError("crash before replace")

    monkeypatch.setattr("cascade.shared.atomic.os.replace", fail_rename)
    with pytest.raises(OSError, match="crash before replace"):
        runner._save_funded_ledger(runner._load_funded_ledger())
    assert runner._load_funded_ledger() == [row]


def test_overlay_cannot_redirect_burn_or_cross_networks():
    from cascade.shared.netconfig import (
        NetConfigError,
        build_config,
        network_identity,
        next_overlay,
    )

    cfg = load_chain_config()
    for overrides in ({"scoring": {"burn_uid": 7}}, {"lane_b": {"remainder": "king"}}):
        with pytest.raises(NetConfigError, match="burn destination"):
            next_overlay(overrides, 28800, None, network=network_identity(cfg))
    body = next_overlay({"scoring": {"min_windows": 201}}, 28800, None,
                        network={"netuid": 259, "chain": "bittensor:test"})
    with pytest.raises(NetConfigError, match="network mismatch"):
        build_config(cfg, body)
