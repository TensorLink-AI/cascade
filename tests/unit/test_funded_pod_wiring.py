"""[round] funded_pods = "rent": per-payer rent → dispatch → teardown wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import cascade.provision.funded as funded_mod
from cascade.funding.queue import FundedQueue
from cascade.funding.vault import PayerKeyVault
from cascade.provision.core import PodAddress
from cascade.provision.state import PodInstance
from cascade.shared.config import RoundConfig, validate_funded_pods
from cascade.trainer.loop import TrainerRunner, _FundedLegSkip
from cascade.trainer.remote import RemoteHost

REF = "ns/gen@sha256:" + "a" * 64
VAULT_REF = "vault/direct@sha256:" + "b" * 64


def _challenger(hotkey: str, ref: str = REF) -> SimpleNamespace:
    return SimpleNamespace(hotkey=hotkey, uid=1, ref=ref)


def _profile(tmp_path) -> RemoteHost:
    key = tmp_path / "op_key"
    key.write_text("private")
    (tmp_path / "op_key.pub").write_text("ssh-ed25519 AAAA op\n")
    return RemoteHost(
        name="op-final-0", host="10.0.0.1", port=22, user="root",
        key_path=str(key), remote_python="/root/cascade/.venv/bin/python",
        workdir="/root/cascade", cuda_device="0",
        chain_toml="/root/cascade/chain.testnet.toml",
        forward_env=("HIPPIUS_S3_ACCESS_KEY",), stage="final",
    )


def _runner(tmp_path, *, sku="RTX4090", image="ghcr.io/x/worker@sha256:" + "c" * 64,
            vault_dir="pv", profile=None, **round_kw):
    rnd = RoundConfig(funded_mode="required", funded_pods="rent",
                      payer_vault_dir=vault_dir, funded_pod_sku=sku,
                      funded_pod_image=image, **round_kw)
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path,
                           _funded_field={}, _funded_leg_failures={})
    for name in ("_funded_queue", "_payer_vault", "_funded_pod_profile",
                 "_funded_ledger_path", "_load_funded_ledger", "_save_funded_ledger",
                 "_ledger_add", "_ledger_remove", "_reconcile_funded_pods",
                 "_record_funded_failure", "_rent_funded_host",
                 "_teardown_funded_pod", "_run_funded_leg", "_settle_funded",
                 "_filter_funded_challengers", "_submissions_path",
                 "_submission_store"):
        setattr(fake, name, getattr(TrainerRunner, name).__get__(fake))
    prof = profile or _profile(tmp_path)
    fake._hosts_for = lambda stage: [prof]
    return fake


def _vault(tmp_path, hotkey: str, key: str = "sk_test123") -> PayerKeyVault:
    v = PayerKeyVault(dir=tmp_path / "pv")
    v.insert(hotkey, key)
    return v


def _pod(hotkey: str = "hkA") -> PodInstance:
    return PodInstance(provider="lium", instance_id="cascade-1-funded-hka",
                      stage="funded", rented_at_iso="2026-09-02T00:00:00Z",
                      sku="RTX4090", gpus=1, payer_hotkey=hotkey)


def _rent_ok(hotkey="hkA"):
    return funded_mod.FundedRentResult(
        hotkey=hotkey, ok=True, pod=_pod(hotkey),
        address=PodAddress(ip="10.9.9.9", ssh_port=2222))


# ── _rent_funded_host ────────────────────────────────────────────────────────

def test_rent_success_builds_host_from_profile_and_ledgers_first(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    seen = {}

    def fake_rent(**kw):
        # write-ahead assertion runs at teardown-time via ledger content below
        seen.update(kw)
        return _rent_ok()

    monkeypatch.setattr(funded_mod, "rent_funded_pod", fake_rent)
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert seen["api_key"] == "sk_test123"
    assert seen["sku"] == "RTX4090"
    assert seen["ssh_pubkey"].startswith("ssh-ed25519")
    # the funded host mirrors the operator final profile (image parity contract)
    assert (host.host, host.port) == ("10.9.9.9", 2222)
    assert host.workdir == "/root/cascade"
    assert host.chain_toml == "/root/cascade/chain.testnet.toml"
    assert host.stage == "final"
    # write-ahead: the pod is on the ledger before any use
    ledger = json.loads((tmp_path / "funded_pods.json").read_text())
    assert [x["instance_id"] for x in ledger] == [pod.instance_id]


def test_rent_missing_key_settles_auth_and_skips(tmp_path):
    runner = _runner(tmp_path)
    PayerKeyVault(dir=tmp_path / "pv")            # empty vault
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (True, "auth", False)


def test_rent_failure_classes_flow_to_settle_verdict(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw:
                        funded_mod.FundedRentResult(
                            hotkey="hkA", ok=False, error="sold out",
                            error_class="no_capacity", burn_attempt=False))
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)


def test_rent_leak_is_surfaced_in_the_settle_error(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw:
                        funded_mod.FundedRentResult(
                            hotkey="hkA", ok=False, error="boom",
                            error_class="infra", burn_attempt=True,
                            leaked_pod="cascade-777-funded-hka"))
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    msg = runner._funded_leg_failures["hkA"][0]
    assert "LEAKED" in msg and "cascade-777-funded-hka" in msg


def test_operator_config_fault_never_burns_the_miner(tmp_path):
    runner = _runner(tmp_path, sku="")            # operator forgot the SKU
    _vault(tmp_path, "hkA")
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, burn) == (False, False)


# ── _run_funded_leg ──────────────────────────────────────────────────────────

class _FakeDisp:
    def __init__(self, fail_rc=None):
        self.fail_rc = fail_rc
        self.calls = []

    def dispatch(self, host, **kw):
        self.calls.append((host, kw))
        if self.fail_rc is not None:
            from cascade.trainer.remote import RemoteDispatchError
            raise RemoteDispatchError("leg failed", returncode=self.fail_rc)
        return SimpleNamespace(hotkey=kw["hotkey"], role=kw["role"])


def _leg_runner(tmp_path, monkeypatch, *, disp):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: _rent_ok())
    torn = []
    runner._teardown_funded_pod = lambda pod: torn.append(pod.instance_id)
    runner._funded_field = {"hkA": REF}
    seeds = SimpleNamespace(base_seed=777)
    contract = SimpleNamespace(arch_preset="toto2-4m")
    return runner, torn, seeds, contract


def test_funded_leg_dispatches_on_the_rented_pod_and_tears_down(tmp_path, monkeypatch):
    disp = _FakeDisp()
    runner, torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=disp)
    entry = runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100,
                                   contract, "", warm_start_ref=None)
    assert entry.hotkey == "hkA"
    (host, kw), = disp.calls
    assert host.host == "10.9.9.9" and kw["role"] == "challenger"
    assert torn == ["cascade-1-funded-hka"]      # teardown ALWAYS runs


def test_funded_leg_failure_classifies_and_still_tears_down(tmp_path, monkeypatch):
    disp = _FakeDisp(fail_rc=255)
    runner, torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=disp)
    with pytest.raises(Exception):
        runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100,
                               contract, "", warm_start_ref=None)
    assert torn == ["cascade-1-funded-hka"]
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "infra", True)


def test_funded_leg_rc3_is_the_miners_fault(tmp_path, monkeypatch):
    disp = _FakeDisp(fail_rc=3)
    runner, torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=disp)
    with pytest.raises(Exception):
        runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100,
                               contract, "", warm_start_ref=None)
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls) == (True, "generator")


def test_vault_ref_leg_stages_the_zip_and_pins_the_pod_env(tmp_path, monkeypatch):
    disp = _FakeDisp()
    runner, torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=disp)
    staged = []

    def fake_stage(host, digest):
        staged.append(digest)
        from dataclasses import replace
        return replace(host, static_env=(("CASCADE_VAULT_DIR",
                                          f"{host.workdir}/_vault_stage"),))

    runner._stage_vault_zip_on = fake_stage
    runner._funded_field = {"hkA": VAULT_REF}
    runner._run_funded_leg(disp, _challenger("hkA", ref=VAULT_REF), seeds, 100,
                           contract, "", warm_start_ref=None)
    assert staged == ["b" * 64]
    (host, kw), = disp.calls
    assert dict(host.static_env) == {"CASCADE_VAULT_DIR": "/root/cascade/_vault_stage"}


# ── teardown + ledger + boundary sweep ───────────────────────────────────────

def test_teardown_confirmed_removes_from_ledger(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    runner._ledger_add(_pod())
    monkeypatch.setattr(funded_mod, "teardown_funded", lambda pods, vault: [])
    runner._teardown_funded_pod(_pod())
    assert runner._load_funded_ledger() == []


def test_teardown_unconfirmed_stays_on_ledger_for_the_sweep(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    pod = _pod()
    runner._ledger_add(pod)
    monkeypatch.setattr(funded_mod, "teardown_funded", lambda pods, vault: [pod])
    runner._teardown_funded_pod(pod)
    assert [x.instance_id for x in runner._load_funded_ledger()] == [pod.instance_id]


def test_boundary_sweep_tears_down_ledgered_leftovers(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    runner._ledger_add(_pod())
    calls = {"teardown": 0, "reconcile": 0}

    def fake_teardown(pods, vault):
        calls["teardown"] += 1
        return []

    def fake_reconcile(owned, vault):
        calls["reconcile"] += 1
        return []

    monkeypatch.setattr(funded_mod, "teardown_funded", fake_teardown)
    monkeypatch.setattr(funded_mod, "reconcile_funded", fake_reconcile)
    runner._reconcile_funded_pods()
    assert calls == {"teardown": 1, "reconcile": 1}
    assert runner._load_funded_ledger() == []


def test_sweep_is_inert_when_funded_pods_off(tmp_path):
    runner = _runner(tmp_path)
    runner.cfg.round = RoundConfig(funded_mode="required")   # funded_pods="off"
    runner._reconcile_funded_pods()                          # must not raise
    assert not (tmp_path / "funded_pods.json").exists()


# ── config + static_env plumbing ─────────────────────────────────────────────

def test_validate_funded_pods():
    assert validate_funded_pods("rent") == "rent"
    with pytest.raises(ValueError):
        validate_funded_pods("on")


def test_round_config_funded_pod_defaults_are_inert():
    rnd = RoundConfig()
    assert rnd.funded_pods == "off"
    assert rnd.payer_vault_dir == ""
    assert rnd.funded_pod_sku == ""
    assert rnd.funded_pod_image == ""


def test_dispatch_static_env_wins_over_forwarded_copies(monkeypatch, tmp_path):
    # The pod-local CASCADE_VAULT_DIR must beat the orchestrator's own export
    # of the same name (its store path means nothing on the pod).
    from cascade.trainer.remote import RemoteDispatcher

    monkeypatch.setenv("CASCADE_VAULT_DIR", "/orchestrator/store")
    host = RemoteHost(name="p", host="1.2.3.4", workdir="/root/cascade",
                      forward_env=("CASCADE_VAULT_DIR",),
                      static_env=(("CASCADE_VAULT_DIR", "/root/cascade/_vault_stage"),))
    captured = {}

    def fake_runner(ssh_argv, timeout, stdin_env):
        captured["stdin"] = stdin_env
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {"hotkey": "hk", "role": "king", "uid": 1, "gen_ref": REF,
             "trained_pointer": "x", "corpus_digest": "d", "n_series": 1,
             "total_points": 1}), stderr="")

    disp = RemoteDispatcher(trainer_spec="m:C", timeout_seconds=5, _runner=fake_runner)
    try:
        disp.dispatch(host, gen_ref=REF, uid=1, hotkey="hk", role="king",
                      base_seed=1, block=1)
    except Exception:
        pass  # receipt shape isn't the point; the env is
    assert "CASCADE_VAULT_DIR=/root/cascade/_vault_stage" in captured["stdin"]
    assert "/orchestrator/store" not in captured["stdin"]
