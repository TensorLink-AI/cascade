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
    import threading
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path,
                           _funded_field={}, _funded_leg_failures={},
                           _funded_claimed_execs=set(),
                           _funded_exec_lock=threading.Lock(),
                           _funded_admission_info={},
                           _funded_round_sku="",
                           _funded_king_host=None,
                           _funded_king_lock=threading.Lock(),
                           _funded_roster={"seated": [], "waiting": [],
                                           "terminal": [], "outcomes": []})
    for name in ("_funded_queue", "_payer_vault", "_funded_pod_profile",
                 "_funded_admission_cap", "_probe_funded_capacity",
                 "_rent_king_host", "_teardown_operator_pod",
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


# ── elastic admission (no-heat field) + executor claims ──────────────────────

def _cap_runner(tmp_path, **kw):
    r = _runner(tmp_path, **kw)
    return r


def test_admission_cap_defaults_to_finalist_cap(tmp_path):
    r = _runner(tmp_path)
    r.cfg.round = RoundConfig(funded_mode="required", finalists=1, max_finalists=3)
    assert r._funded_admission_cap() == 3


def test_funded_field_cap_overrides_finalist_cap(tmp_path):
    r = _runner(tmp_path, funded_field_cap=12)
    assert r._funded_admission_cap() == 12


def test_capacity_probe_clamps_to_market_minus_reserve(tmp_path):
    r = _runner(tmp_path, funded_field_cap=12, funded_capacity_probe=True,
                funded_capacity_reserve=1)
    r._probe_funded_capacity = lambda sku: 5
    assert r._funded_admission_cap() == 4


def test_capacity_probe_failure_clamps_nothing(tmp_path):
    r = _runner(tmp_path, funded_field_cap=12, funded_capacity_probe=True)
    r._probe_funded_capacity = lambda sku: None
    assert r._funded_admission_cap() == 12


def test_capacity_zero_seats_nobody_and_queue_holds(tmp_path):
    r = _runner(tmp_path, funded_field_cap=12, funded_capacity_probe=True,
                funded_capacity_reserve=1)
    r._probe_funded_capacity = lambda sku: 1      # king's reserve eats it
    from cascade.funding.queue import FundedQueue
    FundedQueue(tmp_path / "funded_queue.json").add("hkA", REF, reveal_block=10)
    kept = r._filter_funded_challengers([_challenger("hkA")])
    assert kept == []
    q = FundedQueue(tmp_path / "funded_queue.json")
    assert q.get("hkA").status == "queued"        # holds, unburned


def test_concurrent_rents_claim_distinct_executors(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    _vault(tmp_path, "hkB")
    seen_excludes = []

    def fake_rent(**kw):
        seen_excludes.append(kw["exclude_ids"])
        hk = kw["hotkey"]
        return funded_mod.FundedRentResult(
            hotkey=hk, ok=True, pod=_pod(hk),
            address=PodAddress(ip="10.0.0.9", ssh_port=22),
            machine_id=f"exec-{hk}")

    monkeypatch.setattr(funded_mod, "rent_funded_pod", fake_rent)
    r._rent_funded_host("1", _challenger("hkA"))
    r._rent_funded_host("1", _challenger("hkB"))
    assert seen_excludes[0] == ()
    assert seen_excludes[1] == ("exec-hkA",)
    assert r._funded_claimed_execs == {"exec-hkA", "exec-hkB"}


def test_rent_passes_exclude_ids_into_launch_spec():
    captured = {}

    class _Prov:
        name = "fake"
        def launch(self, spec):
            captured["exclude"] = spec.exclude_ids
            return ["pod-1"]
        def wait_ready(self, pod_id, *, timeout):
            return True
        def get_ip(self, pod_id):
            return PodAddress(ip="1.1.1.1", ssh_port=22)
        def terminate(self, pod_id):
            pass
        def machine_of(self, pod_id):
            return "exec-77"

    res = funded_mod.rent_funded_pod(
        round_id="1", hotkey="hkA", api_key="sk_x", sku="RTX4090",
        image="x@sha256:" + "0" * 64, ssh_pubkey="ssh-ed25519 A",
        exclude_ids=("exec-1", "exec-2"),
        provider_factory=lambda key: _Prov())
    assert captured["exclude"] == ("exec-1", "exec-2")
    assert res.ok and res.machine_id == "exec-77"


# ── transparency: the published roster + its audit check ─────────────────────

def _receipt(challengers):
    # receipt.manifest is the embedded RAW dict, entries included — the test
    # double must mirror the real shape (the settle bug taught us that).
    return SimpleNamespace(manifest={"entries": [
        {"miner_hotkey": h, "role": "challenger"} for h in challengers]},
        round_id="1")


def test_roster_check_passes_on_honest_allocation():
    from cascade.audit.checks import PASS, check_funded_roster
    roster = {"seated": [{"hotkey": "hkA", "reveal_block": 10},
                         {"hotkey": "hkB", "reveal_block": 20}],
              "waiting": [{"hotkey": "hkC", "reveal_block": 30}]}
    r = check_funded_roster(_receipt(["hkA", "hkB"]), roster)
    assert r.status == PASS


def test_roster_check_warns_on_stranger_challenger():
    from cascade.audit.checks import WARN, check_funded_roster
    roster = {"seated": [{"hotkey": "hkA", "reveal_block": 10}], "waiting": []}
    r = check_funded_roster(_receipt(["hkA", "hkGhost"]), roster)
    assert r.status == WARN and "hkGhost" in r.detail


def test_roster_check_warns_on_queue_jump():
    from cascade.audit.checks import WARN, check_funded_roster
    roster = {"seated": [{"hotkey": "hkLate", "reveal_block": 50}],
              "waiting": [{"hotkey": "hkEarly", "reveal_block": 10}]}
    r = check_funded_roster(_receipt(["hkLate"]), roster)
    assert r.status == WARN and "jumped" in r.detail


def test_roster_check_skips_when_unpublished():
    from cascade.audit.checks import SKIP, check_funded_roster
    assert check_funded_roster(_receipt(["hkA"]), None).status == SKIP


def test_roster_publishes_seats_waiting_and_outcomes(tmp_path):
    runner = _runner(tmp_path)
    for name in ("_publish_funded_roster",):
        setattr(runner, name, getattr(TrainerRunner, name).__get__(runner))
    runner._funded_admission_info = {"cap": 2, "configured_cap": 2,
                                     "market_capacity": 5, "reserve": 1}
    runner._funded_roster = {
        "seated": [{"hotkey": "hkA", "ref": REF, "reveal_block": 10}],
        "waiting": [{"hotkey": "hkB", "reveal_block": 20}],
        "terminal": [], "outcomes": [{"hotkey": "hkA", "outcome": "trained"}]}
    published = {}

    class _Store:
        def put_text(self, key, text, **kw):
            published[key] = json.loads(text)

    runner.manifest_store = lambda: _Store()
    runner._publish_funded_roster("777")
    doc = published["funded/round-777.json"]
    assert doc["admission"]["market_capacity"] == 5
    assert doc["seated"][0]["hotkey"] == "hkA"
    assert doc["outcomes"] == [{"hotkey": "hkA", "outcome": "trained"}]
    assert "funded/latest.json" in published


# ── per-round SKU choice + JIT king ──────────────────────────────────────────

def test_multi_sku_picks_most_available(tmp_path):
    r = _runner(tmp_path, funded_pod_skus=("RTX4090", "A6000", "RTX3090"))
    r._probe_funded_capacity = lambda sku: {"RTX4090": 2, "A6000": 9,
                                            "RTX3090": 4}[sku]
    r._funded_admission_cap()
    assert r._funded_round_sku == "A6000"
    assert r._funded_admission_info["sku_capacities"] == {
        "RTX4090": 2, "A6000": 9, "RTX3090": 4}


def test_multi_sku_tie_breaks_toward_preference_order(tmp_path):
    r = _runner(tmp_path, funded_pod_skus=("RTX4090", "A6000"))
    r._probe_funded_capacity = lambda sku: 7
    r._funded_admission_cap()
    assert r._funded_round_sku == "RTX4090"


def test_multi_sku_probe_blackout_falls_back_to_first(tmp_path):
    r = _runner(tmp_path, funded_pod_skus=("A6000", "RTX4090"),
                funded_field_cap=6)
    r._probe_funded_capacity = lambda sku: None
    assert r._funded_admission_cap() == 6            # no clamp
    assert r._funded_round_sku == "A6000"


def test_multi_sku_capacity_clamp_uses_chosen_sku(tmp_path):
    r = _runner(tmp_path, funded_pod_skus=("RTX4090", "A6000"),
                funded_field_cap=10, funded_capacity_probe=True,
                funded_capacity_reserve=1)
    r._probe_funded_capacity = lambda sku: {"RTX4090": 1, "A6000": 4}[sku]
    assert r._funded_admission_cap() == 3            # A6000: 4 - 1 reserve
    assert r._funded_round_sku == "A6000"


def test_rent_uses_the_rounds_chosen_sku(tmp_path, monkeypatch):
    r = _runner(tmp_path, funded_pod_skus=("RTX4090", "A6000"))
    _vault(tmp_path, "hkA")
    r._funded_round_sku = "A6000"
    seen = {}
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: (seen.update(kw), _rent_ok())[1])
    r._rent_funded_host("1", _challenger("hkA"))
    assert seen["sku"] == "A6000"


def test_king_jit_rents_once_ledgers_and_claims_executor(tmp_path):
    import cascade.provision.core as core_mod
    r = _runner(tmp_path, funded_king_rent=True)
    r._funded_round_sku = "A6000"
    launched = []

    class _Prov:
        name = "lium"
        def launch(self, spec):
            launched.append((spec.sku, spec.name_prefix, spec.exclude_ids))
            return ["king-pod-1"]
        def wait_ready(self, pod_id, *, timeout):
            return True
        def get_ip(self, pod_id):
            return PodAddress(ip="9.9.9.9", ssh_port=41000)
        def machine_of(self, pod_id):
            return "exec-king"
        def terminate(self, pod_id):
            pass

    orig = core_mod.LiumProvider
    core_mod.LiumProvider = _Prov
    try:
        h1 = r._rent_king_host("42")
        h2 = r._rent_king_host("42")                  # cached, no second rent
    finally:
        core_mod.LiumProvider = orig
    assert h1 is h2 and len(launched) == 1
    sku, prefix, excl = launched[0]
    assert sku == "A6000" and prefix == "cascade-42-funded-king"
    assert (h1.host, h1.port) == ("9.9.9.9", 41000)
    assert "exec-king" in r._funded_claimed_execs
    ledger = r._load_funded_ledger()
    assert [(x.instance_id, x.payer_hotkey) for x in ledger] == [("king-pod-1", "")]


def test_sweep_routes_operator_pods_off_the_payer_path(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    king = PodInstance(provider="lium", instance_id="king-pod-1", stage="funded",
                       rented_at_iso="2026-09-02T00:00:00Z", sku="A6000",
                       gpus=1, payer_hotkey="")
    r._ledger_add(king)
    r._ledger_add(_pod("hkA"))
    ops, payers = [], []
    r._teardown_operator_pod = lambda pod: (ops.append(pod.instance_id),
                                            r._ledger_remove(pod.instance_id))
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, vault: (payers.extend(
                            p.instance_id for p in pods), [])[1])
    monkeypatch.setattr(funded_mod, "reconcile_funded", lambda o, v: [])
    r._reconcile_funded_pods()
    assert ops == ["king-pod-1"]
    assert payers == ["cascade-1-funded-hka"]
