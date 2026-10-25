"""Regressions for the 2026-10 funded-rental review: create intents that must
(and must not) outlive a failure, Vast teardown confirmation, the operator-side
loss cap across in-leg pod replacements, and the Vast offer query / refusal
classification. Self-contained fakes; no provider API is ever called.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

import cascade.provision.core as core
from cascade.funding.faults import classify_rent_failure
from cascade.funding.providers import (
    PayerCredential,
    begin_create,
    create_intents,
    encode_credentials,
    lease_credentials,
    lease_records,
)
from cascade.funding.queue import OPERATOR_FAULT_RETRIES, FundedQueue
from cascade.funding.vault import PayerKeyVault
from cascade.provision import funded as funded_mod
from cascade.provision.core import (
    LIUM_QUARANTINE_FILE_ENV,
    LaunchSpec,
    LiumProvider,
    PodAddress,
    ProvisionError,
    UncertainCreateError,
    quarantined_hosts,
)
from cascade.provision.funded import (
    LEMON_CLASS,
    funded_pod_name,
    quarantine_lemon_host,
    rent_funded_pod,
    teardown_funded,
    terminate_verified,
)
from cascade.provision.state import PodInstance
from cascade.provision.vast import VastProvider, vast_offer_price

HK = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
BUNDLE_VAST = encode_credentials((PayerCredential("vast", "fixture-vast"),))
NAME = funded_pod_name("777", HK, 91)
IMAGE = "image@sha256:" + "a" * 64


def _listing(rows):
    return 200, {"success": True, "instances": rows, "instances_found": len(rows),
                 "total_instances": len(rows), "next_token": None}


class FakeVast:
    """Vast API double: the instance is allocated server-side on every PUT
    unless ``create_mode`` is a true refusal; ``listing_script`` overrides
    listings call by call ("empty" / "real")."""

    def __init__(self):
        self.offers = [{"id": 31, "machine_id": 310, "num_gpus": 1, "dph_total": 0.35,
                        "gpu_name": "RTX 4090"}]
        self.instances: dict[int, dict] = {}
        self.creates = self.deletes = self.offers_calls = 0
        self.create_mode = "ok"
        self.offers_mode = "ok"
        self.create_reply = None
        self.listing_script: list = []
        self.queries: list[dict] = []
        self.clock = [0.0]

    def __call__(self, method, url, *, headers, json=None, timeout=30.0):
        assert headers["Authorization"] == "Bearer fixture-vast"
        if method == "POST" and url.endswith("/bundles/"):
            self.offers_calls += 1
            self.queries.append(json)
            if self.offers_mode == "500_in_launch" and self.offers_calls > 1:
                self.offers_mode = "ok"            # the pool's probe passed; launch's re-list 5xx'd
                return 500, {"raw": "<html>502 Bad Gateway</html>"}
            return 200, {"offers": list(self.offers)}
        if method == "PUT" and "/asks/" in url:
            if self.create_reply is not None:
                return self.create_reply
            self.creates += 1
            iid = 9000 + self.creates
            self.instances[iid] = {
                "id": iid, "label": json["label"], "actual_status": "running",
                "public_ipaddr": "203.0.113.5", "machine_id": 310, "gpu_name": "RTX 4090",
                "num_gpus": 1, "ports": {"22/tcp": [{"HostPort": "40022"}]}}
            return 200, {"success": True, "new_contract": iid}
        if method == "GET" and "/api/v1/instances/" in url:
            if self.listing_script:
                item = self.listing_script.pop(0)
                return _listing([] if item == "empty" else list(self.instances.values()))
            return _listing(list(self.instances.values()))
        if method == "DELETE" and "/instances/" in url:
            self.deletes += 1
            self.instances.pop(int(url.rstrip("/").rsplit("/", 1)[-1]), None)
            return 200, {"success": True}
        return 404, {"error": "unexpected"}


@pytest.fixture
def market(monkeypatch, tmp_path):
    monkeypatch.setenv(LIUM_QUARANTINE_FILE_ENV, str(tmp_path / "quarantine.json"))
    server = FakeVast()

    def now():
        server.clock[0] += 30.0
        return server.clock[0]

    monkeypatch.setattr(funded_mod, "provider_for_key",
                        lambda provider, key: VastProvider(api_key=key, _http=server,
                                                           _sleep=lambda s: None, _now=now))
    vault = PayerKeyVault(dir=tmp_path / "vault", seal_key=b"x" * 32)
    vault.insert(HK, BUNDLE_VAST)

    def rent():
        return rent_funded_pod(round_id="777", hotkey=HK, api_key=BUNDLE_VAST, sku="RTX4090",
                               skus=("RTX4090",), netuid=91, image=IMAGE,
                               ssh_pubkey="fixture-public", vault=vault)

    return server, vault, rent


# ── 5: a failure BEFORE any create request resolves the intent ───────────────

def test_offer_search_5xx_before_the_create_leaves_no_intent(market):
    server, vault, rent = market
    server.offers_mode = "500_in_launch"
    first = rent()
    assert not first.ok and not first.leaked_pod and server.creates == 0
    assert not create_intents(vault, NAME) and not list(lease_records(vault, HK))
    second = rent()                                     # the API is healthy again
    assert second.ok and server.creates == 1
    assert "create_unresolved" not in second.error


def test_uncertain_create_after_the_request_still_keeps_the_intent(market):
    server, vault, rent = market
    server.create_reply = (200, {"raw": "<html><body>OK</body></html>"})
    first = rent()
    assert not first.ok and first.leaked_pod == NAME + "-0" and not first.burn_attempt
    assert create_intents(vault, NAME)
    server.create_reply = None
    assert "create_unresolved" in rent().error and server.creates == 0


def test_shadeform_signals_refusal_versus_uncertain_create(monkeypatch):
    from cascade.provision.core import ShadeformProvider

    offer = {"cloud": "c", "region": "r", "shade_instance_type": "t", "price_usd_hr": 1.0,
             "sku": "RTX4090"}
    spec = LaunchSpec("RTX4090", 1, IMAGE, "ssh-ed25519 AAAA", name_prefix="pfx")

    def provider(exc):
        p = ShadeformProvider(api_key="fixture")
        monkeypatch.setattr(p, "_offer", lambda *a, **k: offer)

        def post(path, body=None):
            raise exc
        monkeypatch.setattr(p, "_post", post)
        return p

    exc = RuntimeError("422")
    exc.response = SimpleNamespace(status_code=422)
    p = provider(exc)
    with pytest.raises(ProvisionError) as e:
        p.launch(spec)
    assert not isinstance(e.value, UncertainCreateError) and not p.create_requested("pfx-0")
    p = provider(TimeoutError("read timed out"))
    with pytest.raises(UncertainCreateError):
        p.launch(spec)
    assert p.create_requested("pfx-0")
    p = ShadeformProvider(api_key="fixture")
    monkeypatch.setattr(p, "_offer", lambda *a, **k: offer)
    monkeypatch.setattr(p, "_post", lambda path, body=None: {})
    with pytest.raises(UncertainCreateError):                # 2xx without an id
        p.launch(spec)


def test_lium_failure_before_lium_up_is_not_a_create(monkeypatch):
    def run(argv):
        if argv[1] == "ls":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="502 Bad Gateway")
        raise AssertionError(argv)

    prov = LiumProvider(bin="lium", api_key="k", _run=run, _spawn=lambda argv: None,
                        _template=lambda payload: "tmpl", _sleep=lambda s: None)
    monkeypatch.setattr(core, "cooling_executors", lambda: set())
    with pytest.raises(ProvisionError):
        prov.launch(LaunchSpec("RTX4090", 1, IMAGE, "ssh-ed25519 AAAA", name_prefix="pfx"))
    assert not prov.create_requested("pfx-0")


# ── 2: a listing flap during adoption keeps the lease and the intent ─────────

class ScriptedProvider:
    name = "vast"

    def __init__(self, script):
        self.script = list(script)
        self.pods = {"vast-1": NAME + "-0"}
        self.creates = 0

    def list_tagged(self, prefix):
        if self.script:
            return [NAME + "-0"] if self.script.pop(0) == "pod" else []
        return [n for n in self.pods.values() if n.startswith(prefix)]

    def available(self, sku, count, *, gpus=1, exclude_ids=()):
        return True

    def launch(self, spec):
        self.creates += 1
        self.pods[f"vast-{self.creates + 1}"] = spec.name_prefix + "-0"
        return [spec.name_prefix + "-0"]

    def wait_ready(self, pod_id, *, timeout):
        return True

    def get_ip(self, pod_id):
        return PodAddress("192.0.2.1", 22)

    live_pod_address = get_ip

    def pod_identity(self, pod_id):
        return {"id": pod_id, "status": "RUNNING"}

    def terminate(self, pod_id):
        pass


def test_listing_flap_during_adoption_never_rents_a_twin(monkeypatch, tmp_path):
    # pool probe sees the pod → adoption re-list sees nothing → next rent's
    # probe sees nothing: the first allocation still exists and still bills.
    prov = ScriptedProvider(["pod", "pod", "", "", "", ""])
    monkeypatch.setattr(funded_mod, "provider_for_key", lambda provider, key: prov)
    vault = PayerKeyVault(dir=tmp_path / "vault", seal_key=b"x" * 32)
    vault.insert(HK, BUNDLE_VAST)
    token = lease_credentials(vault, HK, PayerCredential("vast", "fixture-vast"), scope=NAME)
    begin_create(vault, token, scope=NAME, provider="vast")
    kw = dict(round_id="777", hotkey=HK, api_key=BUNDLE_VAST, sku="RTX4090", skus=("RTX4090",),
              netuid=91, image=IMAGE, ssh_pubkey="fixture-public", vault=vault)
    first = rent_funded_pod(**kw)
    assert not first.ok and first.leaked_pod == NAME + "-0" and not first.burn_attempt
    assert create_intents(vault, NAME) and len(list(lease_records(vault, HK))) == 1
    second = rent_funded_pod(**kw)
    assert not second.ok and "create_unresolved" in second.error and prov.creates == 0


# ── 3 + 4: Vast "gone" needs an observation or a DELETE ack; teardown rows ───

def test_vast_terminate_verified_never_confirms_gone_from_an_empty_listing_alone(market):
    server, vault, rent = market
    assert rent().ok and server.creates == 1
    server.listing_script = ["empty"] * 3
    fresh = VastProvider(api_key="fixture-vast", _http=server, _sleep=lambda s: None)
    assert not terminate_verified(fresh, NAME + "-0") and server.deletes == 0
    assert server.instances                             # still live, still billing


def test_vast_cleanup_deletes_the_contract_when_the_listing_lags(market):
    server, vault, rent = market
    monkeypatch_wait = VastProvider.wait_ready
    try:
        VastProvider.wait_ready = lambda self, pod_id, *, timeout: False   # lemon
        server.listing_script = ["empty"] * 6                              # listing lags throughout
        res = rent()
    finally:
        VastProvider.wait_ready = monkeypatch_wait
    assert not res.ok and res.error_class == LEMON_CLASS and not res.leaked_pod
    assert server.deletes == 1 and not server.instances
    assert not create_intents(vault, NAME) and not list(lease_records(vault, HK))


def test_teardown_funded_confirms_an_observed_pod_gone_from_an_empty_listing(market):
    server, vault, rent = market
    res = rent()
    assert res.ok and res.pod.pod_uid
    server.instances.clear()                            # the miner destroyed it themselves
    assert teardown_funded([res.pod], vault) == []
    assert not list(lease_records(vault, HK))


def test_teardown_funded_keeps_a_write_ahead_row_whose_create_is_unresolved(market):
    server, vault, rent = market
    server.create_reply = (200, {"raw": "<html><body>OK</body></html>"})
    res = rent()
    assert res.leaked_pod
    token = next(t for t, _, _ in lease_records(vault, HK))
    row = PodInstance(provider="vast", instance_id=NAME + "-0", stage="funded", rented_at_iso="x",
                      sku="RTX4090", gpus=1, payer_hotkey=HK, payer_credential_id=token)
    server.listing_script = ["empty"] * 4
    assert teardown_funded([row], vault) == [row]      # fail closed: still unresolved
    assert create_intents(vault, NAME)


# ── 6 + 8: the offer query and price filter ──────────────────────────────────

def test_vast_offer_query_requires_a_direct_port_and_uses_the_cli_type_spelling(market):
    server, vault, rent = market
    assert rent().ok
    query = server.queries[-1]
    assert query["direct_port_count"] == {"gte": 1} and query["type"] == "on-demand"


def test_vast_lemon_host_is_quarantined_and_not_rented_again(market, monkeypatch):
    server, vault, rent = market
    monkeypatch.setattr(VastProvider, "wait_ready", lambda self, pod_id, *, timeout: False)
    res = rent()
    assert res.error_class == LEMON_CLASS
    assert quarantined_hosts() and set(quarantined_hosts()) == {"vast:310"}
    retry = rent()
    assert retry.error_class == "no_capacity" and server.creates == 1


def test_vast_host_of_pod_names_the_machine_for_the_quarantine_ledger(monkeypatch, tmp_path):
    monkeypatch.setenv(LIUM_QUARANTINE_FILE_ENV, str(tmp_path / "q.json"))
    rows = [{"id": 1, "label": "pfx-0", "machine_id": 9, "actual_status": "running"}]
    prov = VastProvider(api_key="k", _http=lambda method, url, **kw: _listing(rows))
    assert prov.host_of_pod("pfx-0") == "vast:9"
    assert quarantine_lemon_host(prov, "pfx-0", "not ready within 900s") == "vast:9"
    assert "vast:9" in quarantined_hosts()


def test_vast_offer_price_rejects_zero_bool_and_non_finite_prices():
    offers = [{"id": 1, "machine_id": 1, "num_gpus": 1, "dph_total": 0.0},
              {"id": 2, "machine_id": 2, "num_gpus": 1, "dph_total": True},
              {"id": 3, "machine_id": 3, "num_gpus": 1, "dph_total": "0.000001"},
              {"id": 4, "machine_id": 4, "num_gpus": 1, "dph_total": float("inf")},
              {"id": 5, "machine_id": 5, "num_gpus": 1, "dph_total": float("nan")},
              {"id": 6, "machine_id": 6, "num_gpus": 1, "dph_total": -0.5},
              {"id": 7, "machine_id": 7, "num_gpus": 1},
              {"id": 8, "machine_id": 8, "num_gpus": 1, "dph_total": 0.90}]
    assert vast_offer_price({"dph_total": True}) is None and vast_offer_price({"dph_total": 0}) is None
    prov = VastProvider(api_key="k", _http=lambda m, u, **kw: (200, {"offers": offers}))
    assert [o["id"] for o in prov._offers("RTX4090")] == [3, 8]


# ── 7: a balance refusal is auth, never sold-out ─────────────────────────────

@pytest.mark.parametrize("status,body,cls", [
    (200, {"success": False, "msg": "Insufficient credit"}, "auth"),
    (200, {"success": False, "msg": "insufficient_credit"}, "auth"),
    (200, {"success": False, "error": "insufficient_credit", "msg": "Your balance is too low"}, "auth"),
    (400, {"success": False, "msg": "insufficient balance"}, "auth"),
    (200, {"success": False, "msg": "no_such_ask"}, "no_capacity"),
    (200, {"success": False, "msg": "unavailable"}, "no_capacity"),
])
def test_vast_refusal_body_is_classified_before_the_capacity_prefix(market, status, body, cls):
    server, vault, rent = market
    server.create_reply = (status, body)
    res = rent()
    assert not res.ok and res.error_class == cls and not res.leaked_pod and not res.burn_attempt
    assert server.creates == 0 and not create_intents(vault, NAME)


def test_singular_and_snake_case_funding_markers_classify_as_auth():
    for msg in ("Insufficient credit", "insufficient_credit", "insufficient_balance",
                "vast: offer 31 refused (200): {'msg': 'insufficient_funds'}"):
        assert classify_rent_failure(msg) == "auth", msg


# ── 9: a ghost `lium up` resolves on evidence, or stays unresolved ───────────

class GhostLium:
    """`lium ls` lists one executor, `lium up` deploys nothing, `lium ps` never shows it."""

    def __init__(self, rm_stderr="pod not found"):
        self.ups, self.rms, self.ps_calls = [], [], 0
        self.rm_stderr = rm_stderr
        self.clock = [0.0]

    def run(self, argv):
        cmd = argv[1]
        if cmd == "ls":
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(
                [{"id": "exec-1", "gpu_count": 1, "price_per_hour": 0.4}]), stderr="")
        if cmd == "ps":
            self.ps_calls += 1
            return subprocess.CompletedProcess(argv, 0, stdout="[]", stderr="")
        if cmd == "rm":
            self.rms.append(argv[2])
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=self.rm_stderr)
        raise AssertionError(argv)

    def now(self):
        self.clock[0] += 30.0
        return self.clock[0]


def _lium_market(monkeypatch, tmp_path, ghost):
    monkeypatch.setattr(core, "record_executor_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(core, "cooling_executors", lambda: set())
    monkeypatch.setattr(core, "quarantined_hosts", lambda: set())
    monkeypatch.setattr(funded_mod, "provider_for_key", lambda provider, key: LiumProvider(
        bin="lium", api_key=key, _run=ghost.run, _spawn=lambda argv: ghost.ups.append(argv),
        _template=lambda payload: "tmpl-1", _sleep=lambda s: None, _now=ghost.now))
    bundle = encode_credentials((PayerCredential("lium", "fixture-lium"),))
    vault = PayerKeyVault(dir=tmp_path / "vault", seal_key=b"x" * 32)
    vault.insert(HK, bundle)
    return vault, lambda: rent_funded_pod(
        round_id="777", hotkey=HK, api_key=bundle, sku="RTX4090", skus=("RTX4090",), netuid=91,
        image=IMAGE, ssh_pubkey="fixture-public", vault=vault, ready_timeout=900.0)


def test_lium_ghost_up_resolves_the_intent_once_nothing_can_have_been_deployed(monkeypatch, tmp_path):
    ghost = GhostLium()
    vault, rent = _lium_market(monkeypatch, tmp_path, ghost)
    first = rent()
    assert not first.ok and first.error_class == LEMON_CLASS
    assert not first.leaked_pod                        # nothing observed: no "LEAKED" label
    assert ghost.rms == [NAME + "-0"] and ghost.ps_calls >= 2
    assert not create_intents(vault, NAME) and not list(lease_records(vault, HK))
    second = rent()                                    # the leg name is free again
    assert "create_unresolved" not in second.error and len(ghost.ups) == 2


@pytest.mark.parametrize("stderr", ["401 unauthorized", "connection reset by peer"])
def test_lium_ghost_up_stays_unresolved_without_a_not_found_answer(monkeypatch, tmp_path, stderr):
    ghost = GhostLium(rm_stderr=stderr)
    vault, rent = _lium_market(monkeypatch, tmp_path, ghost)
    first = rent()
    assert not first.ok and first.leaked_pod == NAME + "-0"
    assert create_intents(vault, NAME)
    assert "create_unresolved" in rent().error and len(ghost.ups) == 1


def test_lium_ghost_is_not_settled_before_the_appear_window(monkeypatch):
    ghost = GhostLium()
    monkeypatch.setattr(core, "record_executor_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(core, "cooling_executors", lambda: set())
    monkeypatch.setattr(core, "quarantined_hosts", lambda: set())
    prov = LiumProvider(bin="lium", api_key="k", _run=ghost.run, _spawn=lambda argv: None,
                        _template=lambda payload: "tmpl-1", _sleep=lambda s: None,
                        _now=lambda: 0.0)                 # the clock never moves
    prov.launch(LaunchSpec("RTX4090", 1, IMAGE, "ssh-ed25519 AAAA", name_prefix="pfx"))
    assert prov.create_unresolved("pfx-0")
    assert not terminate_verified(prov, "pfx-0")


# ── 1: a systemic post-allocation fault is bounded by the loss cap ───────────

REF = "alice/gen-a@sha256:" + "a" * 64
POD = "cascade-n91-777-funded-hka-0"


def _mainnet_runner(tmp_path, monkeypatch):
    """funded_operator_fallback=false, no operator lane: the mainnet shape."""
    from cascade.trainer.loop import TrainerRunner
    from cascade.trainer.remote import PodHygiene
    from tests.unit.test_funded_pod_wiring import _runner, _vault
    from tests.unit.test_funded_rent_wait import _arm_wait

    r = _runner(tmp_path, funded_operator_fallback=False)
    r.remote_hosts = [replace(r.remote_hosts[0], profile_only=True)]
    r._operator_fallback_lanes = TrainerRunner._operator_fallback_lanes.__get__(r)
    assert r._operator_fallback_lanes() == [] and not r._operator_recovery_available()
    _vault(tmp_path, "hkA")
    q = FundedQueue(tmp_path / "fq.json", entry_ttl_seconds=3600.0)
    q.add("hkA", REF, reveal_block=1)
    r._funded_queue = lambda: q
    _arm_wait(r, deadline_offsets=10 ** 6, capacity_seq=[1] * 1000)
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host, **kw: "")
    monkeypatch.setattr("cascade.trainer.remote.probe_pod_hygiene",
                        lambda host, ckpt_dir, **kw: PodHygiene())
    r._funded_pod_identity_mismatch = lambda pod: None
    r._funded_checkpoint_mismatch = lambda entry, contract: None
    r._teardown_funded_pod = lambda pod: r._ledger_remove(pod.instance_id)
    return r, q


def _lemon_rent(allocations):
    def fake(**kw):
        allocations.append(kw["hotkey"])
        kw["on_provider_selected"]("lium", "tok")
        kw["on_allocated"](f"exec-{len(allocations)}", POD)
        return funded_mod.FundedRentResult(
            hotkey=kw["hotkey"], ok=False, error=f"funded pod {POD} not ready within 900s",
            error_class=LEMON_CLASS, burn_attempt=False)
    return fake


def _drive(r, q, max_legs=12):
    """Emulate rolling._launch_leg._run: train, else settle from the verdict."""
    from tests.unit.test_funded_pod_wiring import _challenger, _FakeDisp

    seeds, contract = SimpleNamespace(base_seed=777), SimpleNamespace(arch_preset="toto2-4m")
    history = []
    for leg in range(1, max_legs + 1):
        q.mark_in_flight("hkA", REF, target_boundary=1, era_index=1, started_block=leg)
        try:
            r._run_funded_leg(_FakeDisp(), _challenger("hkA"), seeds, 100, contract, "",
                              warm_start_ref=None)
            break
        except Exception:  # noqa: BLE001 — same as rolling: settle from the verdict
            msg, miner_fault, cls, burn = r._funded_leg_failures.get(
                "hkA", ("no verdict", False, "operator", False))
            assert msg != "no verdict"
            if miner_fault:
                q.fail("hkA", error=msg, error_class=cls, expect_ref=REF)
            else:
                q.requeue("hkA", error=msg, error_class=cls, burn_attempt=burn)
            e = q.get("hkA")
            history.append((leg, cls, e.operator_faults, msg))
            if "paused after" in msg or e.status == "failed":
                break
    return history


def test_systemic_lemons_bill_at_most_the_operator_loss_cap_before_the_pause(tmp_path, monkeypatch):
    r, q = _mainnet_runner(tmp_path, monkeypatch)
    allocations = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", _lemon_rent(allocations))
    history = _drive(r, q)
    assert "paused after" in history[-1][-1]
    assert len(allocations) <= OPERATOR_FAULT_RETRIES         # was 3 pods × 3 legs = 9
    assert q.get("hkA").operator_faults == len(allocations)   # every billed pod counted once
    assert q.get("hkA").attempts == 0                          # never burned


def test_one_lemon_then_a_good_pod_counts_one_loss_and_runs(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok

    r, q = _mainnet_runner(tmp_path, monkeypatch)
    q.mark_in_flight("hkA", REF, target_boundary=1, era_index=1, started_block=1)
    lemon = funded_mod.FundedRentResult(hotkey="hkA", ok=False, error="funded pod x not ready",
                                        error_class=LEMON_CLASS, burn_attempt=False)
    outcomes = [lemon, _rent_ok()]
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: outcomes.pop(0))
    r._funded_pod_code_mismatch = lambda result, profile: ""
    r._rent_wait_now = lambda: 0.0
    r._funded_rent_wait_deadline = lambda: 1.0
    host, pod = r._rent_funded_host("777", _challenger("hkA"))
    assert outcomes == [] and host.host == "10.9.9.9"
    e = q.get("hkA")
    assert e.operator_faults == 1 and not e.operator_billed   # counted, the leg goes on


def test_in_leg_replacements_stop_at_the_remaining_loss_budget(tmp_path, monkeypatch):
    from cascade.trainer.faults import _FundedLegSkip
    from tests.unit.test_funded_pod_wiring import _challenger

    r, q = _mainnet_runner(tmp_path, monkeypatch)
    q.mark_in_flight("hkA", REF, target_boundary=1, era_index=1, started_block=1)
    for _ in range(OPERATOR_FAULT_RETRIES - 1):               # two earlier legs were lost
        q.set_operator_billed("hkA", True)
        q.set_operator_billed("hkA", False)
    allocations = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", _lemon_rent(allocations))
    r._rent_wait_now = lambda: 0.0
    r._funded_rent_wait_deadline = lambda: 1.0
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    assert len(allocations) == 1                              # one more loss fills the cap
    assert q.get("hkA").operator_faults == OPERATOR_FAULT_RETRIES
    msg, miner_fault, cls, burn = r._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "infra", False) and "bad pod(s)" in msg
