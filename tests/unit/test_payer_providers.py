"""No provider accounts, wallets or network: exercise signed funding and recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.funding.providers import (
    CREDENTIALS_HEADER,
    PayerCredential,
    credentials_from_env,
    decode_credentials,
    encode_credentials,
    header_credentials,
    pod_credential,
)
from cascade.funding.vault import PayerKeyVault
from cascade.miner.cli import build_fund_headers
from cascade.miner.lane_b import build_lane_b_headers
from cascade.miner.lane_c import build_lane_c_headers
from cascade.provision.core import PodAddress, ProvisionError, ShadeformProvider
from cascade.provision.funded import (
    funded_pod_name,
    reconcile_funded,
    rent_funded_pod,
    teardown_funded,
)
from cascade.provision.state import RoundState, load_state, save_state
from tests.unit.test_funding_intake import HK, REF, make_intake

KEYS = (
    PayerCredential("lium", "fixture-lium-secret"),
    PayerCredential("shadeform", "fixture-shade-secret"),
)
BUNDLE = encode_credentials(KEYS)


def sign(message):
    return hashlib.sha256(message).digest()


def verify(hotkey, message, signature):
    return signature == sign(message).hex()


@pytest.mark.parametrize("lane", ["a", "b", "c"])
@pytest.mark.parametrize("all_providers", [False, True])
def test_signed_provider_order_and_keys_bound_across_all_lanes(tmp_path, lane, all_providers):
    keys = KEYS + ((PayerCredential("vast", "fixture-vast"), PayerCredential("runpod", "fixture-runpod"))
                   if all_providers else ())
    bundle = encode_credentials(keys)
    if lane == "a":
        intake, clock = make_intake(tmp_path, require_signature=True, verify=verify)
        headers = build_fund_headers("fund", HK, REF, bundle, sign, now=clock)
        fund = intake.fund
    elif lane == "b":
        from tests.unit.test_lane_b_funded import _intake

        intake, _, _, clock = _intake(tmp_path, sig=True, verify=verify)
        headers = build_lane_b_headers("lane-b-fund", HK, ("energy",), REF, bundle, sign, now=clock)
        fund = intake.lane_b_fund
    else:
        from tests.unit.test_lane_c_trainer_intake import _intake

        intake, _, _, clock = _intake(tmp_path, sig=True, verify=verify)
        headers = build_lane_c_headers(
            "lane-c-fund", HK, "mkt:returns", REF, bundle, sign, now=clock
        )
        fund = intake.lane_c_fund
    tampered = dict(headers, **{CREDENTIALS_HEADER: encode_credentials(reversed(keys))})
    assert fund(tampered)[1]["code"] == "bad_signature"
    assert not intake.vault.has(HK)
    status, body = fund(headers)
    assert status == 202, body
    assert decode_credentials(intake.vault.get(HK)) == keys
    assert "fixture-" not in json.dumps(body)


def test_portfolio_is_explicit_and_uses_environment_names(monkeypatch):
    monkeypatch.setenv("LIUM_API_KEY", "fixture-default")
    monkeypatch.setenv("TEST_SHADE", "fixture-custom-shade")
    args = SimpleNamespace(lium_key_env="LIUM_API_KEY", provider_key_env=["shadeform=TEST_SHADE"])
    assert decode_credentials(credentials_from_env(args)) == (
        PayerCredential("shadeform", "fixture-custom-shade"),
    )
    args.provider_key_env = []
    assert credentials_from_env(args) == "fixture-default"


@pytest.mark.parametrize(
    "specs", [["unknown=ONE"], ["shadeform=literal-secret!"], ["lium=ONE", "lium=TWO"]]
)
def test_bad_or_duplicate_provider_keys_are_refused(monkeypatch, specs):
    monkeypatch.setenv("ONE", "fixture-one")
    monkeypatch.setenv("TWO", "fixture-two")
    with pytest.raises(ValueError):
        credentials_from_env(SimpleNamespace(lium_key_env="LIUM_API_KEY", provider_key_env=specs))


def test_ambiguous_wire_credentials_are_refused():
    with pytest.raises(ValueError):
        header_credentials({CREDENTIALS_HEADER: BUNDLE, "X-Lium-Api-Key": "fixture-other"})
    headers = build_fund_headers("withdraw", HK, REF, BUNDLE, sign)
    assert CREDENTIALS_HEADER not in headers and "X-Lium-Api-Key" not in headers


class Account:
    def __init__(self, name, *, available=True):
        self.name = name
        self.has_capacity = available
        self.pods = {}
        self.calls = []
        self.fail_create = False
        self.ready = True

    def list_tagged(self, prefix):
        return [pid for pid, name in self.pods.items() if name.startswith(prefix)]

    def list_named(self, prefix):
        return {pid: name for pid, name in self.pods.items() if name.startswith(prefix)}

    def available(self, sku, count, *, gpus=1):
        self.calls.append(("available", sku))
        return self.has_capacity

    def launch(self, spec):
        self.calls.append(("create", spec))
        if self.fail_create:
            # Simulate acceptance by the provider followed by a lost response.
            self.pods["ambiguous-id"] = spec.name_prefix + "-0"
            raise ProvisionError("HTTP response timed out after create")
        pid = self.name + "-uuid"
        self.pods[pid] = spec.name_prefix + "-0"
        return [pid]

    def wait_ready(self, pod_id, *, timeout):
        return self.ready

    def get_ip(self, pod_id):
        return PodAddress("192.0.2.1", 22)

    def live_pod_address(self, pod_id):
        return self.get_ip(pod_id) if self.ready else None

    def pod_identity(self, pod_id):
        return {"id": pod_id, "status": "RUNNING"} if pod_id in self.pods else None

    def terminate(self, pod_id):
        self.calls.append(("delete", pod_id))
        self.pods.pop(pod_id, None)


@pytest.fixture
def pool(monkeypatch, tmp_path):
    import cascade.provision.funded as funded

    accounts = {"lium": Account("lium"), "shadeform": Account("shadeform")}
    seen = []

    def factory(provider, key):
        seen.append((provider, key))
        return accounts[provider]

    monkeypatch.setattr(funded, "provider_for_key", factory)
    vault = PayerKeyVault(dir=tmp_path / "vault", seal_key=b"x" * 32)
    vault.insert(HK, BUNDLE)

    def rent(**overrides):
        return rent_funded_pod(
            round_id="777",
            hotkey=HK,
            api_key=BUNDLE,
            sku="H100",
            skus=("H100",),
            netuid=91,
            image="image@sha256:" + "a" * 64,
            ssh_pubkey="fixture-public",
            vault=vault,
            **overrides,
        )

    return accounts, vault, rent, seen


def test_ordered_capacity_fallback_rents_once(pool):
    accounts, vault, rent, seen = pool
    accounts["lium"].has_capacity = False
    result = rent()
    assert result.ok and result.pod.provider == "shadeform"
    assert result.pod.payer_credential_id
    assert not any(c[0] == "create" for c in accounts["lium"].calls)
    assert sum(c[0] == "create" for c in accounts["shadeform"].calls) == 1
    assert pod_credential(vault, result.pod) == KEYS[1]
    assert set(seen) == {(c.provider, c.api_key) for c in KEYS}


def test_priority_wins_over_second_provider(pool):
    accounts, _, rent, _ = pool
    result = rent()
    assert result.ok and result.pod.provider == "lium"
    assert not any(c[0] == "create" for c in accounts["shadeform"].calls)


def test_uncertain_create_never_tries_second_provider(pool):
    accounts, _, rent, _ = pool
    accounts["lium"].fail_create = True
    result = rent()
    assert not result.ok
    assert "ambiguous-id" in accounts["lium"].pods
    assert not any(c[0] == "create" for c in accounts["shadeform"].calls)
    # Next dispatch finds/adopts the existing allocation, without another bill.
    result = rent()
    assert result.ok and result.adopted
    assert sum(c[0] == "create" for c in accounts["lium"].calls) == 1


def test_pending_existing_pod_does_not_create_a_twin(pool):
    accounts, _, rent, _ = pool
    accounts["shadeform"].pods["existing-id"] = funded_pod_name("777", HK, 91) + "-0"
    accounts["shadeform"].ready = False
    result = rent()
    assert not result.ok and not result.burn_attempt
    assert not any(c[0] == "create" for a in accounts.values() for c in a.calls)


def test_two_existing_providers_block_new_rental(pool):
    accounts, _, rent, _ = pool
    for account in accounts.values():
        account.pods[account.name] = funded_pod_name("777", HK, 91) + "-0"
    result = rent()
    assert not result.ok and "multiple providers" in result.error
    assert not any(c[0] == "create" for a in accounts.values() for c in a.calls)


def test_all_providers_sold_out_is_unburned(pool):
    accounts, _, rent, _ = pool
    for account in accounts.values():
        account.has_capacity = False
    result = rent()
    assert not result.ok and result.error_class == "no_capacity" and not result.burn_attempt


def test_rotated_credentials_survive_restart_and_stop_original_account(pool, tmp_path):
    accounts, vault, rent, _ = pool
    accounts["lium"].has_capacity = False
    result = rent()
    vault.insert(HK, encode_credentials((PayerCredential("lium", "fixture-new-account"),)))
    # A different process with the same seal key can still identify and stop it.
    restarted = PayerKeyVault(dir=vault.dir, seal_key=b"x" * 32)
    restarted.hydrate()
    assert pod_credential(restarted, result.pod) == KEYS[1]
    path = tmp_path / "ledger.json"
    save_state(path, RoundState("777", instances=(result.pod,)))
    assert load_state(path).instances[0] == result.pod
    assert "fixture-" not in path.read_text()
    assert all("fixture-shade-secret" not in p.read_text() for p in vault.dir.rglob("*.json"))
    assert teardown_funded([result.pod], restarted) == []
    assert result.pod.instance_id not in accounts["shadeform"].pods


def test_wrong_provider_cannot_use_lease(pool):
    _, vault, rent, _ = pool
    result = rent()
    assert pod_credential(vault, replace(result.pod, provider="shadeform")) is None


def test_orphan_sweep_uses_provider_ids_and_names_after_rotation(pool):
    accounts, vault, rent, _ = pool
    accounts["lium"].has_capacity = False
    result = rent()
    accounts["shadeform"].pods["unrelated"] = "my-personal-job"
    vault.insert(HK, "fixture-new-lium")
    killed = reconcile_funded([], vault, netuid=91)
    assert killed == [result.pod.instance_id]
    assert accounts["shadeform"].pods == {"unrelated": "my-personal-job"}


def test_existing_owned_pod_is_not_reaped(pool):
    _, vault, rent, _ = pool
    result = rent()
    assert reconcile_funded([result.pod], vault, netuid=91) == []


def test_error_text_does_not_disclose_keys(pool):
    accounts, _, rent, _ = pool

    def broken(prefix):
        raise ProvisionError("unauthorized fixture-lium-secret fixture-shade-secret")

    accounts["lium"].list_tagged = broken
    result = rent()
    assert not result.ok and result.error_class == "auth"
    assert "fixture-" not in result.error


def test_shadeform_hourly_and_leg_caps_rechecked_before_create():
    p = ShadeformProvider(api_key="fixture-shade")
    types = {
        "instance_types": [
            {
                "cloud": "cloud",
                "shade_instance_type": "H100.1x",
                "configuration": {"gpu_type": "H100", "num_gpus": 1},
                "hourly_price": 250,
                "availability": [{"region": "r", "available": True}],
            }
        ]
    }
    p._get = lambda *a, **kw: types
    p.max_price_per_hour = 2.0
    assert not p.available("H100", 1)
    p.max_price_per_hour = 3.0
    assert p.available("H100", 1)
    p.max_leg_cost_usd = 1.0
    p.default_wall_seconds = 3600
    assert not p.available("H100", 1)
    assert "fixture-shade" not in repr(p)


def test_shadeform_sends_native_instance_spend_threshold_without_rounding_up():
    from cascade.provision.core import LaunchSpec

    provider = ShadeformProvider(api_key="fixture", max_leg_cost_usd=1.999)
    provider._offer = lambda *a, **kw: {"cloud": "c", "region": "r", "shade_instance_type": "H100",
                                      "price_usd_hr": 0.1, "sku": "H100"}
    sent = []
    provider._post = lambda path, body: sent.append(body) or {"id": "fixture-instance"}
    provider.launch(LaunchSpec("H100", 1, "image@sha256:" + "a" * 64, "fixture-public"))
    assert sent[0]["auto_delete"] == {"spend_threshold": "1.99"}


def test_shadeform_identity_requires_ready_container():
    p = ShadeformProvider(api_key="fixture-shade")
    row = {
        "id": "stable-uuid",
        "status": "active",
        "container_status": "downloading",
        "ip": "192.0.2.1",
    }
    p._get = lambda *a, **kw: row
    assert p.pod_identity("stable-uuid")["status"] != "RUNNING"
    assert p.live_pod_address("stable-uuid") is None
    row["container_status"] = "running"
    assert p.pod_identity("stable-uuid") == {"id": "stable-uuid", "status": "RUNNING"}


def test_legacy_key_rental_also_retains_credential(pool):
    accounts, vault, _, _ = pool
    result = rent_funded_pod(
        round_id="777",
        hotkey=HK,
        api_key=KEYS[0].api_key,
        sku="H100",
        image="image@sha256:" + "a" * 64,
        ssh_pubkey="fixture-public",
        netuid=91,
        vault=vault,
        provider_factory=lambda key: accounts["lium"],
    )
    vault.insert(HK, encode_credentials((KEYS[1],)))
    assert pod_credential(vault, result.pod) == KEYS[0]


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 402 Payment Required",
        "insufficient balance",
        "insufficient funds",
        "HTTP 429 too many requests",
    ],
)
def test_account_failures_never_quarantine_or_rotate_healthy_king(tmp_path, monkeypatch, message):
    from cascade.funding.faults import classify_rent_failure
    from cascade.provision.core import record_host_quarantine
    from cascade.trainer.rolling import king_pod_should_rotate
    from cascade.trainer.runner_rolling import _RollingMixin

    path = tmp_path / "quarantine.json"
    record_host_quarantine("192.0.2.10", message, path=path)
    assert not path.exists()
    assert classify_rent_failure(message) in {"auth", "rate_limited"}
    assert not king_pod_should_rotate(RuntimeError(message), 20)
    retired = []
    runner = SimpleNamespace(_rolling_retire_king_pod=lambda era: retired.append(era))
    _RollingMixin._rolling_rotate_king_pod(runner, SimpleNamespace(index=1), message)
    assert not retired


def test_delayed_payment_failure_is_not_a_bad_host_retry(monkeypatch):
    import cascade.provision.funded as funded
    from tests.unit.test_funded_provision import FakeProvider, _rent

    adapter = FakeProvider(ready=False)
    adapter._up_log_tail = lambda pid: "HTTP 402 insufficient balance"
    quarantined = []
    monkeypatch.setattr(funded, "quarantine_lemon_host", lambda *a: quarantined.append(a))
    result = _rent(adapter, ready_timeout=0)
    assert not result.ok and result.error_class == "auth" and not result.burn_attempt
    assert adapter.terminated and not quarantined


def test_shadeform_nested_container_status_and_on_demand_only():
    from cascade.provision.core import pick_shadeform_offer

    p = ShadeformProvider(api_key="fixture-shade")
    row = {
        "id": "uuid",
        "status": "active",
        "ip": "192.0.2.1",
        "launch_configuration": {"type": "docker", "container_status": "downloading"},
    }
    p._get = lambda *a, **kw: row
    assert p.live_pod_address("uuid") is None
    assert p.pod_identity("uuid")["status"] != "RUNNING"
    row["launch_configuration"]["container_status"] = "running"
    assert p.live_pod_address("uuid") is not None
    row["launch_configuration"].pop("container_status")
    assert p.live_pod_address("uuid") is None
    catalog = {
        "instance_types": [
            {
                "cloud": "c",
                "shade_instance_type": "H100",
                "configuration": {"gpu_type": "H100"},
                "hourly_price": 100,
                "availability": [{"region": "r", "available": True, "rental_type": "spot"}],
            }
        ]
    }
    assert pick_shadeform_offer(catalog, "H100") is None


def test_shadeform_gpu_choice_records_actual_type_and_checks_each_quote():
    from cascade.provision.core import LaunchSpec

    p = ShadeformProvider(api_key="fixture-shade", max_price_per_hour=2.0)

    def catalog(path, params=None):
        sku = params["gpu_type"]
        return {
            "instance_types": [
                {
                    "cloud": "c",
                    "shade_instance_type": sku,
                    "configuration": {"gpu_type": sku},
                    "hourly_price": 300 if sku == "H100" else 100,
                    "availability": [{"region": "r", "available": True}],
                }
            ]
        }

    p._get = catalog
    created = []
    p._post = lambda path, body: created.append(body) or {"id": "uuid"}
    spec = LaunchSpec(
        sku="H100",
        sku_choices=("H100", "RTX4090"),
        count=1,
        image="img@sha256:" + "a" * 64,
        ssh_pubkey="fixture-public",
        name_prefix="fixture",
    )
    assert p.launch(spec) == ["uuid"]
    assert created[0]["shade_instance_type"] == "RTX4090"
    assert p.sku_of("uuid") == "RTX4090"
    assert p.machine_of("uuid").startswith("shadeform:")


def test_shadeform_uses_miner_key_without_modifying_operator_environment(monkeypatch):
    import os

    import requests

    monkeypatch.setenv("SHADEFORM_API_KEY", "fixture-operator")
    monkeypatch.setattr(requests, "Session", lambda: SimpleNamespace(headers={}))
    p = ShadeformProvider(api_key="fixture-miner")
    assert p._http().headers["X-API-KEY"] == "fixture-miner"
    assert os.environ["SHADEFORM_API_KEY"] == "fixture-operator"


def test_confirmed_teardown_forgets_snapshot(pool):
    _, vault, rent, _ = pool
    result = rent()
    assert teardown_funded([result.pod], vault) == []
    assert pod_credential(vault, result.pod) is None


def test_new_key_list_requires_consent_before_wallet_access(monkeypatch):
    from cascade.miner import cli

    monkeypatch.setenv("FIXTURE_SHADE", "fixture-key")

    def refuse_wallet(*args):
        raise AssertionError("must not touch wallet before consent")

    monkeypatch.setattr(cli, "load_signing_wallet", refuse_wallet)
    code = cli.main(
        [
            "credentials",
            "https://intake.example",
            "--ref",
            REF,
            "--provider-key-env",
            "shadeform=FIXTURE_SHADE",
            "--wallet-name",
            "fixture",
            "--wallet-hotkey",
            "fixture",
            "--json",
        ]
    )
    assert code == 5


def test_adoption_failure_retains_pod_and_credential_for_recovery(pool):
    accounts, vault, rent, _ = pool
    existing = funded_pod_name("777", HK, 91) + "-0"
    accounts["shadeform"].pods["existing-uuid"] = existing

    def failed_identity(pod_id):
        raise ProvisionError("temporary identity API failure")

    accounts["shadeform"].pod_identity = failed_identity
    result = rent()
    assert not result.ok and result.leaked_pod == "existing-uuid"
    from cascade.funding.providers import lease_records

    assert any(c == KEYS[1] for _, _, c in lease_records(vault, HK))
    assert not any(c[0] == "create" for a in accounts.values() for c in a.calls)


def test_live_status_preserves_advertised_provider_capabilities():
    from cascade.miner.live_status import status_snapshot
    from tests.unit.test_live_status import NOW, docs

    source = docs()
    source["competition"]["compute"] = {"rolling_providers": ["lium", "shadeform"]}
    assert (
        status_snapshot(source, now=NOW, max_age=180)["compute"] == source["competition"]["compute"]
    )


def test_credentials_update_is_signed_and_does_not_queue_work(monkeypatch, tmp_path, capsys):
    from cascade.miner import cli

    intake, clock = make_intake(tmp_path, require_signature=True, verify=verify)
    # Use the intake's real fresh-clock checks; CLI signs current wall time.
    import time

    clock.t = time.time()
    monkeypatch.setenv("FIXTURE_SHADE", KEYS[1].api_key)
    monkeypatch.setattr(cli, "load_signing_wallet", lambda *a: (HK, sign))

    def post(request, out, **kw):
        assert request.full_url.endswith("/v1/credentials")
        headers = {key.lower(): value for key, value in request.header_items()}

        class CaseInsensitive:
            def get(self, key, default=None):
                return headers.get(key.lower(), default)

        return intake.credentials(CaseInsensitive())

    monkeypatch.setattr(cli, "post_intake", post)
    assert (
        cli.main(
            [
                "credentials",
                "https://intake.example",
                "--ref",
                REF,
                "--provider-key-env",
                "shadeform=FIXTURE_SHADE",
                "--wallet-name",
                "fixture",
                "--wallet-hotkey",
                "fixture",
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert KEYS[1].api_key not in output
    assert json.loads(output)["data"]["providers"] == ["shadeform"]
    assert decode_credentials(intake.vault.get(HK)) == (KEYS[1],)
    assert intake.queue.get(HK) is None


def test_retired_credential_errors_are_redacted(pool, monkeypatch):
    import cascade.provision.funded as funded
    from cascade.funding.providers import lease_credentials

    accounts, vault, rent, _ = pool
    retired = PayerCredential("lium", "fixture-retired-secret")
    lease_credentials(vault, HK, retired, scope=funded_pod_name("777", HK, 91))

    def factory(provider, key):
        if key == retired.api_key:
            raise ProvisionError("invalid api key " + key)
        return accounts[provider]

    monkeypatch.setattr(funded, "provider_for_key", factory)
    result = rent()
    assert not result.ok
    assert retired.api_key not in result.error


def test_claimed_lium_capacity_does_not_block_shadeform_fallback(pool):
    from cascade.provision.payer_pool import payer_capacity

    accounts, _, rent, _ = pool
    seen = []

    def available(sku, count, *, gpus=1, exclude_ids=()):
        seen.append(exclude_ids)
        return "already-claimed" not in exclude_ids

    accounts["lium"].available = available
    result = rent(exclude_ids=("already-claimed",))
    assert result.ok and result.pod.provider == "shadeform"
    assert seen == [("already-claimed",)]
    assert payer_capacity(BUNDLE, ("H100",), {}, ("already-claimed",)) == 1
    assert seen[-1] == ("already-claimed",)


def test_legacy_single_key_uses_durable_pool_adoption_when_vault_is_available(pool):
    accounts, vault, _, _ = pool
    legacy = KEYS[0].api_key
    vault.insert(HK, legacy)
    options = dict(round_id='777', hotkey=HK, api_key=legacy, sku='H100',
        skus=('H100',), netuid=91, image='fixture', ssh_pubkey='fixture', vault=vault)
    result = rent_funded_pod(**options)
    assert result.ok
    created = sum(c[0] == 'create' for c in accounts['lium'].calls)
    adopted = rent_funded_pod(**options)
    assert adopted.ok and adopted.adopted
    assert sum(c[0] == 'create' for c in accounts['lium'].calls) == created


def test_single_key_without_vault_refuses_create_when_existing_pods_cannot_be_checked():
    account = Account('lium')
    def unavailable(prefix):
        raise ProvisionError('HTTP 503 account listing unavailable')
    account.list_tagged = unavailable
    result = rent_funded_pod(round_id='777', hotkey=HK, api_key='fixture-key', sku='H100',
        image='fixture', ssh_pubkey='fixture', provider_factory=lambda key: account)
    assert not result.ok and not result.burn_attempt and result.leaked_pod
    assert not any(c[0] == 'create' for c in account.calls)
