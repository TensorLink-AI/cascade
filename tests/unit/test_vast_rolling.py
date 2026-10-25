"""Vast.ai in the rolling payer pool: the real adapter against a fake Vast API."""
from __future__ import annotations

import pytest

from cascade.funding.providers import (
    PayerCredential,
    credentials_from_env,
    decode_credentials,
    encode_credentials,
)
from cascade.funding.vault import PayerKeyVault
from cascade.provision.funded import funded_pod_name, rent_funded_pod, teardown_funded
from cascade.provision.vast import VastProvider
from tests.unit.test_funding_intake import HK
from tests.unit.test_payer_providers import Account

BUNDLE = encode_credentials((PayerCredential("lium", "fixture-lium"),
                             PayerCredential("vast", "fixture-vast")))


class VastServer:
    """Offers, create, paged v1 listing and destroy, with an optional lost response."""

    def __init__(self):
        self.offers = [{"id": 31, "machine_id": 310, "num_gpus": 1, "dph_total": 0.35,
                        "gpu_name": "RTX 4090"}]
        self.instances: dict[int, dict] = {}
        self.creates = 0
        self.lose_create_response = False
        self.list_lag = False

    def __call__(self, method, url, *, headers, json=None, timeout=30.0):
        assert headers["Authorization"] == "Bearer fixture-vast"
        if method == "POST" and url.endswith("/bundles/"):
            return 200, {"offers": list(self.offers)}
        if method == "PUT" and "/asks/" in url:
            self.creates += 1
            iid = 9000 + self.creates
            self.instances[iid] = {
                "id": iid, "label": json["label"], "actual_status": "running",
                "public_ipaddr": "203.0.113.5", "machine_id": 310, "gpu_name": "RTX 4090",
                "num_gpus": 1, "ports": {"22/tcp": [{"HostPort": "40022"}]}}
            if self.lose_create_response:
                raise TimeoutError("Read timed out")
            return 200, {"success": True, "new_contract": iid}
        if method == "GET" and "/api/v1/instances/" in url:
            rows = [] if self.list_lag else list(self.instances.values())
            return 200, {"success": True, "instances": rows, "instances_found": len(rows),
                         "total_instances": len(rows), "next_token": None}
        if method == "DELETE" and "/instances/" in url:
            self.instances.pop(int(url.rstrip("/").rsplit("/", 1)[-1]), None)
            return 200, {"success": True}
        return 404, {"error": "unexpected"}


@pytest.fixture
def market(monkeypatch, tmp_path):
    import cascade.provision.funded as funded

    server = VastServer()
    lium = Account("lium", available=False)            # Lium sold out
    monkeypatch.setattr(funded, "provider_for_key", lambda provider, key: (
        lium if provider == "lium" else VastProvider(api_key=key, _http=server)))
    vault = PayerKeyVault(dir=tmp_path / "vault", seal_key=b"x" * 32)
    vault.insert(HK, BUNDLE)

    def rent():
        return rent_funded_pod(round_id="777", hotkey=HK, api_key=BUNDLE, sku="RTX4090",
                               skus=("RTX4090",), netuid=91, image="image@sha256:" + "a" * 64,
                               ssh_pubkey="fixture-public", vault=vault)

    return server, lium, vault, rent


def test_vast_is_a_rolling_provider_selected_by_environment_name(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("MY_VAST", "fixture-vast")
    args = SimpleNamespace(lium_key_env="LIUM_API_KEY", provider_key_env=["vast=MY_VAST"])
    assert decode_credentials(credentials_from_env(args)) == (PayerCredential("vast", "fixture-vast"),)


def test_sold_out_lium_falls_back_to_one_vast_rental(market):
    server, lium, vault, rent = market
    result = rent()
    assert result.ok and result.pod.provider == "vast" and server.creates == 1
    assert result.address.ip == "203.0.113.5" and result.pod.payer_credential_id
    assert not any(c[0] == "create" for c in lium.calls)
    # Teardown routes to the Vast account through the retained lease and confirms.
    orphaned = teardown_funded([result.pod], vault)
    assert not orphaned and not server.instances


def test_uncertain_vast_create_never_rents_again(market):
    server, lium, vault, rent = market
    server.lose_create_response = True
    server.list_lag = True                              # the new instance is not listed yet
    first = rent()
    assert not first.ok and first.leaked_pod and not first.burn_attempt
    server.lose_create_response = False
    second = rent()
    assert not second.ok and "create_unresolved" in second.error
    assert server.creates == 1                          # no second bill
    pod_name = funded_pod_name("777", HK, 91) + "-0"
    server.list_lag = False                             # it appears: adopted, not duplicated
    third = rent()
    assert third.ok and third.adopted and server.creates == 1
    assert [r["label"] for r in server.instances.values()] == [pod_name]


def test_vast_refusal_is_capacity_not_an_unresolved_create(market, monkeypatch):
    import cascade.provision.funded as funded

    server, lium, vault, rent = market

    def refuse(method, url, **kw):
        if method == "PUT":
            return 400, {"error": "no_such_ask"}
        return server(method, url, **kw)

    monkeypatch.setattr(funded, "provider_for_key", lambda provider, key: (
        lium if provider == "lium" else VastProvider(api_key=key, _http=refuse)))
    result = rent()
    assert not result.ok and result.error_class == "no_capacity" and not result.leaked_pod
    assert not result.burn_attempt and server.creates == 0


@pytest.mark.parametrize("status,body", [
    (200, None), (200, {}), (200, {"raw": "no_capacity; invalid api key"}),
    (200, {"success": True}), (200, {"success": "false"}),
    (200, {"success": True, "new_contract": False}),
    (200, {"success": False, "new_contract": 9001}),
    (502, {"error": "no_capacity; invalid api key"}),
    (408, {"msg": "request timed out"}),
])
def test_ambiguous_create_response_retains_intent_across_restart(market, monkeypatch, status, body):
    import cascade.provision.funded as funded

    server, lium, vault, rent = market

    def ambiguous(method, url, **kw):
        code, response = server(method, url, **kw)
        return (status, body) if method == "PUT" else (code, response)

    monkeypatch.setattr(funded, "provider_for_key", lambda provider, key: (
        lium if provider == "lium" else VastProvider(api_key=key, _http=ambiguous)))
    first = rent()
    assert not first.ok and first.error_class == "infra" and first.leaked_pod
    assert not first.burn_attempt and server.creates == 1
    server.list_lag = True
    # Reopen the durable vault and use new provider objects, as a trainer restart does.
    reopened = PayerKeyVault(dir=vault.dir, seal_key=b"x" * 32)
    second = rent_funded_pod(round_id="777", hotkey=HK, api_key=BUNDLE, sku="RTX4090",
        skus=("RTX4090",), netuid=91, image="image@sha256:" + "a" * 64,
        ssh_pubkey="fixture-public", vault=reopened)
    assert not second.ok and "create_unresolved" in second.error
    assert server.creates == 1 and len(server.instances) == 1
    server.list_lag = False
    adopted = rent()
    assert adopted.ok and adopted.adopted and server.creates == 1


@pytest.mark.parametrize("body", [
    {"success": False, "msg": "temporary failure"},
    {"success": True, "instances": [], "instances_found": 0,
     "total_instances": 1, "next_token": None},
])
def test_invalid_listing_preserves_cleanup_lease_and_allocation(market, monkeypatch, body):
    import cascade.provision.funded as funded
    from cascade.funding.providers import lease_records

    server, lium, vault, rent = market
    result = rent()
    assert result.ok and server.creates == 1

    def invalid(method, url, **kw):
        if method == "GET":
            return 200, body
        return server(method, url, **kw)

    monkeypatch.setattr(funded, "provider_for_key", lambda provider, key: (
        lium if provider == "lium" else VastProvider(api_key=key, _http=invalid)))
    assert teardown_funded([result.pod], vault) == [result.pod]
    assert len(server.instances) == 1 and lease_records(vault, HK)
    retry = rent()
    assert not retry.ok and not retry.burn_attempt and server.creates == 1


@pytest.mark.parametrize("body", [
    {}, {"success": False},
    {"success": True, "instances": None, "instances_found": 0, "total_instances": 0, "next_token": None},
    {"success": True, "instances": [None], "instances_found": 1, "total_instances": 1, "next_token": None},
    {"success": True, "instances": [{}], "instances_found": 1, "total_instances": 1, "next_token": None},
    {"success": True, "instances": [], "instances_found": False, "total_instances": 0, "next_token": None},
    {"success": True, "instances": [], "instances_found": 1, "total_instances": 0, "next_token": None},
    {"success": True, "instances": [], "instances_found": 0, "total_instances": 1, "next_token": None},
    {"success": True, "instances": [], "instances_found": 0, "total_instances": 0},
    {"success": True, "instances": [], "instances_found": 0, "total_instances": 0, "next_token": ""},
])
def test_malformed_account_listing_never_establishes_absence(body):
    from cascade.provision.core import ProvisionError
    from cascade.provision.funded import terminate_verified

    provider = VastProvider(api_key="fixture-review", _http=lambda *args, **kwargs: (200, body))
    with pytest.raises(ProvisionError):
        terminate_verified(provider, "paid-pod-0")


def test_valid_empty_account_confirms_teardown_only_after_an_observation_or_a_delete_ack():
    from cascade.provision.funded import terminate_verified

    empty = {"success": True, "instances": [], "instances_found": 0,
             "total_instances": 0, "next_token": None}
    # Never observed, nothing to DELETE: an empty listing alone confirms
    # nothing (the v1 listing lags a fresh create; the instance may be live).
    provider = VastProvider(api_key="fixture-review", _http=lambda *args, **kwargs: (200, empty))
    assert not terminate_verified(provider, "already-deleted-pod-0")
    # Observed once (then destroyed and no longer listed): confirmed gone.
    rows = [{"id": 5, "label": "paid-pod-0", "actual_status": "running"}]
    deletes = []

    def http(method, url, **kw):
        if method == "DELETE":
            deletes.append(url)
            rows.clear()
            return 200, {"success": True}
        return 200, {**empty, "instances": list(rows), "instances_found": len(rows),
                     "total_instances": len(rows)}

    provider = VastProvider(api_key="fixture-review", _http=http)
    assert terminate_verified(provider, "paid-pod-0") and len(deletes) == 1
    # Never listed but the create handed back a contract: the DELETE goes to
    # that contract and its acknowledgement (here a 404) is the confirmation.
    seen = []

    def lagging(method, url, **kw):
        if method == "DELETE":
            seen.append(url)
            return 404, {"error": "not found"}
        return 200, empty

    provider = VastProvider(api_key="fixture-review", _http=lagging)
    provider._contract_by_name["paid-pod-0"] = "9001"
    assert terminate_verified(provider, "paid-pod-0") and seen[0].endswith("/instances/9001/")


def test_duplicate_instances_across_pages_do_not_establish_a_complete_listing():
    from cascade.provision.core import ProvisionError

    page = {"success": True, "instances": [{"id": 1, "label": "paid-pod-0"}],
            "instances_found": 1, "total_instances": 2}

    def http(method, url, **kw):
        return 200, {**page, "next_token": None if "after_token" in url else "page2"}

    with pytest.raises(ProvisionError, match="duplicate instance"):
        VastProvider(api_key="fixture-review", _http=http).list_tagged("paid")


def test_transport_error_text_cannot_clear_an_uncertain_create(market, monkeypatch):
    import cascade.provision.funded as funded

    server, lium, vault, rent = market

    def transport_failure(method, url, **kw):
        response = server(method, url, **kw)
        if method == "PUT":
            raise TimeoutError("invalid api key; no_capacity; response lost")
        return response

    monkeypatch.setattr(funded, "provider_for_key", lambda provider, key: (
        lium if provider == "lium" else VastProvider(api_key=key, _http=transport_failure)))
    first = rent()
    assert first.error_class == "infra" and first.leaked_pod and not first.burn_attempt
    server.list_lag = True
    assert "create_unresolved" in rent().error
    assert server.creates == 1
