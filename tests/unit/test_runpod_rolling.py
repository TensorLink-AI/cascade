"""Runpod rolling rentals with a fake HTTP account; no paid or live calls."""
from __future__ import annotations

import pytest

from cascade.funding.providers import PayerCredential, encode_credentials, spending_controls
from cascade.funding.vault import PayerKeyVault
from cascade.provision.core import PriceCapUnavailable, ProvisionError
from cascade.provision.funded import rent_funded_pod, teardown_funded
from cascade.provision.runpod import RunPodProvider
from tests.unit.test_funding_intake import HK


class Server:
    def __init__(self):
        self.pods = []
        self.creates = 0
        self.lag = False
        self.lose_response = False

    def __call__(self, method, url, *, headers, json=None, **kw):
        assert headers["Authorization"] == "Bearer fixture-runpod"
        if method == "GET":
            return 200, [] if self.lag else self.pods
        if method == "POST":
            self.creates += 1
            pod = {"id": f"pod-{self.creates}", "name": json["name"], "desiredStatus": "RUNNING",
                   "publicIp": "192.0.2.10", "portMappings": {"22": 40022},
                   "gpuTypeId": "NVIDIA GeForce RTX 4090", "gpu": {"count": 1}}
            self.pods.append(pod)
            if self.lose_response:
                raise TimeoutError("no_capacity; invalid api key; response lost")
            return 201, pod
        if method == "DELETE":
            self.pods = [p for p in self.pods if not url.endswith(p["id"])]
            return 204, None
        raise AssertionError(method)


@pytest.fixture
def account(tmp_path, monkeypatch):
    import cascade.provision.funded as funded

    server = Server()
    monkeypatch.setattr(funded, "provider_for_key", lambda provider, key:
                        RunPodProvider(api_key=key, _http=server))
    vault = PayerKeyVault(dir=tmp_path / "vault", seal_key=b"x" * 32)
    bundle = encode_credentials((PayerCredential("runpod", "fixture-runpod"),))
    vault.insert(HK, bundle)

    def rent(**kw):
        return rent_funded_pod(round_id="777", hotkey=HK, api_key=bundle, sku="RTX4090",
            skus=("RTX4090",), netuid=91, image="image@sha256:" + "a" * 64,
            ssh_pubkey="fixture-public", vault=vault, **kw)

    return server, vault, rent


def test_uncapped_authorized_runpod_rental_is_adopted_and_cleaned_up(account):
    server, vault, rent = account
    first = rent()
    assert first.ok and first.pod.provider == "runpod" and first.sku == "RTX4090"
    second = rent()
    assert second.ok and second.adopted and server.creates == 1
    assert teardown_funded([second.pod], vault) == [] and not server.pods


def test_price_capped_runpod_never_crosses_paid_boundary(account):
    server, vault, rent = account
    result = rent(price_caps={"max_price_per_hour": 1.0})
    assert not result.ok and result.error_class == "price_cap_unavailable"
    assert not result.burn_attempt and server.creates == 0


def test_lost_create_response_never_authorizes_another_runpod_rental(account):
    server, vault, rent = account
    server.lose_response = True
    result = rent()
    assert not result.ok and result.leaked_pod and not result.burn_attempt
    server.lag = True
    server.lose_response = False
    assert "create_unresolved" in rent().error
    assert server.creates == 1
    server.lag = False
    assert rent().adopted and server.creates == 1


@pytest.mark.parametrize("body", [None, {}, {"error": "failure"}, [None], [{}]])
def test_malformed_listing_is_never_empty_account_evidence(body):
    provider = RunPodProvider(api_key="fixture", _http=lambda *a, **kw: (200, body))
    with pytest.raises(ProvisionError, match="malformed pod listing"):
        provider.list_tagged("cascade")


def test_price_caps_are_rechecked_inside_the_adapter():
    provider = RunPodProvider(api_key="fixture", max_leg_cost_usd=5)
    with pytest.raises(PriceCapUnavailable):
        provider.capacity("RTX4090")
    assert spending_controls()["runpod"]["key_budget_verified"] is False
