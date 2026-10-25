"""Paid boundaries are exercised only through in-memory provider doubles."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.provision.core import LaunchSpec, PodAddress, ProvisionError
from cascade.provision.operator_pool import OperatorPool, SpendBook
from cascade.shared.config import RoundConfig


class FakeProvider:
    def __init__(self, name, capacity=1):
        self.name, self.free = name, capacity
        self.pods, self.effects = {}, []
        self.uncertain = False
        self.delete_fails = False

    def available(self, *args, **kwargs):
        return bool(self.free)

    def capacity(self, *args, **kwargs):
        return self.free

    def list_tagged(self, prefix):
        return [pid for pid, name in self.pods.items() if name.startswith(prefix)]

    def launch(self, spec):
        self.effects.append(spec)
        pid = "opaque-uuid" if self.name == "shadeform" else f"{spec.name_prefix}-0"
        self.pods[pid] = f"{spec.name_prefix}-0"
        if self.uncertain:
            raise TimeoutError("fixture create response lost")
        return [pid]

    def terminate(self, pod):
        if not self.delete_fails:
            self.pods.pop(pod, None)

    def pod_exists(self, pod):
        return pod in self.pods

    def sku_of(self, pod):
        return "H100"

    def machine_of(self, pod):
        return f"{self.name}:machine"

    def live_pod_address(self, pod):
        return PodAddress("127.0.0.1", 22) if pod in self.pods else None


def pool_fixture(tmp_path, monkeypatch, *, clock=lambda: 1000):
    providers = {"lium": FakeProvider("lium", 0), "shadeform": FakeProvider("shadeform")}
    monkeypatch.setenv("LIUM_API_KEY", "fixture-lium")
    monkeypatch.setenv("SHADEFORM_API_KEY", "fixture-shade")
    rnd = replace(
        RoundConfig(),
        ops_operator_providers=("lium", "shadeform"),
        ops_operator_leg_cap_usd=10,
        ops_operator_daily_cap_usd=25,
        funded_max_price_per_hour=1,
    )
    runner = SimpleNamespace(
        work_root=tmp_path,
        cfg=SimpleNamespace(round=rnd),
        _funded_price_caps=lambda: {},
        _leg_wall_seconds=lambda: 3600,
    )

    def factory(name, key):
        return providers[name]

    pool = OperatorPool(runner, factory=factory, clock=clock)
    return pool, providers, runner, factory


def spec(prefix="cascade-fixture-king"):
    return LaunchSpec(
        sku="H100", count=1, image="fixture", ssh_pubkey="fixture", name_prefix=prefix
    )


def test_operator_fallback_keeps_opaque_id_and_original_account_across_restart(
    tmp_path, monkeypatch
):
    pool, providers, runner, factory = pool_fixture(tmp_path, monkeypatch)
    alias = pool.launch(spec())[0]
    assert alias == "cascade-fixture-king-0"
    assert len(providers["shadeform"].effects) == 1 and not providers["lium"].effects
    monkeypatch.setenv("SHADEFORM_API_KEY", "fixture-rotated")
    keys = []

    def restarted_factory(name, key):
        keys.append(key)
        return factory(name, key)

    restarted = OperatorPool(runner, factory=restarted_factory, clock=lambda: 1001)
    assert restarted.launch(spec()) == [alias]
    assert "fixture-shade" in keys
    assert len(providers["shadeform"].effects) == 1
    restarted.terminate(alias)
    assert not providers["shadeform"].pods
    assert restarted.book.rows()[0]["status"] == "closed"


def test_unknown_create_never_fails_over_and_can_adopt_original_pod(tmp_path, monkeypatch):
    pool, providers, runner, factory = pool_fixture(tmp_path, monkeypatch)
    providers["lium"].free = 1
    providers["lium"].uncertain = True
    with pytest.raises(TimeoutError):
        pool.launch(spec())
    assert not providers["shadeform"].effects
    restarted = OperatorPool(runner, factory=factory, clock=lambda: 1001)
    assert restarted.launch(spec()) == ["cascade-fixture-king-0"]
    assert len(providers["lium"].effects) == 1 and not providers["shadeform"].effects


def test_unknown_create_with_empty_listing_keeps_reservation_and_refuses(tmp_path, monkeypatch):
    pool, providers, _, _ = pool_fixture(tmp_path, monkeypatch)
    providers["shadeform"].uncertain = True
    with pytest.raises(TimeoutError):
        pool.launch(spec())
    providers["shadeform"].pods.clear()
    with pytest.raises(ProvisionError, match="unresolved"):
        pool.launch(spec())
    assert len(providers["shadeform"].effects) == 1
    assert pool.book.rows()[0]["reserved_usd"] > 0


def test_failed_shadeform_delete_is_not_treated_as_gone(tmp_path, monkeypatch):
    from cascade.provision.core import ShadeformProvider
    from cascade.provision.funded import terminate_verified

    provider = ShadeformProvider(api_key="fixture")
    monkeypatch.setattr(provider, "_post", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        provider,
        "_get",
        lambda *args, **kwargs: {
            "instances": [{"id": "opaque-uuid", "name": "cascade-fixture", "status": "active"}]
        },
    )
    assert not terminate_verified(provider, "opaque-uuid")
    monkeypatch.setattr(provider, "_get", lambda *args, **kwargs: {"instances": []})
    assert terminate_verified(provider, "opaque-uuid")


def test_caps_persist_after_teardown_and_restart(tmp_path):
    book = SpendBook(tmp_path / "spend.json", clock=lambda: 1000)
    args = dict(
        provider="lium", credential_id="opaque", amount=2, seconds=3600, leg_cap=3, daily_cap=3
    )
    row = book.reserve("leg-a", **args)
    book.update(row["attempt"], status="closed")
    restarted = SpendBook(book.path, clock=lambda: 1001)
    with pytest.raises(ProvisionError, match="leg spending"):
        restarted.reserve("leg-a", **args)
    with pytest.raises(ProvisionError, match="daily spending"):
        restarted.reserve("leg-b", **args)
    assert len(book.rows()) == 1


def test_expiry_reaper_retains_unconfirmed_deletion(tmp_path, monkeypatch):
    now = [1000]
    pool, providers, _, _ = pool_fixture(tmp_path, monkeypatch, clock=lambda: now[0])
    pool.launch(spec())
    now[0] += 100000
    providers["shadeform"].delete_fails = True
    pool.reap_expired()
    assert pool.book.rows()[0]["status"] == "allocated"
    providers["shadeform"].delete_fails = False
    pool.reap_expired()
    assert pool.book.rows()[0]["status"] == "closed"


def test_unusable_unused_operator_account_can_fall_back_without_quarantining_hosts(
    tmp_path, monkeypatch
):
    pool, providers, _, _ = pool_fixture(tmp_path, monkeypatch)

    def rejected(prefix):
        raise ProvisionError("HTTP 402 insufficient balance")

    providers["lium"].list_tagged = rejected
    assert pool.launch(spec()) == ["cascade-fixture-king-0"]
    assert len(providers["shadeform"].effects) == 1
    assert not providers["lium"].effects
