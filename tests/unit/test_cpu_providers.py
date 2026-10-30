"""DEC-CA-0053 CPU VM providers: offer picking, provider order, lifecycle
against fake sessions. No live calls."""

from __future__ import annotations

import pytest

from cascade.provision.core import PodAddress, ProvisionError, pick_shadeform_cpu_offer
from cascade.provision.cpu import (
    CpuShape,
    ShadeformCpuProvider,
    cpu_providers,
    select_cpu_offer,
)
from cascade.provision.targon import (
    TargonCpuProvider,
    parse_cpu_inventory,
    targon_vm_address,
    targon_vm_name,
)
from cascade.shared.config import GenerationConfig

# The live CPU inventory shape (GET /tha/v3/inventory?type=vm, 2026-10-18).
INVENTORY = [
    {"name": "cpu-xlarge-amd", "type": "vm", "gpu": False, "available": 2,
     "cost_per_hour": 1, "spec": {"vcpu": 32000, "memory": 429153}},
    {"name": "cpu-large-amd", "type": "vm", "gpu": False, "available": 5,
     "cost_per_hour": 0.6, "spec": {"vcpu": 16000, "memory": 214576}},
    {"name": "cpu-small-amd", "type": "vm", "gpu": False, "available": 22,
     "cost_per_hour": 0.15, "spec": {"vcpu": 4000, "memory": 53644}},
    {"name": "h200-small", "type": "vm", "gpu": True, "available": 5,
     "cost_per_hour": 2.49, "spec": {"vcpu": 12000, "memory": 115000}},
]


class _Resp:
    def __init__(self, data=None, status=200):
        self._data = data
        self.status_code = status
        self.content = b"" if data is None else b"x"

    def raise_for_status(self):
        if self.status_code >= 400:
            err = RuntimeError(f"HTTP {self.status_code}")
            err.response = self                 # as requests.HTTPError carries it
            raise err

    def json(self):
        return self._data


class _Session:
    """Routes ``(method, path-suffix)`` to canned responses; records calls."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, str, object]] = []

    def _go(self, method, url, **kw):
        self.calls.append((method, url, kw.get("json")))
        for (m, suffix), resp in self.routes.items():
            if m == method and url.endswith(suffix):
                return resp(kw) if callable(resp) else resp
        return _Resp({}, 404)

    def get(self, url, **kw):
        return self._go("get", url, **kw)

    def post(self, url, **kw):
        return self._go("post", url, **kw)

    def delete(self, url, **kw):
        return self._go("delete", url, **kw)


# ── Targon ───────────────────────────────────────────────────────────────────


def test_targon_resolves_exact_vm_name_across_pages(monkeypatch):
    monkeypatch.setenv("TARGON_ORG", "acme")
    queries = []

    def page(kw):
        queries.append(kw["params"])
        if kw["params"].get("cursor") == "next":
            return _Resp({"items": [{"type": "VM", "name": "cgen-x", "uid": "vm2"}]})
        return _Resp({"items": [
            {"type": "VM", "name": "cgen-x", "uid": "vm1"},
            {"type": "VM", "name": "cgen-x-other", "uid": "unrelated"},
            {"type": "BM", "name": "cgen-x", "uid": "baremetal"},
        ], "next_cursor": "next"})

    p = TargonCpuProvider(_session=_Session({("get", "/workloads"): page}))
    assert p.find_by_name("cgen-x") == ["vm1", "vm2"]
    assert queries == [{"type": "VM", "name": "cgen-x", "limit": 1000},
                       {"type": "VM", "name": "cgen-x", "limit": 1000, "cursor": "next"}]


@pytest.mark.parametrize("response", [{}, {"items": None}, {"items": [None]},
    {"items": [{"type": "VM", "name": "cgen-x"}]},
    {"items": [], "next_cursor": "loop"}])
def test_targon_bad_listing_never_returns_false_absence(monkeypatch, response):
    monkeypatch.setenv("TARGON_ORG", "acme")
    p = TargonCpuProvider(_session=_Session({("get", "/workloads"): _Resp(response)}))
    with pytest.raises(ProvisionError):
        p.find_by_name("cgen-x")


def test_inventory_parses_millicores_and_filters():
    shapes = parse_cpu_inventory(INVENTORY, 32, 1.25)
    assert shapes == [CpuShape(vcpus=32, memory_gb=pytest.approx(419.1, abs=0.1),
                               price_per_hour=1.0, provider="targon",
                               type_id="cpu-xlarge-amd")]
    assert [s.type_id for s in parse_cpu_inventory(INVENTORY, 4, 0.7)] == [
        "cpu-small-amd", "cpu-large-amd"]
    assert parse_cpu_inventory(INVENTORY, 32, 0.99) == []          # over the price cap
    assert parse_cpu_inventory(INVENTORY, 64, 10.0) == []          # nothing that wide


def test_inventory_skips_unavailable_and_falls_back_to_static_table():
    items = [{"name": "cpu-xlarge-amd", "type": "vm", "gpu": False, "available": 0,
              "cost_per_hour": 1, "spec": {"vcpu": 32000}},
             {"name": "cpu-large-amd", "type": "vm", "gpu": False, "available": 3}]
    (shape,) = parse_cpu_inventory(items, 8, 5.0)
    assert (shape.type_id, shape.vcpus, shape.price_per_hour) == ("cpu-large-amd", 16, 0.60)


def test_targon_offers_need_no_key(monkeypatch):
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    sess = _Session({("get", "/tha/v3/inventory"): _Resp(INVENTORY)})
    p = TargonCpuProvider(_public=sess)
    assert [s.type_id for s in p.offers(32, 2.0)] == ["cpu-xlarge-amd"]


def test_targon_missing_key_is_lazy(monkeypatch):
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    monkeypatch.setenv("TARGON_ORG", "acme")
    p = TargonCpuProvider(vm_image="ubuntu-24.04")      # construction reads nothing
    with pytest.raises(ProvisionError, match="TARGON_API_KEY"):
        p.launch(parse_cpu_inventory(INVENTORY, 32, 2.0)[0], name="g", ssh_pubkey="ssh-ed25519 A")


def test_targon_launch_registers_key_creates_and_deploys(monkeypatch):
    monkeypatch.setenv("TARGON_ORG", "acme")
    sess = _Session({
        ("post", "/orgs/acme/ssh-keys"): _Resp({"uid": "key-1"}),
        ("post", "/orgs/acme/workloads"): _Resp({"uid": "wl-1"}),
        ("post", "/workloads/wl-1/deploy"): _Resp({"uid": "wl-1"}),
    })
    p = TargonCpuProvider(_session=sess, vm_image="ubuntu-24.04")
    shape = parse_cpu_inventory(INVENTORY, 32, 2.0)[0]
    assert p.launch(shape, name="Cascade_Gen round-7", ssh_pubkey="ssh-ed25519 A") == "wl-1"
    create = next(body for m, url, body in sess.calls if url.endswith("/orgs/acme/workloads"))
    assert create == {"type": "VM", "name": "cascade-gen-round-7", "image": "ubuntu-24.04",
                      "resource_name": "cpu-xlarge-amd", "ssh_keys": ["key-1"]}
    p.launch(shape, name="second", ssh_pubkey="ssh-ed25519 A")
    assert sum(url.endswith("/ssh-keys") for _, url, _ in sess.calls) == 1  # key uid cached


def test_targon_failed_deploy_tears_down(monkeypatch):
    monkeypatch.setenv("TARGON_ORG", "acme")
    sess = _Session({
        ("post", "/ssh-keys"): _Resp({"uid": "k"}),
        ("post", "/orgs/acme/workloads"): _Resp({"uid": "wl-9"}),
        ("post", "/deploy"): _Resp({}, 500),
        ("delete", "/workloads/wl-9"): _Resp(None, 204),
    })
    p = TargonCpuProvider(_session=sess, vm_image="img")
    with pytest.raises(RuntimeError):
        p.launch(parse_cpu_inventory(INVENTORY, 32, 2.0)[0], name="g", ssh_pubkey="k")
    assert ("delete", "https://api.targon.com/tha/v3/orgs/acme/workloads/wl-9", None) in sess.calls


def test_targon_wait_ready_and_address(monkeypatch):
    monkeypatch.setenv("TARGON_ORG", "acme")
    states = iter([{"status": "provisioning"},
                   {"status": "running", "state": {"public_ip": "203.0.113.42", "ssh_port": 4022}}])
    sess = _Session({("get", "/workloads/wl-1/state"): lambda kw: _Resp(next(states))})
    p = TargonCpuProvider(_session=sess, _sleep=lambda s: None)
    assert p.wait_ready("wl-1", timeout=60) is True
    assert targon_vm_address({"public_ip": "1.2.3.4", "ssh_port": 4022}) == PodAddress("1.2.3.4", 4022)
    assert targon_vm_address({"status": "running"}) is None


def test_targon_bad_state_raises(monkeypatch):
    monkeypatch.setenv("TARGON_ORG", "acme")
    sess = _Session({("get", "/state"): _Resp({"status": "error"})})
    with pytest.raises(ProvisionError, match="error"):
        TargonCpuProvider(_session=sess).wait_ready("wl-1", timeout=60)


def test_targon_terminate_is_idempotent(monkeypatch):
    monkeypatch.setenv("TARGON_ORG", "acme")
    sess = _Session({})                       # every call 404s
    TargonCpuProvider(_session=sess).terminate("gone")
    TargonCpuProvider(_session=sess).terminate("gone")


def test_targon_terminate_failure_retries_then_raises(monkeypatch):
    from cascade.provision.core import ProvisionError

    monkeypatch.setenv("TARGON_ORG", "acme")
    sess = _Session({("delete", "/vm-1"): _Resp({}, 503)})
    prov = TargonCpuProvider(_session=sess, _sleep=lambda s: None)
    with pytest.raises(ProvisionError, match="delete vm-1 failed"):
        prov.terminate("vm-1")
    assert [m for m, _, _ in sess.calls] == ["delete"] * 3


def test_targon_vm_name_rules():
    assert targon_vm_name("Cascade_GEN__x") == "cascade-gen-x"
    assert len(targon_vm_name("a" * 50)) == 32
    assert targon_vm_name("!!!") == "cascade-gen"


# ── Shadeform ────────────────────────────────────────────────────────────────


def _st(itype, vcpus, cents, gpus=1, available=True, rental="on_demand", cloud="c", region="r"):
    return {"cloud": cloud, "shade_instance_type": itype, "hourly_price": cents,
            "configuration": {"vcpus": vcpus, "memory_in_gb": 128, "num_gpus": gpus,
                              "gpu_type": "A6000"},
            "availability": [{"region": region, "available": available, "rental_type": rental}]}


def test_shadeform_cpu_offer_filters_and_sorts():
    types = {"instance_types": [
        _st("big-cheap", 48, 90), _st("big-dear", 64, 200), _st("narrow", 16, 30),
        _st("spot-only", 64, 50, rental="spot"), _st("gone", 64, 40, available=False),
    ]}
    assert [o["shade_instance_type"] for o in pick_shadeform_cpu_offer(types, 32, 1.25)] == [
        "big-cheap"]
    assert [o["shade_instance_type"] for o in pick_shadeform_cpu_offer(types, 32, 5.0)] == [
        "big-cheap", "big-dear"]
    assert pick_shadeform_cpu_offer(types, 32, 0.5) == []


def test_shadeform_cpu_offer_prefers_gpuless():
    types = {"instance_types": [_st("gpu-box", 32, 80), _st("cpu-box", 32, 100, gpus=0)]}
    first = pick_shadeform_cpu_offer(types, 32, 2.0)[0]
    assert (first["shade_instance_type"], first["price_usd_hr"]) == ("cpu-box", 1.0)


def test_shadeform_cpu_provider_launch_docker_needs_image():
    sess = _Session({("get", "/instances/types"): _Resp({"instance_types": [_st("x", 32, 90)]}),
                     ("post", "/instances/create"): _Resp({"id": "sf-1"})})
    p = ShadeformCpuProvider(_session=sess)
    (shape,) = p.offers(32, 1.25)
    assert (shape.provider, shape.cloud, shape.region, shape.price_per_hour) == (
        "shadeform", "c", "r", 0.9)
    with pytest.raises(ProvisionError, match="image"):
        p.launch(shape, name="g", ssh_pubkey="k")
    assert p.launch(shape, name="g", ssh_pubkey="k", image="img@sha256:" + "0" * 64) == "sf-1"
    body = sess.calls[-1][2]
    assert body["shade_instance_type"] == "x" and body["launch_configuration"]["type"] == "docker"


def test_shadeform_cpu_provider_vm_mode_and_terminate():
    sess = _Session({("post", "/instances/create"): _Resp({"id": "sf-2"})})
    p = ShadeformCpuProvider(_session=sess, ssh_key_id="acct-key")
    assert p.ssh_user == "shadeform"
    shape = CpuShape(32, 128, 0.9, "shadeform", "x", "r", "c")
    assert p.launch(shape, name="g", ssh_pubkey="k") == "sf-2"
    assert sess.calls[-1][2]["ssh_key_id"] == "acct-key"
    p.terminate("sf-2")                       # delete 404s → logged, not raised


# ── selection ────────────────────────────────────────────────────────────────


class _Fake:
    def __init__(self, name, offers=(), boom=None):
        self.name, self._offers, self._boom = name, list(offers), boom

    def offers(self, min_vcpus, max_price):
        if self._boom:
            raise self._boom
        return self._offers


def test_select_walks_order_and_skips_faults():
    shape = CpuShape(32, 419, 1.0, "shadeform", "x")
    cfg = GenerationConfig()
    chosen = select_cpu_offer([_Fake("targon", boom=RuntimeError("down")),
                               _Fake("shadeform", [shape])], cfg)
    assert chosen is not None and chosen[0].name == "shadeform" and chosen[1] is shape
    first = CpuShape(32, 419, 1.0, "targon", "cpu-xlarge-amd")
    assert select_cpu_offer([_Fake("targon", [first]), _Fake("shadeform", [shape])],
                            cfg)[1] is first
    assert select_cpu_offer([_Fake("targon"), _Fake("shadeform")], cfg) is None


def test_select_propagates_provision_error():
    with pytest.raises(ProvisionError):
        select_cpu_offer([_Fake("targon", boom=ProvisionError("bad config"))], GenerationConfig())


def test_cpu_providers_factory_order(monkeypatch):
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    monkeypatch.delenv("SHADEFORM_API_KEY", raising=False)
    ps = cpu_providers(GenerationConfig(providers=("shadeform", "targon")))
    assert [p.name for p in ps] == ["shadeform", "targon"]
    assert isinstance(ps[1], TargonCpuProvider)


def test_cpu_providers_shadeform_vm_mode_from_env(monkeypatch):
    monkeypatch.delenv("SHADEFORM_API_KEY", raising=False)
    monkeypatch.setenv("SHADEFORM_SSH_KEY_ID", "k-1")
    ps = cpu_providers(GenerationConfig(providers=("shadeform",)))
    assert ps[0].ssh_key_id == "k-1" and ps[0].ssh_user == "shadeform"
    # explicit options still win over the environment default
    ps = cpu_providers(GenerationConfig(providers=("shadeform",)),
                       {"shadeform": {"ssh_key_id": "k-2"}})
    assert ps[0].ssh_key_id == "k-2"
    monkeypatch.setenv("OTHER_KEY_ID", "k-3")
    ps = cpu_providers(GenerationConfig(providers=("shadeform",),
                                        shadeform_ssh_key_id_env="OTHER_KEY_ID"))
    assert ps[0].ssh_key_id == "k-3"
    monkeypatch.delenv("SHADEFORM_SSH_KEY_ID")
    assert cpu_providers(GenerationConfig(providers=("shadeform",)))[0].ssh_key_id == ""


def test_shadeform_cpu_docker_mode_refuses_bare_image_name():
    from cascade.provision.cpu import ShadeformCpuProvider

    p = ShadeformCpuProvider()
    shape = CpuShape(32, 128.0, 1.02, "shadeform", "cpu_32", "dallas-usa-1", "latitude")
    for image in ("", "ubuntu-24-04-lts"):     # the Targon vm_image is not a container
        with pytest.raises(ProvisionError, match="SHADEFORM_SSH_KEY_ID"):
            p.launch(shape, name="n", ssh_pubkey="ssh-ed25519 AAAA x", image=image)


def test_parse_generation_shadeform_key_id_env_round_trip():
    from cascade.shared.config.generation import parse_generation

    assert parse_generation({}).shadeform_ssh_key_id_env == "SHADEFORM_SSH_KEY_ID"
    assert parse_generation({"shadeform_ssh_key_id_env": "X_ID"}).shadeform_ssh_key_id_env == "X_ID"
    with pytest.raises(ValueError, match="environment variable"):
        parse_generation({"shadeform_ssh_key_id_env": "not a name"})
    with pytest.raises(ValueError, match="not a chain.toml field"):
        parse_generation({"shadeform_ssh_key_id": "519203aa"})   # the value never lives here


# ── fallback tiers (owner 2026-10-30: 32 vCPU always; 16 only when 32 is sold out) ──


class _TierFake:
    """Offers only for requests at or below ``max_vcpus``."""

    def __init__(self, name, max_vcpus):
        self.name, self.max_vcpus = name, max_vcpus

    def offers(self, min_vcpus, max_price):
        if min_vcpus > self.max_vcpus:
            return []
        return [CpuShape(self.max_vcpus, 64.0, 0.5, self.name, f"cpu_{self.max_vcpus}")]


def test_tiered_offer_prefers_the_pinned_shape_on_any_provider():
    from cascade.provision.cpu import select_tiered_offer

    cfg = GenerationConfig(vcpus=32, window_seconds=5400, fallback_vcpus=(16,))
    # targon (first) only has 16; shadeform has 32 → the 32 wins
    p, shape, tier = select_tiered_offer([_TierFake("targon", 16), _TierFake("shadeform", 32)], cfg)
    assert (p.name, tier, shape.vcpus) == ("shadeform", 32, 32)
    # no 32 anywhere → the 16 tier
    p, shape, tier = select_tiered_offer([_TierFake("targon", 16), _TierFake("shadeform", 8)], cfg)
    assert (p.name, tier) == ("targon", 16)
    assert select_tiered_offer([_TierFake("targon", 8)], cfg) is None
    # no fallback configured → the single pinned shape, as before
    assert select_tiered_offer([_TierFake("targon", 16)], GenerationConfig(vcpus=32)) is None


def test_fallback_tier_gets_the_same_cpu_budget():
    from cascade.shared.config.generation import parse_generation

    g = parse_generation({"vcpus": 32, "window_seconds": 5400, "fallback_vcpus": [16]})
    assert g.shapes == (32, 16) and g.window_for(16) == 10800 and g.max_window_seconds == 10800
    leg = g.at_shape(16)
    assert (leg.vcpus, leg.window_seconds, leg.n_children) == (16, 10800, 16)
    assert leg.cpu_budget_seconds == g.cpu_budget_seconds == 172800
    assert g.at_shape(32).window_seconds == 5400
    with pytest.raises(ValueError):
        g.at_shape(24)
    for bad in ({"fallback_vcpus": [32]}, {"fallback_vcpus": [8, 16]}, {"fallback_vcpus": [7]},
                {"fallback_vcpus": [16], "children": 4}):
        with pytest.raises(ValueError):
            parse_generation({"vcpus": 32, "window_seconds": 5400, **bad})
    assert parse_generation({}).fallback_vcpus == () and parse_generation({}).shapes == (32,)
