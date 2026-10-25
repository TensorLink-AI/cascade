"""Miner-funded gang rentals on Lium/Vast/RunPod adapters; no marketplace calls."""

from types import SimpleNamespace

import pytest

from cascade.direct.compute import PayerPodProvider, PodProfile, direct_pod_prefix
from cascade.provision.core import PodAddress, ProvisionError
from cascade.provision.direct import (
    Allocation,
    AllocationPending,
    CapacityUnavailable,
    CredentialsRevoked,
    ProviderTransient,
    SpendingRefused,
)
from cascade.provision.funded import payer_pod_pattern, provider_for_key
from cascade.provision.runpod import RunPodProvider
from cascade.provision.vast import VastProvider

PAYER = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
IMAGE = "ghcr.io/tensorlink/cascade-worker:v2@sha256:" + "b" * 64


def profile():
    return PodProfile("H200", 8, "H200", 141, "bfloat16", 30_000_000, 3600, IMAGE, "a" * 64)


class Market:
    """A marketplace double with the Provider seam the direct gang uses."""

    name = "fake"

    def __init__(self):
        self.pods, self.launches, self.terminated = {}, [], []
        self.offers, self.fail, self.stuck = 1, None, False

    def capacity(self, sku, *, gpus=1, exclude_ids=()):
        return self.offers

    def launch(self, spec):
        if self.fail:
            raise ProvisionError(self.fail)
        name = f"{spec.name_prefix}-0"
        self.launches.append(spec)
        self.pods[name] = {"id": f"uid-{len(self.launches)}", "ready": False, "gpus": spec.gpus_per_pod}
        return [name]

    def list_tagged(self, prefix):
        return [n for n in self.pods if n.startswith(prefix)]

    def pod_identity(self, name):
        pod = self.pods.get(name)
        return None if pod is None else {"id": pod["id"], "gpus": pod["gpus"]}

    def live_pod_address(self, name):
        pod = self.pods.get(name)
        return PodAddress("10.0.0.7", 40022) if pod and pod["ready"] else None

    def terminate(self, name):
        self.terminated.append(name)
        if not self.stuck:
            self.pods.pop(name, None)


class Vault:
    def __init__(self, key="payer-key"):
        self.key = key

    def get_shared(self, payer):
        return self.key


def gang(tmp_path, market, *, vault=None, allow_paid=True, clock=lambda: 0.0):
    ledger = SimpleNamespace(state={"jobs": {"job": {"reservation": {"key": "rk"},
                                                     "terms": {"payer": PAYER}}}})
    scans = []

    def scan(ip, port):
        scans.append((ip, port))
        return "ssh-ed25519 AAAApinned"

    provider = PayerPodProvider(ledger, vault or Vault(), "vast", {"h200x8": profile()}, tmp_path,
                                ssh_pubkey="ssh-ed25519 AAAAorchestrator", netuid=91,
                                allow_paid=allow_paid, provider_factory=lambda c, k: market,
                                host_key_scanner=scan, appear_seconds=600, clock=clock)
    return provider, scans


def reserve(provider, now=1.0):
    return provider.reserve("rk", profile="h200x8", world_size=8, cap=10**9, now=now)


def test_paid_rental_requires_explicit_authorization(tmp_path):
    provider, _ = gang(tmp_path, Market(), allow_paid=False)
    with pytest.raises(PermissionError, match="explicit authorization"):
        reserve(provider)


def test_launch_once_then_adopt_by_name_and_pin_identity(tmp_path):
    market = Market()
    provider, scans = gang(tmp_path, market)
    with pytest.raises(AllocationPending):
        reserve(provider)
    with pytest.raises(AllocationPending):  # booting: never a second launch
        reserve(provider, now=2)
    assert len(market.launches) == 1
    spec = market.launches[0]
    assert spec.gpus_per_pod == 8 and spec.image == IMAGE and spec.sku == "H200"
    assert spec.name_prefix == direct_pod_prefix("rk", PAYER, 91)
    market.pods[f"{spec.name_prefix}-0"]["ready"] = True
    held = reserve(provider, now=3)
    assert held == Allocation(f"{spec.name_prefix}-0", tuple(range(8)), 1.0 + 3600, "h200x8",
                              profile().maximum_estimate)
    assert provider.lookup("rk") == held and len(scans) == 1  # host key pinned once
    assert provider.address("rk") == (PodAddress("10.0.0.7", 40022), "ssh-ed25519 AAAApinned")
    # A different pod under our name is the payer's doing, never adopted.
    market.pods[f"{spec.name_prefix}-0"]["id"] = "uid-swapped"
    with pytest.raises(CredentialsRevoked, match="replaced"):
        provider.lookup("rk")


def test_sold_out_is_never_submitted_and_refundable(tmp_path):
    market = Market()
    market.offers = 0
    provider, _ = gang(tmp_path, market)
    with pytest.raises(CapacityUnavailable):
        reserve(provider)
    assert market.launches == [] and provider.no_charge_evidence("rk") == "never_submitted"
    provider.release("rk")


@pytest.mark.parametrize("message, error, evidence", [
    ("vast PUT asks/1/ failed (401): unauthorized", CredentialsRevoked, "provider_rejected"),
    ("vast PUT asks/1/ failed (400): no_such_ask", CapacityUnavailable, "provider_rejected"),
    ("runpod POST pods failed (402): insufficient balance", SpendingRefused, "provider_rejected"),
    ("vast PUT asks/1/ failed (502): bad gateway", ProviderTransient, None),
])
def test_launch_refusals_are_attributed(tmp_path, message, error, evidence):
    market = Market()
    market.fail = message
    provider, _ = gang(tmp_path, market)
    with pytest.raises(error):
        reserve(provider)
    assert provider.no_charge_evidence("rk") == evidence


def test_ambiguous_launch_is_reconciled_by_name_not_relaunched(tmp_path):
    market = Market()
    market.fail = "vast PUT asks/1/ failed (504): gateway timeout"
    provider, _ = gang(tmp_path, market)
    with pytest.raises(ProviderTransient):
        reserve(provider)
    market.fail = None
    with pytest.raises(AllocationPending):
        reserve(provider, now=100)
    assert market.launches == []  # the original submission still owns this key
    with pytest.raises(CapacityUnavailable, match="never appeared"):
        reserve(provider, now=1000)
    assert provider.no_charge_evidence("rk") is None  # not provably free: debit stays


def test_missing_payer_key_is_attributable(tmp_path):
    provider, _ = gang(tmp_path, Market(), vault=Vault(key=None))
    with pytest.raises(CredentialsRevoked, match="no vast key"):
        reserve(provider)


def test_release_confirms_termination_by_listing(tmp_path):
    market = Market()
    provider, _ = gang(tmp_path, market)
    with pytest.raises(AllocationPending):
        reserve(provider)
    name = market.launches[0].name_prefix + "-0"
    market.stuck = True
    with pytest.raises(AllocationPending, match="still listed"):
        provider.release("rk")
    market.stuck = False
    provider.release("rk")
    assert name not in market.pods and market.terminated == [name, name]
    with pytest.raises(CapacityUnavailable, match="cancelled"):
        reserve(provider, now=5)


def test_release_without_payer_key_stays_pending(tmp_path):
    market = Market()
    vault = Vault()
    provider, _ = gang(tmp_path, market, vault=vault)
    with pytest.raises(AllocationPending):
        reserve(provider)
    vault.key = None
    with pytest.raises(AllocationPending, match="payer key gone"):
        provider.release("rk")


def test_direct_pods_are_invisible_to_the_rolling_reaper():
    name = direct_pod_prefix("rk", PAYER, 91) + "-0"
    assert not payer_pod_pattern(PAYER, 91).match(name)
    assert direct_pod_prefix("rk", PAYER, 91) != direct_pod_prefix("other", PAYER, 91)


def test_profile_requires_digest_pinned_image():
    with pytest.raises(ProvisionError, match="digest-pinned"):
        PodProfile("H200", 8, "H200", 141, "bfloat16", 1, 1, "repo:tag", "a" * 64)


# ── marketplace adapters ────────────────────────────────────────────────────


class Http:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def __call__(self, method, url, *, headers, json=None, timeout=30.0):
        self.calls.append((method, url, headers, json))
        for (m, suffix), reply in self.responses.items():
            if m == method and url.endswith(suffix):
                return reply() if callable(reply) else reply
        return 404, {"error": "not found"}


def test_vast_launch_keeps_key_in_header_and_uses_image_entrypoint():
    http = Http({("POST", "bundles/"): (200, {"offers": [
                    {"id": 11, "machine_id": 5, "num_gpus": 8, "dph_total": 30.0},
                    {"id": 12, "machine_id": 6, "num_gpus": 4, "dph_total": 9.0}]}),
                 ("PUT", "asks/11/"): (200, {"success": True, "new_contract": 99})})
    vast = VastProvider(api_key="sekret", _http=http)
    from cascade.provision.core import LaunchSpec

    names = vast.launch(LaunchSpec("H200", 1, IMAGE, "ssh-ed25519 AAAAop", name_prefix="pfx", gpus_per_pod=8))
    assert names == ["pfx-0"]
    method, url, headers, body = http.calls[-1]
    assert headers["Authorization"] == "Bearer sekret" and "sekret" not in str(body)
    assert body["label"] == "pfx-0" and body["runtype"] == "args" and body["image"] == IMAGE
    assert body["env"]["SSH_PUBKEY"] == "ssh-ed25519 AAAAop" and body["env"]["-p 22:22"] == "1"
    query = http.calls[0][3]
    assert query["num_gpus"] == {"eq": 8} and query["gpu_name"] == {"eq": "H200"}


def test_vast_listing_readiness_and_twin_teardown():
    rows = [{"id": 1, "label": "pfx-0", "actual_status": "running", "public_ipaddr": "1.2.3.4",
             "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41022"}]}, "num_gpus": 8},
            {"id": 2, "label": "pfx-0", "actual_status": "loading", "num_gpus": 8},
            {"id": 3, "label": "someone-else", "actual_status": "running"}]
    pages = {None: (200, {"success": True, "instances": rows[:2], "next_token": "t2",
                         "instances_found": 2, "total_instances": 3}),
             "t2": (200, {"success": True, "instances": rows[2:], "next_token": None,
                         "instances_found": 1, "total_instances": 3})}
    calls = []

    def http(method, url, *, headers, json=None, timeout=30.0):
        calls.append((method, url, headers, json))
        if method == "GET":
            assert "/api/v1/instances/?" in url        # the v0 list is deprecated
            return pages["t2" if "after_token=t2" in url else None]
        return 200, {}

    vast = VastProvider(api_key="sekret", _http=http)
    assert vast.list_tagged("pfx") == ["pfx-0", "pfx-0"]
    assert vast.live_pod_address("pfx-0") == PodAddress("1.2.3.4", 41022)
    assert vast.pod_identity("pfx-0")["id"] == "1"
    vast.terminate("pfx-0")
    assert [c[1].rsplit("/", 2)[-2] for c in calls if c[0] == "DELETE"] == ["1", "2"]


def _vast_offers(*offers):
    return Http({("POST", "bundles/"): (200, {"offers": list(offers)})})


def test_vast_queries_the_spelling_its_api_uses():
    http = _vast_offers()
    VastProvider(api_key="k", _http=http).capacity("RTX4090")
    query = http.calls[0][3]
    assert query["gpu_name"] == {"eq": "RTX 4090"} and query["external"] == {"eq": False}
    VastProvider(api_key="k", _http=http).capacity("H100")
    assert http.calls[1][3]["gpu_name"] == {"in": ["H100 SXM", "H100 PCIE", "H100 NVL"]}


def test_vast_enforces_hourly_and_per_leg_caps_and_drops_unpriced_offers():
    offers = [{"id": 1, "machine_id": 1, "num_gpus": 1, "dph_total": float("nan")},
              {"id": 2, "machine_id": 2, "num_gpus": 1},
              {"id": 3, "machine_id": 3, "num_gpus": 1, "dph_total": -1},
              {"id": 4, "machine_id": 4, "num_gpus": 1, "dph_total": 0.90},
              {"id": 5, "machine_id": 5, "num_gpus": 1, "dph_total": 0.40}]
    vast = VastProvider(api_key="k", _http=_vast_offers(*offers))
    assert vast.capacity("RTX4090") == 2
    vast.max_price_per_hour = 1.5
    vast.max_leg_cost_usd, vast.sku_wall_seconds = 1.6, (("RTX4090", 13500),)
    assert [o["id"] for o in vast._offers("RTX4090")] == [5]   # 1.6 $ over 3.75 h
    vast.sku_wall_seconds = ()
    with pytest.raises(ProvisionError, match="positive training wall"):
        vast.capacity("RTX4090")
    vast = VastProvider(api_key="k", _http=_vast_offers(*offers))
    assert vast.capacity("RTX4090", exclude_ids=("vast:5",)) == 1   # a claimed machine


def test_vast_definite_refusal_is_not_an_uncertain_create():
    from cascade.funding.faults import classify_rent_failure
    from cascade.provision.core import LaunchSpec

    offer = {"id": 7, "machine_id": 70, "num_gpus": 1, "dph_total": 0.3}
    spec = LaunchSpec("RTX4090", 1, IMAGE, "ssh-ed25519 AAAAop", name_prefix="pfx")
    for reply, cls in (((400, {"error": "no_such_ask"}), "no_capacity"),
                       ((200, {"success": False, "msg": "unavailable"}), "no_capacity"),
                       ((401, {"msg": "invalid api key"}), "auth"),
                       ((502, {"raw": "bad gateway"}), "infra")):
        http = Http({("POST", "bundles/"): (200, {"offers": [offer]}), ("PUT", "asks/7/"): reply})
        with pytest.raises(ProvisionError) as e:
            VastProvider(api_key="k", _http=http).launch(spec)
        assert classify_rent_failure(str(e.value)) == cls, (reply, str(e.value))
    http = Http({("POST", "bundles/"): (200, {"offers": [offer]}),
                 ("PUT", "asks/7/"): (200, {"success": True, "new_contract": 99})})
    vast = VastProvider(api_key="k", _http=http)
    assert vast.launch(spec) == ["pfx-0"] and http.calls[-1][3]["client_id"] == "me"
    assert vast.machine_of("pfx-0") == "vast:70" and vast.sku_of("pfx-0") == "RTX4090"


def test_vast_recovers_the_gpu_type_after_a_restart():
    rows = [{"id": 1, "label": "pfx-0", "gpu_name": "RTX 4090", "machine_id": 9}]

    def http(method, url, *, headers, json=None, timeout=30.0):
        return 200, {"success": True, "instances": rows, "next_token": None,
                     "instances_found": len(rows), "total_instances": len(rows)}

    vast = VastProvider(api_key="k", _http=http)
    assert vast.sku_of("pfx-0") == "RTX4090" and vast.machine_of("pfx-0") == "vast:9"


def test_vast_repeated_cursor_never_reads_as_gone():
    def http(method, url, *, headers, json=None, timeout=30.0):
        return 200, {"success": True, "instances": [], "next_token": "again",
                     "instances_found": 0, "total_instances": 1}

    with pytest.raises(ProvisionError, match="repeated instance listing cursor"):
        VastProvider(api_key="k", _http=http).list_tagged("pfx")


def test_direct_ambiguous_create_is_not_refused_by_error_prose(tmp_path, monkeypatch):
    from cascade.provision.core import UncertainCreateError

    market = Market()
    provider, _ = gang(tmp_path, market)

    def uncertain(spec):
        raise UncertainCreateError("HTTP 502: no_capacity; invalid api key; balance")

    monkeypatch.setattr(market, "launch", uncertain)
    with pytest.raises(ProviderTransient):
        reserve(provider)
    assert provider.no_charge_evidence("rk") is None
    with pytest.raises(AllocationPending):
        reserve(provider, now=2.0)
    assert not market.launches


def test_vast_errors_never_echo_the_key():
    http = Http({("POST", "bundles/"): (401, {"msg": "bad key sekret"})})
    with pytest.raises(ProvisionError) as e:
        VastProvider(api_key="sekret", _http=http).capacity("H200", gpus=8)
    assert "sekret" not in str(e.value) and "401" in str(e.value)


def test_runpod_create_maps_gpu_and_reads_port_mapping():
    pods = [{"id": "p1", "name": "pfx-0", "desiredStatus": "RUNNING", "publicIp": "5.6.7.8",
             "portMappings": {"22": 10341}, "gpu": {"count": 8, "displayName": "H200"}},
            {"id": "p0", "name": "pfx-0", "desiredStatus": "TERMINATED"}]
    http = Http({("POST", "pods"): (201, {"id": "p1"}), ("GET", "pods"): (200, pods)})
    runpod = RunPodProvider(api_key="sekret", _http=http)
    from cascade.provision.core import LaunchSpec

    runpod.launch(LaunchSpec("H200", 1, IMAGE, "ssh-ed25519 AAAAop", name_prefix="pfx", gpus_per_pod=8))
    body = http.calls[0][3]
    assert body["gpuTypeIds"] == ["NVIDIA H200"] and body["gpuCount"] == 8
    assert body["ports"] == ["22/tcp"] and body["interruptible"] is False
    assert body["env"]["SSH_PUBKEY"] == "ssh-ed25519 AAAAop"
    assert runpod.list_tagged("pfx") == ["pfx-0"]  # terminated rows are not live
    assert runpod.live_pod_address("pfx-0") == PodAddress("5.6.7.8", 10341)
    assert runpod.pod_identity("pfx-0")["gpus"] == 8


def test_provider_factory_dispatches_and_refuses_unknown():
    assert provider_for_key("vast", "k").name == "vast"
    assert provider_for_key("runpod", "k").name == "runpod"
    assert provider_for_key("lium", "k").name == "lium"
    with pytest.raises(ProvisionError, match="unknown compute provider"):
        provider_for_key("aws", "k")
    with pytest.raises(ProvisionError, match="empty"):
        provider_for_key("vast", "")
