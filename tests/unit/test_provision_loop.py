"""Service loop — the full rent→publish→watch→teardown cycle on fakes.

Every boundary (chain, plan, providers, manifest store, health, clock) is
injected, so these tests drive whole rounds without a cloud account, a GPU,
or a real trainer. The hosts.toml the loop publishes is always re-parsed with
the trainer's own loader — the contract, not the string, is asserted."""

from __future__ import annotations

import json
import tomllib

import pytest

from cascade.provision.core import PodAddress, ProvisionError
from cascade.provision.health import CheckResult, HealthReport
from cascade.provision.loop import ProvisionerLoop, RenderSettings, parse_plan_output
from cascade.provision.main import build_policy
from cascade.provision.policy import ProvisionPolicy, StagePolicy
from cascade.provision.state import load_state
from cascade.trainer.remote import RemoteDispatchError, load_hosts

IMG = "reg.example/cascade-worker@sha256:" + "a" * 64


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeProvider:
    """In-memory marketplace: launches get sequential IPs, terminate frees them."""

    def __init__(self, name, *, available=True, avail_raises=None, price=None):
        self.name = name
        self._available = available
        self._avail_raises = avail_raises
        self._price = price
        self._next_ip = 0
        self.live: dict[str, PodAddress] = {}
        self.launched: list[str] = []
        self.terminated: list[str] = []

    def available(self, sku, count, *, gpus=1):
        if self._avail_raises:
            raise self._avail_raises
        return self._available

    def launch(self, spec):
        ids = []
        for i in range(spec.count):
            self._next_ip += 1
            pid = f"{spec.name_prefix}-{i}"
            self.live[pid] = PodAddress(f"10.0.0.{self._next_ip}", 22)
            ids.append(pid)
        self.launched += ids
        return ids

    def wait_ready(self, pod_id, *, timeout):
        return pod_id in self.live

    def get_ip(self, pod_id):
        return self.live.get(pod_id)

    def terminate(self, pod_id):
        self.terminated.append(pod_id)
        self.live.pop(pod_id, None)

    def list_tagged(self, prefix):
        return [pid for pid in self.live if pid.startswith(prefix)]

    def offer_price(self, sku):
        return self._price


class FakeChain:
    def __init__(self, block):
        self.block = block

    def current_block(self):
        return self.block


class FakeStore:
    def __init__(self, texts=None):
        self.texts = dict(texts or {})

    def get_text(self, key):
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _report(ok=True, name="fake"):
    return HealthReport(checks=(CheckResult(name=name, ok=ok),))


def _policy(**kw):
    kw.setdefault("heat", StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=8, max_pods=2,
                                      providers=("lium", "shadeform"), max_price_hr=4.0))
    kw.setdefault("final", StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=2,
                                       providers=("lium", "shadeform"), max_price_hr=3.0))
    kw.setdefault("trigger_margin_blocks", 25)
    kw.setdefault("max_spend_per_round", 25.0)
    return ProvisionPolicy(**kw)


PLAN = {
    "block": 880, "epoch_blocks": 900, "next_boundary_block": 900,
    "blocks_to_boundary": 20, "king": "5King", "resolved": 14,
    "challengers": 13, "eligible_challengers": 12,
    "heat_train_hours": 0.5, "finalists": 1,
}


def make_loop(tmp_path, *, providers=None, block=880, policy=None, plan=None,
              clock=None, store=None, health=None, dry_run=False, plan_calls=None,
              eval_hosts=None, receipt_prefix=""):
    providers = providers if providers is not None else {"lium": FakeProvider("lium")}
    plan_calls = plan_calls if plan_calls is not None else []

    def plan_fn():
        plan_calls.append(1)
        return dict(plan or PLAN)

    return ProvisionerLoop(
        policy=policy or _policy(),
        providers=providers,
        chain_client=FakeChain(block),
        plan_fn=plan_fn,
        render=RenderSettings(image=IMG, ssh_pubkey="ssh-ed25519 AAAA orch",
                              key_path="~/.ssh/cascade_ed25519"),
        hosts_path=tmp_path / "hosts.toml",
        work_root=tmp_path / "work",
        state_path=tmp_path / "state.json",
        epoch_blocks=900,                    # 3h epochs → ttl 3h at ttl_epochs=1
        final_hours=0.25,
        manifest_store=store,
        eval_hosts_path=eval_hosts,
        receipt_prefix=receipt_prefix,
        health_check=health,
        dry_run=dry_run,
        clock=clock or Clock(),
    ), plan_calls


# ── happy path ───────────────────────────────────────────────────────────────


def test_happy_path_rents_publishes_and_records(tmp_path):
    prov = FakeProvider("lium")
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov})
    loop.run_once()

    # 12 eligible → 3 heat slots → one 8-GPU pod; king+1 finalist → one 2-GPU pod.
    assert prov.launched == ["cascade-900-heat-0", "cascade-900-final-0"]
    assert prov.terminated == []

    hosts = load_hosts(tmp_path / "hosts.toml")             # the trainer's own loader
    heat = [h for h in hosts if h.stage == "heat"]
    final = [h for h in hosts if h.stage == "final"]
    assert len(heat) == 8 and len(final) == 2               # per-GPU fan-out
    assert {h.cuda_device for h in heat} == {str(i) for i in range(8)}
    assert all(h.name.startswith("cascade-900-heat") for h in heat)

    st = load_state(tmp_path / "state.json")
    assert st.round_id == "900" and st.published
    assert {(i.stage, i.instance_id) for i in st.instances} == {
        ("heat", "cascade-900-heat-0"), ("final", "cascade-900-final-0")}

    # Rent-once latch: staying inside the margin must not rent again.
    loop.run_once()
    assert plan_calls == [1] and prov.launched == ["cascade-900-heat-0", "cascade-900-final-0"]


def test_no_trigger_outside_margin(tmp_path):
    prov = FakeProvider("lium")
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov}, block=800)
    loop.run_once()
    assert plan_calls == [] and prov.launched == []


def test_dry_run_rents_nothing(tmp_path):
    prov = FakeProvider("lium")
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov}, dry_run=True)
    loop.run_once()
    assert plan_calls == [1]
    assert prov.launched == []
    assert not (tmp_path / "hosts.toml").exists()
    assert load_state(tmp_path / "state.json") is None
    loop.run_once()                                          # latch also applies to dry runs
    assert plan_calls == [1]


# ── provider failure modes ───────────────────────────────────────────────────


def test_provider_down_falls_through_to_next(tmp_path):
    lium = FakeProvider("lium", avail_raises=RuntimeError("api down"))
    shade = FakeProvider("shadeform")
    loop, _ = make_loop(tmp_path, providers={"lium": lium, "shadeform": shade})
    loop.run_once()
    assert lium.launched == []
    assert shade.launched == ["cascade-900-heat-0", "cascade-900-final-0"]


def test_one_stage_without_capacity_still_rents_the_other(tmp_path):
    # lium has capacity but only heat's price is acceptable on it… simpler:
    # heat's only provider is down, final's works → final-only fleet.
    policy = _policy(heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=8, max_pods=2,
                                      providers=("lium",), max_price_hr=4.0),
                     final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=2,
                                       providers=("shadeform",), max_price_hr=3.0))
    lium = FakeProvider("lium", available=False)
    shade = FakeProvider("shadeform")
    loop, _ = make_loop(tmp_path, providers={"lium": lium, "shadeform": shade}, policy=policy)
    loop.run_once()
    assert lium.launched == []
    assert shade.launched == ["cascade-900-final-0"]
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert all(h.stage == "final" for h in hosts)            # degraded, not dead


def test_all_providers_down_clears_hosts_round_never_lost(tmp_path):
    prov = FakeProvider("lium", available=False)
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov})
    loop.run_once()
    assert prov.launched == []
    # Empty hosts file = the trainer's explicit local-fallback signal.
    assert (tmp_path / "hosts.toml").is_file()
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "hosts.toml")
    loop.run_once()                                          # latched: no 30s hammering
    assert plan_calls == [1]


def test_overpriced_offer_skips_that_provider(tmp_path):
    dear = FakeProvider("lium", price=99.0)                  # above both stage caps
    fair = FakeProvider("shadeform", price=2.0)
    loop, _ = make_loop(tmp_path, providers={"lium": dear, "shadeform": fair})
    loop.run_once()
    assert dear.launched == [] and len(fair.launched) == 2


def test_budget_breaker_refuses_to_rent(tmp_path):
    # Prices are under the per-stage caps, but worst-case (ttl = 3h epoch)
    # projection 2 pods × $2.0 × 3h = $12 > the $10 round cap → refuse ALL.
    prov = FakeProvider("lium", price=2.0)
    loop, _ = make_loop(tmp_path, providers={"lium": prov},
                        policy=_policy(max_spend_per_round=10.0))
    loop.run_once()
    assert prov.launched == []
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "hosts.toml")                  # cleared → local fallback


# ── health gate: terminate + one replacement ─────────────────────────────────


def test_unhealthy_pod_replaced_once(tmp_path):
    prov = FakeProvider("lium")
    bad_ips = {"10.0.0.1"}                                   # the first heat pod's IP

    def health(addr, stage, provider="", **shape):
        return _report(ok=addr.ip not in bad_ips)

    loop, _ = make_loop(tmp_path, providers={"lium": prov}, health=health)
    loop.run_once()
    assert prov.terminated == ["cascade-900-heat-0"]
    assert "cascade-900-heat-r0-0" in prov.launched          # the one replacement
    st = load_state(tmp_path / "state.json")
    assert {i.instance_id for i in st.instances} == {
        "cascade-900-heat-r0-0", "cascade-900-final-0"}
    heat = [h for h in load_hosts(tmp_path / "hosts.toml") if h.stage == "heat"]
    assert all(h.host == "10.0.0.2" for h in heat)           # the replacement's IP


def test_replacement_also_unhealthy_drops_the_slot(tmp_path):
    prov = FakeProvider("lium")

    def health(addr, stage, provider="", **shape):
        return _report(ok=(stage != "heat"))                 # every heat pod is a lemon

    loop, _ = make_loop(tmp_path, providers={"lium": prov}, health=health)
    loop.run_once()
    # Original + its single replacement both terminated; no third attempt.
    assert prov.terminated == ["cascade-900-heat-0", "cascade-900-heat-r0-0"]
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert all(h.stage == "final" for h in hosts)            # heat degraded away
    st = load_state(tmp_path / "state.json")
    assert {i.instance_id for i in st.instances} == {"cascade-900-final-0"}


def test_every_pod_unhealthy_clears_hosts(tmp_path):
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov},
                        health=lambda addr, stage, provider="", **shape: _report(ok=False))
    loop.run_once()
    assert prov.live == {}                                    # nothing left billing
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "hosts.toml")


# ── watch + per-stage teardown ───────────────────────────────────────────────


def _provisioned(tmp_path, **kw):
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, **kw)
    loop.run_once()
    assert len(prov.live) == 2
    return loop, prov


def test_heat_marker_tears_down_heat_while_final_runs(tmp_path):
    clock = Clock()
    loop, prov = _provisioned(tmp_path, clock=clock)
    # The trainer settles the heat: work_root/<base_seed>/heat_complete.json.
    # The provisioner keys rounds by boundary block and cannot know base_seed
    # in advance — ANY marker newer than rent time is this round's signal.
    marker_dir = tmp_path / "work" / "54321"
    marker_dir.mkdir(parents=True)
    (marker_dir / "heat_complete.json").write_text(
        json.dumps({"round_id": "54321", "screened": 12, "finalists": ["hk"]}))
    clock.t += 3600.0                                        # 1h in: TTL (3h) not due

    loop.run_once()
    assert prov.terminated == ["cascade-900-heat-0"]
    assert "cascade-900-final-0" in prov.live                # final keeps training
    hosts = load_hosts(tmp_path / "hosts.toml")              # re-rendered final-only
    assert [h.stage for h in hosts] == ["final", "final"]
    st = load_state(tmp_path / "state.json")
    assert {i.stage for i in st.instances} == {"final"}


def test_manifest_tears_down_everything(tmp_path):
    clock = Clock()
    store = FakeStore()
    loop, prov = _provisioned(tmp_path, clock=clock, store=store)
    # Marker lands (teaches the provisioner base_seed 54321)…
    d = tmp_path / "work" / "54321"
    d.mkdir(parents=True)
    (d / "heat_complete.json").write_text("{}")
    clock.t += 1800.0
    loop.run_once()
    assert "cascade-900-final-0" in prov.live
    # …then the round manifest publishes at the learned round id.
    store.texts["manifests/round-54321.json"] = '{"round_id": "54321"}'
    clock.t += 1800.0
    loop.run_once()
    assert prov.live == {}
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "hosts.toml")                  # cleared
    assert load_state(tmp_path / "state.json").instances == ()


def test_latest_pointer_change_also_ends_the_round(tmp_path):
    # No marker ever seen (e.g. trainer crashed mid-write) — the latest.json
    # baseline still detects "a manifest published after we rented".
    clock = Clock()
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _provisioned(tmp_path, clock=clock, store=store)
    clock.t += 1800.0
    loop.run_once()
    assert len(prov.live) == 2                               # unchanged pointer: no teardown
    store.texts["manifests/latest.json"] = '{"round_id": "222"}'
    loop.run_once()
    assert prov.live == {}


def test_ttl_backstop_fires_without_any_signal(tmp_path):
    clock = Clock()
    loop, prov = _provisioned(tmp_path, clock=clock)
    clock.t += 3 * 3600.0 - 1                                # one second shy of 1 epoch
    loop.run_once()
    assert len(prov.live) == 2
    clock.t += 1.0
    loop.run_once()
    assert prov.live == {} and len(prov.terminated) == 2


# ── restart + reconcile ──────────────────────────────────────────────────────


def test_restart_resumes_ledger_and_kills_orphans(tmp_path):
    loop1, prov = _provisioned(tmp_path)
    # A previous crash left a tagged pod the ledger never recorded.
    prov.live["cascade-900-heat-zombie"] = PodAddress("10.0.0.99", 22)

    loop2, plan_calls = make_loop(tmp_path, providers={"lium": prov}, block=885)
    loop2.run_once()
    assert "cascade-900-heat-zombie" in prov.terminated       # orphan reconciled away
    assert "cascade-900-heat-0" in prov.live                  # owned pods untouched
    # The resumed ledger's round_id restores the rent-once latch too.
    assert plan_calls == []


def test_reconcile_never_touches_untagged_pods(tmp_path):
    prov = FakeProvider("lium")
    prov.live["someone-elses-pod"] = PodAddress("10.9.9.9", 22)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=100)
    loop.run_once()
    assert prov.terminated == []


def test_reconcile_never_touches_hand_rented_cascade_pods(tmp_path):
    """The 2026-07-13 incident: operators' pods legitimately share the
    ``cascade-`` prefix (cascade-worker, cascade-final-b) but are NOT the
    provisioner's — only the full cascade-<round>-<stage> scheme is reapable."""
    prov = FakeProvider("lium")
    prov.live["cascade-worker"] = PodAddress("10.9.9.1", 22)
    prov.live["cascade-final-b"] = PodAddress("10.9.9.2", 22)
    prov.live["cascade-heat-2"] = PodAddress("10.9.9.3", 22)   # no round id ⇒ not ours
    prov.live["cascade-900-heat-zombie"] = PodAddress("10.9.9.4", 22)  # ours, orphaned
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=100)
    loop.run_once()
    assert prov.terminated == ["cascade-900-heat-zombie"]


def test_dry_run_never_terminates_anything(tmp_path):
    """--dry-run must gate EVERY provider mutation, not just rentals: the
    reaper (and teardown) once terminated a live pod during a dry-run demo."""
    prov = FakeProvider("lium")
    prov.live["cascade-900-heat-zombie"] = PodAddress("10.9.9.4", 22)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=100)
    loop.dry_run = True
    loop.run_once()
    assert prov.terminated == []


def test_plan_failure_retries_next_tick(tmp_path):
    prov = FakeProvider("lium")
    calls = []

    def flaky_plan():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("subtensor hiccup")
        return dict(PLAN)

    loop, _ = make_loop(tmp_path, providers={"lium": prov})
    loop.plan_fn = flaky_plan
    loop.run_once()
    assert prov.launched == []                                # no latch on plan failure…
    loop.run_once()
    assert len(calls) == 2 and len(prov.launched) == 2        # …so the next tick rents


# ── plan-output parsing + config validation (main.py pure parts) ─────────────


def test_parse_plan_output_takes_last_json_line():
    text = ("btlogging noise\nWARNING something\n"
            '{"old": true}\n' + json.dumps(PLAN) + "\ntrailing banner\n")
    assert parse_plan_output(text)["eligible_challengers"] == 12


def test_parse_plan_output_rejects_json_free_text():
    with pytest.raises(ProvisionError):
        parse_plan_output("no json here\n")


def _raw_config(**over):
    raw = {
        "provisioner": {
            "trigger_margin_blocks": 25,
            "max_spend_per_round": 25.0,
            "heat": {"sku": "NVIDIA RTX A6000", "gpus_per_pod": 8, "max_pods": 4,
                     "providers": ["lium", "shadeform"], "max_price_hr": 4.0},
            "final": {"sku": "NVIDIA L40S", "gpus_per_pod": 2, "max_pods": 2,
                      "providers": ["lium", "shadeform"], "max_price_hr": 3.0},
        }
    }
    raw["provisioner"].update(over)
    return raw


def test_build_policy_happy():
    p = build_policy(_raw_config(), epoch_blocks=900)
    assert p.heat.gpus_per_pod == 8 and p.final.sku == "NVIDIA L40S"
    assert p.heat.slot_overhead == pytest.approx(1.3)         # default
    assert p.ttl_epochs == 1


@pytest.mark.parametrize("bad", [
    {"trigger_margin_blocks": 900},                           # margin must be < epoch
    {"trigger_margin_blocks": 0},
    {"max_spend_per_round": 0},
    {"heat": {"sku": "", "gpus_per_pod": 8, "max_pods": 4,
              "providers": ["lium"], "max_price_hr": 4.0}},   # empty sku
    {"heat": {"sku": "A6000", "gpus_per_pod": 0, "max_pods": 4,
              "providers": ["lium"], "max_price_hr": 4.0}},   # gpus_per_pod >= 1
    {"final": {"sku": "L40S", "gpus_per_pod": 2, "max_pods": 2,
               "providers": ["lium"], "max_price_hr": 0}},    # price > 0
])
def test_build_policy_rejects_bad_config(bad):
    with pytest.raises(ProvisionError):
        build_policy(_raw_config(**bad), epoch_blocks=900)


def test_build_policy_requires_both_stage_tables():
    raw = _raw_config()
    del raw["provisioner"]["heat"]
    with pytest.raises(ProvisionError):
        build_policy(raw, epoch_blocks=900)


def test_hosts_file_round_trips_via_tomllib_too(tmp_path):
    # Belt and braces: the published file is plain valid TOML, not just
    # something load_hosts tolerates.
    loop, _ = make_loop(tmp_path, providers={"lium": FakeProvider("lium")})
    loop.run_once()
    data = tomllib.loads((tmp_path / "hosts.toml").read_text(encoding="utf-8"))
    assert len(data["host"]) == 10                            # 8 heat GPUs + 2 final GPUs


# ── static hosts + bootstrap + unmanaged final ───────────────────────────────


def test_static_hosts_survive_every_publish_and_clear(tmp_path):
    """The operator's long-lived final pod must ride along on every render —
    including failure paths that previously cleared the file outright."""
    static = '[[host]]\nname = "cascade-final-b"\nhost = "216.81.245.151"\nstage = "final"\n'
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=880)
    loop.static_hosts_text = static
    loop.run_once()                               # provisions + publishes
    text = (tmp_path / "hosts.toml").read_text()
    assert "cascade-final-b" in text              # static entry present
    assert "cascade-900-heat" in text             # dynamic heat pods present
    # all-providers-down path: static fleet remains, never an empty file
    prov2 = FakeProvider("lium", available=False)
    loop2, _ = make_loop(tmp_path, providers={"lium": prov2}, block=1780)
    loop2.static_hosts_text = static
    loop2.run_once()
    text2 = (tmp_path / "hosts.toml").read_text()
    assert "cascade-final-b" in text2
    assert "heat" not in text2.replace('stage = "final"', "")


def test_bootstrap_failure_replaces_pod_once(tmp_path):
    calls = []

    def flaky_bootstrap(addr, stage, provider=""):
        calls.append(addr.ip)
        return len(calls) > 1                     # first pod fails, replacement passes

    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=880)
    loop.bootstrap = flaky_bootstrap
    loop.run_once()
    assert len(calls) >= 2                        # failed pod → one replacement attempt
    assert prov.terminated                        # the dud was terminated


def test_unmanaged_final_rents_no_final_pods(tmp_path):
    from cascade.provision.policy import size_fleet

    pol = _policy(final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=1, max_pods=0,
                                    providers=("lium",), max_price_hr=3.0))
    plan = size_fleet(12, 1, 0.5, 3.0, 0.75, pol)
    assert plan.final.pods == 0                   # stage unmanaged: static pods serve it
    assert plan.heat.pods > 0


def test_publish_uses_per_provider_profile(tmp_path):
    """Shadeform VMs land as the 'shadeform' user under /home/shadeform — the
    rendered hosts entries must carry that provider's paths, not lium's root."""
    from dataclasses import replace as _replace

    from cascade.provision.loop import PodProfile

    prov = FakeProvider("shadeform")
    loop, _ = make_loop(tmp_path, providers={"shadeform": prov})
    loop.render = _replace(loop.render, profiles={"shadeform": PodProfile(
        user="shadeform", workdir="/home/shadeform/cascade",
        remote_python="/home/shadeform/cascade/.venv/bin/python")})
    loop.run_once()
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert hosts and all(h.user == "shadeform" for h in hosts)
    assert all(h.workdir == "/home/shadeform/cascade" for h in hosts)


# ── SKU fallback (homogeneous per round) ─────────────────────────────────────


class ShapedProvider(FakeProvider):
    """A marketplace that only stocks specific (sku, gpus) shapes."""

    def __init__(self, name, shapes, price=None):
        super().__init__(name, price=price)
        self.shapes = set(shapes)                       # {(market_sku, gpus)}

    def available(self, sku, count, *, gpus=1):
        return (sku, gpus) in self.shapes


def _fallback_policy():
    from cascade.provision.policy import SkuCandidate
    return _policy(heat=StagePolicy(
        sku="NVIDIA GeForce RTX 4090", market_sku="RTX4090", gpus_per_pod=4,
        max_pods=2, providers=("lium", "shadeform"), max_price_hr=2.60,
        candidates=(SkuCandidate(sku="NVIDIA RTX A6000", market_sku="A6000",
                                 gpus_per_pod=8, max_price_hr=4.50),)))


def test_sku_fallback_takes_first_candidate_with_capacity(tmp_path):
    """Primary 4x4090 out of stock everywhere; the 8xA6000 fallback (on the
    second provider, at ITS shape: 1 pod for 7 slots) serves the whole heat."""
    lium = ShapedProvider("lium", shapes=set())                      # nothing
    shade = ShapedProvider("shadeform", shapes={("A6000", 8)})
    loop, _ = make_loop(tmp_path, providers={"lium": lium, "shadeform": shade},
                        policy=_fallback_policy())
    loop.run_once()
    assert shade.launched == ["cascade-900-heat-0"]                  # 1 × 8x pod
    hosts = [h for h in load_hosts(tmp_path / "hosts.toml") if h.stage == "heat"]
    assert len(hosts) == 8                                           # fallback fan-out
    st = load_state(tmp_path / "state.json")
    heat = [i for i in st.instances if i.stage == "heat"]
    assert heat[0].sku == "NVIDIA RTX A6000" and heat[0].gpus == 8   # rented shape recorded


def test_sku_fallback_health_gate_gets_rented_sku(tmp_path):
    """The gate must assert the device that was ACTUALLY rented, not the primary."""
    seen = []

    def health(addr, stage, provider="", *, sku="", gpus=0):
        seen.append((stage, sku, gpus))
        return _report(ok=True)

    shade = ShapedProvider("shadeform", shapes={("A6000", 8)})
    loop, _ = make_loop(tmp_path, providers={"shadeform": shade},
                        policy=_fallback_policy(), health=health)
    loop.run_once()
    assert ("heat", "NVIDIA RTX A6000", 8) in seen


def test_sku_primary_wins_when_stocked(tmp_path):
    lium = ShapedProvider("lium", shapes={("RTX4090", 4)})
    shade = ShapedProvider("shadeform", shapes={("A6000", 8)})
    loop, _ = make_loop(tmp_path, providers={"lium": lium, "shadeform": shade},
                        policy=_fallback_policy())
    loop.run_once()
    assert lium.launched == ["cascade-900-heat-0"]   # 3 slots (12-field) @ 4x → 1 pod
    assert shade.launched == []


# ── stale chain client (the silent no-trigger of 2026-07-14) ─────────────────


def test_frozen_block_rebuilds_chain_client_and_triggers(tmp_path):
    """A quietly-dead websocket keeps answering with a stale block; after
    stale_block_after_s the loop must rebuild the client and see the real
    height — otherwise it cycles forever and never rents."""
    prov = FakeProvider("lium")
    clock = Clock()
    stale = FakeChain(700)                      # frozen far from the boundary
    fresh = FakeChain(880)                      # the real height (in-window)
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    loop.chain_client = stale
    loop.chain_client_factory = lambda: fresh
    loop.stale_block_after_s = 300.0

    loop.run_once()                             # block seen, baseline set
    clock.t += 200.0
    loop.run_once()                             # frozen, but not stale yet
    assert plan_calls == [] and prov.launched == []
    clock.t += 200.0                            # now 400s frozen > 300s
    loop.run_once()                             # rebuild → block 880 → trigger
    assert plan_calls == [1]
    assert prov.launched != []


def test_raising_chain_client_rebuilds_once(tmp_path):
    class DeadChain:
        def current_block(self):
            raise ConnectionError("ws closed")

    prov = FakeProvider("lium")
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov})
    loop.chain_client = DeadChain()
    loop.chain_client_factory = lambda: FakeChain(880)
    loop.run_once()
    assert plan_calls == [1]                    # rebuilt and proceeded same cycle

# ── elastic eval pod (manifest-triggered; serves the validator) ──────────────


def _eval_policy(**kw):
    kw.setdefault("eval", StagePolicy(sku="NVIDIA L40S", gpus_per_pod=1, max_pods=1,
                                      providers=("lium",), max_price_hr=1.2))
    return _policy(**kw)


def _eval_loop(tmp_path, *, store, prov=None, clock=None, dry_run=False,
               receipt_prefix="receipts/5Val/", block=100):
    """A loop far outside the boundary margin (block=100): only the
    manifest-triggered eval machinery can act."""
    prov = prov or FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=block, clock=clock,
                        store=store, policy=_eval_policy(), dry_run=dry_run,
                        eval_hosts=tmp_path / "eval_hosts.toml",
                        receipt_prefix=receipt_prefix)
    return loop, prov


def test_new_manifest_rents_exactly_one_eval_pod_once(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store)
    loop.run_once()
    assert prov.launched == ["cascade-111-eval-0"]           # one pod, named for the round

    # Round-trip through the trainer's REAL loader (the validator's parser).
    hosts = load_hosts(tmp_path / "eval_hosts.toml")
    assert [h.name for h in hosts] == ["cascade-111-eval-0"]
    assert hosts[0].stage == "any"                           # matches the final/any filter
    assert hosts[0].host == "10.0.0.1" and hosts[0].cuda_device == "0"
    # …and through the validator's own lazy resolver.
    from cascade.validator.eval_offload import make_eval_host_fn
    resolved = make_eval_host_fn(tmp_path / "eval_hosts.toml")()
    assert resolved is not None and resolved.name == "cascade-111-eval-0"

    assert not (tmp_path / "hosts.toml").exists()            # trainer's file untouched
    st = load_state(tmp_path / "state.json")
    assert st.last_evaled_round == "111"                     # persisted rent-once latch
    assert {(i.stage, i.instance_id) for i in st.instances} == {("eval", "cascade-111-eval-0")}

    loop.run_once()                                          # same manifest: idempotent
    loop.run_once()
    assert prov.launched == ["cascade-111-eval-0"]


def test_eval_latch_persists_across_restart(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop1, prov = _eval_loop(tmp_path, store=store)
    loop1.run_once()
    loop2, prov2 = _eval_loop(tmp_path, store=store, prov=prov)  # fresh process, same ledger
    loop2.run_once()
    assert prov.launched == ["cascade-111-eval-0"]           # no double rent
    assert "cascade-111-eval-0" in prov.live                 # and the owned pod survives


def test_receipt_tears_down_eval_pod_and_clears_hosts_file(tmp_path):
    clock = Clock()
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, clock=clock)
    loop.run_once()
    clock.t += 600.0
    loop.run_once()
    assert "cascade-111-eval-0" in prov.live                 # no receipt yet: pod stays
    # The validator publishes the round's receipt under ITS OWN prefix.
    store.texts["receipts/5Val/round-111.json"] = '{"round_id": "111"}'
    clock.t += 600.0
    loop.run_once()
    assert prov.terminated == ["cascade-111-eval-0"] and prov.live == {}
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "eval_hosts.toml")             # cleared → validator falls local
    st = load_state(tmp_path / "state.json")
    assert st.instances == () and st.last_evaled_round == "111"
    loop.run_once()                                          # receipted round never re-rents
    assert prov.launched == ["cascade-111-eval-0"]


def test_newer_manifest_replaces_the_eval_pod(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store)
    loop.run_once()
    store.texts["manifests/latest.json"] = '{"round_id": "222"}'
    loop.run_once()
    # Round 111's evals are moot: its pod dies and round 222 gets its own —
    # teardown runs before the eval check, so the two never coexist.
    assert prov.terminated == ["cascade-111-eval-0"]
    assert prov.launched == ["cascade-111-eval-0", "cascade-222-eval-0"]
    hosts = load_hosts(tmp_path / "eval_hosts.toml")
    assert [h.name for h in hosts] == ["cascade-222-eval-0"]


def test_eval_ttl_backstop_fires_without_receipt_or_newer_manifest(tmp_path):
    clock = Clock()
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, clock=clock)
    loop.run_once()
    clock.t += 3 * 3600.0 - 1                                # one second shy of 1 epoch
    loop.run_once()
    assert "cascade-111-eval-0" in prov.live
    clock.t += 1.0
    loop.run_once()
    assert prov.live == {}                                   # TTL: silent validator ≠ bill
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "eval_hosts.toml")


def test_fresh_start_skips_a_round_already_receipted(tmp_path):
    # Restarted between rounds: the latest manifest is old news and its receipt
    # is up — renting would just buy a pod for the teardown sweep to kill.
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}',
                       "receipts/5Val/round-111.json": "{}"})
    loop, prov = _eval_loop(tmp_path, store=store)
    loop.run_once()
    assert prov.launched == []
    assert load_state(tmp_path / "state.json").last_evaled_round == "111"  # latched anyway


def test_absent_eval_policy_rents_nothing(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=100, store=store,
                        eval_hosts=tmp_path / "eval_hosts.toml")   # policy has no eval
    loop.run_once()
    assert prov.launched == [] and not (tmp_path / "eval_hosts.toml").exists()


def test_absent_eval_hosts_path_rents_nothing(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=100, store=store,
                        policy=_eval_policy())                     # nowhere to publish
    loop.run_once()
    assert prov.launched == []


def test_eval_dry_run_rents_nothing_and_touches_no_files(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, dry_run=True)
    loop.run_once()
    loop.run_once()                                          # in-memory latch: no hammering
    assert prov.launched == []
    assert not (tmp_path / "eval_hosts.toml").exists()
    assert load_state(tmp_path / "state.json") is None       # dry-run writes no ledger


def test_eval_no_capacity_degrades_and_latches(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, prov=FakeProvider("lium", available=False))
    loop.run_once()
    loop.run_once()                                          # latched: no 30s hammering
    assert prov.launched == []                               # validator evals run local
    assert load_state(tmp_path / "state.json").last_evaled_round == "111"


def test_eval_and_trainer_stages_coexist_in_separate_files(tmp_path):
    """A manifest lands while the boundary trigger fires: the eval pod goes to
    eval_hosts.toml, the trainer fleet to hosts.toml — never cross-published."""
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, block=880)
    loop.run_once()
    assert set(prov.launched) == {
        "cascade-111-eval-0", "cascade-900-heat-0", "cascade-900-final-0"}
    assert all(h.stage in ("heat", "final") for h in load_hosts(tmp_path / "hosts.toml"))
    assert [h.name for h in load_hosts(tmp_path / "eval_hosts.toml")] == ["cascade-111-eval-0"]


def test_reaper_accepts_eval_pods_as_self_named(tmp_path):
    from cascade.provision.loop import is_provisioner_pod_name

    assert is_provisioner_pod_name("cascade-900-eval-0")
    assert is_provisioner_pod_name("cascade-900-eval-r0-0")     # replacement suffix
    assert not is_provisioner_pod_name("cascade-eval-1")        # no round id ⇒ not ours
    prov = FakeProvider("lium")
    prov.live["cascade-900-eval-zombie"] = PodAddress("10.9.9.5", 22)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, block=100)
    loop.run_once()
    assert prov.terminated == ["cascade-900-eval-zombie"]       # orphan eval pod reaped


def test_build_policy_eval_table_is_optional():
    p = build_policy(_raw_config(), epoch_blocks=900)
    assert p.eval is None                                       # pre-eval configs unchanged
    raw = _raw_config(eval={"sku": "NVIDIA L40S", "gpus_per_pod": 1, "max_pods": 1,
                            "providers": ["lium"], "max_price_hr": 1.2})
    p = build_policy(raw, epoch_blocks=900)
    assert p.eval.sku == "NVIDIA L40S" and p.eval.max_pods == 1


def test_build_policy_rejects_bad_eval_table():
    raw = _raw_config(eval={"sku": "", "gpus_per_pod": 1, "max_pods": 1,
                            "providers": ["lium"], "max_price_hr": 1.2})
    with pytest.raises(ProvisionError):
        build_policy(raw, epoch_blocks=900)
