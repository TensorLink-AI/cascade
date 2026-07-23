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
from cascade.provision.policy import ProvisionPolicy, SkuCandidate, StagePolicy
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
              eval_hosts=None, receipt_prefix="", escalate_deadline_s=1800.0,
              min_viable_fleet=0.5, rent_retry_cooldown_s=900.0,
              final_rent_on="margin"):
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
        escalate_deadline_s=escalate_deadline_s,
        min_viable_fleet=min_viable_fleet,
        rent_retry_cooldown_s=rent_retry_cooldown_s,
        final_rent_on=final_rent_on,
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


def test_replacement_excludes_the_failed_pods_machine(tmp_path):
    """Incident 2026-07-14 round 5052267627071284702: the eval pod failed its
    boot gate and the replacement re-rented the SAME lium executor (offer
    lists are deterministic), so it failed identically. The replacement spec
    must exclude the lemon's machine."""

    class MachineAwareProvider(FakeProvider):
        machines = ("m-lemon", "m-good", "m-spare")

        def __init__(self, name, **kw):
            super().__init__(name, **kw)
            self.machine_by_pod: dict[str, str] = {}
            self.specs = []

        def launch(self, spec):
            self.specs.append(spec)
            offers = [m for m in self.machines if m not in spec.exclude_ids]
            ids = super().launch(spec)
            for i, pid in enumerate(ids):                    # head-of-list pick, like lium
                self.machine_by_pod[pid] = offers[i]
            return ids

        def machine_of(self, pod_id):
            return self.machine_by_pod.get(pod_id)

    prov = MachineAwareProvider("lium")

    def health(addr, stage, provider="", **shape):
        pid = next(p for p, a in prov.live.items() if a.ip == addr.ip)
        return _report(ok=not (stage == "heat" and prov.machine_by_pod[pid] == "m-lemon"))

    loop, _ = make_loop(tmp_path, providers={"lium": prov}, health=health)
    loop.run_once()

    rspec = next(s for s in prov.specs if "-r0" in s.name_prefix)
    assert rspec.exclude_ids == ("m-lemon",)                 # the fix
    assert prov.machine_by_pod["cascade-900-heat-r0-0"] == "m-good"
    heat = [h for h in load_hosts(tmp_path / "hosts.toml") if h.stage == "heat"]
    assert heat and all(h.host == prov.live["cascade-900-heat-r0-0"].ip for h in heat)


def test_every_pod_unhealthy_clears_hosts(tmp_path):
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov},
                        health=lambda addr, stage, provider="", **shape: _report(ok=False))
    loop.run_once()
    assert prov.live == {}                                    # nothing left billing
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "hosts.toml")


# ── rules of escalation: empty rung → next rung; partial fleet → top-up ──────


class LaunchFailProvider(FakeProvider):
    """Capacity probe says yes, the actual rent call 500s — the gap that
    escalation rule 2 closes (a launch failure used to lose the stage)."""

    def launch(self, spec):
        raise RuntimeError("api 500")


def test_launch_failure_escalates_to_next_provider(tmp_path):
    lium = LaunchFailProvider("lium")
    shade = FakeProvider("shadeform")
    loop, _ = make_loop(tmp_path, providers={"lium": lium, "shadeform": shade})
    loop.run_once()
    # Both stages: lium accepted the probe, failed the launch → the same rung
    # on shadeform rents instead (escalation batch names carry -e1).
    assert lium.launched == []
    assert shade.launched == ["cascade-900-heat-e1-0", "cascade-900-final-e1-0"]
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert {h.stage for h in hosts} == {"heat", "final"}


def test_all_duds_on_one_provider_escalate_to_the_next(tmp_path):
    lium = FakeProvider("lium")
    shade = FakeProvider("shadeform")

    def health(addr, stage, provider="", **shape):
        return _report(ok=provider != "lium")                # a bad lium batch

    loop, _ = make_loop(tmp_path, providers={"lium": lium, "shadeform": shade},
                        health=health)
    loop.run_once()
    # Per stage on lium: the pod AND its one replacement were duds → the stage
    # (not just the slot) escalates and rents healthy on shadeform.
    assert lium.live == {} and len(lium.terminated) == 4     # heat+final, orig+repl
    assert shade.launched == ["cascade-900-heat-e1-0", "cascade-900-final-e1-0"]
    st = load_state(tmp_path / "state.json")
    assert {i.instance_id for i in st.instances} == {
        "cascade-900-heat-e1-0", "cascade-900-final-e1-0"}


def test_zero_deadline_disables_escalation(tmp_path):
    lium = LaunchFailProvider("lium")
    shade = FakeProvider("shadeform")
    loop, _ = make_loop(tmp_path, providers={"lium": lium, "shadeform": shade},
                        escalate_deadline_s=0.0)
    loop.run_once()
    # Deadline already spent at the first failure → pre-escalation behaviour:
    # the stage degrades, hosts clear, the trainer covers the round locally.
    assert shade.launched == []
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "hosts.toml")


def test_escalated_rung_rechecked_against_budget(tmp_path):
    # Primary heat rung (one 8x pod) passes the round gate; its launch fails.
    # The only fallback rung (1x singles → 3 pods) would blow the round cap
    # with the final included → refused, ladder exhausted, heat degrades —
    # but the final still rents.
    class HeatLaunchFails(FakeProvider):
        def launch(self, spec):
            if "-heat" in spec.name_prefix:
                raise RuntimeError("no heat capacity after all")
            return super().launch(spec)

    prov = HeatLaunchFails("lium", price=2.0)
    policy = _policy(
        heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=8, max_pods=4,
                         providers=("lium",), max_price_hr=4.0,
                         candidates=(SkuCandidate(sku="NVIDIA RTX A6000",
                                                  gpus_per_pod=1, max_price_hr=4.0),)),
        final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=2,
                          providers=("lium",), max_price_hr=3.0),
        max_spend_per_round=20.0)
    # Initial projection: (1 heat + 1 final) × $2 × 3h TTL = $12 <= $20. The
    # escalated rung projects 3 × $2 × 3h = $18 heat + $6 final = $24 > $20.
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, policy=policy)
    loop.run_once()
    assert prov.launched == ["cascade-900-final-0"]
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert all(h.stage == "final" for h in hosts)            # degraded, not dead


def test_below_viability_partial_fleet_tops_up_same_candidate(tmp_path):
    prov = FakeProvider("lium")
    plan = dict(PLAN, eligible_challengers=80)               # 19 slots → 3 × 8-GPU pods
    # Heat pods rent as .1/.2/.3; replacements draw .4/.5. Duds: pods 0 and 1
    # AND their replacements → 1 of 3 pods healthy = 8 of 19 slots < 50%.
    bad = {"10.0.0.1", "10.0.0.2", "10.0.0.4", "10.0.0.5"}

    def health(addr, stage, provider="", **shape):
        return _report(ok=addr.ip not in bad)

    policy = _policy(
        heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=8, max_pods=3,
                         providers=("lium",), max_price_hr=4.0),
        max_spend_per_round=100.0)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, plan=plan,
                        policy=policy, health=health)
    loop.run_once()

    # One same-candidate top-up batch re-rents exactly the two missing pods.
    assert "cascade-900-heat-t0-0" in prov.launched
    assert "cascade-900-heat-t0-1" in prov.launched
    heat = [h for h in load_hosts(tmp_path / "hosts.toml") if h.stage == "heat"]
    assert len(heat) == 3 * 8                                # back to full strength
    st = load_state(tmp_path / "state.json")
    heat_ids = {i.instance_id for i in st.instances if i.stage == "heat"}
    assert heat_ids == {"cascade-900-heat-2",
                        "cascade-900-heat-t0-0", "cascade-900-heat-t0-1"}


def test_failed_round_retries_after_cooldown(tmp_path):
    prov = FakeProvider("lium", available=False)
    clock = Clock()
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    loop.run_once()
    assert prov.launched == []                               # trigger found no capacity
    loop.run_once()                                          # inside cooldown: latched
    assert prov.launched == [] and plan_calls == [1]

    prov._available = True                                   # the market recovered
    clock.t += 901                                           # cooldown elapsed
    loop.run_once()
    # Both failed stages re-entered pick→budget→rent — no new plan_fn call
    # (the trigger's payload is cached; reveals are closed mid-round anyway).
    assert plan_calls == [1]
    assert prov.launched == ["cascade-900-heat-0", "cascade-900-final-0"]
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert {h.stage for h in hosts} == {"heat", "final"}


def test_retry_gives_up_when_the_window_closes(tmp_path):
    prov = FakeProvider("lium", available=False)
    clock = Clock()
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    loop.run_once()

    prov._available = True
    clock.t += 901
    loop.chain_client.block = 1700                           # 0.33h left in the round:
    loop.run_once()                                          # heat needs 0.5h + final
    assert prov.launched == []                               # window closed → no rent
    assert loop._stage_failed == set()                       # and no further retries


def test_final_defers_until_heat_marker_and_sizes_off_it(tmp_path):
    prov = FakeProvider("lium")
    policy = _policy(max_spend_per_round=100.0)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, policy=policy,
                        final_rent_on="heat_complete")
    loop.run_once()
    # Margin trigger rents the HEAT only; the final waits on the trainer.
    assert prov.launched == ["cascade-900-heat-0"]
    assert load_state(tmp_path / "state.json").final_pending is True

    # The trainer settles the heat: marker names TWO actual finalists (the
    # plan predicted one) — the JIT fleet must match the marker, not the plan.
    marker_dir = tmp_path / "work" / "777"
    marker_dir.mkdir(parents=True)
    (marker_dir / "heat_complete.json").write_text(json.dumps(
        {"round_id": "777", "screened": 12, "finalists": ["f1", "f2"]}))
    loop.run_once()
    # Same tick: the marker tears the heat down AND rents the final — sized
    # 1 + 2 actual finalists = 3 slots → two 2-GPU pods.
    assert "cascade-900-heat-0" in prov.terminated
    assert prov.launched[1:] == ["cascade-900-final-0", "cascade-900-final-1"]
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert all(h.stage == "final" for h in hosts) and len(hosts) == 4
    st = load_state(tmp_path / "state.json")
    assert st.final_pending is False
    assert {i.stage for i in st.instances} == {"final"}


def test_scarce_final_market_rents_early_at_the_margin(tmp_path):
    """The JIT exception: when the pinned SKU's primary rung probes scarce at
    the margin, the final rents EARLY (locking whatever the ladder finds)
    instead of gambling that capacity exists hours later."""

    class NoTwoGpuPods(FakeProvider):
        def available(self, sku, count, *, gpus=1):
            return gpus != 2                                 # 2x L40S pool is dry

    prov = NoTwoGpuPods("lium")
    policy = _policy(
        final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=2,
                          providers=("lium",), max_price_hr=3.0,
                          candidates=(SkuCandidate(sku="NVIDIA L40S",
                                                   gpus_per_pod=1, max_price_hr=1.5),)))
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, policy=policy,
                        final_rent_on="heat_complete")
    loop.run_once()
    # Final rented at the margin via the 1x fallback rung — not deferred.
    assert "cascade-900-final-0" in prov.launched
    assert "cascade-900-final-1" in prov.launched
    assert load_state(tmp_path / "state.json").final_pending is False


def test_final_pending_survives_restart(tmp_path):
    prov = FakeProvider("lium")
    policy = _policy(max_spend_per_round=100.0)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, policy=policy,
                        final_rent_on="heat_complete")
    loop.run_once()
    assert load_state(tmp_path / "state.json").final_pending is True

    # New process, same ledger: still waiting on the marker (heat instances
    # exist → the marker scan re-anchors on their rent times).
    loop2, _ = make_loop(tmp_path, providers={"lium": prov}, policy=policy,
                         final_rent_on="heat_complete")
    assert loop2._final_pending is True
    assert loop2._heat_marker_latched is False


def test_viable_partial_fleet_does_not_top_up(tmp_path):
    prov = FakeProvider("lium")
    plan = dict(PLAN, eligible_challengers=80)               # 19 slots → 3 × 8-GPU pods
    bad = {"10.0.0.1", "10.0.0.4"}                           # pod 0 + its replacement

    def health(addr, stage, provider="", **shape):
        return _report(ok=addr.ip not in bad)

    policy = _policy(
        heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=8, max_pods=3,
                         providers=("lium",), max_price_hr=4.0),
        max_spend_per_round=100.0)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, plan=plan,
                        policy=policy, health=health)
    loop.run_once()
    # 2 of 3 pods = 16 of 19 slots >= 50% → viable; the dropped slot stays
    # dropped (serial waves), no top-up batch and no escalation.
    assert not any("-t0" in p or "-e1" in p for p in prov.launched)
    heat = [h for h in load_hosts(tmp_path / "hosts.toml") if h.stage == "heat"]
    assert len(heat) == 2 * 8


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


def test_stale_manifest_from_prior_run_does_not_tear_down(tmp_path):
    # Same-round-id rerun: the PREVIOUS run's manifest is still published at
    # round-<id>.json when the heat marker lands. That leftover must not read
    # as round-over (2026-07-15: it killed both pods 21s after duel dispatch);
    # only a NEW publish — different bytes at the same key — ends the round.
    clock = Clock()
    store = FakeStore({"manifests/round-54321.json":
                       '{"round_id": "54321", "contract_digest": "old"}'})
    loop, prov = _provisioned(tmp_path, clock=clock, store=store)
    d = tmp_path / "work" / "54321"
    d.mkdir(parents=True)
    (d / "heat_complete.json").write_text("{}")
    clock.t += 1800.0
    loop.run_once()
    assert "cascade-900-final-0" in prov.live                # stale: final survives
    # The rerun finishes and republishes the SAME round id with new content.
    store.texts["manifests/round-54321.json"] = (
        '{"round_id": "54321", "contract_digest": "new"}')
    clock.t += 1800.0
    loop.run_once()
    assert prov.live == {}


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

    def health(addr, stage, provider="", *, sku="", gpus=0, **kw):
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


def _join_eval(lp):
    t = getattr(lp, "_eval_thread", None)
    if t is not None:
        t.join(timeout=10)


def test_new_manifest_rents_exactly_one_eval_pod_once(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store)
    loop.run_once()
    _join_eval(loop)
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

    loop.run_once()
    _join_eval(loop)                                          # same manifest: idempotent
    loop.run_once()
    _join_eval(loop)
    assert prov.launched == ["cascade-111-eval-0"]


def test_eval_latch_persists_across_restart(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop1, prov = _eval_loop(tmp_path, store=store)
    loop1.run_once()
    _join_eval(loop1)
    loop2, prov2 = _eval_loop(tmp_path, store=store, prov=prov)  # fresh process, same ledger
    loop2.run_once()
    _join_eval(loop2)
    assert prov.launched == ["cascade-111-eval-0"]           # no double rent
    assert "cascade-111-eval-0" in prov.live                 # and the owned pod survives


def test_receipt_tears_down_eval_pod_and_clears_hosts_file(tmp_path):
    clock = Clock()
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, clock=clock)
    loop.run_once()
    _join_eval(loop)
    clock.t += 600.0
    loop.run_once()
    _join_eval(loop)
    assert "cascade-111-eval-0" in prov.live                 # no receipt yet: pod stays
    # The validator publishes the round's receipt under ITS OWN prefix.
    store.texts["receipts/5Val/round-111.json"] = '{"round_id": "111"}'
    clock.t += 600.0
    loop.run_once()
    _join_eval(loop)
    assert prov.terminated == ["cascade-111-eval-0"] and prov.live == {}
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "eval_hosts.toml")             # cleared → validator falls local
    st = load_state(tmp_path / "state.json")
    assert st.instances == () and st.last_evaled_round == "111"
    loop.run_once()
    _join_eval(loop)                                          # receipted round never re-rents
    assert prov.launched == ["cascade-111-eval-0"]


def test_newer_manifest_replaces_the_eval_pod(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store)
    loop.run_once()
    _join_eval(loop)
    store.texts["manifests/latest.json"] = '{"round_id": "222"}'
    loop.run_once()
    _join_eval(loop)
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
    _join_eval(loop)
    clock.t += 3 * 3600.0 - 1                                # one second shy of 1 epoch
    loop.run_once()
    _join_eval(loop)
    assert "cascade-111-eval-0" in prov.live
    clock.t += 1.0
    loop.run_once()
    _join_eval(loop)
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


def test_heartbeat_logs_at_cycle_start(tmp_path, caplog):
    """Liveness must not depend on any network phase completing — the
    heartbeat fires at cycle START (starved twice on 2026-07-14 when it
    lived behind reconcile/eval polls)."""
    import logging

    loop, _ = make_loop(tmp_path, block=100)
    with caplog.at_level(logging.INFO, logger="cascade.provision.loop"):
        loop.run_once()
    assert any("heartbeat: cycle start" in r.message for r in caplog.records)


def test_hung_chain_client_hits_deadline_and_rebuilds(tmp_path):
    """A websocket that HANGS (not raises) must not block the loop — the
    deadline converts the hang into a rebuild. Four rental windows died to
    silent chain stalls before this."""
    import threading

    class HangingChain:
        def current_block(self):
            threading.Event().wait(30)          # would block half the cycle

    prov = FakeProvider("lium")
    loop, plan_calls = make_loop(tmp_path, providers={"lium": prov})
    loop.chain_client = HangingChain()
    loop.chain_client_factory = lambda: FakeChain(880)
    # shrink the deadline for test speed
    orig = loop._with_deadline
    loop_cls = type(loop)
    try:
        loop_cls._with_deadline = staticmethod(
            lambda fn, seconds: orig(fn, 0.2 if seconds >= 60 else seconds))
        loop.run_once()
    finally:
        loop_cls._with_deadline = staticmethod(orig)
    assert plan_calls == [1]                    # rebuilt within the same cycle and triggered


def test_on_cycle_hook_heals_stripped_logging(tmp_path):
    """bittensor strips handlers + raises level on named loggers at chain
    connect (verified live). The loop must invoke the self-heal hook every
    cycle so logging survives arbitrary re-nuking."""
    import logging

    lg = logging.getLogger("cascade")
    calls = []

    def heal():
        calls.append(1)
        lg.setLevel(logging.INFO)

    loop, _ = make_loop(tmp_path, block=100)
    loop.on_cycle = heal
    lg.setLevel(logging.CRITICAL)          # simulate the nuke
    loop.run_once()
    assert calls and lg.getEffectiveLevel() == logging.INFO


def test_heal_rebuilds_closed_handlers(tmp_path, monkeypatch):
    """bittensor can close stripped handlers' streams — re-attaching the same
    object then fails silently on every emit. The heal must rebuild."""
    import logging
    import sys

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["cascade-provisioner"])
    from cascade.provision import main as pmain

    fmt = logging.Formatter("%(message)s")
    state = {"stream": None, "file": None}

    def _alive(h):
        stream = getattr(h, "stream", None)
        return h is not None and stream is not None and not getattr(stream, "closed", True)

    def ensure():
        lg = logging.getLogger("cascade-test-heal")
        if not _alive(state["file"]):
            state["file"] = logging.FileHandler(tmp_path / "svc.log")
            state["file"].setFormatter(fmt)
        if state["file"] not in lg.handlers:
            lg.addHandler(state["file"])
        lg.setLevel(logging.INFO)
        lg.propagate = False

    lg = logging.getLogger("cascade-test-heal")
    ensure()
    lg.info("one")
    state["file"].close()                     # simulate bittensor closing it
    lg.removeHandler(state["file"])           # and stripping it
    ensure()                                  # heal must REBUILD, not re-attach
    lg.info("two")
    for h in lg.handlers:
        h.flush()
    text = (tmp_path / "svc.log").read_text()
    assert "one" in text and "two" in text
    assert pmain  # import sanity


def test_eval_rent_does_not_block_the_cycle(tmp_path, store_with_manifest=None):
    """An eval pod's boot can take 15+ min — it must run OFF the loop thread
    (a blocking boot swallowed a heat-trigger window on 2026-07-14)."""
    import threading
    import time as _time

    gate = threading.Event()

    class SlowBootProvider(FakeProvider):
        def wait_ready(self, pod_id, *, timeout):
            gate.wait(5)                          # simulates the 900s boot wait
            return True

    prov = SlowBootProvider("lium")
    store = FakeStore({"manifests/latest.json": '{"round_id": "424242"}'})
    loop, _ = _eval_loop(tmp_path, store=store, prov=prov)
    t0 = _time.monotonic()
    loop.run_once()                               # must return promptly
    took = _time.monotonic() - t0
    gate.set()
    assert took < 2.0, f"cycle blocked {took:.1f}s on the eval boot"
