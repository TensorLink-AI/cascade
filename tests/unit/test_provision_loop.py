"""Service loop — the eval-pod rent→publish→teardown cycle, the ledger and
the hosts files, on fakes.

Every boundary (chain, providers, manifest store, health, clock) is injected,
so these tests drive whole rentals without a cloud account or a GPU. Every
hosts file the loop publishes is re-parsed with the trainer's own loader —
the contract, not the string, is asserted."""

from __future__ import annotations

import pytest

from cascade.provision.core import PodAddress, ProvisionError
from cascade.provision.health import CheckResult, HealthReport
from cascade.provision.loop import ProvisionerLoop, RenderSettings
from cascade.provision.main import build_policy, provider_names
from cascade.provision.policy import ProvisionPolicy, SkuCandidate, StagePolicy
from cascade.provision.state import PodInstance, RoundState, load_state, save_state
from cascade.trainer.remote import RemoteDispatchError, load_hosts

IMG = "reg.example/cascade-worker@sha256:" + "a" * 64
MANIFEST = {"manifests/latest.json": '{"round_id": "111"}'}


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


class ShapedProvider(FakeProvider):
    """A marketplace that only stocks specific (sku, gpus) shapes."""

    def __init__(self, name, shapes, price=None):
        super().__init__(name, price=price)
        self.shapes = set(shapes)                       # {(market_sku, gpus)}

    def available(self, sku, count, *, gpus=1):
        return (sku, gpus) in self.shapes


def _eval_policy(**kw):
    kw.setdefault("eval", StagePolicy(sku="NVIDIA L40S", gpus_per_pod=1, max_pods=1,
                                      providers=("lium", "shadeform"), max_price_hr=1.2))
    kw.setdefault("max_spend_per_round", 25.0)
    return ProvisionPolicy(**kw)


def make_loop(tmp_path, *, providers=None, block=100, policy=None, clock=None, store=None,
              health=None, dry_run=False, eval_hosts=None, receipt_prefix=""):
    providers = providers if providers is not None else {"lium": FakeProvider("lium")}
    return ProvisionerLoop(
        policy=policy or _eval_policy(),
        providers=providers,
        chain_client=FakeChain(block),
        render=RenderSettings(image=IMG, ssh_pubkey="ssh-ed25519 AAAA orch",
                              key_path="~/.ssh/cascade_ed25519"),
        hosts_path=tmp_path / "hosts.toml",
        work_root=tmp_path / "work",
        state_path=tmp_path / "state.json",
        epoch_blocks=900,                    # 3h epochs → ttl 3h at ttl_epochs=1
        manifest_store=store,
        eval_hosts_path=eval_hosts,
        receipt_prefix=receipt_prefix,
        health_check=health,
        dry_run=dry_run,
        clock=clock or Clock(),
    )


def _eval_loop(tmp_path, *, store=None, prov=None, providers=None, clock=None,
               dry_run=False, receipt_prefix="receipts/5Val/", health=None, policy=None):
    prov = prov or FakeProvider("lium")
    loop = make_loop(tmp_path, providers=providers or {"lium": prov}, clock=clock,
                     store=store if store is not None else FakeStore(MANIFEST),
                     policy=policy, dry_run=dry_run, health=health,
                     eval_hosts=tmp_path / "eval_hosts.toml", receipt_prefix=receipt_prefix)
    return loop, prov


def cycle(loop):
    """One poll tick with the eval rent worker joined, so assertions see the
    settled world."""
    loop.run_once()
    t = getattr(loop, "_eval_thread", None)
    if t is not None:
        t.join(timeout=30)


def _seed_ledger(tmp_path, prov, pods, *, clock=None):
    """A ledger left by an earlier run: ``pods`` = [(stage, instance_id, sku, gpus)],
    live on ``prov``."""
    clock = clock or Clock()
    from datetime import UTC, datetime

    iso = datetime.fromtimestamp(clock(), tz=UTC).isoformat()
    insts = []
    for i, (stage, pid, sku, gpus) in enumerate(pods):
        prov.live[pid] = PodAddress(f"10.0.1.{i + 1}", 22)
        insts.append(PodInstance(provider=prov.name, instance_id=pid, stage=stage,
                                 rented_at_iso=iso, sku=sku, gpus=gpus))
    save_state(tmp_path / "state.json", RoundState(round_id="900", instances=tuple(insts),
                                                   published=True))


def test_new_manifest_rents_exactly_one_eval_pod_once(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store)
    cycle(loop)
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

    cycle(loop)                                               # same manifest: idempotent
    cycle(loop)
    assert prov.launched == ["cascade-111-eval-0"]


def test_eval_latch_persists_across_restart(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop1, prov = _eval_loop(tmp_path, store=store)
    cycle(loop1)
    loop2, prov2 = _eval_loop(tmp_path, store=store, prov=prov)  # fresh process, same ledger
    cycle(loop2)
    assert prov.launched == ["cascade-111-eval-0"]           # no double rent
    assert "cascade-111-eval-0" in prov.live                 # and the owned pod survives


def test_receipt_tears_down_eval_pod_and_clears_hosts_file(tmp_path):
    clock = Clock()
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, clock=clock)
    cycle(loop)
    clock.t += 600.0
    cycle(loop)
    assert "cascade-111-eval-0" in prov.live                 # no receipt yet: pod stays
    # The validator publishes the round's receipt under ITS OWN prefix.
    store.texts["receipts/5Val/round-111.json"] = '{"round_id": "111"}'
    clock.t += 600.0
    cycle(loop)
    assert prov.terminated == ["cascade-111-eval-0"] and prov.live == {}
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "eval_hosts.toml")             # cleared → validator falls local
    st = load_state(tmp_path / "state.json")
    assert st.instances == () and st.last_evaled_round == "111"
    cycle(loop)                                          # receipted round never re-rents
    assert prov.launched == ["cascade-111-eval-0"]


def test_newer_manifest_replaces_the_eval_pod(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store)
    cycle(loop)
    store.texts["manifests/latest.json"] = '{"round_id": "222"}'
    cycle(loop)
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
    cycle(loop)
    clock.t += 3 * 3600.0 - 1                                # one second shy of 1 epoch
    cycle(loop)
    assert "cascade-111-eval-0" in prov.live
    clock.t += 1.0
    cycle(loop)
    assert prov.live == {}                                   # TTL: silent validator ≠ bill
    with pytest.raises(RemoteDispatchError):
        load_hosts(tmp_path / "eval_hosts.toml")


def test_fresh_start_skips_a_round_already_receipted(tmp_path):
    # Restarted between rounds: the latest manifest is old news and its receipt
    # is up — renting would just buy a pod for the teardown sweep to kill.
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}',
                       "receipts/5Val/round-111.json": "{}"})
    loop, prov = _eval_loop(tmp_path, store=store)
    cycle(loop)
    assert prov.launched == []
    assert load_state(tmp_path / "state.json").last_evaled_round == "111"  # latched anyway


def test_absent_eval_policy_rents_nothing(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    prov = FakeProvider("lium")
    loop = make_loop(tmp_path, providers={"lium": prov}, store=store,
                     policy=ProvisionPolicy(max_spend_per_round=25.0),
                     eval_hosts=tmp_path / "eval_hosts.toml")      # policy has no eval
    cycle(loop)
    assert prov.launched == [] and not (tmp_path / "eval_hosts.toml").exists()


def test_absent_eval_hosts_path_rents_nothing(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    prov = FakeProvider("lium")
    loop = make_loop(tmp_path, providers={"lium": prov}, store=store)   # nowhere to publish
    cycle(loop)
    assert prov.launched == []


def test_eval_dry_run_rents_nothing_and_touches_no_files(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, dry_run=True)
    cycle(loop)
    cycle(loop)                                          # in-memory latch: no hammering
    assert prov.launched == []
    assert not (tmp_path / "eval_hosts.toml").exists()
    assert load_state(tmp_path / "state.json") is None       # dry-run writes no ledger


def test_eval_no_capacity_degrades_and_latches(tmp_path):
    store = FakeStore({"manifests/latest.json": '{"round_id": "111"}'})
    loop, prov = _eval_loop(tmp_path, store=store, prov=FakeProvider("lium", available=False))
    cycle(loop)
    cycle(loop)                                          # latched: no 30s hammering
    assert prov.launched == []                               # validator evals run local
    assert load_state(tmp_path / "state.json").last_evaled_round == "111"


def test_reaper_accepts_eval_pods_as_self_named(tmp_path):
    from cascade.provision.loop import is_provisioner_pod_name

    assert is_provisioner_pod_name("cascade-900-eval-0")
    assert is_provisioner_pod_name("cascade-900-eval-r0-0")     # replacement suffix
    assert not is_provisioner_pod_name("cascade-eval-1")        # no round id ⇒ not ours
    prov = FakeProvider("lium")
    prov.live["cascade-900-eval-zombie"] = PodAddress("10.9.9.5", 22)
    loop = make_loop(tmp_path, providers={"lium": prov})
    cycle(loop)
    assert prov.terminated == ["cascade-900-eval-zombie"]       # orphan eval pod reaped


def test_reconcile_never_touches_untagged_pods(tmp_path):
    prov = FakeProvider("lium")
    prov.live["someone-elses-pod"] = PodAddress("10.9.9.9", 22)
    loop = make_loop(tmp_path, providers={"lium": prov})
    cycle(loop)
    assert prov.terminated == []


def test_dry_run_never_terminates_anything(tmp_path):
    """--dry-run must gate EVERY provider mutation, not just rentals: the
    reaper (and teardown) once terminated a live pod during a dry-run demo."""
    prov = FakeProvider("lium")
    prov.live["cascade-900-heat-zombie"] = PodAddress("10.9.9.4", 22)
    loop = make_loop(tmp_path, providers={"lium": prov})
    loop.dry_run = True
    cycle(loop)
    assert prov.terminated == []


def test_heartbeat_logs_at_cycle_start(tmp_path, caplog):
    """Liveness must not depend on any network phase completing — the
    heartbeat fires at cycle START."""
    import logging

    loop = make_loop(tmp_path)
    with caplog.at_level(logging.INFO, logger="cascade.provision.loop"):
        cycle(loop)
    assert any("heartbeat: cycle start" in r.message for r in caplog.records)


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

    loop = make_loop(tmp_path)
    loop.on_cycle = heal
    lg.setLevel(logging.CRITICAL)          # simulate the nuke
    cycle(loop)
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


def test_eval_rent_does_not_block_the_cycle(tmp_path):
    """An eval pod's boot can take 15+ min — it must run OFF the loop thread
    so teardown/heartbeat/reconcile keep ticking."""
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
    loop.run_once()                           # must return promptly
    took = _time.monotonic() - t0
    gate.set()
    assert took < 2.0, f"cycle blocked {took:.1f}s on the eval boot"


# ── provider ladder, budget, health gate (the eval rental) ───────────────────


def test_provider_down_falls_through_to_next(tmp_path):
    lium = FakeProvider("lium", avail_raises=RuntimeError("api down"))
    shade = FakeProvider("shadeform")
    loop, _ = _eval_loop(tmp_path, providers={"lium": lium, "shadeform": shade})
    cycle(loop)
    assert lium.launched == [] and shade.launched == ["cascade-111-eval-0"]


def test_overpriced_offer_skips_that_provider(tmp_path):
    dear = FakeProvider("lium", price=99.0)
    fair = FakeProvider("shadeform", price=1.0)
    loop, _ = _eval_loop(tmp_path, providers={"lium": dear, "shadeform": fair})
    cycle(loop)
    assert dear.launched == [] and fair.launched == ["cascade-111-eval-0"]


def test_budget_breaker_refuses_to_rent(tmp_path):
    # Worst case (ttl = 3h epoch): 1 pod × $1.0 × 3h = $3 > the $2 cap.
    prov = FakeProvider("lium", price=1.0)
    loop, _ = _eval_loop(tmp_path, prov=prov, policy=_eval_policy(max_spend_per_round=2.0))
    cycle(loop)
    assert prov.launched == []
    assert load_state(tmp_path / "state.json").last_evaled_round == "111"   # latched


def test_unhealthy_pod_replaced_once(tmp_path):
    prov = FakeProvider("lium")
    loop, _ = _eval_loop(tmp_path, prov=prov,
                         health=lambda addr, stage, provider="", **k: _report(
                             ok=addr.ip != "10.0.0.1"))
    cycle(loop)
    assert prov.terminated == ["cascade-111-eval-0"]
    assert prov.launched == ["cascade-111-eval-0", "cascade-111-eval-r0-0"]
    hosts = load_hosts(tmp_path / "eval_hosts.toml")
    assert [h.host for h in hosts] == ["10.0.0.2"]           # the replacement


def test_replacement_also_unhealthy_drops_the_pod(tmp_path):
    prov = FakeProvider("lium")
    loop, _ = _eval_loop(tmp_path, prov=prov,
                         health=lambda addr, stage, provider="", **k: _report(ok=False))
    cycle(loop)
    assert prov.terminated == ["cascade-111-eval-0", "cascade-111-eval-r0-0"]
    assert load_state(tmp_path / "state.json").instances == ()
    assert not (tmp_path / "eval_hosts.toml").exists()       # validator evals stay local


def test_replacement_excludes_the_failed_pods_machine(tmp_path):
    """Offer lists are deterministic: the replacement spec must exclude the
    lemon's machine or it re-rents the same failed executor."""

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
        return _report(ok=prov.machine_by_pod[pid] != "m-lemon")

    loop, _ = _eval_loop(tmp_path, prov=prov, health=health)
    cycle(loop)
    rspec = next(s for s in prov.specs if "-r0" in s.name_prefix)
    assert rspec.exclude_ids == ("m-lemon",)
    assert prov.machine_by_pod["cascade-111-eval-r0-0"] == "m-good"


def test_image_digest_failure_excludes_the_facility(tmp_path):
    """A stale facility image cache re-boots the same wrong image in the same
    region: an image_digest gate failure walks the replacement to another."""

    class RegionAwareProvider(FakeProvider):
        regions = ("kansascity-usa-1", "chicago-usa-2")

        def __init__(self, name, **kw):
            super().__init__(name, **kw)
            self.region_by_pod: dict[str, str] = {}
            self.specs = []

        def launch(self, spec):
            self.specs.append(spec)
            region = next(r for r in self.regions if r not in spec.exclude_regions)
            ids = super().launch(spec)
            for pid in ids:
                self.region_by_pod[pid] = region
            return ids

        def region_of(self, pod_id):
            return self.region_by_pod.get(pod_id)

    prov = RegionAwareProvider("lium")

    def health(addr, stage, provider="", **shape):
        pid = next(p for p, a in prov.live.items() if a.ip == addr.ip)
        return _report(ok=prov.region_by_pod[pid] != "kansascity-usa-1", name="image_digest")

    loop, _ = _eval_loop(tmp_path, prov=prov, health=health)
    cycle(loop)
    rspec = next(s for s in prov.specs if "-r0" in s.name_prefix)
    assert rspec.exclude_regions == ("kansascity-usa-1",)
    assert prov.region_by_pod["cascade-111-eval-r0-0"] == "chicago-usa-2"
    assert loop._gate_failures == {}                         # consumed, no leak


def test_bootstrap_failure_replaces_pod_once(tmp_path):
    calls = []

    def flaky_bootstrap(addr, stage, provider=""):
        calls.append(addr.ip)
        return len(calls) > 1                     # first pod fails, replacement passes

    loop, prov = _eval_loop(tmp_path)
    loop.bootstrap = flaky_bootstrap
    cycle(loop)
    assert len(calls) == 2 and prov.terminated == ["cascade-111-eval-0"]


def test_prewarm_hook_fires_per_healthy_pod_and_never_gates(tmp_path):
    loop, prov = _eval_loop(tmp_path)
    calls = []

    def prewarm(addr, stage, provider=""):
        calls.append((stage, provider))
        raise RuntimeError("boom")

    loop.prewarm = prewarm
    cycle(loop)
    assert calls == [("eval", "lium")]
    assert prov.terminated == [] and load_hosts(tmp_path / "eval_hosts.toml")


def test_sku_fallback_takes_first_candidate_with_capacity(tmp_path):
    seen = []

    def health(addr, stage, provider="", *, sku="", gpus=0, **kw):
        seen.append((stage, sku, gpus))
        return _report(ok=True)

    policy = _eval_policy(eval=StagePolicy(
        sku="NVIDIA L40S", gpus_per_pod=1, max_pods=1, providers=("lium", "shadeform"),
        max_price_hr=1.2, candidates=(SkuCandidate(sku="NVIDIA RTX A6000",
                                                   market_sku="A6000", max_price_hr=1.0),)))
    lium = ShapedProvider("lium", shapes=set())
    shade = ShapedProvider("shadeform", shapes={("A6000", 1)})
    loop, _ = _eval_loop(tmp_path, providers={"lium": lium, "shadeform": shade},
                         policy=policy, health=health)
    cycle(loop)
    assert shade.launched == ["cascade-111-eval-0"]
    assert seen == [("eval", "NVIDIA RTX A6000", 1)]          # the gate asserts the RENTED sku
    inst, = load_state(tmp_path / "state.json").instances
    assert (inst.sku, inst.gpus) == ("NVIDIA RTX A6000", 1)


# ── ledger: TTL, restart, re-gate, hosts.toml ────────────────────────────────


def test_ttl_reaps_a_leftover_fleet_pod_and_republishes(tmp_path):
    """A heat/final pod left in the ledger by the retired per-round fleet has
    no teardown signal any more — the TTL alone reaps it."""
    clock = Clock()
    prov = FakeProvider("lium")
    _seed_ledger(tmp_path, prov, [("final", "cascade-900-final-0", "NVIDIA L40S", 2)],
                 clock=clock)
    loop = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    clock.t += 3 * 3600.0 - 1
    cycle(loop)
    assert prov.terminated == []
    clock.t += 1.0
    cycle(loop)
    assert prov.terminated == ["cascade-900-final-0"]
    assert load_state(tmp_path / "state.json").instances == ()


def test_restart_resumes_ledger_and_kills_orphans(tmp_path):
    prov = FakeProvider("lium")
    _seed_ledger(tmp_path, prov, [("final", "cascade-900-final-0", "NVIDIA L40S", 2)])
    prov.live["cascade-900-heat-zombie"] = PodAddress("10.0.0.99", 22)   # never ledgered
    loop = make_loop(tmp_path, providers={"lium": prov})
    cycle(loop)
    assert prov.terminated == ["cascade-900-heat-zombie"]
    assert "cascade-900-final-0" in prov.live                            # owned: untouched


def test_reconcile_never_touches_hand_rented_cascade_pods(tmp_path):
    """Operators' pods legitimately share the ``cascade-`` prefix — only the
    full cascade-<round>-<stage> scheme is reapable."""
    prov = FakeProvider("lium")
    prov.live["cascade-worker"] = PodAddress("10.9.9.1", 22)
    prov.live["cascade-final-b"] = PodAddress("10.9.9.2", 22)
    prov.live["cascade-heat-2"] = PodAddress("10.9.9.3", 22)            # no round id
    prov.live["cascade-n91-42-funded-king-0"] = PodAddress("10.9.9.5", 22)   # the trainer's
    prov.live["cascade-900-heat-zombie"] = PodAddress("10.9.9.4", 22)   # ours, orphaned
    loop = make_loop(tmp_path, providers={"lium": prov})
    cycle(loop)
    assert prov.terminated == ["cascade-900-heat-zombie"]


def _adopted(tmp_path, health):
    prov = FakeProvider("lium")
    _seed_ledger(tmp_path, prov, [("eval", "cascade-111-eval-0", "NVIDIA L40S", 1),
                                  ("final", "cascade-900-final-0", "NVIDIA L40S", 2)])
    loop = make_loop(tmp_path, providers={"lium": prov}, health=health,
                     store=FakeStore(MANIFEST), eval_hosts=tmp_path / "eval_hosts.toml")
    loop._last_evaled_round = "111"
    loop.sleep = lambda s: None          # retry spacing: no real waits in tests
    return loop, prov


def cycle_adopted(loop):
    loop.run_once()
    t = loop._adopt_thread
    if t is not None:
        t.join(timeout=30)


def test_adopted_pods_regate_against_the_ledgered_shape(tmp_path):
    seen = []

    def health(addr, stage, provider="", **shape):
        seen.append((stage, shape.get("sku"), shape.get("gpus"), shape.get("adopted")))
        return _report(ok=True)

    loop, prov = _adopted(tmp_path, health)
    cycle_adopted(loop)
    assert sorted(seen) == [("eval", "NVIDIA L40S", 1, True),
                            ("final", "NVIDIA L40S", 2, True)]
    assert prov.terminated == []


def test_adopted_pod_failing_regate_is_terminated_and_unpublished(tmp_path):
    calls = []

    def health(addr, stage, provider="", **shape):
        calls.append(stage)
        return _report(ok=(stage != "final"))

    loop, prov = _adopted(tmp_path, health)
    cycle_adopted(loop)
    assert prov.terminated == ["cascade-900-final-0"]
    assert [i.stage for i in load_state(tmp_path / "state.json").instances] == ["eval"]
    assert calls.count("final") == 3 and calls.count("eval") == 1   # bounded retry


def test_adopted_pod_survives_a_transient_gate_blip(tmp_path):
    calls = []

    def health(addr, stage, provider="", **shape):
        calls.append(stage)
        return _report(ok=calls.count(stage) > 1)             # blip, then fine

    loop, prov = _adopted(tmp_path, health)
    slept = []
    loop.sleep = slept.append
    cycle_adopted(loop)
    assert prov.terminated == []
    assert slept and all(s == 30.0 for s in slept)
    assert len(calls) == 4                                    # 2 pods × (fail+pass)


def test_adoption_regate_runs_once_and_skips_without_a_gate(tmp_path):
    prov = FakeProvider("lium")
    _seed_ledger(tmp_path, prov, [("final", "cascade-900-final-0", "NVIDIA L40S", 2)])
    loop = make_loop(tmp_path, providers={"lium": prov})
    cycle_adopted(loop)                                       # health=None: no-op
    assert loop._adopt_thread is None
    calls = []
    loop2 = make_loop(tmp_path, providers={"lium": prov},
                      health=lambda a, s, provider="", **k: (calls.append(s), _report())[1])
    cycle_adopted(loop2)
    cycle_adopted(loop2)
    assert calls == ["final"]                                 # once per adopted pod


def test_static_hosts_and_operator_lanes_survive_a_teardown_republish(tmp_path):
    clock = Clock()
    prov = FakeProvider("lium")
    _seed_ledger(tmp_path, prov, [("final", "cascade-900-final-0", "NVIDIA L40S", 2)],
                 clock=clock)
    (tmp_path / "hosts.toml").write_text(
        '[[host]]\nname = "op-lane-1"\nhost = "77.7.7.7"\nstage = "final"\n')
    static = '[[host]]\nname = "cascade-final-b"\nhost = "216.81.245.151"\nstage = "final"\n'
    loop = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    loop.static_hosts_text = static
    clock.t += 3 * 3600.0                                     # TTL reaps the leftover
    cycle(loop)
    names = {h.name for h in load_hosts(tmp_path / "hosts.toml")}
    assert names == {"cascade-final-b", "op-lane-1"}          # static + operator lane


def test_publish_uses_per_provider_profile(tmp_path):
    """Shadeform VMs land as the 'shadeform' user under /home/shadeform — the
    rendered eval entry must carry that provider's paths, not lium's root."""
    from dataclasses import replace as _replace

    from cascade.provision.loop import PodProfile

    prov = FakeProvider("shadeform")
    loop, _ = _eval_loop(tmp_path, providers={"shadeform": prov},
                         policy=_eval_policy(eval=StagePolicy(
                             sku="NVIDIA L40S", gpus_per_pod=1, max_pods=1,
                             providers=("shadeform",), max_price_hr=1.2)))
    loop.render = _replace(loop.render, profiles={"shadeform": PodProfile(
        user="shadeform", workdir="/home/shadeform/cascade",
        remote_python="/home/shadeform/cascade/.venv/bin/python")})
    cycle(loop)
    hosts = load_hosts(tmp_path / "eval_hosts.toml")
    assert hosts and all(h.user == "shadeform" for h in hosts)
    assert all(h.workdir == "/home/shadeform/cascade" for h in hosts)


# ── chain client guards ──────────────────────────────────────────────────────


def test_frozen_block_rebuilds_chain_client(tmp_path):
    """A quietly-dead websocket keeps answering with a stale block; after
    stale_block_after_s the loop rebuilds the client and sees the real height."""
    clock = Clock()
    loop = make_loop(tmp_path, clock=clock)
    loop.chain_client = FakeChain(700)
    loop.chain_client_factory = lambda: FakeChain(880)
    loop.stale_block_after_s = 300.0
    cycle(loop)
    clock.t += 200.0
    cycle(loop)
    assert loop._last_block == 700                            # frozen, not stale yet
    clock.t += 200.0
    cycle(loop)
    assert loop._last_block == 880                            # rebuilt


def test_raising_chain_client_rebuilds_once(tmp_path):
    class DeadChain:
        def current_block(self):
            raise ConnectionError("ws closed")

    loop = make_loop(tmp_path)
    loop.chain_client = DeadChain()
    loop.chain_client_factory = lambda: FakeChain(880)
    cycle(loop)
    assert loop._last_block == 880


def test_hung_chain_client_hits_deadline_and_rebuilds(tmp_path):
    """A websocket that HANGS (not raises) must not block the loop — the
    deadline converts the hang into a rebuild."""
    import threading

    class HangingChain:
        def current_block(self):
            threading.Event().wait(30)

    loop = make_loop(tmp_path)
    loop.chain_client = HangingChain()
    loop.chain_client_factory = lambda: FakeChain(880)
    orig = loop._with_deadline
    loop_cls = type(loop)
    try:
        loop_cls._with_deadline = staticmethod(
            lambda fn, seconds: orig(fn, 0.2 if seconds >= 60 else seconds))
        cycle(loop)
    finally:
        loop_cls._with_deadline = staticmethod(orig)
    assert loop._last_block == 880


# ── config ───────────────────────────────────────────────────────────────────


def _raw_config(**over):
    raw = {"provisioner": {"max_spend_per_round": 25.0}}
    raw["provisioner"].update(over)
    return raw


def test_build_policy_eval_table_is_optional():
    assert build_policy(_raw_config()).eval is None
    raw = _raw_config(eval={"sku": "NVIDIA L40S", "gpus_per_pod": 1, "max_pods": 1,
                            "providers": ["lium"], "max_price_hr": 1.2})
    p = build_policy(raw)
    assert p.eval.sku == "NVIDIA L40S" and p.eval.max_pods == 1 and p.ttl_epochs == 1


@pytest.mark.parametrize("bad", [
    {"max_spend_per_round": 0},
    {"ttl_epochs": 0},
    {"eval": {"sku": "", "gpus_per_pod": 1, "max_pods": 1,
              "providers": ["lium"], "max_price_hr": 1.2}},
    {"eval": {"sku": "L40S", "gpus_per_pod": 0, "max_pods": 1,
              "providers": ["lium"], "max_price_hr": 1.2}},
])
def test_build_policy_rejects_bad_config(bad):
    with pytest.raises(ProvisionError):
        build_policy(_raw_config(**bad))


def test_retired_fleet_tables_still_load_and_name_their_providers():
    """A deployed provision.toml still carries the retired per-round fleet
    (heat/final tables, trigger/escalation keys): it loads unchanged, and the
    providers it names stay wired so a ledger leftover can still be reaped."""
    raw = _raw_config(
        trigger_margin_blocks=25, escalate_deadline_s=1800, final_rent_on="heat_complete",
        heat={"sku": "NVIDIA RTX A6000", "gpus_per_pod": 8, "max_pods": 4,
              "providers": ["lium", "shadeform"], "max_price_hr": 4.0},
        final={"sku": "NVIDIA L40S", "gpus_per_pod": 2, "max_pods": 2,
               "providers": ["shadeform"], "max_price_hr": 3.0},
        eval={"sku": "NVIDIA L40S", "gpus_per_pod": 1, "max_pods": 1,
              "providers": ["lium"], "max_price_hr": 1.2})
    p = build_policy(raw)
    assert p.eval.providers == ("lium",)
    assert provider_names(raw, p) == ["lium", "shadeform"]


def test_eval_hosts_fan_out_to_the_stage_shape_when_the_row_lacks_gpus(tmp_path):
    """A ledger row with no recorded ``gpus`` publishes the stage policy's
    ``gpus_per_pod`` lanes, never a silent single lane."""
    policy = _eval_policy(eval=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=1,
                                           providers=("lium",), max_price_hr=1.2))
    loop, _prov = _eval_loop(tmp_path, policy=policy)
    loop._last_evaled_round = "900"
    inst = PodInstance(provider="lium", instance_id="cascade-900-eval-0", stage="eval",
                       rented_at_iso="2026-01-01T00:00:00+00:00", sku="NVIDIA L40S", gpus=0)
    loop._publish_eval_hosts([(inst, PodAddress("10.0.1.1", 22))])
    text = (tmp_path / "eval_hosts.toml").read_text()
    assert text.count("[[host]]") == 2
    assert loop._stage_gpus("final") == 1          # no policy for the retired fleet
