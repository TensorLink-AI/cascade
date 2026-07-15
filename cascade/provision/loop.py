"""The cascade-provisioner service loop — rent per round, tear down per stage.

One cycle of the machine (each ``poll_seconds``, ~30s):

    WAIT   poll the chain block until :func:`policy.should_trigger` fires
           (inside the last ``trigger_margin_blocks`` of the epoch, once per
           round — that is when timed reveals have landed and the field is
           countable);
    COUNT  ask the trainer for the round plan (``plan_fn``; real impl runs
           ``cascade-trainer --plan-only`` and parses its JSON line);
    SIZE   :func:`policy.size_fleet` — slot-based heat fleet off the eligible
           field, one multi-GPU final pod for king + finalists;
    RENT   providers in each stage's priority order; an adapter failure means
           fewer offers, never a dead loop; every pod is named/tagged
           ``cascade-{round_id}-…`` so reconcile can find strays;
    BOOT   provider-ready → SSH reachable → the seven-check health gate; a
           failed pod is terminated and replaced ONCE, then dropped;
    PUBLISH atomically write hosts.toml (heat + final entries) — the trainer
           picks it up at round start (``--hosts-wait-seconds`` covers boot);
    WATCH  the two teardown signals: the trainer's ``heat_complete.json``
           marker under the shared work-root, and the round manifest in the
           store;
    TEARDOWN per stage — heat pods die on the marker while the final still
           runs (hosts.toml is re-rendered final-only); final pods die on the
           manifest; the TTL (one epoch from rent) is the hard backstop; and
    RECONCILE every cycle: live pods tagged ``cascade-`` that the ledger does
           not own are terminated (a crash between launch and ledger-save must
           not leak a billing box).

The optional EVAL stage rides the opposite phase of the same machine: renting
is triggered by a NEW round manifest appearing in the store (that is when the
validator's heavy evals — GIFT-Eval gate, cascade bench — start needing GPU),
the pod is published to a SEPARATE ``eval_hosts_path`` file that the validator
re-reads lazily per eval (never the trainer's ``hosts_path``), and teardown
comes from the round's receipt publishing under ``receipt_prefix`` (the
validator is done), a newer manifest superseding the round, or the same TTL
backstop. ``last_evaled_round`` persists in the ledger so restarts never
double-rent. See ``_maybe_provision_eval``.

Round-id note: the provisioner keys rounds by the upcoming BOUNDARY BLOCK
(knowable before the round starts), while the trainer keys them by base_seed
(the boundary block's hash — knowable only after). The two are reconciled at
the marker: any ``heat_complete.json`` under the work-root NEWER than our rent
time is this round's (only one round runs at a time), and its directory name
IS the base_seed, which then lets us poll ``manifests/round-<base_seed>.json``
directly. Until a marker appears we also watch ``latest.json`` for a round_id
change since rent — either signal ends the round.

Every boundary is injected (chain client, plan_fn, providers, manifest store,
health check, ssh probe, clock, sleep), so the whole cycle is unit-tested with
fakes; the loop itself is thin glue over ``policy``/``state``/``health``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from ..shared.hippius import MANIFEST_LATEST_KEY, manifest_round_key
from .core import (
    DEFAULT_FORWARD_ENV,
    DEFAULT_READY_TIMEOUT,
    DEFAULT_REMOTE_PYTHON,
    DEFAULT_SSH_PORT,
    DEFAULT_WORKDIR,
    LaunchSpec,
    PodAddress,
    ProvisionError,
    render_hosts_toml,
)
from .health import HealthReport
from .hostsfile import clear_hosts, write_hosts
from .policy import (
    ProvisionPolicy,
    pods_for_slots,
    should_trigger,
    size_fleet,
    teardown_due,
)
from .state import (
    PodInstance,
    RoundState,
    add_instance,
    drop_instances,
    instances_for_stage,
    load_state,
    owned_ids,
    reconcile,
    save_state,
)

log = logging.getLogger("cascade.provision.loop")

POD_TAG = "cascade-"                       # every rented pod's name starts with this

# The FULL naming scheme of pods this service rents: cascade-<round_id>-<stage>
# (+ optional -rN replacement / -gN lane suffixes). The orphan reaper matches on
# this — never on the bare POD_TAG prefix, which operators' hand-rented pods
# (cascade-worker, cascade-final-b, cascade-heat-2) legitimately share. Reaping
# by bare prefix terminated a live hand-rented final pod on 2026-07-13.
_PROVISIONER_POD_RE = re.compile(r"^cascade-\d+-(heat|final|eval)(-|$)")


def is_provisioner_pod_name(name: str) -> bool:
    """True only for pod names this service itself creates (see _PROVISIONER_POD_RE)."""
    return _PROVISIONER_POD_RE.match(str(name)) is not None


@dataclass(frozen=True)
class PodProfile:
    """Per-provider pod paths: providers boot different base users/homes
    (lium pods are root; shadeform VMs land as the ``shadeform`` user)."""

    user: str = "root"
    workdir: str = DEFAULT_WORKDIR
    remote_python: str = DEFAULT_REMOTE_PYTHON


@dataclass(frozen=True)
class RenderSettings:
    """Everything hosts.toml rendering + pod launching needs beyond addresses."""

    image: str                              # digest-pinned worker image
    ssh_pubkey: str                         # injected into pods as $SSH_PUBKEY
    key_path: str                           # orchestrator private key, into hosts.toml
    forward_env: tuple[str, ...] = DEFAULT_FORWARD_ENV
    remote_python: str = DEFAULT_REMOTE_PYTHON
    workdir: str = DEFAULT_WORKDIR
    chain_toml: str | None = None
    ssh_port: int = DEFAULT_SSH_PORT
    # provider name → pod paths override; absent providers use the defaults above.
    profiles: dict[str, PodProfile] = field(default_factory=dict)

    def profile_for(self, provider: str) -> PodProfile:
        return self.profiles.get(
            provider, PodProfile(workdir=self.workdir, remote_python=self.remote_python))


@dataclass
class ProvisionerLoop:
    """One provisioner service instance. All I/O boundaries are injected fields.

    ``providers`` maps name → adapter (the core ``Provider`` verbs, optionally
    ``list_tagged``/``offer_price``); ``plan_fn`` returns the trainer's
    ``--plan-only`` payload dict; ``manifest_store`` duck-types
    ``S3Store.get_text``; ``health_check(addr, stage)`` returns a
    :class:`HealthReport` (``None`` ⇒ gate skipped, e.g. tests/dry-run);
    ``ssh_probe(ip, port)`` is the TCP reachability wait. ``clock`` is seconds
    (epoch) and everything time-based flows through it.
    """

    policy: ProvisionPolicy
    providers: dict[str, object]
    chain_client: object                                    # .current_block() -> int
    plan_fn: Callable[[], dict]
    render: RenderSettings
    hosts_path: Path
    work_root: Path
    state_path: Path
    epoch_blocks: int
    final_hours: float                                      # [training] target_train_hours
    manifest_store: object | None = None
    # The VALIDATOR's eval-offload hosts file (never the trainer's hosts_path):
    # the eval pod is published here on rent and the file is cleared on
    # teardown — safe mid-round because the validator re-resolves it lazily at
    # each eval. None (the default) disables the eval stage even when a
    # [provisioner.eval] policy exists: there is nowhere to publish.
    eval_hosts_path: Path | None = None
    # Where this deployment's validator publishes round receipts — the eval
    # pod's primary teardown signal. Per-validator paths ("receipts/<hotkey>/",
    # see cascade.shared.hippius.receipt_round_key); the watched key is
    # ``<prefix>round-<manifest_round_id>.json``. Empty ⇒ receipts are never
    # seen and the newer-manifest / TTL signals bound the pod's life instead.
    receipt_prefix: str = ""
    health_check: Callable[[PodAddress, str], HealthReport] | None = None
    # Provisions a bare pod over SSH (rsync source + uv sync) before the health
    # gate — the testnet path while no digest-pinned worker image is published.
    # Returns False on failure (the pod is treated like a health-gate dud).
    bootstrap: Callable[[PodAddress, str], bool] | None = None
    # Raw [[host]] TOML appended verbatim to EVERY hosts.toml publish — the
    # operator's static pods (e.g. a long-lived final pod) that the provisioner
    # must never drop. clear/teardown re-renders keep it too: "no dynamic pods"
    # must degrade to "static fleet", never to an empty file while a static
    # final exists.
    static_hosts_text: str = ""
    # Called at the top of EVERY cycle (never raises consequences — errors are
    # suppressed): the logging self-heal hook. bittensor's logging init strips
    # handlers and raises the level (to CRITICAL) on NAMED loggers when a chain
    # client connects — including reconnects — so a one-time logging setup
    # cannot survive; only a per-cycle re-assert can.
    on_cycle: Callable[[], None] | None = None
    ssh_probe: Callable[[str, int], bool] = field(default=lambda ip, port: True)
    # Rebuilds chain_client when the block number freezes: a bittensor
    # websocket can die QUIETLY and keep answering current_block() with a
    # stale value — the loop then cycles forever without ever seeing the
    # trigger window (observed live 2026-07-14: 2h19m of silent no-trigger).
    # None disables the refresh (tests with static FakeChain blocks).
    chain_client_factory: Callable[[], object] | None = None
    stale_block_after_s: float = 300.0
    heartbeat_every_s: float = 600.0
    ready_timeout: float = DEFAULT_READY_TIMEOUT
    poll_seconds: float = 30.0
    dry_run: bool = False
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep

    # internal (rebuilt from the ledger on restart)
    _state: RoundState | None = field(default=None, init=False, repr=False)
    _provisioned_round: int | None = field(default=None, init=False, repr=False)
    _addrs: dict[str, PodAddress] = field(default_factory=dict, init=False, repr=False)
    _manifest_baseline: str | None = field(default=None, init=False, repr=False)
    _last_block: int | None = field(default=None, init=False, repr=False)
    _block_changed_at: float = field(default=0.0, init=False, repr=False)
    _last_heartbeat_at: float = field(default=0.0, init=False, repr=False)
    _eval_inflight: bool = field(default=False, init=False, repr=False)
    _eval_thread: object = field(default=None, init=False, repr=False)
    _state_lock: object = field(default_factory=__import__("threading").RLock,
                                init=False, repr=False)
    _learned_round_id: str | None = field(default=None, init=False, repr=False)
    # The eval stage's rent-once latch (mirrors _provisioned_round for the
    # boundary stages): the manifest round an eval pod was last rented — or
    # deliberately skipped — for. Restored from the ledger so restarts never
    # double-rent; in dry-run it lives in memory only (no disk mutation).
    _last_evaled_round: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._state = load_state(self.state_path)
        if self._state is not None:
            log.info("resumed ledger %s: round=%s, %d instance(s)",
                     self.state_path, self._state.round_id, len(self._state.instances))
            if self._state.round_id.isdigit():
                self._provisioned_round = int(self._state.round_id)
            self._last_evaled_round = self._state.last_evaled_round

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def epoch_hours(self) -> float:
        return self.epoch_blocks * 12.0 / 3600.0            # 12s block time

    @property
    def ttl_hours(self) -> float:
        return self.policy.ttl_epochs * self.epoch_hours

    # ── the cycle ────────────────────────────────────────────────────────────

    def run_once(self) -> None:
        """One poll tick: reconcile strays, tear down what's due, maybe rent.

        Teardown runs before the eval check on purpose: a newer manifest
        first kills the previous round's eval pod, then (same tick) the new
        round rents its own — the two never coexist.
        """
        if self.on_cycle is not None:
            with contextlib.suppress(Exception):
                self.on_cycle()
        now = self.clock()
        if now - self._last_heartbeat_at >= self.heartbeat_every_s:
            # Cycle-START heartbeat: every phase below makes network calls
            # that can crawl on a bad night — liveness must never depend on
            # reaching any of them (2026-07-14, twice: starved heartbeats
            # masked a wedged loop through two rental windows).
            log.info("heartbeat: cycle start, last_block=%s, owned_pods=%d",
                     self._last_block if self._last_block is not None else "?",
                     len(self._state.instances) if self._state else 0)
            self._last_heartbeat_at = now
        self._reconcile_orphans()
        self._teardown_due_pods()
        block = self._current_block()
        self._maybe_provision_eval()
        if should_trigger(block, self.epoch_blocks,
                          self.policy.trigger_margin_blocks, self._provisioned_round):
            round_id = (block // self.epoch_blocks + 1) * self.epoch_blocks
            self._provision_round(round_id)

    @staticmethod
    def _with_deadline(fn, seconds: float):
        """Run ``fn()`` with a HARD deadline in a helper thread.

        bittensor's websocket calls can hang indefinitely (no client-side
        timeout) — four rental windows were lost to silent chain-source stalls.
        A timed-out call leaks its helper thread (daemon; it dies with the
        process), which is an acceptable cost for a loop that must never
        block. Raises ``TimeoutError`` on deadline."""
        import concurrent.futures

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(fn).result(timeout=seconds)
        finally:
            ex.shutdown(wait=False)

    def _current_block(self) -> int:
        """The chain height, with staleness detection and client rebuild.

        A frozen block number for ``stale_block_after_s`` (or a raising
        client) triggers ONE rebuild via ``chain_client_factory`` per
        occurrence; the fresh client's answer is trusted (its own staleness
        clock restarts). Every path also emits a heartbeat log every
        ``heartbeat_every_s`` so a silent stall is visible in the log, not
        just in a missing trigger.
        """
        now = self.clock()
        try:
            block = int(self._with_deadline(self.chain_client.current_block, 60.0))
        except Exception as e:  # noqa: BLE001 — a dead/hung client is rebuildable
            if self.chain_client_factory is None:
                raise
            log.warning("current_block failed/hung (%s); rebuilding chain client",
                        type(e).__name__)
            self.chain_client = self._with_deadline(self.chain_client_factory, 120.0)
            block = int(self._with_deadline(self.chain_client.current_block, 60.0))
            self._block_changed_at = now
        if self._last_block is None or block != self._last_block:
            self._last_block = block
            self._block_changed_at = now
        elif (self.chain_client_factory is not None
              and now - self._block_changed_at > self.stale_block_after_s):
            log.warning("block frozen at %d for %.0fs — rebuilding chain client "
                        "(quietly dead websocket?)", block, now - self._block_changed_at)
            self.chain_client = self._with_deadline(self.chain_client_factory, 120.0)
            block = int(self._with_deadline(self.chain_client.current_block, 60.0))
            self._last_block = block
            self._block_changed_at = now
        return block

    def run_forever(self) -> None:  # pragma: no cover — glue over run_once
        """Poll forever; one bad cycle must never kill the service (pods that
        were rented still need their teardown ticks)."""
        while True:
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001
                log.exception("provisioner cycle failed (retrying): %s", e)
            self.sleep(self.poll_seconds)

    # ── COUNT → SIZE → RENT → PUBLISH ────────────────────────────────────────

    def _provision_round(self, round_id: int) -> None:
        try:
            payload = self.plan_fn()
        except Exception as e:  # noqa: BLE001 — plan failure retries next tick
            log.warning("plan_fn failed (will retry next poll): %s", e)
            return
        # Rent-once latch: set as soon as we HAVE a plan, so failures below
        # (no capacity, over budget) don't hammer providers every 30s. The
        # trainer's local fallback covers the round either way.
        self._provisioned_round = round_id

        fleet = size_fleet(
            int(payload["eligible_challengers"]),
            int(payload["finalists"]),
            float(payload["heat_train_hours"]),
            self.epoch_hours,
            self.final_hours,
            self.policy,
        )
        log.info("round %d plan: eligible=%s → heat %d pod(s)/%d slot(s), final %d pod(s)/%d slot(s)",
                 round_id, payload["eligible_challengers"],
                 fleet.heat.pods, fleet.heat.slots, fleet.final.pods, fleet.final.slots)

        # Per stage: the first (SKU candidate × provider) combination with
        # capacity for the WHOLE stage fleet wins — a stage never mixes SKUs
        # (within-round fairness by construction; see StagePolicy.candidates).
        chosen: dict[str, tuple[object, float, object, int]] = {}
        for stage, sp, fl in (("heat", self.policy.heat, fleet.heat),
                              ("final", self.policy.final, fleet.final)):
            if fl.pods <= 0:
                continue
            picked = self._pick_offer(sp, fl.slots, stage)
            if picked is not None:
                chosen[stage] = picked
        if not chosen:
            # No provider anywhere: publish an EMPTY hosts file so the trainer
            # trains this round locally — the round is degraded, never lost.
            log.error("round %d: no provider has capacity for any stage; "
                      "publishing static fleet only (trainer degrades, never lost)", round_id)
            self._write_hosts([])
            return

        projected = sum(pods * price * self.ttl_hours
                        for (_prov, price, _cand, pods) in chosen.values())
        offers = {stage: price for stage, (_p, price, _c, _n) in chosen.items()}
        ok = projected <= self.policy.max_spend_per_round
        if not ok:
            log.error("round %d REFUSED by budget breaker: worst-case $%.2f > cap $%.2f "
                      "(offers %s); clearing hosts", round_id, projected,
                      self.policy.max_spend_per_round, offers)
            self._write_hosts([])
            return
        log.info("round %d budget ok: worst-case $%.2f <= cap $%.2f",
                 round_id, projected, self.policy.max_spend_per_round)

        if self.dry_run:
            for stage, (prov, price, cand, pods) in chosen.items():
                log.info("[dry-run] round %d %s: WOULD rent %d × %d-GPU %s pod(s) on %s "
                         "@ $%.2f/hr (tag cascade-%d-%s)", round_id, stage, pods,
                         cand.gpus_per_pod, cand.sku, prov.name,
                         price, round_id, stage)
            return

        # Baseline the manifest pointer BEFORE renting: any later change means
        # a round published after our rent — the teardown signal that needs no
        # base_seed knowledge.
        self._manifest_baseline = self._latest_round_id()
        self._learned_round_id = None
        self._state = (RoundState(round_id=str(round_id)) if self._state is None
                       else replace(self._state, round_id=str(round_id), published=False))
        self._save()

        rented: dict[str, list[tuple[PodInstance, PodAddress]]] = {}
        for stage, (prov, _price, cand, pods) in chosen.items():
            healthy = self._rent_stage(round_id, stage, prov, cand, pods)
            if healthy:
                rented[stage] = healthy
        if not rented:
            log.error("round %d: every rented pod failed its health gate; "
                      "publishing static fleet only (trainer degrades, never lost)", round_id)
            self._write_hosts([])
            return
        self._publish_hosts(rented)
        self._state = replace(self._state, published=True)
        self._save()

    def _pick_offer(self, sp, slots: int,
                    stage: str) -> tuple[object, float, object, int] | None:
        """First (SKU candidate × provider) with capacity for the whole fleet.

        Candidates in the stage's configured order, providers in priority order
        within each candidate. The pod count is re-derived per candidate — an
        8× fallback needs fewer pods for the same slots than a 4× primary. Any
        adapter fault just skips that provider: a broken adapter means fewer
        offers, never a dead provisioner. Returns ``(provider, price,
        candidate, pods)`` or ``None``.
        """
        for cand in sp.sku_candidates:
            pods = pods_for_slots(slots, cand.gpus_per_pod, sp.max_pods)
            if pods <= 0:
                continue
            for name in sp.providers:
                prov = self.providers.get(name)
                if prov is None:
                    log.warning("provider %r not configured; skipping", name)
                    continue
                try:
                    if not prov.available(cand.marketplace_sku, pods,
                                          gpus=cand.gpus_per_pod):
                        log.info("provider %s: no capacity for %d × %dx%s",
                                 name, pods, cand.gpus_per_pod, cand.sku)
                        continue
                except Exception as e:  # noqa: BLE001
                    log.warning("provider %s availability probe failed (%s); skipping",
                                name, e)
                    continue
                price = self._offer_price(prov, cand.marketplace_sku)
                if price is None:
                    # Unknown price ⇒ assume the candidate cap: the budget
                    # breaker then projects at the worst price we accepted.
                    price = cand.max_price_hr
                if price > cand.max_price_hr:
                    log.warning("provider %s: %s at $%.2f/hr exceeds cap $%.2f/hr; skipping",
                                name, cand.sku, price, cand.max_price_hr)
                    continue
                if cand is not sp.sku_candidates[0]:
                    log.info("%s stage falling back to %dx%s on %s (primary had no offer)",
                             stage, cand.gpus_per_pod, cand.sku, name)
                return prov, float(price), cand, pods
        return None

    @staticmethod
    def _offer_price(prov: object, sku: str) -> float | None:
        fn = getattr(prov, "offer_price", None)
        if fn is None:
            return None
        try:
            return fn(sku)
        except Exception as e:  # noqa: BLE001 — pricing is advisory, capacity was probed
            log.warning("provider %s offer_price failed (%s); assuming stage cap",
                        getattr(prov, "name", prov), e)
            return None

    # ── RENT + BOOT + HEALTH (with one replacement per failed pod) ───────────

    def _rent_stage(self, round_id: int | str, stage: str, prov: object,
                    cand, pods: int) -> list[tuple[PodInstance, PodAddress]]:
        # round_id is the boundary block for heat/final and the manifest round
        # id for eval — both digits, both satisfying _PROVISIONER_POD_RE.
        spec = LaunchSpec(
            sku=cand.marketplace_sku, count=pods, image=self.render.image,
            ssh_pubkey=self.render.ssh_pubkey, ssh_port=self.render.ssh_port,
            name_prefix=f"{POD_TAG}{round_id}-{stage}", gpus_per_pod=cand.gpus_per_pod,
        )
        try:
            pod_ids = prov.launch(spec)
        except Exception as e:  # noqa: BLE001 — a failed stage is a smaller fleet
            log.error("round %s %s: launch on %s failed: %s", round_id, stage, prov.name, e)
            return []
        # Write-ahead: the ledger owns these ids BEFORE we do anything else
        # with them — a crash from here on still tears them down on restart.
        for pid in pod_ids:
            self._state = add_instance(self._state, self._instance(prov, pid, stage, cand))
        self._save()

        healthy: list[tuple[PodInstance, PodAddress]] = []
        for i, pid in enumerate(pod_ids):
            addr = self._boot_and_gate(prov, pid, stage, cand)
            if addr is not None:
                healthy.append((self._find_instance(pid), addr))
                continue
            # Terminate the dud and try ONE replacement — marketplace pods are
            # lemon-prone, but retrying forever would chase a bad batch all day.
            log.warning("round %s %s: pod %s failed boot/health; replacing once",
                        round_id, stage, pid)
            # Exclude the lemon's machine from the replacement pick — offer
            # listings are deterministic, so the same spec re-rents the same
            # failed executor (observed: eval pod + replacement both dead on
            # 63243c2c…, round 5052267627071284702).
            lemon = getattr(prov, "machine_of", lambda _p: None)(pid)
            self._terminate_and_drop(prov, pid)
            rspec = replace(spec, count=1, name_prefix=f"{POD_TAG}{round_id}-{stage}-r{i}",
                            exclude_ids=spec.exclude_ids + ((lemon,) if lemon else ()))
            try:
                rid = prov.launch(rspec)[0]
            except Exception as e:  # noqa: BLE001
                log.error("round %s %s: replacement launch failed: %s", round_id, stage, e)
                continue
            self._state = add_instance(self._state, self._instance(prov, rid, stage, cand))
            self._save()
            raddr = self._boot_and_gate(prov, rid, stage, cand)
            if raddr is not None:
                healthy.append((self._find_instance(rid), raddr))
            else:
                log.error("round %s %s: replacement %s also failed; dropping the slot",
                          round_id, stage, rid)
                self._terminate_and_drop(prov, rid)
        return healthy

    def _boot_and_gate(self, prov: object, pid: str, stage: str,
                       cand=None) -> PodAddress | None:
        try:
            if not prov.wait_ready(pid, timeout=self.ready_timeout):
                log.warning("pod %s not provider-ready within %.0fs", pid, self.ready_timeout)
                return None
            addr = prov.get_ip(pid)
            if addr is None:
                log.warning("pod %s exposed no IP", pid)
                return None
            if not self.ssh_probe(addr.ip, addr.ssh_port):
                log.warning("pod %s SSH %s:%d unreachable", pid, addr.ip, addr.ssh_port)
                return None
            if self.bootstrap is not None:  # noqa: SIM102 — readability: distinct guard + gate
                # No digest-pinned worker image exists yet (testnet): pods are
                # rented bare and provisioned over SSH (rsync source + uv sync
                # against the pinned lock). The hook runs BEFORE the health
                # gate, so the gate verifies what bootstrap actually produced.
                if not self.bootstrap(addr, stage, prov.name):
                    log.warning("pod %s bootstrap failed", pid)
                    return None
            if self.health_check is not None:
                report = (self.health_check(addr, stage, prov.name,
                                            sku=cand.sku, gpus=cand.gpus_per_pod)
                          if cand is not None else
                          self.health_check(addr, stage, prov.name))
                if not report.ok:
                    log.warning("pod %s failed health gate: %s", pid, report.summary())
                    return None
        except Exception as e:  # noqa: BLE001 — any boot fault is a failed pod, not a dead loop
            log.warning("pod %s boot/health errored: %s", pid, e)
            return None
        self._addrs[pid] = addr
        return addr

    # ── PUBLISH ──────────────────────────────────────────────────────────────

    def _write_hosts(self, sections: list[str]) -> None:
        """Publish dynamic sections + the operator's static entries.

        The static text rides along on EVERY publish — including the
        heat-teardown re-render and the nothing-rented paths — so a long-lived
        hand-rented pod (e.g. the static final) is never dropped by provisioner
        activity. Only with no static text AND no dynamic pods does the file
        clear (the trainer's local-fallback signal).
        """
        content = "".join([self.static_hosts_text, *sections])
        if content.strip():
            write_hosts(self.hosts_path, content)
        else:
            clear_hosts(self.hosts_path)

    def _publish_hosts(self, by_stage: dict[str, list[tuple[PodInstance, PodAddress]]]) -> None:
        sections = []
        for stage in ("heat", "final"):                     # stable order in the file
            entries = by_stage.get(stage)
            if not entries:
                continue
            fleet_gpus = entries[0][0].gpus or _gpus_for(self.policy, stage)
            prof = self.render.profile_for(entries[0][0].provider)
            sections.append(render_hosts_toml(
                [addr for _inst, addr in entries],
                key_path=self.render.key_path,
                forward_env=self.render.forward_env,
                remote_python=prof.remote_python,
                workdir=prof.workdir,
                user=prof.user,
                chain_toml=self.render.chain_toml,
                name_prefix=f"{POD_TAG}{self._state.round_id}-{stage}",
                provider=entries[0][0].provider,
                stage=stage,
                gpus_per_pod=fleet_gpus,
            ))
        self._write_hosts(sections)
        n = sum(len(v) for v in by_stage.values())
        log.info("published %s: %d dynamic pod(s) across %s%s", self.hosts_path, n,
                 sorted(by_stage), " + static entries" if self.static_hosts_text else "")

    def _republish_from_ledger(self) -> None:
        """Re-render hosts.toml from the surviving instances (post-teardown).

        The re-render path after the heat marker: heat entries disappear,
        final entries stay, and the trainer's next ``_hosts_for("final")``
        sees only live pods. With nothing left, clear — an empty file is the
        trainer's local-fallback signal and never lists a dead box.
        """
        by_stage: dict[str, list[tuple[PodInstance, PodAddress]]] = {}
        for stage in ("heat", "final"):
            for inst in instances_for_stage(self._state, stage):
                addr = self._addr_for(inst)
                if addr is not None:
                    by_stage.setdefault(stage, []).append((inst, addr))
        self._publish_hosts(by_stage)

    def _addr_for(self, inst: PodInstance) -> PodAddress | None:
        addr = self._addrs.get(inst.instance_id)
        if addr is not None:
            return addr
        prov = self.providers.get(inst.provider)
        if prov is None:
            return None
        try:
            addr = prov.get_ip(inst.instance_id)
        except Exception:  # noqa: BLE001 — a stale pod re-render is best-effort
            return None
        if addr is not None:
            self._addrs[inst.instance_id] = addr
        return addr

    # ── EVAL POD (manifest-triggered elastic stage) ──────────────────────────

    @property
    def _eval_enabled(self) -> bool:
        """All three legs the stage needs: a policy, a place to publish, and a
        store to watch. Any missing ⇒ the stage does not exist (backward
        compatible with every pre-eval config)."""
        return (self.policy.eval is not None and self.policy.eval.max_pods > 0
                and self.eval_hosts_path is not None and self.manifest_store is not None)

    def _maybe_provision_eval(self) -> None:
        """Rent the round's eval pod when a NEW manifest appears in the store.

        The eval stage is manifest-triggered, not boundary-triggered: the
        validator only needs GPU once a round has PUBLISHED (the GIFT-Eval
        gate and cascade bench score the manifest's checkpoints), which is
        exactly when the trainer fleet is being torn down. ``latest.json``
        moving to a round we have not yet served is the rent signal.

        The ``_last_evaled_round`` latch is set the moment we act — including
        on no-capacity and dry-run — so a failure degrades the round to local
        validator evals instead of hammering providers every poll (same
        discipline as the boundary stages' ``_provisioned_round``). It is
        persisted (write-ahead, before launch) so a crash mid-rent never
        double-rents on restart. A round whose receipt ALREADY exists is
        latched without renting: a provisioner restarted between rounds must
        not buy a pod just to watch the teardown signal kill it.
        """
        if not self._eval_enabled:
            return
        latest = self._latest_round_id()
        if latest is None or latest == self._last_evaled_round:
            return
        if self._eval_inflight:
            return                                # a rent worker is already busy
        if self._state is not None and instances_for_stage(self._state, "eval"):
            # Still holding an eval pod (e.g. dry-run teardown is a no-op):
            # the teardown sweep settles it first; re-check next cycle.
            return
        self._last_evaled_round = latest
        if self._eval_receipt_seen(latest):
            log.info("round %s already has a receipt; skipping its eval pod", latest)
            self._persist_eval_latch(latest)
            return
        picked = self._pick_offer(self.policy.eval, 1, "eval")
        if picked is None:
            log.error("round %s eval: no provider has capacity; the validator "
                      "runs this round's evals locally (degraded, never lost)", latest)
            self._persist_eval_latch(latest)
            return
        prov, price, cand, pods = picked
        if self.dry_run:
            log.info("[dry-run] round %s eval: WOULD rent %d × %d-GPU %s pod(s) on %s "
                     "@ $%.2f/hr (tag cascade-%s-eval)", latest, pods,
                     cand.gpus_per_pod, cand.sku, prov.name, price, latest)
            return
        self._persist_eval_latch(latest)
        # Rent + boot + health can take 15+ minutes (provider readiness alone
        # is a 900s wait) and MUST NOT block the loop: on 2026-07-14 an eval
        # pod's boot wait swallowed the round-5 heat-trigger window whole.
        # The slow leg runs in a daemon worker; the loop keeps cycling (the
        # ledger write-ahead in _rent_stage keeps a crash reconcilable, and
        # _eval_inflight stops a second manifest from double-renting).
        import threading

        def _rent_eval() -> None:
            try:
                healthy = self._rent_stage(latest, "eval", prov, cand, pods)
                if not healthy:
                    log.error("round %s eval: every rented pod failed its health "
                              "gate; the validator runs this round's evals locally",
                              latest)
                    return
                self._publish_eval_hosts(healthy)
            except Exception as e:  # noqa: BLE001 — a failed eval never kills the loop
                log.exception("round %s eval provisioning failed: %s", latest, e)
            finally:
                self._eval_inflight = False

        self._eval_inflight = True
        t = threading.Thread(target=_rent_eval, name=f"eval-rent-{latest}",
                             daemon=True)
        self._eval_thread = t                    # tests join() for determinism
        t.start()

    def _persist_eval_latch(self, round_id: str) -> None:
        """Write the eval latch through to the ledger (never in dry-run —
        dry-run mutates nothing on disk; the in-memory latch suffices)."""
        if self.dry_run:
            return
        self._state = (RoundState(round_id="") if self._state is None else self._state)
        self._state = replace(self._state, last_evaled_round=round_id)
        self._save()

    def _publish_eval_hosts(self, entries: list[tuple[PodInstance, PodAddress]]) -> None:
        """Publish the eval pod to the VALIDATOR's hosts file.

        A separate file from the trainer's ``hosts_path`` by design — the two
        consumers have different lifecycles and clobbering the trainer's fleet
        with an eval pod (or vice versa) must be structurally impossible.
        Rendered ``stage="any"`` so the validator's final/any filter matches,
        and written atomically because the validator may re-read it at any
        moment (it re-resolves lazily per eval).
        """
        inst = entries[0][0]
        prof = self.render.profile_for(inst.provider)
        text = render_hosts_toml(
            [addr for _inst, addr in entries],
            key_path=self.render.key_path,
            forward_env=self.render.forward_env,
            remote_python=prof.remote_python,
            workdir=prof.workdir,
            user=prof.user,
            chain_toml=self.render.chain_toml,
            name_prefix=f"{POD_TAG}{self._last_evaled_round}-eval",
            provider=inst.provider,
            stage="any",
            gpus_per_pod=inst.gpus or 1,
        )
        write_hosts(self.eval_hosts_path, text)
        log.info("published %s: %d eval pod(s) for round %s",
                 self.eval_hosts_path, len(entries), self._last_evaled_round)

    def _clear_eval_hosts(self) -> None:
        """Empty the eval hosts file after teardown. Safe at any moment: the
        validator re-resolves per eval, so an empty file just means its NEXT
        eval runs locally — never a dispatch to a dead box."""
        if self.eval_hosts_path is None:
            return
        clear_hosts(self.eval_hosts_path)
        log.info("cleared %s (eval pod gone; validator evals run locally)",
                 self.eval_hosts_path)

    def _eval_receipt_seen(self, round_id: str) -> bool:
        """Whether ``round_id``'s receipt is published — the eval pod's job is
        done. Receipts live at per-validator keys (``receipts/<hotkey>/
        round-<id>.json``, see ``cascade.validator.loop`` / ``publish_receipt``),
        so the watched prefix is operator config. A missing key or a down
        store both read as 'not yet' — the TTL backstops a store outage."""
        if self.manifest_store is None or not self.receipt_prefix or not round_id:
            return False
        key = f"{self.receipt_prefix.rstrip('/')}/round-{round_id}.json"
        try:
            self.manifest_store.get_text(key)
            return True
        except Exception:  # noqa: BLE001 — not published yet (or store down: TTL backstops)
            return False

    # ── WATCH + TEARDOWN ─────────────────────────────────────────────────────

    def _teardown_due_pods(self) -> None:
        if self._state is None or not self._state.instances:
            return
        now = self.clock()
        marker = self._heat_marker_seen()
        manifest = self._manifest_seen()
        # The eval pod's two signals are only worth store round-trips while an
        # eval pod is actually owned. ``newer_manifest`` compares the CURRENT
        # latest pointer against the round the pod was rented for.
        receipt = newer = False
        if any(i.stage == "eval" for i in self._state.instances):
            receipt = self._eval_receipt_seen(self._last_evaled_round)
            latest = self._latest_round_id()
            newer = (bool(self._last_evaled_round) and latest is not None
                     and latest != self._last_evaled_round)
        dead: set[str] = set()
        for inst in self._state.instances:
            if teardown_due(inst.stage, heat_marker_seen=marker, manifest_seen=manifest,
                            receipt_seen=receipt, newer_manifest=newer,
                            rented_at=_iso_ts(inst.rented_at_iso), now=now,
                            ttl_hours=self.ttl_hours):
                log.info("tearing down %s pod %s (marker=%s manifest=%s receipt=%s "
                         "newer=%s ttl=%.1fh)", inst.stage, inst.instance_id, marker,
                         manifest, receipt, newer, self.ttl_hours)
                prov = self.providers.get(inst.provider)
                if prov is None:
                    log.error("no adapter for provider %r — pod %s may be LEAKED",
                              inst.provider, inst.instance_id)
                    continue
                if self.dry_run:
                    log.info("[dry-run] WOULD terminate %s pod %s", inst.stage,
                             inst.instance_id)
                    continue
                try:
                    prov.terminate(inst.instance_id)
                except Exception as e:  # noqa: BLE001 — keep tearing down the rest
                    log.error("terminate %s failed (may be leaked!): %s", inst.instance_id, e)
                    continue
                dead.add(inst.instance_id)
                self._addrs.pop(inst.instance_id, None)
        if dead:
            dead_eval = {i.instance_id for i in self._state.instances
                         if i.instance_id in dead and i.stage == "eval"}
            self._state = drop_instances(self._state, dead)
            self._save()
            # Each hosts file re-renders only when ITS pods died: the trainer's
            # file must never be touched by eval churn, and vice versa.
            if dead - dead_eval:
                self._republish_from_ledger()
            if dead_eval:
                self._clear_eval_hosts()

    def _heat_marker_seen(self) -> bool:
        """Any ``heat_complete.json`` under the work-root newer than our rent.

        The trainer writes ``work_root/<base_seed>/heat_complete.json`` and the
        provisioner cannot know base_seed in advance (it keys rounds by the
        boundary block; the base seed is that block's hash). Only one round
        runs at a time, so any marker whose mtime postdates our earliest rent
        is this round's — and its directory name teaches us the base_seed for
        direct manifest polling.

        Only heat/final rents anchor the comparison: an eval pod is rented at
        the PREVIOUS round's manifest — before this round's fleet — and
        anchoring on it could resurrect the previous round's marker as a
        false teardown signal.
        """
        if self._state is None:
            return False
        rents = [_iso_ts(i.rented_at_iso) for i in self._state.instances if i.stage != "eval"]
        if not rents:
            return False
        rent_ts = min(rents)
        try:
            markers = sorted(Path(self.work_root).glob("*/heat_complete.json"))
        except OSError:
            return False
        for m in markers:
            try:
                if m.stat().st_mtime >= rent_ts:
                    self._learned_round_id = m.parent.name
                    return True
            except OSError:
                continue
        return False

    def _manifest_seen(self) -> bool:
        """The round's manifest is published — the round is over.

        Two detectors, either suffices: (a) once the marker taught us the
        base_seed, probe ``manifests/round-<base_seed>.json`` directly;
        (b) ``latest.json``'s round_id changed from the pre-rent baseline
        (works even if we never saw a marker; survives nothing — a restart
        loses the baseline, which is fine because (a) re-learns from the
        still-on-disk marker and the TTL backstops everything).
        """
        if self.manifest_store is None:
            return False
        if self._learned_round_id:
            try:
                self.manifest_store.get_text(manifest_round_key(self._learned_round_id))
                return True
            except Exception:  # noqa: BLE001 — not there yet (or store down: TTL backstops)
                pass
        current = self._latest_round_id()
        return (self._manifest_baseline is not None and current is not None
                and current != self._manifest_baseline)

    def _latest_round_id(self) -> str | None:
        if self.manifest_store is None:
            return None
        try:
            return str(json.loads(self.manifest_store.get_text(MANIFEST_LATEST_KEY))["round_id"])
        except Exception:  # noqa: BLE001 — no latest yet, or store down
            return None

    # ── RECONCILE ────────────────────────────────────────────────────────────

    def _reconcile_orphans(self) -> None:
        """Kill live pods the provisioner RENTED but the ledger does not own.

        Runs EVERY cycle (not just at startup): the hole it closes — a crash
        between a provider's create call and the ledger save — can open at any
        time, and an orphan bills until someone notices.

        Only names matching the provisioner's own scheme
        (``cascade-<round_id>-<stage>…``) are candidates: an operator's
        hand-rented pods legitimately share the ``cascade-`` prefix
        (``cascade-worker``, ``cascade-final-b``) and must NEVER be reaped —
        the reaper's mandate is strictly "pods this service created and then
        lost track of," not "pods that look cascade-related." And like every
        provider mutation, termination is gated on ``dry_run``.
        """
        owned = owned_ids(self._state) if self._state is not None else set()
        for name, prov in self.providers.items():
            lister = getattr(prov, "list_tagged", None)
            if lister is None:
                continue
            try:
                live = {p for p in lister(POD_TAG) if is_provisioner_pod_name(p)}
            except Exception as e:  # noqa: BLE001 — a down adapter reconciles next cycle
                log.warning("provider %s list_tagged failed (%s); skipping reconcile", name, e)
                continue
            for orphan in reconcile(owned, live):
                if self.dry_run:
                    log.info("[dry-run] reconcile: WOULD terminate orphan pod %s on %s",
                             orphan, name)
                    continue
                log.warning("reconcile: terminating ORPHAN pod %s on %s "
                            "(provisioner-named but not in the ledger)", orphan, name)
                try:
                    prov.terminate(orphan)
                except Exception as e:  # noqa: BLE001
                    log.error("orphan terminate %s failed: %s", orphan, e)

    # ── small helpers ────────────────────────────────────────────────────────

    def _instance(self, prov: object, pid: str, stage: str, cand=None) -> PodInstance:
        rented_iso = datetime.fromtimestamp(self.clock(), tz=UTC).isoformat()
        return PodInstance(provider=prov.name, instance_id=pid, stage=stage,
                           rented_at_iso=rented_iso,
                           sku=(cand.sku if cand is not None else ""),
                           gpus=(cand.gpus_per_pod if cand is not None else 1))

    def _find_instance(self, pid: str) -> PodInstance:
        return next(i for i in self._state.instances if i.instance_id == pid)

    def _terminate_and_drop(self, prov: object, pid: str) -> None:
        if self.dry_run:
            log.info("[dry-run] WOULD terminate pod %s", pid)
        else:
            try:
                prov.terminate(pid)
            except Exception as e:  # noqa: BLE001 — reconcile/TTL will retry
                log.error("terminate %s failed: %s", pid, e)
        self._state = drop_instances(self._state, {pid})
        self._addrs.pop(pid, None)
        self._save()

    def _save(self) -> None:
        with self._state_lock:
            save_state(self.state_path, self._state)


def _sku_for(policy: ProvisionPolicy, stage: str) -> str:
    return policy.heat.sku if stage == "heat" else policy.final.sku


def _market_sku_for(policy: ProvisionPolicy, stage: str) -> str:
    sp = policy.heat if stage == "heat" else policy.final
    return sp.marketplace_sku


def _gpus_for(policy: ProvisionPolicy, stage: str) -> int:
    return policy.heat.gpus_per_pod if stage == "heat" else policy.final.gpus_per_pod


def _iso_ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def parse_plan_output(text: str) -> dict:
    """The last JSON object line of ``cascade-trainer --plan-only`` output.

    The trainer prints exactly one JSON line, but bittensor/logging banners can
    precede it on stdout — scan from the end, same trick as the worker-receipt
    parse (``remote.parse_receipt``).
    """
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ProvisionError("no JSON plan payload found in --plan-only output")
