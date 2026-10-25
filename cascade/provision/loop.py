"""The cascade-provisioner service loop — the validator's eval pod, the
hosts files, and the pod ledger.

One cycle of the machine (each ``poll_seconds``, ~30s):

    RE-GATE  ledger pods adopted on restart re-run the health gate against
             the config loaded at THIS startup (a previous run's pass is not
             trusted);
    RECONCILE live pods named in this service's scheme that the ledger does
             not own are terminated (a crash between launch and ledger-save
             must not leak a billing box);
    TEARDOWN an eval pod dies when its round's receipt lands or a newer
             manifest supersedes it; every pod dies at its TTL;
    ACTIVATE the DEC-CA-0045 grid switch is learned in-loop (no restart);
    EVAL     a NEW round manifest in the store rents the round's eval pod —
             in a worker thread, since a boot wait must never starve the
             loop: SKU ladder × provider priority, provider-ready → SSH →
             the health gate, one replacement for a dud — and publishes it
             to the VALIDATOR's ``eval_hosts_path`` (never the trainer's
             ``hosts_path``). ``last_evaled_round`` persists in the ledger so
             restarts never double-rent.

The trainer's ``hosts_path`` is rendered from the operator's static fragment
plus any hand-added operator lanes already in the file (preserved by address),
plus any ledger pods still alive; the trainer rents its own pods (payer legs,
the era king — see :mod:`cascade.provision.funded`).

Every boundary is injected (chain client, providers, manifest store, health
check, ssh probe, clock, sleep), so the whole cycle is unit-tested with fakes;
the loop itself is thin glue over ``policy``/``state``/``health``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from ..shared.hippius import MANIFEST_LATEST_KEY
from .core import (
    DEFAULT_FORWARD_ENV,
    DEFAULT_READY_TIMEOUT,
    DEFAULT_REMOTE_PYTHON,
    DEFAULT_SSH_PORT,
    DEFAULT_WORKDIR,
    LaunchSpec,
    PodAddress,
    render_hosts_toml,
)
from .health import HealthReport
from .hostsfile import clear_hosts, parse_host_entries, render_host_entries, write_hosts
from .policy import (
    ProvisionPolicy,
    pods_for_slots,
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
# (+ optional -rN replacement / -gN lane suffixes; funded pods live under the
# trainer's own cascade-n<netuid>-<round_id>-funded-<slug> scheme, NOT this
# one). The orphan reaper matches on
# this — never on the bare POD_TAG prefix, which operators' hand-rented pods
# (cascade-worker, cascade-final-b, cascade-heat-2) legitimately share.
# NEVER add "funded" here: funded pods (payer legs AND the trainer's JIT
# operator king) are the TRAINER's, ledgered in funded_pods.json and named
# under the deployment-scoped cascade-n<netuid>-… scheme — a provisioner that
# matched them would reap the live king mid-leg (it can't see the trainer's
# ledger). heat/final stay in the pattern so a ledger leftover of the retired
# per-round fleet is still reaped.
_PROVISIONER_POD_RE = re.compile(r"^cascade-\d+-(heat|final|eval)(-|$)")
# The exact lane-name shape render_hosts_toml gives THIS service's pods
# (``cascade-<round>-<stage>-<i>-g<g>``, integer slot). A hosts.toml entry of
# that shape whose pod is not in the ledger is a stale self-published lane
# (a previous run's pod), never an operator's — operator lanes must use any
# other shape (e.g. ``cascade-<round>-final-m0-g0``) to be preserved.
_SELF_LANE_RE = re.compile(
    r"^cascade-(?:\d+-(?:heat|final)-\d+|[a-z0-9]+-[0-9a-f]{8})-g\d+$")


def lane_pod_name(provider: str, instance_id: str) -> str:
    """The STABLE per-pod stem of this service's hosts.toml lane names:
    ``cascade-<provider>-<8 hex of the instance id>`` (``-g<n>`` per GPU is
    appended by ``render_hosts_toml``). Derived from pod identity only — never
    the round id or the pod's position — so a pod carried across publishes
    (a second rental, a teardown of a sibling, a round boundary) keeps its
    name and the trainer's name-keyed lane pool never sees a renamed twin
    (which would double-dispatch onto it). Hex-uuid ids (shadeform) use
    their first 8 chars so the name is greppable against the provider
    console; anything else (lium names) is hashed to 8 hex."""
    iid = str(instance_id)
    key = iid[:8] if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]+", iid) else hashlib.sha1(
        iid.encode("utf-8")).hexdigest()[:8]
    prov = re.sub(r"[^a-z0-9]", "", str(provider).lower()) or "pod"
    return f"{POD_TAG}{prov}-{key}"

# Adopted-pod re-gate (the restart path): a pod resumed from the ledger passed
# its health gate under the config of WHOEVER rented it — possibly a stale
# image pin. The old pass is therefore worthless:
# every adopted pod re-runs the SAME gate against the config loaded at THIS
# startup before it stays in the published fleet. The bounded retry exists for
# the mid-training pod: an ssh blip during the re-gate must not read as a bad
# pod, while a deterministic mismatch (wrong digest) fails every attempt.
ADOPT_REGATE_ATTEMPTS = 3
ADOPT_REGATE_RETRY_S = 30.0


def is_provisioner_pod_name(name: str) -> bool:
    """True only for pod names this service itself creates (see _PROVISIONER_POD_RE)."""
    return _PROVISIONER_POD_RE.match(str(name)) is not None


def _scrub_known_host(ip: str, port: int = 22) -> None:
    """Drop any stored host key for a freshly-rented pod's address.

    Marketplace providers recycle IPs and the worker image generates fresh host
    keys per boot (``ssh-keygen -A`` in the entrypoint), so a stale entry makes
    every ``accept-new`` connection hard-fail with "Host key verification
    failed". Scrubbing at rent time keeps trust-on-first-use for the dispatch path."""
    targets = [ip] if port == 22 else [ip, f"[{ip}]:{port}"]
    for t in targets:
        with contextlib.suppress(Exception):
            subprocess.run(["ssh-keygen", "-R", t], capture_output=True, timeout=10)


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
    ``list_tagged``/``offer_price``); ``manifest_store`` duck-types
    ``S3Store.get_text``; ``health_check(addr, stage)`` returns a
    :class:`HealthReport` (``None`` ⇒ gate skipped, e.g. tests/dry-run);
    ``ssh_probe(ip, port)`` is the TCP reachability wait. ``clock`` is seconds
    (epoch) and everything time-based flows through it.
    """

    policy: ProvisionPolicy
    providers: dict[str, object]
    chain_client: object                                    # .current_block() -> int
    render: RenderSettings
    hosts_path: Path
    work_root: Path
    state_path: Path
    epoch_blocks: int
    # Scheduled cadence change, mirroring [round] epoch_blocks_prev /
    # epoch_activation_block. Both 0 ⇒ epoch_blocks always applies. Resolved
    # per tick (never cached) so the provisioner flips at the same block as the
    # trainer and validator without needing its own restart.
    epoch_blocks_prev: int = 0
    epoch_activation_block: int = 0
    # Stake-weighted activation (DEC-CA-0045): called every cycle with
    # ``(chain_client, block)``; returns ``(epoch_blocks, epoch_blocks_prev,
    # epoch_activation_block)`` the moment the fleet's lock-in applies, else
    # None. Resolved IN the loop so no operator restart is ever needed to
    # learn the grid switch.
    activation_fn: object | None = None
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
    # Fired once per healthy pod AFTER the health gate (addr, stage, provider):
    # best-effort warm-up work that must never gate the pod (the deployed
    # chain.toml push, the benchmark-data prewarm).
    prewarm: Callable[[PodAddress, str, str], None] | None = None
    # Raw [[host]] TOML appended verbatim to EVERY hosts.toml publish — the
    # operator's static pods (e.g. a long-lived final pod) that the provisioner
    # must never drop. clear/teardown re-renders keep it too: "no dynamic pods"
    # must degrade to "static fleet", never to an empty file while a static
    # final exists.
    static_hosts_text: str = ""
    # When set, the static fragment is RE-READ on every publish (an operator
    # edit lands without a restart); ``static_hosts_text`` is the last-good
    # copy used when the file is missing or fails to parse.
    static_hosts_path: Path | None = None
    # Called at the top of EVERY cycle (never raises consequences — errors are
    # suppressed): the logging self-heal hook. bittensor's logging init strips
    # handlers and raises the level (to CRITICAL) on NAMED loggers when a chain
    # client connects — including reconnects — so a one-time logging setup
    # cannot survive; only a per-cycle re-assert can.
    on_cycle: Callable[[], None] | None = None
    ssh_probe: Callable[[str, int], bool] = field(default=lambda ip, port: True)
    # Rebuilds chain_client when the block number freezes: a bittensor
    # websocket can die QUIETLY and keep answering current_block() with a
    # stale value — the loop then cycles forever on a frozen chain view.
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
    _addrs: dict[str, PodAddress] = field(default_factory=dict, init=False, repr=False)
    _last_block: int | None = field(default=None, init=False, repr=False)
    _block_changed_at: float = field(default=0.0, init=False, repr=False)
    _last_heartbeat_at: float = field(default=0.0, init=False, repr=False)
    _eval_inflight: bool = field(default=False, init=False, repr=False)
    _eval_thread: object = field(default=None, init=False, repr=False)
    _state_lock: object = field(default_factory=__import__("threading").RLock,
                                init=False, repr=False)
    # The eval stage's rent-once latch: the manifest round an eval pod was
    # last rented — or
    # deliberately skipped — for. Restored from the ledger so restarts never
    # double-rent; in dry-run it lives in memory only (no disk mutation).
    _last_evaled_round: str = field(default="", init=False, repr=False)
    # pod id → names of the health checks it failed (set by _boot_and_gate,
    # consumed once by the replacement path's facility-exclusion decision).
    _gate_failures: dict = field(default_factory=dict, init=False, repr=False)
    # Addresses of pods this service terminated (any path) — their hosts.toml
    # entries must never be preserved as "operator lanes" on a republish.
    _retired_ips: set = field(default_factory=set, init=False, repr=False)
    # Operator lanes already announced as preserved (log once per name).
    _foreign_logged: set = field(default_factory=set, init=False, repr=False)
    # Ledger-adopted pods awaiting their re-gate against the CURRENT config
    # (see _maybe_regate_adopted): set on resume, cleared when the re-gate
    # worker is spawned on the first cycle.
    _adopt_pending: bool = field(default=False, init=False, repr=False)
    _adopt_thread: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._state = load_state(self.state_path)
        if self._state is not None:
            log.info("resumed ledger %s: round=%s, %d instance(s)",
                     self.state_path, self._state.round_id, len(self._state.instances))
            self._last_evaled_round = self._state.last_evaled_round
            # Adoption is NOT trust: these pods passed the gate under the
            # config of the run that RENTED them, which may have carried a
            # different image pin. Flag them for a re-gate against the config
            # loaded at THIS startup (first cycle, off-thread — see
            # _maybe_regate_adopted).
            self._adopt_pending = bool(self._state.instances)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def ttl_hours(self) -> float:
        """Hard pod-lifetime backstop, in hours.

        While a cadence change is pending this takes the LONGER of the two
        epoch lengths. The asymmetry is deliberate: a TTL that is too long only
        delays a backstop that normal receipt-latched teardown reaches first,
        while a TTL that is too short reaps live pods mid-round. Using the
        post-switch 12h here before activation would kill a 24h round's fleet
        at the halfway mark."""
        span = max(int(self.epoch_blocks),
                   int(self.epoch_blocks_prev) if self.epoch_activation_block else 0)
        return self.policy.ttl_epochs * (max(1, span) * 12.0 / 3600.0)

    # ── the cycle ────────────────────────────────────────────────────────────

    def run_once(self) -> None:
        """One poll tick: reconcile strays, tear down what's due, maybe rent
        the eval pod.

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
            # reaching any of them.
            log.info("heartbeat: cycle start, last_block=%s, owned_pods=%d",
                     self._last_block if self._last_block is not None else "?",
                     len(self._state.instances) if self._state else 0)
            self._last_heartbeat_at = now
        self._maybe_regate_adopted()
        self._reconcile_orphans()
        self._teardown_due_pods()
        block = self._current_block()
        # Re-assert AFTER the chain read: constructing the bittensor client
        # strips the cascade tree's handlers, so everything below would log
        # into the void — the cycle-start hook alone is too early.
        if self.on_cycle is not None:
            with contextlib.suppress(Exception):
                self.on_cycle()
        self._maybe_apply_activation(block)
        self._maybe_provision_eval()

    @staticmethod
    def _with_deadline(fn, seconds: float):
        """Run ``fn()`` with a HARD deadline in a helper thread.

        bittensor's websocket calls can hang indefinitely (no client-side
        timeout) — four rental windows were lost to silent chain-source stalls.
        A timeout bounds the caller's wait but does not stop the underlying
        call. Shared with the trainer through ``shared.concurrency``."""
        from ..shared.concurrency import call_with_deadline

        return call_with_deadline(fn, seconds)

    def _maybe_apply_activation(self, block: int) -> None:
        """DEC-CA-0045: learn the fleet's rollover (once it locks in) and
        switch this loop's grid at the same block the trainer and validators
        do. Best-effort under a hard deadline — never blocks the cycle."""
        if self.activation_fn is None:
            return
        try:
            grid = self._with_deadline(
                lambda: self.activation_fn(self.chain_client, int(block)), 120.0)
        except Exception as e:  # noqa: BLE001 — retried next cycle
            log.warning("activation check failed/hung (%s); retrying next cycle",
                        type(e).__name__)
            return
        if not grid:
            return
        eb, prev, act = (int(x) for x in grid)
        if (eb, prev, act) == (self.epoch_blocks, self.epoch_blocks_prev,
                               self.epoch_activation_block):
            return
        log.warning("activation: DEC-CA-0043 rollover ARMED at block %d — grid %d → %d "
                    "from there", act, prev, eb)
        self.epoch_blocks, self.epoch_blocks_prev, self.epoch_activation_block = eb, prev, act

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

    # ── PICK (the SKU ladder × provider priority) ────────────────────────────

    def _iter_offers(self, sp, slots: int, stage: str):
        """Yield every viable ``(provider, price, candidate, pods)``, ladder order.

        Candidates in the stage's configured order, providers in priority order
        within each candidate. Lazy on purpose: capacity is probed when the
        consumer advances. The pod count is re-derived per candidate. Any
        adapter fault just skips that provider: a broken adapter means fewer
        offers, never a dead provisioner.
        """
        for rung, cand in enumerate(sp.sku_candidates):
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
                if rung > 0:
                    log.info("%s stage falling back to %dx%s on %s (earlier rungs had no offer)",
                             stage, cand.gpus_per_pod, cand.sku, name)
                yield prov, float(price), cand, pods

    def _pick_offer(self, sp, slots: int,
                    stage: str) -> tuple[object, float, object, int] | None:
        """First viable rung of :meth:`_iter_offers`, or ``None`` (no capacity)."""
        return next(self._iter_offers(sp, slots, stage), None)

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
                    cand, pods: int, suffix: str = "",
                    ) -> list[tuple[PodInstance, PodAddress]]:
        # round_id is the manifest round id — digits, satisfying
        # _PROVISIONER_POD_RE. ``suffix`` distinguishes a batch's names from
        # an earlier attempt's (same regex, distinct names).
        spec = LaunchSpec(
            sku=cand.marketplace_sku, count=pods, image=self.render.image,
            ssh_pubkey=self.render.ssh_pubkey, ssh_port=self.render.ssh_port,
            name_prefix=f"{POD_TAG}{round_id}-{stage}{suffix}",
            gpus_per_pod=cand.gpus_per_pod,
        )
        try:
            pod_ids = prov.launch(spec)
        except Exception as e:  # noqa: BLE001 — a failed stage is a smaller fleet
            log.error("round %s %s: launch on %s failed: %s", round_id, stage, prov.name, e)
            return []
        # Write-ahead: the ledger owns these ids BEFORE we do anything else
        # with them — a crash from here on still tears them down on restart.
        # Locked: the teardown sweep on the loop thread mutates the same
        # state while a rent worker runs.
        with self._state_lock:
            for pid in pod_ids:
                self._state = add_instance(self._state,
                                           self._instance(prov, pid, stage, cand))
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
            # failed executor.
            lemon = getattr(prov, "machine_of", lambda _p: None)(pid)
            # An image_digest failure means the FACILITY's host image cache is
            # serving a stale template — a same-region replacement predictably
            # re-boots the same wrong image. Walk the replacement to a
            # different region instead of a different machine.
            # A fresh_boot failure (recycled container) is the same facility
            # property — the container cache, not the machine.
            bad_region = None
            facility_faults = {"image_digest", "fresh_boot"} & set(
                self._gate_failures.pop(pid, ()))
            if facility_faults:
                bad_region = getattr(prov, "region_of", lambda _p: None)(pid)
                if bad_region:
                    log.warning("round %s %s: %s failure on %s — "
                                "excluding region %s from the replacement pick",
                                round_id, stage, "/".join(sorted(facility_faults)),
                                pid, bad_region)
            self._terminate_and_drop(prov, pid)
            rspec = replace(spec, count=1,
                            name_prefix=f"{POD_TAG}{round_id}-{stage}{suffix}-r{i}",
                            exclude_ids=spec.exclude_ids + ((lemon,) if lemon else ()),
                            exclude_regions=spec.exclude_regions
                            + ((bad_region,) if bad_region else ()))
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
                self._gate_failures.pop(rid, None)
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
            _scrub_known_host(addr.ip, addr.ssh_port)
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
                # Provider-echoed image digest (empty when unsupported/bare):
                # the gate's fallback attestation for sshd-as-PID-1 images.
                attest_fn = getattr(prov, "launched_image_digest", None)
                attested = (attest_fn(pid) or "") if attest_fn is not None else ""
                report = (self.health_check(addr, stage, prov.name,
                                            sku=cand.sku, gpus=cand.gpus_per_pod,
                                            attested_digest=attested)
                          if cand is not None else
                          self.health_check(addr, stage, prov.name,
                                            attested_digest=attested))
                if not report.ok:
                    log.warning("pod %s failed health gate: %s", pid, report.summary())
                    # Remember WHICH checks failed: the replacement path treats
                    # an image_digest failure as a facility property (stale
                    # host image cache), not a machine lemon.
                    self._gate_failures[pid] = tuple(c.name for c in report.failures)
                    return None
        except Exception as e:  # noqa: BLE001 — any boot fault is a failed pod, not a dead loop
            log.warning("pod %s boot/health errored: %s", pid, e)
            return None
        if self.prewarm is not None:
            try:
                self.prewarm(addr, stage, prov.name)
            except Exception as e:  # noqa: BLE001 — pre-warm never gates a healthy pod
                log.warning("pod %s pre-warm hook errored (ignored): %s", pid, e)
        self._addrs[pid] = addr
        return addr

    # ── ADOPTION RE-GATE (the restart path) ──────────────────────────────────

    def _maybe_regate_adopted(self) -> None:
        """Re-gate every ledger-adopted pod against the CURRENT config, once.

        A restart adopts still-running rentals from the ledger instead of
        re-renting — but their health-gate pass belongs to the run that rented
        them, possibly under a stale image pin (see ``ADOPT_REGATE_ATTEMPTS``).
        The re-gate runs the SAME injected
        ``health_check`` fresh rentals use — bound at startup to the current
        pin — in a daemon worker like the eval rent, so gate ssh
        round-trips never block the poll loop. A pod that fails gets the dud
        treatment (terminate, drop from the ledger) and the affected hosts
        file re-renders without it. Skipped when the gate itself is disabled
        (``health_check is None``) and in dry-run (which mutates nothing).
        """
        if not self._adopt_pending:
            return
        self._adopt_pending = False
        if self.dry_run or self.health_check is None or self._state is None:
            return
        insts = tuple(self._state.instances)
        if not insts:
            return
        log.info("re-gating %d adopted pod(s) against the current config "
                 "(a previous run's gate pass is not trusted across a restart)",
                 len(insts))

        def _worker() -> None:
            try:
                self._regate_adopted(insts)
            except Exception as e:  # noqa: BLE001 — a failed re-gate never kills the loop
                log.exception("adopted-pod re-gate failed: %s", e)

        t = threading.Thread(target=_worker, name="adopt-regate", daemon=True)
        self._adopt_thread = t                   # tests join() for determinism
        t.start()

    def _regate_adopted(self, insts: tuple[PodInstance, ...]) -> None:
        dropped_fleet = dropped_eval = False
        for inst in insts:
            prov = self.providers.get(inst.provider)
            if prov is None:
                log.error("adopted pod %s: no adapter for provider %r — cannot "
                          "re-gate; leaving it to the TTL backstop",
                          inst.instance_id, inst.provider)
                continue
            if self._regate_one(prov, inst):
                log.info("adopted %s pod %s re-passed the health gate",
                         inst.stage, inst.instance_id)
                continue
            log.error("adopted %s pod %s FAILED the re-gate against the current "
                      "config; terminating and dropping it from the fleet",
                      inst.stage, inst.instance_id)
            self._terminate_and_drop(prov, inst.instance_id)
            if inst.stage == "eval":
                dropped_eval = True
            else:
                dropped_fleet = True
        # Same split as the teardown sweep: each hosts file re-renders only
        # when ITS pods died — the trainer's file is never touched by eval
        # churn, and vice versa.
        if dropped_fleet:
            self._republish_from_ledger()
        if dropped_eval:
            self._clear_eval_hosts()

    def _regate_one(self, prov: object, inst: PodInstance) -> bool:
        """One adopted pod through the existing gate, with a bounded retry.

        A fresh rental absorbs transient faults in the boot path (ready-wait,
        auth-injection polling) before its gate runs; an adopted pod is probed
        cold and may be MID-TRAINING — one ssh blip must not kill it, so a
        failed attempt retries up to ``ADOPT_REGATE_ATTEMPTS`` times,
        ``ADOPT_REGATE_RETRY_S`` apart (worker thread — startup never blocks).
        A deterministic mismatch (wrong image digest, wrong SKU) fails every
        attempt and the pod is declared bad.
        """
        for attempt in range(1, ADOPT_REGATE_ATTEMPTS + 1):
            if attempt > 1:
                self.sleep(ADOPT_REGATE_RETRY_S)
            try:
                addr = self._addr_for(inst)
                if addr is None:
                    detail = "provider reports no address"
                else:
                    # Provider-echoed digest: the gate's fallback attestation
                    # for sshd-as-PID-1 images (same as _boot_and_gate).
                    attest_fn = getattr(prov, "launched_image_digest", None)
                    attested = ((attest_fn(inst.instance_id) or "")
                                if attest_fn is not None else "")
                    # The ledgered sku/gpus assert what was ACTUALLY rented
                    # (SKU fallback); a pre-fallback ledger without them falls
                    # back to the stage policy inside the gate factory.
                    shape = ({"sku": inst.sku, "gpus": inst.gpus}
                             if inst.sku else {})
                    report = self.health_check(addr, inst.stage, inst.provider,
                                               attested_digest=attested, adopted=True,
                                               **shape)
                    if report.ok:
                        return True
                    detail = report.summary()
            except Exception as e:  # noqa: BLE001 — a dead transport is a failed attempt
                detail = f"re-gate errored: {e}"
            log.warning("adopted pod %s re-gate attempt %d/%d failed: %s",
                        inst.instance_id, attempt, ADOPT_REGATE_ATTEMPTS, detail)
        return False

    # ── PUBLISH ──────────────────────────────────────────────────────────────

    def _write_hosts(self, sections: list[str]) -> None:
        """Publish dynamic sections + the operator's static entries.

        The static text rides along on EVERY publish, so a long-lived
        hand-rented pod is never dropped by provisioner activity. Only with
        no static text, no operator lanes AND no dynamic pods does the file
        clear.
        """
        foreign = self._foreign_hosts_text(sections)
        content = "".join([self._static_hosts(), *sections, foreign])
        if content.strip():
            write_hosts(self.hosts_path, content)
        else:
            clear_hosts(self.hosts_path)

    def _static_hosts(self) -> str:
        """The operator's static fragment, re-read from disk when a path is set."""
        if self.static_hosts_path is None:
            return self.static_hosts_text
        try:
            text = self.static_hosts_path.read_text(encoding="utf-8")
            parse_host_entries(text)                     # validate before trusting
        except Exception as e:  # noqa: BLE001 — keep the last-good copy
            log.warning("static_hosts %s unreadable (%s); using the last-good copy",
                        self.static_hosts_path, e)
            return self.static_hosts_text
        self.static_hosts_text = text
        return text

    def _foreign_hosts_text(self, sections: list[str]) -> str:
        """Entries in the CURRENT hosts.toml that this service does not own.

        An operator who appends hand-rented lanes to hosts.toml must not lose
        them on the next provisioner publish.
        Ownership is by ADDRESS, not name: an entry is ours iff its host is a
        ledger pod's address (re-rendered from the ledger anyway) or one we
        terminated. Everything else is preserved verbatim-in-substance —
        except entries in this service's exact lane-name shape whose pod is
        gone (``_SELF_LANE_RE``: a previous run's stale lane, never an
        operator's), and names that the static fragment or this publish
        already carries. A torn/unparseable file preserves nothing (never
        re-emit garbage into the trainer's contract file).
        """
        try:
            current = self.hosts_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        try:
            entries = parse_host_entries(current)
        except Exception as e:  # noqa: BLE001 — malformed: preserve nothing
            log.warning("hosts.toml did not parse (%s); operator lanes not preserved", e)
            return ""
        owned_ips = {a.ip for a in self._addrs.values()} | set(self._retired_ips)
        taken: set[str] = set()
        with contextlib.suppress(Exception):
            taken |= {str(e.get("name")) for e in parse_host_entries(
                "".join([self.static_hosts_text, *sections]))}
        keep: list[dict] = []
        for e in entries:
            name, host = str(e.get("name", "")), str(e.get("host", ""))
            if not name or not host or host in owned_ips or name in taken:
                continue
            if _SELF_LANE_RE.match(name):
                log.info("dropping stale self-published lane %s (%s): pod not in the ledger",
                         name, host)
                continue
            keep.append(e)
            taken.add(name)
        for e in keep:
            if e["name"] not in self._foreign_logged:
                self._foreign_logged.add(e["name"])
                log.info("preserving operator lane %s (%s) not in the ledger",
                         e["name"], e["host"])
        return render_host_entries(keep)

    def _stage_gpus(self, stage: str) -> int:
        """The pod shape to fan a ledger row out to when the row records no
        ``gpus``: the stage policy's ``gpus_per_pod`` (1 for a stage with no
        policy — the retired heat/final fleet)."""
        sp = getattr(self.policy, stage, None)
        return max(1, int(getattr(sp, "gpus_per_pod", 1) or 1)) if sp is not None else 1

    def _publish_hosts(self, by_stage: dict[str, list[tuple[PodInstance, PodAddress]]]) -> None:
        sections = []
        for stage in ("heat", "final"):                     # stable order in the file
            entries = by_stage.get(stage)
            if not entries:
                continue
            fleet_gpus = entries[0][0].gpus or self._stage_gpus(stage)
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
                # Stable per-pod stems (H4): a republish must never rename a
                # lane the trainer already holds.
                pod_names=[lane_pod_name(inst.provider, inst.instance_id)
                           for inst, _addr in entries],
            ))
        self._write_hosts(sections)
        n = sum(len(v) for v in by_stage.values())
        log.info("published %s: %d dynamic pod(s) across %s%s", self.hosts_path, n,
                 sorted(by_stage), " + static entries" if self.static_hosts_text else "")

    def _republish_from_ledger(self) -> None:
        """Re-render hosts.toml from the surviving instances (post-teardown).

        Only live ledger pods are listed (plus the static fragment and the
        operator lanes) — the file never lists a dead box.
        """
        if self._state is None:
            self._write_hosts([])                # static-only / empty file
            return
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

        The eval stage is manifest-triggered: the validator only needs GPU
        once a round has PUBLISHED (the GIFT-Eval gate and cascade bench score
        the manifest's checkpoints). ``latest.json`` moving to a round we have
        not yet served is the rent signal.

        The ``_last_evaled_round`` latch is set the moment we act — including
        on no-capacity and dry-run — so a failure degrades the round to local
        validator evals instead of hammering providers every poll. It is
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
        # Deliberately NO ladder escalation for eval (unlike the boundary
        # stages): it is one pod with pick-time fallbacks and one replacement,
        # and the validator's local-eval fallback is cheap — an escalation
        # loop here would only delay the round's evals for marginal gain.
        picked = self._pick_offer(self.policy.eval, 1, "eval")
        if picked is None:
            log.error("round %s eval: no provider has capacity; the validator "
                      "runs this round's evals locally (degraded, never lost)", latest)
            self._persist_eval_latch(latest)
            return
        prov, price, cand, pods = picked
        # Worst-case spend breaker (every pod billed for the full TTL);
        # skipping is cheap because the validator's local CPU evals are
        # genuinely viable.
        projected = pods * float(price) * self.ttl_hours
        if projected > self.policy.max_spend_per_round:
            log.warning("round %s eval: skipped by budget — $%.2f > cap $%.2f "
                        "(validator evals run locally)",
                        latest, projected, self.policy.max_spend_per_round)
            self._persist_eval_latch(latest)
            return
        if self.dry_run:
            log.info("[dry-run] round %s eval: WOULD rent %d × %d-GPU %s pod(s) on %s "
                     "@ $%.2f/hr (tag cascade-%s-eval)", latest, pods,
                     cand.gpus_per_pod, cand.sku, prov.name, price, latest)
            return
        self._persist_eval_latch(latest)
        # Rent + boot + health can take 15+ minutes (provider readiness alone
        # is a 900s wait) and MUST NOT block the loop. The slow leg runs in a daemon worker; the loop keeps cycling (the
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
        with self._state_lock:
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
            gpus_per_pod=inst.gpus or self._stage_gpus("eval"),
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
        """Terminate every owned pod whose teardown is due: an eval pod once
        its round's receipt lands (or a newer manifest supersedes it), any
        pod once its TTL elapses (the backstop that needs no signal)."""
        if self._state is None or not self._state.instances:
            return
        now = self.clock()
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
            if not teardown_due(inst.stage, receipt_seen=receipt, newer_manifest=newer,
                                rented_at=_iso_ts(inst.rented_at_iso), now=now,
                                ttl_hours=self.ttl_hours):
                continue
            log.info("tearing down %s pod %s (receipt=%s newer=%s ttl=%.1fh)",
                     inst.stage, inst.instance_id, receipt, newer, self.ttl_hours)
            prov = self.providers.get(inst.provider)
            if prov is None:
                log.error("no adapter for provider %r — pod %s may be LEAKED",
                          inst.provider, inst.instance_id)
                continue
            if self.dry_run:
                log.info("[dry-run] WOULD terminate %s pod %s", inst.stage, inst.instance_id)
                continue
            try:
                prov.terminate(inst.instance_id)
            except Exception as e:  # noqa: BLE001 — keep tearing down the rest
                log.error("terminate %s failed (may be leaked!): %s", inst.instance_id, e)
                continue
            dead.add(inst.instance_id)
            self._retire_addr(inst.instance_id)
        if dead:
            dead_eval = {i.instance_id for i in self._state.instances
                         if i.instance_id in dead and i.stage == "eval"}
            with self._state_lock:
                self._state = drop_instances(self._state, dead)
                self._save()
            # Each hosts file re-renders only when ITS pods died: the trainer's
            # file must never be touched by eval churn, and vice versa.
            if dead - dead_eval:
                self._republish_from_ledger()
            if dead_eval:
                self._clear_eval_hosts()

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
        if self._eval_inflight:
            # A rent worker is between a provider's create call and its
            # ledger write-ahead at unpredictable moments — reaping now could
            # kill a pod that is owned but not yet recorded. Skipping a tick
            # is free; the reaper runs every cycle.
            return
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
        with self._state_lock:
            self._state = drop_instances(self._state, {pid})
            self._retire_addr(pid)
            self._save()

    def _retire_addr(self, pid: str) -> None:
        """Forget a terminated pod's address, remembering its IP as retired so
        its hosts.toml entry is never preserved as an operator lane."""
        addr = self._addrs.pop(pid, None)
        if addr is not None:
            self._retired_ips.add(addr.ip)

    def _save(self) -> None:
        with self._state_lock:
            save_state(self.state_path, self._state)


def _iso_ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()
