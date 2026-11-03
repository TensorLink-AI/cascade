"""The era king's JIT pod and operator-rented leg pods (TrainerRunner mixin)."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

from .corpus import DIVERGED_MARKER
from .faults import (
    STALL_CLASS,
    _FundedDiverged,
    _FundedLegSkip,
    _FundedOperatorFallback,
    _FundedTamper,
    classify_leg_exception,
)
from .lane_pool import (
    _lium_provider,
    host_cannot_bench,
    king_pod_name_prefix,
)
from .runner_common import (
    claim_executor,
    live_claimed_executors,
    release_executor_of,
    router_kw,
)
from .runner_generation import leg_corpus

log = logging.getLogger("cascade.trainer")


def _wait_out_rent_halt(runner, name_prefix: str) -> None:
    """Hold the king's rent while the box-health halt (a disk halt) stands:
    one cheap check per ``FUNDED_RENT_RETRY_SECONDS``, no provider call. At
    the latest safe start it gives up with :class:`RentsHalted` — which the
    era loop never counts as a king-leg failure (review R2)."""
    from .box_health import RentsHalted, rent_halt_refusal

    sleep = getattr(runner, "_rent_wait_sleep", None) or time.sleep
    now_fn = getattr(runner, "_rent_wait_now", None) or time.time
    while True:
        halted = rent_halt_refusal(runner, king=True)
        if not halted:
            return
        _check_king_cancel(runner)
        if now_fn() >= runner._funded_rent_wait_deadline():
            runner._ledger_remove(f"{name_prefix}-0")
            raise RentsHalted(halted)
        sleep(runner.FUNDED_RENT_RETRY_SECONDS)


def king_pod_ttl_seconds(runner) -> float:
    """Provider TTL of a king pod / king backup: one era's wall + ``[round]
    king_pod_ttl_margin_seconds`` (0 = no TTL). Lium cannot extend a TTL
    (``lium rm --in`` only adds schedules), so this is a create-time
    backstop; the idle-king reaper (money_guard.king_reaper_tick) is the
    enforcing half."""
    from .money_guard import era_wall_seconds

    margin = float(getattr(runner.cfg.round, "king_pod_ttl_margin_seconds", 0) or 0)
    if margin <= 0:
        return 0.0
    return era_wall_seconds(runner.cfg.round) + margin


def _check_king_cancel(runner):
    event = getattr(runner.__dict__.get('_leg_local_obj'), 'king_cancel', None)
    if event is not None and event.is_set():
        from .remote import RemoteDispatchError
        raise RemoteDispatchError('redundant king replica cancelled', returncode=254)


class _OperatorPodMixin:
    """Operator-account pods: the era king's JIT pod and operator-rented legs."""

    def _rent_king_host(self, round_id: str):
        """The era king's JIT pod on the OPERATOR's account (``round_id`` is
        the era's base seed).

        Rents through LiumProvider directly on the operator's own key (never a
        payer's — provision.funded must never bill an operator pod), same
        image and SKU list as every funded challenger. The pod is kept for the
        whole era — the era's benches and payer verifications run on it — and
        ledgered with ``payer_hotkey = ""`` (the operator marker) so a crash
        never leaks it. Rented once per era under a lock; a rent failure
        raises, failing the king leg (the scheduler retries)."""
        with self._funded_king_lock:
            if self._funded_king_host is not None:
                return self._funded_king_host
            from ..provision.core import LaunchSpec, ProvisionError
            from ..provision.funded import quarantine_lemon_host, terminate_verified
            from ..provision.state import PodInstance
            from .host_preflight import sync_host_blocklist
            from .remote import RemoteHost

            sync_host_blocklist(self)
            rnd = self.cfg.round
            profile = self._funded_pod_profile()
            key_path = Path(profile.key_path or "").expanduser()
            ssh_pubkey = (key_path.parent / (key_path.name + ".pub")
                          ).read_text(encoding="utf-8").strip()
            skus = self._funded_skus_for_rent()
            sku = skus[0] if skus else rnd.funded_pod_sku
            from ..provision.operator_pool import operator_price_caps, operator_provider

            # The king is never held by a breaker another rent tripped (F2);
            # only its own 402 trips it.
            provider = operator_provider(self, lambda: _lium_provider(rnd, price_caps=operator_price_caps(self)),
                                         king=True)
            name_prefix = getattr(self.__dict__.get("_leg_local_obj"), "king_primary_prefix", None) or king_pod_name_prefix(self.cfg.subnet.netuid, round_id)

            def _king_remote(addr, host_key: str) -> RemoteHost:
                # pinned_host_key: every ssh/scp to the king pod runs
                # StrictHostKeyChecking=yes against a file holding ONLY the key
                # scanned at readiness — exactly like a payer leg (a Lium
                # ip:port reused by a fresh container breaks the shared
                # known_hosts).
                return RemoteHost(
                    name="funded-king", host=addr.ip, port=addr.ssh_port,
                    user=profile.user, key_path=profile.key_path,
                    remote_python=profile.remote_python, workdir=profile.workdir,
                    # The whole rented pod: every GPU of a layout trains.
                    cuda_device=None, chain_toml=profile.chain_toml,
                    # Credential-free like a payer pod: nothing from the
                    # orchestrator's environment reaches the king pod — the
                    # generator (public Hub refs pull anonymously; vault refs
                    # are staged) and the warm-start init need no login, and
                    # the checkpoint is harvested + uploaded from here rather
                    # than pushed by the pod: a pod that runs miner code must
                    # not hold the bucket/Hub keys.
                    forward_env=(), isolated=True, ssh_options=profile.ssh_options,
                    pinned_host_key=host_key, stage="final", sku=sku,
                    allocation_id=pod_id, allocation_provider=provider.name,
                    allocation_incident=provider.incident_of(pod_id) if callable(getattr(provider, "incident_of", None)) else f"{provider.name}:{pod_id}")

            def _ledger_king(pod_id: str) -> None:
                self._ledger_add(PodInstance(
                    provider=provider.name, instance_id=pod_id, stage="funded",
                    rented_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
                    sku=sku, gpus=1, payer_hotkey=""))

            # Write-ahead (name is deterministic): a crash during the
            # readiness wait must leave a ledger row for the startup sweep —
            # an operator king pod is on NOBODY else's radar.
            _ledger_king(f"{name_prefix}-0")

            def _fail_king(pod_id: str, why: str) -> None:  # noqa: ANN001
                # A king pod that never came up or runs the wrong runtime is
                # a lemon HOST: keep the retry (and every payer leg) off it.
                from .gpu_health import record_pod_failure
                with contextlib.suppress(Exception):
                    record_pod_failure(self, provider, pod_id, sku, why,
                        kind='stale' if 'runtime' in why else 'slow' if 'rejected' in why else 'boot')
                quarantine_lemon_host(provider, pod_id, why)
                from .money_guard import settle_operator_pod

                settle_operator_pod(self, pod_id)        # booked at launch: its real cost
                # Verified teardown, ledger row dropped only when CONFIRMED
                # gone (bare terminate swallows a failed rm as success).
                try:
                    if terminate_verified(provider, pod_id):
                        self._ledger_remove(pod_id)
                    else:
                        log.error("king pod %s still LIVE after terminate — "
                                  "kept on ledger for the sweep", pod_id)
                except Exception as te:  # noqa: BLE001 — row stays for the sweep
                    log.error("king pod %s teardown failed (kept on ledger): "
                              "%s", pod_id, te)
                raise ProvisionError(why)

            def _give_up(why: str) -> None:
                # The king is REQUIRED: once it cannot be had, release every
                # challenger still polling for a GPU so the round fails now
                # (and retries next poll) instead of after their deadline.
                abort = getattr(self, "_funded_wait_abort", None)
                if abort is not None:
                    abort.set()
                raise ProvisionError(why)

            from .remote import probe_worker_runtime
            from .runner_funded_pods import allocation_lock

            # Adopt-before-rent: a retry (or a restart) finds the previous
            # attempt's king pod still up (the startup sweep keeps it).
            # Reusing it is what makes the worker's retry-without-retrain
            # marker reachable for a JIT king — the finished checkpoint lives
            # in that pod's work dir. Same attestation as a fresh rent
            # (deployed chain.toml push + runtime probe); a pod that fails it
            # is torn down and the rent below proceeds as before.
            live = getattr(provider, "live_pod_address", None)
            addr = live(f"{name_prefix}-0") if live is not None else None
            if addr is not None:
                pod_id = f"{name_prefix}-0"
                log.warning("king pod %s is still LIVE at %s:%d from a prior attempt "
                            "of this era — adopting it instead of renting (a "
                            "complete checkpoint there skips the retrain)",
                            pod_id, addr.ip, addr.ssh_port)
                claim_executor(self, provider.machine_of(pod_id) or "", pod_id)
                try:
                    king_host = self._push_deployed_chain_toml(
                        _king_remote(addr, self._pin_king_host_key(pod_id, addr)))
                    why = probe_worker_runtime(king_host, **({"required_flags": ("--local-only", "--continuation-root")} if getattr(self.cfg.round, "ops_continuation_interval_seconds", 0) else {}))
                    if why:
                        _fail_king(pod_id, f"adopted king pod runtime rejected: {why}")
                    _check_king_cancel(self)
                    self._funded_king_host = king_host
                    return self._funded_king_host
                except ProvisionError as e:
                    log.warning("adopting king pod %s failed (%s); renting fresh", pod_id, e)
                except Exception as e:  # noqa: BLE001 — unreachable pod ⇒ rent fresh
                    log.warning("adopting king pod %s failed (%s); tearing it down and "
                                "renting fresh", pod_id, e)
                    with contextlib.suppress(ProvisionError):
                        _fail_king(pod_id, f"adopted king pod unreachable: {e}")

            from .cost_guards import measured_wall_refusal

            unmeasured = measured_wall_refusal(rnd)
            if unmeasured:
                # Same gate as every funded rent: no measured wall for the
                # pinned image ⇒ no JIT king pod either (the write-ahead row
                # goes — nothing was launched under it).
                log.error("king rent: %s", unmeasured)
                self._ledger_remove(f"{name_prefix}-0")
                raise ProvisionError(unmeasured)
            attempt = 0
            while True:
                _check_king_cancel(self)
                # A disk halt holds the king too (its checkpoint could not be
                # saved either): WAIT at the poll cadence — no capacity probe,
                # never a king-leg failure, never a pod rotation (review R2).
                _wait_out_rent_halt(self, name_prefix)
                attempt += 1
                # A sold-out marketplace is not a round failure: wait for a GPU
                # (outside the rent lock) up to the round's latest safe start —
                # the legs then rent independently as machines appear.
                fit = self._skus_fitting_now(skus) or skus
                if not self._provider_capacity(provider, fit, self._claimed_executors()):
                    waited = self._wait_for_funded_capacity(skus, describe="king rent",
                                                            for_king=True, capacity_probe=lambda choices: self._provider_capacity(provider, choices, self._claimed_executors()))
                    if waited == "operator":
                        # Owner-armed hybrid: the king trains on an operator
                        # lane instead of a JIT pod (operator-billed either way).
                        raise _FundedOperatorFallback("king")
                    if not waited:
                        _give_up(f"lium: no {sku} capacity before the round's latest safe start")
                try:
                    # Serialized with every funded rent (see _funded_rent_lock):
                    # the king's `lium up` must not race a challenger's for the
                    # same executor / create-rate window.
                    with allocation_lock(self._funded_rent_lock) as release_allocation:
                        claimed = live_claimed_executors(self)
                        fit = self._skus_fitting_now(skus) or skus
                        spec = LaunchSpec(sku=fit[0], count=1, image=rnd.funded_pod_image,
                                          ssh_pubkey=ssh_pubkey,
                                          name_prefix=name_prefix,
                                          gpus_per_pod=1, exclude_ids=claimed,
                                          sku_choices=fit if len(fit) > 1 else (),
                                          ttl_seconds=king_pod_ttl_seconds(self))
                        from .box_health import RentsHalted, rent_halt_refusal

                        halted = rent_halt_refusal(self, king=True, cached_only=True)
                        if halted:                        # before EVERY king launch (N3, D1)
                            raise RentsHalted(halted)
                        pod_id = provider.launch(spec)[0]
                        _ledger_king(pod_id)
                        sku_fn = getattr(provider, "sku_of", None)
                        landed = (str(sku_fn(pod_id) or "") if callable(sku_fn) else "")
                        sku = landed or fit[0]          # the ledger + floor use the landed type
                        # Claimed either way: a lemon executor must not be
                        # re-picked by the retry or by a sibling leg.
                        machine = provider.machine_of(pod_id) or ""
                        claim_executor(self, machine, pod_id)
                        if machine:
                            release_allocation()
                        if not provider.wait_ready(
                                pod_id, timeout=rnd.funded_ready_timeout_seconds):
                            _fail_king(pod_id, f"king pod {pod_id} not ready in time")
                        addr = provider.get_ip(pod_id)
                        if addr is None:
                            _fail_king(pod_id, f"king pod {pod_id} exposed no IP")
                        # Its real host: never a quarantined/blocklisted one
                        # (a blind listing can still land there).
                        from .host_preflight import host_blocked_reason

                        blocked = host_blocked_reason(self, addr.ip)
                        if blocked:
                            _fail_king(pod_id, f"king pod {pod_id} rejected: {blocked}")
                        try:
                            host_key = self._pin_king_host_key(pod_id, addr)
                        except ProvisionError as e:
                            _fail_king(pod_id, str(e))
                    _ledger_king(pod_id)
                    log.info("king pod %s ready at %s:%d (operator-billed, sku=%s, "
                             "host key pinned)", pod_id, addr.ip, addr.ssh_port, sku)
                    king_host = _king_remote(addr, host_key)
                    # JIT pod ⇒ no provisioner config push: deliver the deployed
                    # chain.toml (correct train_image_digest pin) before the probe.
                    king_host = self._push_deployed_chain_toml(king_host)
                    # Same runtime attestation as the funded legs: lium can boot
                    # a host-cached stale image under the degraded tag ref, and
                    # a stale KING pod trains under the wrong baked contract.
                    why = probe_worker_runtime(king_host, **({"required_flags": ("--local-only", "--continuation-root")} if getattr(self.cfg.round, "ops_continuation_interval_seconds", 0) else {}))
                    if why:
                        _fail_king(pod_id, f"king pod runtime rejected: {why}")
                    # Same throughput gate as the payer legs: the king's wall
                    # is the same as theirs, and a slow king host handicaps
                    # the defence of the throne. Fresh rents only — an
                    # adopted pod already holds its checkpoint.
                    why = self._host_bench_below_floor(king_host, sku, f"king pod {pod_id}")
                    if why:
                        _fail_king(pod_id, f"king pod rejected: {why}")
                    # Multi-GPU preflight: a king host that hangs or crawls an
                    # NCCL collective defends the throne at a fraction of its
                    # rate (the check quarantines the host itself).
                    from .host_preflight import multi_gpu_preflight_failure

                    why = multi_gpu_preflight_failure(self, king_host, sku, f"king pod {pod_id}")
                    if why:
                        _fail_king(pod_id, f"king pod rejected: {why}")
                    # The king pod also runs the era's public benches and the
                    # payer verifications: a driver too old for the bench
                    # sidecar burns every one of them.
                    why = host_cannot_bench(self, king_host)
                    if why:
                        _fail_king(pod_id, f"king pod rejected: {why}")
                except ProvisionError as e:
                    from .box_health import RentsHalted

                    if isinstance(e, RentsHalted):
                        continue                 # the loop top waits it out (R2)
                    from ..provision.core import OperatorRentsHeld

                    if isinstance(e, OperatorRentsHeld):
                        # The runway guard / daily cap holds EVERY operator
                        # rent, the king's too: wait at the poll cadence (no
                        # pod was created), give up at the latest safe start.
                        now_fn = getattr(self, "_rent_wait_now", None) or time.time
                        if now_fn() >= self._funded_rent_wait_deadline():
                            self._ledger_remove(f"{name_prefix}-0")
                            _give_up(f"king rent held at the latest safe start: {e}")
                        log.error("king rent held: %s", e)
                        sleep = getattr(self, "_rent_wait_sleep", None) or time.sleep
                        sleep(self.FUNDED_RENT_RETRY_SECONDS)
                        continue
                    # A pod that never came up / had no IP / ran the wrong
                    # runtime is a LEMON, not a verdict: it is torn down
                    # (_fail_king) and its executor stays excluded — rent again
                    # on another machine while the latest safe start allows.
                    now_fn = getattr(self, "_rent_wait_now", None) or time.time
                    if now_fn() >= self._funded_rent_wait_deadline():
                        _give_up(f"king rent gave up at the latest safe start "
                                 f"(attempt {attempt}): {e}")
                    log.warning("king rent attempt %d failed (%s); renting again on "
                                "another executor", attempt, e)
                    from ..funding.faults import classify_rent_failure

                    if classify_rent_failure(str(e)) in {"no_capacity", "auth", "rate_limited"}:
                        # Sold out (or only claimed executors listed): the next
                        # iteration's capacity check waits at the poll cadence
                        # — never spin on the marketplace API.
                        sleep = getattr(self, "_rent_wait_sleep", None) or time.sleep
                        sleep(self.FUNDED_RENT_RETRY_SECONDS)
                    continue
                _check_king_cancel(self)
                self._funded_king_host = king_host
                return self._funded_king_host

    KING_HOST_KEY_SCAN_TRIES = 6
    KING_HOST_KEY_SCAN_RETRY_SECONDS = 10.0

    def _pin_king_host_key(self, pod_id: str, addr) -> str:
        """``<keytype> <base64>`` of the king pod's sshd, scanned at readiness —
        the same pin a payer leg gets from ``rent_funded_pod`` (sshd may still
        be starting: a few retries). Raises ``ProvisionError`` when nothing
        scans, which the callers turn into a lemon teardown + fresh rent."""
        from ..provision.core import ProvisionError, scan_ssh_host_key

        scanner = getattr(self, "_host_key_scanner", None) or scan_ssh_host_key
        sleep = getattr(self, "_scan_retry_sleep", None) or time.sleep
        last: Exception | None = None
        for i in range(self.KING_HOST_KEY_SCAN_TRIES):
            try:
                key = str(scanner(addr.ip, addr.ssh_port) or "").strip()
                if key:
                    return key
            except Exception as e:  # noqa: BLE001 — retried below
                last = e
            if i + 1 < self.KING_HOST_KEY_SCAN_TRIES:
                sleep(self.KING_HOST_KEY_SCAN_RETRY_SECONDS)
        raise ProvisionError(f"king pod {pod_id}: could not pin its ssh host key "
                             f"at {addr.ip}:{addr.ssh_port} ({last})")

    def _teardown_operator_pod(self, pod) -> None:
        """Verified teardown of an operator-billed ledger pod (king JIT)."""
        from ..provision.core import LiumProvider
        from ..provision.funded import terminate_verified

        try:
            from ..provision.operator_pool import OperatorPool

            provider = OperatorPool(self) if pod.provider == OperatorPool.name else LiumProvider()
            from .money_guard import settle_operator_pod
            from .money_watch import note_teardown

            if terminate_verified(provider, pod.instance_id):
                settle_operator_pod(self, pod.instance_id, true_up=True)
                note_teardown(self, pod.instance_id, ok=True)
                release_executor_of(self, pod.instance_id)
                self._ledger_remove(pod.instance_id)       # may hit ENOSPC: pod already gone
            else:
                note_teardown(self, pod.instance_id, ok=False)
                log.error("operator pod %s still LIVE after terminate — kept "
                          "on ledger for the next sweep", pod.instance_id)
        except Exception as e:  # noqa: BLE001
            from .money_watch import note_teardown

            note_teardown(self, pod.instance_id, ok=False)
            log.error("operator pod %s teardown crashed (kept on ledger): %s",
                      pod.instance_id, e)

    # ── operator-rented legs (payer key expired, no operator lane) ──────────

    OPERATOR_LEG_POD_TAG = "op"
    ADOPT_PROBE_TRIES = 3
    ADOPT_PROBE_RETRY_SECONDS = 20.0

    def _operator_pod_worker_alive(self, addr) -> bool:
        """Whether the pod at ``addr`` still runs a detached worker (the
        sweep's own read-only probe). Unknown (unreachable) is NOT alive here:
        the caller already retried; a dead pod must not be kept forever."""
        try:
            from .remote import probe_pod_runs

            host = self._sweep_probe_host(addr.ip, getattr(addr, "ssh_port", 22) or 22)
            probe = probe_pod_runs(host, runner=getattr(self, "_sweep_ssh_runner", None),
                                   timeout=60)
            return bool(probe.reachable and (probe.live or probe.finished_ok))
        except Exception as e:  # noqa: BLE001
            log.warning("operator pod worker probe failed (%s)", e)
            return False

    def _operator_recovery_available(self, hotkey: str | None = None) -> bool:
        """Whether an operator-billed make-good could actually run: an operator
        lane on file or an allowed operator rental. Without either, flagging an
        entry strands it; the leg retries on the miner's key instead."""
        return bool(self._operator_fallback_lanes()) or self._operator_rent_allowed(hotkey)

    def _operator_rent_allowed(self, hotkey: str | None = None) -> bool:
        """True when an operator-billed leg with no lane on file may RENT an
        operator pod (money safety II, audit E — decoupled):

        * an owner approval (any kind) on file for ``hotkey``'s current leg
          ALWAYS allows it — an approved make-good never strands because
          ``funded_operator_fallback`` is off;
        * otherwise ``[round] funded_operator_fallback`` decides whether a
          non-miner fault may PROPOSE an operator make-good (the gate then
          files the approval request), for a calling leg that opted in.

        ``ops_paid_fallback_after_seconds`` no longer turns this on for every
        caller: the paid fallback is its own path (the capacity wait's
        ``operator_rent`` verdict). Rent mode only."""
        if self.cfg.round.funded_pods != "rent":
            return False
        if hotkey:
            from .money_guard import approval_on_file

            if approval_on_file(self, hotkey):
                return True
        loc = self.__dict__.get("_leg_local_obj")
        return bool(getattr(self.cfg.round, "funded_operator_fallback", False)
                    and loc is not None and getattr(loc, "operator_rent_ok", False))

    def _operator_leg_pod_prefix(self, round_id: str, hotkey: str) -> str:
        from ..provision.funded import funded_pod_name

        return (f"{funded_pod_name(str(round_id), hotkey, self.cfg.subnet.netuid)}"
                f"-{self.OPERATOR_LEG_POD_TAG}")

    def _is_operator_leg_pod_of(self, instance_id: str, hotkeys) -> bool:
        """``instance_id`` is an operator-rented leg pod of one of ``hotkeys``
        (the sweep keeps it while that leg is in flight)."""
        import re

        from ..provision.funded import _SLUG_RE

        m = re.match(rf"^cascade-n{int(self.cfg.subnet.netuid)}-\d+-funded-([a-z0-9]+)"
                     rf"-{self.OPERATOR_LEG_POD_TAG}(-|$)", str(instance_id))
        if not m:
            return False
        slugs = {_SLUG_RE.sub("", str(h).lower())[:12] for h in hotkeys}
        return m.group(1) in slugs

    def _rent_operator_leg_host(self, round_id: str, gen, *, name_prefix=None):
        """Rent ``gen``'s challenger leg pod on the OPERATOR's Lium account →
        ``(RemoteHost, pod_id)``. Same guards as every other rent: the round's
        SKU list (only types that still fit), the price caps, the CPU
        blocklist, claimed executors, the king's first pick, the runtime
        attestation and the host bench floor; a lemon is torn down and its
        executor stays excluded. Adopts a pod of the same name that is still
        up (a trainer restart mid-leg re-attaches to it). Raises
        ``_FundedLegSkip`` (unburned, ``no_capacity``/``infra``) when nothing
        usable can be had before the latest safe start."""
        from ..provision.core import LaunchSpec, ProvisionError
        from ..provision.funded import quarantine_lemon_host, terminate_verified
        from ..provision.state import PodInstance
        from .host_preflight import (
            host_blocked_reason,
            multi_gpu_preflight_failure,
            sync_host_blocklist,
        )
        from .remote import RemoteHost, probe_worker_runtime
        from .runner_funded_pods import allocation_lock

        sync_host_blocklist(self)
        rnd = self.cfg.round
        profile = self._funded_pod_profile()
        key_path = Path(profile.key_path or "").expanduser()
        ssh_pubkey = (key_path.parent / (key_path.name + ".pub")
                      ).read_text(encoding="utf-8").strip()
        skus = self._funded_skus_for_rent()
        from ..provision.operator_pool import operator_price_caps, operator_provider

        provider = operator_provider(self, lambda: _lium_provider(rnd, price_caps=operator_price_caps(self)),
                                     king=name_prefix is not None)
        prefix = name_prefix or self._operator_leg_pod_prefix(round_id, gen.hotkey)
        if name_prefix is not None:
            from ..provision.funded import funded_pod_name
            from ..provision.operator_pool import OperatorPool
            era_prefix = king_pod_name_prefix(self.cfg.subnet.netuid, str(round_id))
            payer_prefix = funded_pod_name(str(round_id), gen.hotkey, self.cfg.subnet.netuid)
            primary = [p for p in self._load_funded_ledger() if p.instance_id != prefix + '-0'
                       and (p.instance_id.startswith(era_prefix + '-replica-') or
                            p.payer_hotkey == gen.hotkey and p.instance_id.startswith(payer_prefix))]
            current = primary[-1] if primary else None
            provider_name = getattr(current, 'provider', '')
            if current is not None and provider_name == OperatorPool.name:
                row = provider._active(current.instance_id)
                provider_name = row['provider'] if row else ''
            alternatives = [p for p in rnd.ops_operator_providers if p != provider_name]
            if provider_name and alternatives:
                self._leg_local.operator_exclude_providers = (provider_name,)
            elif current is not None:
                skus = tuple(s for s in skus if s != current.sku)
                if not skus:
                    raise ProvisionError('no_capacity in an independent king backup failure domain')
            elif len(rnd.ops_operator_providers) > 1:
                self._leg_local.operator_exclude_providers = (rnd.ops_operator_providers[0],)
        pod_id = f"{prefix}-0"
        sku = skus[0] if skus else rnd.funded_pod_sku

        def _remote(addr, host_key: str) -> RemoteHost:
            # Credential-free and host-key pinned, exactly like the king pod:
            # the pod runs miner code, the checkpoint is harvested from here.
            return RemoteHost(
                name=f"operator-leg-{gen.hotkey[:12]}", host=addr.ip, port=addr.ssh_port,
                user=profile.user, key_path=profile.key_path,
                remote_python=profile.remote_python, workdir=profile.workdir,
                cuda_device=None, chain_toml=profile.chain_toml, forward_env=(),  # whole pod
                isolated=True, ssh_options=profile.ssh_options,
                pinned_host_key=host_key, stage="final", sku=sku,
                    allocation_id=pod_id, allocation_provider=provider.name,
                    allocation_incident=provider.incident_of(pod_id) if callable(getattr(provider, "incident_of", None)) else f"{provider.name}:{pod_id}")

        def _ledger(s: str) -> None:
            # payer_hotkey = "" is the operator marker: the startup sweep
            # tears it down through the operator path (kept while in flight).
            self._ledger_add(PodInstance(
                provider=provider.name, instance_id=pod_id, stage="funded",
                rented_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                sku=s, gpus=1, payer_hotkey=""))

        def _skip(msg: str, cls: str) -> None:
            self._record_funded_failure(gen.hotkey, msg, miner_fault=False,
                                        error_class=cls, burn=False)
            raise _FundedLegSkip(gen.hotkey)

        def _lemon(why: str) -> None:
            from .gpu_health import record_pod_failure
            with contextlib.suppress(Exception):
                record_pod_failure(self, provider, pod_id, sku, why,
                    kind='stale' if 'runtime' in why else 'slow' if 'rejected' in why else 'boot')
            quarantine_lemon_host(provider, pod_id, why)
            from .money_guard import settle_operator_pod

            settle_operator_pod(self, pod_id)            # booked at launch: its real cost
            try:
                if terminate_verified(provider, pod_id):
                    self._ledger_remove(pod_id)
                else:
                    log.error("operator leg pod %s still LIVE after terminate — kept "
                              "on ledger for the sweep", pod_id)
            except Exception as te:  # noqa: BLE001 — row stays for the sweep
                log.error("operator leg pod %s teardown failed (kept on ledger): %s",
                          pod_id, te)
            raise ProvisionError(why)

        live = getattr(provider, "live_pod_address", None)
        addr = None
        if live is not None:
            from .money_guard import operator_leg_allocated
            from .runner_common import probe_pod_live

            addr, definitive = probe_pod_live(live, pod_id,
                                              sleep=getattr(self, "_rent_wait_sleep", None))
            if (addr is None and not definitive and name_prefix is None
                    and operator_leg_allocated(self, round_id, gen.hotkey)):
                # A provider blip on a ledgered operator pod (review F1
                # residual): never park, never rent a second pod — requeue
                # transient with the row kept; the next try re-attaches.
                _skip("operator leg pod liveness unknown (provider blip) — re-attach retried, "
                      "nothing rented, not parked", "operator_transient")
        if addr is not None:
            log.warning("operator leg pod %s for %s is still LIVE at %s:%d — adopting it",
                        pod_id, gen.hotkey[:12], addr.ip, addr.ssh_port)
            # Money safety II (audit I): ONE failed probe used to tear a live
            # (possibly mid-training) operator pod down. Retried with backoff;
            # a pod whose worker is still alive is never terminated here.
            sleep = getattr(self, "_rent_wait_sleep", None) or time.sleep
            last = ""
            tries = int(getattr(self, "ADOPT_PROBE_TRIES", 3))
            for attempt in range(tries):
                try:
                    host = self._push_deployed_chain_toml(
                        _remote(addr, self._pin_king_host_key(pod_id, addr)))
                    why = probe_worker_runtime(host, **({"required_flags": ("--local-only", "--continuation-root")} if getattr(self.cfg.round, "ops_continuation_interval_seconds", 0) else {}))
                    if not why:
                        _ledger(sku)
                        return host, pod_id
                    last = f"runtime rejected: {why}"
                    break                         # a definite verdict: no retry
                except Exception as e:  # noqa: BLE001 — retried with backoff
                    last = str(e)
                    log.warning("adopting operator leg pod %s failed (try %d/%d: %s)",
                                pod_id, attempt + 1, tries, e)
                    if attempt + 1 < tries:
                        sleep(float(getattr(self, "ADOPT_PROBE_RETRY_SECONDS", 0.0)) * 2 ** attempt)
            if _OperatorPodMixin._operator_pod_worker_alive(self, addr):
                _skip(f"operator leg pod {pod_id} is live with a running worker but could not "
                      f"be adopted yet ({last[:160]}) — kept, re-attach retried, nothing rented",
                      "operator_transient")
            log.warning("adopted operator leg pod %s unusable (%s); renting fresh", pod_id, last)
            with contextlib.suppress(ProvisionError):
                _lemon("adopted operator leg pod unusable")

        from .cost_guards import measured_wall_refusal, provider_ttl_seconds
        from .runner_common import disk_guard_refusal

        unmeasured = measured_wall_refusal(rnd)
        if unmeasured:
            log.error("operator leg %s: %s", gen.hotkey[:12], unmeasured)
            _skip(unmeasured, "operator_blocked")
        # Disk guard (+ box-health halt + canary): a NEW leg pod only (the
        # adopt path returned above); a king backup (``name_prefix``) is held
        # only by a disk halt.
        if name_prefix is None:
            low = disk_guard_refusal(self)
        else:
            from .box_health import rent_halt_refusal

            low = rent_halt_refusal(self, king=True)
        if low:
            from ..funding.queue import OPERATOR_DISK_CLASS

            log.error("operator leg %s: %s", gen.hotkey[:12], low)
            _skip(low, OPERATOR_DISK_CLASS)
        from ..funding.queue import OPERATOR_BLOCKED_CLASS
        from .money_guard import OperatorBreaker, operator_leg_gate, take_deferred_gate

        deferred = take_deferred_gate(self) if name_prefix is None else None
        if deferred:
            # This leg skipped the approval because its operator pod was on
            # the ledger (F1) — but that pod is gone: a FRESH rent is gated.
            operator_leg_gate(self, gen, kind=deferred, rent=True)
        breaker = OperatorBreaker.for_runner(self)
        tripped = breaker.open_record() if name_prefix is None else None
        if tripped:
            # 2026-11-02: one 402 must pause every operator rent, not re-try
            # per leg every poll.
            _skip(f"operator rents paused (circuit breaker: {tripped.get('reason', '')}); "
                  "requeued, nothing rented, nothing burned", OPERATOR_BLOCKED_CLASS)
        if name_prefix is None:
            # Money safety II: the same doomed-leg gate as a payer rent — an
            # operator pod is never rented for a leg that can only land after
            # the settlement it must make (nothing rented, requeued unburned).
            from .cost_guards import doomed_leg_reason

            doomed = doomed_leg_reason(self, tuple(skus))
            if doomed:
                log.error("operator leg %s: %s", gen.hotkey[:12], doomed)
                _skip(doomed, "no_capacity")
        # A king backup (``name_prefix``) may become the era's king pod, which
        # lives for the era: the king's era TTL (0 = none; see the king rent).
        ttl = (provider_ttl_seconds(self, kind="leg") if name_prefix is None
               else king_pod_ttl_seconds(self))
        bad = 0
        while True:
            _check_king_cancel(self)
            if name_prefix is None:
                self._yield_to_king(f"operator leg {gen.hotkey[:12]}")
            fit = self._skus_fitting_now(skus) or skus
            if not self._provider_capacity(provider, fit, self._claimed_executors()):
                waited = self._wait_for_funded_capacity(
                    skus, describe=f"operator leg {gen.hotkey[:12]}", hotkey=gen.hotkey,
                    capacity_probe=lambda choices: self._provider_capacity(provider, choices, self._claimed_executors()))
                if waited is not True:
                    _skip("operator-billed leg: no GPU on the operator's account before "
                          "the latest safe start (requeued, operator-billed)", "no_capacity")
            from .box_health import rent_halt_refusal

            # the full check (it may probe) BEFORE the rent lock (D1)
            halted = rent_halt_refusal(self, king=name_prefix is not None)
            if halted:
                _skip(f"operator disk: {halted} — nothing rented", "operator_disk")
            try:
                with allocation_lock(self._funded_rent_lock) as release_allocation:
                    claimed = live_claimed_executors(self)
                    fit = self._skus_fitting_now(skus) or skus
                    halted = rent_halt_refusal(self, king=name_prefix is not None,
                                               cached_only=True)
                    if halted:                            # re-checked before EVERY launch (N3)
                        _skip(f"operator disk: {halted} — nothing rented", "operator_disk")
                    _ledger(fit[0])                       # write-ahead (deterministic name)
                    spec = LaunchSpec(sku=fit[0], count=1, image=rnd.funded_pod_image,
                                      ssh_pubkey=ssh_pubkey, name_prefix=prefix,
                                      gpus_per_pod=1, exclude_ids=claimed,
                                      sku_choices=fit if len(fit) > 1 else (),
                                      ttl_seconds=ttl)
                    pod_id = provider.launch(spec)[0]
                    _ledger(fit[0])
                    sku_fn = getattr(provider, "sku_of", None)
                    sku = (str(sku_fn(pod_id) or "") if callable(sku_fn) else "") or fit[0]
                    machine = provider.machine_of(pod_id) or ""
                    claim_executor(self, machine, pod_id)
                    if machine:
                        release_allocation()
                    if not provider.wait_ready(pod_id, timeout=rnd.funded_ready_timeout_seconds):
                        _lemon(f"operator leg pod {pod_id} not ready in time")
                    addr = provider.get_ip(pod_id)
                    if addr is None:
                        _lemon(f"operator leg pod {pod_id} exposed no IP")
                    # Its real host: never a quarantined/blocklisted one.
                    blocked = host_blocked_reason(self, addr.ip)
                    if blocked:
                        _lemon(f"operator leg pod {pod_id} rejected: {blocked}")
                    try:
                        host_key = self._pin_king_host_key(pod_id, addr)
                    except ProvisionError as e:
                        _lemon(str(e))
                _ledger(sku)
                host = self._push_deployed_chain_toml(_remote(addr, host_key))
                why = (probe_worker_runtime(host, **({"required_flags": ("--local-only", "--continuation-root")} if getattr(self.cfg.round, "ops_continuation_interval_seconds", 0) else {}))
                       or self._host_bench_below_floor(host, sku, f"operator leg pod {pod_id}")
                       or multi_gpu_preflight_failure(self, host, sku,
                                                      f"operator leg pod {pod_id}"))
                if why:
                    _lemon(f"operator leg pod rejected: {why}")
                log.warning("operator leg pod %s ready at %s:%d for %s (sku=%s) — "
                            "operator-billed (payer key expired)", pod_id, addr.ip,
                            addr.ssh_port, gen.hotkey[:12], sku)
                _check_king_cancel(self)
                if name_prefix is None:
                    # A usable pod is up: only now is the owner's approval used
                    # for this attempt and the operator spend booked (review F3)
                    # — a sold-out market, a lemon or a rent error keeps both.
                    from .money_guard import note_operator_launch

                    price_fn = getattr(provider, "price_of", None)
                    try:
                        landed = price_fn(pod_id) if callable(price_fn) else None
                    except Exception:  # noqa: BLE001 — unknown: the projection is booked
                        landed = None
                    if not note_operator_launch(self, gen.hotkey, landed_rate=landed,
                                                pod_id=pod_id):
                        # the landed cost broke the approval's ceiling (money
                        # safety II): release the pod (not a lemon — the host
                        # is fine) and park for the owner's re-approval
                        from ..funding.queue import APPROVAL_CLASS
                        from .money_guard import settle_operator_pod

                        settle_operator_pod(self, pod_id)
                        if terminate_verified(provider, pod_id):
                            self._ledger_remove(pod_id)
                        _skip("landed price above the approved ceiling — pod released, "
                              "awaiting the owner's re-approval (nothing burned)",
                              APPROVAL_CLASS)
                return host, pod_id
            except ProvisionError as e:
                from ..provision.core import OperatorRentsHeld

                if isinstance(e, OperatorRentsHeld):
                    # runway guard / daily cap: nothing was created — never a
                    # bad pod, never a retry loop against an empty account
                    self._ledger_remove(pod_id)
                    _skip(f"operator rents held: {e}", OPERATOR_BLOCKED_CLASS)
                bad += 1
                tripped = breaker.open_record() if name_prefix is None else None
                if tripped:
                    # A balance refusal (this rent or another) opened the
                    # breaker: stop now — no retry against an empty account.
                    _skip(f"operator rents paused (circuit breaker: "
                          f"{tripped.get('reason', '')}); last: {e}", OPERATOR_BLOCKED_CLASS)
                now_fn = getattr(self, "_rent_wait_now", None) or time.time
                if (bad >= self.FUNDED_MAX_STALE_PODS
                        or now_fn() >= self._funded_rent_wait_deadline()):
                    from ..funding.faults import classify_rent_failure

                    # Our account, our pods: a rent refusal is transient on our
                    # side — backed off (never the flat "infra" loop of
                    # 2026-11-02) and never counted toward the fault cap.
                    cls = "no_capacity" if classify_rent_failure(str(e)) == "no_capacity" \
                        else "operator_transient"
                    _skip(f"operator-billed leg: {bad} bad pod(s), last: {e}", cls)
                log.warning("operator leg %s: rent attempt %d failed (%s); renting again",
                            gen.hotkey[:12], bad, e)
                sleep = getattr(self, "_rent_wait_sleep", None) or time.sleep
                sleep(self.FUNDED_RENT_RETRY_SECONDS)

    def _run_operator_rented_leg(self, disp, gen, seeds, block: int, contract, suffix: str,
                                 *, warm_start_ref: str | None,
                                 contract_block: int | None = None,
                                 router_slow: bool = False, role="challenger",
                                 domain=None, expert_selection=None, finetune_recipe=None):
        """Rent (or adopt) an operator pod → stage the vault ZIP → dispatch the
        challenger leg → tear the pod down, whatever happened. The checkpoint
        is harvested by the rolling dispatcher (credential-free pod)."""
        from ..funding.store import parse_vault_ref
        from ..provision.state import PodInstance
        from .money_guard import operator_leg_allocated, would_rent_refusal
        from .runner_funded_legs import _record_generation_failure

        # Money safety II: the money gates as a side-effect-free "would it
        # rent?" verdict BEFORE the corpus — no CPU VM is paid for a leg
        # that cannot rent (nothing rented, requeued unburned).
        why, cls = (("", "") if operator_leg_allocated(self, str(seeds.base_seed), gen.hotkey)
                    else would_rent_refusal(self, gen, operator=True))
        if why:
            log.warning("operator leg %s: %s — no corpus generated", gen.hotkey[:12], why)
            self._record_funded_failure(gen.hotkey, why, miner_fault=False, error_class=cls,
                                        burn=False)
            raise _FundedLegSkip(gen.hotkey)
        try:
            corpus = leg_corpus(self, gen, seeds, contract)  # DEC-CA-0053, before the rent
        except Exception as e:  # noqa: BLE001 — classify, record, re-raise
            _record_generation_failure(self, gen, e)
            raise
        try:
            host, pod_id = self._rent_operator_leg_host(str(seeds.base_seed), gen)
        except _FundedLegSkip:
            from .money_guard import release_reservation

            release_reservation(self)            # no rent: free the daily-cap headroom
            raise
        _loc = self.__dict__.get("_leg_local_obj")
        if _loc is not None:
            _loc.leg_trained = False           # set by the harvest (harvest_guard)

        def ledger_pod():
            pods = getattr(self, "_load_funded_ledger", lambda: [])()
            pod = next((p for p in pods if p.instance_id == pod_id), None)
            if pod is None:
                from ..provision.operator_pool import OperatorPool, enabled
                pooled = hasattr(self, "cfg") and enabled(self)
                pod = PodInstance(provider=OperatorPool.name if pooled else "lium",
                                  instance_id=pod_id, stage="funded", rented_at_iso="")
            return pod

        released: list[bool] = []

        def release_pod() -> None:
            # Owner 2026-11-01: never benched here and torn down at leg end
            # anyway — once the harvested copy is verified on the box, stop
            # the operator's bill before the upload (harvest_release).
            if not released:
                # DEC-CA-0063: the leg is harvested (its worker exited) — a
                # parked random-init build of this seed epoch runs here first;
                # its thread then tears the pod down
                hook = getattr(self, "_rolling_init_on_leg_pod", None)
                try:
                    claimed = bool(callable(hook) and hook(
                        host, seeds.base_seed, lambda: self._teardown_operator_pod(ledger_pod())))
                except Exception as e:  # noqa: BLE001 — never keep the pod over it
                    log.warning("operator leg %s: init build hook failed: %s",
                                gen.hotkey[:12], e)
                    claimed = False
                if not claimed:
                    self._teardown_operator_pod(ledger_pod())
                released.append(True)

        # A challenger receipt is harvested INSIDE dispatch (the isolated
        # host's harvest hook, same thread): armed for the whole leg body.
        from .harvest_release import arm_release, disarm_release

        armed = arm_release(self, release_pod)
        try:
            from .progress import report_progress

            digest = parse_vault_ref(gen.ref)
            if digest is not None:
                host = self._stage_vault_zip_on(host, digest)
            report_progress(self, "training")
            specialist = role in {"expert", "mkt_finetune"}
            # The same progress watchdog (throughput / first-step deadline)
            # as a payer leg; an abort quarantines the host and is ours.
            from types import SimpleNamespace

            from .runner_funded_legs import _dispatch_watched, _with_leg_watchdog

            disp = _with_leg_watchdog(
                self, disp, gen, SimpleNamespace(instance_id=pod_id, payer_hotkey=""),
                contract, host)
            receipt = _dispatch_watched(
                disp, host, lane_count=1, gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey,
                role=role, base_seed=seeds.base_seed, block=int(block),
                arch_preset=contract.arch_preset, warm_start_ref=warm_start_ref,
                contract_block=contract_block, **corpus,
                **router_kw(router_slow),
                **({"repo_suffix": suffix} if suffix else {}),
                **({"local_checkpoint": True, "domain": domain, "expert_selection": expert_selection,
                    "finetune_recipe": finetune_recipe} if specialist else {}),
            )
            if specialist:
                report_progress(self, "harvesting")
                # The armed release (above) lets the pod go once the
                # checkpoint is verified — on the box (ssh) or on R2 (r2
                # transport) — before the Hippius push.
                return self._harvest_funded_checkpoint(host, receipt, contract, seeds, suffix)
            return receipt
        except _FundedTamper as e:
            # The harvest hook's ingest guard: the same verdict as a payer leg.
            self._record_funded_failure(gen.hotkey, f"checkpoint: {e}", miner_fault=True,
                                        error_class="tamper", burn=False)
            raise
        except _FundedDiverged as e:
            self._record_funded_failure(gen.hotkey, f"{DIVERGED_MARKER}: {e}",
                                        miner_fault=True, error_class="generator",
                                        burn=False)
            raise
        except Exception as e:  # noqa: BLE001 — classify, record, re-raise
            from .harvest_guard import handle_our_side_fault

            loc = self.__dict__.get("_leg_local_obj")
            if handle_our_side_fault(self, gen, e, pod_id=pod_id, payer=False,
                                     trained=bool(getattr(loc, "leg_trained", False)),
                                     released=bool(released)):
                # OUR disk failed with the run complete on this pod: keep it
                # (ledgered; the requeued leg adopts it by name and harvests).
                released.append(False)
                raise
            # An OPERATOR pod: the miner controls nothing on it, so only the
            # worker's own rc 3 (their generator) is theirs; anything else is
            # ours and never burns one of their attempts.
            q = self._funded_queue()
            prior = q.get(gen.hotkey) if q is not None else None
            miner_fault, error_class, _burn = classify_leg_exception(
                self, gen, seeds, e, rc=getattr(e, "returncode", None), text=str(e),
                stalled_before=bool(prior is not None
                                    and prior.last_error_class == STALL_CLASS),
                shards_leg=bool(corpus.get("corpus_files")))
            if error_class == "infra":
                # infra on OUR pod (it died, transport): transient (review F4)
                error_class = "operator_transient"
            self._record_funded_failure(gen.hotkey, str(e), miner_fault=miner_fault,
                                        error_class=error_class, burn=False)
            raise
        finally:
            disarm_release(self, armed)
            if not released:
                self._teardown_operator_pod(ledger_pod())
