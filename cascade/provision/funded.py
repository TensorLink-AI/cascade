"""Per-payer rentals: one pod, one challenger, one miner's provider account.

The stage fleets (``loop.ProvisionerLoop``) rent homogeneous batches on the
OPERATOR's account. Funded challenger legs (DEC-CA-0036) are the opposite
shape: each pod is billed to its miner's own key, so every rental gets its own
:class:`~cascade.provision.core.LiumProvider` bound to that key — the port of
PRISM's per-submission backend (``PayerBackendFactory.resolve``). Three
consequences this module owns:

* **Classification, not judgement.** A funded rent that fails is classified
  (:mod:`cascade.funding.faults`) and reported; whether it burns an attempt is
  the queue's call (`burn_attempt` follows the taxonomy: only ``infra`` does).
  Nothing here touches the trainer's submission-burn machinery.
* **Teardown needs the payer's key.** A pod on a miner's account is invisible
  to (and unterminatable by) the operator's key. Every result carries the
  payer hotkey precisely so restart/reap paths can hydrate the right key from
  the vault first; :func:`reconcile_funded` is the reaper's per-payer twin.
* **A rent is never auto-retried here.** Each ``lium up`` attempt spends the
  MINER's budget (base's client excludes ``/rent`` from its backoff for the
  same reason) — retry cadence belongs to the caller, on queue cooldowns.

The king's leg, the confirmation leg, and every eval pod stay on the
operator's account and the existing stage machinery — this module must never
grow a path that rents operator-billed pods.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from ..funding.faults import classify_rent_failure
from ..funding.vault import PayerKeyVault
from .core import (
    LaunchSpec,
    LiumProvider,
    PodAddress,
    Provider,
    ProvisionError,
    UncertainCreateError,
    record_host_quarantine,
)
from .state import PodInstance

__all__ = [
    "FundedRentResult",
    "funded_pod_name",
    "lium_provider_for_key",
    "payer_pod_pattern",
    "reconcile_funded",
    "rent_funded_pod",
    "teardown_funded",
    "terminate_verified",
]

log = logging.getLogger("cascade.provision.funded")

FUNDED_STAGE = "funded"

_SLUG_RE = re.compile(r"[^0-9a-z]")


def funded_pod_name(round_id: str, hotkey: str, netuid: int) -> str:
    """``cascade-n<netuid>-<round>-funded-<slug>``: deployment-scoped, payer-attributable.

    The slug is the hotkey's first 12 chars lowercased (SS58 is alnum, so this
    is stable and collision-safe within any real field). The ``n<netuid>``
    token scopes the name to THIS deployment: a miner may fund a testnet and
    a mainnet cascade from ONE Lium account, and without the discriminator
    either side's :func:`reconcile_funded` sweep would read the other's live
    leg as an off-ledger orphan and kill it mid-round (review 2026-09-02).
    Deliberately NOT the provisioner's ``cascade-<round>-<stage>`` scheme —
    funded pods are the trainer's, ledgered in ``funded_pods.json``; the
    provisioner's orphan reaper must never consider them its own.
    """
    slug = _SLUG_RE.sub("", hotkey.lower())[:12]
    if not slug:
        raise ProvisionError(f"cannot derive a pod slug from hotkey {hotkey!r}")
    return f"cascade-n{int(netuid)}-{round_id}-funded-{slug}"


def lium_provider_for_key(api_key: str) -> Provider:
    """A Lium adapter billing ``api_key``'s account (the default factory)."""
    if not api_key:
        raise ProvisionError("funded rental needs the payer's api key (empty)")
    return LiumProvider(api_key=api_key)


# Marketplaces a miner may fund with. Each payer key scopes rentals to that
# miner's own account; see cascade.provision.{vast,runpod} for the key scopes.
COMPUTE_PROVIDERS = ("lium", "vast", "runpod")


def provider_for_key(provider: str, api_key: str) -> Provider:
    """An adapter for ``provider`` billing ``api_key``'s account."""
    if not api_key:
        raise ProvisionError("funded rental needs the payer's api key (empty)")
    if provider == "lium":
        from .core import LiumProvider as PayerLiumProvider

        return PayerLiumProvider(api_key=api_key)
    if provider == "shadeform":
        from .core import ShadeformProvider

        return ShadeformProvider(api_key=api_key)
    if provider == "vast":
        from .vast import VastProvider

        return VastProvider(api_key=api_key)
    if provider == "runpod":
        from .runpod import RunPodProvider

        return RunPodProvider(api_key=api_key)
    raise ProvisionError(f"unknown compute provider {provider!r}")


def apply_price_caps(provider: Provider, *, max_price_per_hour: float = 0.0,
                     max_leg_cost_usd: float = 0.0,
                     sku_wall_seconds: tuple[tuple[str, int], ...] = (),
                     default_wall_seconds: float = 0.0) -> Provider:
    """Hand the ``[round]`` price guards (and the per-SKU wall table behind
    the per-leg cap) to a provider with the seam (``LiumProvider``); others
    are returned untouched."""
    if hasattr(provider, "max_price_per_hour"):
        provider.max_price_per_hour = float(max_price_per_hour or 0.0)
        provider.max_leg_cost_usd = float(max_leg_cost_usd or 0.0)
        provider.sku_wall_seconds = tuple(sku_wall_seconds or ())
        provider.default_wall_seconds = float(default_wall_seconds or 0.0)
    return provider


def apply_cpu_blocklist(provider: Provider, blocklist: tuple[str, ...]) -> Provider:
    """Hand ``[round] funded_cpu_blocklist`` to a provider that can steer its
    listings by CPU model (``LiumProvider.cpu_blocklist``); providers without
    the seam (other clouds, test fakes) are returned untouched."""
    if blocklist and hasattr(provider, "cpu_blocklist"):
        provider.cpu_blocklist = tuple(blocklist)
    return provider


def terminate_verified(provider: Provider, pod_id: str) -> bool:
    """Terminate and CONFIRM by re-listing; True only when the pod is gone.

    ``LiumProvider.terminate`` deliberately swallows a failed ``lium rm`` as
    already-terminated (operator-fleet idempotency) — on a revoked miner key
    that turns a 401 into silence, so every funded-path teardown must believe
    the listing, not the call (review 2026-08-29). A provider without
    ``list_tagged`` cannot be re-checked; the call's own success is then the
    best evidence available. A provider that can say its view is incomplete
    (``create_unresolved``: Lium's never-listed ``lium up``, Vast's label never
    observed and no DELETE acknowledged) is never believed on an empty listing
    alone.
    """
    unresolved = getattr(provider, "create_unresolved", lambda _pid: False)
    uncertain = unresolved(pod_id)
    provider.terminate(pod_id)
    if uncertain and unresolved(pod_id):
        return False
    present = getattr(provider, "pod_exists", None)
    if present is not None:
        return not present(pod_id)
    lister = getattr(provider, "list_tagged", None)
    if lister is None:
        return True
    return pod_id not in set(lister(pod_id))


# A pod that the platform accepted but never brought up (or brought up wrong)
# is a LEMON: the host's fault, never the payer's — it neither burns one of the
# miner's attempts nor ends the leg; the caller rents again elsewhere while the
# round's latest safe start allows. Its host is quarantined so the retry (and
# every sibling leg) lands on a different machine, not the same one under
# another executor id (2026-09-13: 91.224.44.222 × 4 ids).
LEMON_CLASS = "lemon"


class LemonPodError(ProvisionError):
    """The rented pod never became usable — a host fault, not a verdict."""


def quarantine_lemon_host(provider: Provider, pod_id: str, reason: str) -> str:
    """Quarantine the host ``pod_id`` sits on (best-effort; "" when unknown).
    Call BEFORE tearing the pod down — its record is how the host is found."""
    finder = getattr(provider, "host_of_pod", None)
    host = ""
    if callable(finder):
        try:
            host = str(finder(pod_id) or "")
        except Exception as e:  # noqa: BLE001 — a lookup failure must not mask the lemon
            log.warning("lium: could not resolve the host of lemon pod %s: %s", pod_id, e)
    if host:
        record_host_quarantine(host, f"{pod_id}: {reason}")
    else:
        log.warning("lium: lemon pod %s has no resolvable host — nothing quarantined",
                    pod_id)
    return host


@dataclass(frozen=True)
class FundedRentResult:
    """One funded rent attempt, success or classified failure."""

    hotkey: str
    ok: bool
    pod: PodInstance | None = None
    address: PodAddress | None = None
    error: str = ""
    error_class: str = ""           # cascade.funding.faults class; "" on success
    # Marketplace machine id the pod landed on ("" when unknown): concurrent
    # funded rents feed these back as ``exclude_ids`` so N challengers claim N
    # DISTINCT executors instead of racing for the listing's first row.
    machine_id: str = ""
    adopted: bool = False           # a live pod of this name was reused, not rented
    burn_attempt: bool = False      # True only for "infra" (the taxonomy's rule)
    # Platform identity of the pod that answered (Lium pod id + huid) and the
    # container's SSH host key, both read once the pod is READY: the trainer
    # pins them for the leg and re-checks before dispatch and at return, so
    # a pod the payer replaced under the same name is caught as tamper.
    pod_uid: str = ""
    host_key: str = ""              # "<keytype> <base64>" from ssh-keyscan
    # A half-launched pod the failure-path cleanup could NOT confirm dead —
    # billing the MINER until someone acts. The caller must surface it (queue
    # error text, operator alert), never drop it on the floor.
    leaked_pod: str = ""
    # GPU type the pod landed on (open-market rents choose per leg); ``sku``
    # as requested when the adapter cannot tell.
    sku: str = ""


def _log_key_cannot_stop(hotkey: str, pod_name: str, err: Exception, api_key: str) -> None:
    """``PAYER KEY CANNOT STOP POD`` (2026-10-31 'Invalid API key'): the one
    line ops grep to tell a miner their key no longer controls their pod."""
    msg = str(err)
    if api_key:
        msg = msg.replace(api_key, "<redacted>")
    log.error("PAYER KEY CANNOT STOP POD: hotkey=%s pod=%s — the vaulted key cannot list "
              "pods on the payer's account (%s); no new rental on this key", hotkey,
              pod_name, msg[-300:])


def rent_funded_pod(
    *,
    round_id: str,
    hotkey: str,
    api_key: str,
    sku: str,
    image: str,
    ssh_pubkey: str,
    netuid: int = 0,
    gpus_per_pod: int = 1,
    ready_timeout: float = 900.0,
    exclude_ids: tuple[str, ...] = (),
    cpu_blocklist: tuple[str, ...] = (),
    skus: tuple[str, ...] = (),
    price_caps: dict | None = None,
    provider_factory: Callable[[str], Provider] = lium_provider_for_key,
    now_iso: Callable[[], str] = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    host_key_scanner: Callable[[str, int], str] | None = None,
    on_allocated: Callable[[str, str], None] | None = None,
    vault: PayerKeyVault | None = None,
    on_provider_selected: Callable[[str, str], None] | None = None,
    existing_only: bool = False,
    on_create_started: Callable[[], None] | None = None,
    gpu_health=None,
    ttl_seconds: float = 0.0,
) -> FundedRentResult:
    """Rent ONE pod for ``hotkey``'s challenger leg on ``hotkey``'s own key.

    Success returns the ledgered :class:`PodInstance` (stage ``"funded"``,
    ``payer_hotkey`` set — the caller appends it to the round ledger and saves
    BEFORE using the pod, same write-ahead rule as the stage fleets) plus its
    SSH address. Failure tears down anything half-launched (with the payer's
    key), classifies the error, and reports whether the attempt should burn.
    """

    from ..funding.providers import PREFIX
    from .layout import layout_gpus

    if api_key.startswith(PREFIX) or (vault is not None and provider_factory is lium_provider_for_key):
        from .payer_pool import rent_from_pool

        return rent_from_pool(dict(
            round_id=round_id, hotkey=hotkey, api_key=api_key, sku=sku, image=image,
            ssh_pubkey=ssh_pubkey, netuid=netuid, gpus_per_pod=gpus_per_pod,
            ready_timeout=ready_timeout, exclude_ids=exclude_ids,
            cpu_blocklist=cpu_blocklist, skus=skus, price_caps=price_caps,
            now_iso=now_iso, host_key_scanner=host_key_scanner, on_allocated=on_allocated,
            vault=vault, on_provider_selected=on_provider_selected,
            gpu_health=gpu_health, ttl_seconds=ttl_seconds), rent_funded_pod)

    def _fail(err: Exception | str, *, leaked_pod: str = "") -> FundedRentResult:
        msg = str(err)
        if api_key:
            msg = msg.replace(api_key, "<redacted>")
        cls = "infra" if isinstance(err, UncertainCreateError) else classify_rent_failure(msg)
        log.warning("funded rent for %s failed [%s]: %s", hotkey, cls, msg[-300:])
        return FundedRentResult(
            hotkey=hotkey, ok=False, error=msg[-500:], error_class=cls,
            burn_attempt=(cls == "infra"), leaked_pod=leaked_pod,
        )

    credential_id = ""
    name = funded_pod_name(round_id, hotkey, netuid)
    try:
        provider = apply_cpu_blocklist(provider_factory(api_key), cpu_blocklist)
        provider = apply_price_caps(provider, **(price_caps or {}))
        if gpu_health is not None and gpu_health.threshold:
            from ..trainer.gpu_health import GuardedProvider

            provider = GuardedProvider(provider, gpu_health)
        if vault is not None:
            from ..funding.providers import PayerCredential, lease_credentials

            credential_id = lease_credentials(vault, hotkey, PayerCredential(provider.name, api_key), scope=name)
            if on_provider_selected is not None:
                on_provider_selected(provider.name, credential_id)
    except Exception as e:  # noqa: BLE001 — a bad key must classify, not crash the round
        return _fail(e)

    # ``skus`` (open market, owner 2026-09-20): the adapter picks the cheapest
    # fitting executor across the types; ``sku`` stays the nominal first pick.
    # A pod by this name already RUNNING on the payer's account is THIS leg's
    # pod from a prior attempt (a trainer restart, a retry after a transport
    # drop): adopt it — the worker on it is still training or has finished
    # (the dispatcher attaches to its run). Renting again created a duplicate
    # under the same name and the identity pin then saw the booting twin
    # (2026-09-24: four legs burned as "tamper", three pods per leg rented
    # and torn down). Adoption never launches; a failed probe rents as before.
    # ``name`` is the launch PREFIX; the provider names the pod ``<name>-<i>``
    # (2026-09-24 12:46: probing the bare prefix found nothing and seven twins
    # were rented — list the prefix, then ask each listed pod).
    live_fn = getattr(provider, "live_pod_address", None)
    live_name, live_addr = "", None
    if callable(live_fn):
        try:
            candidates = sorted(provider.list_tagged(name))
            if len(candidates) > 1:
                return replace(_fail("multiple allocations need reconciliation; no new rental", leaked_pod=candidates[0]), burn_attempt=False)
            for cand in candidates:
                addr = live_fn(cand)
                if addr is not None:
                    live_name, live_addr = cand, addr
                    break
            if candidates and live_addr is None:
                return replace(_fail("existing allocation is not ready; no new rental", leaked_pod=candidates[0]), burn_attempt=False)
        except Exception as e:  # noqa: BLE001 — a listing blip must not block the rent
            # This listing is also the teardown-key check: it is the first
            # call of every verified teardown, so a key that cannot list
            # cannot stop the pod it would rent — nothing is rented.
            log.warning("funded rent for %s: existing allocation cannot be inspected; no new rental", hotkey)
            _log_key_cannot_stop(hotkey, name + "-0", e, api_key)
            return replace(_fail(e, leaked_pod=name + "-0"), burn_attempt=False)
    if live_addr is not None:
        try:
            ident_fn = getattr(provider, "pod_identity", None)
            ident = ident_fn(live_name) if callable(ident_fn) else None
            pod_uid = str((ident or {}).get("id") or "")
            if ident_fn is not None and not pod_uid:
                raise ProvisionError(f"funded pod {live_name}: platform identity unavailable")
            host_key = ""
            if host_key_scanner is not None:
                host_key = host_key_scanner(live_addr.ip, live_addr.ssh_port)
            sku_fn = getattr(provider, "sku_of", None)
            landed = (str(sku_fn(live_name) or "") if callable(sku_fn) else "") or (
                skus[0] if skus else sku)
            pod = PodInstance(
                provider=provider.name, instance_id=live_name, stage=FUNDED_STAGE,
                rented_at_iso=now_iso(), sku=landed, gpus=max(gpus_per_pod, layout_gpus(landed)),
                payer_hotkey=hotkey, pod_uid=pod_uid, payer_credential_id=credential_id,
            )
            log.warning("funded pod %s is still LIVE at %s:%d from a prior attempt — "
                        "adopting it for %s (no rent; the dispatcher attaches to its run)",
                        live_name, live_addr.ip, live_addr.ssh_port, hotkey)
            machine = ""
            getter = getattr(provider, "machine_of", None)
            if getter is not None:
                machine = getter(live_name) or ""
            return FundedRentResult(hotkey=hotkey, ok=True, pod=pod, address=live_addr,
                                    machine_id=machine, pod_uid=pod_uid,
                                    host_key=host_key, sku=landed, adopted=True)
        except Exception as e:  # noqa: BLE001 — never tear the live pod down; retry later
            res = _fail(f"live pod {live_name} could not be adopted: {e}", leaked_pod=live_name)
            return replace(res, error_class="infra", burn_attempt=False)

    if existing_only:
        # The pool's probe listed an allocation moments ago and this re-list
        # did not: a listing flap, not a vanished pod. The allocation is still
        # the leg's (``leaked_pod`` keeps its lease and create intent — the
        # pool forgets both on a bare failure, and the next rent then created
        # a twin under the same name).
        return FundedRentResult(hotkey, False,
                                error="existing allocation is booting or cannot yet be adopted; no new rental",
                                error_class="infra", burn_attempt=False, leaked_pod=name + "-0")

    choices = tuple(skus) if len(tuple(skus)) > 1 else ()
    spec = LaunchSpec(
        sku=(skus[0] if skus else sku), count=1, image=image, ssh_pubkey=ssh_pubkey,
        name_prefix=name, gpus_per_pod=gpus_per_pod, exclude_ids=exclude_ids,
        sku_choices=choices, ttl_seconds=float(ttl_seconds or 0.0),
    )
    launched: list[str] = []
    launch_attempted = False
    try:
        if on_create_started is not None:
            try:
                on_create_started()
            except Exception:
                return FundedRentResult(hotkey, False, error="cannot persist rental intent; no create sent",
                                        error_class="infra", burn_attempt=False)
        launch_attempted = True
        launched = provider.launch(spec)
        pod_id = launched[0]
        if on_allocated is not None:
            getter = getattr(provider, "machine_of", None)
            machine = str(getter(pod_id) or "") if callable(getter) else ""
            on_allocated(machine, pod_id)
        if not provider.wait_ready(pod_id, timeout=ready_timeout):
            # Surface the captured `lium up` output INTO the error: launch is
            # fire-and-forget, so a key revoked (or balance exhausted) between
            # `ls` and `up` otherwise reads as a generic timeout → classified
            # "infra" → burns a miner-fixable fault (review 2026-09-02). The
            # taxonomy classifies on this text; _fail scrubs the key from it.
            tail = ""
            tail_fn = getattr(provider, "_up_log_tail", None)
            if callable(tail_fn):
                tail = tail_fn(pod_id)
            why = (f"funded pod {pod_id} not ready within {ready_timeout:.0f}s"
                   + (f"; lium up said: {tail}" if tail else ""))
            # The platform took the rent and delivered nothing usable: a lemon
            # host (2026-09-13: RUNNING with no ports for 900 s, four ids on
            # one machine). Remember the HOST before the pod record vanishes.
            if classify_rent_failure(why) in {"auth", "rate_limited"}:
                raise ProvisionError(why)
            quarantine_lemon_host(provider, pod_id, why)
            raise LemonPodError(why)
        addr = provider.get_ip(pod_id)
        if addr is None:
            raise ProvisionError(f"funded pod {pod_id} exposed no IP")
        # Identity pins (fail CLOSED: a leg we cannot pin is a leg we cannot
        # trust — it skips as infra, unburned, rather than run unpinned).
        ident = None
        ident_fn = getattr(provider, "pod_identity", None)
        if callable(ident_fn):
            ident = ident_fn(pod_id)
        pod_uid = str((ident or {}).get("id") or "")
        if ident_fn is not None and not pod_uid:
            raise ProvisionError(f"funded pod {pod_id}: platform identity unavailable")
        host_key = ""
        if host_key_scanner is not None:
            last = None
            for _ in range(6):
                try:
                    host_key = host_key_scanner(addr.ip, addr.ssh_port)
                    break
                except Exception as e:  # noqa: BLE001 — sshd may still be starting
                    last = e
                    time.sleep(10)
            if not host_key:
                raise ProvisionError(f"funded pod {pod_id}: could not pin its "
                                     f"ssh host key ({last})")
        sku_fn = getattr(provider, "sku_of", None)
        landed = (str(sku_fn(pod_id) or "") if callable(sku_fn) else "") or spec.sku
        pod = PodInstance(
            provider=provider.name, instance_id=pod_id, stage=FUNDED_STAGE,
            rented_at_iso=now_iso(), sku=landed, gpus=max(gpus_per_pod, layout_gpus(landed)),
            payer_hotkey=hotkey, pod_uid=pod_uid, payer_credential_id=credential_id,
        )
        log.info("funded pod %s ready for %s at %s:%d (%s, billed to payer)",
                 pod_id, hotkey, addr.ip, addr.ssh_port, landed)
        machine = ""
        getter = getattr(provider, "machine_of", None)
        if getter is not None:
            machine = getter(pod_id) or ""
        return FundedRentResult(hotkey=hotkey, ok=True, pod=pod, address=addr,
                                machine_id=machine, pod_uid=pod_uid,
                                host_key=host_key, sku=landed)
    except Exception as e:  # noqa: BLE001 — classify everything; the taxonomy decides
        leaked = ""
        for pid in launched:
            try:
                if not terminate_verified(provider, pid):
                    raise ProvisionError("still listed live after terminate")
            except Exception as te:  # noqa: BLE001 — record loudly; never silent
                log.error("funded cleanup of %s could NOT confirm teardown — pod "
                          "may be LEAKED on payer %s's account (revoked key?): %s",
                          pid, hotkey, te)
                leaked = pid
        if launch_attempted and not launched:
            # ``launch`` raised: did a create reach the provider? An adapter
            # that tracks its requests (``create_requested``) answers exactly —
            # an offer search or template step failing BEFORE any create used
            # to leave the write-ahead intent on disk for good (no listing can
            # resolve a create that was never sent) and blocked the leg name
            # until the epoch ended. Without the seam, any prose that is not a
            # definite refusal reads as uncertain (fail closed).
            requested = getattr(provider, "create_requested", None)
            if callable(requested):
                sent = isinstance(e, UncertainCreateError) or bool(requested(name + "-0"))
            else:
                sent = (isinstance(e, UncertainCreateError)
                        or classify_rent_failure(str(e)) not in {"auth", "no_capacity", "rate_limited"})
            if sent:
                # A lost create response is not proof of rejection. Retain the
                # intent and credential so recovery can find the allocation by name.
                leaked = name + "-0"
        if credential_id and vault is not None and not leaked:
            from ..funding.providers import forget_lease

            forget_lease(vault, credential_id)
        res = _fail(e, leaked_pod=leaked)
        if leaked:
            res = replace(res, burn_attempt=False)
        if isinstance(e, LemonPodError):
            # Not the taxonomy's "infra" (which spends one of the miner's
            # attempts): the caller rents again on another host.
            res = replace(res, error_class=LEMON_CLASS, burn_attempt=False)
        return res


def teardown_funded(
    instances: Iterable[PodInstance],
    vault: PayerKeyVault,
    *,
    provider_factory: Callable[[str], Provider] = lium_provider_for_key,
) -> list[PodInstance]:
    """Terminate funded pods, each with its own payer's key from the vault.

    Returns every instance that could NOT be CONFIRMED gone: the payer's key
    missing from the vault, a terminate that crashed, or — the sneaky case —
    a terminate that "succeeded" while the pod is still listed live.
    ``LiumProvider.terminate`` deliberately treats a failed ``lium rm`` as
    already-terminated (idempotency for the operator fleets), which on a
    REVOKED miner key turns a 401 into silence — so this path re-lists the
    payer's pods after terminating and believes only the listing (audit
    2026-08-29). Unconfirmed pods bill the miner until someone acts, which is
    why the vault TTL must exceed every legitimate pod lifetime and why the
    caller must surface the returned leftovers loudly, never swallow them.
    """
    orphaned: list[PodInstance] = []
    for inst in instances:
        if inst.stage != FUNDED_STAGE:
            continue
        from ..funding.providers import pod_credential

        credential = pod_credential(vault, inst) if inst.payer_hotkey else None
        key = credential.api_key if credential is not None else None
        if not key:
            log.error("no vaulted key for payer %s — cannot stop pod %s on their "
                      "account; the miner must `lium rm` it themselves",
                      inst.payer_hotkey or "<unset>", inst.instance_id)
            orphaned.append(inst)
            continue
        try:
            adapter = (provider_factory(key) if provider_factory is not lium_provider_for_key
                       else provider_for_key(inst.provider, key))
            ids = [inst.instance_id]
            from ..funding.providers import create_intents

            if (inst.payer_credential_id and not inst.pod_uid
                    and create_intents(vault, inst.instance_id.removesuffix("-0"))
                    and not adapter.list_tagged(inst.instance_id)):
                orphaned.append(inst)
                continue
            if inst.provider != "lium" and inst.instance_id.startswith("cascade-"):
                # Write-ahead intent: Shadeform assigns a UUID only after create.
                ids = list(adapter.list_tagged(inst.instance_id))
            if ids:
                confirmed_gone = all(terminate_verified(adapter, pid) for pid in ids)
            else:
                # Nothing listed under the name. Gone when the pod was once
                # OBSERVED (``pod_uid`` is read off a READY pod) or no create
                # is pending for its scope: a replayed teardown, a rental the
                # payer or the host already ended — not a row to keep as
                # "still billing" forever. A write-ahead row whose create is
                # still unresolved stays (the guard above): an eventually
                # consistent empty listing cannot resolve a create.
                confirmed_gone = bool(inst.pod_uid) or not create_intents(
                    vault, inst.instance_id.removesuffix("-0"))
        except Exception as e:  # noqa: BLE001 — one pod's failure must not skip the rest
            log.error("funded teardown of %s (payer %s) failed: %s",
                      inst.instance_id, inst.payer_hotkey, e)
            orphaned.append(inst)
            continue
        if confirmed_gone:
            from ..funding.providers import forget_lease

            try:
                forget_lease(vault, getattr(inst, "payer_credential_id", ""))
            except OSError as e:
                # The pod IS gone; a lease row left behind by a full disk
                # (ENOSPC) only ages out by TTL — never report it as live.
                log.warning("funded pod %s terminated; forgetting its lease failed "
                            "(%s) — the row expires by TTL", inst.instance_id, e)
        if not confirmed_gone:
            log.error("funded pod %s still LIVE after terminate on payer %s's "
                      "account (revoked key?) — miner is billed until it stops",
                      inst.instance_id, inst.payer_hotkey)
            orphaned.append(inst)
    return orphaned


def payer_pod_pattern(hotkey: str, netuid: int) -> re.Pattern:
    """The ONLY names reconcile may touch on ``hotkey``'s account.

    ``cascade-n<netuid>-<digits>-funded-<this payer's slug>`` (plus
    replacement/lane suffixes) — never the generic provisioner scheme, and
    never another deployment's funded scheme. A miner may run their own
    cascade deployment on the same Lium account they fund with; matching
    ``cascade-<n>-heat-…`` there would kill hardware the operator never
    rented — the 2026-07-13 over-reap failure mode, aimed at someone else's
    fleet (audit 2026-08-29). The netuid token additionally stops a testnet
    deployment sweeping a mainnet deployment's live funded legs (and vice
    versa) when both share the payer's account (review 2026-09-02).
    """
    slug = _SLUG_RE.sub("", hotkey.lower())[:12]
    if not slug:
        raise ProvisionError(f"cannot derive a pod slug from hotkey {hotkey!r}")
    return re.compile(rf"^cascade-n{int(netuid)}-\d+-funded-{re.escape(slug)}(-|$)")


def reconcile_funded(
    owned: Iterable[PodInstance],
    vault: PayerKeyVault,
    *,
    netuid: int = 0,
    provider_factory: Callable[[str], Provider] = lium_provider_for_key,
    hotkeys: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
) -> list[str]:
    """The orphan reaper's per-payer sweep; returns the pod names CONFIRMED gone.

    ``hotkeys`` restricts the sweep to those payers (a finished leg's
    teardown fallback sweeps ITS payer only); ``exclude`` skips payers whose
    leg is still in flight (their pod is the leg — the restart sweep must
    reap everyone else's orphans without waiting for the field to empty,
    2026-09-29: four pods idled 5–14 h because the sweep only ran when no leg
    was in flight, which under rolling eras is never).

    The operator-account reaper cannot see pods on miners' accounts, so this
    walks every hotkey that has a vaulted key, lists THAT account's pods, and
    kills any matching :func:`payer_pod_pattern` for THAT payer that the
    ledger does not own — the crash-between-launch-and-ledger hole, closed
    per payer, scoped so a miner's own unrelated cascade pods are untouchable.

    Termination is VERIFIED (:func:`terminate_verified` re-lists after rm):
    ``LiumProvider.terminate`` swallows a failed ``lium rm`` on a revoked key
    as already-terminated, so an unverified reap would report a still-billing
    orphan as killed every sweep (review 2026-08-29). Only confirmed kills are
    returned; a pod that could not be confirmed dead is logged as a leak.
    """
    from .payer_pool import reconcile_payer_accounts

    return reconcile_payer_accounts(
        owned, vault, netuid=netuid, provider_factory=provider_factory,
        hotkeys=hotkeys, exclude=exclude)
