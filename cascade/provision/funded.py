"""Per-payer rentals: one pod, one challenger, one miner's Lium key.

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
    best evidence available.
    """
    provider.terminate(pod_id)
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
) -> FundedRentResult:
    """Rent ONE pod for ``hotkey``'s challenger leg on ``hotkey``'s own key.

    Success returns the ledgered :class:`PodInstance` (stage ``"funded"``,
    ``payer_hotkey`` set — the caller appends it to the round ledger and saves
    BEFORE using the pod, same write-ahead rule as the stage fleets) plus its
    SSH address. Failure tears down anything half-launched (with the payer's
    key), classifies the error, and reports whether the attempt should burn.
    """

    def _fail(err: Exception | str, *, leaked_pod: str = "") -> FundedRentResult:
        msg = str(err)
        if api_key:
            msg = msg.replace(api_key, "<redacted>")
        cls = classify_rent_failure(msg)
        log.warning("funded rent for %s failed [%s]: %s", hotkey, cls, msg[-300:])
        return FundedRentResult(
            hotkey=hotkey, ok=False, error=msg[-500:], error_class=cls,
            burn_attempt=(cls == "infra"), leaked_pod=leaked_pod,
        )

    name = funded_pod_name(round_id, hotkey, netuid)
    try:
        provider = apply_cpu_blocklist(provider_factory(api_key), cpu_blocklist)
        provider = apply_price_caps(provider, **(price_caps or {}))
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
    live_fn = getattr(provider, "live_pod_address", None)
    live_addr = None
    if callable(live_fn):
        try:
            live_addr = live_fn(name)
        except Exception as e:  # noqa: BLE001 — a listing blip must not block the rent
            log.warning("funded rent for %s: live-pod probe failed (%s) — renting",
                        hotkey, str(e)[-200:])
    if live_addr is not None:
        try:
            ident_fn = getattr(provider, "pod_identity", None)
            ident = ident_fn(name) if callable(ident_fn) else None
            pod_uid = str((ident or {}).get("id") or "")
            if ident_fn is not None and not pod_uid:
                raise ProvisionError(f"funded pod {name}: platform identity unavailable")
            host_key = ""
            if host_key_scanner is not None:
                host_key = host_key_scanner(live_addr.ip, live_addr.ssh_port)
            sku_fn = getattr(provider, "sku_of", None)
            landed = (str(sku_fn(name) or "") if callable(sku_fn) else "") or (
                skus[0] if skus else sku)
            pod = PodInstance(
                provider=provider.name, instance_id=name, stage=FUNDED_STAGE,
                rented_at_iso=now_iso(), sku=landed, gpus=gpus_per_pod,
                payer_hotkey=hotkey, pod_uid=pod_uid,
            )
            log.warning("funded pod %s is still LIVE at %s:%d from a prior attempt — "
                        "adopting it for %s (no rent; the dispatcher attaches to its run)",
                        name, live_addr.ip, live_addr.ssh_port, hotkey)
            machine = ""
            getter = getattr(provider, "machine_of", None)
            if getter is not None:
                machine = getter(name) or ""
            return FundedRentResult(hotkey=hotkey, ok=True, pod=pod, address=live_addr,
                                    machine_id=machine, pod_uid=pod_uid,
                                    host_key=host_key, sku=landed, adopted=True)
        except Exception as e:  # noqa: BLE001 — never tear the live pod down; retry later
            res = _fail(f"live pod {name} could not be adopted: {e}")
            return replace(res, error_class="infra", burn_attempt=False)

    choices = tuple(skus) if len(tuple(skus)) > 1 else ()
    spec = LaunchSpec(
        sku=(skus[0] if skus else sku), count=1, image=image, ssh_pubkey=ssh_pubkey,
        name_prefix=name, gpus_per_pod=gpus_per_pod, exclude_ids=exclude_ids,
        sku_choices=choices,
    )
    launched: list[str] = []
    try:
        launched = provider.launch(spec)
        pod_id = launched[0]
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
            rented_at_iso=now_iso(), sku=landed, gpus=gpus_per_pod,
            payer_hotkey=hotkey, pod_uid=pod_uid,
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
        res = _fail(e, leaked_pod=leaked)
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
        key = vault.get(inst.payer_hotkey) if inst.payer_hotkey else None
        if not key:
            log.error("no vaulted key for payer %s — cannot stop pod %s on their "
                      "account; the miner must `lium rm` it themselves",
                      inst.payer_hotkey or "<unset>", inst.instance_id)
            orphaned.append(inst)
            continue
        try:
            confirmed_gone = terminate_verified(provider_factory(key), inst.instance_id)
        except Exception as e:  # noqa: BLE001 — one pod's failure must not skip the rest
            log.error("funded teardown of %s (payer %s) failed: %s",
                      inst.instance_id, inst.payer_hotkey, e)
            orphaned.append(inst)
            continue
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
) -> list[str]:
    """The orphan reaper's per-payer sweep; returns the pod names CONFIRMED gone.

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
    owned_ids = {i.instance_id for i in owned if i.stage == FUNDED_STAGE}
    killed: list[str] = []
    for hotkey in vault.hotkeys():
        key = vault.get(hotkey)
        if not key:
            continue
        provider = provider_factory(key)
        lister = getattr(provider, "list_tagged", None)
        if lister is None:
            continue
        mine = payer_pod_pattern(hotkey, netuid)
        try:
            tagged = lister("cascade-")
        except Exception as e:  # noqa: BLE001 — one payer's API trouble must not stop the sweep
            log.warning("funded reconcile: listing payer %s failed: %s", hotkey, e)
            continue
        for pod_name in tagged:
            if mine.match(pod_name) and pod_name not in owned_ids:
                log.warning("funded reconcile: killing orphan %s on payer %s's account",
                            pod_name, hotkey)
                try:
                    confirmed = terminate_verified(provider, pod_name)
                except Exception as e:  # noqa: BLE001 — keep sweeping the rest
                    log.error("funded reconcile: terminate %s failed — may still "
                              "bill payer %s: %s", pod_name, hotkey, e)
                    continue
                if confirmed:
                    killed.append(pod_name)
                else:
                    log.error("funded reconcile: %s still LIVE after terminate on "
                              "payer %s's account (revoked key?) — still billing",
                              pod_name, hotkey)
    return killed
