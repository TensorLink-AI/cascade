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
from dataclasses import dataclass

from ..funding.faults import classify_rent_failure
from ..funding.vault import PayerKeyVault
from .core import LaunchSpec, LiumProvider, PodAddress, Provider, ProvisionError
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
        provider = provider_factory(api_key)
    except Exception as e:  # noqa: BLE001 — a bad key must classify, not crash the round
        return _fail(e)

    spec = LaunchSpec(
        sku=sku, count=1, image=image, ssh_pubkey=ssh_pubkey,
        name_prefix=name, gpus_per_pod=gpus_per_pod, exclude_ids=exclude_ids,
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
            raise ProvisionError(
                f"funded pod {pod_id} not ready within {ready_timeout:.0f}s"
                + (f"; lium up said: {tail}" if tail else ""))
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
        pod = PodInstance(
            provider=provider.name, instance_id=pod_id, stage=FUNDED_STAGE,
            rented_at_iso=now_iso(), sku=sku, gpus=gpus_per_pod,
            payer_hotkey=hotkey, pod_uid=pod_uid,
        )
        log.info("funded pod %s ready for %s at %s:%d (billed to payer)",
                 pod_id, hotkey, addr.ip, addr.ssh_port)
        machine = ""
        getter = getattr(provider, "machine_of", None)
        if getter is not None:
            machine = getter(pod_id) or ""
        return FundedRentResult(hotkey=hotkey, ok=True, pod=pod, address=addr,
                                machine_id=machine, pod_uid=pod_uid,
                                host_key=host_key)
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
        return _fail(e, leaked_pod=leaked)


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
