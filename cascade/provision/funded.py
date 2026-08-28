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
    "reconcile_funded",
    "rent_funded_pod",
    "teardown_funded",
]

log = logging.getLogger("cascade.provision.funded")

FUNDED_STAGE = "funded"

_SLUG_RE = re.compile(r"[^0-9a-z]")


def funded_pod_name(round_id: str, hotkey: str) -> str:
    """``cascade-<round>-funded-<slug>``: reaper-matchable, payer-attributable.

    The slug is the hotkey's first 12 chars lowercased (SS58 is alnum, so this
    is stable and collision-safe within any real field). The name must satisfy
    ``loop.is_provisioner_pod_name`` or the orphan reaper would skip — or
    worse, a rename would orphan — funded pods.
    """
    slug = _SLUG_RE.sub("", hotkey.lower())[:12]
    if not slug:
        raise ProvisionError(f"cannot derive a pod slug from hotkey {hotkey!r}")
    return f"cascade-{round_id}-funded-{slug}"


def lium_provider_for_key(api_key: str) -> Provider:
    """A Lium adapter billing ``api_key``'s account (the default factory)."""
    if not api_key:
        raise ProvisionError("funded rental needs the payer's api key (empty)")
    return LiumProvider(api_key=api_key)


@dataclass(frozen=True)
class FundedRentResult:
    """One funded rent attempt, success or classified failure."""

    hotkey: str
    ok: bool
    pod: PodInstance | None = None
    address: PodAddress | None = None
    error: str = ""
    error_class: str = ""           # cascade.funding.faults class; "" on success
    burn_attempt: bool = False      # True only for "infra" (the taxonomy's rule)


def rent_funded_pod(
    *,
    round_id: str,
    hotkey: str,
    api_key: str,
    sku: str,
    image: str,
    ssh_pubkey: str,
    gpus_per_pod: int = 1,
    ready_timeout: float = 900.0,
    provider_factory: Callable[[str], Provider] = lium_provider_for_key,
    now_iso: Callable[[], str] = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
) -> FundedRentResult:
    """Rent ONE pod for ``hotkey``'s challenger leg on ``hotkey``'s own key.

    Success returns the ledgered :class:`PodInstance` (stage ``"funded"``,
    ``payer_hotkey`` set — the caller appends it to the round ledger and saves
    BEFORE using the pod, same write-ahead rule as the stage fleets) plus its
    SSH address. Failure tears down anything half-launched (with the payer's
    key), classifies the error, and reports whether the attempt should burn.
    """

    def _fail(err: Exception | str) -> FundedRentResult:
        msg = str(err)
        if api_key:
            msg = msg.replace(api_key, "<redacted>")
        cls = classify_rent_failure(msg)
        log.warning("funded rent for %s failed [%s]: %s", hotkey, cls, msg[-300:])
        return FundedRentResult(
            hotkey=hotkey, ok=False, error=msg[-500:], error_class=cls,
            burn_attempt=(cls == "infra"),
        )

    name = funded_pod_name(round_id, hotkey)
    try:
        provider = provider_factory(api_key)
    except Exception as e:  # noqa: BLE001 — a bad key must classify, not crash the round
        return _fail(e)

    spec = LaunchSpec(
        sku=sku, count=1, image=image, ssh_pubkey=ssh_pubkey,
        name_prefix=name, gpus_per_pod=gpus_per_pod,
    )
    launched: list[str] = []
    try:
        launched = provider.launch(spec)
        pod_id = launched[0]
        if not provider.wait_ready(pod_id, timeout=ready_timeout):
            raise ProvisionError(f"funded pod {pod_id} not ready within {ready_timeout:.0f}s")
        addr = provider.get_ip(pod_id)
        if addr is None:
            raise ProvisionError(f"funded pod {pod_id} exposed no IP")
        pod = PodInstance(
            provider=provider.name, instance_id=pod_id, stage=FUNDED_STAGE,
            rented_at_iso=now_iso(), sku=sku, gpus=gpus_per_pod,
            payer_hotkey=hotkey,
        )
        log.info("funded pod %s ready for %s at %s:%d (billed to payer)",
                 pod_id, hotkey, addr.ip, addr.ssh_port)
        return FundedRentResult(hotkey=hotkey, ok=True, pod=pod, address=addr)
    except Exception as e:  # noqa: BLE001 — classify everything; the taxonomy decides
        for pid in launched:
            try:
                provider.terminate(pid)
            except Exception as te:  # noqa: BLE001 — best-effort, reconcile is the backstop
                log.error("funded teardown of %s (payer %s) failed — may be leaked "
                          "on the MINER's account: %s", pid, hotkey, te)
        return _fail(e)


def teardown_funded(
    instances: Iterable[PodInstance],
    vault: PayerKeyVault,
    *,
    provider_factory: Callable[[str], Provider] = lium_provider_for_key,
) -> list[PodInstance]:
    """Terminate funded pods, each with its own payer's key from the vault.

    Returns the instances that could NOT be handled because the payer's key is
    gone (vault expiry after a very long outage). Those are the one leak class
    this design cannot fix by itself — the miner is billed until they stop the
    pod, which is why the vault TTL must exceed every legitimate pod lifetime
    and why the caller must surface the leftovers loudly, never swallow them.
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
        provider_factory(key).terminate(inst.instance_id)
    return orphaned


def reconcile_funded(
    owned: Iterable[PodInstance],
    vault: PayerKeyVault,
    *,
    provider_factory: Callable[[str], Provider] = lium_provider_for_key,
    is_ours: Callable[[str], bool] | None = None,
) -> list[str]:
    """The orphan reaper's per-payer sweep; returns the pod names it killed.

    The operator-account reaper cannot see pods on miners' accounts, so this
    walks every hotkey that has a vaulted key, lists THAT account's
    ``cascade-``-tagged pods, and kills any the ledger does not own — the
    crash-between-launch-and-ledger hole, closed per payer. ``is_ours``
    defaults to the loop's naming gate and exists as a seam for tests.
    """
    from .loop import is_provisioner_pod_name

    ours = is_ours or is_provisioner_pod_name
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
        try:
            tagged = lister("cascade-")
        except Exception as e:  # noqa: BLE001 — one payer's API trouble must not stop the sweep
            log.warning("funded reconcile: listing payer %s failed: %s", hotkey, e)
            continue
        for pod_name in tagged:
            if ours(pod_name) and pod_name not in owned_ids:
                log.warning("funded reconcile: killing orphan %s on payer %s's account",
                            pod_name, hotkey)
                provider.terminate(pod_name)
                killed.append(pod_name)
    return killed
