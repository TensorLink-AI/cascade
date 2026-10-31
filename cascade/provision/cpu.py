"""CPU VM providers for the generation tier (DEC-CA-0053).

The generation tier rents one operator-paid CPU VM per leg and runs the
generator there, so the GPU pod never runs miner code. Lium has no CPU
executors, so the providers are Targon (CPU VMs, :mod:`.targon`) and
Shadeform (a vCPU-rich cheap shape from its GPU type listing, below).

A :class:`CpuProvider` mirrors the GPU :class:`~.core.Provider` lifecycle for a
single VM: ``offers`` is the non-destructive probe, then ``launch`` →
``wait_ready`` → ``get_ip`` → ``terminate`` (idempotent). Nothing here runs
unless ``[generation] enabled`` — the provisioner is the only caller.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..shared.config import GenerationConfig
from .core import (
    DEFAULT_SSH_PORT,
    LaunchSpec,
    PodAddress,
    ProvisionError,
    ShadeformProvider,
    pick_shadeform_cpu_offer,
    shadeform_create_body,
)

log = logging.getLogger("cascade.provision.cpu")


@dataclass(frozen=True)
class CpuShape:
    """One rentable CPU VM shape. ``type_id`` is the provider's resource id
    (Targon ``resource_name``, Shadeform ``shade_instance_type``); ``cloud`` is
    Shadeform's underlying cloud (empty for Targon)."""

    vcpus: int
    memory_gb: float
    price_per_hour: float
    provider: str
    type_id: str
    region: str = ""
    cloud: str = ""


class CpuProvider(Protocol):
    name: str
    # The login user on a launched VM (Targon VMs: always ``ubuntu``).
    ssh_user: str

    def offers(self, min_vcpus: int, max_price: float) -> list[CpuShape]: ...
    def launch(self, shape: CpuShape, *, name: str, ssh_pubkey: str,
               image: str = "") -> str: ...
    def wait_ready(self, vm_id: str, *, timeout: float) -> bool: ...
    def get_ip(self, vm_id: str) -> PodAddress | None: ...
    def terminate(self, vm_id: str) -> None: ...


def select_cpu_offer(
    providers: Sequence[CpuProvider], cfg: GenerationConfig
) -> tuple[CpuProvider, CpuShape] | None:
    """The cheapest offer of the first provider (in the given order) that has
    one with ``cfg.vcpus`` vCPUs under ``cfg.max_price_per_hour``. A provider
    whose probe fails is logged and skipped — one adapter's fault never masks
    the rest (as :func:`.core.select_provider`). ``None`` ⇒ no capacity."""
    for p in providers:
        try:
            offers = p.offers(cfg.vcpus, cfg.max_price_per_hour)
        except ProvisionError:
            raise
        except Exception as e:  # noqa: BLE001 — never let one adapter's fault mask the rest
            log.warning("cpu provider %s offer probe failed: %s", p.name, e)
            continue
        log.info("cpu provider %s: %d offer(s) for %d vCPU ≤ $%.2f/h",
                 p.name, len(offers), cfg.vcpus, cfg.max_price_per_hour)
        if offers:
            return p, offers[0]
    return None


def select_tiered_offer(
    providers: Sequence[CpuProvider], cfg: GenerationConfig
) -> tuple[CpuProvider, CpuShape, int] | None:
    """:func:`select_cpu_offer` over the tiers in ``cfg.shapes``: every provider
    at the pinned ``vcpus`` first, a fallback size only when NO provider has
    the larger one. Returns ``(provider, offer, tier_vcpus)`` — the tier, not
    the offer's own core count, sets the leg's children and window."""
    from dataclasses import replace

    for v in cfg.shapes:
        pick = select_cpu_offer(providers, replace(cfg, vcpus=v, fallback_vcpus=()))
        if pick is not None:
            if v != cfg.vcpus:
                log.info("cpu tier fallback: no %d-vCPU offer — renting a %d-vCPU tier "
                         "(window %ds, same CPU budget)", cfg.vcpus, v, cfg.window_for(v))
            return pick[0], pick[1], v
    return None


# ── Shadeform as a CPU provider ──────────────────────────────────────────────


class ShadeformCpuProvider(ShadeformProvider):
    """Shadeform's cheapest vCPU-rich shape, rented for its cores.

    Reuses :class:`~.core.ShadeformProvider`'s HTTP session, ``wait_ready``,
    ``get_ip`` and idempotent ``terminate``. Docker mode needs ``image`` (the
    digest-pinned sandbox image); with ``ssh_key_id`` set it boots a bare VM
    (user ``shadeform``) instead. Unverified: whether Shadeform ever lists a
    GPU-less type — :func:`~.core.pick_shadeform_cpu_offer` prefers one if so.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "shadeform"
        self.ssh_user = "shadeform" if self.ssh_key_id else "root"

    def offers(self, min_vcpus: int, max_price: float) -> list[CpuShape]:
        types = self._get("/instances/types", {"available": "true"})
        return [
            CpuShape(vcpus=o["vcpus"], memory_gb=o["memory_gb"],
                     price_per_hour=o["price_usd_hr"], provider=self.name,
                     type_id=str(o["shade_instance_type"]), region=str(o["region"] or ""),
                     cloud=str(o["cloud"] or ""))
            for o in pick_shadeform_cpu_offer(types, min_vcpus, max_price)
        ]

    def launch(self, shape: CpuShape, *, name: str, ssh_pubkey: str,  # type: ignore[override]
               image: str = "") -> str:
        if not self.ssh_key_id and "@sha256:" not in image:
            # A bare provider image name (e.g. the Targon ``vm_image``) is not
            # a container: refuse before anything is rented.
            raise ProvisionError("shadeform cpu: docker mode needs a digest-pinned image "
                                 "(or set $SHADEFORM_SSH_KEY_ID for VM mode)")
        spec = LaunchSpec(sku="CPU", count=1, image=image, ssh_pubkey=ssh_pubkey,
                          ssh_port=DEFAULT_SSH_PORT, name_prefix=name)
        offer = {"cloud": shape.cloud, "region": shape.region,
                 "shade_instance_type": shape.type_id}
        body = shadeform_create_body(spec, offer, name=name,
                                     ssh_key_id=self.ssh_key_id or None)
        resp = self._post("/instances/create", body)
        iid = resp.get("id")
        if not iid:
            raise ProvisionError(f"shadeform create returned no id: {resp}")
        log.info("shadeform cpu create → %s (%s/%s, %d vCPU)", iid, shape.cloud,
                 shape.region, shape.vcpus)
        self._region_by_pod[str(iid)] = shape.region
        return str(iid)


def cpu_providers(cfg: GenerationConfig, options: dict[str, dict] | None = None
                  ) -> list[CpuProvider]:
    """CPU adapters in ``cfg.providers`` order. ``options`` maps a provider
    name to constructor kwargs; the Shadeform VM-mode key id defaults to
    ``$<[generation] shadeform_ssh_key_id_env>`` (an account key id, not a
    secret). Construction reads no credentials."""
    import os

    from .targon import TargonCpuProvider

    options = dict(options or {})
    key_id = os.environ.get(cfg.shadeform_ssh_key_id_env, "").strip()
    if key_id:
        options["shadeform"] = {"ssh_key_id": key_id, **options.get("shadeform", {})}

    factories = {"targon": TargonCpuProvider, "shadeform": ShadeformCpuProvider}
    out: list[CpuProvider] = []
    for name in cfg.providers:
        factory = factories.get(name)
        if factory is None:
            raise ProvisionError(f"unknown cpu provider {name!r}; known: {sorted(factories)}")
        out.append(factory(**(options or {}).get(name, {})))
    return out
