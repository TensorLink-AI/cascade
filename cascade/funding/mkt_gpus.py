"""MarketCast GPU choice (owner 2026-11-24): any Lium GPU, at the miner's own risk.

A MarketCast leg fine-tunes the pinned Timestral-28M base in ONE worker
process on ``cuda:0`` (``cascade-train-worker --role mkt_finetune``; no
torchrun, no world size), so a single-GPU pod (1xRTX4090, 1xA100, 1xH100, ...)
trains it as well as an 8-GPU one. Lane B's rule applies:

* ``[marketcast] funded_pod_skus`` is the RECOMMENDED list. A fund choosing one
  of them gets no warning. A fund with no choice rents the cheapest fitting
  one (``()`` = ``[round] funded_pod_skus``, the behaviour before this rule).
* any other well-formed GPU type is accepted AT THE MINER'S OWN RISK: the reply
  carries :data:`OWN_RISK_WARNING`. The leg is still capped by ``[marketcast]
  max_leg_hours`` and the recipe's own ``leg_hours``. It rents on the miner's
  own Lium key, and nothing is refunded if the type is slow, cannot boot, OOMs
  or crashes.
* GPUs rent on Lium only, never Shadeform: an own-risk type needs a plain Lium
  key (``X-Lium-Api-Key``), and the trainer rents every MarketCast leg on the
  Lium credential of a multi-provider bundle only (:func:`lium_only_key`).

This is trainer and intake policy only. No validator, satellite or audit path
reads the GPU choice or this list for a MarketCast verdict or receipt.

Stdlib only: importing it never imports torch.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import replace

from .gpu_choice import GpuCatalog

__all__ = [
    "LIUM_ONLY",
    "OWN_RISK_WARNING",
    "mkt_catalog",
    "mkt_rent_scope",
    "mkt_rent_skus",
    "mkt_skus",
    "lium_only_key",
    "own_risk_gpu",
    "own_risk_warning",
    "with_own_risk",
]

LIUM_ONLY = "marketcast_lium_only"
OWN_RISK_WARNING = (
    "{gpu} is not on the recommended MarketCast GPU list ([marketcast] funded_pod_skus: "
    "{listed}). It is accepted AT YOUR OWN RISK: the leg rents on your own Lium key and is "
    "still capped by [marketcast] max_leg_hours / your recipe's leg_hours, and nothing is "
    "refunded if the type is slow, cannot boot, runs out of memory or crashes.")


def mkt_skus(cfg) -> tuple[str, ...]:
    """The recommended MarketCast GPU types: ``[marketcast] funded_pod_skus``,
    else ``[round] funded_pod_skus`` (unset = the pre-2026-11-24 behaviour)."""
    own = tuple(getattr(getattr(cfg, "marketcast", None), "funded_pod_skus", ()) or ())
    if own:
        return own
    return tuple(getattr(getattr(cfg, "round", None), "funded_pod_skus", ()) or ())


def mkt_catalog(cfg) -> GpuCatalog:
    """What a MarketCast fund's GPU headers are checked against: the
    recommended list, no wall table (a leg's wall is its recipe's hours), the
    operator's ``funded_max_price_per_hour`` as the default cap."""
    return GpuCatalog(skus=mkt_skus(cfg), walls=(), wall_cap_seconds=0.0,
                      default_max_price_per_hour=float(
                          getattr(cfg.round, "funded_max_price_per_hour", 0.0) or 0.0))


def own_risk_gpu(gpu: str, catalog: GpuCatalog | None, *, bundle: bool) -> str:
    """``gpu`` stripped when it is an own-risk type: well-formed, not on the
    recommended list, and funded with a plain Lium key (``bundle`` False);
    else ``""``."""
    from ..shared.config.marketcast import is_gpu_type

    g = str(gpu or "").strip()
    if not g or bundle or not is_gpu_type(g):
        return ""
    if catalog is not None and catalog.skus is not None and catalog.canonical(g) is not None:
        return ""
    return g


def with_own_risk(catalog: GpuCatalog, gpu: str) -> GpuCatalog:
    """``catalog`` with the own-risk type ``gpu`` added (so the shared header
    parser accepts it)."""
    skus = tuple(catalog.skus or ())
    return replace(catalog, skus=(*skus, gpu))


def own_risk_warning(gpu: str, catalog: GpuCatalog | None) -> str:
    listed = ", ".join((catalog.skus or ()) if catalog is not None else ()) or "none"
    return OWN_RISK_WARNING.format(gpu=gpu, listed=listed)


def mkt_rent_skus(cfg, choice) -> tuple[str, ...]:
    """The types a MarketCast leg may rent: the recommended list, plus the
    miner's chosen type when it is an own-risk one (rented exactly as chosen)."""
    skus = mkt_skus(cfg)
    gpu = str(getattr(choice, "gpu", "") or "").strip()
    if gpu and not any(s.lower() == gpu.lower() for s in skus):
        from ..shared.config.marketcast import is_gpu_type

        if is_gpu_type(gpu):
            skus = (*skus, gpu)
    return tuple(skus)


_UNSET = object()


@contextmanager
def mkt_rent_scope(runner, skus: tuple[str, ...]):
    """While the block runs on this thread, the runner's rent path
    (``runner_capacity``) rents from ``skus`` (a MarketCast leg's list) and
    only on Lium (``runner_funded_pods`` filters the key)."""
    loc = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    prev = getattr(loc, "mkt_rent_skus", _UNSET)
    loc.mkt_rent_skus = tuple(skus)
    try:
        yield loc
    finally:
        if prev is _UNSET:
            del loc.mkt_rent_skus
        else:
            loc.mkt_rent_skus = prev


def lium_only_key(raw: str) -> str:
    """A vaulted payer key reduced to Lium: a plain Lium key unchanged; a
    multi-provider bundle re-encoded with its Lium credential only; ``""``
    when the bundle has none (the leg cannot rent — Shadeform never)."""
    from .providers import PREFIX, decode_credentials, encode_credentials

    if not raw or not raw.startswith(PREFIX):
        return raw
    lium = [c for c in decode_credentials(raw) if c.provider == "lium"]
    return encode_credentials(lium) if lium else ""
