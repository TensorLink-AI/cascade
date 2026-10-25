"""Miner-chosen GPU type for a funded leg (DEC-CA-0060).

Every leg trains the contract's token budget inside the contract's wall
(DEC-CA-0001); what a miner may choose is WHICH GPU type its own key rents —
money for speed, never for compute. A fund request may carry three optional
headers:

    X-Gpu-Type:           a SKU from ``[round] funded_pod_skus`` (case-
                          insensitive; stored in the catalog's spelling)
    X-Gpu-Allow-Fallback: ``1``/``true``/``yes`` or ``0``/``false``/``no``
                          (absent = false: the leg WAITS for the chosen type
                          and requeues like any capacity miss; true = rent the
                          cheapest fitting type when the chosen one has none)
    X-Gpu-Max-Price:      the miner's own $/h cap on the chosen type (absent
                          or 0 = no cap — the miner accepts the listed price)

A chosen type is rented under the MINER's cap only: the operator's
``funded_max_price_per_hour`` / ``funded_max_leg_cost_usd`` exist to protect
miners from the default (cheapest-fitting) choice, not from their own.

Signing (backward compatible): a request with none of the headers signs
today's v2 canonical message byte for byte. A request with any of them signs
the v3 message, which appends :func:`gpu_signed_segment` — built from the
header values EXACTLY as sent — so stripping, adding or editing any of the
three breaks the signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "GPU_FALLBACK_HEADER",
    "GPU_HEADER",
    "GPU_MAX_PRICE_HEADER",
    "GpuCatalog",
    "GpuChoice",
    "gpu_signed_segment",
    "parse_gpu_headers",
    "wall_warning",
]

GPU_HEADER = "X-Gpu-Type"
GPU_FALLBACK_HEADER = "X-Gpu-Allow-Fallback"
GPU_MAX_PRICE_HEADER = "X-Gpu-Max-Price"

_TRUE = ("1", "true", "yes")
_FALSE = ("0", "false", "no")


@dataclass(frozen=True)
class GpuChoice:
    """One entry's GPU choice; the default (``gpu == ""``) is today's path:
    the cheapest fitting type under the operator caps."""

    gpu: str = ""
    allow_fallback: bool = False
    max_price_per_hour: float = 0.0     # the miner's $/h cap; 0 = none

    @property
    def chosen(self) -> bool:
        return bool(self.gpu)

    def as_dict(self) -> dict:
        """The response / status fields (empty for the default choice)."""
        if not self.gpu:
            return {}
        return {"gpu": self.gpu, "allow_fallback": bool(self.allow_fallback),
                "max_price_per_hour": float(self.max_price_per_hour)}


@dataclass(frozen=True)
class GpuCatalog:
    """What the intake checks a choice against: the allowed SKUs (``None`` =
    no config loaded — any well-formed SKU, dev only; ``()`` = choices off),
    the measured per-SKU walls and the contract's wall cap (seconds) behind
    the fund-time warning."""

    skus: tuple[str, ...] | None = None
    walls: tuple[tuple[str, int], ...] = ()
    wall_cap_seconds: float = 0.0

    @classmethod
    def from_config(cls, cfg) -> GpuCatalog:
        rnd = cfg.round
        per_leg = bool(getattr(rnd, "funded_sku_per_leg", False))
        skus = tuple(getattr(rnd, "funded_pod_skus", ()) or ()) if per_leg else ()
        try:
            cap = float(max(int(c.max_train_seconds) for c in cfg.throne_contracts()))
        except Exception:  # noqa: BLE001 — no contract ⇒ no warning, never a refusal
            cap = 0.0
        return cls(skus=skus, walls=tuple(getattr(rnd, "funded_sku_wall_seconds", ()) or ()),
                   wall_cap_seconds=cap)

    def canonical(self, sku: str) -> str | None:
        """The catalog's spelling of ``sku`` (case-insensitive), else None."""
        want = (sku or "").strip()
        if self.skus is None:
            return want or None
        for s in self.skus:
            if s.lower() == want.lower():
                return s
        return None


def gpu_signed_segment(gpu_raw: str, fallback_raw: str, max_price_raw: str) -> str:
    """The v3 field the signature appends, from the header values as sent
    (stripped); ``""`` when none is present — the v2 message, unchanged."""
    g, f, p = (str(x or "").strip() for x in (gpu_raw, fallback_raw, max_price_raw))
    if not (g or f or p):
        return ""
    return f"gpu={g};fallback={f};max_price={p}"


def parse_gpu_headers(headers, catalog: GpuCatalog | None
                      ) -> tuple[GpuChoice, str, dict | None]:
    """``(choice, signed_segment, None)`` or ``(GpuChoice(), "", error_body)``.

    A request without the headers is the default choice and the empty segment.
    ``catalog`` None behaves like ``GpuCatalog()`` (any well-formed SKU)."""
    cat = catalog if catalog is not None else GpuCatalog()
    g = (headers.get(GPU_HEADER) or "").strip()
    f = (headers.get(GPU_FALLBACK_HEADER) or "").strip()
    p = (headers.get(GPU_MAX_PRICE_HEADER) or "").strip()
    segment = gpu_signed_segment(g, f, p)
    if not segment:
        return GpuChoice(), "", None
    if not g:
        return GpuChoice(), "", {
            "code": "gpu_options_without_gpu",
            "message": f"{GPU_FALLBACK_HEADER} / {GPU_MAX_PRICE_HEADER} apply to a chosen "
                       f"GPU type — send {GPU_HEADER} too (or neither)"}
    if cat.skus is not None and not cat.skus:
        return GpuChoice(), "", {
            "code": "gpu_choice_unavailable",
            "message": "this network locks one GPU type ([round] funded_sku_per_leg off "
                       "or no funded_pod_skus) — fund without a GPU choice"}
    sku = cat.canonical(g)
    if sku is None or any(c in sku for c in ";=:|"):
        listed = ", ".join(cat.skus or ())
        return GpuChoice(), "", {
            "code": "unknown_gpu",
            "message": f"{g!r} is not a funded GPU type; choose one of: {listed}"}
    if f.lower() in _TRUE:
        fallback = True
    elif not f or f.lower() in _FALSE:
        fallback = False
    else:
        return GpuChoice(), "", {
            "code": "bad_gpu_fallback",
            "message": f"{GPU_FALLBACK_HEADER} must be 1/true/yes or 0/false/no"}
    price = 0.0
    if p:
        try:
            price = float(p)
        except ValueError:
            price = -1.0
        if not math.isfinite(price) or price < 0:
            return GpuChoice(), "", {
                "code": "bad_gpu_max_price",
                "message": f"{GPU_MAX_PRICE_HEADER} must be a non-negative USD/h figure "
                           "(0 = no cap)"}
    return GpuChoice(gpu=sku, allow_fallback=fallback, max_price_per_hour=price), segment, None


def wall_warning(choice: GpuChoice, catalog: GpuCatalog | None) -> str:
    """Why ``choice``'s type may not finish the token budget inside the wall
    (``[round] funded_sku_wall_seconds``; no entry ⇒ wall-bound), or ``""``.
    Advisory only: a slow choice is the miner's risk, never a refusal — a
    leg that stops at the wall is judged as it stands."""
    if not choice.gpu or catalog is None:
        return ""
    cap = float(catalog.wall_cap_seconds or 0.0)
    measured = None
    for name, secs in catalog.walls:
        if name.lower() == choice.gpu.lower():
            measured = float(secs)
            break
    if measured is None:
        return (f"{choice.gpu} has no measured full-budget wall ([round] "
                f"funded_sku_wall_seconds) — assumed wall-bound: the leg may stop at the "
                f"{cap:.0f}s wall before finishing the token budget and is judged as it "
                f"stands" if cap else
                f"{choice.gpu} has no measured full-budget wall ([round] "
                f"funded_sku_wall_seconds) — assumed wall-bound")
    if cap and measured >= cap:
        return (f"{choice.gpu} needs ~{measured:.0f}s for the full token budget, at or "
                f"past the {cap:.0f}s wall — the leg may stop at the wall and is judged "
                f"as it stands")
    return ""
