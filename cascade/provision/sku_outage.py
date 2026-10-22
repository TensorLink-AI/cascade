"""GPU-type outage cooldown: correlated pod loss steers rents to other types.

Host quarantine (``record_host_quarantine``) keeps ONE machine out of the
listings. A supply-side outage takes down many machines of the same type at
once, and per-host quarantine then just walks the trainer onto the next dying
machine of that type: 2026-10-21 19:03, three H100 pods (the era-2593 king and
two payer legs) went unreachable together; the king re-rented an H100 at 19:21
and lost it 13 min after boot, while an L40 rented at 20:07 trained the full
leg. This module records pod losses per GPU type and reports a type as
COOLING once ``MIN_EVENTS`` distinct pods of it were lost inside ``WINDOW``;
it stays cooling for ``COOL_SECONDS`` after the last loss. Callers PREFER
other types while one cools (never forbid: if only cooling types are offered
they are still used — a wait for capacity is worse than a risky rent).

Presentational to consensus: rent policy only, no contract or manifest change.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

SKU_OUTAGE_FILE_ENV = "CASCADE_LIUM_SKU_OUTAGE_FILE"
WINDOW_SECONDS = 3600.0       # losses this close together are one outage
MIN_EVENTS = 2                # distinct pods lost inside the window
COOL_SECONDS = 3 * 3600.0     # prefer other types this long after the last loss
KEEP_SECONDS = 24 * 3600.0    # events older than this are pruned


def sku_outage_path() -> Path:
    return Path(os.environ.get(
        SKU_OUTAGE_FILE_ENV, "~/.cache/cascade/lium_sku_outage.json")).expanduser()


def _load(path: Path) -> list[dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    return [e for e in raw if isinstance(e, dict) and "t" in e and "sku" in e] \
        if isinstance(raw, list) else []


def record_pod_loss(sku: str, pod: str, reason: str, *, path: Path | None = None,
                    now: float | None = None) -> None:
    """Record that a pod of ``sku`` was lost (unreachable mid-leg)."""
    sku = str(sku or "").strip()
    if not sku:
        return
    path = path or sku_outage_path()
    now = time.time() if now is None else float(now)
    try:
        events = [e for e in _load(path) if now - float(e["t"]) < KEEP_SECONDS]
        events.append({"t": now, "sku": sku, "pod": str(pod)[:120],
                       "reason": str(reason)[:200]})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(events), encoding="utf-8")
        tmp.replace(path)
        if sku in cooling_skus(path=path, now=now):
            log.warning("lium: %s pods lost %d+ times within %.0f min — preferring other "
                        "GPU types for %.1f h (%s)", sku, MIN_EVENTS, WINDOW_SECONDS / 60,
                        COOL_SECONDS / 3600, reason)
    except OSError as e:
        log.warning("lium: could not record %s pod loss: %s", sku, e)


def cooling_skus(*, path: Path | None = None, now: float | None = None) -> set[str]:
    """GPU types with ``MIN_EVENTS`` distinct pod losses inside ``WINDOW_SECONDS``
    and the last loss less than ``COOL_SECONDS`` ago (case-insensitive keys)."""
    path = path or sku_outage_path()
    now = time.time() if now is None else float(now)
    by_sku: dict[str, list[tuple[float, str]]] = {}
    for e in _load(path):
        t = float(e["t"])
        if now - t < KEEP_SECONDS:
            by_sku.setdefault(str(e["sku"]).lower(), []).append((t, str(e.get("pod", ""))))
    out: set[str] = set()
    for sku, evs in by_sku.items():
        evs.sort()
        if now - evs[-1][0] >= COOL_SECONDS:
            continue
        for i, (t0, _p) in enumerate(evs):
            pods = {p for t, p in evs[i:] if t - t0 <= WINDOW_SECONDS}
            if len(pods) >= MIN_EVENTS:
                out.add(sku)
                break
    return out


def prefer_healthy(skus: tuple[str, ...], *, path: Path | None = None,
                   now: float | None = None) -> tuple[str, ...]:
    """``skus`` without the cooling types — unless that leaves nothing, in
    which case ``skus`` unchanged (prefer, never forbid)."""
    cooling = cooling_skus(path=path, now=now)
    if not cooling:
        return skus
    kept = tuple(s for s in skus if s.lower() not in cooling)
    return kept or skus
