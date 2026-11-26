"""Lane D frozen window sets (LANE_D_PLAN §1.3 "Evidence accumulates, then
freezes", the W5b row, T18e, B14 / D29).

An entry's eligible set is the union, over every ``pool_d`` snapshot since
its reveal, of the windows :func:`~cascade.validator.lane_d_forward.eligible_windows`
admits for ``max(T_E, T_K)``, deduplicated by window id. At every check the
union is capped so that no single source holds more than
``[lane_d] max_source_share`` of it (:func:`cap_source_share`, the fewest
windows dropped, the drop seeded by the settlement). If the capped set meets
``min_windows`` and ``min_clusters`` the set freezes; once frozen it never
changes. If the capped set misses a floor the entry keeps its uncapped union
and the cap runs again at the next check with that settlement's seed.

The progress record (:func:`advance_frozen_set`) is plain JSON so it can live
in ``lane_d_state.json``::

    {"frozen": bool, "window_ids": [sorted ids], "sources": {id: source},
     "set_sha256": sha256("\\n".join(window_ids)) when frozen else None,
     "seed": the freezing seed when frozen else None}

Light imports only (stdlib + the forward rule): this module is on the
validator package's import path.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable

from .lane_d_forward import eligible_windows


def set_sha256(window_ids: Iterable[str]) -> str:
    """The repo's window-set digest: sha256 of the sorted ids joined by ``\\n``
    (the ``cohorts[].set_sha256`` convention)."""
    return hashlib.sha256("\n".join(sorted(window_ids)).encode("utf-8")).hexdigest()


def _rank(seed: str, wid: str) -> str:
    return hashlib.sha256(f"lane-d/cap\n{seed}\n{wid}".encode()).hexdigest()


def _dedup(windows) -> list[dict]:
    """One dict per ``window_id`` (the lowest source on a conflict, so the
    choice does not depend on input order), sorted by id."""
    best: dict[str, dict] = {}
    for w in windows:
        wid = str(w["window_id"])
        if wid not in best or str(w["source"]) < str(best[wid]["source"]):
            best[wid] = w
    return [best[k] for k in sorted(best)]


def _quota(counts: dict[str, int], max_share: float) -> int:
    """``Q``: the largest integer ``<= max n_s`` with
    ``Q <= max_share * sum_s min(n_s, Q)`` (``Q = 0`` always qualifies)."""
    share = float(max_share)
    for q in range(max(counts.values(), default=0), 0, -1):
        if q <= share * sum(min(n, q) for n in counts.values()):
            return q
    return 0


def cap_source_share(windows, *, max_share: float, seed: str) -> list[dict]:
    """Drop the fewest windows so no source holds more than ``max_share`` of
    the kept set (README "The cap"): each source keeps ``min(n_s, Q)``.

    A capped source keeps the ``Q`` windows ranked first by
    ``sha256(seed, window_id)``, so the kept list is a function of ``seed``
    and the ids only. Only ``window_id`` and ``source`` are read; the kept
    input dicts are returned sorted by ``window_id``."""
    ws = _dedup(windows)
    by_source: dict[str, list[dict]] = {}
    for w in ws:
        by_source.setdefault(str(w["source"]), []).append(w)
    q = _quota({s: len(v) for s, v in by_source.items()}, max_share)
    kept: list[dict] = []
    for members in by_source.values():
        if len(members) <= q:
            kept.extend(members)
        else:
            kept.extend(sorted(members, key=lambda w: _rank(seed, str(w["window_id"])))[:q])
    return sorted(kept, key=lambda w: str(w["window_id"]))


def _meets_floors(windows: list[dict], cfg) -> bool:
    return (len(windows) >= int(cfg.min_windows)
            and len({str(w["source"]) for w in windows}) >= int(cfg.min_clusters))


def advance_frozen_set(progress, pools, *, t_entry: int, t_king: int, cfg, seed: str,
                       exclude=()) -> dict:
    """One check of an entry's set (see the module docstring).

    ``progress`` is ``None`` (a new entry) or an earlier return value, possibly
    reloaded from JSON; it is never mutated. ``pools`` are ``load_pool``
    snapshots; ids in ``exclude`` (windows the entry was already judged on)
    never join. A frozen ``progress`` is returned unchanged."""
    if progress is not None and progress.get("frozen"):
        return copy.deepcopy(progress)
    skip = {str(x) for x in exclude}
    sources: dict[str, str] = {}
    if progress is not None:
        sources = {str(k): str(v) for k, v in (progress.get("sources") or {}).items()
                   if str(k) not in skip}
    fresh: dict[str, str] = {}
    for pool in pools:
        verdicts = eligible_windows(pool, t_entry=int(t_entry), t_king=int(t_king), cfg=cfg)
        for w, v in zip(pool["windows"], verdicts, strict=True):
            wid = str(w["window_id"])
            if not wid or not v.eligible or wid in skip or wid in sources:
                continue
            src = str(w["source"])
            if wid not in fresh or src < fresh[wid]:
                fresh[wid] = src
    sources.update(fresh)
    union = [{"window_id": k, "source": sources[k]} for k in sorted(sources)]
    capped = cap_source_share(union, max_share=float(cfg.max_source_share), seed=str(seed))
    if _meets_floors(capped, cfg):
        ids = [w["window_id"] for w in capped]
        return {"frozen": True, "window_ids": ids,
                "sources": {k: sources[k] for k in ids},
                "set_sha256": set_sha256(ids), "seed": str(seed)}
    return {"frozen": False, "window_ids": sorted(sources), "sources": dict(sorted(sources.items())),
            "set_sha256": None, "seed": None}


__all__ = ["advance_frozen_set", "cap_source_share", "set_sha256"]
