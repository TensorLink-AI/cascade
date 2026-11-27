"""Lane D seeded per-bin window draw (LANE_D_PLAN rev6.10 (b), rev6.11 (a)/(b),
rev6.12 (a)/(b); §3.0 W5j row).

Each settlement's judged windows are a seeded random draw, equal per
``(timeframe, bin)`` cell, from the boundary block hash: one draw per cohort
over its intersection eligible set, identical for KING_D, the house and every
member. The algorithm (``tests/accept/w5j/README.md``, pinned there):

1. **Seed** ``f"{boundary_block_hash}:lane_d_windows"``; one stream
   ``np.random.default_rng(_seed_to_int(seed))`` (lane A's seed map,
   ``cascade/eval/bootstrap.py``), consumed only in the order below.
2. **Budget**: ``budget < min_windows`` draws nothing (``window_budget``).
3. **Capacity** of a cell: non-overlapping candidates at ``h_lo``,
   ``Σ_series Σ_runs floor(run / h_lo)`` over the maximal eligible runs.
4. **Quota**: lane A's even pick (``_pick_series``, ``windows.py``): the
   smallest-capacity cell first (ties by key), scarce cells whole, the
   surplus re-split.
5. **Picks**, cells in key order: ``h`` uniform in ``[h_lo, h_hi]`` (fixed
   when equal), the candidate uniform over the valid ``(series, origin)``
   pairs (no target point overlapping a window already drawn from that
   series), **fresh first** (rev6.12 (b)): the embargo candidates only when no
   fresh one is left; none left ends the cell (no refill).
6. **Source cap** (``cap_source_share``, W5b); no refill.
7. **Minimum fresh share** ``s = min_fresh_window_share``: with ``F`` fresh
   and ``E`` embargo windows left, keep the largest ``K <= E`` with
   ``F >= s * (F + K)`` (the stated rule, "at least ``s`` of the set is
   fresh", i.e. ``K <= F (1 - s) / s``), the last-drawn embargo windows
   dropped first. The plan text's ``F * s / (1 - s)`` is an erratum; the two
   agree at the default ``s = 0.5`` (``K <= F``).
8. **Floors after** (``min_windows`` windows from ``min_clusters`` sources):
   a set that met them before step 7 but not after is ``fresh_share_short``
   (the entry stays pending; the share is never relaxed).

There is no confirmation draw (rev6.11 (b)). Pure and deterministic: the
input is never mutated and its order doesn't matter.

Light imports only: this module is on the validator package's import path.
"""

from __future__ import annotations

import numpy as np

from ..eval.bootstrap import _seed_to_int
from .lane_d_frozen import cap_source_share

SEED_TAG = "lane_d_windows"
DEFERRED_BUDGET = "window_budget"
SHORT_FRESH = "fresh_share_short"


# rev6.9 / W7a: the class of a pool ``freq`` (exact match); anything else,
# hourly and finer included, is ``subdaily``.
FREQ_CLASS = {"D": "daily", "W": "weekly", "M": "monthly", "ME": "monthly", "MS": "monthly",
              "Q": "quarterly", "QE": "quarterly", "QS": "quarterly",
              "A": "yearly", "Y": "yearly", "YE": "yearly"}


def freq_class(freq) -> str:
    return FREQ_CLASS.get(str(freq), "subdaily")


def draw_seed(boundary_block_hash: str) -> str:
    return f"{boundary_block_hash}:{SEED_TAG}"


def cell_key(cell: dict) -> str:
    return f"{cell['timeframe']}|{cell['bin']}"


def _runs(flags) -> list[int]:
    out, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def cell_capacity(cell: dict) -> int:
    lo = int(cell["h_lo"])
    return sum(r // lo for s in cell["series"] for r in _runs(s["eligible"]))


def split_quota(caps: dict[str, int], total: int) -> dict[str, int]:
    """Lane A's even pick: smallest capacity first (ties by key), scarce
    cells whole, the surplus re-split over the rest; cells never reached 0."""
    quota = {k: 0 for k in caps}
    remaining = dict(sorted(caps.items()))
    want = int(total)
    while remaining and want > 0:
        share = max(1, want // len(remaining))
        key = min(remaining, key=lambda k: (remaining[k], k))
        cap = remaining.pop(key)
        take = min(cap, share if remaining else want)
        quota[key] = take
        want -= take
    return quota


def fresh_keep(n_embargo: int, n_fresh: int, share: float) -> int:
    """The most embargo windows ``k <= n_embargo`` with
    ``n_fresh >= share * (n_fresh + k)`` (``share = 1``: none; 0: all)."""
    for k in range(int(n_embargo), -1, -1):
        if n_fresh >= share * (n_fresh + k):
            return k
    return 0


def _overlaps(spans, a: int, b: int) -> bool:
    return any(a < e and s < b for s, e in spans)


def _candidates(series: list[dict], h: int, taken: dict) -> list[tuple[str, int, dict]]:
    out = []
    for s in series:
        el, sid = s["eligible"], s["series_id"]
        spans = taken.get(sid, ())
        for i in range(0, len(el) - h + 1):
            if all(el[i:i + h]) and not _overlaps(spans, i, i + h):
                out.append((sid, i, s))
    return out


def _floors(ws, cfg) -> bool:
    return (len(ws) >= int(cfg.min_windows)
            and len({str(w["source"]) for w in ws}) >= int(cfg.min_clusters))


def draw_windows(cells, *, boundary_block_hash: str, cfg, budget: int | None = None) -> dict:
    """The seeded per-bin draw over ``cells`` (see the module docstring).

    Each cell is ``{"timeframe", "bin", "kind", "h_lo", "h_hi", "series":
    [{"series_id", "source", "ts", "eligible", "e0"}]}``. Returns plain JSON:
    ``seed``, ``boundary_block_hash``, ``max_windows_per_settlement``,
    ``capacity`` / ``quota`` per ``"<timeframe>|<bin>"``, the kept
    ``windows`` sorted by ``window_id``, ``fresh_windows``,
    ``embargo_windows``, ``meets_floors``, ``short_reason`` and ``deferred``.
    """
    seed = draw_seed(boundary_block_hash)
    ordered = sorted(({**c, "series": sorted(c["series"], key=lambda s: str(s["series_id"]))}
                      for c in cells), key=cell_key)
    caps = {cell_key(c): cell_capacity(c) for c in ordered}
    max_w = int(cfg.max_windows_per_settlement)
    total = max_w if budget is None else int(budget)
    out = {"seed": seed, "boundary_block_hash": str(boundary_block_hash),
           "max_windows_per_settlement": max_w, "capacity": caps}
    if total < int(cfg.min_windows):
        return {**out, "quota": {k: 0 for k in caps}, "windows": [], "fresh_windows": 0,
                "embargo_windows": 0, "meets_floors": False, "short_reason": None,
                "deferred": DEFERRED_BUDGET}
    quota = split_quota(caps, total)
    rng = np.random.default_rng(_seed_to_int(seed))
    taken: dict[str, list[tuple[int, int]]] = {}
    picks: list[dict] = []
    for cell in ordered:
        lo, hi = int(cell["h_lo"]), int(cell["h_hi"])
        for _ in range(quota[cell_key(cell)]):
            h = lo if lo == hi else int(rng.integers(lo, hi + 1))
            cands = _candidates(cell["series"], h, taken)
            fresh = [c for c in cands if all(c[2]["e0"][c[1]:c[1] + h])]
            group = fresh or cands
            if not group:
                break
            sid, i, s = group[int(rng.integers(len(group)))]
            taken.setdefault(sid, []).append((i, i + h))
            origin = int(s["ts"][i])
            picks.append({"window_id": f"{sid}:{origin}:h{h}", "series_id": str(sid),
                          "source": str(s["source"]), "timeframe": str(cell["timeframe"]),
                          "bin": str(cell["bin"]), "kind": str(cell["kind"]), "h": int(h),
                          "origin": origin, "embargo_window": not all(s["e0"][i:i + h])})
    kept_ids = {w["window_id"] for w in cap_source_share(
        [{"window_id": w["window_id"], "source": w["source"]} for w in picks],
        max_share=float(cfg.max_source_share), seed=seed)}
    kept = [w for w in picks if w["window_id"] in kept_ids]          # draw order
    emb = [w for w in kept if w["embargo_window"]]
    fresh_kept = [w for w in kept if not w["embargo_window"]]
    emb = emb[:fresh_keep(len(emb), len(fresh_kept), float(cfg.min_fresh_window_share))]
    final = sorted(fresh_kept + emb, key=lambda w: w["window_id"])
    meets = _floors(final, cfg)
    short = SHORT_FRESH if not meets and _floors(kept, cfg) else None
    return {**out, "quota": quota, "windows": final, "fresh_windows": len(fresh_kept),
            "embargo_windows": len(emb), "meets_floors": meets, "short_reason": short,
            "deferred": None}


# ── per-window horizon bands (owner 2026-11-27) ─────────────────────────────


def balance_cells(ws: list[dict], min_cell: int) -> tuple[list[dict], dict, dict]:
    """Equal weight per ``(frequency class, band)`` cell by equal counts:
    a cell with fewer than ``min_cell`` windows is left out, every other cell
    keeps its first ``m`` windows in draw order (``m`` = the smallest kept
    cell's count). ``(kept in draw order, {cell: n kept}, {cell: n dropped
    cell})``. Equal counts make the unweighted per-window mean (the statistic
    every checker re-derives from the rows) the equal-weight-per-cell mean."""
    by: dict[str, list[dict]] = {}
    for w in ws:
        by.setdefault(w["cell"], []).append(w)
    small = {k: len(v) for k, v in by.items() if len(v) < int(min_cell)}
    big = {k: v for k, v in by.items() if k not in small}
    if not big:
        return [], {}, dict(sorted(small.items()))
    m = min(len(v) for v in big.values())
    keep = {w["window_id"] for v in big.values() for w in v[:m]}
    return ([w for w in ws if w["window_id"] in keep], {k: m for k in sorted(big)},
            dict(sorted(small.items())))


def draw_band_windows(cands: list[dict], *, boundary_block_hash: str, cfg,
                      budget: int | None = None) -> dict:
    """The seeded draw of a cohort's band candidates (owner 2026-11-27).

    ``cands``: ``{"window_id", "series_id", "source", "freq", "band", "h",
    "t0", "t1", "embargo_window"}`` (``t0`` / ``t1`` the target's first / last
    data time). Steps, one ``np.random.default_rng(_seed_to_int(seed))``
    stream, ``seed = draw_seed(boundary_block_hash)``:

    1. cells ``"<frequency class>|<band>"``; capacity = candidates; budget <
       ``min_windows`` draws nothing;
    2. quota: :func:`split_quota` (the live draw's even split);
    3. per cell, ONE permutation of its fresh candidates then one of its
       embargo candidates (cells in key order); picks go round-robin over the
       cells in key order, each taking its next candidate whose target does
       not overlap (in data time) a window already drawn from that series
       (any band): fresh first, the band and the cut of every window are thus
       the block hash's;
    4. the source cap (:func:`cap_source_share`), then :func:`balance_cells`
       and the minimum fresh share (:func:`fresh_keep`, the last-drawn embargo
       windows dropped first), repeated until neither changes the set;
    5. the floors after.
    """
    from ..shared.lane_d_horizons import cell_class

    seed = draw_seed(boundary_block_hash)
    max_w = int(cfg.max_windows_per_settlement)
    total = max_w if budget is None else int(budget)
    by_cell: dict[str, list[dict]] = {}
    for c in sorted(cands, key=lambda c: str(c["window_id"])):
        cell = f"{cell_class(c['freq'])}|{c['band']}"
        by_cell.setdefault(cell, []).append({**c, "cell": cell})
    caps = {k: len(v) for k, v in sorted(by_cell.items())}
    out = {"seed": seed, "boundary_block_hash": str(boundary_block_hash),
           "max_windows_per_settlement": max_w, "capacity": caps}
    if total < int(cfg.min_windows):
        return {**out, "quota": {k: 0 for k in caps}, "windows": [], "fresh_windows": 0,
                "embargo_windows": 0, "meets_floors": False, "short_reason": None,
                "deferred": DEFERRED_BUDGET, "cells": {}, "cells_dropped": {}}
    quota = split_quota(caps, total)
    rng = np.random.default_rng(_seed_to_int(seed))
    queues = {}
    for k in sorted(by_cell):
        fresh = [c for c in by_cell[k] if not c["embargo_window"]]
        emb = [c for c in by_cell[k] if c["embargo_window"]]
        queues[k] = ([fresh[i] for i in rng.permutation(len(fresh))]
                     + [emb[i] for i in rng.permutation(len(emb))])
    pos = {k: 0 for k in queues}
    left = {k: int(quota[k]) for k in queues}
    taken: dict[str, list[tuple[int, int]]] = {}
    picks: list[dict] = []
    live = [k for k in sorted(queues) if left[k] > 0]
    while live:
        nxt = []
        for k in live:
            q = queues[k]
            while pos[k] < len(q):
                c = q[pos[k]]
                pos[k] += 1
                spans = taken.get(c["series_id"], ())
                a, b = int(c["t0"]), int(c["t1"])
                if any(a <= e and s <= b for s, e in spans):
                    continue
                taken.setdefault(c["series_id"], []).append((a, b))
                picks.append(c)
                left[k] -= 1
                break
            if left[k] > 0 and pos[k] < len(q):
                nxt.append(k)
        live = nxt
    kept_ids = {w["window_id"] for w in cap_source_share(
        [{"window_id": w["window_id"], "source": w["source"]} for w in picks],
        max_share=float(cfg.max_source_share), seed=seed)}
    kept = [w for w in picks if w["window_id"] in kept_ids]           # draw order
    share = float(cfg.min_fresh_window_share)
    ws, cells, dropped = kept, {}, {}
    while True:
        ws, cells, dropped_now = balance_cells(ws, int(cfg.min_cell_windows))
        dropped.update({k: max(v, dropped.get(k, 0)) for k, v in dropped_now.items()})
        emb = [w for w in ws if w["embargo_window"]]
        fresh_w = [w for w in ws if not w["embargo_window"]]
        k_emb = fresh_keep(len(emb), len(fresh_w), share)
        if k_emb == len(emb):
            break
        gone = {w["window_id"] for w in emb[k_emb:]}
        ws = [w for w in ws if w["window_id"] not in gone]
    final = sorted(ws, key=lambda w: w["window_id"])
    meets = _floors(final, cfg)
    short = SHORT_FRESH if not meets and _floors(kept, cfg) else None
    n_emb = sum(1 for w in final if w["embargo_window"])
    return {**out, "quota": quota, "windows": [{k: v for k, v in w.items()} for w in final],
            "fresh_windows": len(final) - n_emb, "embargo_windows": n_emb,
            "meets_floors": meets, "short_reason": short, "deferred": None,
            "cells": cells, "cells_dropped": dict(sorted(dropped.items()))}


__all__ = ["DEFERRED_BUDGET", "SHORT_FRESH", "balance_cells", "cell_capacity", "cell_key",
           "draw_band_windows", "draw_seed", "draw_windows", "fresh_keep", "split_quota"]
