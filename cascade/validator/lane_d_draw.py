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


__all__ = ["DEFERRED_BUDGET", "SHORT_FRESH", "cell_capacity", "cell_key", "draw_seed",
           "draw_windows", "fresh_keep", "split_quota"]
