"""Spec, fixture builders and the reference draw for the W5j tests.

Copied from docs/plans/LANE_D_PLAN.md: rev6.10 (b) "Seeded random draw from
every horizon bin", rev6.11 (a) (5000) and (b) (no confirmation draw), rev6.12 (b)
(fresh-first draw, minimum fresh share) and the §3.0 W5j row. README.md states the
module contract; :func:`reference_draw` is that contract as code, so an
independent re-derivation (W7d, W9) has one exact algorithm to match.

Nothing here imports a module in PATHS at import time: on main every test
fails inside its own body, at its import of ``cascade.validator.lane_d_draw``
(a ``ModuleNotFoundError`` for a path in PATHS) or at an ``assert``.
"""

from __future__ import annotations

import copy
import hashlib
import random

import numpy as np

HORIZON = 64
ERA = 3600
ARM_BLOCK = 2700 * ERA
HASH_A = "0x" + hashlib.sha256(b"w5j boundary A").hexdigest()
HASH_B = "0x" + hashlib.sha256(b"w5j boundary B").hexdigest()
SEED_TAG = "lane_d_windows"

# Point spacing per pool freq (seconds), for building timestamps only.
STEP = {"10s": 10, "1min": 60, "5min": 300, "15min": 900, "H": 3600}
T0 = 1_790_000_000


# rev6.18 F4 (A3): these fixtures pin the live-only rule: the public half off
# (public_weight = 0, a spec setting) and no per-window ratio clamp in effect
# (a floor far below any fixture ratio), so their pre-F4 numbers stand. F4's
# own tests (tests/accept/w5e/test_w5e_combined.py) pin the combined rule.
LIVE_ONLY = {"public_weight": 0.0, "window_ratio_floor": 1e-9}
# Owner 2026-11-27: these fixtures pin the fixed [lane_d] horizon rule (their
# pools hold fixed-length targets); tests/accept/w_hz pins the band rule.
FIXED_HORIZON = {"horizon_bands": []}


def lane_d_cfg(**over):
    """The real parse of an armed [lane_d] table with small floors."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "horizon": HORIZON, "min_windows": 24, "min_clusters": 4,
           "max_source_share": 0.25, "bootstrap_B": 200,
           "max_windows_per_settlement": 60, "min_fresh_window_share": 0.5}
    return parse_lane_d({**raw, **LIVE_ONLY, **FIXED_HORIZON, **over}, era_blocks_at=lambda _block: ERA)


def seed_string(block_hash: str) -> str:
    return f"{block_hash}:{SEED_TAG}"


def seed_int(block_hash: str) -> int:
    """Lane A's blake2b seed map (``cascade/eval/bootstrap.py:_seed_to_int``)."""
    h = hashlib.blake2b(seed_string(block_hash).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def cell_key(cell: dict) -> str:
    return f"{cell['timeframe']}|{cell['bin']}"


# ── fixtures ────────────────────────────────────────────────────────────────


def series(series_id: str, source: str, freq: str, n: int, *, eligible=None, e0=None,
           start: int = T0) -> dict:
    """One series of ``n`` points. ``eligible`` / ``e0`` are per-point flags
    (default: every point eligible and passing at e = 0)."""
    ts = [int(start + i * STEP[freq]) for i in range(n)]
    el = [True] * n if eligible is None else [bool(x) for x in eligible]
    z = list(el) if e0 is None else [bool(x) and bool(y) for x, y in zip(e0, el, strict=True)]
    return {"series_id": series_id, "source": source, "ts": ts, "eligible": el, "e0": z}


def forward_cell(freq: str, members: list[dict], h: int = HORIZON) -> dict:
    return {"timeframe": freq, "bin": "forward", "kind": "forward", "h_lo": h, "h_hi": h,
            "series": members}


def cells_of(spec: dict[str, dict[str, int]], *, windows_per_series: int = 1,
             tag: str = "s") -> list[dict]:
    """``spec[freq][source]`` series per (freq, source), each holding
    ``windows_per_series`` non-overlapping h = 64 windows' worth of points."""
    out = []
    for freq, by_src in spec.items():
        members = []
        for src, k in by_src.items():
            for i in range(k):
                members.append(series(f"{tag}-{freq}-{src}-{i}", src, freq,
                                      HORIZON * windows_per_series))
        out.append(forward_cell(freq, members))
    return out


def shuffled(cells: list[dict], seed: int) -> list[dict]:
    """The same cells and series in another order (the draw sorts)."""
    rnd = random.Random(seed)
    out = copy.deepcopy(cells)
    for c in out:
        rnd.shuffle(c["series"])
    rnd.shuffle(out)
    return out


# ── the reference draw (README "Algorithm") ─────────────────────────────────


def _runs(flags: list[bool]) -> list[int]:
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


def capacity(cell: dict) -> int:
    lo = int(cell["h_lo"])
    return sum(r // lo for s in cell["series"] for r in _runs(s["eligible"]))


def split_quota(caps: dict[str, int], total: int) -> dict[str, int]:
    """Lane A's even pick (``_pick_series``): smallest capacity first (ties by
    key), scarce cells whole, the surplus re-split over the rest."""
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


def _overlaps(spans: list[tuple[int, int]], a: int, b: int) -> bool:
    return any(a < e and s < b for s, e in spans)


def _candidates(cell: dict, h: int, taken: dict) -> list[tuple[str, int, dict]]:
    out = []
    for s in sorted(cell["series"], key=lambda x: x["series_id"]):
        el, sid = s["eligible"], s["series_id"]
        for i in range(0, len(el) - h + 1):
            if all(el[i:i + h]) and not _overlaps(taken.get(sid, []), i, i + h):
                out.append((sid, i, s))
    return out


def _cap_kept(picks: list[dict], max_share: float, seed: str) -> set[str]:
    from cascade.validator.lane_d_frozen import cap_source_share

    kept = cap_source_share([{"window_id": w["window_id"], "source": w["source"]}
                             for w in picks], max_share=max_share, seed=seed)
    return {w["window_id"] for w in kept}


def embargo_keep(n_embargo: int, n_fresh: int, s: float) -> int:
    """rev6.12 (b) step 3: the most embargo windows ``k <= n_embargo`` that
    keep the fresh share ``n_fresh / (n_fresh + k) >= s``, i.e.
    ``n_fresh >= s * (n_fresh + k)`` (``s = 1``: none; ``s = 0``: all)."""
    for k in range(n_embargo, -1, -1):
        if n_fresh >= s * (n_fresh + k):
            return k
    raise AssertionError("unreachable: k = 0 always fits")


def _floors(ws, min_windows, min_clusters) -> bool:
    return len(ws) >= int(min_windows) and len({w["source"] for w in ws}) >= int(min_clusters)


def reference_draw(cells: list[dict], *, block_hash: str, min_windows: int,
                   min_clusters: int, max_source_share: float, fresh_share: float,
                   max_windows: int, budget: int | None = None) -> dict:
    caps = {cell_key(c): capacity(c) for c in cells}
    total = int(max_windows if budget is None else budget)
    seed = seed_string(block_hash)
    if total < int(min_windows):
        return {"seed": seed, "capacity": caps, "quota": {k: 0 for k in caps},
                "windows": [], "deferred": "window_budget", "meets_floors": False,
                "short_reason": None, "fresh_windows": 0, "embargo_windows": 0}
    quota = split_quota(caps, total)
    rng = np.random.default_rng(seed_int(block_hash))
    taken: dict[str, list[tuple[int, int]]] = {}
    picks: list[dict] = []
    for cell in sorted(cells, key=cell_key):
        for _ in range(quota[cell_key(cell)]):
            lo, hi = int(cell["h_lo"]), int(cell["h_hi"])
            h = lo if lo == hi else int(rng.integers(lo, hi + 1))
            cands = _candidates(cell, h, taken)
            fresh = [c for c in cands if all(c[2]["e0"][c[1]:c[1] + h])]
            group = fresh or cands          # fresh first, then fill from embargo
            if not group:
                break
            sid, i, s = group[int(rng.integers(len(group)))]
            taken.setdefault(sid, []).append((i, i + h))
            picks.append({"window_id": f"{sid}:{s['ts'][i]}:h{h}", "series_id": sid,
                          "source": s["source"], "timeframe": cell["timeframe"],
                          "bin": cell["bin"], "kind": cell["kind"], "h": h,
                          "origin": s["ts"][i],
                          "embargo_window": not all(s["e0"][i:i + h])})
    kept_ids = _cap_kept(picks, max_source_share, seed)
    kept = [w for w in picks if w["window_id"] in kept_ids]          # draw order
    emb = [w for w in kept if w["embargo_window"]]
    fresh_kept = [w for w in kept if not w["embargo_window"]]
    k = embargo_keep(len(emb), len(fresh_kept), fresh_share)
    emb = emb[:k]                                                   # last-drawn dropped first
    final = sorted(fresh_kept + emb, key=lambda w: w["window_id"])
    meets = _floors(final, min_windows, min_clusters)
    short = None
    if not meets and _floors(kept, min_windows, min_clusters):
        short = "fresh_share_short"
    return {"seed": seed, "capacity": caps, "quota": quota, "windows": final,
            "deferred": None, "meets_floors": meets, "short_reason": short,
            "fresh_windows": len(fresh_kept), "embargo_windows": len(emb)}


def ref_for(cells, cfg, block_hash: str, **kw) -> dict:
    """:func:`reference_draw` at the test config's values (see ``lane_d_cfg``)."""
    vals = {"min_windows": 24, "min_clusters": 4, "max_source_share": 0.25,
            "fresh_share": 0.5, "max_windows": 60}
    vals.update(kw)
    return reference_draw(cells, block_hash=block_hash, **vals)


def draw(cells, cfg, block_hash: str, **kw) -> dict:
    from cascade.validator.lane_d_draw import draw_windows

    return draw_windows(copy.deepcopy(cells), boundary_block_hash=block_hash, cfg=cfg, **kw)


WINDOW_FIELDS = ("window_id", "series_id", "source", "timeframe", "bin", "kind", "h",
                 "origin", "embargo_window")


def view(result: dict) -> list[dict]:
    """The pinned per-window fields of a draw result, in result order."""
    out = []
    for w in result["windows"]:
        missing = [f for f in WINDOW_FIELDS if f not in w]
        assert not missing, f"drawn window {w.get('window_id')!r} lacks {missing}"
        out.append({f: w[f] for f in WINDOW_FIELDS})
    return out
