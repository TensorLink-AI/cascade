"""Lane D public windows: the per-settlement draw from the public sources,
reading the drawn series and their per-window scores (LANE_D_PLAN rev6.18
A1, A2, A4; F3, W5k).

Each settlement draws, per public source (``gift_eval_pretrain``,
``rmisc``; :data:`~cascade.eval.lane_d_public_index.PUBLIC_SOURCES`),
:data:`PUBLIC_WINDOWS_PER_SOURCE` (5000, A1) windows from that source's
index (W6p) and the settlement block hash. The draw (:func:`draw_public`,
pinned in ``tests/accept/w5k/README.md``) is a pure function of (block
hash, index, horizon):

1. **Slots.** A series of ``L`` points has ``k = max(0, (L - m) // h)``
   non-overlapping target slots, ``m`` = :data:`PUBLIC_MIN_HISTORY` history
   points before the first cut, ``h`` = ``[lane_d] horizon``. ``k = 0``
   series are never drawn.
2. **Cells** = ``(family, freq)`` (``"<family>|<freq>"``); year-split folders
   share one family (W6p :func:`~cascade.eval.lane_d_public_index.family_of`).
   Capacity = the cell's series with ``k >= 1`` (GiftEvalPretrain: one window
   per series) or ``Σ k`` (RMISC, ``multi_window``: a small folder with few
   long series gives several windows per series).
3. **Quota**: the live draw's even split (:func:`~cascade.validator.lane_d_draw.split_quota`,
   W5j, the same code): smallest-capacity cell first, scarce cells taken
   whole, the surplus re-split.
4. **Picks**, cells in key order, in rounds ``r = 1, 2, …`` (round 1 only
   without ``multi_window``): the round's pool is the cell's series holding
   exactly ``r - 1`` windows and ``k >= r``; they are ranked by
   ``H(":pick:", ordinal, r)`` (ties by ordinal) and the first
   ``remaining`` taken, each getting one cut ``c`` (the target is ``[c, c +
   h)``, the history everything before it):

   - one window per series (GiftEvalPretrain): ``c = m + H(":cut:",
     ordinal, 1) mod (L - h - m + 1)``, uniform in ``[m, L - h]``;
   - several (RMISC): the series' ``k`` slots sit at ``c_j = m + o + j·h``
     (``j < k``) with one random offset ``o = H(":offset:", ordinal, 0) mod
     (L - m - k·h + 1)``; round ``r`` takes the unused slot at index
     ``H(":slot:", ordinal, r) mod (#unused)`` (unused slots ascending).
     Targets never overlap and every slot is reachable, so a series yields
     exactly ``k`` windows; together the cuts cover ``[m, L - h]``.

   The cell ends when its quota is met or its pool is empty.
5. ``H(kind, ordinal, r) = int(sha256(f"{seed}:{kind}:{ordinal}:{r}"))``
   with ``seed = f"{boundary_block_hash}:lane_d_public:{source}"``. Hash
   ranking (not a numpy stream) keeps the draw independent of the numpy
   version, so any auditor re-derives it from the index alone.

The same window list serves KING_D, the house and every challenger (one draw
per settlement, never per entry). The box draws from the index alone (no
pyarrow, no data). :func:`read_public_windows` reads ONLY the drawn series
(where the mirror lives) and builds each window: history = the series before
the cut (at most :data:`PUBLIC_MAX_HISTORY` points), target = the next ``h``
points; a ``(C, L)`` series is one multichannel window with F2's channel
rule (:mod:`cascade.eval.lane_d_channels`: empty channels dropped, at most
``[generator] max_channels`` kept by the block hash). A window is dropped,
for every model alike, when its target holds a non-finite value
(``public_nonfinite_target``), its finite history suffix is shorter than
``m`` (``public_short_history``), no channel can be scored
(``public_no_channel``) or its target has no scale (``public_no_scale``).

:func:`score_public_windows` scores each model's forecasts through F2's
per-window API (:func:`~cascade.eval.lane_d_scorer.score_window`) and
returns per-source per-window scores. Combining the halves is F4's.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from ..eval.lane_d_channels import apply_pick, channel_record, window_pick
from ..eval.lane_d_public_index import PUBLIC_SOURCES, iter_index, load_manifest, read_series
from ..eval.seasonality import get_seasonality
from .lane_d_draw import split_quota

PUBLIC_SCHEMA = "lane-d-public/1"
PUBLIC_SEED_TAG = "lane_d_public"
PUBLIC_WINDOWS_PER_SOURCE = 5000
PUBLIC_MIN_HISTORY = 128
PUBLIC_MAX_HISTORY = 131072
DROP_REASONS = ("public_nonfinite_target", "public_short_history", "public_no_channel",
                "public_no_scale")
# Owner 2026-11-27: a kept public window left out by the box so every
# (frequency class, band) cell of the public half holds the same count.
BALANCE_DROP = "public_cell_balance"


def public_seed(boundary_block_hash: str, source: str) -> str:
    return f"{boundary_block_hash}:{PUBLIC_SEED_TAG}:{source}"


def balance_public(windows: list[dict], *, boundary_block_hash: str, min_cell: int
                   ) -> tuple[list[dict], list[dict]]:
    """Owner 2026-11-27: equal counts per ``(frequency class, band)`` cell of
    the kept public windows (``{window_id, freq, band, ...}``): a cell below
    ``min_cell`` is left out, every other keeps its ``m`` windows ranked first
    by ``sha256(<block hash>:lane_d_public:balance:<window_id>)``.
    ``(kept, [{window_id, reason: BALANCE_DROP}])``, both by window id."""
    from ..shared.lane_d_horizons import cell_class

    seed = public_seed(str(boundary_block_hash), "balance")
    by: dict[str, list[dict]] = {}
    for w in windows:
        by.setdefault(f"{cell_class(w['freq'])}|{w['band']}", []).append(w)
    big = {k: v for k, v in by.items() if len(v) >= int(min_cell)}
    m = min((len(v) for v in big.values()), default=0)
    keep = set()
    for v in big.values():
        ranked = sorted(v, key=lambda w: (hashlib.sha256(
            f"{seed}:{w['window_id']}".encode()).hexdigest(), w["window_id"]))
        keep |= {w["window_id"] for w in ranked[:m]}
    kept = sorted((w for w in windows if w["window_id"] in keep), key=lambda w: w["window_id"])
    gone = sorted(({"window_id": w["window_id"], "reason": BALANCE_DROP} for w in windows
                   if w["window_id"] not in keep), key=lambda x: x["window_id"])
    return kept, gone


def series_slots(length: int, horizon: int, min_history: int) -> int:
    return max(0, (int(length) - int(min_history)) // int(horizon))


def _hint(seed: str, kind: str, ordinal: int, r: int) -> int:
    return int(hashlib.sha256(f"{seed}:{kind}:{ordinal}:{r}".encode()).hexdigest(), 16)


def pick_cut(seed: str, ordinal: int, r: int, length: int, horizon: int, min_history: int,
             multi: bool, used: list[int]) -> int:
    """The cut of series ``ordinal`` in round ``r`` (module docstring, step
    4). ``used``: the cuts already drawn from it (multi-window only)."""
    L, h, m = int(length), int(horizon), int(min_history)
    if not multi:
        return m + _hint(seed, "cut", ordinal, 1) % (L - h - m + 1)
    k = series_slots(L, h, m)
    o = _hint(seed, "offset", ordinal, 0) % (L - m - k * h + 1)
    unused = [m + o + j * h for j in range(k) if m + o + j * h not in set(used)]
    return unused[_hint(seed, "slot", ordinal, r) % len(unused)]


def cell_of(row: Mapping) -> str:
    return f"{row['family']}|{row['freq']}"


def series_band(seed: str, row: Mapping, bands, *, min_history: int,
                max_horizon: int) -> tuple[str, int] | None:
    """Owner 2026-11-27: the series' drawn band and its ``h``: uniform over
    the GIFT-Eval bands of its ``freq`` that fit its length (at least one
    target slot after ``min_history``; the whole history is public, so every
    band that fits is allowed), by ``H(":band:", ordinal, 0)``; ``None``
    when none fits."""
    from ..shared.lane_d_horizons import bands_for

    fits = [(b, h) for b, h in bands_for(row["freq"], bands, max_horizon=max_horizon).items()
            if series_slots(row["length"], h, min_history) >= 1]
    if not fits:
        return None
    return fits[_hint(seed, "band", int(row["ordinal"]), 0) % len(fits)]


def draw_public(index_dir, *, source: str, boundary_block_hash: str, horizon: int,
                n_windows: int = PUBLIC_WINDOWS_PER_SOURCE,
                min_history: int = PUBLIC_MIN_HISTORY, bands=None,
                max_horizon: int = 2048) -> dict:
    """The settlement's public draw for ``source`` from its index at
    ``index_dir`` (two streaming passes: cells, then the drawn rows). Plain
    JSON; ``windows`` sorted by ``window_id``.

    ``bands`` (owner 2026-11-27; ``None`` = every window at ``horizon``):
    each series first draws its band (:func:`series_band`); its slots,
    capacity and cuts then use that band's ``h`` (the record's ``horizon``
    stays ``[lane_d] horizon``; each window carries its ``h`` and ``band``)."""
    spec = PUBLIC_SOURCES[source]
    manifest = load_manifest(index_dir)
    if manifest["source"] != source:
        raise ValueError(f"{index_dir}: index of {manifest['source']!r}, not {source!r}")
    h, m, multi = int(horizon), int(min_history), bool(spec.multi_window)
    seed = public_seed(str(boundary_block_hash), source)
    cells: dict[str, list[tuple[int, int]]] = {}
    band_of: dict[int, tuple[str, int]] = {}
    for row in iter_index(index_dir):
        if bands:
            got = series_band(seed, row, bands, min_history=m, max_horizon=int(max_horizon))
            if got is None:
                continue
            band_of[int(row["ordinal"])] = got
        k = series_slots(row["length"], band_of.get(int(row["ordinal"]), (None, h))[1], m)
        if k >= 1:
            cells.setdefault(cell_of(row), []).append((int(row["ordinal"]), int(row["length"])))

    def h_of(o: int) -> int:
        return band_of[o][1] if bands else h

    caps = {c: (sum(series_slots(L, h_of(o), m) for o, L in ms) if multi else len(ms))
            for c, ms in cells.items()}
    quota = split_quota(caps, int(n_windows))
    picks: dict[int, list[int]] = {}
    for cell in sorted(cells):
        members = sorted(cells[cell])
        length = dict(members)
        remaining, r = int(quota.get(cell, 0)), 1
        while remaining > 0:
            pool = [o for o, L in members
                    if len(picks.get(o, ())) == r - 1 and series_slots(L, h_of(o), m) >= r]
            if not pool:
                break
            ranked = sorted(pool, key=lambda o: (_hint(seed, "pick", o, r), o))
            for o in ranked[:remaining]:
                cut = pick_cut(seed, o, r, length[o], h_of(o), m, multi, picks.get(o, []))
                picks.setdefault(o, []).append(cut)
                remaining -= 1
            if not multi:
                break
            r += 1
    windows = []
    for row in iter_index(index_dir):
        o = int(row["ordinal"])
        for cut in picks.get(o, ()):
            hw = h_of(o)
            windows.append({"window_id": f"{source}:{row['series_id']}:{cut}:h{hw}",
                            "ordinal": o, "series_id": row["series_id"],
                            "dataset": row["dataset"], "family": row["family"],
                            "freq": row["freq"], "cut": int(cut), "h": hw,
                            "length": int(row["length"]), "n_channels": int(row["n_channels"]),
                            **({"band": band_of[o][0]} if bands else {})})
    windows.sort(key=lambda w: w["window_id"])
    return {"source": source, "repo": spec.repo, "revision": spec.revision,
            "index_sha256": manifest["index_sha256"], "files_sha256": manifest["files_sha256"],
            "seed": seed, "boundary_block_hash": str(boundary_block_hash),
            "n_windows": int(n_windows), "horizon": h, "min_history": m,
            "multi_window": multi, "capacity": dict(sorted(caps.items())),
            "quota": dict(sorted(quota.items())), "windows": windows}


def public_record(draws: list[dict], dropped: Mapping[str, list[dict]] | None = None) -> dict:
    """The signed report's ``public`` block (W7a): per source the pinned
    revision, ``index_sha256``, ``files_sha256``, the draw parameters and the
    drawn window ids (with their cell, cut and series length, so the audit
    re-derives the draw from the index), plus the windows dropped at read
    time (``{window_id, reason}``)."""
    keep = ("window_id", "ordinal", "family", "freq", "cut", "h", "length", "band")
    out = []
    for d in sorted(draws, key=lambda d: d["source"]):
        out.append({**{k: d[k] for k in ("source", "repo", "revision", "index_sha256",
                                          "files_sha256", "seed", "n_windows", "horizon",
                                          "min_history", "multi_window", "capacity", "quota")},
                    "windows": [{k: w[k] for k in keep if k in w} for w in d["windows"]],
                    "dropped": sorted((dict(x) for x in (dropped or {}).get(d["source"], [])),
                                      key=lambda x: x["window_id"])})
    return {"schema": PUBLIC_SCHEMA, "sources": out}


def seasonal_period(freq: str) -> int:
    f = str(freq)
    p = get_seasonality(f)
    if p == 1 and f[-1:].islower() and f[-1:] in "hdwts":
        p = get_seasonality(f[:-1] + f[-1].upper())
    return int(p)


def _finite_suffix(hist: np.ndarray) -> int:
    """Start index of the longest all-finite suffix (over every channel)."""
    ok = np.isfinite(hist).all(axis=0)
    bad = np.flatnonzero(~ok)
    return int(bad[-1]) + 1 if bad.size else 0


def build_window(series: np.ndarray, w: Mapping, *, block_hash: str | None, max_channels: int,
                 min_history: int = PUBLIC_MIN_HISTORY) -> tuple[dict | None, str | None]:
    """One drawn window from its ``(C, L)`` series: ``(window, None)`` or
    ``(None, drop reason)``. Deterministic in (data, block hash)."""
    a = np.asarray(series, dtype=np.float64)
    a = a.reshape(1, -1) if a.ndim == 1 else a
    cut, h = int(w["cut"]), int(w["h"])
    hist = a[:, max(0, cut - PUBLIC_MAX_HISTORY):cut]
    tgt = a[:, cut:cut + h]
    base = {"window_id": w["window_id"], "series_id": w["series_id"],
            "source": str(w["window_id"]).split(":", 1)[0], "family": w["family"],
            "freq": w["freq"], "seasonal_period": seasonal_period(w["freq"])}
    rec = None
    if a.shape[0] >= 2:
        pick = window_pick({**base, "history": hist, "target": tgt}, block_hash, max_channels)
        if pick is None or not pick.scorable:
            return None, "public_no_channel"
        sliced = apply_pick({**base, "history": hist, "target": tgt}, pick)
        if sliced is None:
            return None, "public_no_channel"
        rec = channel_record(pick)
        hist = np.atleast_2d(sliced["history"])
        tgt = np.atleast_2d(sliced["target"])
    if tgt.shape[1] != h or not np.isfinite(tgt).all():
        return None, "public_nonfinite_target"
    hist = hist[:, _finite_suffix(hist):]
    if hist.shape[1] < int(min_history):
        return None, "public_short_history"
    if not (np.abs(tgt).sum(axis=1) > 0).any():
        return None, "public_no_scale"
    uni = hist.shape[0] == 1
    return {**base, "history": hist[0].copy() if uni else hist.copy(),
            "target": tgt[0].copy() if uni else tgt.copy(), "channels": rec}, None


def _ordered_map(fn, items: list, workers: int):
    """``fn`` over ``items`` on at most ``workers`` threads, yielded in
    ``items`` order with at most ``2 * workers`` results in flight (bounded
    memory). The first exception cancels what has not started and is
    raised."""
    if workers <= 1 or len(items) <= 1:
        for x in items:
            yield fn(x)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    ex = ThreadPoolExecutor(max_workers=int(workers))
    pending: deque = deque()
    it = iter(items)
    try:
        for x in it:
            pending.append(ex.submit(fn, x))
            if len(pending) >= 2 * int(workers):
                break
        while pending:
            got = pending.popleft().result()
            for x in it:
                pending.append(ex.submit(fn, x))
                break
            yield got
    finally:
        for f in pending:
            f.cancel()
        ex.shutdown(wait=True, cancel_futures=True)


def read_public_windows(draw: dict, *, index_dir, data_root=None, block_hash: str | None,
                        max_channels: int, reader=None,
                        workers: int = 1) -> tuple[list[dict], list[dict]]:
    """Read ONLY the drawn series of ``draw`` and build its windows:
    ``(windows, dropped)``, both in ``window_id`` order. The series come
    from ``reader(row) -> (C, L)`` when given (the Hippius mirror,
    :meth:`cascade.eval.lane_d_public_mirror.PublicMirror.reader`), else
    from the local mirror at ``data_root``. Needs ``pyarrow`` (the eval host
    / builder, never the box). ``index_dir`` is the source's index (the one
    ``draw`` was drawn from). ``workers > 1`` reads that many series at once
    (a bounded thread pool; the result is identical to the serial read: the
    windows, the drops and their order depend on the data only)."""
    if reader is None and data_root is None:
        raise ValueError("read_public_windows: a data_root or a reader is required")
    source = draw["source"]
    by_series: dict[str, list[dict]] = {}
    for w in draw["windows"]:
        by_series.setdefault(w["series_id"], []).append(w)
    rows = {}
    want = {int(w["ordinal"]) for w in draw["windows"]}
    if load_manifest(index_dir)["index_sha256"] != draw["index_sha256"]:
        raise ValueError(f"{index_dir}: not the index the draw was made from")
    for row in iter_index(index_dir):
        if int(row["ordinal"]) in want:
            rows[row["series_id"]] = row
    windows, dropped = [], []
    sids = sorted(by_series)

    def load(sid):
        row = rows[sid]
        return (reader(row) if reader is not None
                else read_series(data_root, row, source=source))

    for sid, series in zip(sids, _ordered_map(load, sids, int(workers)), strict=True):
        for w in by_series[sid]:
            win, why = build_window(series, w, block_hash=block_hash, max_channels=max_channels,
                                    min_history=int(draw["min_history"]))
            if win is None:
                dropped.append({"window_id": w["window_id"], "reason": why})
            else:
                windows.append(win)
    windows.sort(key=lambda w: w["window_id"])
    dropped.sort(key=lambda d: d["window_id"])
    return windows, dropped


def score_public_windows(windows: list[dict], forecasts: Mapping[str, Mapping]) -> dict:
    """Per-source per-window scores through F2's per-window API:
    ``{digest: {source: [{window_id, family, crps, mase}]}}`` (windows in
    ``window_id`` order). Every digest must forecast every window (the same
    windows for KING_D, the house and every challenger); a missing or
    malformed forecast is a ``ValueError``."""
    from ..eval.lane_d_scorer import score_window

    out: dict[str, dict[str, list[dict]]] = {}
    ordered = sorted(windows, key=lambda w: w["window_id"])
    for dg in sorted(forecasts):
        fc = forecasts[dg]
        per: dict[str, list[dict]] = {}
        for w in ordered:
            if w["window_id"] not in fc:
                raise ValueError(f"{dg}: no forecast for public window {w['window_id']}")
            crps, mase = score_window(np.asarray(fc[w["window_id"]], dtype=np.float64),
                                      w["target"], w["history"], int(w["seasonal_period"]))
            per.setdefault(w["source"], []).append(
                {"window_id": w["window_id"], "family": w["family"], "crps": float(crps),
                 "mase": float(mase)})
        out[dg] = per
    return out


__all__ = ["BALANCE_DROP", "DROP_REASONS", "balance_public", "series_band", "PUBLIC_MAX_HISTORY", "PUBLIC_MIN_HISTORY", "PUBLIC_SCHEMA",
           "PUBLIC_SEED_TAG", "PUBLIC_WINDOWS_PER_SOURCE", "build_window", "cell_of",
           "draw_public", "pick_cut", "public_record", "public_seed", "read_public_windows",
           "score_public_windows", "seasonal_period", "series_slots"]
