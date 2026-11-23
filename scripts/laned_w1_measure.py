#!/usr/bin/env python3
"""Lane D S1 measurement W1 — forward-eligible window counts from consecutive
pinned daily pool snapshots, as a SNAPSHOT-DIFF BOUND.

REPORT-ONLY and read-only: it reads unpacked snapshot directories and never
touches ``cascade/pool/`` or any bucket.

Pool records carry no timestamps and no first-seen times, so nothing here is
"measured first-seen". Instead, for each series present in consecutive
snapshots the arrays are aligned (the older array's suffix against the newer
array's prefix) to find how many NEW TRAILING points appeared; changed values
inside the overlap are counted separately as "revised". Every point is given
the build time of the first snapshot it appears in (chained across the
sequence). A target window of ``h`` = 64 points (the scorer's cut: the
series' last 64 points) is forward-eligible for a commit at time T when all
64 points first appear in a snapshot built after ``T + margin`` (default
6 h). That assigns each point the LATEST time it could have first appeared,
and series new to the pool (or that cannot be aligned) count as entirely
new, so every count is labelled "snapshot-diff bound (upper bound on eligible
counts)".

Reported per snapshot pair and per frequency (metadata ``freq``): eligible
window counts and distinct sources over a sweep of T, and whether the floors
(``min_windows`` 1000, ``min_clusters`` 20) are met under the bound. Also the
series / sources with history >= L + 64 for L in 4096 / 8192 / 16384 and
32768 / 65536 / 131072 — the pool truncates history to ``context_length +
keep_horizon`` (4096 + 720 on mainnet), so those counts are a LOWER bound of
what the forge mirror holds, and any L + 64 above the truncation length is
"not answerable from pool snapshots: needs forge mirror (TSFORGE_S3)".

Input: ``--snapshots-dir`` holding >= 2 unpacked snapshots, one sub-directory
each (``*.npy`` + ``metadata.json`` [+ ``provenance.json``], the layout
:mod:`cascade.validator.pool` reads). Each needs a build time:
``--build-time NAME=ISO8601`` (preferred), else the pool index
(``--index pool/index.json``: the entry whose ``effective_block`` equals the
sub-directory name, its ``as_of`` DATE taken as 00:00Z — day granularity,
flagged), else the snapshot's ``provenance.json`` ``summary.as_of`` (same
caveat). Each snapshot's sha256: the tree digest of the unpacked directory
(computed) and the tar sha256 from ``--snapshot-sha256 NAME=HEX`` or the
index entry (recorded as given, not re-verified against the directory).

Usage:
  laned_w1_measure.py --snapshots-dir SNAPS --out w1.json \\
      [--build-time 9493200=2026-11-19T02:10:00Z ...] [--index index.json] \\
      [--snapshot-sha256 9493200=<hex> ...] [--margin-hours 6] \\
      [--commit-time 2026-11-19T00:00:00Z ...] [--t-step-hours 6] \\
      [--forge-dir ./tsforge --embargo-hours 48 --min-fresh-share 0.5 \\
       --max-windows 5000 --sweep-days 3]

FORGE MIRROR (``--forge-dir``, label "measured from forge mirror"; either or
both of ``--snapshots-dir`` / ``--forge-dir``): the ``forge_mirror`` key holds,
per cadence bucket (10s, 1min, 5min, 10min, 15min, 30min, hourly, daily,
slower), sources, series and the first-seen-proxy latency; for commit times T
swept over the mirror's last ``--sweep-days`` the fresh and embargo window
counts (b = T + ``--forward-margin-hours`` (0, rev6.14), embargo e =
``--embargo-hours``; judging from T + ``--eval-delay-hours`` (6)) at the clocks T+6 h,
T+9 h, T+12 h, T+24 h, the rev6.12 (b) set rule (fresh first, embargo trimmed
to the minimum fresh share, ``--max-windows`` cap, floors) and its median/min
over T; and the long-history inventory per context 4096…131072 (+ horizon)
against the arming gate (``min_long_series`` from LaneDConfig, ≥ 20 sources).
The forge block is described next to its code below.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import socket
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SCHEMA = "cascade-laned-w1/1"
BOUND_LABEL = "snapshot-diff bound (upper bound on eligible counts)"
NOT_ANSWERABLE = "not answerable from pool snapshots: needs forge mirror (TSFORGE_S3)"
DEFAULT_HORIZON = 64
DEFAULT_MIN_CONTEXT = 64          # horizon_draw's scored-path eligibility floor
DEFAULT_MARGIN_HOURS = 6.0
DEFAULT_MIN_WINDOWS = 1000
DEFAULT_MIN_CLUSTERS = 20
HISTORY_LEVELS = (4096, 8192, 16384, 32768, 65536, 131072)
MAINNET_KEEP_LENGTH = 4096 + 720  # context_length + max(scored_horizons)
ANCHORS = 8
MAX_MATCHES_PER_ANCHOR = 256
MAX_CANDIDATES = 64
MIN_MATCH_FRAC = 0.5
MIN_OVERLAP = 16

# How a point got its first-seen time (chained across snapshots).
CARRIED, APPENDED, REVISED, BACKFILLED, APPEARED, UNALIGNED = range(6)


# ── provenance ───────────────────────────────────────────────────────────────


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path | str) -> str:
    """sha256 over every file under ``root`` (sorted relative path, NUL, size,
    NUL, bytes) — a content digest of the UNPACKED directory."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0" + str(p.stat().st_size).encode() + b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def provenance(argv: Sequence[str] | None) -> dict:
    args = list(sys.argv[1:] if argv is None else argv)
    return {
        "command_line": shlex.join([sys.argv[0] or "laned_w1_measure.py", *args]),
        "host": socket.gethostname(),
        "timestamp_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def parse_time(s: str) -> float:
    """ISO-8601 → UTC epoch seconds (a naive time is taken as UTC)."""
    t = dt.datetime.fromisoformat(str(s).strip())
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    return t.timestamp()


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── alignment ────────────────────────────────────────────────────────────────


def align(a: np.ndarray, b: np.ndarray) -> dict | None:
    """Align an older array ``a`` with a newer ``b`` (both ``(L,)`` or
    ``(C, L)``): b's last ``k`` points are new; b[Lb-k-ov : Lb-k] overlaps
    a[La-ov:]. Candidates for k come from exact matches of a's last
    ``ANCHORS`` values inside b; the candidate with the highest overlap match
    fraction wins (ties → smaller k). ``None`` = unaligned (no candidate with
    >= ``MIN_OVERLAP`` points matching >= ``MIN_MATCH_FRAC``).

    Returns ``{"k", "ov", "eq"}`` with ``eq`` the per-overlap-point equality
    mask (False = revised)."""
    a2 = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b2 = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a2.shape[0] != b2.shape[0]:
        return None
    la, lb = a2.shape[-1], b2.shape[-1]
    votes: Counter = Counter()
    for t in range(min(ANCHORS, la)):
        v = a2[:, la - 1 - t]
        hits = np.flatnonzero(np.all(b2 == v[:, None], axis=0))
        if hits.size == 0 or hits.size > MAX_MATCHES_PER_ANCHOR:
            continue
        for j in hits.tolist():
            k = lb - 1 - j - t
            if 0 <= k <= lb:
                votes[k] += 1
    best = None
    for k, _v in votes.most_common(MAX_CANDIDATES):
        ov = min(la, lb - k)
        if ov < MIN_OVERLAP:
            continue
        eq = np.all(a2[:, la - ov:] == b2[:, lb - k - ov: lb - k], axis=0)
        frac = float(eq.mean())
        key = (frac, -k)
        if best is None or key > best[0]:
            best = (key, {"k": int(k), "ov": int(ov), "eq": eq})
    if best is None or best[0][0] < MIN_MATCH_FRAC:
        return None
    return best[1]


# ── snapshot loading ─────────────────────────────────────────────────────────


def _kv(specs: Sequence[str] | None, what: str) -> dict[str, str]:
    out = {}
    for s in specs or ():
        k, sep, v = s.partition("=")
        if not sep or not k or not v:
            raise SystemExit(f"{what} must be NAME=VALUE; got {s!r}")
        out[k] = v
    return out


def _index_entries(index_path: str | None) -> dict[str, dict]:
    if not index_path:
        return {}
    doc = json.loads(Path(index_path).read_text(encoding="utf-8"))
    out = {}
    for e in doc.get("snapshots", []):
        block = e.get("effective_block", e.get("effective_round"))
        if block is not None:
            out[str(int(block))] = e
        if e.get("key"):
            out.setdefault(Path(str(e["key"])).name.split(".")[0], e)
    return out


def snapshot_info(d: Path, *, build_times: dict, shas: dict, index: dict) -> dict:
    """Name, build time (+ its source), digests and truncation length for one
    unpacked snapshot directory."""
    name = d.name
    prov = {}
    pf = d / "provenance.json"
    if pf.is_file():
        try:
            prov = json.loads(pf.read_text(encoding="utf-8"))
        except ValueError:
            prov = {}
    entry = index.get(name)
    if name in build_times:
        t, src = parse_time(build_times[name]), "--build-time"
    elif entry and entry.get("as_of"):
        t = parse_time(str(entry["as_of"])[:10] + "T00:00:00+00:00")
        src = "pool index as_of (DATE only, taken as 00:00Z — day granularity)"
    elif (prov.get("summary") or {}).get("as_of"):
        t = parse_time(str(prov["summary"]["as_of"])[:10] + "T00:00:00+00:00")
        src = "provenance.json summary.as_of (DATE only, taken as 00:00Z — day granularity)"
    else:
        raise SystemExit(f"{d}: no build time — pass --build-time {name}=ISO8601 or --index")
    cfg = prov.get("config") or {}
    if cfg.get("context_length"):
        kh = cfg.get("keep_horizon") or cfg.get("horizon") or 0
        keep, keep_src = int(cfg["context_length"]) + int(kh), "provenance.json config"
    else:
        keep, keep_src = MAINNET_KEEP_LENGTH, "assumed mainnet 4096 + 720 (no provenance config)"
    tar_sha = shas.get(name) or (entry or {}).get("sha256")
    return {"name": name, "dir": str(d), "build_time": iso(t), "build_ts": t,
            "build_time_source": src, "tree_sha256": tree_sha256(d),
            "tar_sha256": tar_sha or "not run: no --snapshot-sha256 / index entry",
            "tar_sha256_source": ("--snapshot-sha256" if name in shas else
                                  "pool index (recorded, not re-verified)" if tar_sha else None),
            "keep_length": keep, "keep_length_source": keep_src}


def load_snapshot(d: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    from cascade.validator.calibration import load_pool_series

    series, ids, md = load_pool_series(d)
    return dict(zip(ids, series, strict=True)), dict(zip(ids, md, strict=True))


# ── diff + chained first-seen ────────────────────────────────────────────────


def advance(prev: dict | None, series: dict[str, np.ndarray], t_build: float) -> tuple[dict, dict]:
    """Chain first-seen times into a new snapshot. ``prev`` maps id →
    ``(array, first_seen, kind)`` of the previous snapshot (``None`` for the
    first: its points are CARRIED at its build time). Returns the new state
    and the transition counts."""
    state: dict = {}
    tr = Counter()
    for sid, b in series.items():
        lb = np.atleast_2d(b).shape[-1]
        if prev is None:
            state[sid] = (b, np.full(lb, t_build), np.full(lb, CARRIED, dtype=np.int8))
            continue
        if sid not in prev:
            tr["appeared"] += 1
            state[sid] = (b, np.full(lb, t_build), np.full(lb, APPEARED, dtype=np.int8))
            continue
        a, fs_a, kind_a = prev[sid]
        al = align(a, b)
        if al is None:
            tr["unaligned"] += 1
            state[sid] = (b, np.full(lb, t_build), np.full(lb, UNALIGNED, dtype=np.int8))
            continue
        tr["common"] += 1
        la = np.atleast_2d(a).shape[-1]
        k, ov, eq = al["k"], al["ov"], al["eq"]
        fs = np.full(lb, t_build)
        kind = np.full(lb, APPENDED, dtype=np.int8)
        lo, hi = lb - k - ov, lb - k
        fs[lo:hi] = np.where(eq, fs_a[la - ov:], t_build)
        kind[lo:hi] = np.where(eq, kind_a[la - ov:], REVISED)
        kind[:lo] = BACKFILLED
        n_rev = int((~eq).sum())
        if k:
            tr["appended_series"] += 1
            tr["appended_points"] += k
        if n_rev:
            tr["revised_series"] += 1
            tr["revised_points"] += n_rev
        if ov < la:
            tr["retruncated_series"] += 1
            tr["dropped_head_points"] += la - ov
        if lo > 0:
            tr["backfilled_series"] += 1
            tr["backfilled_head_points"] += lo
        state[sid] = (b, fs, kind)
    if prev is not None:
        tr["disappeared"] = sum(1 for sid in prev if sid not in series)
    return state, dict(tr)


def _source(sid: str, md: dict) -> str:
    s = md.get("source")
    return str(s) if s else f"__series:{sid}"


def _floors(windows: int, sources: int, min_windows: int, min_clusters: int) -> dict:
    return {"min_windows": min_windows, "min_clusters": min_clusters,
            "met": bool(windows >= min_windows and sources >= min_clusters),
            "label": "under the snapshot-diff upper bound"}


def eligibility(state: dict, metadata: dict, commit_ts: float, *, margin_s: float,
                horizon: int, min_context: int, min_windows: int, min_clusters: int) -> dict:
    """Eligible window counts (total and per freq) for one commit time."""
    cutoff = commit_ts + margin_s
    per: dict[str, dict] = {}
    tot = {"windows": 0, "sources": set(), "by_kind": Counter()}
    for sid, (_b, fs, kind) in state.items():
        if fs.shape[0] < horizon + min_context:
            continue
        if not np.all(fs[-horizon:] > cutoff):
            continue
        k = kind[-horizon:]
        if np.any(k == UNALIGNED):
            why = "unaligned_series"
        elif np.any(k == APPEARED):
            why = "appeared_series"
        elif np.any(k == REVISED) or np.any(k == BACKFILLED):
            why = "with_revised_or_backfilled"
        else:
            why = "appended"
        md = metadata.get(sid, {})
        freq = str(md.get("freq", "") or "?")
        src = _source(sid, md)
        cell = per.setdefault(freq, {"windows": 0, "sources": set(), "by_kind": Counter()})
        for c in (cell, tot):
            c["windows"] += 1
            c["sources"].add(src)
            c["by_kind"][why] += 1

    def fin(c):
        n_src = len(c["sources"])
        return {"windows": c["windows"], "sources": n_src, "by_kind": dict(c["by_kind"]),
                "floors": _floors(c["windows"], n_src, min_windows, min_clusters)}

    return {"commit_time": iso(commit_ts), "eligible_after": iso(cutoff), "label": BOUND_LABEL,
            "total": fin(tot), "per_freq": {f: fin(c) for f, c in sorted(per.items())}}


def long_history(series: dict[str, np.ndarray], metadata: dict, keep_length: int,
                 horizon: int) -> dict:
    out = {"truncation_length": keep_length,
           "note": (f"the pool truncates history to {keep_length} points (context_length + "
                    "keep_horizon): counts are a LOWER bound of what the forge mirror holds")}
    lengths = {sid: np.atleast_2d(s).shape[-1] for sid, s in series.items()}
    for L in HISTORY_LEVELS:
        need = L + horizon
        key = f">={L}+{horizon}"
        if need > keep_length:
            out[key] = NOT_ANSWERABLE
            continue
        ids = [sid for sid, n in lengths.items() if n >= need]
        out[key] = {"series": len(ids),
                    "sources": len({_source(s, metadata.get(s, {})) for s in ids}),
                    "label": "lower bound (pool-truncated)"}
    return out


def commit_sweep(t_first: float, t_last: float, step_h: float) -> list[float]:
    step = step_h * 3600.0
    if step <= 0:
        raise ValueError("--t-step-hours must be positive")
    out, t = [], t_first
    while t < t_last:
        out.append(t)
        t += step
    return out


def measure(snaps: list[dict], *, commit_times: list[float] | None, step_h: float,
            margin_h: float, horizon: int, min_context: int, min_windows: int,
            min_clusters: int, loader=load_snapshot) -> dict:
    """The W1 report body for build-time-ordered ``snaps`` (``snapshot_info``
    dicts). Two snapshots are in memory at a time."""
    if len(snaps) < 2:
        raise ValueError("need >= 2 snapshots")
    snaps = sorted(snaps, key=lambda s: s["build_ts"])
    if len({s["build_ts"] for s in snaps}) != len(snaps):
        raise ValueError("two snapshots share a build time — consecutive order is ambiguous")
    t0 = snaps[0]["build_ts"]
    sweep = sorted(commit_times) if commit_times else commit_sweep(
        t0, snaps[-1]["build_ts"], step_h)
    before = [t for t in sweep if t < t0]
    sweep = [t for t in sweep if t >= t0]
    state = None
    out_snaps, pairs = [], []
    for i, s in enumerate(snaps):
        series, md = loader(Path(s["dir"]))
        snap = {k: v for k, v in s.items() if k != "build_ts"}
        snap["n_series"] = len(series)
        snap["long_history"] = long_history(series, md, int(s["keep_length"]), horizon)
        out_snaps.append(snap)
        state, tr = advance(state, series, s["build_ts"])
        if i == 0:
            continue
        ts = [t for t in sweep if t < s["build_ts"]]
        pairs.append({
            "from": snaps[i - 1]["name"], "to": s["name"],
            "from_build_time": iso(snaps[i - 1]["build_ts"]), "to_build_time": iso(s["build_ts"]),
            "transition": tr, "label": BOUND_LABEL,
            "eligibility": [eligibility(state, md, t, margin_s=margin_h * 3600.0,
                                        horizon=horizon, min_context=min_context,
                                        min_windows=min_windows, min_clusters=min_clusters)
                            for t in ts]
            or f"not run: no commit time in [{iso(t0)}, {iso(s['build_ts'])})",
        })
        del series
    return {"label": BOUND_LABEL, "snapshots": out_snaps, "pairs": pairs,
            "commit_times_dropped_before_first_snapshot": [iso(t) for t in before]}


# ── forge mirror (measured) ──────────────────────────────────────────────────
#
# ``--forge-dir`` reads the synced tsbench-forge mirror (``sources.yaml`` +
# ``data/<feed>/<YYYY-MM-DD>.parquet``) with the pool reader's own helpers
# (:class:`cascade.pool.sources.tsbench_forge.TsbenchForgeSource`: file list,
# timestamp parsing + as_of cutoff, densest/named value column, freshest
# contiguous segment, frequency mapping, staleness rule), so a "series" here
# is the series the reader would emit. Data times are the parquet
# ``timestamp`` column. A point's first-seen time is not recorded in the
# mirror (that is W6fs), so it is bracketed by a PROXY from the first
# partition holding the point:
#   lower = max(data time, 00:00Z of that partition's date)
#   upper = the partition object's time: ``min`` (default) = the earlier of
#           its file mtime (``aws s3 sync`` sets it to the object's
#           LastModified) and the end of its partition date; ``object`` = the
#           mtime only; ``partition`` = the end of its partition date only.
# Window eligibility is counted CONSERVATIVELY under the bracket: the "not
# seen before" conditions use the lower bound and "available by the clock"
# uses the upper bound.

FORGE_LABEL = "measured from forge mirror"
FORGE_NOTE = ("first-seen is a PROXY (partition date / mirror object time), not the W6fs "
              "record; eligibility uses the conservative side of the bracket")
DEFAULT_EMBARGO_HOURS = 48.0
DEFAULT_FORWARD_MARGIN_HOURS = 0.0   # plan rev6.14: fresh = data time and first-seen > T + 0
DEFAULT_EVAL_DELAY_HOURS = 6.0       # plan rev6.14: judging starts at T + 6 h
DEFAULT_MIN_FRESH_SHARE = 0.5
DEFAULT_MAX_WINDOWS = 5000
DEFAULT_SWEEP_DAYS = 3.0
CLOCK_HOURS = (6, 9, 12, 24)
FIRST_SEEN_CLOCKS = ("min", "object", "partition")
# Cadence buckets (raw catalog cadence in seconds, [lo, hi)).
FREQ_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("10s", 0.0, 60.0), ("1min", 60.0, 300.0), ("5min", 300.0, 600.0),
    ("10min", 600.0, 900.0), ("15min", 900.0, 1800.0), ("30min", 1800.0, 3600.0),
    ("hourly", 3600.0, 86_400.0), ("daily", 86_400.0, 86_401.0),
    ("slower", 86_401.0, float("inf")),
)
_PART = "__w1_part"
_TS = "__w1_ts"


def freq_bucket(raw_sec: float | None) -> str:
    if raw_sec is None:
        return "unparseable"
    for name, lo, hi in FREQ_BUCKETS:
        if lo <= float(raw_sec) < hi:
            return name
    return "slower"


def lane_d_defaults() -> dict:
    """``min_long_series`` (the ``[lane_d]`` default) and the scorer's
    per-variant source floor, read from the code, never restated here."""
    from cascade.eval.lane_d_scorer import GATE_MIN_SOURCES
    from cascade.shared.config.lane_d import LaneDConfig

    return {"min_long_series": int(LaneDConfig().min_long_series),
            "min_sources": int(GATE_MIN_SOURCES)}


def _median(xs) -> float | None:
    xs = [float(x) for x in xs]
    return float(np.median(xs)) if xs else None


def _day_bounds(stamp: dt.date | None) -> tuple[float | None, float | None]:
    if stamp is None:
        return None, None
    t0 = dt.datetime.combine(stamp, dt.time(), tzinfo=dt.UTC).timestamp()
    return t0, t0 + 86_400.0


def _file_upper(mtime: float, day_end: float | None, clock: str) -> float:
    if clock == "object" or day_end is None:
        return float(mtime)
    if clock == "partition":
        return float(day_end)
    return float(min(mtime, day_end))


def mirror_listing(forge_dir: Path) -> list[dict]:
    """Every ``data/<feed>/*.parquet`` with size, mtime and partition date."""
    from cascade.pool.sources.tsbench_forge import _stem_date

    data = forge_dir / "data"
    out = []
    if not data.is_dir():
        return out
    for feed_dir in sorted(d for d in data.iterdir() if d.is_dir()):
        for p in sorted(feed_dir.glob("*.parquet")):
            st = p.stat()
            out.append({"feed": feed_dir.name, "path": p, "name": p.name, "size": st.st_size,
                        "mtime": float(st.st_mtime), "date": _stem_date(p)})
    return out


def listing_sha256(listing: list[dict]) -> str:
    h = hashlib.sha256()
    for f in listing:
        h.update(f"{f['feed']}/{f['name']}\0{f['size']}\0{f['mtime']:.3f}\n".encode())
    return h.hexdigest()


def _feed_series(src, pd, pq, files: list[dict], ctx, *, horizon: int, columns,
                 clock: str, max_series: int) -> tuple[list[dict], dict]:
    """The reader's series of one feed, each with its data times and the
    first-seen proxy bracket per point; plus the feed's file stats."""
    from cascade.pool.sources.tsbench_forge import MIN_CONTEXT

    frames, kept_files = [], []
    corrupt = 0
    for f in files:
        try:
            t = pq.read_table(f["path"]).to_pandas()
        except Exception:  # noqa: BLE001 — the reader skips a corrupt partition too
            corrupt += 1
            continue
        t[_PART] = len(kept_files)
        kept_files.append(f)
        frames.append(t)
    stats = {"partitions": len(files), "corrupt_partitions": corrupt}
    if not frames:
        return [], stats
    df = pd.concat(frames, ignore_index=True)
    df, ts = src._sorted_frame(pd, df, ctx)
    if ts is None:
        stats["no_timestamps"] = True
        return [], stats
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df = df.copy()
    df[_TS] = ((ts - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    panel_cols = [c for c in df.columns if c.startswith("_panel_")]
    keys = [_TS, *panel_cols]
    df["__w1_first"] = df.groupby(keys, sort=False)[_PART].transform("min")
    df = df.drop_duplicates(subset=keys, keep="last")     # the reader keeps the newest scrape
    lo_day = np.array([_day_bounds(f["date"])[0] if f["date"] else np.nan for f in kept_files])
    hi_file = np.array([_file_upper(f["mtime"], _day_bounds(f["date"])[1], clock)
                        for f in kept_files])
    excluded = {"timestamp", *panel_cols, _PART, _TS, "__w1_first"}
    groups = df.groupby(panel_cols, sort=True) if panel_cols else [((), df)]
    out = []
    for key, sub in groups:
        if max_series and len(out) >= max_series:
            break
        if columns:
            chans = src._named_channels(sub, columns[:1], excluded)
            if chans is None or int(np.isfinite(chans[0]).sum()) < 2:
                continue
        else:
            best_finite, best = src._densest_column(sub, excluded)
            if best is None or best_finite < 2:
                continue
        secs = sub[_TS].to_numpy(dtype=np.int64)
        lo, hi = src._freshest_segment(secs.astype("datetime64[s]"), len(secs),
                                       min_len=horizon + MIN_CONTEXT)
        secs = secs[lo:hi]
        first = sub["__w1_first"].to_numpy(dtype=np.int64)[lo:hi]
        day0 = lo_day[first]
        fs_lo = np.where(np.isnan(day0), secs, np.maximum(secs, np.nan_to_num(day0)))
        key_t = key if isinstance(key, tuple) else (key,)
        out.append({"panel": {c.removeprefix("_panel_"): str(v)
                              for c, v in zip(panel_cols, key_t, strict=False)},
                    "length": int(secs.size), "data_ts": secs.astype(np.float64),
                    "fs_lo": fs_lo.astype(np.float64), "fs_hi": hi_file[first],
                    "baseline": first == 0})
    stats["first_partition"] = str(kept_files[0]["date"] or kept_files[0]["name"])
    stats["last_partition"] = str(kept_files[-1]["date"] or kept_files[-1]["name"])
    stats["mirror_end"] = float(hi_file.max())
    return out, stats


def _latency(series: list[dict]) -> dict:
    """Mirror latency of the feed's NON-baseline points (first seen in a
    later partition than the feed's first one; the first partition re-delivers
    history), in hours: upper = proxy upper − data time, lower = max(0, proxy
    lower − data time)."""
    up = [s["fs_hi"][~s["baseline"]] - s["data_ts"][~s["baseline"]] for s in series]
    dn = [s["fs_lo"][~s["baseline"]] - s["data_ts"][~s["baseline"]] for s in series]
    up = np.concatenate(up) if up else np.empty(0)
    dn = np.concatenate(dn) if dn else np.empty(0)
    if not up.size:
        return {"points": 0, "median_upper_hours": None, "p90_upper_hours": None,
                "median_lower_hours": None,
                "note": "not run: no point first seen after the feed's first partition"}
    return {"points": int(up.size),
            "median_upper_hours": float(np.median(up)) / 3600.0,
            "p90_upper_hours": float(np.quantile(up, 0.9)) / 3600.0,
            "median_lower_hours": float(np.median(np.maximum(dn, 0.0))) / 3600.0,
            "max_lead_hours": float(max(0.0, -float(up.min()))) / 3600.0}


def load_forge(forge_dir: Path | str, *, horizon: int, clock: str = "min",
               max_series_per_source: int = 200, max_stale_days: int = 4) -> dict:
    """Read the mirror once: per catalog feed its reader series (data times +
    first-seen bracket) and stats. Feeds the reader would not emit (disabled,
    slower than daily or unparseable cadence, stale, no files) are listed with
    the reason and contribute no series."""
    from cascade.pool.source import HarvestContext
    from cascade.pool.sources.tsbench_forge import (
        TsbenchForgeSource,
        _iso_duration_seconds,
        is_fresh,
        resolve_frequency,
    )

    forge_dir = Path(forge_dir)
    src = TsbenchForgeSource(forge_dir, max_stale_days=max_stale_days,
                             max_series_per_source=max_series_per_source)
    pd, pq, yaml = src._require_deps()
    cat_path = forge_dir / "sources.yaml"
    if not cat_path.is_file():
        raise SystemExit(f"{forge_dir}: no sources.yaml (not a forge mirror)")
    catalog = yaml.safe_load(cat_path.read_text(encoding="utf-8")) or []
    listing = mirror_listing(forge_dir)
    by_feed: dict[str, list[dict]] = {}
    for f in listing:
        by_feed.setdefault(f["feed"], []).append(f)
    dates = [f["date"] for f in listing if f["date"]]
    if not dates:
        raise SystemExit(f"{forge_dir}: no dated data/<feed>/<YYYY-MM-DD>.parquet partitions")
    as_of = max(dates)
    ctx = HarvestContext(as_of=as_of, horizon=horizon)
    feeds: dict[str, dict] = {}
    catalogued = set()
    for entry in catalog:
        fid = entry.get("id")
        if not fid:
            continue
        fid = str(fid)
        catalogued.add(fid)
        freq_s = str(entry.get("frequency", ""))
        raw_sec = _iso_duration_seconds(freq_s)
        mapped = resolve_frequency(freq_s)
        info = {"frequency": freq_s, "raw_seconds": raw_sec, "bucket": freq_bucket(raw_sec),
                "pandas_freq": mapped[0] if mapped else None, "series": [], "stats": {},
                "skipped": None}
        feeds[fid] = info
        files = [f for f in by_feed.get(fid, []) if f["date"] is None or f["date"] <= as_of]
        if entry.get("disabled"):
            info["skipped"] = "disabled"
        elif mapped is None:
            info["skipped"] = ("slower than daily (the reader skips it)" if raw_sec
                               else "unparseable frequency")
        elif not files:
            info["skipped"] = "no partitions"
        elif not is_fresh(as_of=as_of, cadence_sec=raw_sec, max_stale_days=max_stale_days,
                          newest_file=files[-1]["date"], newest_obs=None):
            info["skipped"] = f"stale (newest partition > {max_stale_days} d before {as_of})"
        if info["skipped"] and info["skipped"] != "slower than daily (the reader skips it)":
            info["stats"] = {"partitions": len(files)}
            continue
        mvc = entry.get("mv_channels") or None
        series, stats = _feed_series(src, pd, pq, files, ctx, horizon=horizon,
                                     columns=[str(c) for c in mvc] if mvc else None,
                                     clock=clock, max_series=max_series_per_source)
        info["series"], info["stats"] = series, stats
    uncatalogued = sorted(set(by_feed) - catalogued)
    ends = [i["stats"]["mirror_end"] for i in feeds.values() if "mirror_end" in i["stats"]]
    return {"forge_dir": str(forge_dir), "as_of": as_of.isoformat(),
            "catalog_sha256": file_sha256(cat_path), "catalog_entries": len(catalog),
            "partitions": len(listing), "listing_sha256": listing_sha256(listing),
            "newest_partition": as_of.isoformat(), "uncatalogued_feed_dirs": uncatalogued,
            "mirror_end_ts": max(ends) if ends else None, "feeds": feeds}


def _windows(s: dict, *, b: float, e: float, clock_ts: float, horizon: int,
             min_context: int) -> tuple[int, int]:
    """(fresh, embargo) DISJOINT target windows of one series for commit
    boundary ``b`` (= T + forward margin) seen at ``clock_ts``. A target point is
    eligible iff ``data_ts > b − e``, ``first_seen + e > b`` (proxy lower
    bound) and it was available by the clock (proxy upper bound ≤ clock), and
    it has ≥ ``min_context`` points of history before it. Each maximal run of
    eligible points is tiled from its newest end: fresh windows (every point
    ``data_ts > b`` and first-seen lower bound ``> b``: passes at e = 0) first,
    then embargo windows (fully embargoed or mixed) from the rest."""
    ts, lo, hi = s["data_ts"], s["fs_lo"], s["fs_hi"]
    elig = (ts > b - e) & (lo + e > b) & (hi <= clock_ts)
    elig[:max(0, min_context - int(s.get("offset", 0)))] = False
    if not elig.any():
        return 0, 0
    fresh = elig & (ts > b) & (lo > b)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], elig.astype(np.int8), [0]))))
    fw = ew = 0
    for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
        run = fresh[start:stop][::-1]
        tail = int(np.argmin(run)) if not run.all() else int(run.size)
        f = tail // horizon
        fw += f
        ew += (stop - start - f * horizon) // horizon
    return int(fw), int(ew)


def _set_rule(F: int, E: int, src_fresh: int, src_any: int, *, min_share: float,
              max_windows: int, min_windows: int, min_clusters: int) -> dict:
    """The rev6.12 (b) set rule on the counts: fresh first, embargo windows
    trimmed to ``F × (1 − s) / s`` (s = min_fresh_share), then the
    ``max_windows`` cap (fresh first), then the floors on the kept set."""
    if min_share >= 1.0:
        e_cap = 0
    elif min_share <= 0.0:
        e_cap = E
    else:
        e_cap = min(E, int(np.floor(F * (1.0 - min_share) / min_share + 1e-9)))
    kf = min(F, max_windows)
    ke = min(e_cap, max_windows - kf)
    n = kf + ke
    sources = src_any if ke else src_fresh
    share = (kf / n) if n else None
    return {"fresh_kept": kf, "embargo_kept": ke, "windows": n, "sources": sources,
            "fresh_share": share,
            "floors_met": bool(n >= min_windows and sources >= min_clusters),
            "fresh_share_met": bool(n and share >= min_share),
            "met": bool(n >= min_windows and sources >= min_clusters and n
                        and share >= min_share)}


def forge_sweep_times(mirror_end: float, *, sweep_days: float, step_h: float) -> list[float]:
    """Commit times T on the ``step_h`` grid, the newest with T + 24 h ≤ the
    mirror's end (so every clock is observable), back ``sweep_days``."""
    step = step_h * 3600.0
    if step <= 0:
        raise ValueError("--t-step-hours must be positive")
    last = np.floor((mirror_end - max(CLOCK_HOURS) * 3600.0) / step) * step
    n = int(np.floor(sweep_days * 86_400.0 / step + 1e-9))
    return [float(last - i * step) for i in range(n, -1, -1)]


def forge_measure(mirror: dict, *, commit_times: list[float] | None, sweep_days: float,
                  step_h: float, forward_margin_h: float, eval_delay_h: float,
                  embargo_h: float, horizon: int,
                  min_context: int, min_windows: int, min_clusters: int,
                  min_fresh_share: float, max_windows: int, min_long_series: int,
                  min_sources: int) -> dict:
    """The ``forge_mirror`` block of w1.json (label: measured from forge mirror).

    Plan rev6.14: the boundary is ``b = T + forward_margin`` (default 0, so a
    fresh target is entirely after T, data time and first-seen), and judging
    starts at ``T + eval_delay`` (default 6 h): a clock before that is not
    counted."""
    feeds = mirror["feeds"]
    end = mirror["mirror_end_ts"]
    # ── per-freq inventory + latency ─────────────────────────────────────────
    per_freq: dict[str, dict] = {}
    sources_out: dict[str, dict] = {}
    for fid in sorted(feeds):
        f = feeds[fid]
        lat = _latency(f["series"]) if f["series"] else None
        sources_out[fid] = {"frequency": f["frequency"], "bucket": f["bucket"],
                            "pandas_freq": f["pandas_freq"], "series": len(f["series"]),
                            "skipped": f["skipped"],
                            **{k: v for k, v in f["stats"].items() if k != "mirror_end"},
                            "latency": lat}
        if f["skipped"] and not f["series"]:
            continue
        c = per_freq.setdefault(f["bucket"], {"sources": 0, "series": 0, "points": 0,
                                              "pandas_freqs": set(), "lat_up": [],
                                              "lat_dn": [], "reader_emits": True})
        c["sources"] += 1 if f["series"] else 0
        c["series"] += len(f["series"])
        c["points"] += sum(s["length"] for s in f["series"])
        if f["pandas_freq"]:
            c["pandas_freqs"].add(f["pandas_freq"])
        if f["skipped"]:
            c["reader_emits"] = False
        if lat and lat["median_upper_hours"] is not None:
            c["lat_up"].append(lat["median_upper_hours"])
            c["lat_dn"].append(lat["median_lower_hours"])
    pf_out = {}
    for name, *_ in FREQ_BUCKETS:
        c = per_freq.get(name)
        if c is None:
            pf_out[name] = {"sources": 0, "series": 0, "label": FORGE_LABEL}
            continue
        pf_out[name] = {
            "sources": c["sources"], "series": c["series"], "points": c["points"],
            "pandas_freqs": sorted(c["pandas_freqs"]), "reader_emits": c["reader_emits"],
            "first_seen_latency_hours": {
                "median_over_sources_upper": _median(c["lat_up"]),
                "min_over_sources_upper": min(c["lat_up"]) if c["lat_up"] else None,
                "max_over_sources_upper": max(c["lat_up"]) if c["lat_up"] else None,
                "median_over_sources_lower": _median(c["lat_dn"]),
                "basis": "first-seen proxy − data time, non-baseline points"},
            "label": FORGE_LABEL}
    # ── fresh / embargo windows over the T sweep ─────────────────────────────
    scored = [(fid, f["bucket"], s) for fid in sorted(feeds) for s in feeds[fid]["series"]
              if not feeds[fid]["skipped"]]
    sweep: list[dict] | str
    if end is None:
        sweep = "not run: no partition with readable timestamps"
        times = []
    else:
        times = sorted(commit_times) if commit_times else forge_sweep_times(
            end, sweep_days=sweep_days, step_h=step_h)
        sweep = []
    margin_s, e_s = forward_margin_h * 3600.0, embargo_h * 3600.0
    rule = dict(min_share=min_fresh_share, max_windows=max_windows,
                min_windows=min_windows, min_clusters=min_clusters)
    if times:
        # Only points that can be targets for the earliest T are kept (a
        # target point needs data_ts > b − e); ``offset`` keeps min_context
        # counted against the full series.
        floor_ts = min(times) + margin_s - e_s
        trimmed = []
        for fid, bucket, s in scored:
            cut = int(np.searchsorted(s["data_ts"], floor_ts, side="right"))
            trimmed.append((fid, bucket, {k: s[k][cut:] for k in ("data_ts", "fs_lo", "fs_hi")}
                            | {"offset": cut}))
        scored_w = trimmed
    else:
        scored_w = []
    for T in times:
        b = T + margin_s
        clocks = {}
        for ch in CLOCK_HOURS:
            clock_ts = T + ch * 3600.0
            key = f"T+{ch}h"
            if ch < eval_delay_h:
                clocks[key] = (f"not run: clock T+{ch}h is before judging starts "
                               f"(T + eval_delay {eval_delay_h:g} h)")
                continue
            if clock_ts > end:
                clocks[key] = f"not run: clock {iso(clock_ts)} is after the mirror's end {iso(end)}"
                continue
            cells: dict[str, dict] = {}
            for fid, bucket, s in scored_w:
                fw, ew = _windows(s, b=b, e=e_s, clock_ts=clock_ts, horizon=horizon,
                                  min_context=min_context)
                if not (fw or ew):
                    continue
                for name in (bucket, "__total"):
                    c = cells.setdefault(name, {"fresh": 0, "embargo": 0, "src_fresh": set(),
                                                "src_any": set()})
                    c["fresh"] += fw
                    c["embargo"] += ew
                    c["src_any"].add(fid)
                    if fw:
                        c["src_fresh"].add(fid)

            def fin(c):
                return {"fresh": c["fresh"], "embargo": c["embargo"],
                        "sources": len(c["src_any"]), "sources_fresh": len(c["src_fresh"])}

            tot = cells.get("__total", {"fresh": 0, "embargo": 0, "src_fresh": set(),
                                        "src_any": set()})
            clocks[key] = {
                "clock": iso(clock_ts),
                "per_freq": {n: fin(cells[n]) for n, *_ in FREQ_BUCKETS if n in cells},
                "total": fin(tot),
                "set": _set_rule(tot["fresh"], tot["embargo"], len(tot["src_fresh"]),
                                 len(tot["src_any"]), **rule)}
        sweep.append({"commit_time": iso(T), "boundary_b": iso(b),
                      "embargo_from": iso(b - e_s),
                      "judging_from": iso(T + eval_delay_h * 3600.0), "clocks": clocks})
    summary = {}
    if isinstance(sweep, list):
        for ch in CLOCK_HOURS:
            key = f"T+{ch}h"
            rows = [r["clocks"][key] for r in sweep if isinstance(r["clocks"][key], dict)]
            if not rows:
                summary[key] = "not run: no commit time with this clock inside the mirror"
                continue

            def stat(vals):
                return {"median": _median(vals), "min": min(vals)} if vals else None

            pf = {}
            for n, *_ in FREQ_BUCKETS:
                fr = [r["per_freq"].get(n, {}).get("fresh", 0) for r in rows]
                em = [r["per_freq"].get(n, {}).get("embargo", 0) for r in rows]
                sc = [r["per_freq"].get(n, {}).get("sources", 0) for r in rows]
                if any(fr) or any(em):
                    pf[n] = {"fresh": stat(fr), "embargo": stat(em), "sources": stat(sc)}
            sets = [r["set"] for r in rows]
            shares = [s["fresh_share"] for s in sets if s["fresh_share"] is not None]
            summary[key] = {
                "commit_times": len(rows),
                "fresh": stat([r["total"]["fresh"] for r in rows]),
                "embargo": stat([r["total"]["embargo"] for r in rows]),
                "sources": stat([r["total"]["sources"] for r in rows]),
                "sources_fresh": stat([r["total"]["sources_fresh"] for r in rows]),
                "set_windows": stat([s["windows"] for s in sets]),
                "set_sources": stat([s["sources"] for s in sets]),
                "set_fresh_share": stat(shares),
                "met_share_of_T": sum(s["met"] for s in sets) / len(sets),
                "met_all_T": all(s["met"] for s in sets),
                "per_freq": pf, "label": FORGE_LABEL}
    # ── long-history inventory ───────────────────────────────────────────────
    rungs = []
    for L in HISTORY_LEVELS:
        need = L + horizon
        hits = [(fid, s) for fid, _b, s in scored if s["length"] >= need]
        n_src = len({fid for fid, _ in hits})
        rungs.append({"context": L, "need_points": need, "series": len(hits),
                      "sources": n_src,
                      "gate_met": bool(len(hits) >= min_long_series and n_src >= min_sources)})
    return {
        "label": FORGE_LABEL, "note": FORGE_NOTE,
        "mirror": {k: mirror[k] for k in ("forge_dir", "as_of", "catalog_sha256",
                                          "catalog_entries", "partitions", "listing_sha256",
                                          "newest_partition", "uncatalogued_feed_dirs")},
        "mirror_end": iso(end) if end is not None else None,
        "per_freq": pf_out, "sources": sources_out,
        "windows": {"label": FORGE_LABEL, "sweep": sweep, "summary": summary},
        "long_history": {
            "label": FORGE_LABEL,
            "basis": ("series = the reader's freshest contiguous segment per (feed, panel "
                      "row), per-feed cap as --max-series-per-source"),
            "gate": {"min_long_series": min_long_series, "min_sources": min_sources,
                     "source": "LaneDConfig.min_long_series default + "
                               "lane_d_scorer.GATE_MIN_SOURCES (readiness gate (1) only; "
                               "T7 is gate (2))"},
            "rungs": rungs},
    }


def print_summary(doc: dict, out=sys.stdout) -> None:
    fm = doc.get("forge_mirror")
    if fm:
        print(f"W1 — {fm['label']} (mirror end {fm['mirror_end']})", file=out)
        for name, c in fm["per_freq"].items():
            if c["series"]:
                lat = (c.get("first_seen_latency_hours") or {}).get("median_over_sources_upper")
                print(f"  {name}: {c['sources']} sources / {c['series']} series, "
                      f"median first-seen latency <= {lat} h", file=out)
        for key, s in fm["windows"]["summary"].items():
            if isinstance(s, str):
                print(f"  {key}: {s}", file=out)
                continue
            print(f"  {key}: fresh med {s['fresh']['median']} / embargo med "
                  f"{s['embargo']['median']}, set med {s['set_windows']['median']} from "
                  f"{s['set_sources']['median']} sources; met in "
                  f"{s['met_share_of_T']:.0%} of {s['commit_times']} T", file=out)
        for r in fm["long_history"]["rungs"]:
            print(f"  >={r['context']}+{r['need_points'] - r['context']}: {r['series']} series"
                  f" / {r['sources']} sources (gate met: {r['gate_met']})", file=out)
    if not isinstance(doc.get("pairs"), list):
        return
    print(f"W1 — {doc['label']}", file=out)
    for p in doc["pairs"]:
        tr = p["transition"]
        print(f"  {p['from']} -> {p['to']}: common {tr.get('common', 0)}, appended "
              f"{tr.get('appended_series', 0)} series / {tr.get('appended_points', 0)} pts, "
              f"revised {tr.get('revised_series', 0)} / {tr.get('revised_points', 0)}, "
              f"appeared {tr.get('appeared', 0)}, disappeared {tr.get('disappeared', 0)}, "
              f"unaligned {tr.get('unaligned', 0)}", file=out)
        if isinstance(p["eligibility"], str):
            print(f"    {p['eligibility']}", file=out)
            continue
        for e in p["eligibility"]:
            t = e["total"]
            print(f"    T={e['commit_time']}: <= {t['windows']} windows / {t['sources']} sources"
                  f" (floors met: {t['floors']['met']})", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshots-dir", default=None,
                    help="directory of >= 2 unpacked snapshots, one sub-directory each")
    ap.add_argument("--forge-dir", default=None,
                    help="synced forge mirror (sources.yaml + data/): the measured block")
    ap.add_argument("--embargo-hours", type=float, default=DEFAULT_EMBARGO_HOURS)
    ap.add_argument("--forward-margin-hours", type=float, default=DEFAULT_FORWARD_MARGIN_HOURS,
                    help="forge: b = T + this (plan rev6.14; --margin-hours stays the "
                         "snapshot-diff bound's)")
    ap.add_argument("--eval-delay-hours", type=float, default=DEFAULT_EVAL_DELAY_HOURS,
                    help="forge: judging starts at T + this; earlier clocks are not counted")
    ap.add_argument("--min-fresh-share", type=float, default=DEFAULT_MIN_FRESH_SHARE)
    ap.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    ap.add_argument("--min-long-series", type=int, default=None,
                    help="default: LaneDConfig().min_long_series")
    ap.add_argument("--sweep-days", type=float, default=DEFAULT_SWEEP_DAYS,
                    help="forge: commit-time sweep length back from the mirror's end")
    ap.add_argument("--first-seen-clock", choices=FIRST_SEEN_CLOCKS, default="min",
                    help="forge: first-seen proxy upper bound (see module docs)")
    ap.add_argument("--max-series-per-source", type=int, default=200,
                    help="forge: per-feed series cap (the reader's default; 0 = none)")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--build-time", action="append", default=None, help="NAME=ISO8601")
    ap.add_argument("--index", default=None, help="pool/index.json (as_of dates, tar sha256)")
    ap.add_argument("--snapshot-sha256", action="append", default=None, help="NAME=HEX (tar)")
    ap.add_argument("--commit-time", action="append", default=None,
                    help="ISO8601 commit time T (repeatable); default: a sweep")
    ap.add_argument("--t-step-hours", type=float, default=6.0)
    ap.add_argument("--margin-hours", type=float, default=DEFAULT_MARGIN_HOURS)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--min-context", type=int, default=DEFAULT_MIN_CONTEXT)
    ap.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS)
    ap.add_argument("--min-clusters", type=int, default=DEFAULT_MIN_CLUSTERS)
    args = ap.parse_args(argv)
    if not args.snapshots_dir and not args.forge_dir:
        ap.error("give --snapshots-dir, --forge-dir or both")
    if not 0.0 <= args.min_fresh_share <= 1.0:
        ap.error("--min-fresh-share must be in [0, 1]")
    if args.max_windows < args.min_windows:
        ap.error("--max-windows must be >= --min-windows")
    commit_times = [parse_time(t) for t in args.commit_time or ()] or None
    if args.snapshots_dir:
        root = Path(args.snapshots_dir)
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and (d / "metadata.json").is_file())
        if len(dirs) < 2:
            raise SystemExit(f"{root}: need >= 2 snapshot sub-directories with metadata.json")
        bt = _kv(args.build_time, "--build-time")
        shas = _kv(args.snapshot_sha256, "--snapshot-sha256")
        index = _index_entries(args.index)
        snaps = [snapshot_info(d, build_times=bt, shas=shas, index=index) for d in dirs]
        body = measure(snaps, commit_times=commit_times,
                       step_h=args.t_step_hours, margin_h=args.margin_hours, horizon=args.horizon,
                       min_context=args.min_context, min_windows=args.min_windows,
                       min_clusters=args.min_clusters)
    else:
        body = {"label": BOUND_LABEL, "snapshots": [],
                "pairs": "not run: no --snapshots-dir",
                "commit_times_dropped_before_first_snapshot": []}
    forge_params = None
    if args.forge_dir:
        gate = lane_d_defaults()
        if args.min_long_series is not None:
            gate["min_long_series"] = int(args.min_long_series)
        mirror = load_forge(args.forge_dir, horizon=args.horizon, clock=args.first_seen_clock,
                            max_series_per_source=args.max_series_per_source)
        body["forge_mirror"] = forge_measure(
            mirror, commit_times=commit_times, sweep_days=args.sweep_days,
            step_h=args.t_step_hours, forward_margin_h=args.forward_margin_hours,
            eval_delay_h=args.eval_delay_hours, embargo_h=args.embargo_hours, horizon=args.horizon, min_context=args.min_context,
            min_windows=args.min_windows, min_clusters=args.min_clusters,
            min_fresh_share=args.min_fresh_share, max_windows=args.max_windows,
            min_long_series=gate["min_long_series"], min_sources=gate["min_sources"])
        forge_params = {"embargo_hours": args.embargo_hours,
                        "forward_margin_hours": args.forward_margin_hours,
                        "eval_delay_hours": args.eval_delay_hours,
                        "min_fresh_share": args.min_fresh_share,
                        "max_windows": args.max_windows, "sweep_days": args.sweep_days,
                        "clocks_hours": list(CLOCK_HOURS),
                        "first_seen_clock": args.first_seen_clock,
                        "max_series_per_source": args.max_series_per_source,
                        "freq_buckets_seconds": {n: [lo, None if hi == float("inf") else hi]
                                                 for n, lo, hi in FREQ_BUCKETS},
                        **gate}
    doc = dict(provenance(argv), schema=SCHEMA,
               labels={"pairs": BOUND_LABEL,
                       "forge_mirror": FORGE_LABEL if args.forge_dir
                       else "not run: no --forge-dir"},
               forge_params=forge_params or "not run: no --forge-dir",
               index_sha256=(file_sha256(args.index) if args.index
                             else "not run: no --index given"),
               snapshot_sha256={s["name"]: {"tree": s["tree_sha256"], "tar": s["tar_sha256"]}
                                for s in body["snapshots"]},
               seeds="none: the measurement is deterministic (no sampling)",
               params={"horizon": args.horizon, "margin_hours": args.margin_hours,
                       "min_context": args.min_context, "min_windows": args.min_windows,
                       "min_clusters": args.min_clusters, "t_step_hours": args.t_step_hours,
                       "align": {"anchors": ANCHORS, "min_match_frac": MIN_MATCH_FRAC,
                                 "min_overlap": MIN_OVERLAP},
                       "first_seen": "build time of the first snapshot holding the point "
                                     "(revised / backfilled / new-series / unaligned points "
                                     "take the newer snapshot's time)"},
               **body)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(doc)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
