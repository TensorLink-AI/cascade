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
      [--commit-time 2026-11-19T00:00:00Z ...] [--t-step-hours 6]
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


def print_summary(doc: dict, out=sys.stdout) -> None:
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
    ap.add_argument("--snapshots-dir", required=True,
                    help="directory of >= 2 unpacked snapshots, one sub-directory each")
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
    root = Path(args.snapshots_dir)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "metadata.json").is_file())
    if len(dirs) < 2:
        raise SystemExit(f"{root}: need >= 2 snapshot sub-directories with metadata.json")
    bt = _kv(args.build_time, "--build-time")
    shas = _kv(args.snapshot_sha256, "--snapshot-sha256")
    index = _index_entries(args.index)
    snaps = [snapshot_info(d, build_times=bt, shas=shas, index=index) for d in dirs]
    body = measure(snaps, commit_times=[parse_time(t) for t in args.commit_time or ()] or None,
                   step_h=args.t_step_hours, margin_h=args.margin_hours, horizon=args.horizon,
                   min_context=args.min_context, min_windows=args.min_windows,
                   min_clusters=args.min_clusters)
    doc = dict(provenance(argv), schema=SCHEMA,
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
