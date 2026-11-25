"""First-seen recording for the lane D ``pool_d`` builder (plan W6fs).

An append-only ingest log, kept in the forge mirror path (by default
``<forge mirror>/first_seen/``, which :class:`TsbenchForgeSource` never
reads), records for every point of every ``data/<feed>/<day>.parquet``
partition the first time **our mirror** saw it. Lane D's forward-only rule
reads it on both clocks (``first_seen(p) > b`` and the rev6.10 release time
``first_seen(p) + e``), so it is built to be re-derivable and attested:

* ``records.jsonl``: one canonical-JSON record per (re)read partition, with
  only the point keys ``(panel_json, data_ts)`` never seen before for that
  feed. Rewrites and backfills therefore keep each point's FIRST time.
* ``roots.jsonl``: per 3-hourly period (rev6.11 (d)) the Merkle root of that
  period's records, committed once the period has ended. A period with a
  root never takes another record (``period_committed``), a rewritten log no
  longer matches its roots (``root_mismatch``), and a snapshot built at
  ``as_of`` may use only records whose root was committed by ``as_of``.
* per feed, the measured mirror lag (median ``first_seen − data_ts`` over
  points recorded after the feed's first, baseline, read) and the maximum
  lead ``data_ts − first_seen`` (> 0 = publish-ahead; W6b then pins e = 0).

The release time is never stored (``release_ts`` derives it from the signed
record and the snapshot's pinned ``e``). Producer side only: validators read
the pinned ``pool_d`` snapshot, never this log. Needs the ``pool-forge`` extra.

Unreadable partitions (2026-11-25: two forge objects with no parquet footer
aborted every release) are skipped, logged loudly to stderr and appended to
``quarantine.jsonl`` (append-only, keyed by feed, partition name, size and
sha256). A quarantined partition NAME stays quarantined forever: when a later
version of it becomes readable, its points not already known are written to
the quarantine as ``blocked`` (never to ``records.jsonl``) and are never
recorded as first seen afterwards, from any partition; the index also drops
any record of a quarantined name seen after its quarantine (an older release
that did not know the file). Old (e.g. July) points of a repaired object can
therefore never count as fresh in lane D. Points already recorded from other
partitions before the quarantine keep their genuine first-seen time. More
than :func:`unreadable_limit` unreadable partitions in one run (a systemic
read failure) aborts as before (``too_many_unreadable``), writing nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

PERIOD_SECONDS = 10_800
LOG_SCHEMA = "pool-d-first-seen/1"
EVIDENCE_SCHEMA = "laned-evidence/w6fs/1"
RECORDS_FILE = "records.jsonl"
ROOTS_FILE = "roots.jsonl"
QUARANTINE_FILE = "quarantine.jsonl"
QUARANTINE_SCHEMA = "pool-d-first-seen-quarantine/1"
DEFAULT_LOG_DIRNAME = "first_seen"
# The systemic-read-failure guard: more unreadable partitions than
# unreadable_limit(...) in one run aborts (nothing written).
DEFAULT_MAX_UNREADABLE_FILES = 50
DEFAULT_MAX_UNREADABLE_FRAC = 0.001
_REASON_MAX = 300


class FirstSeenError(ValueError):
    """A first-seen log defect; ``code`` is a stable tag."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def period_start(ts: int) -> int:
    ts = int(ts)
    return ts - ts % PERIOD_SECONDS


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def panel_json(panel: Mapping | str) -> str:
    """The canonical panel key of a point (``"{}"`` when unpaneled)."""
    if isinstance(panel, str):
        return panel
    return canonical_json({str(k): str(v) for k, v in dict(panel).items()}).decode()


def merkle_root(leaves: Sequence[bytes]) -> str:
    """Hex Merkle root: leaf ``sha256(0x00 || leaf)``, node
    ``sha256(0x01 || left || right)``, an odd last node promoted unchanged."""
    if not leaves:
        raise ValueError("merkle_root of no leaves")
    level = [hashlib.sha256(b"\x00" + bytes(leaf)).digest() for leaf in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest())
            else:
                nxt.append(level[i])
        level = nxt
    return level[0].hex()


def release_ts(first_seen_ts: int, embargo_hours: float) -> int:
    """rev6.10 ``release(p) = first_seen(p) + e`` (derived, never stored)."""
    return int(first_seen_ts) + round(float(embargo_hours) * 3600)


def unreadable_limit(n_partitions: int, *, max_files: int = DEFAULT_MAX_UNREADABLE_FILES,
                     max_frac: float = DEFAULT_MAX_UNREADABLE_FRAC) -> int:
    """How many unreadable partitions one run tolerates: ``max_files``, or
    ``max_frac`` of the partitions when that is smaller (never below 1)."""
    by_frac = max(1, math.floor(float(max_frac) * max(0, int(n_partitions))))
    return max(0, min(int(max_files), by_frac))


def unreadable_reason(exc: BaseException) -> str:
    """A short, stable reason string for a partition that failed to read."""
    msg = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {msg}"[:_REASON_MAX]


def _warn(msg: str) -> None:
    print(f"first_seen: WARNING {msg}", file=sys.stderr, flush=True)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _epoch_seconds(pd, raw):
    """``(ok mask, int64 epoch seconds)`` of a timestamp column, parsed the
    way the forge reader parses it (mixed ISO, then its fallback formats)."""
    from .sources.tsbench_forge import _TS_FALLBACK_FORMATS

    ts = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
    if len(ts) and ts.notna().mean() <= 0.9:
        for fmt in _TS_FALLBACK_FORMATS:
            alt = pd.to_datetime(raw, errors="coerce", utc=True, format=fmt)
            if alt.notna().mean() > ts.notna().mean():
                ts = alt
    ok = ts.notna().to_numpy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    secs = np.zeros(len(ts), dtype=np.int64)
    if ok.any():
        secs[ok] = ((ts[ok] - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    return ok, secs


def partition_points(path: Path) -> list[tuple[str, int]]:
    """The ``(panel_json, data_ts)`` keys of one forge parquet partition."""
    from .sources.tsbench_forge import TsbenchForgeSource

    pd, pq, _yaml = TsbenchForgeSource._require_deps()
    df = pq.read_table(path).to_pandas()
    if "timestamp" not in df.columns or df.empty:
        return []
    ok, secs = _epoch_seconds(pd, df["timestamp"])
    panel_cols = [c for c in df.columns if c.startswith("_panel_")]
    cols = [[str(v) for v in df[c].tolist()] for c in panel_cols]
    names = [c.removeprefix("_panel_") for c in panel_cols]
    out = set()
    for i in np.flatnonzero(ok):
        panel = {n: col[i] for n, col in zip(names, cols, strict=True)}
        out.add((panel_json(panel), int(secs[i])))
    return sorted(out)


class FirstSeenIndex:
    """First-seen lookups over the committed records up to ``until_ts``."""

    def __init__(self, entries: dict, until_ts: int):
        self._entries = entries            # (feed, panel_json, ts) -> (seen, root, baseline)
        self.until_ts = int(until_ts)

    def __len__(self) -> int:
        return len(self._entries)

    def _get(self, feed, panel, data_ts):
        return self._entries.get((str(feed), panel_json(panel), int(data_ts)))

    def first_seen(self, feed: str, panel: Mapping | str, data_ts: int) -> int | None:
        hit = self._get(feed, panel, data_ts)
        return None if hit is None else hit[0]

    def root_for(self, feed: str, panel: Mapping | str, data_ts: int) -> str | None:
        hit = self._get(feed, panel, data_ts)
        return None if hit is None else hit[1]

    def feed_points(self, feed: str) -> dict[tuple[str, int], int]:
        """``{(panel_json, data_ts): first_seen}`` for one feed."""
        return {(p, t): v[0] for (f, p, t), v in self._entries.items() if f == feed}

    def feed_lag(self) -> dict[str, dict]:
        lags: dict[str, list[int]] = {}
        leads: dict[str, int] = {}
        for (feed, _panel, ts), (seen, _root, baseline) in self._entries.items():
            lead = ts - seen
            leads[feed] = lead if feed not in leads else max(leads[feed], lead)
            lags.setdefault(feed, [])
            if not baseline and ts <= seen:
                lags[feed].append(seen - ts)
        out = {}
        for feed in sorted(leads):
            lag = lags.get(feed) or []
            lead_h = leads[feed] / 3600.0
            out[feed] = {
                "mirror_lag_hours": (statistics.median(lag) / 3600.0) if lag else None,
                "max_lead_hours": lead_h,
                "publish_ahead": bool(lead_h > 0),
            }
        return out


class FirstSeenLog:
    """The append-only ingest log in ``log_dir``."""

    def __init__(self, log_dir: Path | str):
        self.log_dir = Path(log_dir)

    @property
    def records_path(self) -> Path:
        return self.log_dir / RECORDS_FILE

    @property
    def roots_path(self) -> Path:
        return self.log_dir / ROOTS_FILE

    @property
    def quarantine_path(self) -> Path:
        return self.log_dir / QUARANTINE_FILE

    # ── reading ─────────────────────────────────────────────────────────────

    @staticmethod
    def _lines(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        out = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise FirstSeenError("log_corrupt", f"{path}:{n}: {e}") from e
        return out

    def records(self) -> list[dict]:
        return self._lines(self.records_path)

    def quarantine(self) -> list[dict]:
        """Every quarantine entry (``kind`` ``unreadable`` or ``blocked``)."""
        return self._lines(self.quarantine_path)

    def quarantined(self, *, until_ts: int | None = None) -> dict[tuple[str, str], int]:
        """``{(feed, partition name): first quarantined_ts}`` of the names
        quarantined (unreadable) at or before ``until_ts`` (all when None)."""
        out: dict[tuple[str, str], int] = {}
        for e in self.quarantine():
            if e.get("kind") != "unreadable":
                continue
            qts = int(e["quarantined_ts"])
            if until_ts is not None and qts > int(until_ts):
                continue
            key = (str(e["feed"]), str(e["partition"]))
            out[key] = min(qts, out.get(key, qts))
        return out

    def quarantine_report(self, *, until_ts: int) -> list[dict]:
        """The quarantine entries in effect at ``until_ts``, for the signed
        ``pool_d`` metadata (``blocked`` entries carry a point count only)."""
        out = []
        for e in self.quarantine():
            if int(e["quarantined_ts"]) > int(until_ts):
                continue
            row = {k: e[k] for k in ("kind", "feed", "partition", "partition_sha256", "size",
                                     "quarantined_ts")}
            if e.get("kind") == "unreadable":
                row["reason"] = str(e.get("reason", ""))
            else:
                row["n_blocked_points"] = len(e.get("points") or [])
            out.append(row)
        return sorted(out, key=lambda r: (r["feed"], r["partition"], r["quarantined_ts"],
                                          r["kind"], r["partition_sha256"]))

    def roots(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for entry in self._lines(self.roots_path):
            p = int(entry["period_start"])
            if p in out and out[p]["root"] != entry["root"]:
                raise FirstSeenError("root_conflict", f"period {p} has two different roots")
            out.setdefault(p, entry)
        return out

    # ── writing ─────────────────────────────────────────────────────────────

    def _append(self, path: Path, docs: list[dict]) -> None:
        if not docs:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for doc in docs:
                f.write(canonical_json(doc).decode() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def record(self, forge_dir: Path | str, *, now_ts: int,
               max_unreadable_files: int = DEFAULT_MAX_UNREADABLE_FILES,
               max_unreadable_frac: float = DEFAULT_MAX_UNREADABLE_FRAC) -> dict:
        """Log every new point of the forge mirror's partitions at ``now_ts``.

        An unreadable partition is skipped and quarantined (module docstring);
        more than :func:`unreadable_limit` unreadable partitions (new ones plus
        quarantined versions still present) raises ``too_many_unreadable``
        before anything is written. The result gains ``quarantined`` /
        ``blocked`` / ``unreadable`` only when non-zero."""
        now_ts = int(now_ts)
        p = period_start(now_ts)
        if p in self.roots():
            raise FirstSeenError("period_committed",
                                 f"period {p} already has a committed root; now_ts={now_ts}")
        existing = self.records()
        quarantine = self.quarantine()
        seen_parts = {(r["feed"], r["partition"], r["partition_sha256"]) for r in existing}
        known: dict[str, set] = {}
        for r in existing:
            known.setdefault(r["feed"], set()).update((pj, int(t)) for pj, t in r["points"])
        q_names = {(str(e["feed"]), str(e["partition"])) for e in quarantine
                   if e.get("kind") == "unreadable"}
        q_versions = {(str(e["feed"]), str(e["partition"]), str(e["partition_sha256"])):
                      e.get("kind") for e in quarantine}
        for e in quarantine:
            if e.get("kind") == "blocked":
                known.setdefault(str(e["feed"]), set()).update(
                    (pj, int(t)) for pj, t in e.get("points") or [])
        logged_feeds = {r["feed"] for r in existing}
        data = Path(forge_dir) / "data"
        new: list[dict] = []
        new_q: list[dict] = []
        unreadable: list[str] = []
        n_points = n_blocked = n_parts = 0
        feed_dirs = sorted(d for d in data.iterdir() if d.is_dir()) if data.is_dir() else []
        for feed_dir in feed_dirs:
            feed = feed_dir.name
            for part in sorted(feed_dir.glob("*.parquet")):
                n_parts += 1
                sha = _file_sha(part)
                if (feed, part.name, sha) in seen_parts:
                    continue
                kind = q_versions.get((feed, part.name, sha))
                if kind == "unreadable":                 # the same corrupt bytes
                    unreadable.append(f"{feed}/{part.name}")
                    continue
                if kind == "blocked":                    # already blocked
                    continue
                size = part.stat().st_size
                try:
                    points = partition_points(part)
                except Exception as e:  # noqa: BLE001 — any read failure quarantines it
                    reason = unreadable_reason(e)
                    _warn(f"QUARANTINE unreadable partition data/{feed}/{part.name} "
                          f"(size={size} sha256={sha}): {reason}")
                    unreadable.append(f"{feed}/{part.name}")
                    new_q.append({"schema": QUARANTINE_SCHEMA, "kind": "unreadable",
                                  "feed": feed, "partition": part.name,
                                  "partition_sha256": sha, "size": int(size),
                                  "reason": reason, "quarantined_ts": now_ts})
                    q_versions[(feed, part.name, sha)] = "unreadable"
                    q_names.add((feed, part.name))
                    continue
                have = known.setdefault(feed, set())
                fresh = sorted(set(points) - have)
                have.update(fresh)
                if (feed, part.name) in q_names:
                    # A quarantined partition readable again: its unknown
                    # points are blocked forever, never first-seen now.
                    _warn(f"quarantined partition data/{feed}/{part.name} is readable again "
                          f"(size={size} sha256={sha}); {len(fresh)} unseen point(s) blocked "
                          f"from lane D, not recorded as first seen")
                    n_blocked += len(fresh)
                    new_q.append({"schema": QUARANTINE_SCHEMA, "kind": "blocked",
                                  "feed": feed, "partition": part.name,
                                  "partition_sha256": sha, "size": int(size),
                                  "quarantined_ts": now_ts,
                                  "points": [[pj, int(t)] for pj, t in fresh]})
                    q_versions[(feed, part.name, sha)] = "blocked"
                    continue
                n_points += len(fresh)
                new.append({
                    "schema": LOG_SCHEMA, "feed": feed, "partition": part.name,
                    "partition_sha256": sha, "seen_ts": now_ts, "period_start": p,
                    "baseline": feed not in logged_feeds,
                    "points": [[pj, int(t)] for pj, t in fresh],
                })
        limit = unreadable_limit(n_parts, max_files=max_unreadable_files,
                                 max_frac=max_unreadable_frac)
        if len(unreadable) > limit:
            raise FirstSeenError(
                "too_many_unreadable",
                f"{len(unreadable)} of {n_parts} partitions are unreadable (limit {limit}); "
                f"a systemic read failure? nothing recorded. First: "
                f"{', '.join(unreadable[:10])}")
        # Quarantine first: a crash between the two appends errs on the safe side.
        self._append(self.quarantine_path, new_q)
        self._append(self.records_path, new)
        out = {"records": len(new), "points": n_points}
        n_q = sum(1 for e in new_q if e["kind"] == "unreadable")
        if n_q:
            out["quarantined"] = n_q
        if n_blocked or any(e["kind"] == "blocked" for e in new_q):
            out["blocked"] = n_blocked
        if unreadable:
            out["unreadable"] = len(unreadable)
            _warn(f"{len(unreadable)} unreadable partition(s) skipped (limit {limit}): "
                  f"{', '.join(unreadable[:20])}")
        return out

    def commit(self, *, now_ts: int) -> list[dict]:
        """Commit a root for every ended period that has records and no root."""
        now_ts = int(now_ts)
        roots = self.roots()
        by_period: dict[int, list[dict]] = {}
        for r in self.records():
            by_period.setdefault(int(r["period_start"]), []).append(r)
        added = []
        for p in sorted(by_period):
            if p in roots or p + PERIOD_SECONDS > now_ts:
                continue
            recs = by_period[p]
            added.append({"period_start": p,
                          "root": merkle_root([canonical_json(r) for r in recs]),
                          "n_records": len(recs), "committed_ts": now_ts})
        self._append(self.roots_path, added)
        return added

    # ── the index ───────────────────────────────────────────────────────────

    def index(self, *, until_ts: int) -> FirstSeenIndex:
        """Points of the periods whose root was committed by ``until_ts``,
        after checking every committed root against the log."""
        until_ts = int(until_ts)
        records = self.records()
        roots = self.roots()
        by_period: dict[int, list[dict]] = {}
        for r in records:
            if int(r["period_start"]) != period_start(int(r["seen_ts"])):
                raise FirstSeenError("root_mismatch",
                                     f"record of {r.get('feed')} has a wrong period_start")
            by_period.setdefault(int(r["period_start"]), []).append(r)
        for p, entry in roots.items():
            recs = by_period.get(p, [])
            got = merkle_root([canonical_json(r) for r in recs]) if recs else None
            if got != entry["root"] or len(recs) != int(entry["n_records"]):
                raise FirstSeenError("root_mismatch",
                                     f"period {p}: log no longer matches its committed root")
        usable = {p: e["root"] for p, e in roots.items() if int(e["committed_ts"]) <= until_ts}
        quarantined = self.quarantined(until_ts=until_ts)
        entries: dict = {}
        for r in records:
            p = int(r["period_start"])
            if p not in usable:
                continue
            seen = int(r["seen_ts"])
            q_ts = quarantined.get((str(r["feed"]), str(r["partition"])))
            if q_ts is not None and seen >= q_ts:
                # A quarantined name recorded at/after its quarantine (e.g. by
                # a release that did not know quarantine.jsonl): never fresh.
                continue
            for pj, t in r["points"]:
                key = (str(r["feed"]), str(pj), int(t))
                if key not in entries or seen < entries[key][0]:
                    entries[key] = (seen, usable[p], bool(r["baseline"]))
        return FirstSeenIndex(entries, until_ts)


def verify_snapshot(meta: Mapping, log_dir: Path | str) -> list[str]:
    """Problems of a ``pool_d/1`` metadata dict against the log (``[]`` = ok)."""
    problems: list[str] = []
    try:
        as_of = int(meta["as_of_ts"])
    except (KeyError, TypeError, ValueError):
        return ["metadata has no integer as_of_ts"]
    try:
        idx = FirstSeenLog(log_dir).index(until_ts=as_of)
    except FirstSeenError as e:
        return [f"first-seen log: {e}"]
    listed = set(meta.get("first_seen_roots") or [])
    for s in meta.get("series") or []:
        sid = s.get("series_id")
        ts, seen = s.get("target_ts") or [], s.get("first_seen_ts") or []
        if len(ts) != len(seen):
            problems.append(f"{sid}: target_ts and first_seen_ts differ in length")
            continue
        feed, panel = s.get("feed") or s.get("source"), s.get("panel") or {}
        for t, claimed in zip(ts, seen, strict=True):
            got = idx.first_seen(feed, panel, t)
            if got is None:
                problems.append(f"{sid}: point {t} has no first-seen record committed by "
                                f"as_of {as_of}")
                continue
            if int(claimed) != got:
                problems.append(f"{sid}: point {t} first_seen_ts {claimed} != log {got}")
            root = idx.root_for(feed, panel, t)
            if root not in listed:
                problems.append(f"{sid}: point {t} root {root} not in first_seen_roots")
    return problems


# ── CLI ─────────────────────────────────────────────────────────────────────


def _evidence(log: FirstSeenLog, now_ts: int) -> dict:
    records = log.records()
    roots = sorted(log.roots().values(), key=lambda e: int(e["period_start"]))
    doc = {
        "schema": EVIDENCE_SCHEMA, "wp": "W6fs", "source": "builder",
        "generated_by": "cascade.pool.first_seen evidence",
        "host": socket.gethostname(), "generated_ts": int(now_ts),
        "n_records": len(records), "n_points": sum(len(r["points"]) for r in records),
        "roots": [{"period_start": int(e["period_start"]), "root": e["root"],
                   "n_records": int(e["n_records"]), "committed_ts": int(e["committed_ts"])}
                  for e in roots],
        "feeds": log.index(until_ts=now_ts).feed_lag(),
    }
    doc["digest"] = hashlib.sha256(canonical_json(doc)).hexdigest()
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m cascade.pool.first_seen",
                                 description="pool_d first-seen ingest log (lane D W6fs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "commit", "evidence"):
        p = sub.add_parser(name)
        if name != "commit":
            p.add_argument("--forge-dir", type=Path, required=True)
        if name == "record":
            p.add_argument("--max-unreadable-files", type=int,
                           default=DEFAULT_MAX_UNREADABLE_FILES)
            p.add_argument("--max-unreadable-frac", type=float,
                           default=DEFAULT_MAX_UNREADABLE_FRAC)
        p.add_argument("--log-dir", type=Path, default=None,
                       help="default: <forge-dir>/first_seen")
        p.add_argument("--now-ts", type=int, default=None, help="default: the wall clock")
        if name == "evidence":
            p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    now = int(time.time()) if args.now_ts is None else int(args.now_ts)
    log_dir = args.log_dir
    if log_dir is None:
        if getattr(args, "forge_dir", None) is None:
            ap.error("--log-dir is required")
        log_dir = args.forge_dir / DEFAULT_LOG_DIRNAME
    log = FirstSeenLog(log_dir)
    try:
        if args.cmd == "record":
            print(json.dumps(log.record(args.forge_dir, now_ts=now,
                                        max_unreadable_files=args.max_unreadable_files,
                                        max_unreadable_frac=args.max_unreadable_frac),
                             sort_keys=True))
        elif args.cmd == "commit":
            print(json.dumps(log.commit(now_ts=now), sort_keys=True))
        else:
            doc = _evidence(log, now)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
            print(f"wrote {args.out} ({doc['n_records']} records, {len(doc['roots'])} roots)")
    except FirstSeenError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_MAX_UNREADABLE_FILES",
    "DEFAULT_MAX_UNREADABLE_FRAC",
    "EVIDENCE_SCHEMA",
    "LOG_SCHEMA",
    "PERIOD_SECONDS",
    "QUARANTINE_FILE",
    "QUARANTINE_SCHEMA",
    "FirstSeenError",
    "FirstSeenIndex",
    "FirstSeenLog",
    "canonical_json",
    "main",
    "merkle_root",
    "panel_json",
    "partition_points",
    "period_start",
    "release_ts",
    "unreadable_limit",
    "unreadable_reason",
    "verify_snapshot",
]
