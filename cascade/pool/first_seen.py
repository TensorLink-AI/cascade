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

Incremental scans (2026-11-28; every build re-hashed all ~630k forge
partitions and re-parsed the whole 20 GB ``records.jsonl`` three times):
with a ``cache_dir`` (the CLIs default to ``<log_dir>.cache``) the log keeps

* ``partitions.json``: per partition ``(size, mtime_ns, inode) -> sha256``,
  so only new or changed partitions are hashed;
* ``records.meta.json``: per ``records.jsonl`` line its byte span, feed,
  partition, sha256, ``seen_ts``, ``period_start``, point count, data-time
  range and Merkle leaf hash, plus a fingerprint of the bytes it covers; a
  run parses only the lines appended since (the log is append-only), and
  ``record`` loads a feed's known points only from the lines whose data-time
  range meets the new partition's.

Every answer equals the uncached one (same records, roots, index; tests on
fixtures). A cache that no longer fingerprints its log prefix is dropped (a
full scan); ``full=True`` (CLI ``--full``) ignores it; and a full scan runs
anyway once the cache's last full verification is ``full_verify_seconds``
(24 h) old, warning loudly when the cache had diverged. The cache lives
outside the three log files: ``state-push`` never uploads it.

Root rule ``roots="ended"`` (2026-11-28, the pool_d builder): a snapshot at
``as_of`` uses every committed period that ENDED by ``as_of`` (its records
are all first seen before ``as_of``), including the root this build's own
``commit`` wrote after ``as_of``; ``"committed"`` (the default) keeps the
rev6.11 (d) rule (root committed by ``as_of``), one build later. ``ended``
refuses (``uncommitted_period``) when an ended period has records but no
root, so the usable set is final when the snapshot is cut and anyone
re-derives it from the log; ``record`` refuses a ``now_ts`` whose period
ended before the newest commit (``period_frozen``), so no record can later
enter a period an earlier snapshot relied on being empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import sys
import tempfile
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
CACHE_SCHEMA = "pool-d-first-seen-cache/1"
CACHE_META_FILE = "records.meta.json"
CACHE_PARTS_FILE = "partitions.json"
FULL_VERIFY_SECONDS = 86_400
ROOT_RULES = ("committed", "ended")
_FP_BYTES = 1 << 16


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
    return _merkle_root_of_hashes([_leaf_hash(leaf) for leaf in leaves])


def _leaf_hash(leaf: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + bytes(leaf)).digest()


def _merkle_root_of_hashes(level: list[bytes]) -> str:
    """:func:`merkle_root` from the leaf hashes (``sha256(0x00 || leaf)``)."""
    if not level:
        raise ValueError("merkle_root of no leaves")
    level = list(level)
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


def default_cache_dir(log_dir: Path | str) -> Path:
    """``<log_dir>.cache``: beside the log, never one of its pushed files."""
    d = Path(log_dir)
    return d.with_name(d.name + ".cache")


def _fingerprint(path: Path, size: int) -> str:
    """sha256 over the first and the last ``_FP_BYTES`` of ``path[:size]``
    (an append-only log keeps both; a rewrite or a truncation changes them)."""
    h = hashlib.sha256(str(int(size)).encode())
    if size:
        with open(path, "rb") as f:
            h.update(f.read(min(_FP_BYTES, size)))
            f.seek(max(0, size - _FP_BYTES))
            h.update(f.read(size - max(0, size - _FP_BYTES)))
    return h.hexdigest()


def _meta_of(off: int, n: int, r: Mapping) -> list:
    """The cached metadata of one ``records.jsonl`` line: ``[offset, length,
    feed, partition, partition_sha256, seen_ts, period_start, n_points,
    min data_ts, max data_ts, leaf hash hex]``."""
    ts = [int(t) for _pj, t in r["points"]]
    return [int(off), int(n), r["feed"], r["partition"], r["partition_sha256"],
            int(r["seen_ts"]), int(r["period_start"]), len(ts),
            min(ts) if ts else None, max(ts) if ts else None,
            _leaf_hash(canonical_json(r)).hex()]


def _write_json_atomic(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(canonical_json(doc))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_cache(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") == CACHE_SCHEMA else None


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
    del df
    # One panel_json string per distinct panel (not per row): the same keys,
    # far fewer string objects in the per-feed ``known`` set downstream.
    pj_of: dict[tuple, str] = {}
    out = set()
    for i in np.flatnonzero(ok):
        vals = tuple(col[i] for col in cols)
        pj = pj_of.get(vals)
        if pj is None:
            pj = pj_of[vals] = panel_json(dict(zip(names, vals, strict=True)))
        out.add((pj, int(secs[i])))
    return sorted(out)


def _lag_entry(lag: list[int], lead: int) -> dict:
    lead_h = lead / 3600.0
    return {
        "mirror_lag_hours": (statistics.median(lag) / 3600.0) if lag else None,
        "max_lead_hours": lead_h,
        "publish_ahead": bool(lead_h > 0),
    }


def _feed_lag_of(points) -> tuple[list[int], int] | None:
    """``(lags, max lead)`` of one feed's ``(data_ts, (seen, root, baseline))``
    pairs (``None`` when the feed has no points)."""
    lags: list[int] = []
    lead: int | None = None
    for ts, (seen, _root, baseline) in points:
        d = ts - seen
        lead = d if lead is None else max(lead, d)
        if not baseline and ts <= seen:
            lags.append(seen - ts)
    return None if lead is None else (lags, lead)


class FirstSeenIndex:
    """First-seen lookups over the committed records up to ``until_ts``
    (every feed in memory at once: :meth:`FirstSeenLog.index`)."""

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

    def feed_lag(self, feeds: Sequence[str] | None = None) -> dict[str, dict]:
        """Per feed with points (only ``feeds`` when given): the measured
        mirror lag, the maximum lead and ``publish_ahead``."""
        want = None if feeds is None else {str(f) for f in feeds}
        by_feed: dict[str, list] = {}
        for (feed, _panel, ts), v in self._entries.items():
            if want is None or feed in want:
                by_feed.setdefault(feed, []).append((ts, v))
        out = {}
        for feed in sorted(by_feed):
            lag, lead = _feed_lag_of(by_feed[feed])
            out[feed] = _lag_entry(lag, lead)
        return out


class FirstSeenFeedIndex:
    """The same lookups as :class:`FirstSeenIndex`, holding ONE feed at a time.

    2026-11-26: ``pool_d_build`` over a 20 GB / 7,002-feed ``records.jsonl``
    reached 39.3 GB RSS in 5 min building the whole-log dict. This index keeps
    only per-feed byte spans of the usable records (verified against their
    roots by :meth:`FirstSeenLog.feed_index`); a lookup for a feed loads that
    feed's points from the log and drops the previous feed's. Answers are
    identical to :meth:`FirstSeenLog.index`; callers that alternate feeds stay
    correct but re-read, so group lookups by feed."""

    def __init__(self, records_path: Path, spans: dict[str, list[tuple[int, int]]],
                 usable: dict[int, str], until_ts: int):
        self._path = Path(records_path)
        self._spans = spans                # feed -> [(offset, length)] in file order
        self._usable = usable              # period_start -> root
        self.until_ts = int(until_ts)
        self._cur_feed: str | None = None
        self._cur: dict = {}
        self.loads = 0                     # feed loads so far (tests, diagnostics)

    def feeds(self) -> list[str]:
        return sorted(self._spans)

    def _load(self, feed: str) -> dict:
        """``{(panel_json, ts): (seen, root, baseline)}`` of ``feed``: the
        records in log order with the whole-log rule (earliest seen wins, the
        first record on a tie)."""
        entries: dict = {}
        intern: dict[str, str] = {}
        for r in FirstSeenLog._iter_spans(self._path, self._spans.get(feed)):
            seen = int(r["seen_ts"])
            val = (seen, self._usable[int(r["period_start"])], bool(r["baseline"]))
            for pj, t in r["points"]:
                pj = str(pj)
                key = (intern.setdefault(pj, pj), int(t))
                if key not in entries or seen < entries[key][0]:
                    entries[key] = val
        return entries

    def _feed(self, feed) -> dict:
        feed = str(feed)
        if feed != self._cur_feed:
            self._cur = {}                 # free the previous feed before loading
            self._cur_feed = None
            self._cur = self._load(feed)
            self._cur_feed = feed
            self.loads += 1
        return self._cur

    def release(self) -> None:
        """Drop the loaded feed."""
        self._cur, self._cur_feed = {}, None

    def __len__(self) -> int:
        n = 0
        for feed in self.feeds():
            n += len(self._load(feed))
        return n

    def _get(self, feed, panel, data_ts):
        return self._feed(feed).get((panel_json(panel), int(data_ts)))

    def first_seen(self, feed: str, panel: Mapping | str, data_ts: int) -> int | None:
        hit = self._get(feed, panel, data_ts)
        return None if hit is None else hit[0]

    def root_for(self, feed: str, panel: Mapping | str, data_ts: int) -> str | None:
        hit = self._get(feed, panel, data_ts)
        return None if hit is None else hit[1]

    def feed_points(self, feed: str) -> dict[tuple[str, int], int]:
        return {k: v[0] for k, v in self._feed(feed).items()}

    def feed_lag(self, feeds: Sequence[str] | None = None) -> dict[str, dict]:
        """As :meth:`FirstSeenIndex.feed_lag`, one feed loaded at a time."""
        names = self.feeds() if feeds is None else sorted({str(f) for f in feeds})
        out = {}
        for feed in names:
            if feed not in self._spans:
                continue
            got = _feed_lag_of((t, v) for (_p, t), v in self._feed(feed).items())
            if got is not None:
                out[feed] = _lag_entry(*got)
        return out

    def to_index(self) -> FirstSeenIndex:
        """The whole-log :class:`FirstSeenIndex` (every feed in memory)."""
        entries: dict = {}
        for feed in self.feeds():
            for (pj, t), v in self._load(feed).items():
                entries[(feed, pj, t)] = v
        return FirstSeenIndex(entries, self.until_ts)


class FirstSeenLog:
    """The append-only ingest log in ``log_dir``."""

    def __init__(self, log_dir: Path | str, *, cache_dir: Path | str | None = None,
                 full: bool = False, full_verify_seconds: float = FULL_VERIFY_SECONDS,
                 clock=time.time):
        """``cache_dir`` None = no cache (every scan reads everything);
        ``full`` = ignore the cache this run (and rewrite it from a full scan)."""
        self.log_dir = Path(log_dir)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.full = bool(full)
        self.full_verify_seconds = float(full_verify_seconds)
        self._clock = clock

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
        return [doc for _off, _len, doc in FirstSeenLog._scan(path)]

    @staticmethod
    def _scan(path: Path, start: int = 0):
        """Yield ``(byte offset, byte length, parsed doc)`` per non-blank line
        from byte ``start`` (a line boundary), one line in memory at a time
        (``log_corrupt`` on a bad line)."""
        if not path.is_file():
            return
        with open(path, "rb") as f:
            f.seek(int(start))
            off = int(start)
            for n, raw in enumerate(f, 1):
                start, off = off, off + len(raw)
                line = raw.decode("utf-8")
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    raise FirstSeenError("log_corrupt", f"{path}:{n}: {e}") from e
                yield start, len(raw), doc

    def records(self) -> list[dict]:
        return self._lines(self.records_path)

    # ── the incremental scan cache (module docstring) ───────────────────────

    def _full_due(self, cached: Mapping | None) -> bool:
        if self.full or cached is None:
            return True
        last = cached.get("full_verify_ts")
        return not isinstance(last, int | float) or \
            self._clock() - float(last) >= self.full_verify_seconds

    def _records_meta(self) -> list[list]:
        """Per-line metadata (:func:`_meta_of`) of every ``records.jsonl``
        line, in file order: the cached lines plus the lines appended since,
        or a full scan (no / stale / mismatching cache, ``full``, or the
        periodic full verification). Raises what :meth:`_scan` raises."""
        path = self.records_path
        size = path.stat().st_size if path.is_file() else 0
        cpath = self.cache_dir / CACHE_META_FILE
        cached = _read_cache(cpath)
        prefix_ok = (cached is not None and isinstance(cached.get("size"), int)
                     and cached["size"] <= size
                     and cached.get("fingerprint") == _fingerprint(path, cached["size"]))
        if cached is not None and not prefix_ok and not self.full:
            _warn(f"first-seen scan cache {cpath} does not match {path} (rewritten or "
                  f"restored log?); full scan")
        if self._full_due(cached) or not prefix_ok:
            lines = [_meta_of(off, n, r) for off, n, r in self._scan(path)]
            if prefix_ok and lines[:len(cached["lines"])] != cached["lines"]:
                _warn(f"first-seen scan cache {cpath} DIVERGED from a full scan of {path}; "
                      f"the full scan is used and the cache rebuilt")
            full_ts = self._clock()
        else:
            lines = list(cached["lines"])
            lines.extend(_meta_of(off, n, r) for off, n, r in self._scan(path, cached["size"]))
            full_ts = cached["full_verify_ts"]
        end_ok = size == 0
        if size:
            with open(path, "rb") as f:
                f.seek(size - 1)
                end_ok = f.read(1) == b"\n"
        if end_ok:                         # cache only a prefix that ends a line
            _write_json_atomic(cpath, {"schema": CACHE_SCHEMA, "size": size,
                                       "fingerprint": _fingerprint(path, size),
                                       "full_verify_ts": full_ts, "lines": lines})
        return lines

    def _partition_cache(self) -> tuple[dict, float | None]:
        """``({"feed/name": [size, mtime_ns, inode, sha256]}, last full
        verification ts)``; the ts is None when this run must hash everything."""
        cached = _read_cache(self.cache_dir / CACHE_PARTS_FILE)
        entries = cached.get("entries") if cached is not None else None
        entries = entries if isinstance(entries, dict) else {}
        if self._full_due(cached):
            return entries, None           # every partition re-hashed, compared
        return entries, float(cached["full_verify_ts"])

    def _partition_sha(self, feed: str, part: Path, cache: dict, out: dict,
                       full: bool) -> str:
        """``part``'s sha256: the cached one when its size, mtime_ns and inode
        are unchanged (never under ``full``), else hashed."""
        st = part.stat()
        key, stat = f"{feed}/{part.name}", [int(st.st_size), int(st.st_mtime_ns), int(st.st_ino)]
        hit = cache.get(key)
        if not full and isinstance(hit, list) and len(hit) == 4 and hit[:3] == stat:
            sha = str(hit[3])
        else:
            sha = _file_sha(part)
            if isinstance(hit, list) and len(hit) == 4 and hit[:3] == stat and hit[3] != sha:
                _warn(f"partition sha cache DIVERGED for data/{key} (same size/mtime/inode, "
                      f"new bytes); the hash is used and the cache rebuilt")
        out[key] = [*stat, sha]
        return sha

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

    @staticmethod
    def _write_docs(f, docs) -> None:
        for doc in docs:
            f.write(canonical_json(doc) + b"\n")

    def _append_staged(self, path: Path, staged) -> None:
        """Append a staged (binary, canonical-JSON lines) temp file to ``path``
        like :meth:`_append`: nothing when empty, fsync after the copy."""
        if staged.tell() == 0:
            return
        staged.flush()
        staged.seek(0)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as f:
            shutil.copyfileobj(staged, f, 1 << 20)
            f.flush()
            os.fsync(f.fileno())

    def _scan_history(self) -> dict:
        """One streaming pass over ``records.jsonl`` and ``quarantine.jsonl``
        keeping only what is small: the logged (feed, partition, sha256)
        triples, the logged feeds, the quarantine keys and, per feed, the byte
        spans of the lines carrying points (``known`` is then loaded one feed
        at a time by :meth:`_feed_known`, never the whole history at once)."""
        seen_parts: set = set()
        logged_feeds: set = set()
        rec_spans: dict[str, list[tuple[int, int]]] = {}
        # cached runs only: per feed [(offset, length, min data_ts, max data_ts)]
        rec_ranges: dict[str, list[tuple[int, int, int, int]]] = {}
        if self.cache_dir is not None:
            for off, n, feed, part, sha, _seen, _p, npts, lo, hi, _leaf in self._records_meta():
                seen_parts.add((feed, part, sha))
                logged_feeds.add(feed)
                if npts:
                    rec_spans.setdefault(feed, []).append((off, n))
                    rec_ranges.setdefault(feed, []).append((off, n, lo, hi))
        else:
            for off, n, r in self._scan(self.records_path):
                seen_parts.add((r["feed"], r["partition"], r["partition_sha256"]))
                logged_feeds.add(r["feed"])
                if r["points"]:
                    rec_spans.setdefault(r["feed"], []).append((off, n))
        q_names: set = set()
        q_versions: dict = {}
        blk_spans: dict[str, list[tuple[int, int]]] = {}
        for off, n, e in self._scan(self.quarantine_path):
            key = (str(e["feed"]), str(e["partition"]), str(e["partition_sha256"]))
            if e.get("kind") == "unreadable":
                q_names.add(key[:2])
            q_versions[key] = e.get("kind")
            if e.get("kind") == "blocked" and e.get("points"):
                blk_spans.setdefault(key[0], []).append((off, n))
        out = {"seen_parts": seen_parts, "logged_feeds": logged_feeds,
               "rec_spans": rec_spans, "q_names": q_names, "q_versions": q_versions,
               "blk_spans": blk_spans}
        if self.cache_dir is not None:
            out["rec_ranges"] = rec_ranges
        return out

    @staticmethod
    def _read_spans(path: Path, spans) -> list[dict]:
        if not spans:
            return []
        out = []
        with open(path, "rb") as f:
            for off, n in spans:
                f.seek(off)
                out.append(json.loads(f.read(n).decode("utf-8")))
        return out

    @staticmethod
    def _iter_spans(path: Path, spans):
        """:meth:`_read_spans` one parsed line at a time."""
        if not spans:
            return
        with open(path, "rb") as f:
            for off, n in spans:
                f.seek(off)
                yield json.loads(f.read(n).decode("utf-8"))

    def _feed_known(self, feed: str, hist: dict) -> set:
        """Every point key already logged for ``feed``: its records' points
        plus its ``blocked`` quarantine points (the same set the whole-history
        load built, for this one feed)."""
        known: set = set()
        intern: dict[str, str] = {}
        for r in self._read_spans(self.records_path, hist["rec_spans"].get(feed)):
            known.update((intern.setdefault(pj, pj), int(t)) for pj, t in r["points"])
        for e in self._read_spans(self.quarantine_path, hist["blk_spans"].get(feed)):
            known.update((intern.setdefault(pj, pj), int(t)) for pj, t in e.get("points") or [])
        return known

    def _feed_known_in(self, feed: str, hist: dict, lo: int, hi: int, have: set,
                       loaded: set) -> None:
        """Cached runs: add to ``have`` the points of ``feed``'s records whose
        data-time range meets ``[lo, hi]`` (not loaded yet). A logged point
        with data time in ``[lo, hi]`` lies in such a record, so ``points -
        have`` equals ``points - (every known point)`` for points in range."""
        spans = [(off, n) for off, n, rlo, rhi in hist["rec_ranges"].get(feed, ())
                 if (off, n) not in loaded and rlo <= hi and rhi >= lo]
        intern: dict[str, str] = {}
        for r in self._iter_spans(self.records_path, spans):
            have.update((intern.setdefault(pj, pj), int(t)) for pj, t in r["points"])
        loaded.update(spans)

    def _feed_blocked(self, feed: str, hist: dict) -> set:
        """``feed``'s ``blocked`` quarantine points (the small half of
        :meth:`_feed_known`)."""
        known: set = set()
        for e in self._read_spans(self.quarantine_path, hist["blk_spans"].get(feed)):
            known.update((pj, int(t)) for pj, t in e.get("points") or [])
        return known

    def record(self, forge_dir: Path | str, *, now_ts: int,
               max_unreadable_files: int = DEFAULT_MAX_UNREADABLE_FILES,
               max_unreadable_frac: float = DEFAULT_MAX_UNREADABLE_FRAC) -> dict:
        """Log every new point of the forge mirror's partitions at ``now_ts``.

        An unreadable partition is skipped and quarantined (module docstring);
        more than :func:`unreadable_limit` unreadable partitions (new ones plus
        quarantined versions still present) raises ``too_many_unreadable``
        before anything is written. The result gains ``quarantined`` /
        ``blocked`` / ``unreadable`` only when non-zero.

        Memory is bounded by one feed (2026-11-25: a 7,002-feed baseline held
        every point of every feed and OOM-killed a 39 GB builder): feeds are
        processed one at a time, each feed's ``known`` set is loaded from the
        log for that feed only and dropped when it is done, and its new lines
        go to two unlinked staging files in ``log_dir``. Only after the
        unreadable check passes are they appended, quarantine first, so a
        systemic failure still records nothing and the logs get exactly the
        bytes, in the same order, that the whole-forge version wrote."""
        now_ts = int(now_ts)
        p = period_start(now_ts)
        roots = self.roots()
        if p in roots:
            raise FirstSeenError("period_committed",
                                 f"period {p} already has a committed root; now_ts={now_ts}")
        last_commit = max((int(e["committed_ts"]) for e in roots.values()), default=None)
        if last_commit is not None and p + PERIOD_SECONDS <= last_commit:
            # A commit at last_commit froze every period ended by then (a
            # roots="ended" snapshot relied on them); never record into one.
            raise FirstSeenError("period_frozen",
                                 f"period {p} ended before the last commit at {last_commit}; "
                                 f"now_ts={now_ts} is in the past")
        hist = self._scan_history()
        cached = self.cache_dir is not None
        if cached:
            part_cache, part_verified = self._partition_cache()
            part_full = part_verified is None
            part_out: dict = {}
        seen_parts, logged_feeds = hist["seen_parts"], hist["logged_feeds"]
        q_names, q_versions = hist["q_names"], hist["q_versions"]
        data = Path(forge_dir) / "data"
        unreadable: list[str] = []
        n_points = n_blocked = n_parts = n_records = n_q = 0
        any_blocked = False
        feed_dirs = sorted(d for d in data.iterdir() if d.is_dir()) if data.is_dir() else []
        # The staging files live in log_dir (same filesystem, never tmpfs);
        # dirs made only for them are removed again when nothing was written.
        made_dirs = [d for d in (self.log_dir, *self.log_dir.parents) if not d.exists()]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryFile(dir=self.log_dir) as stage_q, \
                    tempfile.TemporaryFile(dir=self.log_dir) as stage_r:
                for feed_dir in feed_dirs:
                    feed = feed_dir.name
                    have: set | None = None       # this feed's known points, loaded lazily
                    loaded: set = set()           # cached runs: record spans already in `have`
                    for part in sorted(feed_dir.glob("*.parquet")):
                        n_parts += 1
                        sha = (self._partition_sha(feed, part, part_cache, part_out, part_full)
                               if cached else _file_sha(part))
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
                            self._write_docs(stage_q, [{
                                "schema": QUARANTINE_SCHEMA, "kind": "unreadable",
                                "feed": feed, "partition": part.name,
                                "partition_sha256": sha, "size": int(size),
                                "reason": reason, "quarantined_ts": now_ts}])
                            n_q += 1
                            q_versions[(feed, part.name, sha)] = "unreadable"
                            q_names.add((feed, part.name))
                            continue
                        if cached:
                            if have is None:
                                have = self._feed_blocked(feed, hist)
                            if points:
                                ts = [t for _pj, t in points]
                                self._feed_known_in(feed, hist, min(ts), max(ts), have, loaded)
                        elif have is None:
                            have = self._feed_known(feed, hist)
                        fresh = sorted(set(points) - have)
                        del points
                        have.update(fresh)
                        if (feed, part.name) in q_names:
                            # A quarantined partition readable again: its unknown
                            # points are blocked forever, never first-seen now.
                            _warn(f"quarantined partition data/{feed}/{part.name} is readable "
                                  f"again (size={size} sha256={sha}); {len(fresh)} unseen "
                                  f"point(s) blocked from lane D, not recorded as first seen")
                            n_blocked += len(fresh)
                            any_blocked = True
                            self._write_docs(stage_q, [{
                                "schema": QUARANTINE_SCHEMA, "kind": "blocked",
                                "feed": feed, "partition": part.name,
                                "partition_sha256": sha, "size": int(size),
                                "quarantined_ts": now_ts,
                                "points": [[pj, int(t)] for pj, t in fresh]}])
                            q_versions[(feed, part.name, sha)] = "blocked"
                            continue
                        n_points += len(fresh)
                        n_records += 1
                        self._write_docs(stage_r, [{
                            "schema": LOG_SCHEMA, "feed": feed, "partition": part.name,
                            "partition_sha256": sha, "seen_ts": now_ts, "period_start": p,
                            "baseline": feed not in logged_feeds,
                            "points": [[pj, int(t)] for pj, t in fresh],
                        }])
                        del fresh
                    del have, loaded                      # free this feed before the next
                    self._feed_state_freed(feed)
                limit = unreadable_limit(n_parts, max_files=max_unreadable_files,
                                         max_frac=max_unreadable_frac)
                if len(unreadable) > limit:
                    raise FirstSeenError(
                        "too_many_unreadable",
                        f"{len(unreadable)} of {n_parts} partitions are unreadable (limit "
                        f"{limit}); a systemic read failure? nothing recorded. First: "
                        f"{', '.join(unreadable[:10])}")
                # Quarantine first: a crash between the two appends errs on the safe side.
                self._append_staged(self.quarantine_path, stage_q)
                self._append_staged(self.records_path, stage_r)
                if cached:
                    _write_json_atomic(self.cache_dir / CACHE_PARTS_FILE, {
                        "schema": CACHE_SCHEMA, "entries": part_out,
                        "full_verify_ts": self._clock() if part_full else part_verified})
        finally:
            for d in made_dirs:                           # deepest first
                try:
                    d.rmdir()                             # only succeeds when nothing was written
                except OSError:
                    break
        out = {"records": n_records, "points": n_points}
        if n_q:
            out["quarantined"] = n_q
        if n_blocked or any_blocked:
            out["blocked"] = n_blocked
        if unreadable:
            out["unreadable"] = len(unreadable)
            _warn(f"{len(unreadable)} unreadable partition(s) skipped (limit {limit}): "
                  f"{', '.join(unreadable[:20])}")
        return out

    def _feed_state_freed(self, feed: str) -> None:
        """Hook called after a feed's per-feed state is dropped (tests)."""

    def commit(self, *, now_ts: int) -> list[dict]:
        """Commit a root for every ended period that has records and no root.

        Streams ``records.jsonl``: only the 32-byte leaf hashes of the periods
        to commit are kept, never the records themselves."""
        now_ts = int(now_ts)
        roots = self.roots()
        by_period: dict[int, list[bytes]] = {}
        if self.cache_dir is not None:
            for m in self._records_meta():
                pr = m[6]
                if pr in roots or pr + PERIOD_SECONDS > now_ts:
                    continue
                by_period.setdefault(pr, []).append(bytes.fromhex(m[10]))
        for _off, _n, r in (() if self.cache_dir is not None else self._scan(self.records_path)):
            pr = int(r["period_start"])
            if pr in roots or pr + PERIOD_SECONDS > now_ts:
                continue
            by_period.setdefault(pr, []).append(_leaf_hash(canonical_json(r)))
        added = []
        for pr in sorted(by_period):
            hashes = by_period[pr]
            added.append({"period_start": pr, "root": _merkle_root_of_hashes(hashes),
                          "n_records": len(hashes), "committed_ts": now_ts})
        self._append(self.roots_path, added)
        return added

    # ── the index ───────────────────────────────────────────────────────────

    def index(self, *, until_ts: int, roots: str = "committed") -> FirstSeenIndex:
        """Points of the periods whose root was committed by ``until_ts``
        (``roots`` as :meth:`feed_index`), after checking every committed root
        against the log (every feed in memory; :meth:`feed_index` holds one
        feed at a time)."""
        return self.feed_index(until_ts=until_ts, roots=roots).to_index()

    def feed_index(self, *, until_ts: int, roots: str = "committed") -> FirstSeenFeedIndex:
        """:meth:`index` that holds one feed at a time.

        ``roots`` picks the usable periods (module docstring): ``"committed"``
        = root committed by ``until_ts``; ``"ended"`` = period ended by
        ``until_ts`` and committed (any time), refusing ``uncommitted_period``
        when an ended period has records but no root.

        One streaming pass over ``records.jsonl`` checks every committed root
        (32-byte leaf hashes only) and keeps, per feed, the byte spans of the
        usable records: a committed-by-``until_ts`` period, not a quarantined
        name recorded at/after its quarantine, with points. The errors (and
        which one wins when several apply) are :meth:`index`'s before the
        2026-11-26 fix: ``log_corrupt`` in the records, then the roots file,
        then a wrong ``period_start``, then ``root_mismatch``, then the
        quarantine file."""
        until_ts = int(until_ts)
        if roots not in ROOT_RULES:
            raise ValueError(f"roots must be one of {ROOT_RULES}; got {roots!r}")
        rule = roots
        try:
            roots = self.roots()
            roots_err = None
        except Exception as e:  # noqa: BLE001 — re-raised after the records scan
            roots, roots_err = {}, e
        try:
            quarantined = self.quarantined(until_ts=until_ts)
            q_err = None
        except Exception as e:  # noqa: BLE001 — re-raised after the root checks
            quarantined, q_err = {}, e
        if rule == "ended":
            usable = {p: e["root"] for p, e in roots.items() if p + PERIOD_SECONDS <= until_ts}
        else:
            usable = {p: e["root"] for p, e in roots.items()
                      if int(e["committed_ts"]) <= until_ts}
        hashes: dict[int, list[bytes]] = {}
        spans: dict[str, list[tuple[int, int]]] = {}
        open_ended: set[int] = set()       # ended by until_ts, records, no root
        period_err: FirstSeenError | None = None
        if self.cache_dir is not None:
            lines = ((m[0], m[1], m[2], m[3], m[5], m[6], m[7], bytes.fromhex(m[10]))
                     for m in self._records_meta())
        else:
            lines = ((off, n, r["feed"], r["partition"], int(r["seen_ts"]),
                      int(r["period_start"]), len(r["points"]), _leaf_hash(canonical_json(r)))
                     for off, n, r in self._scan(self.records_path))
        for off, n, feed, part, seen, p, npts, leaf in lines:
            if period_err is None and p != period_start(seen):
                period_err = FirstSeenError(
                    "root_mismatch", f"record of {feed} has a wrong period_start")
            if p in roots:
                hashes.setdefault(p, []).append(leaf)
            elif p + PERIOD_SECONDS <= until_ts:
                open_ended.add(p)
            if p not in usable or not npts:
                continue
            q_ts = quarantined.get((str(feed), str(part)))
            if q_ts is not None and seen >= q_ts:
                # A quarantined name recorded at/after its quarantine (e.g. by
                # a release that did not know quarantine.jsonl): never fresh.
                continue
            spans.setdefault(str(feed), []).append((off, n))
        if roots_err is not None:
            raise roots_err
        if period_err is not None:
            raise period_err
        for p, entry in roots.items():
            hs = hashes.get(p, [])
            got = _merkle_root_of_hashes(hs) if hs else None
            if got != entry["root"] or len(hs) != int(entry["n_records"]):
                raise FirstSeenError("root_mismatch",
                                     f"period {p}: log no longer matches its committed root")
        if q_err is not None:
            raise q_err
        if rule == "ended" and open_ended:
            raise FirstSeenError(
                "uncommitted_period",
                f"period(s) {sorted(open_ended)} ended by {until_ts} hold records but no "
                f"committed root; run `commit` first (roots='ended' needs every ended "
                f"period final)")
        return FirstSeenFeedIndex(self.records_path, spans, usable, until_ts)


def verify_snapshot(meta: Mapping, log_dir: Path | str, *, roots: str = "committed",
                    cache_dir: Path | str | None = None) -> list[str]:
    """Problems of a ``pool_d/1`` metadata dict against the log (``[]`` = ok);
    ``roots="ended"`` for a snapshot the builder cut under that rule."""
    problems: list[str] = []
    try:
        as_of = int(meta["as_of_ts"])
    except (KeyError, TypeError, ValueError):
        return ["metadata has no integer as_of_ts"]
    try:
        idx = FirstSeenLog(log_dir, cache_dir=cache_dir).feed_index(until_ts=as_of, roots=roots)
    except FirstSeenError as e:
        return [f"first-seen log: {e}"]
    listed = set(meta.get("first_seen_roots") or [])
    series = list(meta.get("series") or [])
    # Checked grouped by feed (one feed loaded at a time), reported in order.
    per: dict[int, list[str]] = {}
    order = sorted(range(len(series)),
                   key=lambda i: str(series[i].get("feed") or series[i].get("source")))
    for i in order:
        s = series[i]
        out = per[i] = []
        sid = s.get("series_id")
        ts, seen = s.get("target_ts") or [], s.get("first_seen_ts") or []
        if len(ts) != len(seen):
            out.append(f"{sid}: target_ts and first_seen_ts differ in length")
            continue
        feed, panel = s.get("feed") or s.get("source"), s.get("panel") or {}
        for t, claimed in zip(ts, seen, strict=True):
            got = idx.first_seen(feed, panel, t)
            if got is None:
                out.append(f"{sid}: point {t} has no first-seen record committed by "
                           f"as_of {as_of}")
                continue
            if int(claimed) != got:
                out.append(f"{sid}: point {t} first_seen_ts {claimed} != log {got}")
            root = idx.root_for(feed, panel, t)
            if root not in listed:
                out.append(f"{sid}: point {t} root {root} not in first_seen_roots")
    idx.release()
    for i in range(len(series)):
        problems.extend(per[i])
    return problems


# ── CLI ─────────────────────────────────────────────────────────────────────


def _evidence(log: FirstSeenLog, now_ts: int) -> dict:
    n_records = n_points = 0
    for _off, _n, r in log._scan(log.records_path):     # streamed, never all at once
        n_records += 1
        n_points += len(r["points"])
    roots = sorted(log.roots().values(), key=lambda e: int(e["period_start"]))
    doc = {
        "schema": EVIDENCE_SCHEMA, "wp": "W6fs", "source": "builder",
        "generated_by": "cascade.pool.first_seen evidence",
        "host": socket.gethostname(), "generated_ts": int(now_ts),
        "n_records": n_records, "n_points": n_points,
        "roots": [{"period_start": int(e["period_start"]), "root": e["root"],
                   "n_records": int(e["n_records"]), "committed_ts": int(e["committed_ts"])}
                  for e in roots],
        "feeds": log.feed_index(until_ts=now_ts).feed_lag(),
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
        else:
            p.add_argument("--cache-dir", type=Path, default=None,
                           help="incremental scan cache (default: <log-dir>.cache)")
            p.add_argument("--no-cache", action="store_true",
                           help="scan everything, read and write no cache")
            p.add_argument("--full", action="store_true",
                           help="ignore the cache this run: full scan, cache rebuilt")
    args = ap.parse_args(argv)
    now = int(time.time()) if args.now_ts is None else int(args.now_ts)
    log_dir = args.log_dir
    if log_dir is None:
        if getattr(args, "forge_dir", None) is None:
            ap.error("--log-dir is required")
        log_dir = args.forge_dir / DEFAULT_LOG_DIRNAME
    cache_dir = None
    if args.cmd != "evidence" and not args.no_cache:
        cache_dir = args.cache_dir or default_cache_dir(log_dir)
    log = FirstSeenLog(log_dir, cache_dir=cache_dir, full=getattr(args, "full", False))
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
    "CACHE_META_FILE",
    "CACHE_PARTS_FILE",
    "CACHE_SCHEMA",
    "DEFAULT_MAX_UNREADABLE_FILES",
    "DEFAULT_MAX_UNREADABLE_FRAC",
    "EVIDENCE_SCHEMA",
    "FULL_VERIFY_SECONDS",
    "ROOT_RULES",
    "LOG_SCHEMA",
    "PERIOD_SECONDS",
    "QUARANTINE_FILE",
    "QUARANTINE_SCHEMA",
    "FirstSeenError",
    "FirstSeenFeedIndex",
    "FirstSeenIndex",
    "FirstSeenLog",
    "canonical_json",
    "default_cache_dir",
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
