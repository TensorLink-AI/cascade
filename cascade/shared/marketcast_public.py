"""The public MarketCast bar history (owner 2026-11-27): the recorder's bars
OLDER than the training embargo, as Parquet, for miners to download.

Lane C recipes must end their training data ``[marketcast] embargo_days``
before submission and the eval draws from the ``embargo_eval_days`` before
dispatch, so a bar older than ``today - embargo_days`` is one every recipe may
already train on. This module is the one definition of that cut, the layout
and the index both the publisher (``scripts/publish_marketcast_public.py``,
scheduled by ``.github/workflows/publish-mkt-public.yml``) and the miner
download command (``cascade lane-c data``) use. numpy only at import; pyarrow
is imported lazily by the Parquet writer/reader (publisher side, the
``pool-forge`` extra).

Layout (identical on the HF dataset and the public Hippius prefix)::

    data/<class>/<SYMBOL>/<tf>/<YYYY-MM>.parquet     tf in 1m / 5m / 15m / 1h
        timestamp  timestamp[ms, UTC]  bar OPEN time
        open high low close volume     float64 (NaN never written: a bar
                                       nobody recorded is an absent row)
    data/<class>/<SYMBOL>/funding/<YYYY-MM>.parquet
        timestamp  timestamp[ms, UTC]  sample time
        rate       float64             hourly funding rate
        premium    float64             NaN when the API gave none
    index.json                         every file: sha256, bytes, rows, first/last day
    README.md                          the dataset card

A month file holds only the month's days strictly before the cut, so the
current month's file grows by a day per day and is re-uploaded (its content
hash changes); a closed month never changes again.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

HF_REPO = "tensorlink-dev/cascade-marketcast-bars"
HIPPIUS_PREFIX = "marketcast/public-bars"
INDEX_NAME = "index.json"
INDEX_SCHEMA = "cascade-marketcast-public-bars/1"
BAR_FIELDS = ("open", "high", "low", "close", "volume")
FUNDING_FIELDS = ("rate", "premium")
TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}   # the recorder's banks (minutes)
FUNDING = "funding"
DAY_S = 86400
PUBLIC_SOURCES = ("hyperliquid", "hyperliquid_xyz")     # recorded by MktBarStore

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


class EmbargoViolation(AssertionError):
    """A row on or after the embargo cut reached a public file."""


# ── the embargo cut ──────────────────────────────────────────────────────────


def embargo_cutoff(today: date, embargo_days: int) -> date:
    """The first EMBARGOED day: a day is public iff ``day < cutoff``
    (strictly older than ``today - embargo_days``)."""
    if int(embargo_days) < 1:
        raise ValueError(f"embargo_days must be >= 1, got {embargo_days}")
    return today - timedelta(days=int(embargo_days))


def cutoff_ms(cutoff: date) -> int:
    return int(datetime(cutoff.year, cutoff.month, cutoff.day, tzinfo=UTC).timestamp()) * 1000


def is_public_day(day: date, cutoff: date) -> bool:
    return day < cutoff


def assert_before_cutoff(ts_ms: Iterable[int] | np.ndarray, cutoff: date, *, what: str = "") -> None:
    """Raise :class:`EmbargoViolation` if any timestamp is at/after the cut."""
    a = np.asarray(ts_ms, dtype=np.int64)
    if a.size and int(a.max()) >= cutoff_ms(cutoff):
        last = datetime.fromtimestamp(int(a.max()) / 1000, tz=UTC).isoformat()
        raise EmbargoViolation(f"{what or 'file'}: row at {last} is inside the embargo "
                               f"(cut {cutoff.isoformat()})")


def parse_day(s: str) -> date:
    if not _DAY_RE.match(s):
        raise ValueError(f"not a YYYY-MM-DD day: {s!r}")
    return date.fromisoformat(s)


# ── layout ───────────────────────────────────────────────────────────────────


def file_path(cls: str, symbol: str, tf: str, month: str) -> str:
    """``data/<class>/<SYMBOL>/<tf>/<YYYY-MM>.parquet`` (tf may be ``funding``)."""
    for v in (cls, symbol):
        if not _NAME_RE.match(v):
            raise ValueError(f"unsafe name {v!r}")
    if tf not in TIMEFRAMES and tf != FUNDING:
        raise ValueError(f"unknown timeframe {tf!r}")
    if not _MONTH_RE.match(month):
        raise ValueError(f"not a YYYY-MM month: {month!r}")
    return f"data/{cls}/{symbol}/{tf}/{month}.parquet"


def parse_file_path(path: str) -> tuple[str, str, str, str] | None:
    """``(class, symbol, tf, month)`` of a layout path, else None."""
    m = re.match(r"^data/([^/]+)/([^/]+)/([^/]+)/(\d{4}-\d{2})\.parquet$", path)
    if not m:
        return None
    cls, sym, tf, month = m.groups()
    if tf not in TIMEFRAMES and tf != FUNDING:
        return None
    return cls, sym, tf, month


# ── reading the recorder's bank (MktBarStore layout) ─────────────────────────


def _bar_dir(root: Path, cls: str, symbol: str, tf: str) -> Path:
    d = root / cls / symbol
    return d if tf == "1m" else d / tf


def public_days(root: Path, cls: str, symbol: str, tf: str, cutoff: date) -> list[date]:
    """The recorded day directories of ``(cls, symbol, tf)`` strictly before
    the cut (sorted)."""
    d = _bar_dir(Path(root), cls, symbol, tf)
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        if p.is_dir() and _DAY_RE.match(p.name):
            day = date.fromisoformat(p.name)
            if is_public_day(day, cutoff):
                out.append(day)
    return sorted(out)


def read_bar_days(root: Path, cls: str, symbol: str, tf: str,
                  days: Iterable[date]) -> dict[str, np.ndarray]:
    """``{"timestamp": int64 ms, open..volume: float64}`` of ``days``, rows
    where every field is NaN dropped (a minute nobody recorded)."""
    step = 60 * TIMEFRAMES[tf]
    per_day = DAY_S // step
    ts, cols = [], {f: [] for f in BAR_FIELDS}
    base = _bar_dir(Path(root), cls, symbol, tf)
    for day in days:
        d = base / day.isoformat()
        arrs = {}
        for f in BAR_FIELDS:
            p = d / f"{f}.npy"
            a = np.load(p, allow_pickle=False) if p.is_file() else None
            arrs[f] = (a.astype(np.float64) if a is not None and a.shape == (per_day,)
                       else np.full(per_day, np.nan))
        keep = ~np.all(np.stack([np.isnan(arrs[f]) for f in BAR_FIELDS]), axis=0)
        day0 = cutoff_ms(day)
        ts.append(day0 + np.arange(per_day, dtype=np.int64)[keep] * step * 1000)
        for f in BAR_FIELDS:
            cols[f].append(arrs[f][keep])
    out = {"timestamp": np.concatenate(ts) if ts else np.zeros(0, np.int64)}
    for f in BAR_FIELDS:
        out[f] = np.concatenate(cols[f]) if cols[f] else np.zeros(0, np.float64)
    return out


def read_funding(root: Path, cls: str, symbol: str, cutoff: date) -> dict[str, np.ndarray]:
    """The recorded funding samples strictly before the cut, sorted by time."""
    p = Path(root) / cls / symbol / "funding.jsonl"
    rows: dict[int, tuple[float, float]] = {}
    if p.is_file():
        lim = cutoff_ms(cutoff)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            t = int(r["t"])
            if t < lim:
                prem = r.get("premium")
                rows[t] = (float(r["rate"]), float("nan") if prem is None else float(prem))
    ks = sorted(rows)
    return {"timestamp": np.asarray(ks, dtype=np.int64),
            "rate": np.asarray([rows[k][0] for k in ks], dtype=np.float64),
            "premium": np.asarray([rows[k][1] for k in ks], dtype=np.float64)}


def split_by_month(cols: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Partition a column dict by the UTC month of ``timestamp``."""
    ts = cols["timestamp"]
    if not ts.size:
        return {}
    months = np.array([datetime.fromtimestamp(int(t) // 1000, tz=UTC).strftime("%Y-%m")
                       for t in ts])
    out = {}
    for m in sorted(set(months.tolist())):
        sel = months == m
        out[m] = {k: v[sel] for k, v in cols.items()}
    return out


def content_sha256(cols: dict[str, np.ndarray]) -> str:
    """A hash of the column VALUES (names, dtypes, bytes): the idempotence key,
    independent of the Parquet writer's version string."""
    h = hashlib.sha256()
    for k in sorted(cols):
        a = np.ascontiguousarray(cols[k])
        h.update(k.encode() + b"\0" + str(a.dtype).encode() + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


# ── Parquet (pyarrow, lazy) ──────────────────────────────────────────────────


def write_parquet(path: Path, cols: dict[str, np.ndarray], cutoff: date) -> None:
    """Write one month file (zstd, float64 preserved, timestamp[ms, UTC]);
    refuses any row at/after the cut."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert_before_cutoff(cols["timestamp"], cutoff, what=str(path))
    fields = [k for k in cols if k != "timestamp"]
    arrays = [pa.array(cols["timestamp"].astype(np.int64), type=pa.int64())
              .cast(pa.timestamp("ms", tz="UTC"))]
    arrays += [pa.array(np.asarray(cols[f], dtype=np.float64), type=pa.float64()) for f in fields]
    table = pa.Table.from_arrays(arrays, names=["timestamp", *fields])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=9,
                   use_dictionary=False, write_statistics=True)


def read_parquet_timestamps_ms(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    col = pq.read_table(path, columns=["timestamp"]).column("timestamp")
    return np.asarray(col.cast("int64").to_numpy(), dtype=np.int64)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── index + selection (download side) ────────────────────────────────────────


def empty_index() -> dict:
    return {"schema": INDEX_SCHEMA, "files": {}}


def parse_index(text: str | bytes) -> dict:
    doc = json.loads(text)
    if not isinstance(doc, dict) or doc.get("schema") != INDEX_SCHEMA \
            or not isinstance(doc.get("files"), dict):
        raise ValueError("not a cascade-marketcast-public-bars/1 index")
    return doc


def _month_span(month: str) -> tuple[date, date]:
    y, m = (int(x) for x in month.split("-"))
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def select_files(index: dict, *, symbols: Iterable[str], timeframes: Iterable[str],
                 start: date | None = None, end: date | None = None,
                 classes: Iterable[str] | None = None) -> list[str]:
    """The index paths of ``symbols`` x ``timeframes`` (``funding`` allowed)
    whose month overlaps ``[start, end]`` (inclusive; None = open)."""
    syms = {s.upper() for s in symbols}
    tfs = set(timeframes)
    cls_ok = set(classes) if classes else None
    out = []
    for p in sorted(index.get("files", {})):
        parsed = parse_file_path(p)
        if parsed is None:
            continue
        cls, sym, tf, month = parsed
        if sym.upper() not in syms or tf not in tfs or (cls_ok and cls not in cls_ok):
            continue
        lo, hi = _month_span(month)
        if (start and hi < start) or (end and lo > end):
            continue
        out.append(p)
    return out


def hippius_public_url(endpoint: str, bucket: str, key: str) -> str:
    return f"{endpoint.rstrip('/')}/{bucket}/{key.lstrip('/')}"


__all__ = [
    "BAR_FIELDS", "FUNDING", "FUNDING_FIELDS", "HF_REPO", "HIPPIUS_PREFIX", "INDEX_NAME",
    "INDEX_SCHEMA", "PUBLIC_SOURCES", "TIMEFRAMES", "EmbargoViolation", "assert_before_cutoff",
    "content_sha256", "cutoff_ms", "embargo_cutoff", "empty_index", "file_path", "file_sha256",
    "hippius_public_url", "is_public_day", "parse_day", "parse_file_path", "parse_index",
    "public_days", "read_bar_days", "read_funding", "read_parquet_timestamps_ms",
    "select_files", "split_by_month", "write_parquet",
]
