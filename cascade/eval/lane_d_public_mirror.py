"""Lane D public mirror on Hippius S3: content-addressed Parquet shards, the
shard map, the eval host's ranged fetcher and its byte-budget cache
(owner 2026-11-24, issue #894; F3 / W6p, eval-host side W8c).

The public half's canonical mirror is an object set on Hippius S3 (R2 as the
fallback copy), not a 908 GiB local copy on every eval host. **The W6p index
does not change**: ``index.jsonl.gz`` + ``manifest.json`` are byte-identical
to the ones :func:`~cascade.eval.lane_d_public_index.build_index` writes over
a full local mirror of the pinned HF revision (``file`` / ``rows`` still name
the HF files, ``files_sha256`` still hashes them), so ``index_sha256`` and
``files_sha256`` in the signed report, the draw and the audit are untouched.
The mirror adds two files next to the index, keyed by its ``index_sha256``:

* ``shards.json`` (:data:`SHARDS_SCHEMA`): the source, repo, revision,
  ``index_sha256`` and ``files_sha256`` it was built from, ``n_series``,
  ``shardmap_sha256`` and the shard list ``{key, size, sha256, tail_offset,
  tail_length, tail_sha256, n_row_groups, n_series}``;
* ``shardmap.jsonl.gz``: one canonical JSON line per index row, in ordinal
  order: ``{ordinal, series_id, shard, row_group, row, offset, length,
  sha256, n_channels, n_points, values_sha256}``. ``offset`` / ``length`` /
  ``sha256`` are the byte range of the series' Parquet row group inside the
  shard; ``values_sha256`` (:func:`values_sha256`) is the hash of the series
  exactly as :func:`~cascade.eval.lane_d_public_index.read_series` returns it
  from the HF files (float64 ``(C, L)``, NaN canonical), so a decoded series
  is checked against the original data, not just against the shard bytes.

**Shards** (``shards/<sha256>.parquet``, ~256 MB, never spanning datasets):
columns ``series_id`` (string), ``n_channels`` (int32), ``n_points`` (int64),
``values`` (``list<float32>``) and ``values64`` (``list<float64>``), zstd,
byte-stream-split floats; a row holds its ``(C, L)`` series row-major in
``values`` when the float32 round trip is exact, else in ``values64`` (the
other is null), so the decoded series is bit-identical to ``read_series``.
Row groups hold whole series, closed at :data:`ROW_GROUP_TARGET_BYTES`
(raw), so a drawn series costs one row group GET plus the shard's tail
(footer) once.

Object layout under ``<prefix>/<source>/<revision>/`` (:func:`base_key`)::

    index/<index_sha256>/{index.jsonl.gz, manifest.json, shards.json, shardmap.jsonl.gz}
    shards/<sha256>.parquet

**Eval host** (:class:`PublicMirror`): after the box's deterministic draw it
fetches the index files once (verified against the draw's ``index_sha256`` /
``files_sha256``), then ONLY the drawn series' row groups by ranged GET:
volume cache first (:class:`BlobCache`, sha256-addressed, byte-budget LRU,
every read re-verified, a corrupt blob dropped and refetched), then Hippius,
then R2. A missing or wiped cache simply refills. Credentials are env-only
(``<key_env>_ACCESS_KEY`` / ``_SECRET_KEY``); without them the client is
anonymous (a public-read bucket).

Stdlib + numpy at import; ``pyarrow`` and ``boto3`` load inside the functions
that need them (the box never imports them through this module).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import logging
import os
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lane_d_public_index import (
    INDEX_FILE,
    MANIFEST_FILE,
    PUBLIC_SOURCES,
    canonical_json,
    iter_index,
    load_manifest,
)

log = logging.getLogger("cascade.eval.lane_d_public_mirror")

SHARDS_SCHEMA = "lane-d-public-shards/1"
SHARDS_FILE = "shards.json"
SHARDMAP_FILE = "shardmap.jsonl.gz"
INDEX_FILES = (INDEX_FILE, MANIFEST_FILE, SHARDS_FILE, SHARDMAP_FILE)
SHARDMAP_FIELDS = ("ordinal", "series_id", "shard", "row_group", "row", "offset", "length",
                   "sha256", "n_channels", "n_points", "values_sha256")
SHARD_FIELDS = ("key", "size", "sha256", "tail_offset", "tail_length", "tail_sha256",
                "n_row_groups", "n_series")
DEFAULT_PREFIX = "lane_d_public"
SHARD_TARGET_BYTES = 256 * 2**20
ROW_GROUP_TARGET_BYTES = 1 * 2**20
# The shard tail fetched once per shard: the footer, and at least this many
# bytes, so pyarrow's speculative footer read never leaves the fetched range.
TAIL_MIN_BYTES = 64 * 1024
ZSTD_LEVEL = 6
MIRROR_CONFIG_PATH = (Path(__file__).resolve().parents[2] / "deploy" / "laned"
                      / "public_mirror.toml")
MIRROR_CONFIG_SCHEMA = "laned_public_mirror/1"


class MirrorError(RuntimeError):
    """The mirror cannot serve a series: every tier failed, or the bytes do
    not verify. Infrastructure, never the miner's fault."""


def base_key(prefix: str, source: str) -> str:
    spec = PUBLIC_SOURCES[source]
    return f"{str(prefix).strip('/')}/{source}/{spec.revision}"


def index_key(prefix: str, source: str, index_sha: str, name: str) -> str:
    return f"{base_key(prefix, source)}/index/{index_sha}/{name}"


def shard_key(sha: str) -> str:
    """A shard's key RELATIVE to :func:`base_key` (what the shard map holds)."""
    return f"shards/{sha}.parquet"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_series(a) -> np.ndarray:
    """``(C, L)`` float64, C-contiguous, every NaN the canonical NaN."""
    x = np.asarray(a, dtype=np.float64)
    x = x.reshape(1, -1) if x.ndim == 1 else x
    x = np.ascontiguousarray(x)
    nan = np.isnan(x)
    if nan.any():
        x = x.copy()
        x[nan] = np.nan
    return x


def values_sha256(a) -> str:
    """sha256 of ``"<C>x<L>:"`` + the little-endian float64 bytes of the
    canonical series (:func:`canonical_series`)."""
    x = canonical_series(a)
    h = hashlib.sha256(f"{x.shape[0]}x{x.shape[1]}:".encode())
    h.update(x.astype("<f8", copy=False).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Shard map / shards.json (stdlib)
# ---------------------------------------------------------------------------

def _check_map_row(row: dict, ordinal: int) -> None:
    if set(row) != set(SHARDMAP_FIELDS):
        raise ValueError(f"shard map row {ordinal}: fields {sorted(row)}")
    if row["ordinal"] != ordinal:
        raise ValueError(f"shard map row {ordinal}: ordinal {row['ordinal']!r}")
    for k in ("row_group", "row", "offset", "length", "n_channels", "n_points"):
        v = row[k]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"shard map row {ordinal}: {k} must be a non-negative integer")
    for k in ("sha256", "values_sha256"):
        if not (isinstance(row[k], str) and len(row[k]) == 64):
            raise ValueError(f"shard map row {ordinal}: {k} must be a sha256 hex")
    if not (isinstance(row["shard"], str) and row["shard"].startswith("shards/")):
        raise ValueError(f"shard map row {ordinal}: bad shard key {row['shard']!r}")


def shardmap_lines(rows: Iterable[dict]) -> Iterator[bytes]:
    for i, row in enumerate(rows):
        _check_map_row(row, i)
        yield canonical_json(row) + b"\n"


def write_shard_index(out_dir, *, manifest: Mapping, map_rows: list[dict],
                      shards: list[dict]) -> dict:
    """Write ``shardmap.jsonl.gz`` + ``shards.json`` next to the index whose
    ``manifest`` is given; return the shards document."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    n = 0
    with gzip.GzipFile(out / SHARDMAP_FILE, "wb", mtime=0) as gz:
        for line in shardmap_lines(map_rows):
            gz.write(line)
            h.update(line)
            n += 1
    if n != int(manifest["n_series"]):
        raise ValueError(f"shard map has {n} rows, the index {manifest['n_series']}")
    doc = {"schema": SHARDS_SCHEMA, "source": manifest["source"], "repo": manifest["repo"],
           "revision": manifest["revision"], "index_sha256": manifest["index_sha256"],
           "files_sha256": manifest["files_sha256"], "n_series": n,
           "shardmap_sha256": h.hexdigest(), "codec": "zstd",
           "shards": sorted(({k: s[k] for k in SHARD_FIELDS} for s in shards),
                            key=lambda s: s["key"])}
    (out / SHARDS_FILE).write_bytes(canonical_json(doc) + b"\n")
    return doc


def load_shards(index_dir, *, index_sha256: str | None = None,
                files_sha256: str | None = None) -> dict:
    """``shards.json`` checked against the index manifest next to it (and,
    when given, the draw's ``index_sha256`` / ``files_sha256``)."""
    d = Path(index_dir)
    doc = json.loads((d / SHARDS_FILE).read_text(encoding="utf-8"))
    if doc.get("schema") != SHARDS_SCHEMA:
        raise ValueError(f"{d}: shards schema {doc.get('schema')!r}")
    m = load_manifest(d)
    for k in ("source", "repo", "revision", "index_sha256", "files_sha256", "n_series"):
        if doc.get(k) != m.get(k):
            raise ValueError(f"{d}: shards.json {k} is not the index's")
    if index_sha256 is not None and doc["index_sha256"] != index_sha256:
        raise ValueError(f"{d}: shards.json is for index {doc['index_sha256']}, not {index_sha256}")
    if files_sha256 is not None and doc["files_sha256"] != files_sha256:
        raise ValueError(f"{d}: shards.json files_sha256 is not the draw's")
    for s in doc["shards"]:
        if set(s) != set(SHARD_FIELDS) or s["key"] != shard_key(s["sha256"]):
            raise ValueError(f"{d}: bad shard record {s!r}")
    return doc


def iter_shardmap(index_dir, shards_doc: Mapping) -> Iterator[dict]:
    """Stream the shard map; at the end its bytes must hash to
    ``shardmap_sha256`` (``ValueError`` otherwise)."""
    h = hashlib.sha256()
    n = 0
    with gzip.open(Path(index_dir) / SHARDMAP_FILE, "rb") as gz:
        for line in gz:
            h.update(line)
            row = json.loads(line)
            _check_map_row(row, n)
            n += 1
            yield row
    if h.hexdigest() != shards_doc["shardmap_sha256"] or n != int(shards_doc["n_series"]):
        raise ValueError(f"{index_dir}: shard map does not hash to shardmap_sha256")


def lookup(index_dir, shards_doc: Mapping, ordinals: Iterable[int]) -> dict[int, dict]:
    """The shard map rows of ``ordinals`` (the whole map is read, so its hash
    is checked before anything is returned)."""
    want = {int(o) for o in ordinals}
    out = {int(r["ordinal"]): r for r in iter_shardmap(index_dir, shards_doc)
           if int(r["ordinal"]) in want}
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"{index_dir}: shard map lacks ordinals {missing[:5]}")
    return out


# ---------------------------------------------------------------------------
# Byte-budget, sha256-addressed blob cache (the Lium volume)
# ---------------------------------------------------------------------------

class BlobCache:
    """``<root>/blobs/<sha[:2]>/<sha>``: content-addressed, every read
    re-hashed (a corrupt blob is deleted and reported a miss), byte-budget LRU
    by last use (``put`` evicts the least recently used blobs until the new
    one fits; ``total_bytes() <= budget_bytes`` holds). The same shape as the
    W15 checkpoint cache's budget (``eval_cache_budget_bytes``), for blobs.
    A failed write (full or missing volume) never fails the caller: the bytes
    are served uncached."""

    def __init__(self, root, *, budget_bytes: int):
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) \
                or budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be a positive int; got {budget_bytes!r}")
        self.root = Path(root)
        self.budget_bytes = budget_bytes

    def _path(self, sha: str) -> Path:
        return self.root / "blobs" / sha[:2] / sha

    def _blobs(self) -> list[Path]:
        d = self.root / "blobs"
        if not d.is_dir():
            return []
        return [p for p in d.glob("*/*") if p.is_file() and len(p.name) == 64]

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._blobs())

    def get(self, sha: str) -> bytes | None:
        p = self._path(sha)
        try:
            data = p.read_bytes()
        except OSError:
            return None
        if sha256_hex(data) != sha:
            log.warning("lane_d_public_cache_corrupt %s: dropped, refetching", sha[:16])
            with contextlib.suppress(OSError):
                p.unlink()
            return None
        with contextlib.suppress(OSError):
            os.utime(p)
        return data

    def put(self, sha: str, data: bytes) -> bool:
        if sha256_hex(data) != sha:
            raise MirrorError(f"refusing to cache bytes that do not hash to {sha[:16]}")
        if len(data) > self.budget_bytes:
            return False
        try:
            self._evict(len(data), keep=sha)
            p = self._path(sha)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, p)
            return True
        except OSError as e:
            log.warning("lane_d_public_cache_write_failed %s (%s): served uncached",
                        sha[:16], e)
            return False

    def _evict(self, incoming: int, *, keep: str) -> None:
        blobs = []
        for p in self._blobs():
            with contextlib.suppress(OSError):
                st = p.stat()
                blobs.append((st.st_mtime_ns, p.name, st.st_size, p))
        total = sum(b[2] for b in blobs)
        for _mt, name, size, p in sorted(blobs):
            if total + incoming <= self.budget_bytes:
                break
            if name == keep:
                continue
            with contextlib.suppress(OSError):
                p.unlink()
                total -= size
                log.info("lane_d_public_cache_evict %s bytes=%d", name[:16], size)


# ---------------------------------------------------------------------------
# Object stores (Hippius, R2): ranged GET
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StoreSpec:
    name: str
    endpoint: str
    region: str
    bucket: str
    key_env: str              # <key_env>_ACCESS_KEY / <key_env>_SECRET_KEY (names only)


class S3RangeStore:
    """One S3-compatible store: ``get(key, offset=, length=)``. Credentials
    by env NAME; absent ⇒ an anonymous client (a public-read bucket)."""

    def __init__(self, spec: StoreSpec, *, client=None):
        self.spec = spec
        self.name = spec.name
        self._client = client

    def client(self):
        if self._client is None:
            import boto3  # type: ignore
            from botocore import UNSIGNED  # type: ignore
            from botocore.config import Config  # type: ignore

            access = os.environ.get(f"{self.spec.key_env}_ACCESS_KEY")
            secret = os.environ.get(f"{self.spec.key_env}_SECRET_KEY")
            cfg = dict(connect_timeout=10, read_timeout=60,
                       retries={"max_attempts": 3, "mode": "standard"})
            if access and secret:
                self._client = boto3.client(
                    "s3", endpoint_url=self.spec.endpoint, region_name=self.spec.region,
                    aws_access_key_id=access, aws_secret_access_key=secret,
                    config=Config(**cfg))
            else:
                self._client = boto3.client(
                    "s3", endpoint_url=self.spec.endpoint, region_name=self.spec.region,
                    config=Config(signature_version=UNSIGNED, **cfg))
        return self._client

    def get(self, key: str, *, offset: int | None = None, length: int | None = None) -> bytes:
        kw = {"Bucket": self.spec.bucket, "Key": key}
        if offset is not None:
            kw["Range"] = f"bytes={int(offset)}-{int(offset) + int(length) - 1}"
        body = self.client().get_object(**kw)["Body"]
        try:
            return body.read()
        finally:
            with contextlib.suppress(Exception):
                body.close()


def load_mirror_config(path=None) -> dict:
    """``deploy/laned/public_mirror.toml`` (names and locations only)."""
    p = Path(path or MIRROR_CONFIG_PATH)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema") != MIRROR_CONFIG_SCHEMA:
        raise ValueError(f"{p}: schema {raw.get('schema')!r}")
    stores = []
    for s in raw.get("store", []):
        if set(s) != {"name", "endpoint", "region", "bucket", "key_env"}:
            raise ValueError(f"{p}: [[store]] keys must be name, endpoint, region, bucket, "
                             f"key_env; got {sorted(s)}")
        stores.append(StoreSpec(**{k: str(v) for k, v in s.items()}))
    if not stores:
        raise ValueError(f"{p}: no [[store]]")
    budget = raw.get("cache_budget_bytes")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"{p}: cache_budget_bytes must be a positive int")
    return {"prefix": str(raw.get("prefix", DEFAULT_PREFIX)), "stores": stores,
            "cache_dir": str(raw["cache_dir"]), "cache_budget_bytes": budget}


# ---------------------------------------------------------------------------
# Reading a row group without the rest of the shard
# ---------------------------------------------------------------------------

class _SparseFile(io.RawIOBase):
    """A read-only file of ``size`` bytes of which only ``pieces``
    ({offset: bytes}) are known; any read outside them raises ``OSError``
    (never zeros), so pyarrow decodes only what was fetched and verified."""

    def __init__(self, size: int, pieces: Mapping[int, bytes]):
        super().__init__()
        self._size = int(size)
        self._pieces = sorted((int(o), bytes(b)) for o, b in pieces.items())
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, base + int(offset))
        return self._pos

    def readinto(self, buf) -> int:
        n = min(len(buf), max(0, self._size - self._pos))
        if n == 0:
            return 0
        a, b = self._pos, self._pos + n
        for off, data in self._pieces:
            if off <= a and b <= off + len(data):
                buf[:n] = data[a - off:b - off]
                self._pos = b
                return n
        raise OSError(f"lane D public mirror: bytes [{a}, {b}) were not fetched")


def decode_row(table_row: Mapping) -> np.ndarray:
    """``(C, L)`` float64 from one shard row (``values`` or ``values64``)."""
    c, n = int(table_row["n_channels"]), int(table_row["n_points"])
    v = table_row["values"] if table_row["values"] is not None else table_row["values64"]
    if v is None:
        raise ValueError("shard row holds neither values nor values64")
    a = np.asarray(v, dtype=np.float64)
    if a.size != c * n:
        raise ValueError(f"shard row holds {a.size} values, not {c}x{n}")
    return a.reshape(c, n)


def read_row_group_series(size: int, pieces: Mapping[int, bytes], row_group: int,
                          row: int) -> tuple[str, np.ndarray]:
    """``(series_id, (C, L) float64)`` of one row of one row group, decoded
    from the shard's fetched tail + row-group bytes only."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(pa.PythonFile(_SparseFile(size, pieces), mode="r"))
    t = pf.read_row_group(int(row_group))
    rec = {k: t.column(k)[int(row)].as_py() for k in ("series_id", "n_channels", "n_points")}
    for k in ("values", "values64"):
        col = t.column(k)[int(row)]
        rec[k] = None if not col.is_valid else col.values.to_numpy(zero_copy_only=False)
    return str(rec["series_id"]), decode_row(rec)


# ---------------------------------------------------------------------------
# The eval host's reader
# ---------------------------------------------------------------------------

class PublicMirror:
    """The eval host's view of one source's Hippius mirror: index files under
    ``<cache_root>/index/<source>/<index_sha256>/`` and the row-group blobs in
    the :class:`BlobCache`; ``stores`` are tried in order (Hippius, then R2)
    on a cache miss. :meth:`reader` returns the ``row -> (C, L)`` callable
    :func:`~cascade.validator.lane_d_public.read_public_windows` takes."""

    def __init__(self, *, cache: BlobCache, stores: Sequence, prefix: str = DEFAULT_PREFIX):
        if not stores:
            raise ValueError("lane D public mirror: no store configured")
        self.cache = cache
        self.stores = list(stores)
        self.prefix = prefix
        self.fetched: list[tuple[str, str, int | None, int | None]] = []

    # -- raw fetch with verification and fallback ---------------------------
    def _fetch(self, key: str, *, sha: str | None, offset: int | None = None,
               length: int | None = None) -> bytes:
        errors = []
        for st in self.stores:
            try:
                data = st.get(key, offset=offset, length=length)
            except Exception as e:  # noqa: BLE001 — the next tier
                errors.append(f"{getattr(st, 'name', st)}: {type(e).__name__}: {e}")
                continue
            self.fetched.append((getattr(st, "name", "?"), key, offset, length))
            if length is not None and len(data) != int(length):
                errors.append(f"{getattr(st, 'name', st)}: {len(data)} bytes, not {length}")
                continue
            if sha is not None and sha256_hex(data) != sha:
                log.warning("lane_d_public_tampered %s key=%s range=%s+%s: sha256 mismatch",
                            getattr(st, "name", st), key, offset, length)
                errors.append(f"{getattr(st, 'name', st)}: sha256 mismatch")
                continue
            return data
        raise MirrorError(f"lane D public mirror: {key} [{offset}+{length}] unavailable: "
                          + "; ".join(errors))

    def blob(self, key: str, *, sha: str, offset: int, length: int) -> bytes:
        got = self.cache.get(sha)
        if got is not None:
            return got
        data = self._fetch(key, sha=sha, offset=offset, length=length)
        self.cache.put(sha, data)
        return data

    # -- index files --------------------------------------------------------
    def index_dir(self, draw: Mapping) -> Path:
        """The verified index + shard map the draw was made from (fetched once
        into ``<cache_root>/index/<source>/<index_sha256>/``)."""
        source, sha = str(draw["source"]), str(draw["index_sha256"])
        d = self.cache.root / "index" / source / sha
        try:
            # Reuse: the cheap checks here; the index and shard-map bytes are
            # re-hashed by every pass that reads them (iter_index, lookup).
            return self._verify_index_dir(d, draw, full=False)
        except (OSError, ValueError, KeyError):
            pass
        tmp = d.with_name(f".{sha}.{os.getpid()}.tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        for name in INDEX_FILES:
            data = self._fetch(index_key(self.prefix, source, sha, name), sha=None)
            (tmp / name).write_bytes(data)
        self._verify_index_dir(tmp, draw, full=True)
        if d.exists():
            import shutil

            shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)
        return d

    @staticmethod
    def _verify_index_dir(d: Path, draw: Mapping, *, full: bool) -> Path:
        m = load_manifest(d)
        if m["source"] != draw["source"] or m["index_sha256"] != draw["index_sha256"] \
                or m["files_sha256"] != draw["files_sha256"]:
            raise ValueError(f"{d}: not the index the draw was made from")
        doc = load_shards(d, index_sha256=draw["index_sha256"], files_sha256=draw["files_sha256"])
        if full:
            for _ in iter_index(d):        # the index bytes hash to index_sha256
                pass
            for _ in iter_shardmap(d, doc):  # the shard map hashes to shardmap_sha256
                pass
        return d

    # -- one series -----------------------------------------------------------
    def reader(self, draw: Mapping) -> tuple[Path, Callable[[Mapping], np.ndarray]]:
        """``(index_dir, row -> (C, L) float64)`` for the drawn rows of
        ``draw``: only their row groups (and each touched shard's tail) are
        fetched."""
        d = self.index_dir(draw)
        doc = load_shards(d, index_sha256=draw["index_sha256"],
                          files_sha256=draw["files_sha256"])
        shards = {s["key"]: s for s in doc["shards"]}
        where = lookup(d, doc, (int(w["ordinal"]) for w in draw["windows"]))
        base = base_key(self.prefix, str(draw["source"]))

        def read(row: Mapping) -> np.ndarray:
            e = where[int(row["ordinal"])]
            if e["series_id"] != row["series_id"]:
                raise MirrorError(f"shard map row {e['ordinal']} is {e['series_id']!r}, the "
                                  f"index's {row['series_id']!r}")
            s = shards[e["shard"]]
            key = f"{base}/{e['shard']}"
            tail = self.blob(key, sha=s["tail_sha256"], offset=s["tail_offset"],
                             length=s["tail_length"])
            rg = self.blob(key, sha=e["sha256"], offset=e["offset"], length=e["length"])
            sid, arr = read_row_group_series(s["size"], {s["tail_offset"]: tail,
                                                         e["offset"]: rg},
                                             e["row_group"], e["row"])
            if sid != row["series_id"]:
                raise MirrorError(f"{key} rg {e['row_group']} row {e['row']} holds {sid!r}, "
                                  f"not {row['series_id']!r}")
            if arr.shape != (e["n_channels"], e["n_points"]) or arr.shape[1] != row["length"]:
                raise MirrorError(f"{row['series_id']}: shape {arr.shape} is not the map's / "
                                  f"index length {row['length']}")
            if values_sha256(arr) != e["values_sha256"]:
                raise MirrorError(f"{row['series_id']}: decoded values do not hash to the "
                                  f"shard map's values_sha256")
            return canonical_series(arr)

        return d, read


def mirror_from_config(cfg: Mapping, *, cache_dir=None, budget_bytes: int | None = None,
                       store_factory: Callable | None = None) -> PublicMirror:
    make = store_factory or S3RangeStore
    cache = BlobCache(cache_dir or cfg["cache_dir"],
                      budget_bytes=int(budget_bytes or cfg["cache_budget_bytes"]))
    return PublicMirror(cache=cache, stores=[make(s) for s in cfg["stores"]],
                        prefix=cfg["prefix"])


__all__ = ["DEFAULT_PREFIX", "INDEX_FILES", "MIRROR_CONFIG_PATH", "ROW_GROUP_TARGET_BYTES",
           "SHARDMAP_FIELDS", "SHARDMAP_FILE", "SHARDS_FILE", "SHARDS_SCHEMA",
           "SHARD_FIELDS", "SHARD_TARGET_BYTES", "BlobCache", "MirrorError", "PublicMirror",
           "S3RangeStore", "StoreSpec", "base_key", "canonical_series", "decode_row",
           "index_key", "iter_shardmap", "load_mirror_config", "load_shards", "lookup",
           "mirror_from_config", "read_row_group_series", "shard_key", "shardmap_lines",
           "values_sha256", "write_shard_index"]
