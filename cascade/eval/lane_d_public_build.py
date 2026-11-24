"""Lane D public mirror builder: HF → canonical Parquet shards on Hippius S3
(owner 2026-11-24, issue #894; F3 / W6p). Runs on a Shadeform CPU VM, never
the validator box; ``scripts/laned_public_mirror.py build`` is the CLI.

Per dataset folder (sorted; ``--workers`` folders at once), the builder

1. downloads the folder's files at the pinned revision into a scratch dir
   (each size checked against the HF listing, each LFS file's sha256 against
   its LFS oid) — at most the folders in flight are on disk, never 908 GiB;
2. indexes the folder with the W6p code itself
   (:func:`~cascade.eval.lane_d_public_index.index_gift_dataset` /
   :func:`~cascade.eval.lane_d_public_index.index_rmisc_dataset`) and hashes
   every file, so the final index and ``files_sha256`` are byte-identical to
   :func:`~cascade.eval.lane_d_public_index.build_index` over a full mirror;
3. converts each series (exactly what ``read_series`` returns) into
   content-addressed Parquet shards (:mod:`cascade.eval.lane_d_public_mirror`),
   uploading each shard to every store as it closes and journalling it;
4. records the folder as done and deletes its scratch copy.

``finalize`` (after the last folder) assigns ordinals, writes the W6p index +
manifest and the shard map + ``shards.json``, and uploads them under
``index/<index_sha256>/``.

**Resumable and idempotent.** State lives under ``<work>/state/<source>/``:
``<dataset>.shards.jsonl`` (one line per uploaded shard) and
``<dataset>.done.json`` (the folder's index rows and file hashes). A rerun
skips done folders, and inside a partly built folder skips the series its
journalled shards hold (shard boundaries are a pure function of the series
stream, so the resumed shards are the ones an uninterrupted run writes). An
upload is skipped when the store already holds the key at the same size
(keys are content addresses).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

from . import lane_d_public_index as idx
from . import lane_d_public_mirror as pm

STATE_DIR = "state"
SCRATCH_DIR = "scratch"
SHARD_WORK_DIR = "shards"


# ---------------------------------------------------------------------------
# Series extraction (what read_series returns, one file at a time)
# ---------------------------------------------------------------------------

def _gift_target_array(col) -> np.ndarray:
    """``(C, L)`` float64 of a one-row ``target`` slice, as ``read_series``
    builds it (None -> NaN)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    vt = col.type.value_type
    nested = pa.types.is_list(vt) or pa.types.is_large_list(vt)
    if not nested:
        flat = pc.list_flatten(col)
        a = flat.cast(pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64)
        return a.reshape(1, -1)
    inner = pc.list_flatten(col)
    c = len(inner)
    if c == 0:
        return np.zeros((1, 0), dtype=np.float64)
    flat = pc.list_flatten(inner)
    a = flat.cast(pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64)
    return a.reshape(c, -1)


def iter_series(data_root, rows: Sequence[Mapping], *, source: str
                ) -> Iterator[tuple[Mapping, np.ndarray]]:
    """``(row, (C, L) float64)`` for ``rows`` (index rows of local files),
    in the given order. GiftEvalPretrain: each arrow file opened once;
    RMISC: :func:`~cascade.eval.lane_d_public_index.read_series`."""
    layout = idx.PUBLIC_SOURCES[source].layout
    if layout != "gift_arrow":
        for r in rows:
            yield r, idx.read_series(data_root, r, source=source)
        return
    open_file, table = None, None
    for r in rows:
        if r["file"] != open_file:
            table = idx._open_arrow(Path(data_root) / r["file"])
            open_file = r["file"]
        (i, _j), = r["rows"]
        yield r, _gift_target_array(table.column("target").slice(i, 1))


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def _schema():
    import pyarrow as pa

    return pa.schema([("series_id", pa.string()), ("n_channels", pa.int32()),
                      ("n_points", pa.int64()), ("values", pa.list_(pa.float32())),
                      ("values64", pa.list_(pa.float64()))])


def _row_group_ranges(path: Path) -> list[tuple[int, int]]:
    import pyarrow.parquet as pq

    md = pq.read_metadata(str(path))
    out = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        lo, hi = None, 0
        for c in range(rg.num_columns):
            cc = rg.column(c)
            start = cc.data_page_offset
            if cc.has_dictionary_page and cc.dictionary_page_offset is not None \
                    and 0 < cc.dictionary_page_offset < start:
                start = cc.dictionary_page_offset
            lo = start if lo is None else min(lo, start)
            hi = max(hi, start + cc.total_compressed_size)
        out.append((int(lo), int(hi - lo)))
    return out


def _tail(data: bytes) -> tuple[int, int]:
    if data[-4:] != b"PAR1":
        raise ValueError("not a parquet file")
    footer = int.from_bytes(data[-8:-4], "little")
    n = min(len(data), max(pm.TAIL_MIN_BYTES, footer + 8))
    return len(data) - n, n


class ShardWriter:
    """Packs series into row groups (closed at ``rg_target`` raw bytes) and
    shards (closed at ``shard_target`` bytes on disk); :meth:`close_shard`
    returns ``(shard record, map entries, path)``."""

    def __init__(self, work: Path, *, shard_target: int = pm.SHARD_TARGET_BYTES,
                 rg_target: int = pm.ROW_GROUP_TARGET_BYTES):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.shard_target = int(shard_target)
        self.rg_target = int(rg_target)
        self._writer = None
        self._path = None
        self._rg: list[dict] = []
        self._rg_bytes = 0
        self._entries: list[dict] = []
        self._n_rg = 0

    @property
    def open(self) -> bool:
        return self._writer is not None or bool(self._rg)

    def add(self, row: Mapping, arr: np.ndarray) -> bool:
        """Add one series; True when the shard is full (call close_shard)."""
        a = pm.canonical_series(arr)
        a32 = a.astype(np.float32)
        exact = np.array_equal(a32.astype(np.float64), a, equal_nan=True)
        self._rg.append({"series_id": str(row["series_id"]), "n_channels": int(a.shape[0]),
                         "n_points": int(a.shape[1]),
                         "values": a32.ravel() if exact else None,
                         "values64": None if exact else a.ravel(),
                         "_key": (row["file"], int(row["rows"][0][0])),
                         "_vsha": pm.values_sha256(a)})
        self._rg_bytes += a.size * (4 if exact else 8)
        if self._rg_bytes >= self.rg_target:
            self._flush_rg()
            return os.path.getsize(self._path) >= self.shard_target
        return False

    def _flush_rg(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self._rg:
            return
        if self._writer is None:
            self._path = self.work / f".shard.{os.getpid()}.{time.monotonic_ns()}.parquet"
            self._writer = pq.ParquetWriter(
                str(self._path), _schema(), compression="zstd",
                compression_level=pm.ZSTD_LEVEL, use_dictionary=False,
                write_statistics=False,
                use_byte_stream_split=["values.list.element", "values64.list.element"])
        schema = _schema()
        t = pa.table({f.name: pa.array([r[f.name] for r in self._rg], type=f.type)
                      for f in schema}, schema=schema)
        self._writer.write_table(t, row_group_size=len(self._rg))
        for i, r in enumerate(self._rg):
            self._entries.append({"series_id": r["series_id"], "row_group": self._n_rg, "row": i,
                                  "n_channels": r["n_channels"], "n_points": r["n_points"],
                                  "values_sha256": r["_vsha"], "file": r["_key"][0],
                                  "first_row": r["_key"][1]})
        self._n_rg += 1
        self._rg, self._rg_bytes = [], 0

    def close_shard(self) -> tuple[dict, list[dict], Path] | None:
        self._flush_rg()
        if self._writer is None:
            return None
        self._writer.close()
        path, entries, n_rg = self._path, self._entries, self._n_rg
        self._writer, self._path, self._entries, self._n_rg = None, None, [], 0
        data = path.read_bytes()
        sha = pm.sha256_hex(data)
        ranges = _row_group_ranges(path)
        if len(ranges) != n_rg:
            raise RuntimeError(f"{path}: {len(ranges)} row groups, wrote {n_rg}")
        t_off, t_len = _tail(data)
        final = self.work / f"{sha}.parquet"
        os.replace(path, final)
        for e in entries:
            off, ln = ranges[e["row_group"]]
            e.update(shard=pm.shard_key(sha), offset=off, length=ln,
                     sha256=pm.sha256_hex(data[off:off + ln]))
        rec = {"key": pm.shard_key(sha), "size": len(data), "sha256": sha, "tail_offset": t_off,
               "tail_length": t_len, "tail_sha256": pm.sha256_hex(data[t_off:]),
               "n_row_groups": n_rg, "n_series": len(entries)}
        return rec, entries, final


# ---------------------------------------------------------------------------
# Stores (writes need credentials)
# ---------------------------------------------------------------------------

class S3PutStore:
    """``head(key) -> size | None``, ``put_file(key, path)``. Keys by env
    NAME (``<key_env>_ACCESS_KEY`` / ``_SECRET_KEY``); values never printed."""

    def __init__(self, spec: pm.StoreSpec):
        self.spec = spec
        self.name = spec.name
        self._client = None

    def client(self):
        if self._client is None:
            from ..shared.hippius import S3Config, _s3_client

            self._client = _s3_client(S3Config(
                endpoint=self.spec.endpoint, region=self.spec.region, bucket=self.spec.bucket,
                access_key_env=f"{self.spec.key_env}_ACCESS_KEY",
                secret_key_env=f"{self.spec.key_env}_SECRET_KEY"))
        return self._client

    def head(self, key: str) -> int | None:
        try:
            return int(self.client().head_object(Bucket=self.spec.bucket,
                                                 Key=key).get("ContentLength", -1))
        except Exception:  # noqa: BLE001 — absent
            return None

    def put_file(self, key: str, path: Path) -> None:
        self.client().upload_file(str(path), self.spec.bucket, key)


def upload(stores: Sequence, key: str, path: Path) -> list[str]:
    """Put ``path`` at ``key`` on every store (skip when the same size is
    already there); returns the store names written."""
    size = Path(path).stat().st_size
    wrote = []
    for st in stores:
        if st.head(key) == size:
            continue
        st.put_file(key, Path(path))
        if st.head(key) != size:
            raise RuntimeError(f"{getattr(st, 'name', st)}: {key} did not land at {size} bytes")
        wrote.append(getattr(st, "name", "?"))
    return wrote


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def dataset_files(listing: Sequence[Mapping], dataset: str) -> list[dict]:
    """The folder's direct files (what ``build_index`` hashes): ``<ds>/<name>``."""
    return sorted(({"path": e["path"], "size": int(e.get("size") or 0),
                    "lfs": (e.get("lfs") or {}).get("oid")}
                   for e in listing if e.get("type", "file") == "file"
                   and e["path"].count("/") == 1 and e["path"].split("/", 1)[0] == dataset),
                  key=lambda f: f["path"])


class HFFetcher:
    """Download one file of the pinned revision to ``dest`` (huggingface_hub;
    picklable, so ``--workers`` processes can carry it)."""

    def __init__(self, source: str):
        self.source = source

    def __call__(self, path: str, dest: Path) -> None:
        from huggingface_hub import hf_hub_download

        spec = idx.PUBLIC_SOURCES[self.source]
        tmp = dest.parent / ".hf"
        got = hf_hub_download(repo_id=spec.repo, repo_type="dataset", revision=spec.revision,
                              filename=path, local_dir=str(tmp))
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, dest)


def hf_fetcher(source: str) -> Callable[[str, Path], None]:
    return HFFetcher(source)


class Builder:
    def __init__(self, *, source: str, listing: Sequence[Mapping], work_dir, stores: Sequence,
                 fetch: Callable[[str, Path], None], prefix: str = pm.DEFAULT_PREFIX,
                 shard_target: int = pm.SHARD_TARGET_BYTES,
                 rg_target: int = pm.ROW_GROUP_TARGET_BYTES, echo: Callable = print):
        self.source = source
        self.listing = list(listing)
        self.work = Path(work_dir)
        self.stores = list(stores)
        self.fetch = fetch
        self.prefix = prefix
        self.shard_target = shard_target
        self.rg_target = rg_target
        self.echo = echo
        self.state = self.work / STATE_DIR / source
        self.state.mkdir(parents=True, exist_ok=True)

    # -- state ----------------------------------------------------------------
    def _done_path(self, ds: str) -> Path:
        return self.state / f"{ds}.done.json"

    def _journal_path(self, ds: str) -> Path:
        return self.state / f"{ds}.shards.jsonl"

    def journal(self, ds: str) -> list[dict]:
        p = self._journal_path(ds)
        if not p.is_file():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                with contextlib.suppress(json.JSONDecodeError):  # a torn last line
                    out.append(json.loads(line))
        return out

    def done(self, ds: str) -> dict | None:
        p = self._done_path(ds)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    # -- one folder -----------------------------------------------------------
    def _download(self, ds: str, root: Path) -> list[dict]:
        recs = []
        for f in dataset_files(self.listing, ds):
            dest = root / f["path"]
            if not (dest.is_file() and dest.stat().st_size == f["size"]):
                self.fetch(f["path"], dest)
            if dest.stat().st_size != f["size"]:
                raise RuntimeError(f"{f['path']}: {dest.stat().st_size} bytes, the listing "
                                   f"says {f['size']}")
            sha = idx.file_sha256(dest)
            if f["lfs"] and sha != f["lfs"]:
                raise RuntimeError(f"{f['path']}: sha256 {sha} is not the LFS oid {f['lfs']}")
            recs.append({"path": f["path"], "size": f["size"], "sha256": sha})
        if not recs:
            raise RuntimeError(f"{ds}: no files in the listing")
        return recs

    def build_dataset(self, ds: str) -> dict:
        got = self.done(ds)
        if got is not None:
            self.echo(f"SKIP {self.source} {ds} (done)")
            return got
        t0 = time.monotonic()
        root = self.work / SCRATCH_DIR / self.source
        files = self._download(ds, root)
        fn = (idx.index_gift_dataset if idx.PUBLIC_SOURCES[self.source].layout == "gift_arrow"
              else idx.index_rmisc_dataset)
        rows, _ = fn(root, ds)
        rows = sorted(rows, key=lambda r: (r["file"], r["rows"][0][0]))
        journal = self.journal(ds)
        covered = {(e["file"], e["first_row"]) for j in journal for e in j["entries"]}
        keys = [(r["file"], r["rows"][0][0]) for r in rows]
        k = 0
        while k < len(keys) and keys[k] in covered:
            k += 1
        if covered - set(keys[:k]):
            raise RuntimeError(f"{ds}: the shard journal is not a prefix of the series stream; "
                               f"delete {self._journal_path(ds)} to rebuild the folder")
        if k:
            self.echo(f"RESUME {self.source} {ds}: {k}/{len(rows)} series already in "
                      f"{len(journal)} shard(s)")
        w = ShardWriter(self.work / SHARD_WORK_DIR / self.source, shard_target=self.shard_target,
                        rg_target=self.rg_target)
        for row, arr in iter_series(root, rows[k:], source=self.source):
            if w.add(row, arr):
                self._commit(ds, w.close_shard())
        out = w.close_shard()
        if out is not None:
            self._commit(ds, out)
        journal = self.journal(ds)
        raw = sum(f["size"] for f in files)
        packed = sum(j["shard"]["size"] for j in journal)
        doc = {"dataset": ds, "rows": rows, "files": files, "raw_bytes": raw,
               "shard_bytes": packed, "n_shards": len(journal),
               "seconds": round(time.monotonic() - t0, 1)}
        tmp = self._done_path(ds).with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._done_path(ds))
        shutil.rmtree(root / ds, ignore_errors=True)
        self.echo(f"DONE {self.source} {ds} series={len(rows)} raw={raw} shards={len(journal)} "
                  f"shard_bytes={packed} ratio={packed / max(raw, 1):.3f} secs={doc['seconds']}")
        return doc

    def _commit(self, ds: str, out) -> None:
        rec, entries, path = out
        key = f"{pm.base_key(self.prefix, self.source)}/{rec['key']}"
        wrote = upload(self.stores, key, path)
        with open(self._journal_path(ds), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"shard": rec, "entries": entries}, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        path.unlink(missing_ok=True)
        self.echo(f"SHARD {self.source} {ds} {rec['key']} size={rec['size']} "
                  f"series={rec['n_series']} uploaded={','.join(wrote) or 'skipped'}")

    # -- all folders, then the index ------------------------------------------
    def finalize(self, datasets: Sequence[str], index_dir) -> dict:
        docs = []
        for ds in sorted(datasets):
            d = self.done(ds)
            if d is None:
                raise RuntimeError(f"{ds} is not built yet; run build again")
            docs.append(d)
        rows = idx.assign_ordinals([r for d in docs for r in d["rows"]])
        files = [f for d in docs for f in d["files"]]
        out = Path(index_dir)
        manifest = idx.write_index(out, source=self.source, rows=rows, files=files)
        where = {}
        shards = {}
        for ds in sorted(datasets):
            for j in self.journal(ds):
                shards[j["shard"]["key"]] = j["shard"]
                for e in j["entries"]:
                    where[(e["file"], e["first_row"])] = e
        map_rows = []
        for r in rows:
            e = where.get((r["file"], r["rows"][0][0]))
            if e is None or e["series_id"] != r["series_id"]:
                raise RuntimeError(f"series {r['series_id']} has no shard")
            map_rows.append({"ordinal": r["ordinal"], **{k: e[k] for k in pm.SHARDMAP_FIELDS
                                                         if k != "ordinal"}})
        doc = pm.write_shard_index(out, manifest=manifest, map_rows=map_rows,
                                   shards=list(shards.values()))
        for name in pm.INDEX_FILES:
            upload(self.stores, pm.index_key(self.prefix, self.source, manifest["index_sha256"],
                                             name), out / name)
        raw = sum(d["raw_bytes"] for d in docs)
        packed = sum(s["size"] for s in shards.values())
        self.echo(f"INDEX {self.source} series={manifest['n_series']} "
                  f"index_sha256={manifest['index_sha256']} "
                  f"files_sha256={manifest['files_sha256']} "
                  f"shardmap_sha256={doc['shardmap_sha256']} shards={len(shards)} "
                  f"raw={raw} shard_bytes={packed} ratio={packed / max(raw, 1):.3f}")
        return {"manifest": manifest, "shards": doc, "raw_bytes": raw, "shard_bytes": packed}


def _worker(kw: dict, ds: str) -> dict:
    b = Builder(**kw)
    return b.build_dataset(ds)


def build_all(builder_kw: Mapping, datasets: Sequence[str], *, workers: int = 1) -> list[dict]:
    """Build every folder (``workers`` processes at once; 1 = in-process)."""
    if workers <= 1:
        b = Builder(**builder_kw)
        return [b.build_dataset(ds) for ds in sorted(datasets)]
    from concurrent.futures import ProcessPoolExecutor

    # Biggest folders first so the long ones never start last.
    sizes = {ds: sum(f["size"] for f in dataset_files(builder_kw["listing"], ds))
             for ds in datasets}
    order = sorted(datasets, key=lambda d: (-sizes[d], d))
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        return list(ex.map(_worker, [dict(builder_kw)] * len(order), order))


def estimate(done_docs: Sequence[Mapping], total_raw: int) -> dict:
    """Extrapolate the compressed size and the conversion wall from sample
    folders' ``done`` records to ``total_raw`` bytes."""
    raw = sum(int(d["raw_bytes"]) for d in done_docs)
    packed = sum(int(d["shard_bytes"]) for d in done_docs)
    secs = sum(float(d["seconds"]) for d in done_docs)
    ratio = packed / max(raw, 1)
    return {"sample_raw": raw, "sample_shards": packed, "ratio": round(ratio, 4),
            "projected_shard_bytes": int(total_raw * ratio),
            "projected_single_worker_hours": round(total_raw / max(raw, 1) * secs / 3600, 2)}


__all__ = ["Builder", "S3PutStore", "ShardWriter", "build_all", "dataset_files", "estimate",
           "hf_fetcher", "iter_series", "upload"]
