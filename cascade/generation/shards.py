"""The corpus shard format the CPU generation tier writes (DEC-CA-0053).

A materialised corpus is a directory::

    corpus.json              the index (below)
    c<child>-<n>.parquet     shards, one parquet file each

Each shard row is one series: ``channels`` (int32), ``length`` (int32) and
``values`` (the ``(C, L)`` array as little-endian float32 bytes, row-major).
Schema 2 adds nullable ``groups`` (one uint8 label byte per channel). A
values-only shard, including explicit ``groups=None``, keeps schema 1 and
the exact original parquet bytes under legacy encoding. Explicit
``float32_bss`` encoding uses schema 3: typed float32 lists, byte-stream split
and Zstandard, with the same rounded values and logical corpus digest. An
index records the highest schema used by any shard.

float32 is what the model trains on; every reader widens it to float64
exactly, so the stream digest is over the stored values, not the generator's
pre-rounding float64.

The tier runs ``n`` sandbox children, child ``i`` on seed ``child_seed(seed,
i)``. Child output is a deterministic prefix of that seed's stream, so
``corpus.json`` records each child's series count and an auditor regenerates
exactly that prefix. Training order is round-robin over children, one series
at a time, a child dropping out when it runs out — fixed by the counts alone.

``corpus.json``::

    {"schema": 1, "seed": int, "corpus_digest": hex, "n_series": int,
     "points": int, "children": [{"index", "seed", "n_series", "points",
     "shards": [{"name", "sha256", "n_series", "points"}]}], ...meta}

``corpus_digest`` is the :class:`cascade.trainer.stream._StreamDigest` framing
over every series in training order — the same digest a ``shards`` stream
reports when consumed to the end.

A REMOTE corpus (the GPU pod) is ``corpus.json`` plus ``urls.json`` — one
pre-signed download link per shard (:mod:`.transfer`). Readers fetch each
shard when its child reaches it, check its sha256, and delete it once read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

SCHEMA = 1
GROUPS_SCHEMA = 2
FLOAT32_SCHEMA = 3
INDEX_NAME = "corpus.json"
URLS_NAME = "urls.json"      # remote corpus: {shard name: pre-signed GET link}


_POOL_SET = False


class ShardError(ValueError):
    """A corpus directory or shard is malformed or fails its digest."""


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:  # pragma: no cover - env guard
        raise ShardError("corpus shards need pyarrow (pip install -e '.[generation]')") from e
    global _POOL_SET
    if not _POOL_SET:
        # The tier writes shards from worker threads. pyarrow's default
        # mimalloc pool segfaults when buffers allocated in a thread that has
        # since exited are touched again (reproduced on pyarrow 25: thread
        # writes, main thread reads). The system allocator does not.
        pa.set_memory_pool(pa.system_memory_pool())
        _POOL_SET = True
    return pa, pq


def child_seed(seed: int, index: int) -> int:
    """Child ``index``'s generation seed: a 63-bit hash of the leg seed."""
    h = hashlib.sha256(f"cascade-gen-child:{int(seed)}:{int(index)}".encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def as_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ShardError(f"series must be (L,) or (C, L); got shape {a.shape}")
    return a


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_shard(path: Path | str, series: Iterable[np.ndarray | dict], *,
                encoding: str = "legacy", compression_level: int = 3) -> dict:
    """Write ``series`` to one parquet shard; returns its index entry."""
    if encoding not in ("legacy", "float32_bss"):
        raise ShardError(f"unsupported shard encoding {encoding!r}")
    if not 1 <= compression_level <= 9:
        raise ShardError("compression_level must be 1–9")
    pa, pq = _pa()
    chans: list[int] = []
    lens: list[int] = []
    vals: list[bytes] = []
    points = 0
    groups: list[bytes | None] = []
    for s in series:
        group = None
        if isinstance(s, dict):
            extras = set(s) - {"values", "groups"}
            if extras:
                raise ShardError(f"unsupported shard fields: {sorted(extras)}")
            group = s.get("groups")
            s = s["values"]
        a = as_2d(s).astype("<f4")
        if group is not None:
            from ..interface.generator import check_groups

            check_groups(group, a.shape[0])
            group = np.asarray(group, dtype=np.uint8).tobytes()
        groups.append(group)
        chans.append(int(a.shape[0]))
        lens.append(int(a.shape[1]))
        vals.append(np.ascontiguousarray(a).tobytes())
        points += int(a.size)
    table = pa.table({
        "channels": pa.array(chans, type=pa.int32()),
        "length": pa.array(lens, type=pa.int32()),
        "values": pa.array(vals, type=pa.large_binary()),
    })
    grouped = any(g is not None for g in groups)
    if grouped:
        table = table.append_column("groups", pa.array(groups, type=pa.binary()))
        table = table.replace_schema_metadata({b"cascade_schema": b"2"})
    path = Path(path)
    if encoding == "legacy":
        pq.write_table(table, path, compression="zstd")
    else:
        # Lossless byte-stream split exposes corresponding float32 bytes to
        # Zstd. No quantization, float16 conversion or rounding beyond legacy f32.
        offsets = np.concatenate(([0], np.cumsum([len(v) // 4 for v in vals])))
        flat = np.frombuffer(b"".join(vals), dtype="<f4")
        column = pa.LargeListArray.from_arrays(pa.array(offsets, type=pa.int64()), pa.array(flat))
        table = table.set_column(2, "values", column).replace_schema_metadata(
            {b"cascade_schema": b"3", b"cascade_encoding": b"float32_bss"})
        pq.write_table(table, path, compression="zstd", compression_level=compression_level,
                       use_dictionary=False, use_byte_stream_split=["values.list.element"])
    entry = {"name": path.name, "sha256": file_sha256(path),
             "n_series": len(chans), "points": points}
    if encoding == "float32_bss":
        entry.update(schema=FLOAT32_SCHEMA, encoding=encoding)
    elif grouped:
        entry["schema"] = GROUPS_SCHEMA
    return entry


def iter_shard(path: Path | str) -> Iterator[np.ndarray | dict]:
    """Yield widened float64 values, carrying schema-2 groups when present."""
    _, pq = _pa()
    pf = pq.ParquetFile(path)
    pa, _ = _pa()
    typed = pa.types.is_large_list(pf.schema_arrow.field("values").type) or pa.types.is_list(pf.schema_arrow.field("values").type)
    grouped = "groups" in pf.schema_arrow.names
    columns = ["channels", "length", "values"] + (["groups"] if grouped else [])
    for batch in pf.iter_batches(batch_size=1024, columns=columns):
        cs = batch.column(0).to_pylist()
        ls = batch.column(1).to_pylist()
        if typed:
            values_column = batch.column(2)
            offsets = values_column.offsets.to_numpy(zero_copy_only=False)
            flat = values_column.values.to_numpy(zero_copy_only=False)
            vs = (flat[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True))
        else:
            vs = batch.column(2).to_pylist()
        gs = batch.column(3).to_pylist() if grouped else [None] * len(cs)
        for c, n, v, g in zip(cs, ls, vs, gs, strict=True):
            a = np.asarray(v, dtype="<f4") if typed else np.frombuffer(v, dtype="<f4")
            if a.size != c * n:
                raise ShardError(f"{path}: row holds {a.size} values, expected {c}×{n}")
            values = a.reshape(c, n).astype(np.float64)
            if g is None:
                yield values
            else:
                from ..interface.generator import check_groups

                labels = np.frombuffer(g, dtype=np.uint8).copy()
                try:
                    check_groups(labels, c)
                except ValueError as exc:
                    raise ShardError(f"{path}: {exc}") from exc
                yield {"values": values, "groups": labels}


def read_index(corpus_dir: Path | str) -> dict:
    p = Path(corpus_dir) / INDEX_NAME
    try:
        idx = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ShardError(f"unreadable corpus index {p}: {e}") from e
    if idx.get("schema") not in (SCHEMA, GROUPS_SCHEMA, FLOAT32_SCHEMA):
        raise ShardError(f"{p}: schema {idx.get('schema')!r}, expected 1, 2 or 3")
    return idx


def write_index(corpus_dir: Path | str, index: dict) -> Path:
    p = Path(corpus_dir) / INDEX_NAME
    p.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return p


def read_urls(corpus_dir: Path | str) -> dict[str, str]:
    """``urls.json`` — ``{shard name: pre-signed GET link}`` — when the corpus
    is remote (the GPU pod's case), else ``{}``."""
    p = Path(corpus_dir) / URLS_NAME
    if not p.is_file():
        return {}
    try:
        urls = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ShardError(f"unreadable {p}: {e}") from e
    if not isinstance(urls, dict):
        raise ShardError(f"{p} must map shard names to links")
    return urls


def verify_shards(corpus_dir: Path | str, index: dict | None = None) -> None:
    """Every listed shard exists and hashes to its recorded sha256. A shard
    that is not on disk but has a download link (``urls.json``) is verified
    when it is fetched instead."""
    d = Path(corpus_dir)
    idx = index or read_index(d)
    urls = read_urls(d)
    for child in idx["children"]:
        for sh in child["shards"]:
            p = d / sh["name"]
            if not p.is_file():
                if sh["name"] in urls:
                    continue
                raise ShardError(f"missing shard {p}")
            if file_sha256(p) != sh["sha256"]:
                raise ShardError(f"shard {p} fails its sha256")


def _fetch_shard(corpus_dir: Path, sh: dict, url: str) -> Path:
    """Download one remote shard and check it against the index's sha256."""
    from .transfer import get_file

    p = corpus_dir / sh["name"]
    try:
        get_file(url, p)
    except OSError as e:
        raise ShardError(f"shard {sh['name']} could not be downloaded: {e}") from e
    if file_sha256(p) != sh["sha256"]:
        p.unlink(missing_ok=True)
        raise ShardError(f"downloaded shard {sh['name']} fails its sha256")
    return p


def _child_iter(corpus_dir: Path, child: dict, urls: dict[str, str]) -> Iterator[np.ndarray]:
    """One child's series in order. Local shards are read in place; remote
    ones (``urls``) are downloaded one ahead on a background thread, verified,
    and deleted once consumed, so a pod holds about two shards per child."""
    import concurrent.futures as cf

    shards = child["shards"]
    on_disk = [sh["name"] not in urls or (corpus_dir / sh["name"]).is_file() for sh in shards]
    if all(on_disk):             # nothing to download: the plain path
        for sh in shards:
            yield from iter_shard(corpus_dir / sh["name"])
        return
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        def start(i: int):
            sh = shards[i]
            if (corpus_dir / sh["name"]).is_file():
                return None
            return ex.submit(_fetch_shard, corpus_dir, sh, urls[sh["name"]])

        pending = start(0) if shards else None
        for i, sh in enumerate(shards):
            if pending is not None:
                pending.result()
            pending = start(i + 1) if i + 1 < len(shards) else None
            path = corpus_dir / sh["name"]
            yield from iter_shard(path)
            if sh["name"] in urls:
                path.unlink(missing_ok=True)
    finally:
        # Generator finalizers can run inside CPython thread bookkeeping while
        # its shutdown lock is held. Joining another thread there deadlocks.
        # Cancel unstarted reads; an active bounded HTTP read finishes itself.
        ex.shutdown(wait=False, cancel_futures=True)


def iter_corpus(corpus_dir: Path | str, *, verify: bool = True) -> Iterator[np.ndarray]:
    """Yield the corpus in training order: round-robin over children, one
    series each, a child dropping out once exhausted."""
    d = Path(corpus_dir)
    idx = read_index(d)
    if verify:
        verify_shards(d, idx)
    urls = read_urls(d)
    live = [_child_iter(d, c, urls) for c in sorted(idx["children"], key=lambda c: c["index"])]
    try:
        while live:
            nxt = []
            for it in live:
                try:
                    yield next(it)
                except StopIteration:
                    continue
                nxt.append(it)
            live = nxt
    finally:
        # Close every child if a sibling fails or a training budget stops early.
        for it in live:
            it.close()


def stream_digest(series: Iterable[np.ndarray]) -> tuple[str, int, int]:
    """``(digest, n_series, points)`` under the ``_StreamDigest`` framing."""
    from ..trainer.stream import _StreamDigest

    dg = _StreamDigest()
    n = points = 0
    for s in series:
        if isinstance(s, dict):
            a = as_2d(s["values"])
            dg.update(s)
        else:
            a = as_2d(s)
            dg.update(a)
        n += 1
        points += int(a.size)
    return dg.hexdigest(), n, points


def finalize_index(corpus_dir: Path | str, *, seed: int, children: list[dict],
                   meta: dict | None = None) -> dict:
    """Compute the corpus digest over the shards in training order and write
    ``corpus.json``. ``children`` are ``{"index", "seed", "shards": [...]}``."""
    d = Path(corpus_dir)
    kids = []
    for c in sorted(children, key=lambda c: c["index"]):
        kids.append({**c,
                     "n_series": sum(s["n_series"] for s in c["shards"]),
                     "points": sum(s["points"] for s in c["shards"])})
    schema = max((sh.get("schema", SCHEMA) for c in kids for sh in c["shards"]), default=SCHEMA)
    index = {**(meta or {}), "schema": schema, "seed": int(seed), "children": kids}
    write_index(d, {**index, "corpus_digest": "", "n_series": 0, "points": 0})
    digest, n, points = stream_digest(iter_corpus(d, verify=True))
    index.update(corpus_digest=digest, n_series=n, points=points)
    write_index(d, index)
    return index
