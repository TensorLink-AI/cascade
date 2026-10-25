"""The corpus dataset on Hippius S3 (DEC-CA-0053).

Layout under ``[generation] storage_prefix`` (default ``corpora/``)::

    corpora/round-<id>/<hotkey>/corpus.json     the shard index (shards.py)
    corpora/round-<id>/<hotkey>/c<i>-<n>.parquet
    corpora/round-<id>/<hotkey>/meta.json       tags: round, era, uid, hotkey,
                                                generator ref, seed, digest,
                                                counts, length/channel
                                                histograms, CPU, provider
    corpora/index.json                          one row per leg; scores filled
                                                after settlement

Keyed by hotkey; the UID is only a tag (UIDs recycle). Shards and meta are
written once; scores land only in the index (:func:`record_scores`). The
index is read-modify-written by its single writer, the orchestrator.

Every corpus is operator-private. :func:`publish_king_corpus` copies a crowned
king's corpus to the public store (all shards, or the recipe plus sample
shards); :func:`expire_corpora` deletes non-king corpora past retention.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from collections import Counter
from pathlib import Path

from ..shared.config import GenerationConfig
from ..shared.hippius import ObjectNotFound
from .shards import INDEX_NAME, ShardError, read_index, verify_shards

INDEX_KEY = "index.json"
META_NAME = "meta.json"
SCORE_FIELDS = ("crps_geomean", "mase_geomean", "verdict", "crowned", "bench")


# The orchestrator writes the index from several threads (rolling legs, lane-B
# legs): every read-modify-write of corpora/index.json holds this lock.
_INDEX_LOCK = threading.RLock()


def _index_locked(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with _INDEX_LOCK:
            return fn(*a, **kw)
    return wrapper


def leg_prefix(round_id: str, hotkey: str, prefix: str = "corpora/") -> str:
    if not hotkey or "/" in hotkey:
        raise ValueError(f"bad hotkey {hotkey!r}")
    if not str(round_id) or "/" in str(round_id):
        raise ValueError(f"bad round id {round_id!r}")
    return f"{prefix}round-{round_id}/{hotkey}/"


def index_key(prefix: str = "corpora/") -> str:
    return f"{prefix}{INDEX_KEY}"


# ── store helpers (S3Store has no list/delete; fakes may provide them) ──────


def _put_file(store, key: str, path: Path) -> None:
    if hasattr(store, "put_file"):
        store.put_file(key, path)
        return
    if hasattr(store, "cfg") and hasattr(store, "client"):
        store._ensure_bucket()
        store.client().upload_file(str(path), store.cfg.bucket, key)
        return
    store.put_bytes(key, path.read_bytes())


def _list_keys(store, prefix: str) -> list[str]:
    if hasattr(store, "list_keys"):
        return list(store.list_keys(prefix))
    keys: list[str] = []
    pages = store.client().get_paginator("list_objects_v2").paginate(
        Bucket=store.cfg.bucket, Prefix=prefix)
    for page in pages:
        keys.extend(o["Key"] for o in page.get("Contents", []) or [])
    return keys


def _delete_key(store, key: str) -> None:
    if hasattr(store, "delete"):
        store.delete(key)
        return
    store.client().delete_object(Bucket=store.cfg.bucket, Key=key)


def _put_public(store, key: str, data: bytes, content_type: str) -> None:
    try:
        store.put_bytes(key, data, content_type=content_type, acl="public-read")
    except TypeError:
        store.put_bytes(key, data)


# ── the index ───────────────────────────────────────────────────────────────


def read_dataset_index(store, prefix: str = "corpora/") -> list[dict]:
    try:
        rows = json.loads(store.get_text(index_key(prefix)))
    except ObjectNotFound:
        return []
    if not isinstance(rows, list):
        raise ShardError(f"{index_key(prefix)} is not a list")
    return rows


def _write_dataset_index(store, rows: list[dict], prefix: str) -> None:
    rows = sorted(rows, key=lambda r: (str(r["round_id"]), r["hotkey"]))
    store.put_text(index_key(prefix), json.dumps(rows, indent=1, sort_keys=True),
                   content_type="application/json")


def _row(rows: list[dict], round_id: str, hotkey: str) -> dict | None:
    for r in rows:
        if str(r["round_id"]) == str(round_id) and r["hotkey"] == hotkey:
            return r
    return None


# ── tags ────────────────────────────────────────────────────────────────────


def _pow2_bucket(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def shape_histograms(corpus_dir: Path | str, index: dict | None = None) -> tuple[dict, dict]:
    """``(length histogram over pow2 buckets, channel histogram)`` read from
    the shards' ``length``/``channels`` columns only (no values)."""
    import pyarrow.parquet as pq

    d = Path(corpus_dir)
    idx = index or read_index(d)
    lengths: Counter = Counter()
    chans: Counter = Counter()
    for child in idx["children"]:
        for sh in child["shards"]:
            t = pq.read_table(d / sh["name"], columns=["channels", "length"])
            for c, n in zip(t.column(0).to_pylist(), t.column(1).to_pylist(), strict=True):
                lengths[_pow2_bucket(n)] += 1
                chans[int(c)] += 1
    return ({str(k): v for k, v in sorted(lengths.items())},
            {str(k): v for k, v in sorted(chans.items())})


def build_meta(corpus_dir: Path | str, *, round_id: str, hotkey: str, uid: int | None,
               era: int | None, provider: str, shape: str) -> dict:
    idx = read_index(corpus_dir)
    return meta_from_index(idx, round_id=round_id, hotkey=hotkey, uid=uid, era=era,
                           provider=provider, shape=shape,
                           histograms=None if "length_histogram" in idx
                           else shape_histograms(corpus_dir, idx))


def meta_from_index(idx: dict, *, round_id: str, hotkey: str, uid: int | None,
                    era: int | None, provider: str, shape: str,
                    histograms: tuple[dict, dict] | None = None) -> dict:
    """A leg's ``meta.json`` from its ``corpus.json`` alone (the tier records
    the shape histograms while generating, so no shard is read)."""
    lengths, chans = histograms or (idx.get("length_histogram", {}),
                                    idx.get("channels_histogram", {}))
    return {
        "round_id": str(round_id), "era": era, "uid": uid, "hotkey": hotkey,
        "generator_ref": idx.get("generator_ref", ""), "seed": idx["seed"],
        "corpus_digest": idx["corpus_digest"], "n_series": idx["n_series"],
        "points": idx["points"], "length_histogram": lengths,
        "channels_histogram": chans, "cpu_seconds": idx.get("cpu_seconds"),
        "wall_seconds": idx.get("wall_seconds"), "cpu_model": idx.get("cpu_model", ""),
        "vcpus": idx.get("vcpus"), "stop_reason": idx.get("stop_reason", ""),
        "provider": provider, "shape": shape,
    }


# ── operations ──────────────────────────────────────────────────────────────


def upload_corpus(store, corpus_dir: Path | str, *, round_id: str, hotkey: str,
                  uid: int | None, era: int | None, provider: str, shape: str,
                  prefix: str = "corpora/", now: float | None = None) -> dict:
    """Verify and upload one leg's corpus, write its meta, add its index row.
    Returns the row."""
    d = Path(corpus_dir)
    idx = read_index(d)
    verify_shards(d, idx)
    base = leg_prefix(round_id, hotkey, prefix)
    for child in idx["children"]:
        for sh in child["shards"]:
            _put_file(store, base + sh["name"], d / sh["name"])
    meta = build_meta(d, round_id=round_id, hotkey=hotkey, uid=uid, era=era,
                      provider=provider, shape=shape)
    store.put_text(base + META_NAME, json.dumps(meta, indent=1, sort_keys=True),
                   content_type="application/json")
    # The index last: a leg is listed only once every shard is in place.
    store.put_text(base + INDEX_NAME, (d / INDEX_NAME).read_text(encoding="utf-8"),
                   content_type="application/json")
    return _add_row(store, meta, round_id=round_id, hotkey=hotkey, uid=uid, era=era,
                    provider=provider, shape=shape, prefix=prefix, now=now)


def register_corpus(store, index: dict, *, round_id: str, hotkey: str,
                    uid: int | None, era: int | None, provider: str, shape: str,
                    prefix: str = "corpora/", now: float | None = None) -> dict:
    """Tag a corpus the tier uploaded itself (pre-signed links): write its
    ``meta.json`` and index row from ``corpus.json`` alone. Returns the row."""
    base = leg_prefix(round_id, hotkey, prefix)
    meta = meta_from_index(index, round_id=round_id, hotkey=hotkey, uid=uid, era=era,
                           provider=provider, shape=shape)
    store.put_text(base + META_NAME, json.dumps(meta, indent=1, sort_keys=True),
                   content_type="application/json")
    return _add_row(store, meta, round_id=round_id, hotkey=hotkey, uid=uid, era=era,
                    provider=provider, shape=shape, prefix=prefix, now=now)


@_index_locked
def _add_row(store, meta: dict, *, round_id: str, hotkey: str, uid: int | None,
             era: int | None, provider: str, shape: str, prefix: str,
             now: float | None) -> dict:
    rows = read_dataset_index(store, prefix)
    row = {
        "round_id": str(round_id), "hotkey": hotkey, "uid": uid, "era": era,
        "generator_ref": meta["generator_ref"], "corpus_digest": meta["corpus_digest"],
        "n_series": meta["n_series"], "points": meta["points"], "provider": provider,
        "shape": shape, "uploaded_at": float(time.time() if now is None else now),
        "crps_geomean": None, "mase_geomean": None, "verdict": None,
        "crowned": False, "bench": None, "published": None, "expired": False,
    }
    rows = [r for r in rows if not (str(r["round_id"]) == str(round_id)
                                    and r["hotkey"] == hotkey)] + [row]
    _write_dataset_index(store, rows, prefix)
    return row


@_index_locked
def record_scores(store, round_id: str, hotkey: str, *, prefix: str = "corpora/",
                  **scores) -> dict:
    """Fill a leg's score columns after settlement. Shards are untouched."""
    unknown = set(scores) - set(SCORE_FIELDS)
    if unknown:
        raise ValueError(f"unknown score fields {sorted(unknown)}; expected {SCORE_FIELDS}")
    rows = read_dataset_index(store, prefix)
    row = _row(rows, round_id, hotkey)
    if row is None:
        raise KeyError(f"no corpus for round {round_id} hotkey {hotkey}")
    row.update(scores)
    _write_dataset_index(store, rows, prefix)
    return row


def _sample_shard_names(idx: dict, n: int) -> list[str]:
    """The first ``n`` shards in consumption order: shard position first,
    then child index (round-robin training reads every child's first shard
    before any second shard)."""
    ordered = sorted(
        ((pos, c["index"], sh["name"])
         for c in idx["children"] for pos, sh in enumerate(c["shards"])),
    )
    return [name for _, _, name in ordered[:n]]


@_index_locked
def publish_king_corpus(src_store, public_store, round_id: str, hotkey: str,
                        gcfg: GenerationConfig) -> dict:
    """Publish a crowned king's corpus: ``full`` copies every shard;
    ``recipe`` copies meta + corpus.json + the first ``sample_shards``
    shards (``stream_cpu`` regenerates the rest from ref + seed)."""
    prefix = gcfg.storage_prefix
    base = leg_prefix(round_id, hotkey, prefix)
    idx = json.loads(src_store.get_text(base + INDEX_NAME))
    if gcfg.king_publish == "full":
        names = [sh["name"] for c in idx["children"] for sh in c["shards"]]
    else:
        names = _sample_shard_names(idx, gcfg.sample_shards)
    for name in names:
        _put_public(public_store, base + name, src_store.get_bytes(base + name),
                    "application/octet-stream")
    for name in (META_NAME, INDEX_NAME):
        _put_public(public_store, base + name, src_store.get_bytes(base + name),
                    "application/json")
    rows = read_dataset_index(src_store, prefix)
    row = _row(rows, round_id, hotkey)
    if row is not None:
        row.update(crowned=True, published=gcfg.king_publish)
        _write_dataset_index(src_store, rows, prefix)
    return {"round_id": str(round_id), "hotkey": hotkey, "mode": gcfg.king_publish,
            "shards": names}


@_index_locked
def expire_corpora(store, now: float, gcfg: GenerationConfig,
                   keep: set[tuple[str, str]] = frozenset()) -> list[tuple[str, str]]:
    """Delete every non-king corpus older than ``retention_days``. Crowned
    rows and ``keep`` (``(round_id, hotkey)``) survive; expired rows stay in
    the index (scores kept) with ``expired = true``."""
    prefix = gcfg.storage_prefix
    cutoff = float(now) - gcfg.retention_days * 86400
    rows = read_dataset_index(store, prefix)
    gone: list[tuple[str, str]] = []
    for r in rows:
        k = (str(r["round_id"]), r["hotkey"])
        if r.get("expired") or r.get("crowned") or k in keep:
            continue
        if float(r.get("uploaded_at") or 0) >= cutoff:
            continue
        for key in _list_keys(store, leg_prefix(*k, prefix)):
            _delete_key(store, key)
        r["expired"] = True
        gone.append(k)
    if gone:
        _write_dataset_index(store, rows, prefix)
    return gone
