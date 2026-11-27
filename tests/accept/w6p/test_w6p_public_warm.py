"""Lane D public half throughput (owner 2026-11-27, "fix all those things").

One settlement draws 10,000 public windows and the eval host fetched every
row group serially (measured 31-45 blobs/min against a ~588 GiB mirror): the
3600 s eval timeout could never be met. This file pins the eval-host side:

* :meth:`PublicMirror.warm` fills the volume cache with a source's whole
  mirror ahead of time: the shard map walked shard by shard, adjacent row
  groups coalesced into few ranged GETs, every blob verified, a byte budget
  respected WITHOUT eviction, resumable (a rerun fetches only what is
  missing);
* the ``warm-public`` host CLI (``python -m cascade.eval.lane_d_forecasts``);
* the :class:`BlobCache` no longer scans the whole blob tree on every put
  (a warm cache holds ~10^6 blobs) and is safe under the read pool;
* concurrent readers of one shard fetch its tail once (single flight);
* every upload is ``public-read`` (anonymous GETs were 403, so external
  validators could not reproduce the public half) and ``fix-acl`` sweeps the
  objects uploaded before.

No pyarrow: the synthetic mirror is opaque bytes (warm never decodes), the
index is W7a's index-only fixture. Nothing touches a network.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import pytest

SRC = "gift_eval_pretrain"


def _pm():
    from cascade.eval import lane_d_public_mirror as pm

    return pm


def _need_warm():
    pm = _pm()
    assert hasattr(pm.PublicMirror, "warm") and hasattr(pm.PublicMirror, "warm_plan"), (
        "2026-11-27: PublicMirror has no warm sweep (the public half cannot be pre-fetched)")
    return pm


class Store:
    """An in-memory S3 store with ranged ``get`` (every call recorded);
    ``corrupt`` = keys whose ranged reads come back with a flipped byte,
    ``fail_first`` = this many GETs raise before the store works."""

    def __init__(self, name, objects, *, delay=0.0):
        self.name = name
        self.objects = objects
        self.gets: list[tuple[str, int | None, int | None]] = []
        self.delay = delay
        self.corrupt: set[str] = set()
        self.fail_first = 0
        self._lock = threading.Lock()

    def get(self, key, *, offset=None, length=None):
        with self._lock:
            self.gets.append((key, offset, length))
            if self.fail_first > 0:
                self.fail_first -= 1
                raise ConnectionError(f"{self.name}: flaky")
        if self.delay:
            time.sleep(self.delay)
        data = self.objects[key]
        data = data if offset is None else data[offset:offset + length]
        if key in self.corrupt and data:
            data = bytes([data[0] ^ 0xFF]) + data[1:]
        return data


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def synthetic_mirror(tmp: Path, *, rg_bytes=3000, rgs_per_shard=4, per_rg=2) -> dict:
    """The gift_eval_pretrain index-only fixture (22 series) plus opaque
    shards on an object dict: ``{objects, index_sha256, blobs, bytes}``."""
    import numpy as np

    from cascade.eval.lane_d_public_index import (
        INDEX_FILE,
        MANIFEST_FILE,
        iter_index,
        load_manifest,
    )
    from tests.accept.w7a import pub

    pm = _pm()
    ix = pub.index_only(tmp / "ix")[SRC]
    manifest = load_manifest(ix)
    rows = list(iter_index(ix))
    rng = np.random.default_rng(7)
    shards, map_rows, objects = [], [], {}
    base = pm.base_key(pm.DEFAULT_PREFIX, SRC)
    blobs = nbytes = 0
    i = 0
    while i < len(rows):
        body = b""
        rgs = []
        for _ in range(rgs_per_shard):
            if i >= len(rows):
                break
            rg = rng.integers(0, 256, rg_bytes, dtype=np.uint8).tobytes()
            rgs.append((len(body), rg, rows[i:i + per_rg]))
            body += rg
            i += per_rg
        tail = rng.integers(0, 256, 200, dtype=np.uint8).tobytes()
        data = body + tail
        sha = _sha(data)
        key = pm.shard_key(sha)
        for g, (off, rg, members) in enumerate(rgs):
            for r, row in enumerate(members):
                map_rows.append({"ordinal": row["ordinal"], "series_id": row["series_id"],
                                 "shard": key, "row_group": g, "row": r, "offset": off,
                                 "length": len(rg), "sha256": _sha(rg), "n_channels": 1,
                                 "n_points": row["length"], "values_sha256": "0" * 64})
            blobs += 1
            nbytes += len(rg)
        shards.append({"key": key, "size": len(data), "sha256": sha, "tail_offset": len(body),
                       "tail_length": len(tail), "tail_sha256": _sha(tail),
                       "n_row_groups": len(rgs), "n_series": sum(len(m) for *_, m in rgs)})
        blobs += 1
        nbytes += len(tail)
        objects[f"{base}/{key}"] = data
    doc = pm.write_shard_index(ix, manifest=manifest, map_rows=map_rows, shards=shards)
    for name in (INDEX_FILE, MANIFEST_FILE, pm.SHARDS_FILE, pm.SHARDMAP_FILE):
        objects[pm.index_key(pm.DEFAULT_PREFIX, SRC, manifest["index_sha256"], name)] = \
            (ix / name).read_bytes()
    return {"objects": objects, "index_sha256": manifest["index_sha256"], "blobs": blobs,
            "bytes": nbytes, "shards": doc["shards"], "map": map_rows}


def _mirror(tmp, objects, *, budget=10**9, stores=None, **kw):
    pm = _pm()
    stores = stores or [Store("hippius", objects)]
    return pm.PublicMirror(cache=pm.BlobCache(tmp / "cache", budget_bytes=budget),
                           stores=stores, **kw), stores


def _span_gets(store, base_prefix="lane_d_public"):
    return [g for g in store.gets if "/shards/" in g[0]]


# ── the warm sweep ──────────────────────────────────────────────────────────


def test_warm_fills_every_blob_with_few_ranged_gets_and_a_rerun_fetches_nothing(tmp_path):
    pm = _need_warm()
    m = synthetic_mirror(tmp_path)
    mirror, (st,) = _mirror(tmp_path, m["objects"])
    got = mirror.warm(SRC, m["index_sha256"], workers=4)
    assert got["failed_spans"] == 0 and got["fits"]
    assert got["fetched_blobs"] == m["blobs"] and got["fetched_bytes"] == m["bytes"]
    for r in m["map"]:
        assert mirror.cache.get(r["sha256"]) is not None, "every row group is cached, verified"
    for s in m["shards"]:
        assert mirror.cache.get(s["tail_sha256"]) is not None, "every shard tail is cached"
    spans = _span_gets(st)
    assert len(spans) == len(m["shards"]) < m["blobs"], (
        "adjacent row groups + the tail of one shard are ONE ranged GET, not one per blob")
    n = len(st.gets)
    again = mirror.warm(SRC, m["index_sha256"], workers=4)
    assert again["fetched_blobs"] == 0 and again["cached"] == m["blobs"]
    assert len(_span_gets(st)) == len(spans) and len(st.gets) == n, \
        "resumable: a warm cache plans nothing and fetches nothing"
    assert pm.WARM_SPAN_BYTES >= 2**20


def test_warm_spans_are_capped_so_memory_stays_bounded(tmp_path):
    _need_warm()
    m = synthetic_mirror(tmp_path)
    mirror, (st,) = _mirror(tmp_path, m["objects"])
    plan = mirror.warm_plan(SRC, m["index_sha256"], span_bytes=6500)
    assert all(s["end"] - s["offset"] <= 6500 for s in plan["spans"])
    got = mirror.warm(SRC, m["index_sha256"], workers=3, span_bytes=6500)
    assert got["fetched_blobs"] == m["blobs"]
    assert max(g[2] for g in _span_gets(st)) <= 6500


def test_warm_respects_the_fill_budget_and_never_evicts(tmp_path):
    _need_warm()
    m = synthetic_mirror(tmp_path)
    pm = _pm()
    budget = m["bytes"] // 2
    mirror, _ = _mirror(tmp_path, m["objects"], budget=m["bytes"] * 2)
    keep = b"pre-existing blob the warm must not evict"
    mirror.cache.put(_sha(keep), keep)
    got = mirror.warm(SRC, m["index_sha256"], workers=2, budget_bytes=budget)
    assert not got["fits"], "the plan reports the source does not fit the fill budget"
    assert 0 < mirror.cache.total_bytes() <= budget
    assert mirror.cache.get(_sha(keep)) == keep, "warm never evicts"
    small = pm.BlobCache(tmp_path / "small", budget_bytes=4000)
    m2, _ = _mirror(tmp_path / "x", m["objects"])
    m2.cache = small
    m2.warm(SRC, m["index_sha256"], workers=2)
    assert small.total_bytes() <= 4000, "the cache's own budget caps any fill"


def test_warm_verifies_every_blob_and_falls_back_per_blob(tmp_path):
    _need_warm()
    m = synthetic_mirror(tmp_path)
    bad, good = Store("hippius", m["objects"]), Store("r2", m["objects"])
    key0 = next(k for k in m["objects"] if "/shards/" in k)
    bad.corrupt.add(key0)
    mirror, _ = _mirror(tmp_path, m["objects"], stores=[bad, good])
    got = mirror.warm(SRC, m["index_sha256"], workers=2)
    assert got["failed_spans"] == 0 and got["fetched_blobs"] == m["blobs"]
    for r in m["map"]:
        data = mirror.cache.get(r["sha256"])
        assert data is not None and _sha(data) == r["sha256"]
    assert any(g[0] == key0 for g in good.gets), "the tampered shard's blobs came from R2"


def test_an_interrupted_warm_resumes(tmp_path):
    _need_warm()
    m = synthetic_mirror(tmp_path)
    st = Store("hippius", m["objects"])
    mirror, _ = _mirror(tmp_path, m["objects"], stores=[st])
    mirror.index_dir_for(SRC, m["index_sha256"])     # the index set lands first
    st.fail_first = 6
    first = mirror.warm(SRC, m["index_sha256"], workers=1)
    assert first["failed_spans"] > 0 and first["fetched_blobs"] < m["blobs"]
    second = mirror.warm(SRC, m["index_sha256"], workers=2)
    assert second["failed_spans"] == 0
    assert first["fetched_blobs"] + second["fetched_blobs"] == m["blobs"]
    assert second["cached"] == first["fetched_blobs"]


def test_warm_public_cli_plans_and_fills(tmp_path, monkeypatch, capsys):
    pm = _need_warm()
    from cascade.eval import lane_d_forecasts as fc

    m = synthetic_mirror(tmp_path)
    store = Store("hippius", m["objects"])
    monkeypatch.setattr(pm, "S3RangeStore", lambda spec: store)
    cfg = tmp_path / "mirror.toml"
    cfg.write_text(f'schema = "laned_public_mirror/1"\nprefix = "lane_d_public"\n'
                   f'cache_dir = "{tmp_path / "vol"}"\ncache_budget_bytes = 1000000000\n'
                   '[[store]]\nname = "hippius"\nendpoint = "https://x"\nregion = "r"\n'
                   'bucket = "b"\nkey_env = "NONE"\n', encoding="utf-8")
    arg = ["warm-public", "--index", f"{SRC}={m['index_sha256']}", "--mirror-config", str(cfg)]
    assert fc.main(arg + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PLAN ") and f"blobs={m['blobs']} cached=0" in out
    assert not _span_gets(store), "--dry-run fetches no shard bytes"
    assert fc.main(arg + ["--workers", "3"]) == 0
    out = capsys.readouterr().out
    assert "WARMED " in out and "failed=0" in out
    assert fc.main(arg + ["--dry-run"]) == 0
    assert f"cached={m['blobs']} " in capsys.readouterr().out
    assert fc.main(["warm-public", "--index", "rmisc=nothex", "--mirror-config", str(cfg)]) == 2


# ── the blob cache under the read pool ─────────────────────────────────────


def test_a_put_under_budget_does_not_scan_the_blob_tree(tmp_path, monkeypatch):
    pm = _pm()
    c = pm.BlobCache(tmp_path / "c", budget_bytes=10**9)
    calls = []
    real = c._blobs
    monkeypatch.setattr(c, "_blobs", lambda: calls.append(1) or real())
    for i in range(200):
        b = f"blob {i}".encode()
        assert c.put(_sha(b), b)
    assert len(calls) <= 2, (
        f"{len(calls)} blob-tree scans for 200 puts: a warm cache (~10^6 blobs) makes every "
        "cold read O(cache size)")
    assert c.total_bytes() == sum(len(f"blob {i}".encode()) for i in range(200))


def test_concurrent_readers_of_one_shard_fetch_its_tail_once(tmp_path):
    pm = _pm()
    tail = b"t" * 1000
    st = Store("hippius", {"k": tail}, delay=0.05)
    mirror = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c", budget_bytes=10**6),
                             stores=[st])
    out = []
    ts = [threading.Thread(target=lambda: out.append(
        mirror.blob("k", sha=_sha(tail), offset=0, length=len(tail)))) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == [tail] * 8
    assert len(st.gets) == 1, f"{len(st.gets)} GETs of one tail by 8 concurrent readers"


def test_read_workers_come_from_the_mirror_config(tmp_path):
    pm = _pm()
    cfg = pm.load_mirror_config()
    assert cfg.get("read_workers", 0) >= 8, (
        "the release config reads the drawn series on a pool (read_workers)")
    mirror = pm.mirror_from_config(cfg, cache_dir=tmp_path, store_factory=lambda s: object())
    assert mirror.workers == cfg["read_workers"]
    assert pm.mirror_from_config(cfg, cache_dir=tmp_path, store_factory=lambda s: object(),
                                 workers=3).workers == 3
    bad = Path(pm.MIRROR_CONFIG_PATH).read_text(encoding="utf-8").replace(
        "read_workers = ", "read_workers = 0 # ")
    p = tmp_path / "bad.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="read_workers"):
        pm.load_mirror_config(p)


# ── public-read uploads + the fix-acl sweep ────────────────────────────────


class S3Client:
    """A boto3-shaped fake: ``upload_file``, ``list_objects_v2`` (paginated
    by 2), ``put_object_acl``, ``head_object``, ``copy_object``."""

    def __init__(self, keys=(), *, acl_upload=True, acl_put=True):
        self.keys = list(keys)
        self.calls: list[tuple] = []
        self.acl_upload, self.acl_put = acl_upload, acl_put

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.calls.append(("upload", key, dict(ExtraArgs or {})))
        if ExtraArgs and "ACL" in ExtraArgs and not self.acl_upload:
            raise RuntimeError("NotImplemented: x-amz-acl")

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Key))
        if Key not in self.keys:
            raise KeyError(Key)
        return {"ContentLength": 1, "ContentType": "application/octet-stream"}

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        ks = [k for k in self.keys if k.startswith(Prefix)]
        i = int(ContinuationToken or 0)
        return {"Contents": [{"Key": k} for k in ks[i:i + 2]], "IsTruncated": i + 2 < len(ks),
                "NextContinuationToken": str(i + 2)}

    def put_object_acl(self, Bucket, Key, ACL):
        self.calls.append(("acl", Key, ACL))
        if not self.acl_put:
            raise RuntimeError("NotImplemented")

    def copy_object(self, **kw):
        self.calls.append(("copy", kw["Key"], kw["ACL"]))


def _script():
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("laned_public_mirror_t",
                                                  root / "scripts" / "laned_public_mirror.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_uploads_are_public_read_and_r2_falls_back_without_the_acl(tmp_path):
    s = _script()
    (tmp_path / "a").write_bytes(b"x")
    c = S3Client()
    s.upload_tree(c, "b", tmp_path, "pfx")
    ups = [x for x in c.calls if x[0] == "upload"]
    assert ups == [("upload", "pfx/a", {"ACL": "public-read"})], (
        "upload_tree must upload public-read (anonymous GETs of the mirror were 403)")
    r2 = S3Client(acl_upload=False)
    s.upload_tree(r2, "b", tmp_path, "pfx")
    ups = [x for x in r2.calls if x[0] == "upload"]
    assert ups[-1] == ("upload", "pfx/a", {}), "a store refusing canned ACLs still gets the bytes"
    from cascade.eval import lane_d_public_build as pb

    st = pb.S3PutStore(_pm().StoreSpec("hippius", "https://x", "r", "b", "NONE"))
    st._client = S3Client()
    st.put_file("k", tmp_path / "a")
    assert st._client.calls == [("upload", "k", {"ACL": "public-read"})], \
        "the canonical build's shard + index uploads are public-read too"


def test_fix_acl_sweeps_every_object_under_the_prefix(capsys):
    s = _script()
    assert hasattr(s, "fix_acl"), "scripts/laned_public_mirror.py has no fix-acl sweep"
    keys = [f"lane_d_public/{SRC}/rev/shards/{i}.parquet" for i in range(5)] + ["other/x"]
    c = S3Client(keys)
    got = s.fix_acl(c, "b", f"lane_d_public/{SRC}/", every=2)
    assert got == {"objects": 5, "acl": 5, "copy": 0, "failed": 0}
    assert sorted(x[1] for x in c.calls if x[0] == "acl") == sorted(keys[:5])
    dry = S3Client(keys)
    assert s.fix_acl(dry, "b", f"lane_d_public/{SRC}/", dry_run=True)["objects"] == 5
    assert dry.calls == [], "--dry-run changes nothing"
    nocl = S3Client(keys, acl_put=False)
    got = s.fix_acl(nocl, "b", f"lane_d_public/{SRC}/")
    assert got["copy"] == 5 and got["failed"] == 0, "no PutObjectAcl: an in-place ACL copy"
    assert "ACL " in capsys.readouterr().out
