"""W6p (owner 2026-11-24, issue #894): the public mirror as a Hippius S3
object set. Content-addressed Parquet shards + the shard map, built by a
resumable streaming job; the eval host fetches ONLY the drawn series by
ranged GET (volume cache, then Hippius, then R2), verifying every sha256.
Contract: README.md "Hippius mirror". Offline: the stores are in-memory
fakes, the "HF" fetch copies the fixture mirror; nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from . import fixtures as fx

pytest.importorskip("pyarrow")


def _pm():
    return importlib.import_module("cascade.eval.lane_d_public_mirror")


def _pb():
    return importlib.import_module("cascade.eval.lane_d_public_build")


def _idx():
    return importlib.import_module("cascade.eval.lane_d_public_index")


class FakeStore:
    """An in-memory S3 store: ranged ``get``, ``head``, ``put_file``; every
    call is recorded."""

    def __init__(self, name: str, objects: dict | None = None):
        self.name = name
        self.objects: dict[str, bytes] = {} if objects is None else objects
        self.gets: list[tuple[str, int | None, int | None]] = []
        self.puts: list[str] = []
        self.down = False

    def get(self, key, *, offset=None, length=None):
        self.gets.append((key, offset, length))
        if self.down:
            raise ConnectionError(f"{self.name} down")
        data = self.objects[key]
        return data if offset is None else data[offset:offset + length]

    def head(self, key):
        return len(self.objects[key]) if key in self.objects else None

    def put_file(self, key, path):
        self.puts.append(key)
        self.objects[key] = Path(path).read_bytes()


def _listing(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            e = {"type": "file", "path": p.relative_to(root).as_posix(), "size": p.stat().st_size}
            if p.suffix in (".arrow", ".parquet"):
                e["lfs"] = {"oid": hashlib.sha256(p.read_bytes()).hexdigest()}
            out.append(e)
    return out


def _fetcher(root: Path, log: list | None = None):
    def fetch(path, dest):
        if log is not None:
            log.append(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, dest)
    return fetch


def _build(tmp_path: Path, source: str, *, stores=None, rg_target=4096, shard_target=20000,
           fetch_log=None):
    if source == fx.GIFT:
        root, ds = fx.gift_mirror(tmp_path)
    else:
        root, ds = fx.rmisc_mirror(tmp_path)
    stores = stores if stores is not None else [FakeStore("hippius"), FakeStore("r2")]
    kw = {"source": source, "listing": _listing(root), "work_dir": tmp_path / "work",
          "stores": stores, "fetch": _fetcher(root, fetch_log), "rg_target": rg_target,
          "shard_target": shard_target, "echo": lambda *_: None}
    pb = _pb()
    pb.build_all(kw, ds)
    out = pb.Builder(**kw).finalize(ds, tmp_path / f"built_{source}")
    return {"root": root, "datasets": ds, "stores": stores, "kw": kw, "out": out,
            "index": tmp_path / f"built_{source}"}


@pytest.mark.parametrize("source", [fx.GIFT, fx.RMISC])
def test_the_built_index_is_the_w6p_index_of_a_full_mirror(tmp_path, source):
    """The validator binds index_sha256 / files_sha256: the streaming build
    must produce the very index build_index writes over a local mirror."""
    b = _build(tmp_path, source)
    full = _idx().build_index(b["root"], source=source, datasets=b["datasets"],
                              out_dir=tmp_path / "full")
    m = b["out"]["manifest"]
    assert (m["index_sha256"], m["files_sha256"], m["n_series"]) == (
        full["index_sha256"], full["files_sha256"], full["n_series"])
    assert (b["index"] / "index.jsonl.gz").read_bytes() == (tmp_path / "full" /
                                                            "index.jsonl.gz").read_bytes()
    assert (b["index"] / "manifest.json").read_bytes() == (tmp_path / "full" /
                                                           "manifest.json").read_bytes()


@pytest.mark.parametrize("source", [fx.GIFT, fx.RMISC])
def test_objects_are_content_addressed_and_the_index_set_uploaded(tmp_path, source):
    b = _build(tmp_path, source)
    pm = _pm()
    doc = pm.load_shards(b["index"])
    base = pm.base_key(pm.DEFAULT_PREFIX, source)
    for st in b["stores"]:
        for s in doc["shards"]:
            data = st.objects[f"{base}/{s['key']}"]
            assert hashlib.sha256(data).hexdigest() == s["sha256"] and len(data) == s["size"]
            assert s["key"] == f"shards/{s['sha256']}.parquet"
        for name in pm.INDEX_FILES:
            key = pm.index_key(pm.DEFAULT_PREFIX, source, doc["index_sha256"], name)
            assert st.objects[key] == (b["index"] / name).read_bytes()
    assert len(doc["shards"]) > 1, "the tiny shard target must split the source"


@pytest.mark.parametrize("source", [fx.GIFT, fx.RMISC])
def test_every_series_decodes_to_read_series_exactly(tmp_path, source):
    """Same canonical series: the shard map's values_sha256 is the hash of
    read_series over the HF files, and the ranged read returns those bytes."""
    b = _build(tmp_path, source)
    pm = _pm()
    idx = _idx()
    doc = pm.load_shards(b["index"])
    m = idx.load_manifest(b["index"])
    rows = list(idx.iter_index(b["index"]))
    mirror = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "cache", budget_bytes=10**9),
                             stores=b["stores"])
    draw = {"source": source, "index_sha256": m["index_sha256"],
            "files_sha256": m["files_sha256"],
            "windows": [{"ordinal": r["ordinal"]} for r in rows]}
    _d, read = mirror.reader(draw)
    for r in rows:
        want = idx.read_series(b["root"], r, source=source)
        got = read(r)
        assert got.dtype == np.float64 and got.shape == want.shape
        assert np.array_equal(got, want, equal_nan=True), r["series_id"]
        assert pm.values_sha256(want) == pm.values_sha256(got)
    assert doc["n_series"] == len(rows)


def test_nan_and_multichannel_series_round_trip(tmp_path):
    pm = _pm()
    pb = _pb()
    idx = _idx()
    root = tmp_path / "gift"
    fx.write_gift(root, "nanny", [300, 300], freq="H", channels=2, nan_at={0: [(1, 7), (0, 0)]})
    kw = {"source": fx.GIFT, "listing": _listing(root), "work_dir": tmp_path / "w",
          "stores": [FakeStore("hippius")], "fetch": _fetcher(root), "echo": lambda *_: None}
    pb.build_all(kw, ["nanny"])
    pb.Builder(**kw).finalize(["nanny"], tmp_path / "ix")
    m = idx.load_manifest(tmp_path / "ix")
    rows = list(idx.iter_index(tmp_path / "ix"))
    mirror = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c", budget_bytes=10**8),
                             stores=kw["stores"])
    _d, read = mirror.reader({"source": fx.GIFT, "index_sha256": m["index_sha256"],
                              "files_sha256": m["files_sha256"],
                              "windows": [{"ordinal": 0}, {"ordinal": 1}]})
    a = read(rows[0])
    assert a.shape == (2, 300) and np.isnan(a[1, 7]) and np.isnan(a[0, 0])
    assert np.array_equal(a, idx.read_series(root, rows[0], source=fx.GIFT), equal_nan=True)


def test_ranged_fetch_verifies_and_falls_back_to_r2(tmp_path):
    b = _build(tmp_path, fx.GIFT)
    pm = _pm()
    idx = _idx()
    hip, r2 = (FakeStore("hippius", dict(b["stores"][0].objects)),
               FakeStore("r2", dict(b["stores"][1].objects)))
    m = idx.load_manifest(b["index"])
    rows = list(idx.iter_index(b["index"]))
    draw = {"source": fx.GIFT, "index_sha256": m["index_sha256"],
            "files_sha256": m["files_sha256"], "windows": [{"ordinal": 3}]}
    mirror = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c", budget_bytes=10**9),
                             stores=[hip, r2])
    _d, read = mirror.reader(draw)
    hip.down = True
    a = read(rows[3])
    assert np.array_equal(a, idx.read_series(b["root"], rows[3], source=fx.GIFT), equal_nan=True)
    shard_gets = [g for g in r2.gets if "/shards/" in g[0]]
    assert shard_gets and all(g[1] is not None and g[2] is not None for g in shard_gets), (
        "shard bytes are fetched by ranged GET only, never the whole shard")
    r2.down = True
    mirror2 = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c2", budget_bytes=10**9),
                              stores=[hip, r2])
    with pytest.raises(pm.MirrorError):
        mirror2.reader(draw)


def test_a_tampered_shard_is_refused_and_the_backup_used(tmp_path):
    b = _build(tmp_path, fx.GIFT)
    pm = _pm()
    idx = _idx()
    m = idx.load_manifest(b["index"])
    rows = list(idx.iter_index(b["index"]))
    doc = pm.load_shards(b["index"])
    where = pm.lookup(b["index"], doc, [5])[5]
    key = f"{pm.base_key(pm.DEFAULT_PREFIX, fx.GIFT)}/{where['shard']}"
    hip = FakeStore("hippius", dict(b["stores"][0].objects))
    bad = bytearray(hip.objects[key])
    bad[where["offset"] + where["length"] // 2] ^= 0xFF
    hip.objects[key] = bytes(bad)
    draw = {"source": fx.GIFT, "index_sha256": m["index_sha256"],
            "files_sha256": m["files_sha256"], "windows": [{"ordinal": 5}]}
    alone = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c1", budget_bytes=10**9),
                            stores=[hip])
    _d, read = alone.reader(draw)
    with pytest.raises(pm.MirrorError, match="sha256"):
        read(rows[5])
    assert not list((tmp_path / "c1" / "blobs").glob(f"*/{where['sha256']}")), (
        "tampered bytes never enter the cache")
    both = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c2", budget_bytes=10**9),
                           stores=[hip, FakeStore("r2", dict(b["stores"][1].objects))])
    _d, read = both.reader(draw)
    assert np.array_equal(read(rows[5]), idx.read_series(b["root"], rows[5], source=fx.GIFT),
                          equal_nan=True)


def test_a_tampered_index_set_is_refused(tmp_path):
    b = _build(tmp_path, fx.GIFT)
    pm = _pm()
    idx = _idx()
    m = idx.load_manifest(b["index"])
    hip = FakeStore("hippius", dict(b["stores"][0].objects))
    key = pm.index_key(pm.DEFAULT_PREFIX, fx.GIFT, m["index_sha256"], pm.SHARDMAP_FILE)
    import gzip

    lines = gzip.decompress(hip.objects[key]).splitlines(keepends=True)
    row = json.loads(lines[0])
    row["offset"] += 1
    lines[0] = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    hip.objects[key] = gzip.compress(b"".join(lines))
    mirror = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c", budget_bytes=10**9),
                             stores=[hip])
    draw = {"source": fx.GIFT, "index_sha256": m["index_sha256"],
            "files_sha256": m["files_sha256"], "windows": [{"ordinal": 0}]}
    with pytest.raises(ValueError, match="shardmap_sha256"):
        mirror.reader(draw)
    other = {**draw, "files_sha256": "0" * 64}
    with pytest.raises((ValueError, pm.MirrorError)):
        pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c3", budget_bytes=10**9),
                        stores=[FakeStore("h", dict(b["stores"][0].objects))]).reader(other)


def test_the_cache_serves_repeat_reads_and_reverifies_every_read(tmp_path):
    b = _build(tmp_path, fx.GIFT)
    pm = _pm()
    idx = _idx()
    m = idx.load_manifest(b["index"])
    rows = list(idx.iter_index(b["index"]))
    hip = FakeStore("hippius", dict(b["stores"][0].objects))
    cache = pm.BlobCache(tmp_path / "vol", budget_bytes=10**9)
    draw = {"source": fx.GIFT, "index_sha256": m["index_sha256"],
            "files_sha256": m["files_sha256"], "windows": [{"ordinal": 2}]}
    _d, read = pm.PublicMirror(cache=cache, stores=[hip]).reader(draw)
    want = read(rows[2])
    n = len(hip.gets)
    _d, read = pm.PublicMirror(cache=cache, stores=[hip]).reader(draw)
    assert np.array_equal(read(rows[2]), want, equal_nan=True)
    assert len(hip.gets) == n, "a warm volume serves the series and the index with no GET"
    # Corrupt every cached blob: each read re-hashes, drops it and refetches.
    for p in (tmp_path / "vol" / "blobs").glob("*/*"):
        p.write_bytes(b"x" + p.read_bytes()[1:])
    _d, read = pm.PublicMirror(cache=cache, stores=[hip]).reader(draw)
    assert np.array_equal(read(rows[2]), want, equal_nan=True)
    assert len(hip.gets) > n
    # A wiped volume refills from Hippius.
    shutil.rmtree(tmp_path / "vol")
    _d, read = pm.PublicMirror(cache=cache, stores=[hip]).reader(draw)
    assert np.array_equal(read(rows[2]), want, equal_nan=True)


def test_cache_eviction_keeps_the_byte_budget_lru(tmp_path):
    pm = _pm()
    c = pm.BlobCache(tmp_path / "c", budget_bytes=250)
    blobs = [bytes([i]) * 100 for i in range(4)]
    shas = [hashlib.sha256(b).hexdigest() for b in blobs]
    import os
    import time

    for i, (s, b) in enumerate(zip(shas, blobs[:2], strict=False)):
        assert c.put(s, b)
        os.utime(c._path(s), ns=(time.time_ns() - (10 - i) * 10**9,) * 2)
    assert c.get(shas[0]) == blobs[0]          # touch: blob 1 is now the LRU one
    assert c.put(shas[2], blobs[2])
    assert c.total_bytes() <= 250
    assert c.get(shas[1]) is None and c.get(shas[0]) == blobs[0] and c.get(shas[2]) == blobs[2]
    assert not c.put(hashlib.sha256(b"y" * 300).hexdigest(), b"y" * 300), "over budget: uncached"
    with pytest.raises(pm.MirrorError):
        c.put(shas[3], blobs[0])                # bytes that do not hash to the key


def test_draw_then_fetch_touches_only_the_drawn_series(tmp_path):
    """After the box's deterministic draw the host GETs the drawn series'
    row groups (+ each touched shard's tail) and nothing else."""
    from cascade.validator.lane_d_public import draw_public, read_public_windows

    b = _build(tmp_path, fx.GIFT, rg_target=1)          # one series per row group
    pm = _pm()
    hip = FakeStore("hippius", dict(b["stores"][0].objects))
    draw = draw_public(b["index"], source=fx.GIFT, boundary_block_hash=fx.BLOCK_HASH,
                       horizon=24, n_windows=5)
    mirror = pm.PublicMirror(cache=pm.BlobCache(tmp_path / "c", budget_bytes=10**9),
                             stores=[hip])
    d, reader = mirror.reader(draw)
    wins, drops = read_public_windows(draw, index_dir=d, reader=reader,
                                      block_hash=fx.BLOCK_HASH, max_channels=32)
    local = read_public_windows(draw, index_dir=b["index"], data_root=b["root"],
                                block_hash=fx.BLOCK_HASH, max_channels=32)
    assert [w["window_id"] for w in wins] == [w["window_id"] for w in local[0]]
    assert drops == local[1]
    for x, y in zip(wins, local[0], strict=True):
        assert np.array_equal(x["history"], y["history"], equal_nan=True)
        assert np.array_equal(x["target"], y["target"], equal_nan=True)
    doc = pm.load_shards(d)
    where = pm.lookup(d, doc, [w["ordinal"] for w in draw["windows"]])
    shards = {s["key"]: s for s in doc["shards"]}
    allowed = {(e["shard"], e["offset"], e["length"]) for e in where.values()}
    allowed |= {(shards[e["shard"]]["key"], shards[e["shard"]]["tail_offset"],
                 shards[e["shard"]]["tail_length"]) for e in where.values()}
    shard_gets = {(k.split("/", 3)[3], o, n) for k, o, n in hip.gets if "/shards/" in k}
    assert shard_gets and shard_gets <= allowed, "a GET outside the drawn series' bytes"
    n_series = doc["n_series"]
    assert len({e["ordinal"] for e in where.values()}) < n_series


def test_the_build_resumes_by_shard_and_is_idempotent(tmp_path):
    """Killed after a few shards, a rerun uploads only the missing shards,
    and the result is byte-identical to an uninterrupted build."""
    pb = _pb()
    pm = _pm()
    ref = _build(tmp_path / "ref", fx.GIFT)
    root, ds = fx.gift_mirror(tmp_path / "run")
    hip = FakeStore("hippius")
    kw = {"source": fx.GIFT, "listing": _listing(root), "work_dir": tmp_path / "run" / "work",
          "stores": [hip], "fetch": _fetcher(root), "rg_target": 4096, "shard_target": 20000,
          "echo": lambda *_: None}

    class Boom(Exception):
        pass

    real = pb.upload
    calls = {"n": 0}

    def dying(stores, key, path):
        calls["n"] += 1
        if calls["n"] == 4:
            raise Boom()
        return real(stores, key, path)

    pb.upload = dying
    try:
        with pytest.raises(Boom):
            pb.build_all(kw, ds)
    finally:
        pb.upload = real
    before = set(hip.objects)
    assert before, "some shards landed before the crash"
    hip.puts.clear()
    pb.build_all(kw, ds)
    out = pb.Builder(**kw).finalize(ds, tmp_path / "run" / "ix")
    assert not set(hip.puts) & before, "a resumed build never re-uploads a landed shard"
    assert out["manifest"]["index_sha256"] == ref["out"]["manifest"]["index_sha256"]
    assert out["shards"]["shardmap_sha256"] == ref["out"]["shards"]["shardmap_sha256"]
    assert {k.split("/", 3)[3] for k in hip.objects if "/shards/" in k} == {
        s["key"] for s in pm.load_shards(ref["index"])["shards"]}
    # Idempotent: a third run does no download and no upload.
    hip.puts.clear()
    log: list = []
    kw2 = {**kw, "fetch": _fetcher(root, log)}
    pb.build_all(kw2, ds)
    pb.Builder(**kw2).finalize(ds, tmp_path / "run" / "ix2")
    assert hip.puts == [] and log == []


def test_build_refuses_a_file_that_is_not_the_listing(tmp_path):
    pb = _pb()
    root, ds = fx.gift_mirror(tmp_path)
    listing = _listing(root)
    for e in listing:
        if e["path"].endswith(".arrow"):
            e["lfs"]["oid"] = "0" * 64
            break
    kw = {"source": fx.GIFT, "listing": listing, "work_dir": tmp_path / "w",
          "stores": [FakeStore("h")], "fetch": _fetcher(root), "echo": lambda *_: None}
    with pytest.raises(RuntimeError, match="LFS oid"):
        pb.build_all(kw, ds)


def test_fetch_index_gives_the_box_a_verified_index(tmp_path, monkeypatch, capsys):
    import importlib.util

    b = _build(tmp_path, fx.GIFT)
    pm = _pm()
    objects = dict(b["stores"][0].objects)
    monkeypatch.setattr(pm, "S3RangeStore", lambda spec: FakeStore(spec.name, objects))
    path = Path(__file__).resolve().parents[3] / "scripts" / "laned_public_mirror.py"
    spec = importlib.util.spec_from_file_location("laned_public_mirror_fetch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sha = b["out"]["manifest"]["index_sha256"]
    out = tmp_path / "box_idx"
    assert mod.main(["fetch-index", "--source", fx.GIFT, "--index-sha256", sha,
                     "--index-dir", str(out)]) == 0
    assert f"index_sha256={sha}" in capsys.readouterr().out
    assert (out / "index.jsonl.gz").read_bytes() == (b["index"] / "index.jsonl.gz").read_bytes()
    with pytest.raises(SystemExit):
        mod.main(["fetch-index", "--source", fx.GIFT, "--index-sha256", "0" * 64,
                  "--index-dir", str(tmp_path / "nope")])


def test_estimate_extrapolates_the_measured_ratio():
    est = _pb().estimate([{"raw_bytes": 1000, "shard_bytes": 500, "seconds": 36.0},
                          {"raw_bytes": 3000, "shard_bytes": 1500, "seconds": 108.0}], 40000)
    assert est["ratio"] == 0.5 and est["projected_shard_bytes"] == 20000
    assert est["projected_single_worker_hours"] == 0.4


def test_mirror_config_names_stores_and_the_volume_cache():
    pm = _pm()
    cfg = pm.load_mirror_config()
    assert [s.name for s in cfg["stores"]] == ["hippius", "r2"]
    assert cfg["stores"][0].endpoint == "https://s3.hippius.com"
    assert cfg["cache_dir"].startswith("/workspace/")
    text = pm.MIRROR_CONFIG_PATH.read_text()
    assert "SECRET" not in text.replace("_SECRET_KEY", "") and "key_env" in text
