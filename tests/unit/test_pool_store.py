"""Daily eval-pool snapshots over S3: publish/index/select/fetch primitives,
the consensus-safe BucketWindowSource, and the publish CLI — all over an
in-memory fake store (no boto3, no network)."""

from __future__ import annotations

import dataclasses
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.pool.builder import PoolBuildConfig, build_pool
from cascade.pool.source import HarvestContext, HarvestedSeries
from cascade.shared import hippius
from cascade.shared.config import load_chain_config
from cascade.shared.hippius import (
    StorageError,
    fetch_pool_snapshot,
    pack_dir_to_tar,
    pool_s3_store,
    publish_pool_snapshot,
    read_pool_index,
    select_snapshot,
)
from cascade.validator.pool import BucketWindowSource

CTX = HarvestContext(as_of=dt.date(2026, 6, 1), context_length=128, horizon=16, max_series=1000)
CFG = PoolBuildConfig(context_length=128, horizon=16, min_context=32)


class _FakeS3Store:
    """In-memory stand-in for S3Store (bytes + text surface, StorageError on miss)."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, key, data, *, content_type="application/octet-stream"):
        self.objects[key] = bytes(data)

    def put_text(self, key, text, *, content_type="text/plain"):
        self.objects[key] = text.encode("utf-8")

    def get_bytes(self, key):
        if key not in self.objects:
            raise StorageError(f"missing: {key}")
        return self.objects[key]

    def get_text(self, key):
        return self.get_bytes(key).decode("utf-8")


class _ListSource:
    name = "list"

    def __init__(self, items):
        self.items = items

    def harvest(self, fetch, ctx):
        yield from self.items


def _make_pool_tar(tmp_path, name, n=5, phase=0.0):
    items = [
        HarvestedSeries(
            f"s{i}", 10 + np.sin(np.arange(200) / (3.0 + i) + phase), "H", "weather", 24
        )
        for i in range(n)
    ]
    out = tmp_path / name
    build_pool([_ListSource(items)], out, CTX, CFG, fetch=None)
    return pack_dir_to_tar(out)


# ── primitives ──────────────────────────────────────────────────────────────


def test_publish_index_and_fetch_round_trip(tmp_path):
    store = _FakeS3Store()
    tar = _make_pool_tar(tmp_path, "p1", n=5)
    meta = publish_pool_snapshot(
        store, tar, effective_round=10, as_of="2026-06-01", n_series=5,
        context_length=128, horizon=16,
    )
    assert meta.effective_round == 10 and meta.key == "pool/snapshots/10.tar"

    index = read_pool_index(store)
    assert [m.effective_round for m in index] == [10]
    out = fetch_pool_snapshot(store, index[0], tmp_path / "restored")
    assert (out / "metadata.json").is_file()
    assert len(list(out.glob("*.npy"))) == 5


def test_read_index_empty_when_absent():
    assert read_pool_index(_FakeS3Store()) == []


def test_select_snapshot_picks_greatest_le_round(tmp_path):
    store = _FakeS3Store()
    for er in (5, 10, 20):
        publish_pool_snapshot(
            store, _make_pool_tar(tmp_path, f"p{er}", phase=er), effective_round=er,
            as_of="2026-06-01", n_series=5, context_length=128, horizon=16,
        )
    index = read_pool_index(store)
    assert select_snapshot(index, 4).effective_round == 5     # below all → earliest floor
    assert select_snapshot(index, 5).effective_round == 5
    assert select_snapshot(index, 15).effective_round == 10
    assert select_snapshot(index, 999).effective_round == 20
    assert select_snapshot([], 7) is None


def test_publish_is_idempotent_per_round_and_trims(tmp_path):
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "a"), effective_round=10,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "b", phase=1.0), effective_round=10,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    assert [m.effective_round for m in read_pool_index(store)] == [10]  # replaced, not duplicated

    for er in range(11, 16):
        publish_pool_snapshot(store, _make_pool_tar(tmp_path, f"c{er}", phase=er),
                              effective_round=er, as_of="d", n_series=5,
                              context_length=128, horizon=16, max_keep=3)
    kept = [m.effective_round for m in read_pool_index(store)]
    assert kept == [13, 14, 15]  # trimmed to the most recent max_keep


def test_fetch_rejects_digest_mismatch(tmp_path):
    store = _FakeS3Store()
    meta = publish_pool_snapshot(store, _make_pool_tar(tmp_path, "p"), effective_round=1,
                                 as_of="d", n_series=5, context_length=128, horizon=16)
    store.objects[meta.key] = b"corrupted"  # tamper with the stored tar
    with pytest.raises(StorageError, match="digest_mismatch"):
        fetch_pool_snapshot(store, meta, tmp_path / "x")


# ── BucketWindowSource (consensus-safe per-round selection) ──────────────────


def _cfg_small():
    base = load_chain_config()
    return dataclasses.replace(base, eval=dataclasses.replace(base.eval, context_length=128, horizon=16))


def test_bucket_source_selects_by_round_and_rotates(tmp_path):
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "old", n=5, phase=0.0), effective_round=1,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "new", n=8, phase=1.0), effective_round=100,
                          as_of="d", n_series=8, context_length=128, horizon=16)

    src = BucketWindowSource(_cfg_small(), store, cache_dir=tmp_path / "cache")

    # round 50 → snapshot@1 (5 series); round 200 → snapshot@100 (8 series)
    w_old = src.windows_for_round(50, 50)
    w_new = src.windows_for_round(200, 50)
    assert len(w_old) == 5 and len(w_new) == 8

    # same snapshot, different round → different rotation order
    a = [w.series_id for w in src.windows_for_round(50, 50)]
    b = [w.series_id for w in src.windows_for_round(51, 50)]
    assert sorted(a) == sorted(b) and a != b


def test_bucket_source_picks_up_new_snapshot(tmp_path):
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "v1", n=5), effective_round=1,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    src = BucketWindowSource(_cfg_small(), store, cache_dir=tmp_path / "cache")
    assert len(src.windows_for_round(10, 50)) == 5

    # orchestrator publishes a bigger snapshot effective from round 20
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "v2", n=9, phase=2.0), effective_round=20,
                          as_of="d", n_series=9, context_length=128, horizon=16)
    assert len(src.windows_for_round(15, 50)) == 5   # still on v1 for round 15
    assert len(src.windows_for_round(25, 50)) == 9   # switched to v2 for round 25


def test_bucket_source_raises_when_no_snapshot(tmp_path):
    src = BucketWindowSource(_cfg_small(), _FakeS3Store(), cache_dir=tmp_path)
    with pytest.raises(Exception, match="no eval-pool snapshot"):
        src.windows_for_round(1, 50)


# ── credential / backend resolution ─────────────────────────────────────────


def test_pool_s3_store_defaults_to_hippius(monkeypatch):
    monkeypatch.delenv("POOL_S3_ACCESS_KEY", raising=False)
    storage = SimpleNamespace(
        pool_bucket="cascade-eval-pool", pool_s3_endpoint="", pool_s3_region="",
        s3_endpoint="https://s3.hippius.com", s3_region="decentralized",
    )
    store = pool_s3_store(storage)
    assert store.cfg.bucket == "cascade-eval-pool"
    assert store.cfg.endpoint == "https://s3.hippius.com"
    assert store.cfg.access_key_env == "HIPPIUS_S3_ACCESS_KEY"


def test_pool_s3_store_uses_r2_when_configured(monkeypatch):
    monkeypatch.setenv("POOL_S3_ACCESS_KEY", "r2key")
    storage = SimpleNamespace(
        pool_bucket="pool", pool_s3_endpoint="https://acct.r2.cloudflarestorage.com",
        pool_s3_region="auto", s3_endpoint="https://s3.hippius.com", s3_region="decentralized",
    )
    store = pool_s3_store(storage)
    assert store.cfg.endpoint.endswith("r2.cloudflarestorage.com")
    assert store.cfg.region == "auto"
    assert store.cfg.access_key_env == "POOL_S3_ACCESS_KEY"


def test_publish_cli_end_to_end(tmp_path, monkeypatch):
    from cascade.pool import cli

    store = _FakeS3Store()
    cfg = dataclasses.replace(
        _cfg_small(),
        storage=dataclasses.replace(load_chain_config().storage, pool_bucket="cascade-eval-pool"),
    )
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(hippius, "pool_s3_store", lambda *_a, **_k: store)

    rc = cli.main(
        ["publish", "--sources", "synthetic", "--out", str(tmp_path / "stage"),
         "--effective-round", "7", "--context-length", "128", "--horizon", "16", "--min-context", "32"]
    )
    assert rc == 0
    index = read_pool_index(store)
    assert [m.effective_round for m in index] == [7]
    # the published snapshot is loadable for a round it governs
    src = BucketWindowSource(cfg, store, cache_dir=tmp_path / "c")
    assert len(src.windows_for_round(9, 50)) > 0
