"""Daily eval-pool snapshots over S3: publish/index/select/fetch primitives,
the consensus-safe BucketWindowSource, and the publish CLI — all over an
in-memory fake store (no boto3, no network)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.pool.builder import PoolBuildConfig, build_pool
from cascade.pool.source import HarvestContext, HarvestedSeries
from cascade.shared import hippius
from cascade.shared.config import load_chain_config
from cascade.shared.hippius import (
    ObjectNotFound,
    S3MirrorStore,
    S3Store,
    StorageError,
    fetch_pool_snapshot,
    pack_dir_to_tar,
    pool_backup_s3_store,
    pool_s3_store,
    publish_pool_snapshot,
    read_pool_index,
    select_snapshot,
)
from cascade.validator.pool import BucketWindowSource

CTX = HarvestContext(as_of=dt.date(2026, 6, 1), context_length=128, horizon=16, max_series=1000)
CFG = PoolBuildConfig(context_length=128, horizon=16, min_context=32)


class _FakeS3Store:
    """In-memory stand-in for S3Store (bytes + text surface).

    Mirrors real S3 semantics: a missing key raises :class:`ObjectNotFound`
    (the ``NoSuchKey`` case), NOT a bare ``StorageError``. ``fail`` forces every
    read AND write to raise a plain ``StorageError`` instead, modelling an
    unreadable/unwritable bucket (auth/network/5xx) — distinct from a genuinely
    absent object."""

    def __init__(self, *, fail: bool = False):
        self.objects: dict[str, bytes] = {}
        self.fail = fail

    def put_bytes(self, key, data, *, content_type="application/octet-stream", acl=None):
        if self.fail:
            raise StorageError(f"s3_put_failed: {key}: 500")
        self.objects[key] = bytes(data)

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type, acl=acl)

    def get_bytes(self, key):
        if self.fail:
            raise StorageError(f"bucket_unavailable: {key}")
        if key not in self.objects:
            raise ObjectNotFound(f"s3_get_missing: {key}")
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
        store, tar, effective_block=7200, as_of="2026-06-01", n_series=5,
        context_length=128, horizon=16,
    )
    assert meta.effective_block == 7200 and meta.key == "pool/snapshots/block-7200.tar"

    index = read_pool_index(store)
    assert [m.effective_block for m in index] == [7200]
    out = fetch_pool_snapshot(store, index[0], tmp_path / "restored")
    assert (out / "metadata.json").is_file()
    assert len(list(out.glob("*.npy"))) == 5


def test_read_index_empty_when_absent():
    assert read_pool_index(_FakeS3Store()) == []


def test_read_index_raises_when_bucket_unreadable():
    # A read FAILURE (auth/network/wrong bucket) must NOT masquerade as an empty
    # index — the old bare ``except StorageError: return []`` turned a pool-bucket
    # blip into "no snapshot" and the validator rejected every pinned round.
    with pytest.raises(StorageError, match="bucket_unavailable"):
        read_pool_index(_FakeS3Store(fail=True))


def test_is_missing_object_distinguishes_absent_from_unreadable():
    # A missing OBJECT (NoSuchKey / 404) is absence; a missing BUCKET or a 403
    # is a read failure and must not be reported as "not there".
    missing_key = SimpleNamespace(response={"Error": {"Code": "NoSuchKey"}})
    http_404 = SimpleNamespace(response={"ResponseMetadata": {"HTTPStatusCode": 404}})
    missing_bucket = SimpleNamespace(response={"Error": {"Code": "NoSuchBucket"}})
    forbidden = SimpleNamespace(response={"ResponseMetadata": {"HTTPStatusCode": 403}})
    assert hippius._is_missing_object(missing_key) is True
    assert hippius._is_missing_object(http_404) is True
    assert hippius._is_missing_object(missing_bucket) is False
    assert hippius._is_missing_object(forbidden) is False
    assert hippius._is_missing_object(RuntimeError("no .response attr")) is False


def test_provenance_absent_returns_empty_but_unreadable_propagates(tmp_path):
    # Absent index → ("", "") (legacy unpinned semantics, receipt carries no pin).
    src = BucketWindowSource(_cfg_small(), _FakeS3Store(), cache_dir=tmp_path / "a")
    assert src.provenance_for_round(123, block=7200) == ("", "")
    # Unreadable index → propagate, so the pin gate reports a distinct
    # "provenance lookup failed" reject rather than "resolved no snapshot".
    bad = BucketWindowSource(_cfg_small(), _FakeS3Store(fail=True), cache_dir=tmp_path / "b")
    with pytest.raises(StorageError, match="bucket_unavailable"):
        bad.provenance_for_round(123, block=7200)


def test_index_reads_legacy_effective_round_key():
    # A v1 index keyed snapshots by ``effective_round``; it must still parse
    # (a redeploy republishes with ``effective_block``).
    store = _FakeS3Store()
    store.objects[hippius.POOL_INDEX_KEY] = json.dumps({"schema": 1, "snapshots": [
        {"effective_round": 7200, "key": "pool/snapshots/7200.tar", "sha256": "a" * 64,
         "size_bytes": 1, "as_of": "d", "n_series": 5, "context_length": 128, "horizon": 16},
    ]}).encode()
    index = read_pool_index(store)
    assert [m.effective_block for m in index] == [7200]


def test_select_snapshot_picks_greatest_le_epoch_block(tmp_path):
    store = _FakeS3Store()
    for blk in (7200, 14400, 28800):
        publish_pool_snapshot(
            store, _make_pool_tar(tmp_path, f"p{blk}", phase=blk), effective_block=blk,
            as_of="2026-06-01", n_series=5, context_length=128, horizon=16,
        )
    index = read_pool_index(store)
    assert select_snapshot(index, 7000).effective_block == 7200     # below all → earliest floor
    assert select_snapshot(index, 7200).effective_block == 7200
    assert select_snapshot(index, 20000).effective_block == 14400
    assert select_snapshot(index, 999999).effective_block == 28800
    assert select_snapshot([], 7) is None


def test_publish_is_idempotent_per_block_and_trims(tmp_path):
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "a"), effective_block=7200,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "b", phase=1.0), effective_block=7200,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    assert [m.effective_block for m in read_pool_index(store)] == [7200]  # replaced, not duplicated

    for i in range(1, 6):
        publish_pool_snapshot(store, _make_pool_tar(tmp_path, f"c{i}", phase=i),
                              effective_block=7200 + i * 7200, as_of="d", n_series=5,
                              context_length=128, horizon=16, max_keep=3)
    kept = [m.effective_block for m in read_pool_index(store)]
    assert kept == [28800, 36000, 43200]  # trimmed to the most recent max_keep


def test_fetch_rejects_digest_mismatch(tmp_path):
    store = _FakeS3Store()
    meta = publish_pool_snapshot(store, _make_pool_tar(tmp_path, "p"), effective_block=7200,
                                 as_of="d", n_series=5, context_length=128, horizon=16)
    store.objects[meta.key] = b"corrupted"  # tamper with the stored tar
    with pytest.raises(StorageError, match="digest_mismatch"):
        fetch_pool_snapshot(store, meta, tmp_path / "x")


# ── BucketWindowSource (consensus-safe per-round selection) ──────────────────


def _cfg_small():
    base = load_chain_config()
    return dataclasses.replace(base, eval=dataclasses.replace(base.eval, context_length=128, horizon=16))


def test_bucket_source_selects_by_epoch_block_and_rotates(tmp_path):
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "old", n=5, phase=0.0), effective_block=7200,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "new", n=8, phase=1.0), effective_block=720000,
                          as_of="d", n_series=8, context_length=128, horizon=16)

    src = BucketWindowSource(_cfg_small(), store, cache_dir=tmp_path / "cache")

    # epoch block 10000 → snapshot@7200 (5 series); 800000 → snapshot@720000 (8)
    w_old = src.windows_for_round(50, 50, block=10000)
    w_new = src.windows_for_round(200, 50, block=800000)
    assert len(w_old) == 5 and len(w_new) == 8

    # same snapshot (same block), different round seed → different rotation order
    a = [w.series_id for w in src.windows_for_round(50, 50, block=10000)]
    b = [w.series_id for w in src.windows_for_round(51, 50, block=10000)]
    assert sorted(a) == sorted(b) and a != b


def test_bucket_source_ignores_nonmonotonic_round_id(tmp_path):
    # The regression: round ids are block-HASH seeds (huge, random, unordered).
    # Selection must key on the epoch BLOCK, so a later round with a *smaller*
    # random round id still lands on the newer snapshot.
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "old", n=5), effective_block=7200,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "new", n=8, phase=1.0), effective_block=14400,
                          as_of="d", n_series=8, context_length=128, horizon=16)
    src = BucketWindowSource(_cfg_small(), store, cache_dir=tmp_path / "cache")

    # round in the LATER epoch (block 15000) but with a TINY random round id →
    # must still get the newer snapshot; the old round-id compare would have
    # picked the older (or fallen through), diverging validators.
    assert len(src.windows_for_round(3, 50, block=15000)) == 8
    # round in the EARLIER epoch (block 8000) with a HUGE random round id →
    # still the old snapshot.
    assert len(src.windows_for_round(10**19, 50, block=8000)) == 5


def test_bucket_source_no_block_uses_newest(tmp_path):
    # A caller that can't supply the epoch block gets the newest snapshot — a
    # safe deterministic default, never the broken round-id comparison.
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "v1", n=5), effective_block=7200,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "v2", n=9, phase=2.0), effective_block=14400,
                          as_of="d", n_series=9, context_length=128, horizon=16)
    src = BucketWindowSource(_cfg_small(), store, cache_dir=tmp_path / "cache")
    assert len(src.windows_for_round(123, 50)) == 9   # newest, block omitted


def test_bucket_source_picks_up_new_snapshot(tmp_path):
    store = _FakeS3Store()
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "v1", n=5), effective_block=7200,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    src = BucketWindowSource(_cfg_small(), store, cache_dir=tmp_path / "cache")
    assert len(src.windows_for_round(10, 50, block=8000)) == 5

    # orchestrator publishes a bigger snapshot effective from epoch block 14400
    publish_pool_snapshot(store, _make_pool_tar(tmp_path, "v2", n=9, phase=2.0), effective_block=14400,
                          as_of="d", n_series=9, context_length=128, horizon=16)
    assert len(src.windows_for_round(15, 50, block=10000)) == 5   # still on v1 for epoch 10000
    assert len(src.windows_for_round(25, 50, block=15000)) == 9   # switched to v2 for epoch 15000


def test_bucket_source_raises_when_no_snapshot(tmp_path):
    src = BucketWindowSource(_cfg_small(), _FakeS3Store(), cache_dir=tmp_path)
    with pytest.raises(Exception, match="no eval-pool snapshot"):
        src.windows_for_round(1, 50, block=7200)


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
         "--effective-block", "7200", "--context-length", "128", "--horizon", "16",
         "--min-context", "32"]
    )
    assert rc == 0
    index = read_pool_index(store)
    assert [m.effective_block for m in index] == [7200]
    # the published snapshot is loadable for a round in an epoch it governs
    src = BucketWindowSource(cfg, store, cache_dir=tmp_path / "c")
    assert len(src.windows_for_round(9, 50, block=8000)) > 0


def test_publish_cli_deprecated_effective_round_alias(tmp_path, monkeypatch):
    # --effective-round still works (maps to --effective-block) so existing
    # cron invocations don't break; the value is a BLOCK now.
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
         "--effective-round", "14400", "--context-length", "128", "--horizon", "16",
         "--min-context", "32"]
    )
    assert rc == 0
    assert [m.effective_block for m in read_pool_index(store)] == [14400]


def test_resolve_effective_block_auto_projects_next_epoch(monkeypatch):
    # auto reads the manifest created_block, floors to the epoch grid, and adds
    # round_buffer epochs — a FUTURE epoch, never one already scored.
    from types import SimpleNamespace

    from cascade.pool import cli

    cfg = _cfg_small()
    cfg = dataclasses.replace(cfg, round=dataclasses.replace(cfg.round, epoch_blocks=7200))

    class _Manifest:
        created_block = 3 * 7200 + 123   # mid-epoch 3
    monkeypatch.setattr("cascade.shared.manifest.load_manifest", lambda *_a: _Manifest())
    monkeypatch.setattr("cascade.shared.hippius.read_latest_manifest", lambda *_a: "{}")
    monkeypatch.setattr("cascade.shared.hippius.S3Store", lambda *_a, **_k: object())
    args = SimpleNamespace(effective_block="auto", round_buffer=1)
    # epoch_start = 3*7200 = 21600; +1 epoch = 28800
    assert cli._resolve_effective_block(args, cfg) == 28800
    args.round_buffer = 2
    assert cli._resolve_effective_block(args, cfg) == 36000


# ── R2 live backup of the eval-pool bucket ───────────────────────────────────
#
# The pool store gains the same S3MirrorStore treatment the manifest bucket
# already has: dual-write every snapshot + pool/index.json to R2, and fall back
# to R2 on a Hippius 5xx — so an outage no longer strands the pool-pin gate as
# ``pool_pin_unverifiable``. Endpoint/region/creds are REUSED from the manifest
# backup R2 (BACKUP_S3_*); only the bucket differs. Hippius stays primary.


def _pool_storage(**over):
    base = dict(
        pool_bucket="cascade-eval-pool", pool_s3_endpoint="", pool_s3_region="",
        s3_endpoint="https://s3.hippius.com", s3_region="decentralized",
        pool_backup_bucket="", backup_s3_endpoint="", backup_s3_region="",
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_pool_backup_store_none_when_unconfigured():
    # No pool_backup_bucket ⇒ no mirror, even when the manifest backup endpoint
    # is already set: the pool mirror is keyed off its OWN bucket, so it stays
    # off until explicitly named (flipping it on is a one-line config change).
    assert pool_backup_s3_store(_pool_storage(), bucket="cascade-eval-pool") is None
    assert pool_backup_s3_store(
        _pool_storage(backup_s3_endpoint="https://acct.r2.cloudflarestorage.com"),
        bucket="cascade-eval-pool") is None


def test_pool_backup_store_reuses_manifest_r2_creds():
    store = pool_backup_s3_store(
        _pool_storage(
            pool_backup_bucket="cascade-eval-pool",
            backup_s3_endpoint="https://acct.r2.cloudflarestorage.com",
        ),
        bucket="cascade-eval-pool",
    )
    assert isinstance(store, S3Store)
    assert store.cfg.endpoint == "https://acct.r2.cloudflarestorage.com"
    assert store.cfg.region == "auto"                        # R2 default
    assert store.cfg.bucket == "cascade-eval-pool"
    # REUSED from the manifest mirror — NOT a new POOL_* pair
    assert store.cfg.access_key_env == "BACKUP_S3_ACCESS_KEY"
    assert store.cfg.secret_key_env == "BACKUP_S3_SECRET_KEY"


def test_pool_s3_store_plain_when_no_backup(monkeypatch):
    monkeypatch.delenv("POOL_S3_ACCESS_KEY", raising=False)
    assert isinstance(pool_s3_store(_pool_storage()), S3Store)


def test_pool_s3_store_mirrored_when_backup_bucket_set(monkeypatch):
    monkeypatch.delenv("POOL_S3_ACCESS_KEY", raising=False)
    store = pool_s3_store(_pool_storage(
        pool_backup_bucket="cascade-eval-pool",
        backup_s3_endpoint="https://acct.r2.cloudflarestorage.com",
    ))
    assert isinstance(store, S3MirrorStore)
    assert isinstance(store.primary, S3Store)
    assert store.primary.cfg.endpoint == "https://s3.hippius.com"        # Hippius primary
    assert store.mirror.cfg.endpoint.endswith("r2.cloudflarestorage.com")  # R2 backup
    assert store.mirror.cfg.bucket == "cascade-eval-pool"


def _pool_mirror(*, primary_fail=False, mirror_fail=False):
    primary = _FakeS3Store(fail=primary_fail)
    r2 = _FakeS3Store(fail=mirror_fail)
    return S3MirrorStore(primary, r2), primary, r2


def test_pool_publish_dual_writes_snapshot_and_index_to_r2(tmp_path):
    s, _primary, r2 = _pool_mirror()
    meta = publish_pool_snapshot(
        s, _make_pool_tar(tmp_path, "p", n=4), effective_block=7200,
        as_of="d", n_series=4, context_length=128, horizon=16,
    )
    # both the snapshot tar AND pool/index.json land on the R2 backup...
    assert meta.key in r2.objects
    assert hippius.POOL_INDEX_KEY in r2.objects
    # ...so a validator could read the whole index straight from R2 alone
    assert [m.effective_block for m in read_pool_index(r2)] == [7200]


def test_pool_read_falls_back_to_r2_on_hippius_outage(tmp_path):
    # Publish while healthy so R2 holds a complete copy...
    s, primary, _r2 = _pool_mirror()
    publish_pool_snapshot(s, _make_pool_tar(tmp_path, "p", n=6), effective_block=7200,
                          as_of="d", n_series=6, context_length=128, horizon=16)
    # ...then Hippius 5xx's: the validator still reads the index and fetches the
    # verified tar off R2 — no pool_pin_unverifiable self-exclusion.
    primary.fail = True
    index = read_pool_index(s)
    assert [m.effective_block for m in index] == [7200]
    out = fetch_pool_snapshot(s, index[0], tmp_path / "restored")
    assert len(list(out.glob("*.npy"))) == 6


def test_pool_bootstrap_absent_through_mirror_reads_empty():
    # No pool published yet: the index is absent on BOTH primary and R2. The
    # mirror must preserve ObjectNotFound so read_pool_index sees genuine
    # absence (→ []) rather than a spurious read error that rejects every round.
    s, _primary, _r2 = _pool_mirror()
    assert read_pool_index(s) == []


def test_pool_mirror_recovers_index_after_primary_data_loss(tmp_path):
    # The data-loss case the dual-write exists for: the primary lost the index
    # object but R2 still holds it. It is served from the mirror, not reported
    # absent (ObjectNotFound from the primary alone must not short-circuit).
    s, primary, _r2 = _pool_mirror()
    publish_pool_snapshot(s, _make_pool_tar(tmp_path, "p"), effective_block=7200,
                          as_of="d", n_series=5, context_length=128, horizon=16)
    del primary.objects[hippius.POOL_INDEX_KEY]          # primary drops the index
    assert [m.effective_block for m in read_pool_index(s)] == [7200]  # recovered from R2


def test_pool_backup_write_failure_is_not_fatal(tmp_path):
    # An R2-only failure must never break publishing — Hippius stays primary and
    # the pool backup is a backup, not a hard dependency.
    s, primary, _r2 = _pool_mirror(mirror_fail=True)
    meta = publish_pool_snapshot(s, _make_pool_tar(tmp_path, "p"), effective_block=7200,
                                 as_of="d", n_series=5, context_length=128, horizon=16)
    assert meta.key in primary.objects                    # primary copy intact
    assert hippius.POOL_INDEX_KEY in primary.objects
