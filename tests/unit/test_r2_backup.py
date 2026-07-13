"""S3MirrorStore — Hippius S3 primary with a Cloudflare R2 (S3-compatible) backup
that receives a copy of EVERY write. Unlike the HF failover, the happy path
dual-writes both stores, so R2 always holds a complete copy; reads fall back to
R2 only when the primary is down."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.shared.hippius import (
    HFFallbackStore,
    S3MirrorStore,
    S3Store,
    StorageError,
    backup_s3_store,
    open_manifest_store,
)


class _FakeS3:
    """S3Store stand-in whose ops can be flipped to raise (simulate an outage),
    optionally rejecting canned ACLs the way R2 does."""

    def __init__(self, up=True, *, reject_acl=False, bucket="fake"):
        self.up = up
        self.reject_acl = reject_acl
        self.objects: dict[str, bytes] = {}
        self.cfg = SimpleNamespace(bucket=bucket)

    def put_bytes(self, key, data, *, content_type="application/octet-stream", acl=None):
        if not self.up:
            raise StorageError(f"s3_put_failed: {key}: 500")
        if acl and self.reject_acl:
            raise StorageError(f"s3_put_failed: {key}: ACL unsupported")
        self.objects[key] = data

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type, acl=acl)

    def get_bytes(self, key):
        if not self.up:
            raise StorageError(f"s3_get_failed: {key}: 500")
        if key not in self.objects:
            raise StorageError(f"s3_get_failed: {key}: NoSuchKey")
        return self.objects[key]

    def get_text(self, key):
        return self.get_bytes(key).decode("utf-8")


def _mirror(primary_up=True, mirror_up=True, *, reject_acl=False):
    primary = _FakeS3(up=primary_up, bucket="cascade-manifests")
    r2 = _FakeS3(up=mirror_up, reject_acl=reject_acl, bucket="cascade-r2-backup")
    return S3MirrorStore(primary, r2), primary, r2


def test_happy_path_writes_both_stores():
    s, primary, r2 = _mirror()
    s.put_text("manifests/latest.json", '{"round":1}')
    # dual-write: the object lands on BOTH the primary and the R2 backup
    assert primary.objects["manifests/latest.json"] == b'{"round":1}'
    assert r2.objects["manifests/latest.json"] == b'{"round":1}'


def test_read_prefers_primary():
    s, primary, r2 = _mirror()
    primary.objects["k"] = b"from-primary"
    r2.objects["k"] = b"from-r2"
    assert s.get_bytes("k") == b"from-primary"


def test_read_falls_back_to_r2_when_primary_down():
    s, primary, r2 = _mirror(primary_up=False)
    r2.objects["receipts/latest.json"] = b'{"round":"9"}'
    assert s.get_text("receipts/latest.json") == '{"round":"9"}'


def test_write_still_lands_on_r2_when_primary_down():
    s, primary, r2 = _mirror(primary_up=False)
    s.put_text("manifests/round-7.json", "payload")
    # not lost — the object is on R2 even though the primary put failed
    assert r2.objects["manifests/round-7.json"] == b"payload"


def test_r2_backup_failure_is_not_fatal_when_primary_ok():
    # a backup outage must never break the round loop
    s, primary, r2 = _mirror(mirror_up=False)
    s.put_text("manifests/latest.json", "ok")
    assert primary.objects["manifests/latest.json"] == b"ok"  # primary copy intact
    assert r2.objects == {}                                   # backup skipped, no raise


def test_both_down_raises():
    s, primary, r2 = _mirror(primary_up=False, mirror_up=False)
    with pytest.raises(StorageError, match="both primary and R2 put"):
        s.put_text("manifests/latest.json", "x")


def test_both_down_read_raises():
    s, primary, r2 = _mirror(primary_up=False, mirror_up=False)
    with pytest.raises(StorageError, match="both primary and R2 get"):
        s.get_text("manifests/latest.json")


def test_public_read_acl_retried_without_acl_on_r2():
    # R2 rejects canned object ACLs; the mirror write retries without one so the
    # backup bytes still land (the primary keeps the public-read receipt).
    s, primary, r2 = _mirror(reject_acl=True)
    s.put_text("receipts/latest.json", "signed", acl="public-read")
    assert primary.objects["receipts/latest.json"] == b"signed"   # ACL honoured here
    assert r2.objects["receipts/latest.json"] == b"signed"        # landed sans ACL


def test_r2_stacks_on_top_of_hf_fallback():
    # primary may itself be an HFFallbackStore — R2 mirror + HF failover coexist
    hf = HFFallbackStore(_FakeS3(bucket="cascade-manifests"), "acct/mirror")
    r2 = _FakeS3(bucket="cascade-r2-backup")
    s = S3MirrorStore(hf, r2)
    s.put_text("manifests/latest.json", "v")
    assert r2.objects["manifests/latest.json"] == b"v"


# ── factory wiring ───────────────────────────────────────────────────────────

def _storage(**over):
    base = dict(
        manifest_bucket="cascade-manifests",
        hf_backup_repo="",
        s3_endpoint="https://s3.hippius.com",
        s3_region="decentralized",
        backup_bucket="",
        backup_s3_endpoint="",
        backup_s3_region="",
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_backup_store_none_when_unconfigured():
    assert backup_s3_store(_storage(), bucket="cascade-manifests") is None


def test_backup_store_built_when_endpoint_set():
    store = backup_s3_store(
        _storage(backup_s3_endpoint="https://acct.r2.cloudflarestorage.com"),
        bucket="cascade-manifests",
    )
    assert isinstance(store, S3Store)
    assert store.cfg.endpoint == "https://acct.r2.cloudflarestorage.com"
    assert store.cfg.region == "auto"                     # R2 default
    assert store.cfg.bucket == "cascade-manifests"        # defaults to primary name
    assert store.cfg.access_key_env == "BACKUP_S3_ACCESS_KEY"
    assert store.cfg.secret_key_env == "BACKUP_S3_SECRET_KEY"


def test_backup_bucket_and_region_override():
    store = backup_s3_store(
        _storage(
            backup_s3_endpoint="https://acct.r2.cloudflarestorage.com",
            backup_bucket="cascade-r2-mirror",
            backup_s3_region="wnam",
        ),
        bucket="cascade-manifests",
    )
    assert store.cfg.bucket == "cascade-r2-mirror"
    assert store.cfg.region == "wnam"


def test_factory_plain_s3_when_unconfigured():
    assert isinstance(open_manifest_store(_storage()), S3Store)


def test_factory_mirror_when_backup_configured():
    store = open_manifest_store(
        _storage(backup_s3_endpoint="https://acct.r2.cloudflarestorage.com")
    )
    assert isinstance(store, S3MirrorStore)
    assert isinstance(store.primary, S3Store)             # no HF ⇒ plain primary
    assert store.mirror.cfg.bucket == "cascade-manifests"


def test_factory_stacks_hf_under_r2_when_both_set():
    store = open_manifest_store(
        _storage(
            hf_backup_repo="acct/mirror",
            backup_s3_endpoint="https://acct.r2.cloudflarestorage.com",
        )
    )
    assert isinstance(store, S3MirrorStore)
    assert isinstance(store.primary, HFFallbackStore)     # HF failover under R2
