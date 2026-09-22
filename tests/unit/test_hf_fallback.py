"""HFFallbackStore — Hippius S3 primary with a HuggingFace fallback that engages
only when S3 is down. Happy path never touches HF; an S3 outage transparently
reads/writes the HF mirror."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.shared.hippius import (
    HFFallbackStore,
    ObjectNotFound,
    S3Store,
    StorageError,
    open_manifest_store,
)


class _FakeS3:
    """S3Store stand-in whose ops can be flipped to raise (simulate an outage)."""

    def __init__(self, up=True):
        self.up = up
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, key, data, *, content_type="application/octet-stream", acl=None):
        if not self.up:
            raise StorageError(f"s3_put_failed: {key}: 500")
        self.objects[key] = data

    def get_bytes(self, key):
        if not self.up:
            raise StorageError(f"s3_get_failed: {key}: 500")
        if key not in self.objects:
            raise StorageError(f"s3_get_failed: {key}: NoSuchKey")
        return self.objects[key]


class _FakeHFApi:
    """Records uploads so the get path can serve them; counts calls."""

    def __init__(self, store):
        self.store = store
        self.uploads = 0

    def create_repo(self, *a, **k):
        pass

    def upload_file(self, *, path_or_fileobj, path_in_repo, **k):
        self.uploads += 1
        self.store[path_in_repo] = path_or_fileobj.read()


def _store(primary_up=True):
    s = HFFallbackStore(_FakeS3(up=primary_up), "acct/cascade-mirror")
    hf_objs: dict[str, bytes] = {}
    api = _FakeHFApi(hf_objs)
    s._api = api
    s._ensured = True
    # patch the download to read from the fake HF object store
    s._hf_get = lambda key: hf_objs[key]
    return s, api, hf_objs


def test_happy_path_never_touches_hf():
    s, api, hf = _store(primary_up=True)
    s.put_text("manifests/latest.json", '{"round":1}')
    assert s.get_text("manifests/latest.json") == '{"round":1}'
    assert api.uploads == 0 and hf == {}          # zero HF traffic when S3 is up


def test_write_falls_back_to_hf_when_s3_down():
    s, api, hf = _store(primary_up=False)
    s.put_text("manifests/round-7.json", "payload")
    assert api.uploads == 1
    assert hf["manifests/round-7.json"] == b"payload"    # not lost — landed on HF


def test_read_falls_back_to_hf_when_s3_down():
    s, api, hf = _store(primary_up=False)
    hf["receipts/latest.json"] = b'{"round":"9"}'         # mirror has it
    assert s.get_text("receipts/latest.json") == '{"round":"9"}'


def test_outage_round_trip_end_to_end():
    # trainer writes during the outage → validator reads it back, all via HF
    s, api, hf = _store(primary_up=False)
    s.put_text("manifests/latest.json", "the-manifest")
    assert s.get_text("manifests/latest.json") == "the-manifest"


def test_both_down_raises():
    s, api, hf = _store(primary_up=False)
    def boom(key):
        raise RuntimeError("hf 503")
    s._hf_get = boom
    with pytest.raises(StorageError, match="both S3 and HF"):
        s.get_text("manifests/latest.json")


def test_factory_plain_s3_when_unconfigured():
    storage = SimpleNamespace(manifest_bucket="cascade-manifests", hf_backup_repo="",
                              s3_endpoint="https://s3.hippius.com", s3_region="decentralized")
    assert isinstance(open_manifest_store(storage), S3Store)


def test_factory_hf_backed_when_configured():
    storage = SimpleNamespace(manifest_bucket="cascade-manifests",
                              hf_backup_repo="acct/mirror",
                              s3_endpoint="https://s3.hippius.com", s3_region="decentralized")
    store = open_manifest_store(storage)
    assert isinstance(store, HFFallbackStore) and store.hf_repo == "acct/mirror"


class _MissingS3:
    """Primary that is UP and says the key does not exist."""

    def get_bytes(self, key):
        raise ObjectNotFound(f"s3_get_failed: {key}: NoSuchKey")


class _HFMissing(Exception):
    response = SimpleNamespace(status_code=404)


def test_absent_on_both_layers_reads_as_object_not_found():
    # Absence is a verdict; callers separate it from an outage (retry).
    s = HFFallbackStore(_MissingS3(), "acct/cascade-mirror")

    def hf_missing(key):
        raise _HFMissing("404 Client Error: Entry Not Found")

    s._hf_get = hf_missing
    with pytest.raises(ObjectNotFound):
        s.get_bytes("benchmarks/round-1.json")
    # Primary absent but HF unreachable: existence unknown → plain StorageError.
    def hf_boom(key):
        raise RuntimeError("timeout")

    s._hf_get = hf_boom
    with pytest.raises(StorageError) as ei:
        s.get_bytes("benchmarks/round-1.json")
    assert not isinstance(ei.value, ObjectNotFound)
    # Primary DOWN (not absent) + HF 404: S3 may still hold it → not absence.
    s2 = HFFallbackStore(_FakeS3(up=False), "acct/cascade-mirror")
    s2._hf_get = hf_missing
    with pytest.raises(StorageError) as ei:
        s2.get_bytes("benchmarks/round-1.json")
    assert not isinstance(ei.value, ObjectNotFound)
