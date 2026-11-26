"""W-LAUNCH pool_d builder state-push (2026-11-26): the first-seen log streams.

Every 3-hourly build cycle exited non-zero at ``cascade-laned elastic
state-push``: ``StorageError: s3_put_failed: pool_d/_builder/first_seen/
records.jsonl: ... (EntityTooLarge) ... PutObject``. The log is 20 GB (7,002
feeds) and grows on every build; the push read it (and the bucket copy) whole
and sent it as one PUT, which Hippius refuses above its single-PUT cap.

Contract, against a fake S3 client whose single PUT is capped:

* the log goes up as a multipart upload above one part, never one body larger
  than a part; a failed part aborts the upload and leaves the bucket copy and
  ``HEAD.json`` as they were, so the next push retries;
* the append-only check never downloads the bucket copy: HEAD's sha256 is
  compared with the sha256 of the local file's first ``size`` bytes;
* ``HEAD.json`` keeps ``{name: sha256}`` and gains ``sizes``; a file equal to
  the bucket copy is not re-uploaded;
* ``state-pull`` streams (bounded reads), verifies sha256 and size against
  HEAD before replacing anything, and keeps a warm local copy equal to HEAD;
* a HEAD and pull mark from before ``sizes`` still push.

No torch, no network.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import pytest

KIB = 1024
PART = 64 * KIB
RECORDS = "pool_d/_builder/first_seen/records.jsonl"
HEAD = "pool_d/_builder/first_seen/HEAD.json"


class _Missing(Exception):
    def __init__(self, key):
        super().__init__(f"NoSuchKey: {key}")
        self.response = {"Error": {"Code": "NoSuchKey"},
                         "ResponseMetadata": {"HTTPStatusCode": 404}}


class _Body:
    def __init__(self, key, data, reads):
        self._key, self._f, self._reads = key, io.BytesIO(data), reads

    def read(self, n=-1):
        self._reads.append((self._key, n))
        return self._f.read(n)


class FakeS3:
    """Hippius-like: a single PUT above ``max_put`` is EntityTooLarge."""

    def __init__(self, max_put=PART):
        self.objects, self.uploads, self.calls = {}, {}, []
        self.bodies, self.reads, self.aborted = [], [], []
        self.max_put, self.fail_part = max_put, None

    def reset(self):
        self.calls, self.bodies, self.reads = [], [], []

    def put_object(self, Bucket, Key, Body, ContentType=None):
        data = Body if isinstance(Body, bytes) else Body.read()
        self.calls.append(("put_object", Key))
        self.bodies.append(len(data))
        if len(data) > self.max_put:
            raise RuntimeError("An error occurred (EntityTooLarge) when calling the PutObject "
                               "operation: Your proposed upload exceeds the maximum allowed size")
        self.objects[Key] = bytes(data)

    def get_object(self, Bucket, Key):
        self.calls.append(("get_object", Key))
        if Key not in self.objects:
            raise _Missing(Key)
        return {"Body": _Body(Key, self.objects[Key], self.reads),
                "ContentLength": len(self.objects[Key])}

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", Key))
        if Key not in self.objects:
            raise _Missing(Key)
        return {"ContentLength": len(self.objects[Key])}

    def create_multipart_upload(self, Bucket, Key, ContentType=None):
        self.calls.append(("create_multipart_upload", Key))
        uid = f"up{len(self.uploads) + len(self.aborted)}"
        self.uploads[uid] = {}
        return {"UploadId": uid}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        self.calls.append(("upload_part", Key))
        self.bodies.append(len(Body))
        if self.fail_part == PartNumber:
            raise RuntimeError("connection reset by peer")
        self.uploads[UploadId][PartNumber] = bytes(Body)
        return {"ETag": hashlib.md5(Body).hexdigest()}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        self.calls.append(("complete_multipart_upload", Key))
        parts = self.uploads.pop(UploadId)
        self.objects[Key] = b"".join(parts[p["PartNumber"]] for p in MultipartUpload["Parts"])

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.calls.append(("abort_multipart_upload", Key))
        self.uploads.pop(UploadId, None)
        self.aborted.append(UploadId)

    def ops(self, name, key=None):
        return sum(1 for op, k in self.calls if op == name and (key is None or k == key))


@pytest.fixture
def env(monkeypatch):
    import importlib

    pub = importlib.import_module("cascade.pool.pool_d_publish")
    m = importlib.import_module("cascade.laned.elastic")
    from cascade.shared.hippius import S3Config

    monkeypatch.setattr(pub, "MULTIPART_CHUNK_BYTES", PART)
    monkeypatch.setattr(pub, "STREAM_CHUNK_BYTES", 4 * KIB, raising=False)
    fake = FakeS3()
    store = pub.HippiusPoolStore(S3Config(endpoint="https://s3.example", region="r",
                                          bucket="pool-d"), _client=fake)
    return m, store, fake


def _push(m, store, log_dir):
    from cascade.shared.hippius import StorageError

    try:
        return m.state_push(store, prefix="pool_d", log_dir=log_dir, pool_d=None)
    except StorageError as e:
        raise AssertionError(f"state-push failed: {e}") from e


def _pull(m, store, log_dir, tmp_path):
    return m.state_pull(store, prefix="pool_d", log_dir=log_dir,
                        prior_root=tmp_path / "prior")


def _lines(n_bytes):
    out = bytearray()
    i = 0
    while len(out) < n_bytes:
        out += json.dumps({"feed": f"f{i % 7:03d}", "pad": os.urandom(24).hex()}).encode() + b"\n"
        i += 1
    return bytes(out)


def _head(fake):
    return json.loads(fake.objects[HEAD])


def test_log_above_the_single_put_cap_goes_multipart(env, tmp_path):
    m, store, fake = env
    a = tmp_path / "vm1" / "first_seen"
    _pull(m, store, a, tmp_path)
    data = _lines(6 * PART + 100)
    (a / "records.jsonl").write_bytes(data)
    _push(m, store, a)
    assert fake.objects[RECORDS] == data
    assert fake.ops("put_object", RECORDS) == 0
    assert fake.ops("upload_part", RECORDS) == 7 and fake.ops("complete_multipart_upload") == 1
    assert max(fake.bodies) <= PART and not fake.aborted
    head = _head(fake)
    assert head["records.jsonl"] == hashlib.sha256(data).hexdigest()
    assert head["sizes"] == {"records.jsonl": len(data), "roots.jsonl": 0,
                             "quarantine.jsonl": 0}


def test_push_never_downloads_the_bucket_copy_and_skips_unchanged(env, tmp_path):
    m, store, fake = env
    a = tmp_path / "vm1" / "first_seen"
    _pull(m, store, a, tmp_path)
    first = _lines(3 * PART)
    (a / "records.jsonl").write_bytes(first)
    (a / "roots.jsonl").write_bytes(b'{"p":1}\n')
    _push(m, store, a)
    b = tmp_path / "vm2" / "first_seen"
    _pull(m, store, b, tmp_path)
    grown = first + _lines(PART)
    (b / "records.jsonl").write_bytes(grown)
    fake.reset()
    out = _push(m, store, b)
    assert fake.ops("get_object", RECORDS) == 0          # prefix proven from HEAD's sha256
    assert fake.objects[RECORDS] == grown
    assert out["uploaded"] == ["records.jsonl"]
    assert "roots.jsonl" in out["skipped"]
    fake.reset()
    out = _push(m, store, b)                              # nothing changed
    assert "records.jsonl" in out["skipped"] and fake.ops("upload_part") == 0
    assert fake.ops("put_object", RECORDS) == 0


def test_rewritten_prefix_is_refused_without_download(env, tmp_path):
    m, store, fake = env
    a = tmp_path / "vm1" / "first_seen"
    _pull(m, store, a, tmp_path)
    data = _lines(2 * PART)
    (a / "records.jsonl").write_bytes(data)
    _push(m, store, a)
    (a / "records.jsonl").write_bytes(b"X" + data[1:] + _lines(100))
    fake.reset()
    with pytest.raises(m.ElasticError) as ei:
        m.state_push(store, prefix="pool_d", log_dir=a, pool_d=None)
    assert ei.value.code == "state_conflict"
    assert fake.ops("get_object", RECORDS) == 0 and fake.objects[RECORDS] == data


def test_failed_part_aborts_and_keeps_the_previous_state(env, tmp_path):
    m, store, fake = env
    from cascade.shared.hippius import StorageError

    a = tmp_path / "vm1" / "first_seen"
    _pull(m, store, a, tmp_path)
    small = _lines(PART // 2)
    (a / "records.jsonl").write_bytes(small)
    _push(m, store, a)
    head_before = fake.objects[HEAD]
    (a / "records.jsonl").write_bytes(small + _lines(4 * PART))
    fake.fail_part = 2
    with pytest.raises(StorageError):
        m.state_push(store, prefix="pool_d", log_dir=a, pool_d=None)
    assert fake.aborted == ["up0"] and fake.uploads == {}
    assert fake.objects[RECORDS] == small and fake.objects[HEAD] == head_before
    fake.fail_part = None
    _push(m, store, a)                                    # the next cycle retries
    assert fake.objects[RECORDS] == (a / "records.jsonl").read_bytes()


def test_restore_streams_and_verifies(env, tmp_path):
    m, store, fake = env
    a = tmp_path / "vm1" / "first_seen"
    _pull(m, store, a, tmp_path)
    data = _lines(5 * PART)
    (a / "records.jsonl").write_bytes(data)
    _push(m, store, a)
    fake.reset()
    b = tmp_path / "vm2" / "first_seen"
    _pull(m, store, b, tmp_path)
    assert (b / "records.jsonl").read_bytes() == data
    reads = [n for k, n in fake.reads if k == RECORDS]
    assert reads and all(0 < n <= 4 * KIB for n in reads)          # never read() whole
    # same size, different bytes: refused, nothing replaced, no temp left behind
    fake.objects[RECORDS] = b"Y" + data[1:]
    c = tmp_path / "vm3" / "first_seen"
    with pytest.raises(m.ElasticError) as ei:
        _pull(m, store, c, tmp_path)
    assert ei.value.code == "state_conflict"
    assert not (c / "records.jsonl").exists()
    assert [p.name for p in c.iterdir() if p.name.endswith(".tmp")] == []


def test_warm_builder_equal_to_head_is_not_downloaded(env, tmp_path):
    m, store, fake = env
    a = tmp_path / "vm1" / "first_seen"
    _pull(m, store, a, tmp_path)
    (a / "records.jsonl").write_bytes(_lines(3 * PART))
    _push(m, store, a)
    fake.reset()
    _pull(m, store, a, tmp_path)
    assert fake.ops("get_object", RECORDS) == 0
    _push(m, store, a)
    assert fake.ops("upload_part") == 0


def test_head_and_mark_from_before_sizes_still_push(env, tmp_path):
    m, store, fake = env
    old = _lines(PART // 2)
    fake.objects[RECORDS] = old
    fake.objects["pool_d/_builder/first_seen/roots.jsonl"] = b""
    head = {"records.jsonl": hashlib.sha256(old).hexdigest(),
            "roots.jsonl": hashlib.sha256(b"").hexdigest()}
    fake.objects[HEAD] = json.dumps(head, sort_keys=True).encode()
    a = tmp_path / "vm" / "first_seen"
    a.mkdir(parents=True)
    (a / "records.jsonl").write_bytes(old + _lines(3 * PART))
    (a / ".elastic_pulled.json").write_text(json.dumps(head, sort_keys=True))   # legacy mark
    out = _push(m, store, a)
    assert fake.objects[RECORDS] == (a / "records.jsonl").read_bytes()
    assert _head(fake)["sizes"]["records.jsonl"] == len(fake.objects[RECORDS])
    assert out["head"]["records.jsonl"] == hashlib.sha256(fake.objects[RECORDS]).hexdigest()
