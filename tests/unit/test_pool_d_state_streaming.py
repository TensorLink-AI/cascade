"""pool_d store streaming: multipart upload, streamed download, object size.

``cascade-laned elastic state-push`` failed on the pool_d builder with
``EntityTooLarge``: the 20 GB ``first_seen/records.jsonl`` went up as one
``PutObject`` (and was read whole into memory first). The stores now carry
``put_file_sha`` / ``get_file_sha`` / ``object_size``; this file pins them
against a fake S3 client (no network, no boto3 transfer manager, no torch).
"""

from __future__ import annotations

import hashlib
import io
import os

import pytest

from cascade.pool import pool_d_publish as pub
from cascade.shared.hippius import ObjectNotFound, S3Config, StorageError

KIB = 1024
CHUNK = 64 * KIB


class _Missing(Exception):
    def __init__(self, key):
        super().__init__(f"NoSuchKey: {key}")
        self.response = {"Error": {"Code": "NoSuchKey"},
                         "ResponseMetadata": {"HTTPStatusCode": 404}}


class _Body:
    """A streaming body that records every read size (``read()`` = -1)."""

    def __init__(self, data: bytes, reads: list, *, cut: int | None = None):
        self._f = io.BytesIO(data if cut is None else data[:cut])
        self._reads = reads

    def read(self, n: int = -1) -> bytes:
        self._reads.append(n)
        return self._f.read(n)


class FakeS3:
    """The boto3 S3 client subset the pool store uses."""

    def __init__(self, *, max_put: int | None = None):
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict] = {}
        self.max_put = max_put
        self.calls: list[tuple[str, str]] = []
        self.bodies: list[int] = []
        self.reads: list[int] = []
        self.aborted: list[str] = []
        self.fail_part: int | None = None
        self.fail_complete = False
        self.on_part = None
        self.cut: int | None = None

    def put_object(self, Bucket, Key, Body, ContentType=None):
        data = Body if isinstance(Body, bytes) else Body.read()
        self.calls.append(("put_object", Key))
        self.bodies.append(len(data))
        if self.max_put is not None and len(data) > self.max_put:
            raise RuntimeError("An error occurred (EntityTooLarge) when calling the PutObject "
                               "operation: Your proposed upload exceeds the maximum allowed size")
        self.objects[Key] = bytes(data)

    def get_object(self, Bucket, Key):
        self.calls.append(("get_object", Key))
        if Key not in self.objects:
            raise _Missing(Key)
        data = self.objects[Key]
        return {"Body": _Body(data, self.reads, cut=self.cut), "ContentLength": len(data)}

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", Key))
        if Key not in self.objects:
            raise _Missing(Key)
        return {"ContentLength": len(self.objects[Key])}

    def create_multipart_upload(self, Bucket, Key, ContentType=None):
        self.calls.append(("create_multipart_upload", Key))
        uid = f"up{len(self.uploads) + len(self.aborted)}"
        self.uploads[uid] = {"key": Key, "parts": {}}
        return {"UploadId": uid}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        self.calls.append(("upload_part", Key))
        self.bodies.append(len(Body))
        if self.on_part is not None:
            self.on_part(PartNumber)
        if self.fail_part == PartNumber:
            raise RuntimeError("connection reset by peer")
        self.uploads[UploadId]["parts"][PartNumber] = bytes(Body)
        return {"ETag": hashlib.md5(Body).hexdigest()}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        self.calls.append(("complete_multipart_upload", Key))
        if self.fail_complete:
            raise RuntimeError("InternalError")
        up = self.uploads.pop(UploadId)
        nums = [p["PartNumber"] for p in MultipartUpload["Parts"]]
        assert nums == list(range(1, len(nums) + 1))
        self.objects[Key] = b"".join(up["parts"][n] for n in nums)

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.calls.append(("abort_multipart_upload", Key))
        self.uploads.pop(UploadId, None)
        self.aborted.append(UploadId)

    def ops(self, name: str) -> int:
        return sum(1 for op, _ in self.calls if op == name)


def _store(fake: FakeS3) -> pub.HippiusPoolStore:
    return pub.HippiusPoolStore(S3Config(endpoint="https://s3.example", region="r",
                                         bucket="b"), _client=fake)


@pytest.fixture
def small_parts(monkeypatch):
    monkeypatch.setattr(pub, "MULTIPART_CHUNK_BYTES", CHUNK)
    monkeypatch.setattr(pub, "STREAM_CHUNK_BYTES", 4 * KIB)


def _file(tmp_path, n: int, name: str = "f.bin"):
    p = tmp_path / name
    p.write_bytes(os.urandom(n))
    return p


def test_part_size_stays_inside_the_s3_part_limit():
    gib = 1 << 30
    assert pub.MAX_MULTIPART_PARTS == 10_000
    assert pub.multipart_part_size(0) == pub.MULTIPART_CHUNK_BYTES
    assert pub.multipart_part_size(20 * gib) == pub.MULTIPART_CHUNK_BYTES   # 80 parts
    for size in (20 * gib, 3 * 1024 * gib, 5 * 1024 * gib, 10 * 1024 * gib + 7):
        part = pub.multipart_part_size(size)
        assert part % (1 << 20) == 0 and part >= pub.MULTIPART_CHUNK_BYTES
        assert -(-size // part) <= pub.MAX_MULTIPART_PARTS


def test_small_file_is_one_put(tmp_path, small_parts):
    fake = FakeS3(max_put=CHUNK)
    p = _file(tmp_path, CHUNK)                       # exactly the threshold
    got = _store(fake).put_file_sha("k", p)
    assert got == {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": CHUNK,
                   "parts": 0}
    assert [op for op, _ in fake.calls] == ["put_object"]
    assert fake.objects["k"] == p.read_bytes()


def test_put_bytes_path_unchanged(small_parts):
    fake = FakeS3()
    _store(fake).put_bytes("pool_d/index.json", b"{}")
    assert fake.calls == [("put_object", "pool_d/index.json")]
    assert fake.objects["pool_d/index.json"] == b"{}"


def test_large_file_goes_multipart_one_part_in_memory(tmp_path, small_parts):
    fake = FakeS3(max_put=CHUNK)
    n = 5 * CHUNK + 123
    p = _file(tmp_path, n)
    got = _store(fake).put_file_sha("big", p)
    data = p.read_bytes()
    assert got == {"sha256": hashlib.sha256(data).hexdigest(), "size": n, "parts": 6}
    assert fake.objects["big"] == data
    assert fake.ops("put_object") == 0 and fake.ops("upload_part") == 6
    assert fake.ops("complete_multipart_upload") == 1 and not fake.aborted
    assert max(fake.bodies) <= CHUNK and fake.uploads == {}


def test_part_size_grows_past_the_part_limit(tmp_path, small_parts, monkeypatch):
    monkeypatch.setattr(pub, "MAX_MULTIPART_PARTS", 3)
    fake = FakeS3()
    p = _file(tmp_path, 3 * (1 << 20) + 1)           # needs > 1 MiB parts for 3 parts
    got = _store(fake).put_file_sha("big", p)
    assert got["parts"] <= 3 and fake.objects["big"] == p.read_bytes()


@pytest.mark.parametrize("where", ["part", "complete"])
def test_failure_aborts_the_multipart_upload(tmp_path, small_parts, where):
    fake = FakeS3()
    fake.objects["big"] = b"previous"
    if where == "part":
        fake.fail_part = 3
    else:
        fake.fail_complete = True
    p = _file(tmp_path, 4 * CHUNK)
    with pytest.raises(StorageError) as ei:
        _store(fake).put_file_sha("big", p)
    assert "s3_put_failed: big" in str(ei.value) and "aborted" in str(ei.value)
    assert fake.aborted == ["up0"] and fake.uploads == {}
    assert fake.objects["big"] == b"previous"        # the old object is untouched
    if where == "part":
        assert fake.ops("complete_multipart_upload") == 0


def test_file_that_grows_mid_upload_is_aborted(tmp_path, small_parts):
    fake = FakeS3()
    p = _file(tmp_path, 3 * CHUNK)

    def grow(num):
        if num == 1:
            with open(p, "ab") as f:
                f.write(b"x" * 10)

    fake.on_part = grow
    with pytest.raises(StorageError, match="changed during upload"):
        _store(fake).put_file_sha("big", p)
    assert fake.aborted == ["up0"] and "big" not in fake.objects


def test_get_file_sha_streams_and_checks_length(tmp_path, small_parts):
    fake = FakeS3()
    data = os.urandom(3 * CHUNK + 5)
    fake.objects["k"] = data
    store = _store(fake)
    got = store.get_file_sha("k", tmp_path / "out" / "k.bin")
    assert got == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    assert (tmp_path / "out" / "k.bin").read_bytes() == data
    assert fake.reads and all(0 < r <= pub.STREAM_CHUNK_BYTES for r in fake.reads)
    with pytest.raises(ObjectNotFound):
        store.get_file_sha("nope", tmp_path / "x")
    fake.cut = 100
    with pytest.raises(StorageError, match="short read"):
        store.get_file_sha("k", tmp_path / "y")


def test_object_size(small_parts):
    fake = FakeS3()
    fake.objects["k"] = b"abc"
    store = _store(fake)
    assert store.object_size("k") == 3
    with pytest.raises(ObjectNotFound):
        store.object_size("nope")


def test_fs_store_streams_the_same_contract(tmp_path, small_parts):
    store = pub.FsPoolStore(tmp_path / "bucket")
    p = _file(tmp_path, 2 * CHUNK + 1)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert store.put_file_sha("a/b", p) == {"sha256": sha, "size": 2 * CHUNK + 1, "parts": 0}
    assert store.object_size("a/b") == 2 * CHUNK + 1
    assert store.get_file_sha("a/b", tmp_path / "back") == {"sha256": sha,
                                                            "size": 2 * CHUNK + 1}
    assert (tmp_path / "back").read_bytes() == p.read_bytes()
    with pytest.raises(ObjectNotFound):
        store.object_size("a/c")
    with pytest.raises(ObjectNotFound):
        store.get_file_sha("a/c", tmp_path / "x")
