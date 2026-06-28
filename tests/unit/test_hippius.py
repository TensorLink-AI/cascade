"""Hippius storage helpers — pure parts (tar packing, CID grammar, S3 manifest +
log layout over a fake S3 client). No real IPFS node / boto3 endpoint needed."""

from __future__ import annotations

import pytest

from metronome.shared import hippius


def test_is_cid_accepts_v0_and_v1_rejects_garbage():
    assert hippius.is_cid("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG")
    assert hippius.is_cid("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")
    for bad in ("", "Qmshort", "not-a-cid", "QmYwAP!Jzv5", "metro-v1:gen"):
        assert not hippius.is_cid(bad)


def test_pack_dir_is_deterministic_and_round_trips(tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "config.json").write_text('{"a": 1}')
    sub = src / "nested"
    sub.mkdir()
    (sub / "weights.bin").write_bytes(b"\x00\x01\x02\x03")

    blob1 = hippius.pack_dir_to_tar(src)
    blob2 = hippius.pack_dir_to_tar(src)
    assert blob1 == blob2  # reproducible ⇒ stable CID
    assert hippius.tar_cid_digest(blob1) == hippius.tar_cid_digest(blob2)

    out = hippius.unpack_tar_to_dir(blob1, tmp_path / "restored")
    assert (out / "config.json").read_text() == '{"a": 1}'
    assert (out / "nested" / "weights.bin").read_bytes() == b"\x00\x01\x02\x03"


def test_unpack_rejects_path_traversal(tmp_path):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        data = b"x"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(hippius.StorageError):
        hippius.unpack_tar_to_dir(buf.getvalue(), tmp_path / "dest")


class _FakeS3Store:
    """In-memory stand-in for S3Store (same put_text/get_text surface)."""

    def __init__(self):
        self.objects: dict[str, str] = {}

    def put_text(self, key, text, *, content_type="text/plain"):
        self.objects[key] = text

    def get_text(self, key):
        return self.objects[key]


def test_publish_and_read_latest_manifest():
    store = _FakeS3Store()
    key = hippius.publish_manifest(store, '{"round_id":"42"}', "42")
    assert key == "manifests/round-42.json"
    assert store.objects[key] == '{"round_id":"42"}'
    assert hippius.read_latest_manifest(store) == '{"round_id":"42"}'


def test_log_sink_accumulates_and_flushes_jsonl():
    store = _FakeS3Store()
    sink = hippius.LogSink(store, round_id="7", role="king")
    assert sink.flush() is None  # nothing buffered yet
    sink.emit({"step": 1, "loss": 0.5})
    sink.emit({"step": 2, "loss": 0.4})
    key = sink.flush()
    assert key == "logs/round-7/king.jsonl"
    lines = store.objects[key].strip().split("\n")
    assert len(lines) == 2
    assert '"loss":0.5' in lines[0] and '"step":2' in lines[1]
