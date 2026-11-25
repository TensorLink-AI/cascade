"""R13: builder state on object storage -- first-seen stays append-only."""

from __future__ import annotations

import hashlib
import json

import pytest

from .helpers import mod


def _store(tmp_path):
    from cascade.pool.pool_d_publish import FsPoolStore

    return FsPoolStore(tmp_path / "bucket")


def _snapshot(d, name):
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"snapshot": name, "as_of_ts": 1}))


def test_round_trip_and_append_only(tmp_path):
    m = mod()
    store = _store(tmp_path)
    a = tmp_path / "vm1" / "first_seen"
    m.state_pull(store, prefix="pool_d", log_dir=a, prior_root=tmp_path / "vm1" / "prior")
    (a / "records.jsonl").write_text('{"r":1}\n')
    (a / "roots.jsonl").write_text('{"p":1}\n')
    _snapshot(tmp_path / "vm1" / "out", "pool_d_1")
    m.state_push(store, prefix="pool_d", log_dir=a, pool_d=tmp_path / "vm1" / "out")
    store.put_bytes("pool_d/index.json", json.dumps(
        {"snapshots": [{"snapshot": "pool_d_1", "as_of_ts": 1}]}).encode())

    # a fresh VM restores the log and the prior metadata
    b = tmp_path / "vm2" / "first_seen"
    got = m.state_pull(store, prefix="pool_d", log_dir=b, prior_root=tmp_path / "vm2" / "prior")
    assert (b / "records.jsonl").read_text() == '{"r":1}\n'
    assert got["prior"] == ["pool_d_1"]
    assert (tmp_path / "vm2" / "prior" / "pool_d_1" / "metadata.json").is_file()
    (b / "records.jsonl").write_text('{"r":1}\n{"r":2}\n')
    m.state_push(store, prefix="pool_d", log_dir=b, pool_d=None)
    assert store.get_bytes("pool_d/_builder/first_seen/records.jsonl") == b'{"r":1}\n{"r":2}\n'


def test_rewritten_log_is_refused(tmp_path):
    m = mod()
    store = _store(tmp_path)
    a = tmp_path / "vm" / "first_seen"
    m.state_pull(store, prefix="pool_d", log_dir=a, prior_root=tmp_path / "prior")
    (a / "records.jsonl").write_text('{"r":1}\n')
    m.state_push(store, prefix="pool_d", log_dir=a, pool_d=None)
    (a / "records.jsonl").write_text('{"r":"rewritten"}\n')
    with pytest.raises(m.ElasticError) as ei:
        m.state_push(store, prefix="pool_d", log_dir=a, pool_d=None)
    assert ei.value.code == "state_conflict"
    assert store.get_bytes("pool_d/_builder/first_seen/records.jsonl") == b'{"r":1}\n'


def test_concurrent_builder_is_refused(tmp_path):
    m = mod()
    store = _store(tmp_path)
    a, b = tmp_path / "a" / "first_seen", tmp_path / "b" / "first_seen"
    m.state_pull(store, prefix="pool_d", log_dir=a, prior_root=tmp_path / "pa")
    m.state_pull(store, prefix="pool_d", log_dir=b, prior_root=tmp_path / "pb")
    (a / "records.jsonl").write_text('{"r":1}\n')
    m.state_push(store, prefix="pool_d", log_dir=a, pool_d=None)
    (b / "records.jsonl").write_text('{"r":1}\n{"r":9}\n')
    with pytest.raises(m.ElasticError) as ei:
        m.state_push(store, prefix="pool_d", log_dir=b, pool_d=None)
    assert ei.value.code == "state_conflict"


def test_tampered_bucket_copy_is_refused_on_pull(tmp_path):
    m = mod()
    store = _store(tmp_path)
    a = tmp_path / "a" / "first_seen"
    m.state_pull(store, prefix="pool_d", log_dir=a, prior_root=tmp_path / "pa")
    (a / "records.jsonl").write_text('{"r":1}\n')
    m.state_push(store, prefix="pool_d", log_dir=a, pool_d=None)
    store.put_bytes("pool_d/_builder/first_seen/records.jsonl", b'{"r":"x"}\n')
    with pytest.raises(m.ElasticError) as ei:
        m.state_pull(store, prefix="pool_d", log_dir=tmp_path / "c" / "first_seen",
                     prior_root=tmp_path / "pc")
    assert ei.value.code == "state_conflict"


def test_quarantine_round_trips_and_old_head_pulls_empty(tmp_path):
    """W6fs quarantine.jsonl rides the same append-only state; a HEAD written
    before it existed restores it as empty, and a bucket copy HEAD does not
    name is refused."""
    m = mod()
    assert "quarantine.jsonl" in m.FIRST_SEEN_FILES
    store = _store(tmp_path)
    # a HEAD from before the quarantine file existed
    store.put_bytes("pool_d/_builder/first_seen/records.jsonl", b'{"r":1}\n')
    store.put_bytes("pool_d/_builder/first_seen/roots.jsonl", b"")
    store.put_bytes("pool_d/_builder/first_seen/HEAD.json", json.dumps({
        "records.jsonl": hashlib.sha256(b'{"r":1}\n').hexdigest(),
        "roots.jsonl": hashlib.sha256(b"").hexdigest()}, sort_keys=True).encode())
    a = tmp_path / "a" / "first_seen"
    m.state_pull(store, prefix="pool_d", log_dir=a, prior_root=tmp_path / "pa")
    assert (a / "quarantine.jsonl").read_bytes() == b""
    (a / "quarantine.jsonl").write_text('{"q":1}\n')
    m.state_push(store, prefix="pool_d", log_dir=a, pool_d=None)
    assert store.get_bytes("pool_d/_builder/first_seen/quarantine.jsonl") == b'{"q":1}\n'
    b = tmp_path / "b" / "first_seen"
    m.state_pull(store, prefix="pool_d", log_dir=b, prior_root=tmp_path / "pb")
    assert (b / "quarantine.jsonl").read_text() == '{"q":1}\n'
    # a quarantine copy the HEAD does not name
    store.put_bytes("pool_d/_builder/first_seen/HEAD.json", json.dumps({
        "records.jsonl": hashlib.sha256(b'{"r":1}\n').hexdigest(),
        "roots.jsonl": hashlib.sha256(b"").hexdigest()}, sort_keys=True).encode())
    with pytest.raises(m.ElasticError) as ei:
        m.state_pull(store, prefix="pool_d", log_dir=tmp_path / "c" / "first_seen",
                     prior_root=tmp_path / "pc")
    assert ei.value.code == "state_conflict"


class _Pager:
    def __init__(self, keys):
        self.keys = keys

    def paginate(self, Bucket):
        yield {"Contents": [{"Key": k, "Size": len(v)} for k, v in self.keys.items()]}


class _S3:
    def __init__(self, keys):
        self.keys = keys
        self.got = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Pager(self.keys)

    def download_file(self, bucket, key, dest):
        self.got.append(key)
        with open(dest, "wb") as f:
            f.write(self.keys[key])


def test_mirror_sync_never_touches_first_seen(tmp_path):
    m = mod()
    forge = tmp_path / "tsforge"
    (forge / "first_seen").mkdir(parents=True)
    (forge / "first_seen" / "records.jsonl").write_text("ours\n")
    s3 = _S3({"data/f/2026-11-20.parquet": b"abc", "first_seen/records.jsonl": b"theirs",
              "data/f/2026-11-19.parquet": b"xy"})
    (forge / "data" / "f").mkdir(parents=True)
    (forge / "data" / "f" / "2026-11-19.parquet").write_bytes(b"xy")
    out = m.mirror_sync(s3, bucket="b", forge_dir=forge)
    assert s3.got == ["data/f/2026-11-20.parquet"]
    assert out == {"downloaded": 1, "unchanged": 1}
    assert (forge / "first_seen" / "records.jsonl").read_text() == "ours\n"
