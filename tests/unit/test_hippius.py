"""Hippius storage helpers — pure parts (Hub ref grammar, tar packing for S3
pool snapshots, S3 manifest + log layout over a fake S3 client). No real Hub /
boto3 endpoint needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.shared import hippius


def test_hub_ref_parses_and_rejects_garbage():
    ref = hippius.HubRef.parse("alice/metro-gen@sha256:" + "a" * 64)
    assert ref.repo == "alice/metro-gen"
    assert ref.digest == "sha256:" + "a" * 64
    assert ref.immutable_ref == "alice/metro-gen@sha256:" + "a" * 64
    # hf: digests are accepted too (a genesis/eval artefact mirrored on HF).
    assert hippius.is_hub_ref("ns/name@hf:" + "b" * 40)
    for bad in (
        "",
        "not-a-ref",                              # no @digest
        "alice/gen@sha256:short",                 # truncated digest
        "alice/gen@deadbeef",                     # missing sha256: prefix
        "@sha256:" + "a" * 64,                    # empty repo
        "Alice Gen/x@sha256:" + "a" * 64,         # space in repo
    ):
        assert not hippius.is_hub_ref(bad)
    with pytest.raises(hippius.StorageError):
        hippius.HubRef.parse("no-at-sign")


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


_FETCH_REF = "cascade/ckpt-test@sha256:" + "b" * 64


def _fake_hub(monkeypatch, downloader):
    """Install a fake ``hippius_hub`` module whose snapshot_download is ``downloader``."""
    import sys
    import types

    mod = types.ModuleType("hippius_hub")
    mod.snapshot_download = downloader
    monkeypatch.setitem(sys.modules, "hippius_hub", mod)
    monkeypatch.setattr(hippius, "_resolve_hub_token_for_pull", lambda _msg: "tok")


def test_fetch_from_hub_is_atomic_reused_and_marked_complete(tmp_path, monkeypatch):
    calls = {"n": 0}

    def dl(*, repo_id, revision, local_dir, allow_patterns, token):
        calls["n"] += 1
        p = Path(local_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "weights.safetensors").write_bytes(b"w")

    _fake_hub(monkeypatch, dl)
    dest = tmp_path / "sha256-b"
    out = hippius.fetch_from_hub(_FETCH_REF, dest)
    assert out == dest
    assert (dest / "weights.safetensors").read_bytes() == b"w"
    assert (dest / hippius.FETCH_COMPLETE_MARKER).exists()
    # The private download dir is gone whether the rename won or lost.
    assert not list(tmp_path.glob(".sha256-b.fetch-*"))
    # A digest-pinned snapshot is immutable: the second fetch must reuse it
    # (the old rmtree+redownload deleted it out from under sibling lanes).
    hippius.fetch_from_hub(_FETCH_REF, dest)
    assert calls["n"] == 1


def test_fetch_from_hub_replaces_unmarked_partial_dir(tmp_path, monkeypatch):
    def dl(*, repo_id, revision, local_dir, allow_patterns, token):
        p = Path(local_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}")

    _fake_hub(monkeypatch, dl)
    dest = tmp_path / "sha256-b"
    dest.mkdir()
    (dest / "stale.bin").write_bytes(b"partial download, no marker")
    hippius.fetch_from_hub(_FETCH_REF, dest)
    assert (dest / "config.json").exists()
    assert not (dest / "stale.bin").exists()


def test_fetch_from_hub_yields_to_sibling_completed_snapshot(tmp_path, monkeypatch):
    dest = tmp_path / "sha256-b"

    def dl(*, repo_id, revision, local_dir, allow_patterns, token):
        p = Path(local_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ours.bin").write_bytes(b"x")
        # While we were downloading, a sibling lane completed the same digest.
        dest.mkdir()
        (dest / "theirs.bin").write_bytes(b"y")
        (dest / hippius.FETCH_COMPLETE_MARKER).touch()

    _fake_hub(monkeypatch, dl)
    out = hippius.fetch_from_hub(_FETCH_REF, dest)
    assert out == dest
    # The sibling's snapshot (identical bytes by digest) is kept untouched.
    assert (dest / "theirs.bin").exists()
    assert not (dest / "ours.bin").exists()
    assert not list(tmp_path.glob(".sha256-b.fetch-*"))


def test_is_retryable_hub_error_classifies_transient_vs_permanent():
    # The exact message from the failing miner upload is a transient read stall.
    assert hippius._is_retryable_hub_error(RuntimeError("The read operation timed out"))
    assert hippius._is_retryable_hub_error(TimeoutError())
    assert hippius._is_retryable_hub_error(ConnectionError("connection reset by peer"))
    assert hippius._is_retryable_hub_error(RuntimeError("503 Server Error: Service Unavailable"))
    # Deterministic failures must not be retried.
    assert not hippius._is_retryable_hub_error(hippius.HubAuthError("no token"))
    assert not hippius._is_retryable_hub_error(RuntimeError("404 repo not found"))


def test_is_retryable_hub_error_walks_the_cause_chain():
    # hippius_hub wraps transport drops as bare RuntimeError("chunk N failed")
    # with the retryable detail in __cause__ — the classifier must walk it
    # (2026-08-19: six transient chunk-0 drops each burned a whole dispatch
    # because the top-level message matched nothing).
    transport = RuntimeError("HTTP transport error: error decoding response body")
    chunk = RuntimeError("chunk 0 failed")
    chunk.__cause__ = transport
    assert hippius._is_retryable_hub_error(chunk)

    # The observed mainnet shape: chunk failure caused by a local I/O error.
    io_err = RuntimeError("local I/O error")
    chunk2 = RuntimeError("chunk 0 failed")
    chunk2.__cause__ = io_err
    assert hippius._is_retryable_hub_error(chunk2)
    assert hippius._is_retryable_hub_error(RuntimeError("local I/O error"))

    # Implicit chaining (__context__) counts too.
    ctx = RuntimeError("chunk 3 failed")
    ctx.__context__ = TimeoutError()
    assert hippius._is_retryable_hub_error(ctx)

    # A 401-wrapped chunk failure has no transport substring anywhere in the
    # chain — stays permanent (the deliberate carve-out).
    denied = RuntimeError("chunk 0 failed")
    denied.__cause__ = RuntimeError("server returned 401 (Failed chunk bytes)")
    assert not hippius._is_retryable_hub_error(denied)

    # Self-referential chains must not loop forever.
    loop_exc = RuntimeError("chunk 1 failed")
    loop_exc.__cause__ = loop_exc
    assert not hippius._is_retryable_hub_error(loop_exc)


def test_retry_hub_op_recovers_after_transient_timeouts():
    calls = {"n": 0}
    slept: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("The read operation timed out")
        return "sha256:" + "a" * 64

    out = hippius._retry_hub_op(flaky, "upload of ./gen to alice/gen", sleep=slept.append)
    assert out == "sha256:" + "a" * 64
    assert calls["n"] == 3
    # Exponential backoff between the two failed attempts: 2s then 4s.
    assert slept == [2.0, 4.0]


def test_retry_hub_op_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_times_out():
        calls["n"] += 1
        raise RuntimeError("The read operation timed out")

    with pytest.raises(hippius.StorageError) as ei:
        hippius._retry_hub_op(always_times_out, "upload of ./gen to alice/gen",
                              attempts=3, sleep=lambda _: None)
    assert calls["n"] == 3  # tried exactly `attempts` times, no more
    assert "after 3 attempt(s)" in str(ei.value)
    assert "read operation timed out" in str(ei.value).lower()


def test_retry_hub_op_does_not_retry_permanent_errors():
    calls = {"n": 0}

    def not_found():
        calls["n"] += 1
        raise RuntimeError("404 Client Error: repo not found")

    with pytest.raises(hippius.StorageError):
        hippius._retry_hub_op(not_found, "fetch of alice/gen@sha256:...", sleep=lambda _: None)
    assert calls["n"] == 1  # permanent error surfaces on the first attempt


def test_retry_hub_op_passes_auth_error_through_unretried():
    calls = {"n": 0}

    def auth_fails():
        calls["n"] += 1
        raise hippius.HubAuthError("set HIPPIUS_HUB_TOKEN")

    # HubAuthError is a StorageError, so it propagates unchanged (not re-wrapped).
    with pytest.raises(hippius.HubAuthError):
        hippius._retry_hub_op(auth_fails, "upload of ./gen to alice/gen", sleep=lambda _: None)
    assert calls["n"] == 1


class _FakeS3Store:
    """In-memory stand-in for S3Store (same put_text/get_text surface)."""

    def __init__(self):
        self.objects: dict[str, str] = {}

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
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


def test_pull_token_falls_back_to_anonymous(monkeypatch, tmp_path):
    """No Hub creds anywhere ⇒ pulls use the library's anonymous sentinel
    (False) instead of raising — public repos need no account (2026-07-18:
    an external validator's checkpoint fetch died on HubAuthError although
    the registry serves the pull anonymously). Uploads stay strict."""
    for name in (*hippius.HUB_TOKEN_ENV_NAMES, *hippius.HUB_USERNAME_ENV_NAMES,
                 *hippius.HUB_PASSWORD_ENV_NAMES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hippius, "HUB_TOKEN_PATH", tmp_path / "absent-token")
    assert hippius._resolve_hub_token_for_pull("Downloading x/y@sha256:z") is False
    with pytest.raises(hippius.HubAuthError):
        hippius._resolve_hub_token("Uploading ./d to x/y")


def test_pull_token_prefers_real_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv(hippius.HUB_TOKEN_ENV_NAMES[0], "tok-123")
    monkeypatch.setattr(hippius, "HUB_TOKEN_PATH", tmp_path / "absent-token")
    assert hippius._resolve_hub_token_for_pull("Downloading x/y@sha256:z") == "tok-123"
