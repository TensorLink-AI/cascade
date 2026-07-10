"""Hippius storage helpers — pure parts (Hub ref grammar, tar packing for S3
pool snapshots, S3 manifest + log layout over a fake S3 client). No real Hub /
boto3 endpoint needed."""

from __future__ import annotations

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


def test_is_retryable_hub_error_classifies_transient_vs_permanent():
    # The exact message from the failing miner upload is a transient read stall.
    assert hippius._is_retryable_hub_error(RuntimeError("The read operation timed out"))
    assert hippius._is_retryable_hub_error(TimeoutError())
    assert hippius._is_retryable_hub_error(ConnectionError("connection reset by peer"))
    assert hippius._is_retryable_hub_error(RuntimeError("503 Server Error: Service Unavailable"))
    # Deterministic failures must not be retried.
    assert not hippius._is_retryable_hub_error(hippius.HubAuthError("no token"))
    assert not hippius._is_retryable_hub_error(RuntimeError("404 repo not found"))


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


def test_hub_op_timeout_env_overrides_and_disables(monkeypatch):
    monkeypatch.delenv(hippius.HUB_OP_TIMEOUT_ENV, raising=False)
    assert hippius._hub_op_timeout() == hippius.HUB_OP_TIMEOUT_DEFAULT_S  # unset ⇒ default
    monkeypatch.setenv(hippius.HUB_OP_TIMEOUT_ENV, "30")
    assert hippius._hub_op_timeout() == 30.0
    monkeypatch.setenv(hippius.HUB_OP_TIMEOUT_ENV, "0")
    assert hippius._hub_op_timeout() is None  # <= 0 disables the watchdog
    monkeypatch.setenv(hippius.HUB_OP_TIMEOUT_ENV, "-5")
    assert hippius._hub_op_timeout() is None
    monkeypatch.setenv(hippius.HUB_OP_TIMEOUT_ENV, "not-a-number")
    assert hippius._hub_op_timeout() == hippius.HUB_OP_TIMEOUT_DEFAULT_S  # malformed ⇒ default


def test_run_with_timeout_returns_value_and_runs_inline_when_unbounded():
    assert hippius._run_with_timeout(lambda: 42, 5.0) == 42
    assert hippius._run_with_timeout(lambda: 7, None) == 7  # None ⇒ inline, no watchdog


def test_run_with_timeout_propagates_op_error():
    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        hippius._run_with_timeout(boom, 5.0)


def test_run_with_timeout_abandons_a_hung_op():
    import threading

    never = threading.Event()  # never set ⇒ the op blocks until the watchdog fires
    try:
        with pytest.raises(TimeoutError, match="never responded"):
            hippius._run_with_timeout(never.wait, 0.05)
    finally:
        never.set()  # release the daemon worker so it can exit cleanly


def test_retry_hub_op_retries_a_hung_op_then_gives_up():
    # The exact production symptom: the Hub op never returns AND never raises.
    # The watchdog converts each hang into a retryable TimeoutError, so the loop
    # retries `attempts` times and then surfaces a StorageError (rather than
    # blocking forever, which would strand `cascade deploy`'s HF fallback).
    import threading

    never = threading.Event()
    slept: list[float] = []
    try:
        with pytest.raises(hippius.StorageError) as ei:
            hippius._retry_hub_op(
                never.wait, "upload of ./gen to alice/gen",
                attempts=2, sleep=slept.append, op_timeout=0.05,
            )
    finally:
        never.set()
    assert "after 2 attempt(s)" in str(ei.value)
    assert "exceeded" in str(ei.value).lower()  # the TimeoutError text is carried through
    assert slept == [2.0]  # one exponential backoff between the two bounded attempts


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
