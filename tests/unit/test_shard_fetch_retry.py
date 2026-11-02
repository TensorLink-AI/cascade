"""Layer A (worker side): a remote shard fetch retries a failed download OR a
sha256 mismatch before the leg gives up — and a retry never changes what trains.

2026-11-02 01:22 UTC: a make-good leg on an 8xH200 died at step 7650/23841 on
``downloaded shard c4-00030.parquet fails its sha256`` — one bad download, no
retry, ~1.2 h of pod time lost.
"""
from __future__ import annotations

import io
import json
import shutil
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from cascade.generation import shards as S
from cascade.generation import transfer as T


def _corpus(d: Path, seed: int = 5) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kids = []
    for i in range(3):
        sh = [S.write_shard(d / f"c{i}-{n:05d}.parquet",
                            [rng.standard_normal((1, 32 + 4 * j)) for j in range(4)])
              for n in range(3)]
        kids.append({"index": i, "seed": S.child_seed(seed, i), "shards": sh})
    return S.finalize_index(d, seed=seed, children=kids, meta={"generator_ref": "g@sha256:1"})


def _remote(src: Path, dst: Path) -> dict:
    dst.mkdir()
    shutil.copy(src / S.INDEX_NAME, dst / S.INDEX_NAME)
    idx = S.read_index(src)
    urls = {sh["name"]: f"https://store.invalid/{sh['name']}"
            for c in idx["children"] for sh in c["shards"]}
    (dst / S.URLS_NAME).write_text(json.dumps(urls))
    return urls


@pytest.fixture
def sleeps(monkeypatch):
    seen: list[float] = []
    monkeypatch.setattr(S, "_SLEEP", seen.append)
    monkeypatch.setattr(S, "_JITTER", lambda: 1.0)
    return seen


class _Flaky:
    """A ``get_file`` double serving ``src``: for each shard in ``bad`` the first
    try raises a transport error and the second writes corrupt bytes; every
    later try (and every other shard) gets the real bytes."""

    def __init__(self, src: Path, bad: set[str]):
        self.src, self.bad = src, bad
        self.tries: dict[str, int] = {}
        self.attempts_args: list[int] = []

    def __call__(self, url: str, path, *, timeout: float = 600.0, attempts: int = 5):
        self.attempts_args.append(attempts)
        name = url.rsplit("/", 1)[-1]
        n = self.tries[name] = self.tries.get(name, 0) + 1
        if name in self.bad and n == 1:
            raise OSError("connection reset by peer")
        if name in self.bad and n == 2:
            Path(path).write_bytes(b"truncated garbage")
            return
        shutil.copy(self.src / name, path)


def test_a_flaky_fetch_that_fails_twice_streams_the_identical_series(tmp_path, monkeypatch, sleeps):
    idx = _corpus(tmp_path / "src")
    _remote(tmp_path / "src", tmp_path / "pod")
    bad = {idx["children"][0]["shards"][1]["name"], idx["children"][2]["shards"][2]["name"]}
    flaky = _Flaky(tmp_path / "src", bad)
    monkeypatch.setattr(T, "get_file", flaky)

    local = list(S.iter_corpus(tmp_path / "src"))
    remote = list(S.iter_corpus(tmp_path / "pod"))

    assert len(local) == len(remote) == 36
    assert all(np.array_equal(a, b) for a, b in zip(local, remote, strict=True))
    # Same stream digest ⇒ same training: the retries changed nothing that trains.
    assert S.stream_digest(remote)[0] == idx["corpus_digest"] == S.stream_digest(local)[0]
    assert all(flaky.tries[n] == 3 for n in bad)
    assert sorted(sleeps) == [2.0, 2.0, 4.0, 4.0]  # 2 retries per bad shard, exponential
    assert set(flaky.attempts_args) == {1}, "the shard loop owns retries: no double exponential"
    assert not list((tmp_path / "pod").glob("*.parquet*")), "no partial or consumed file left"


def test_a_shard_that_never_verifies_fails_after_bounded_tries(tmp_path, monkeypatch, sleeps):
    idx = _corpus(tmp_path / "src")
    _remote(tmp_path / "src", tmp_path / "pod")
    victim = idx["children"][1]["shards"][0]["name"]

    def get(url, path, **kw):
        name = url.rsplit("/", 1)[-1]
        if name == victim:
            Path(path).write_bytes(b"wrong")
        else:
            shutil.copy(tmp_path / "src" / name, path)

    monkeypatch.setattr(T, "get_file", get)
    with pytest.raises(S.ShardError, match=r"fails its sha256 \(after 5 tries\)"):
        list(S.iter_corpus(tmp_path / "pod"))
    assert len(sleeps) == S.SHARD_FETCH_ATTEMPTS - 1
    assert not (tmp_path / "pod" / victim).exists(), "a bad download is deleted"


def test_backoff_is_exponential_capped_and_jittered(monkeypatch):
    monkeypatch.setattr(S, "_JITTER", lambda: 1.0)
    assert [S._backoff_s(a) for a in range(1, 8)] == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    monkeypatch.setattr(S, "_JITTER", lambda: 0.0)
    assert S._backoff_s(3) == 4.0                  # jitter floor: half the step


def test_an_expired_link_raises_its_own_class_without_retrying(tmp_path, monkeypatch, sleeps):
    def get(url, path, **kw):
        raise T.LinkExpiredError("HTTP 403 on an expired link")

    monkeypatch.setattr(T, "get_file", get)
    with pytest.raises(S.ShardLinkExpired, match="expired"):
        S._fetch_shard(tmp_path, {"name": "c0-00000.parquet", "sha256": "0" * 64}, "https://x/y")
    assert sleeps == []
    assert issubclass(S.ShardLinkExpired, S.ShardError)


def test_transport_errors_exhaust_into_could_not_be_downloaded(tmp_path, monkeypatch, sleeps):
    def get(url, path, **kw):
        raise OSError("timed out")

    monkeypatch.setattr(T, "get_file", get)
    with pytest.raises(S.ShardError, match=r"could not be downloaded: timed out \(after 3 tries\)"):
        S._fetch_shard(tmp_path, {"name": "c0-00000.parquet", "sha256": "0" * 64},
                       "https://x/y", attempts=3)
    assert sleeps == [2.0, 4.0]


# ── transfer.get_file: short reads, expired links, coherent retry ──────────


class _Resp(io.BytesIO):
    def __init__(self, body: bytes, length: int | None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_get_file_refuses_a_short_body_and_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(T.urllib.request, "urlopen", lambda url, timeout: _Resp(b"abc", 10))
    with pytest.raises(OSError, match="short read: 3 of 10"):
        T.get_file("https://x/a.parquet", tmp_path / "a.parquet", attempts=1)
    assert list(tmp_path.iterdir()) == []


def test_get_file_writes_a_complete_body(tmp_path, monkeypatch):
    monkeypatch.setattr(T.urllib.request, "urlopen", lambda url, timeout: _Resp(b"abcdef", 6))
    T.get_file("https://x/a.parquet", tmp_path / "a.parquet", attempts=1)
    assert (tmp_path / "a.parquet").read_bytes() == b"abcdef"
    assert not (tmp_path / "a.parquet.part").exists()


def _http_error(code):
    def opener(url, timeout):
        raise urllib.error.HTTPError(url, code, "Forbidden", {}, None)
    return opener


def test_get_file_403_on_an_expired_link_is_link_expired_and_not_retried(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(T.time, "sleep", calls.append)
    monkeypatch.setattr(T.urllib.request, "urlopen", _http_error(403))
    old = "https://s3/b/k?X-Amz-Date=20200101T000000Z&X-Amz-Expires=60&X-Amz-Signature=s"
    with pytest.raises(T.LinkExpiredError):
        T.get_file(old, tmp_path / "a.parquet")       # default attempts: still no retry
    assert calls == []


def test_get_file_403_on_a_live_link_is_a_plain_transient_error(tmp_path, monkeypatch):
    monkeypatch.setattr(T.urllib.request, "urlopen", _http_error(403))
    live = "https://s3/b/k?X-Amz-Date=20990101T000000Z&X-Amz-Expires=60&X-Amz-Signature=s"
    with pytest.raises(OSError, match="HTTP 403") as ei:
        T.get_file(live, tmp_path / "a.parquet", attempts=1)
    assert not isinstance(ei.value, T.LinkExpiredError)


def test_link_expiry_parses_sigv4_and_sigv2():
    v4 = "https://s3/b/k?X-Amz-Date=20261102T000000Z&X-Amz-Expires=86400&X-Amz-Signature=s"
    assert T.link_expiry_epoch(v4) == 1793577600 + 86400
    assert T.link_expired(v4, now=1793577600 + 86400)
    assert not T.link_expired(v4, now=1793577600 + 86399)
    assert T.link_expiry_epoch("https://s3/b/k?Expires=1700000000&Signature=s") == 1700000000
    assert T.link_expiry_epoch("file:///tmp/x") == 0 and not T.link_expired("file:///tmp/x")


# ── fault class: our storage, never the miner's attempt ─────────────────────


def _dispatch_failure(stderr: str, rc: int = 1) -> str:
    """The message RemoteDispatcher raises for a failed leg (rc != 3)."""
    from cascade.trainer.remote import summarize_failure

    cause, detail = summarize_failure(stderr, rc=rc)
    return f"remote challenger on pod-a failed (rc={rc}) {cause}" + (f"\n{detail}" if detail else "")


def _traceback(exc_line: str) -> str:
    return ("2026-11-02 01:22:51,580 ERROR cascade.trainer.worker: worker training failed\n"
            "Traceback (most recent call last):\n"
            '  File "/root/cascade/cascade/generation/shards.py", line 266, in _fetch_shard\n'
            "    raise ShardError(...)\n"
            f"{exc_line}\n")


_SHARD_EXC_LINES = (
    "cascade.generation.shards.ShardError: downloaded shard c4-00030.parquet fails its sha256",
    "cascade.generation.shards.ShardError: downloaded shard c4-00030.parquet fails its sha256 "
    "(after 5 tries)",
    "cascade.generation.shards.ShardError: shard c1-00002.parquet could not be downloaded: "
    "timed out (after 5 tries)",
    "cascade.generation.shards.ShardLinkExpired: shard c0-00001.parquet: download link expired "
    "(HTTP 403 on an expired link) — re-sign the corpus links",
)


def test_a_shard_fetch_failure_on_a_shards_leg_is_an_unburned_operator_fault():
    from cascade.funding.queue import OPERATOR_FAULT_CLASS
    from cascade.trainer.faults import classify_funded_worker_failure, shard_fetch_failure

    for exc in _SHARD_EXC_LINES:
        text = _dispatch_failure(_traceback(exc))
        assert shard_fetch_failure(text), exc
        assert classify_funded_worker_failure(1, text, stalled_before=False, shards_leg=True) == (
            False, OPERATOR_FAULT_CLASS, False)


def test_forged_shard_text_in_a_streaming_leg_stays_burned():
    """A streaming leg runs the miner's generator on the pod, so it can print
    anything — including our exact shard-failure traceback. Off a shards leg
    the text is never consulted: the exit stays infra and burns an attempt."""
    from cascade.trainer.faults import classify_funded_worker_failure

    for exc in _SHARD_EXC_LINES:
        forged = _dispatch_failure(_traceback(exc))
        assert classify_funded_worker_failure(1, forged, stalled_before=False) == (
            False, "infra", True)
        assert classify_funded_worker_failure(1, forged, stalled_before=False,
                                              shards_leg=False) == (False, "infra", True)


def test_only_the_final_traceback_exception_counts():
    from cascade.trainer.faults import classify_funded_worker_failure, shard_fetch_failure

    shard = _SHARD_EXC_LINES[0]
    # The phrase as a log line, then a different real death: not a shard fault.
    other = _dispatch_failure(f"{shard}\n" + _traceback("RuntimeError: CUDA error: device lost"))
    assert not shard_fetch_failure(other)
    assert classify_funded_worker_failure(1, other, stalled_before=False, shards_leg=True) == (
        False, "infra", True)
    # The phrase with no traceback at all, or not fully qualified: not matched.
    assert not shard_fetch_failure(_dispatch_failure(f"WARNING {shard}\nexiting\n"))
    assert not shard_fetch_failure(_dispatch_failure(_traceback(
        "ShardError: downloaded shard c4-00030.parquet fails its sha256")))
    assert not shard_fetch_failure(shard)              # a bare string is not a traceback block
    # A miner rejection stays the miner's even on a shards leg.
    assert classify_funded_worker_failure(
        3, _dispatch_failure(_traceback(shard), rc=3), stalled_before=False,
        shards_leg=True) == (True, "generator", False)
    # Prefetch log lines never masquerade as a worker shard failure.
    assert not shard_fetch_failure(_dispatch_failure(
        "cascade-prefetch: WARNING: c4-00030.parquet: gave up after 5 tries (curl rc=28, "
        "HTTP 000); the worker fetches it itself\n"))


def test_an_http_incomplete_read_is_retried_like_a_transport_error(tmp_path, monkeypatch, sleeps):
    import http.client

    src = tmp_path / "src"
    idx = _corpus(src)
    sh = idx["children"][0]["shards"][0]
    tries = []

    def get(url, path, **kw):
        tries.append(1)
        if len(tries) == 1:
            raise http.client.IncompleteRead(b"abc", 10)
        shutil.copy(src / sh["name"], path)

    monkeypatch.setattr(T, "get_file", get)
    (tmp_path / "pod").mkdir()
    p = S._fetch_shard(tmp_path / "pod", sh, "https://x/" + sh["name"])
    assert S.file_sha256(p) == sh["sha256"] and len(tries) == 2 and sleeps == [2.0]


def test_transfer_retry_covers_http_exceptions(monkeypatch):
    import http.client

    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    calls = []

    def once():
        calls.append(1)
        if len(calls) < 3:
            raise http.client.RemoteDisconnected("closed")
        return "ok"

    assert T._retry("download x", once, attempts=3) == "ok" and len(calls) == 3
