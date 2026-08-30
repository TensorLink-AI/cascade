"""cascade.funding.intake: wire contract, auth gate, and key custody."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from cascade.funding.intake import (
    FundingIntake,
    canonical_fund_message,
)
from cascade.funding.queue import FundedQueue
from cascade.funding.vault import PayerKeyVault

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "cascade-gen-abc123@sha256:" + "a" * 64


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_intake(tmp_path, *, require_signature=False, verify=None,
                resolve=lambda hk, ref: 500, clock=None):
    clock = clock or FakeClock()
    return FundingIntake(
        FundedQueue(tmp_path / "queue.json", clock=clock),
        PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=resolve,
        require_signature=require_signature,
        verify=verify or (lambda hk, msg, sig: False),
        clock=clock,
    ), clock


# ── behaviour, no HTTP ───────────────────────────────────────────────────────


def test_fund_happy_path_queues_and_vaults(tmp_path):
    intake, _ = make_intake(tmp_path)
    status, body = intake.fund({
        "X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk-live",
    })
    assert (status, body["status"]) == (202, "queued")
    assert intake.queue.get(HK).reveal_block == 500
    assert intake.vault.get(HK) == "sk-live"
    # Idempotent re-fund: 200, and the vault key is refreshed, not dropped.
    status, body = intake.fund({
        "X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk-live-2",
    })
    assert (status, body["status"]) == (200, "already-queued")
    assert intake.vault.get(HK) == "sk-live-2"


@pytest.mark.parametrize("headers,code", [
    ({"X-Commit-Ref": REF, "X-Lium-Api-Key": "sk"}, "missing_hotkey"),
    ({"X-Miner-Hotkey": HK, "X-Lium-Api-Key": "sk"}, "missing_ref"),
    ({"X-Miner-Hotkey": HK, "X-Commit-Ref": REF}, "missing_lium_api_key"),
])
def test_fund_fail_closed_codes(tmp_path, headers, code):
    intake, _ = make_intake(tmp_path)
    status, body = intake.fund(headers)
    assert status == 400 and body["code"] == code


def test_fund_unrevealed_ref_is_403(tmp_path):
    intake, _ = make_intake(tmp_path, resolve=lambda hk, ref: None)
    status, body = intake.fund({
        "X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk",
    })
    assert status == 403 and body["code"] == "not_revealed"
    assert intake.vault.get(HK) is None      # a refused fund stores nothing


def test_signature_gate(tmp_path):
    clock = FakeClock()
    seen = {}

    def verify(hotkey, msg, sig):
        seen["msg"] = msg
        return sig == "beef"

    intake, _ = make_intake(tmp_path, require_signature=True, verify=verify, clock=clock)
    base = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk",
            "X-Timestamp": str(int(clock.t))}
    status, body = intake.fund({**base, "X-Signature": "dead"})
    assert (status, body["code"]) == (400, "bad_signature")
    status, body = intake.fund({**base, "X-Signature": "beef"})
    assert status == 202
    assert seen["msg"] == canonical_fund_message("fund", HK, REF, str(int(clock.t)))
    # Stale timestamp rejected even with a "valid" signature.
    stale = {**base, "X-Timestamp": str(int(clock.t) - 3600), "X-Signature": "beef"}
    status, body = intake.fund(stale)
    assert (status, body["code"]) == (400, "stale_timestamp")


def test_withdraw_lifecycle(tmp_path):
    intake, _ = make_intake(tmp_path)
    headers = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk"}
    intake.fund(headers)
    status, body = intake.withdraw({"X-Miner-Hotkey": HK, "X-Commit-Ref": REF})
    assert (status, body["status"]) == (200, "withdrawn")
    assert intake.vault.get(HK) is None      # withdrawing forgets the key
    status, body = intake.withdraw({"X-Miner-Hotkey": HK, "X-Commit-Ref": REF})
    assert (status, body["code"]) == (409, "not_queued")


def test_canonical_message_rejects_unknown_action():
    with pytest.raises(ValueError):
        canonical_fund_message("steal", HK, REF, "0")


# ── over real HTTP ───────────────────────────────────────────────────────────


@pytest.fixture()
def live_server(tmp_path):
    intake, _ = make_intake(tmp_path)
    server = intake.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield intake, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _request(url, method="GET", headers=None):
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _raw_post(base, path, extra_headers):
    """Send a hand-built POST (for headers urllib refuses, like a bad length)."""
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(base)
    host, port = parts.hostname, parts.port
    lines = [f"POST {path} HTTP/1.1", f"Host: {host}:{port}", "Connection: close"]
    lines += extra_headers
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode()
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(raw)
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    status_line = buf.split(b"\r\n", 1)[0].decode()
    return int(status_line.split()[1])


def test_negative_content_length_is_rejected_not_an_unbounded_read(live_server):
    # A negative Content-Length must be a clean 400, never rfile.read(-1)
    # (read-to-EOF past the cap) and never an uncaught int() ValueError that
    # kills the connection with no reply (review 2026-08-29).
    _, base = live_server
    assert _raw_post(base, "/v1/submit", ["Content-Length: -1"]) == 400
    assert _raw_post(base, "/v1/fund", ["Content-Length: -1"]) == 400
    assert _raw_post(base, "/v1/submit", ["Content-Length: not-a-number"]) == 400


def test_http_fund_and_queue_feed(live_server):
    intake, base = live_server
    status, body = _request(f"{base}/health")
    assert (status, body["status"]) == (200, "ok")
    status, body = _request(f"{base}/v1/fund", method="POST", headers={
        "X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk-live",
    })
    assert (status, body["status"]) == (202, "queued")
    status, body = _request(f"{base}/v1/queue")
    assert status == 200 and body["queued_depth"] == 1
    assert "sk-live" not in json.dumps(body)   # the feed never carries keys
    status, body = _request(f"{base}/v1/fund", method="POST", headers={
        "X-Miner-Hotkey": HK, "X-Commit-Ref": REF,
    })
    assert (status, body["code"]) == (400, "missing_lium_api_key")
    status, body = _request(f"{base}/nope", method="POST")
    assert status == 404


# ── DoS backstops: socket timeout + bounded concurrency ──────────────────────


def _connect(base):
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(base)
    return socket.create_connection((parts.hostname, parts.port), timeout=5)


def _serve(intake, **server_kw):
    server = intake.make_server("127.0.0.1", 0, **server_kw)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_slowloris_connection_is_closed_not_parked_forever(tmp_path):
    # A client dribbling half a request must lose its connection after
    # request_timeout_s — the stdlib default is NO socket timeout, i.e. a
    # permanently pinned handler thread per slow client.
    intake, _ = make_intake(tmp_path)
    server, base = _serve(intake, request_timeout_s=0.3)
    try:
        with _connect(base) as s:
            s.settimeout(5)
            s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nX-Drib")  # …stall
            t0 = time.monotonic()
            try:
                data = s.recv(4096)
            except OSError:
                data = b""
            assert data == b""                      # server hung up on us
            assert time.monotonic() - t0 < 4.0      # promptly, not never
        status, body = _request(f"{base}/health")   # thread was reclaimed
        assert (status, body["status"]) == (200, "ok")
    finally:
        server.shutdown()
        server.server_close()


def test_connection_flood_gets_fast_503_and_the_cap_recovers(tmp_path):
    intake, _ = make_intake(tmp_path)
    server, base = _serve(intake, max_connections=2, request_timeout_s=5.0)
    try:
        # Two idle connections pin both handler slots (a slot is taken at
        # accept, before any bytes arrive — that is exactly the flood shape).
        holders = [_connect(base) for _ in range(2)]
        deadline = time.monotonic() + 5
        while server._slots._value > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server._slots._value == 0

        # The third connection gets an immediate 503, not a queued thread.
        with _connect(base) as s3:
            s3.settimeout(5)
            s3.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            buf = b""
            while True:
                chunk = s3.recv(4096)
                if not chunk:
                    break
                buf += chunk
        assert buf.startswith(b"HTTP/1.1 503")
        assert b"Retry-After" in buf and b"overloaded" in buf

        # Releasing the holders frees the slots and service resumes.
        for h in holders:
            h.close()
        status = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                status, _body = _request(f"{base}/health")
                if status == 200:
                    break
            except OSError:
                pass
            time.sleep(0.05)
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()


# ── ChainRevealResolver: the reveal poll runs under a deadline ───────────────

# The resolver round-trips refs through the real commit grammar, which wants
# a full Hub ``owner/name@sha256:…`` ref (unlike the header-plumbing tests
# above, where REF is opaque).
HUB_REF = "miner/gen-abc@sha256:" + "a" * 64


def _commitment(hotkey, ref, block):
    from cascade.interface.validation import format_commit

    return SimpleNamespace(hotkey=hotkey, payload=format_commit(ref),
                           commit_block=block)


def test_resolver_resolves_caches_and_refuses_foreign_refs():
    from cascade.funding.main import ChainRevealResolver

    calls = []

    def poll_commitments(include_history=True):
        calls.append(1)
        return [_commitment(HK, HUB_REF, 42)]

    r = ChainRevealResolver(SimpleNamespace(poll_commitments=poll_commitments),
                            cache_seconds=60.0, deadline_seconds=5.0)
    assert r(HK, HUB_REF) == 42
    assert r(HK, "other-repo@sha256:" + "b" * 64) is None
    assert r("5SomeoneElse", HUB_REF) is None
    assert len(calls) == 1                      # served from the cache


def test_resolver_deadline_serves_stale_and_never_stacks_polls():
    from cascade.funding.main import ChainRevealResolver

    started = []
    release = threading.Event()

    def poll_commitments(include_history=True):
        started.append(threading.current_thread())
        release.wait(10)
        return [_commitment(HK, HUB_REF, 42)]

    r = ChainRevealResolver(SimpleNamespace(poll_commitments=poll_commitments),
                            cache_seconds=0.0, deadline_seconds=0.05)
    t0 = time.monotonic()
    assert r(HK, HUB_REF) is None          # deadline hit → stale (empty) table
    assert time.monotonic() - t0 < 2.0  # …and the caller was NOT pinned
    assert r(HK, HUB_REF) is None          # a second call must not stack a poll
    assert len(started) == 1           # on the one still hung
    release.set()                      # the hung poll finally finishes…
    started[0].join(5)
    assert r(HK, HUB_REF) == 42            # …and the next refresh harvests it


def test_resolver_poll_error_reaches_the_caller():
    from cascade.funding.main import ChainRevealResolver

    def boom(include_history=True):
        raise RuntimeError("substrate down")

    r = ChainRevealResolver(SimpleNamespace(poll_commitments=boom),
                            cache_seconds=60.0, deadline_seconds=5.0)
    with pytest.raises(RuntimeError, match="substrate down"):
        r(HK, HUB_REF)
