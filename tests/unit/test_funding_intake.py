"""cascade.funding.intake: wire contract, auth gate, and key custody."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

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
