"""The miner→operator funding endpoint: the one HTTP surface cascade exposes.

Everything else miners do rides the chain (commit/reveal) or public storage
(pull-only status files) — see ``docs/COMMUNICATION.md``. An API key cannot
ride either: a chain commit is public forever once revealed, and storage is
world-readable. So funding is a tiny authenticated POST straight to the
operator, mirroring PRISM's intake (``base`` repo, ``prism-challenge``): the
key travels as the ``X-Lium-Api-Key`` header, lands in the
:class:`~cascade.funding.vault.PayerKeyVault`, and NOTHING else ever sees it —
the queue entry it creates carries hotkey/ref/reveal-block only.

Wire contract (header names kept byte-compatible with PRISM so miner tooling
transfers):

    POST /v1/fund
        X-Miner-Hotkey:  SS58 hotkey (identity)
        X-Lium-Api-Key:  the miner's Lium key (payment)
        X-Commit-Ref:    the revealed generator ref this funds (repo@digest)
        X-Timestamp:     unix seconds, ±FRESHNESS_WINDOW_SECONDS
        X-Signature:     hex sr25519 by the hotkey over canonical_fund_message
    → 202 {"status": "queued" | "replaced"}  |  200 "already-queued"
    → 400 missing_lium_api_key / missing_hotkey / missing_ref /
          stale_timestamp / bad_signature      |  403 not_revealed

    POST /v1/submit     (body = the generator ZIP; same identity headers,
        X-Content-Digest: sha256:<hex> of the body — the SIGNED binding —
        and optionally X-Lium-Api-Key to fund in the same request)
    → 201 {"status": "stored", "ref": "vault/direct@sha256:…",
           "commit_payload": "metro-v1:gen:hippius:vault/direct@sha256:…",
           "funding": "pending_reveal" | "already-funded" | "none" |
                      "blocked-by-existing-entry"}  # last carries a funding_note
    → 400 digest_mismatch / missing_digest / bad_content_length | 409 digest_owned
    → 413 zip_too_large                     |  503 submissions_disabled

    POST /v1/withdraw   (same auth headers, no key needed)
    GET  /v1/queue      (public transparency feed — no key material)
    GET  /health
    (any path)          → 503 overloaded + Retry-After past the connection cap

Direct submission (DEC-CA-0036's private-code half): the ZIP never touches a
miner-hosted repo — it lands in the operator's private
:class:`~cascade.funding.store.SubmissionStore`, the miner chain-commits the
returned vault ref, and the code goes public only if it takes the throne
(the champion publisher). A submit that also carries the Lium key parks a
``pending_reveal`` queue entry that auto-promotes to ``queued`` the moment
the chain reveal resolves — one request funds the whole entry.

Fail-closed at intake: a request that cannot fund a rental is rejected with
the code the miner needs to fix it, never accepted-and-stuck. There is
deliberately NO balance pre-check (PRISM's choice too) — the key is validated
by use, and an underfunded key surfaces as a classified rent failure that
requeues without burning.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .queue import FundedQueue
from .vault import PayerKeyVault

__all__ = [
    "FRESHNESS_WINDOW_SECONDS",
    "FundingIntake",
    "canonical_fund_message",
    "verify_hotkey_signature",
]

log = logging.getLogger("cascade.funding.intake")

# Replay bound on the signed timestamp. Funding is idempotent, so the only
# replay of consequence is re-queueing after a withdraw — a five-minute
# window plus per-request freshness closes it without clock heroics.
FRESHNESS_WINDOW_SECONDS = 300.0

_ACTIONS = ("fund", "withdraw", "submit")


def canonical_fund_message(action: str, hotkey: str, ref: str, timestamp: str) -> bytes:
    """The byte string the miner's hotkey signs. Versioned; fields ``:``-joined.

    ``ref`` is a Hub ``repo@digest`` and hotkeys are SS58 — neither contains
    ``:``-ambiguity in practice, and the version tag pins the layout so a
    future field addition is a new version, not a silent re-parse.
    """
    if action not in _ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    return f"cascade-{action}:v1:{hotkey}:{ref}:{timestamp}".encode()


def verify_hotkey_signature(hotkey: str, message: bytes, signature_hex: str) -> bool:
    """sr25519 verification against the hotkey's SS58 address.

    Same lazy-import shape as ``shared.manifest.verify_signature``: bittensor
    is the ``chain`` extra, and a missing lib or malformed signature is a
    False, never a crash — the caller decides whether unverifiable means
    rejected (it does, whenever signatures are required).
    """
    try:
        from bittensor import Keypair  # type: ignore

        kp = Keypair(ss58_address=hotkey)
        return bool(kp.verify(message, bytes.fromhex(signature_hex)))
    except Exception:  # noqa: BLE001 — any failure to verify is a non-verify
        return False


class FundingIntake:
    """The intake's behaviour, separated from HTTP plumbing for testability.

    ``resolve_reveal(hotkey, ref) -> int | None`` is the eligibility oracle:
    the reveal block of ``hotkey``'s revealed commitment matching ``ref``, or
    ``None`` when no such reveal exists (funding an unrevealed or foreign ref
    is refused — the queue orders by reveal block, so an unresolvable one has
    no seniority to claim). Injected so the service can run against the chain
    poller in production and a table in tests.
    """

    def __init__(
        self,
        queue: FundedQueue,
        vault: PayerKeyVault,
        *,
        resolve_reveal: Callable[[str, str], int | None],
        require_signature: bool = True,
        verify: Callable[[str, bytes, str], bool] = verify_hotkey_signature,
        clock: Callable[[], float] = time.time,
        store: object | None = None,   # SubmissionStore; None = submissions off
    ) -> None:
        self.queue = queue
        self.vault = vault
        self.resolve_reveal = resolve_reveal
        self.require_signature = require_signature
        self.verify = verify
        self.clock = clock
        self.store = store

    def sweep_pending(self) -> int:
        """Promote submit-with-key entries whose chain reveal has landed.

        Lazy, request-driven (no background thread): runs at the top of every
        fund/submit/queue read, so a pending entry goes live within one
        interaction — or at latest the trainer's own queue read — of its
        reveal resolving. Never raises: a chain hiccup here must not fail the
        request that triggered the sweep.
        """
        try:
            return self.queue.promote_pending(self.resolve_reveal)
        except Exception:  # noqa: BLE001 — sweep is best-effort by design
            log.exception("pending-reveal sweep failed (will retry next request)")
            return 0

    # ── request handling (returns (http_status, body_dict)) ──────────────────

    def _auth(self, action: str, headers) -> tuple[int, dict] | tuple[None, dict]:
        """Shared identity/signature gate; (None, ctx) when the request passes."""
        hotkey = (headers.get("X-Miner-Hotkey") or "").strip()
        if not hotkey:
            return 400, {"code": "missing_hotkey",
                         "message": "send your SS58 hotkey as X-Miner-Hotkey"}
        ref = (headers.get("X-Commit-Ref") or "").strip()
        if not ref:
            return 400, {"code": "missing_ref",
                         "message": "send the revealed generator ref as X-Commit-Ref"}
        if self.require_signature:
            ts = (headers.get("X-Timestamp") or "").strip()
            sig = (headers.get("X-Signature") or "").strip()
            try:
                skew = abs(self.clock() - float(ts))
            except ValueError:
                skew = float("inf")
            if skew > FRESHNESS_WINDOW_SECONDS:
                return 400, {"code": "stale_timestamp",
                             "message": "X-Timestamp must be current unix seconds "
                                        f"(±{FRESHNESS_WINDOW_SECONDS:.0f}s)"}
            msg = canonical_fund_message(action, hotkey, ref, ts)
            if not sig or not self.verify(hotkey, msg, sig):
                return 400, {"code": "bad_signature",
                             "message": "X-Signature must be the hotkey's sr25519 "
                                        "signature over the canonical message"}
        return None, {"hotkey": hotkey, "ref": ref}

    def submit_gate(self, headers) -> tuple[int, dict] | tuple[None, dict]:
        """Identity + signature gate for a submit, checkable BEFORE the body.

        The signature is over the DECLARED digest, so the whole gate runs on
        headers alone — the HTTP layer calls it before reading a byte of the
        upload, which is what stops an unauthenticated client from making N
        handler threads each buffer a cap-sized body on the orchestrator
        (review 2026-08-29). (With ``require_signature`` off — dev only —
        the body is still gated on the cheap header checks.)
        """
        if self.store is None:
            return 503, {"code": "submissions_disabled",
                         "message": "this intake accepts funding only; submit via "
                                    "the Hub path (cascade deploy)"}
        hotkey = (headers.get("X-Miner-Hotkey") or "").strip()
        if not hotkey:
            return 400, {"code": "missing_hotkey",
                         "message": "send your SS58 hotkey as X-Miner-Hotkey"}
        declared = (headers.get("X-Content-Digest") or "").strip()
        if not declared.startswith("sha256:"):
            return 400, {"code": "missing_digest",
                         "message": "send sha256:<hex> of the ZIP as X-Content-Digest"}
        if self.require_signature:
            ts = (headers.get("X-Timestamp") or "").strip()
            sig = (headers.get("X-Signature") or "").strip()
            try:
                skew = abs(self.clock() - float(ts))
            except ValueError:
                skew = float("inf")
            if skew > FRESHNESS_WINDOW_SECONDS:
                return 400, {"code": "stale_timestamp",
                             "message": "X-Timestamp must be current unix seconds "
                                        f"(±{FRESHNESS_WINDOW_SECONDS:.0f}s)"}
            msg = canonical_fund_message("submit", hotkey, declared, ts)
            if not sig or not self.verify(hotkey, msg, sig):
                return 400, {"code": "bad_signature",
                             "message": "X-Signature must be the hotkey's sr25519 "
                                        "signature over the canonical message "
                                        "(which binds X-Content-Digest)"}
        return None, {"hotkey": hotkey, "declared": declared}

    def submit(self, headers, body: bytes) -> tuple[int, dict]:
        """Store a generator ZIP; optionally fund it in the same request.

        The signature binds the CONTENT: the miner signs over
        ``X-Content-Digest`` (``sha256:<hex>`` of the ZIP they built), and the
        server recomputes the digest from the received body — a tampered or
        truncated upload fails ``digest_mismatch`` before anything stores.
        """
        import hashlib

        status, ctx = self.submit_gate(headers)
        if status is not None:
            return status, ctx
        self.sweep_pending()
        hotkey, declared = ctx["hotkey"], ctx["declared"]
        actual = f"sha256:{hashlib.sha256(body).hexdigest()}"
        if actual != declared:
            return 400, {"code": "digest_mismatch",
                         "message": f"body hashes to {actual}, header declared "
                                    f"{declared} — corrupted upload?"}
        from ..interface.validation import format_commit
        from ..shared.hippius import StorageError
        from .store import DigestOwned, SubmissionTooLarge, vault_ref

        # Dispatch HTTP status on the exception TYPE, never by substring-matching
        # the message — the message can carry the attacker-chosen member name,
        # so a member named "zip_too_large" must not steer the response code
        # (review 2026-08-29). Operator-side faults (OSError ENOSPC, MemoryError)
        # are NOT StorageError, so they propagate to a 500 rather than a 400.
        try:
            digest = self.store.put(body, hotkey)
        except SubmissionTooLarge as e:
            return 413, {"code": "zip_too_large", "message": str(e)}
        except DigestOwned as e:
            return 409, {"code": "digest_owned", "message": str(e)}
        except StorageError as e:
            return 400, {"code": "bad_zip", "message": str(e)}
        ref = vault_ref(digest)
        funding, note = "none", ""
        api_key = (headers.get("X-Lium-Api-Key") or "").strip()
        if api_key:
            # One-request flow: the key vaults now, the entry parks until the
            # miner's chain reveal resolves (the sweep promotes it). Vault
            # AFTER the store accepted, mirroring fund(). The response tells
            # the truth about what parked: a live entry for a DIFFERENT ref
            # is never silently replaced — the miner re-funds the new ref
            # deliberately (or withdraws the old one first).
            outcome = self.queue.add_pending(hotkey, ref)
            self.vault.insert(hotkey, api_key)
            if outcome in ("pending_reveal", "already-pending"):
                funding = "pending_reveal"
            else:
                # add_pending returned "already-queued": a live queued/in_round
                # entry blocked it. Distinguish the idempotent SAME-ref
                # re-submit (already funded — the truth) from a genuine
                # different-ref collision (the miner must withdraw first).
                existing = self.queue.get(hotkey)
                if existing is not None and existing.ref == ref:
                    funding = "already-funded"
                else:
                    funding = "blocked-by-existing-entry"
                    note = ("your queue already holds a live entry for a "
                            "different ref — withdraw it or let it settle, then "
                            "`cascade fund` this new ref")
        log.info("submit %s: stored %s (funding=%s)", hotkey, digest, funding)
        body_out = {"status": "stored", "ref": ref,
                    "commit_payload": format_commit(ref), "funding": funding}
        if note:
            body_out["funding_note"] = note
        return 201, body_out

    def fund(self, headers) -> tuple[int, dict]:
        self.sweep_pending()
        status, ctx = self._auth("fund", headers)
        if status is not None:
            return status, ctx
        api_key = (headers.get("X-Lium-Api-Key") or "").strip()
        if not api_key:
            return 400, {"code": "missing_lium_api_key",
                         "message": "a funded entry needs your Lium API key "
                                    "as X-Lium-Api-Key"}
        hotkey, ref = ctx["hotkey"], ctx["ref"]
        reveal_block = self.resolve_reveal(hotkey, ref)
        if reveal_block is None:
            return 403, {"code": "not_revealed",
                         "message": "no revealed commitment for this hotkey matches "
                                    "X-Commit-Ref — reveal first, then fund"}
        outcome = self.queue.add(hotkey, ref, int(reveal_block))
        # Vault AFTER the queue accepts (mirrors PRISM: creds stored only once
        # the row is real) — and on every accepted outcome, so a re-fund of a
        # queued entry refreshes a key that may be nearing its TTL.
        self.vault.insert(hotkey, api_key)
        log.info("fund %s: %s (reveal_block=%d)", hotkey, outcome, reveal_block)
        if outcome == "already-queued":
            return 200, {"status": "already-queued"}
        return 202, {"status": outcome}

    def withdraw(self, headers) -> tuple[int, dict]:
        status, ctx = self._auth("withdraw", headers)
        if status is not None:
            return status, ctx
        hotkey = ctx["hotkey"]
        if not self.queue.withdraw(hotkey):
            return 409, {"code": "not_queued",
                         "message": "only a queued entry can withdraw — an entry "
                                    "already in a round runs to its verdict"}
        self.vault.remove(hotkey)
        log.info("withdraw %s", hotkey)
        return 200, {"status": "withdrawn"}

    def queue_view(self) -> tuple[int, dict]:
        self.sweep_pending()
        return 200, self.queue.public_view()

    # ── HTTP server ──────────────────────────────────────────────────────────

    def make_server(self, host: str, port: int, *,
                    max_connections: int = 64,
                    request_timeout_s: float = 30.0) -> ThreadingHTTPServer:
        """The intake's HTTP server, with in-app DoS backstops.

        Volumetric DDoS is the FRONT PROXY's job (the runbook mandates one);
        these are the defence-in-depth floors that keep the orchestrator — the
        box holding the trainer wallet and eval pool — degrading gracefully
        even if the proxy is missing or misconfigured:

        * ``request_timeout_s`` is the per-socket read/write timeout, so a
          slowloris client dribbling header bytes cannot pin a handler thread
          forever (the stdlib default is NO timeout).
        * ``max_connections`` bounds concurrent handler threads — the stdlib
          ``ThreadingHTTPServer`` spawns unboundedly. Over the cap, the
          connection gets an immediate 503 with Retry-After and is closed
          WITHOUT spawning a thread, so a connection flood costs one small
          write each instead of a thread + fd each.
        """
        intake = self

        class Handler(BaseHTTPRequestHandler):
            # Per-socket read/write timeout. StreamRequestHandler.setup() calls
            # settimeout() with this, and handle_one_request turns the timeout
            # into a closed connection — the thread is reclaimed instead of
            # parked on a client that stopped sending.
            timeout = request_timeout_s

            # Default BaseHTTPRequestHandler logging writes the request line to
            # stderr; route through our logger and keep it header-free (the
            # request line never carries the key — it is a header — but stay
            # deliberate about what gets logged here).
            def log_message(self, fmt: str, *args) -> None:  # noqa: A003
                log.debug("%s %s", self.address_string(), fmt % args)

            def _reply(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _dispatch(self, fn) -> None:
                # A handler exception must come back as a clean 500, not a
                # connection reset: under ThreadingHTTPServer a raced write or
                # any internal error would otherwise kill just this thread and
                # leave the miner with a hung/EOF'd request and no error code.
                try:
                    status, body = fn()
                except Exception:  # noqa: BLE001 — the reply is the error channel
                    log.exception("intake handler failed for %s", self.path)
                    status, body = 500, {"code": "internal_error",
                                         "message": "intake error; retry shortly"}
                self._reply(status, body)

            def do_GET(self) -> None:  # noqa: N802 — http.server API
                if self.path == "/health":
                    self._reply(200, {"status": "ok"})
                elif self.path == "/v1/queue":
                    self._dispatch(intake.queue_view)
                else:
                    self._reply(404, {"code": "not_found"})

            def do_POST(self) -> None:  # noqa: N802 — http.server API
                # A Content-Length that is missing → 0, and one that is
                # non-numeric or NEGATIVE is rejected outright: int() on junk
                # would raise (killing the connection with no reply), and a
                # negative length would make rfile.read(length) an unbounded
                # read-to-EOF, slipping past the declared-length cap
                # (review 2026-08-29).
                raw_len = self.headers.get("Content-Length")
                if raw_len is None:
                    length = 0
                else:
                    try:
                        length = int(raw_len)
                    except ValueError:
                        length = -1
                    if length < 0:
                        self._reply(400, {"code": "bad_content_length",
                                          "message": "Content-Length must be a "
                                                     "non-negative integer"})
                        return
                if self.path == "/v1/submit":
                    cap = getattr(getattr(intake, "store", None), "max_bytes", 0) or (1 << 20)
                    if length > cap:
                        # Reject on the DECLARED length before reading — a
                        # too-large upload must not stream through first.
                        self._reply(413, {"code": "zip_too_large",
                                          "message": f"declared {length} bytes > cap {cap}"})
                        return
                    # Identity/signature gate BEFORE buffering the upload: an
                    # unauthenticated request never costs more than headers.
                    status, err = intake.submit_gate(self.headers)
                    if status is not None:
                        self._reply(status, err)
                        return
                    body = self.rfile.read(length) if length else b""
                    self._dispatch(lambda: intake.submit(self.headers, body))
                    return
                # Everything else is header-only; drain the body so keep-alive
                # clients are not desynced.
                if length:
                    self.rfile.read(min(length, 1 << 16))
                if self.path == "/v1/fund":
                    self._dispatch(lambda: intake.fund(self.headers))
                elif self.path == "/v1/withdraw":
                    self._dispatch(lambda: intake.withdraw(self.headers))
                else:
                    self._reply(404, {"code": "not_found"})

        overloaded = json.dumps(
            {"code": "overloaded", "message": "intake at connection capacity; "
                                              "retry shortly"}).encode()
        reject = (b"HTTP/1.1 503 Service Unavailable\r\n"
                  b"Content-Type: application/json\r\n"
                  b"Content-Length: " + str(len(overloaded)).encode() + b"\r\n"
                  b"Retry-After: 2\r\n"
                  b"Connection: close\r\n"
                  b"\r\n" + overloaded)

        class BoundedServer(ThreadingHTTPServer):
            # Re-assert the ThreadingHTTPServer default we depend on: daemon
            # handler threads are never joined at server_close(), so a client
            # mid-request cannot hold shutdown hostage.
            daemon_threads = True

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._slots = threading.BoundedSemaphore(max_connections)

            def process_request(self, request, client_address) -> None:
                if not self._slots.acquire(blocking=False):
                    try:
                        request.settimeout(request_timeout_s)
                        request.sendall(reject)
                    except OSError:
                        pass  # client already gone — the close below is enough
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    # Thread never started (spawn failure) → the slot would
                    # leak; on success process_request_thread owns the release.
                    self._slots.release()
                    raise

            def process_request_thread(self, request, client_address) -> None:
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self._slots.release()

        return BoundedServer((host, port), Handler)
