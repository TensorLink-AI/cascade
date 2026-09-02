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
        X-Timestamp:     unix seconds, ±FRESHNESS_WINDOW_SECONDS, strictly
                         increasing per (action, hotkey) — a re-signed retry
                         needs a fresh timestamp
        X-Signature:     hex sr25519 by the hotkey over canonical_fund_message
                         (v2: binds sha256 of the API key too, so an on-path
                         replay cannot attach or swap a key)
    → 202 {"status": "queued" | "replaced"}  |  200 "already-queued"
    → 400 missing_lium_api_key / missing_hotkey / missing_ref /
          stale_timestamp / bad_signature      |  403 not_revealed / not_registered
    → 409 replayed_timestamp | 503 registration_unavailable

    POST /v1/submit     (body = the generator ZIP; same identity headers,
        X-Content-Digest: sha256:<hex> of the body — the SIGNED binding —
        and optionally X-Lium-Api-Key to fund in the same request)
    → 201 {"status": "stored", "ref": "vault/direct@sha256:…",
           "commit_payload": "metro-v1:gen:hippius:vault/direct@sha256:…",
           "funding": "pending_reveal" | "already-funded" | "none" |
                      "blocked-by-existing-entry"}  # last carries a funding_note
    → 400 digest_mismatch / missing_digest / bad_content_length | 409 digest_owned
    → 403 not_registered (submissions require a hotkey registered on the
          subnet — self-minted keypairs cannot fill the private store)
    → 413 zip_too_large | 429 quota_exceeded | 503 submissions_disabled / busy

    POST /v1/withdraw   (same auth headers, no key needed; with no live queue
        entry it still forgets the vaulted key → 200 "key-forgotten")
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
    "header_key_digest",
    "verify_hotkey_signature",
]

log = logging.getLogger("cascade.funding.intake")

# Replay bound on the signed timestamp. Funding is idempotent, so the only
# replay of consequence is re-queueing after a withdraw — a five-minute
# window plus per-request freshness closes it without clock heroics.
FRESHNESS_WINDOW_SECONDS = 300.0

_ACTIONS = ("fund", "withdraw", "submit")


def canonical_fund_message(action: str, hotkey: str, ref: str, timestamp: str,
                           key_digest: str = "-") -> bytes:
    """The byte string the miner's hotkey signs. Versioned; fields ``:``-joined.

    ``ref`` is a Hub ``repo@digest`` and hotkeys are SS58, and the version tag
    pins the layout so a future field addition is a new version, not a silent
    re-parse. v2 (review 2026-09-02) appends ``key_digest`` — sha256 hex of
    the ``X-Lium-Api-Key`` value, or ``-`` when the request carries no key —
    so the signature binds WHICH key funds the entry: an on-path replay of a
    captured request can no longer attach its own key header (overwriting the
    victim's vaulted working key) or strip/swap the one that was sent.
    """
    if action not in _ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    return f"cascade-{action}:v2:{hotkey}:{ref}:{timestamp}:{key_digest}".encode()


def header_key_digest(headers) -> str:
    """The canonical-message key field for a request's headers."""
    import hashlib

    api_key = (headers.get("X-Lium-Api-Key") or "").strip()
    if not api_key:
        return "-"
    return hashlib.sha256(api_key.encode()).hexdigest()


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

    ``resolve_registered(hotkey) -> bool | None`` is the subnet-membership
    oracle: True/False for a definite answer, ``None`` when the chain view is
    unavailable (→ 503, fail closed). ``None`` for the whole oracle disables
    the check (dev / --trust-refs). Fund is already gated by the reveal check
    (a reveal requires a registered commit), but submit happens BEFORE any
    chain action — without this gate any self-minted keypair could fill the
    private store (review 2026-09-02).
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
        resolve_registered: Callable[[str], bool | None] | None = None,
    ) -> None:
        self.queue = queue
        self.vault = vault
        self.resolve_reveal = resolve_reveal
        self.require_signature = require_signature
        self.verify = verify
        self.clock = clock
        self.store = store
        self.resolve_registered = resolve_registered
        # Replay floor: highest signature-verified timestamp per (action,
        # hotkey). In-memory only — the freshness window is 300s, so a
        # restart's forgotten floor re-opens at most that window.
        self._replay_lock = threading.Lock()
        self._last_ts: dict[tuple[str, str], float] = {}

    def sweep_pending(self) -> int:
        """Promote submit-with-key entries whose chain reveal has landed.

        Lazy, request-driven (no background thread): runs at the top of every
        fund/submit/queue read, so a pending entry goes live within one
        interaction — or at latest the trainer's own queue read — of its
        reveal resolving. Never raises: a chain hiccup here must not fail the
        request that triggered the sweep.
        """
        try:
            # Piggy-back the vault's TTL purge on the same request-driven
            # cadence: without it an expired key's plaintext survives on disk
            # until that hotkey is get()'d or the process restarts.
            self.vault.purge_expired()
        except Exception:  # noqa: BLE001 — purge is best-effort housekeeping
            log.exception("vault TTL purge failed (will retry next request)")
        try:
            return self.queue.promote_pending(self.resolve_reveal)
        except Exception:  # noqa: BLE001 — sweep is best-effort by design
            log.exception("pending-reveal sweep failed (will retry next request)")
            return 0

    # ── request handling (returns (http_status, body_dict)) ──────────────────

    def _parse_fresh_timestamp(self, ts: str) -> float | None:
        """The parsed timestamp when fresh, else ``None``.

        The comparison is ``not (skew <= window)`` rather than ``skew >
        window``: ``float("nan")`` parses, and NaN compares False BOTH ways —
        the ``>`` form would treat a ``X-Timestamp: nan`` signature as fresh
        forever, an eternal replay token (review 2026-09-02).
        """
        try:
            tsf = float(ts)
            skew = abs(self.clock() - tsf)
        except ValueError:
            return None
        if not (skew <= FRESHNESS_WINDOW_SECONDS):
            return None
        return tsf

    def _register_timestamp(self, action: str, hotkey: str, tsf: float) -> bool:
        """Record a VERIFIED signature's timestamp; False = replay.

        Timestamps must be strictly increasing per (action, hotkey): within
        the freshness window a captured request replays verbatim — the
        damaging case is a re-played withdraw landing after the miner
        re-funded (review 2026-09-02). Runs only after signature
        verification, so unverified garbage cannot poison the floor.
        """
        now = self.clock()
        with self._replay_lock:
            if len(self._last_ts) > 4096:   # bound the table; old floors are dead
                cutoff = now - 2 * FRESHNESS_WINDOW_SECONDS
                self._last_ts = {k: v for k, v in self._last_ts.items()
                                 if v >= cutoff}
            key = (action, hotkey)
            last = self._last_ts.get(key)
            if last is not None and tsf <= last:
                return False
            self._last_ts[key] = tsf
            return True

    _REPLAY_BODY = {"code": "replayed_timestamp",
                    "message": "this (action, hotkey) already accepted an equal "
                               "or newer X-Timestamp — sign a fresh one and retry"}

    def _registration_gate(self, hotkey: str) -> tuple[int, dict] | None:
        """403/503 when the hotkey is not (provably) registered on the subnet."""
        if self.resolve_registered is None:
            return None
        try:
            registered = self.resolve_registered(hotkey)
        except Exception:  # noqa: BLE001 — an oracle crash is "unavailable"
            log.exception("registration oracle failed for %s", hotkey)
            registered = None
        if registered is None:
            return 503, {"code": "registration_unavailable",
                         "message": "cannot verify subnet registration right "
                                    "now; retry shortly"}
        if not registered:
            return 403, {"code": "not_registered",
                         "message": "this hotkey is not registered on the "
                                    "subnet — register first, then retry"}
        return None

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
            tsf = self._parse_fresh_timestamp(ts)
            if tsf is None:
                return 400, {"code": "stale_timestamp",
                             "message": "X-Timestamp must be current unix seconds "
                                        f"(±{FRESHNESS_WINDOW_SECONDS:.0f}s)"}
            msg = canonical_fund_message(action, hotkey, ref, ts,
                                         header_key_digest(headers))
            if not sig or not self.verify(hotkey, msg, sig):
                return 400, {"code": "bad_signature",
                             "message": "X-Signature must be the hotkey's sr25519 "
                                        "signature over the canonical message "
                                        "(v2 — it binds the key header too)"}
            if not self._register_timestamp(action, hotkey, tsf):
                return 409, dict(self._REPLAY_BODY)
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
            tsf = self._parse_fresh_timestamp(ts)
            if tsf is None:
                return 400, {"code": "stale_timestamp",
                             "message": "X-Timestamp must be current unix seconds "
                                        f"(±{FRESHNESS_WINDOW_SECONDS:.0f}s)"}
            msg = canonical_fund_message("submit", hotkey, declared, ts,
                                         header_key_digest(headers))
            if not sig or not self.verify(hotkey, msg, sig):
                return 400, {"code": "bad_signature",
                             "message": "X-Signature must be the hotkey's sr25519 "
                                        "signature over the canonical message "
                                        "(v2 — binds X-Content-Digest and the "
                                        "key header)"}
            if not self._register_timestamp("submit", hotkey, tsf):
                return 409, dict(self._REPLAY_BODY)
        # Subnet-membership gate, still header-only: storing code (and the
        # validation extraction it costs) is for registered participants.
        gate = self._registration_gate(hotkey)
        if gate is not None:
            return gate
        return None, {"hotkey": hotkey, "declared": declared}

    def submit(self, headers, body: bytes,
               gate_ctx: dict | None = None) -> tuple[int, dict]:
        """Store a generator ZIP; optionally fund it in the same request.

        The signature binds the CONTENT: the miner signs over
        ``X-Content-Digest`` (``sha256:<hex>`` of the ZIP they built), and the
        server recomputes the digest from the received body — a tampered or
        truncated upload fails ``digest_mismatch`` before anything stores.

        ``gate_ctx`` is the context an earlier :meth:`submit_gate` call
        returned (the HTTP layer gates before reading the body). Passing it
        skips re-running the gate — which would trip the strictly-increasing
        timestamp floor on the request's own second check.
        """
        import hashlib

        if gate_ctx is None:
            status, ctx = self.submit_gate(headers)
            if status is not None:
                return status, ctx
        else:
            ctx = gate_ctx
        self.sweep_pending()
        hotkey, declared = ctx["hotkey"], ctx["declared"]
        actual = f"sha256:{hashlib.sha256(body).hexdigest()}"
        if actual != declared:
            return 400, {"code": "digest_mismatch",
                         "message": f"body hashes to {actual}, header declared "
                                    f"{declared} — corrupted upload?"}
        from ..interface.validation import format_commit
        from ..shared.hippius import StorageError
        from .store import (DigestOwned, SubmissionQuotaExceeded,
                            SubmissionTooLarge, vault_ref)

        # Dispatch HTTP status on the exception TYPE, never by substring-matching
        # the message — the message can carry the attacker-chosen member name,
        # so a member named "zip_too_large" must not steer the response code
        # (review 2026-08-29). Operator-side faults (OSError ENOSPC, MemoryError)
        # are NOT StorageError, so they propagate to a 500 rather than a 400.
        try:
            digest = self.store.put(body, hotkey)
        except SubmissionQuotaExceeded as e:
            return 429, {"code": "quota_exceeded", "message": str(e)}
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
        # Belt over the reveal check's braces: a reveal implies the hotkey was
        # registered when it committed, but a since-deregistered hotkey has no
        # seat to fund.
        gate = self._registration_gate(ctx["hotkey"])
        if gate is not None:
            return gate
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
        if self.queue.withdraw(hotkey):
            self.vault.remove(hotkey)
            log.info("withdraw %s", hotkey)
            return 200, {"status": "withdrawn"}
        entry = self.queue.get(hotkey)
        if entry is not None and entry.status == "in_round":
            return 409, {"code": "not_queued",
                         "message": "an entry already in a round runs to its "
                                    "verdict — the key is retained until then "
                                    "(teardown needs it)"}
        # No live entry: nothing to unqueue, but honor the custody half — a
        # terminal (done/failed) or absent entry has no teardown claim on the
        # key, so the miner can make us forget it NOW rather than waiting out
        # the 36h TTL (review 2026-09-02).
        self.vault.remove(hotkey)
        log.info("withdraw %s: no live entry — key forgotten", hotkey)
        return 200, {"status": "key-forgotten"}

    def queue_view(self) -> tuple[int, dict]:
        self.sweep_pending()
        return 200, self.queue.public_view()

    # ── HTTP server ──────────────────────────────────────────────────────────

    def make_server(self, host: str, port: int, *,
                    max_connections: int = 64,
                    request_timeout_s: float = 30.0,
                    request_deadline_s: float = 120.0,
                    max_concurrent_uploads: int = 4) -> ThreadingHTTPServer:
        """The intake's HTTP server, with in-app DoS backstops.

        Volumetric DDoS is the FRONT PROXY's job (the runbook mandates one);
        these are the defence-in-depth floors that keep the orchestrator — the
        box holding the trainer wallet and eval pool — degrading gracefully
        even if the proxy is missing or misconfigured:

        * ``request_timeout_s`` is the per-socket read/write timeout, so a
          slowloris client dribbling header bytes cannot pin a handler thread
          forever (the stdlib default is NO timeout).
        * ``request_deadline_s`` is a whole-CONNECTION wall clock, enforced by
          a reaper thread that force-closes any socket older than it. The
          per-operation timeout alone is defeated by a drip-feed client that
          sends one byte per interval — each read succeeds, the connection
          never ends (review 2026-09-02). Connections are one-request
          (HTTP/1.0 close semantics), so no legitimate request outlives this.
        * ``max_connections`` bounds concurrent handler threads — the stdlib
          ``ThreadingHTTPServer`` spawns unboundedly. Over the cap, the
          connection gets an immediate 503 with Retry-After and is closed
          WITHOUT spawning a thread, so a connection flood costs one small
          write each instead of a thread + fd each.
        * ``max_concurrent_uploads`` bounds how many submit bodies buffer at
          once: each is up to the ZIP cap in RAM (plus the store's validation
          copy), so the worst case is this × ~3 × cap instead of
          ``max_connections`` × ~3 × cap.
        """
        intake = self
        upload_slots = threading.BoundedSemaphore(max_concurrent_uploads)

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
                    # The returned ctx rides into submit() — re-gating there
                    # would trip the strictly-increasing timestamp floor.
                    status, gate_out = intake.submit_gate(self.headers)
                    if status is not None:
                        self._reply(status, gate_out)
                        return
                    if not upload_slots.acquire(timeout=10.0):
                        self._reply(503, {"code": "busy",
                                          "message": "too many concurrent "
                                                     "uploads; retry shortly"})
                        return
                    try:
                        body = self.rfile.read(length) if length else b""
                        self._dispatch(lambda: intake.submit(
                            self.headers, body, gate_ctx=gate_out))
                    finally:
                        upload_slots.release()
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
                self._live_lock = threading.Lock()
                self._live: dict[int, tuple] = {}   # id(sock) → (sock, started)
                self._reaper_stop = threading.Event()
                threading.Thread(target=self._reap_over_deadline, daemon=True,
                                 name="intake-conn-reaper").start()

            def _reap_over_deadline(self) -> None:
                # The whole-connection deadline: force-close any socket older
                # than request_deadline_s. The handler's blocked read then
                # raises OSError, the thread unwinds, and its slot frees — a
                # drip-feed client cannot hold a slot past the deadline.
                import socket as _socket

                while not self._reaper_stop.wait(5.0):
                    now = time.monotonic()
                    with self._live_lock:
                        stuck = [(key, sock) for key, (sock, t0) in
                                 self._live.items()
                                 if now - t0 > request_deadline_s]
                    for key, sock in stuck:
                        log.warning("closing connection past the %.0fs "
                                    "deadline", request_deadline_s)
                        try:
                            sock.shutdown(_socket.SHUT_RDWR)
                        except OSError:
                            pass
                        try:
                            sock.close()
                        except OSError:
                            pass
                        with self._live_lock:
                            self._live.pop(key, None)

            def server_close(self) -> None:
                self._reaper_stop.set()
                super().server_close()

            def process_request(self, request, client_address) -> None:
                if not self._slots.acquire(blocking=False):
                    try:
                        request.settimeout(request_timeout_s)
                        request.sendall(reject)
                    except OSError:
                        pass  # client already gone — the close below is enough
                    self.shutdown_request(request)
                    return
                with self._live_lock:
                    self._live[id(request)] = (request, time.monotonic())
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    # Thread never started (spawn failure) → the slot would
                    # leak; on success process_request_thread owns the release.
                    self._release(request)
                    raise

            def _release(self, request) -> None:
                with self._live_lock:
                    self._live.pop(id(request), None)
                self._slots.release()

            def process_request_thread(self, request, client_address) -> None:
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self._release(request)

        return BoundedServer((host, port), Handler)
