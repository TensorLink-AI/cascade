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

    POST /v1/withdraw   (same auth headers, no key needed)
    GET  /v1/queue      (public transparency feed — no key material)
    GET  /health

Fail-closed at intake: a request that cannot fund a rental is rejected with
the code the miner needs to fix it, never accepted-and-stuck. There is
deliberately NO balance pre-check (PRISM's choice too) — the key is validated
by use, and an underfunded key surfaces as a classified rent failure that
requeues without burning.
"""

from __future__ import annotations

import json
import logging
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

_ACTIONS = ("fund", "withdraw")


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
    ) -> None:
        self.queue = queue
        self.vault = vault
        self.resolve_reveal = resolve_reveal
        self.require_signature = require_signature
        self.verify = verify
        self.clock = clock

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

    def fund(self, headers) -> tuple[int, dict]:
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
        return 200, self.queue.public_view()

    # ── HTTP server ──────────────────────────────────────────────────────────

    def make_server(self, host: str, port: int) -> ThreadingHTTPServer:
        intake = self

        class Handler(BaseHTTPRequestHandler):
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

            def do_GET(self) -> None:  # noqa: N802 — http.server API
                if self.path == "/health":
                    self._reply(200, {"status": "ok"})
                elif self.path == "/v1/queue":
                    self._reply(*intake.queue_view())
                else:
                    self._reply(404, {"code": "not_found"})

            def do_POST(self) -> None:  # noqa: N802 — http.server API
                # Bodies are ignored (the contract is header-only); drain so
                # keep-alive clients are not desynced.
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(min(length, 1 << 16))
                if self.path == "/v1/fund":
                    self._reply(*intake.fund(self.headers))
                elif self.path == "/v1/withdraw":
                    self._reply(*intake.withdraw(self.headers))
                else:
                    self._reply(404, {"code": "not_found"})

        return ThreadingHTTPServer((host, port), Handler)
