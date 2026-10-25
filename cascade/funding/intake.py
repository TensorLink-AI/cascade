"""The miner→operator funding endpoint: the one HTTP surface cascade exposes.

Everything else miners do rides the chain (commit/reveal) or public storage
(pull-only status files) — see ``docs/ARCHITECTURE.md``. An API key cannot
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
        X-Submission-Label: optional display name (≤32 chars, [A-Za-z0-9._-])
                         shown beside the hotkey in the public queue and
                         roster — presentational, unsigned, never identity
                         (DEC-CA-0043)
        X-Gpu-Type / X-Gpu-Allow-Fallback / X-Gpu-Max-Price: optional
                         miner-chosen GPU type (DEC-CA-0060,
                         :mod:`cascade.funding.gpu_choice`); when ANY is sent
                         the signature is over the v3 message, which appends
                         their values — none sent ⇒ the v2 bytes, unchanged
    → 202 {"status": "queued" | "replaced"}  |  200 "already-queued"
          (+ "gpu", "allow_fallback", "max_price_per_hour" and an advisory
          "warning" when a GPU type was chosen)
    → 400 missing_lium_api_key / missing_hotkey / missing_ref / bad_label /
          stale_timestamp / bad_signature / unknown_gpu / bad_gpu_fallback /
          bad_gpu_max_price / gpu_options_without_gpu / gpu_choice_unavailable
          | 403 not_revealed / not_registered
    → 409 replayed_timestamp | 503 registration_unavailable

    POST /v1/credentials  (DEC-CA-0055: rotate the Lium key a hotkey's legs —
        the v2 king's own leg included — rent on; same identity headers as
        /v1/fund: X-Miner-Hotkey, X-Lium-Api-Key, X-Commit-Ref (a REVEALED
        commitment of the hotkey), X-Timestamp, X-Signature over
        canonical_fund_message("credentials", …) binding sha256 of the key)
    → 200 {"status": "stored"} — the key replaces the hotkey's payer-vault key;
          nothing is queued and the one lifetime submission is untouched
    → 400 missing_lium_api_key / missing_hotkey / missing_ref / stale_timestamp /
          bad_signature | 403 not_revealed / not_registered | 409 replayed_timestamp

    POST /v1/submit     (body = the generator ZIP; same identity headers,
        X-Content-Digest: sha256:<hex> of the body — the SIGNED binding —
        and optionally X-Lium-Api-Key to fund in the same request, and
        X-Submission-Label as above)
    → 201 {"status": "stored", "ref": "vault/direct@sha256:…",
           "commit_payload": "metro-v1:gen:hippius:vault/direct@sha256:…",
           "funding": "pending_reveal" | "already-funded" | "none" |
                      "blocked-by-existing-entry"}  # last carries a funding_note
    → 400 digest_mismatch / missing_digest / bad_content_length | 409 digest_owned
    → 403 not_registered (submissions require a hotkey registered on the
          subnet — self-minted keypairs cannot fill the private store)
    → 413 zip_too_large | 429 quota_exceeded | 503 submissions_disabled / busy

    POST /v2/direct/credentials   (direct-MoE gangs; same identity headers)
        X-Compute-Provider: lium | vast | runpod — the marketplace to rent on
        X-Compute-Api-Key:  that marketplace's key, ideally scoped to instance
                            create/delete only (see cascade.provision.vast/runpod)
        X-Signature:        over canonical_fund_message("credentials", …) whose
                            key field is sha256("<provider>:<key>")
    → 200 {"status": "stored", "provider": …} | 400 unsupported_compute_provider /
          missing_compute_api_key | 403 not_revealed / not_registered

    POST /v1/lane-c/fund   (DEC-CA-0056 lane C — fund ONE fine-tune leg of the
        hotkey's revealed lane-C commitment
        ``metro-v1:ft:<target>:hippius:<ref>``; the ref is a recipe tree with
        ``marketcast.json`` and an optional generator)
        X-Miner-Hotkey, X-Lium-Api-Key, X-Commit-Ref (the recipe tree ref),
        X-Lane-C-Target: the declared target, one of ``[lane_c] targets``
                         (a forward-only competition's — today MarketCast
                         ``mkt:<suite>``; anything else is ``bad_target``)
        X-Timestamp, X-Signature over canonical_fund_message("lane-c-fund",
                         hotkey, "<target>|<ref>", ts, sha256(key)) — binds
                         the key digest, the target AND the ref
        X-Submission-Label (optional, as above)
        X-Gpu-Type / X-Gpu-Allow-Fallback / X-Gpu-Max-Price (optional, as
                         /v1/fund — the v3 message then appends them)
    → 202 {"status": "queued" | "replaced" | "queued-behind", "lane": "c",
           "target": …} | 200 "already-queued"
    → 400 missing_target / bad_target / unknown_target / missing_lium_api_key /
          stale_timestamp / bad_signature | 403 not_revealed / not_registered |
          409 replayed_timestamp | 409 competition_ended (its until_block
          passed: nothing new is admitted) | 503 lane_c_disabled
      The entry queues into the LANE-C queue (``lane_c_queue.json`` beside the
      challenger queue — never the challenger queue) and the key lands in the
      shared payer vault (one key per hotkey). It consumes NO lane-A lifetime
      submission. Infra faults requeue without burning; an invalid
      ``marketcast.json`` (or a generator fault) is the miner's and ends the
      entry.
    POST /v1/lane-c/withdraw  (same headers minus the key, signed as
        "lane-c-withdraw") → 200 "withdrawn" | 409 not_queued
    GET  /v1/lane-c/queue     (public lane-C queue: hotkey, target, ref, status)

    POST /v1/lane-b/fund   (DEC-CA-0059 item 8 — fund ONE domain-expert leg of
        the hotkey's revealed lane-B commitment
        ``metro-v1:expert:<domain|codes>:hippius:<ref>``)
        X-Miner-Hotkey, X-Lium-Api-Key, X-Commit-Ref (the generator ref),
        X-Lane-B-Targets: the declared targets, comma-separated full names
                         (``energy`` | ``energy,web_cloudops,horizon:720``),
                         1..``[lane_b] max_domains_per_leg`` of ``[lane_b]
                         targets`` — exactly the commitment's
        X-Timestamp, X-Signature over canonical_fund_message("lane-b-fund",
                         hotkey, "<targets as sent>|<ref>", ts, sha256(key))
        X-Submission-Label, X-Gpu-Type / X-Gpu-Allow-Fallback /
                         X-Gpu-Max-Price (optional, as /v1/fund)
    → 202 {"status": "queued" | "replaced" | "queued-behind", "lane": "b",
           "targets": […]} | 200 "already-queued"
    → 400 missing_targets / bad_targets / unknown_target / too_many_targets /
          missing_lium_api_key / stale_timestamp / bad_signature |
          403 not_revealed / not_registered | 409 replayed_timestamp |
          503 lane_b_disabled
      Queues into the LANE-B queue (``lane_b_queue.json`` beside the
      challenger queue); the key lands in the shared payer vault. No lane-A
      lifetime submission is consumed; infra faults requeue without burning,
      a capacity miss waits (``no_capacity``), the operator never pays.
    POST /v1/lane-b/withdraw  (same headers minus the key, signed as
        "lane-b-withdraw") → 200 "withdrawn" | 409 not_queued
    GET  /v1/lane-b/queue     (public lane-B queue: hotkey, targets, ref, status)

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

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .gpu_choice import GpuCatalog, GpuChoice, parse_gpu_headers, wall_warning
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

_ACTIONS = ("fund", "withdraw", "submit", "credentials",
            # DEC-CA-0056 lane C: the signed ref field is lane_c_signed_ref()
            "lane-c-fund", "lane-c-withdraw",
            # DEC-CA-0059 lane B: the signed ref field is lane_b_signed_ref()
            "lane-b-fund", "lane-b-withdraw")

LANE_C_TARGET_HEADER = "X-Lane-C-Target"
LANE_B_TARGETS_HEADER = "X-Lane-B-Targets"


def lane_b_targets(header_value: str) -> tuple[str, ...]:
    """The declared targets of an ``X-Lane-B-Targets`` value, in the order
    sent (each stripped; empty items dropped)."""
    return tuple(t.strip() for t in str(header_value or "").split(",") if t.strip())


def lane_b_signed_ref(targets, ref: str) -> str:
    """The ``ref`` field a lane-B request signs: ``"<t1>,<t2>|<ref>"`` with
    the targets EXACTLY as the request lists them — the signature binds the
    declared targets and the ref (and, through the canonical message, the key
    digest). The queue stores them in the ``[lane_b]`` slot order."""
    from .queue import lane_b_entry_ref

    return lane_b_entry_ref(targets, ref)


def lane_c_signed_ref(target: str, ref: str) -> str:
    """The ``ref`` field a lane-C request signs: ``"<target>|<ref>"`` — the
    signature binds the declared target as well as the ref (and, through the
    canonical message, the key digest)."""
    from .queue import lane_c_entry_ref

    return lane_c_entry_ref(target, ref)


def canonical_fund_message(action: str, hotkey: str, ref: str, timestamp: str,
                           key_digest: str = "-", gpu_segment: str = "") -> bytes:
    """The byte string the miner's hotkey signs. Versioned; fields ``:``-joined.

    ``ref`` is a Hub ``repo@digest`` and hotkeys are SS58, and the version tag
    pins the layout so a future field addition is a new version, not a silent
    re-parse. v2 (review 2026-09-02) appends ``key_digest`` — sha256 hex of
    the ``X-Lium-Api-Key`` value or the entire ``X-Payer-Credentials`` bundle,
    or ``-`` when the request carries no key —
    so the signature binds WHICH key funds the entry: an on-path replay of a
    captured request can no longer attach its own key header (overwriting the
    victim's vaulted working key) or strip/swap the one that was sent.

    v3 (DEC-CA-0060) appends ``gpu_segment`` — the miner's GPU choice
    (:func:`cascade.funding.gpu_choice.gpu_signed_segment`) — and is used ONLY
    when the request carries one: ``gpu_segment == ""`` is the v2 message,
    byte for byte, so every existing tool keeps signing what it signs today.
    """
    if action not in _ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    if gpu_segment:
        return (f"cascade-{action}:v3:{hotkey}:{ref}:{timestamp}:{key_digest}:"
                f"{gpu_segment}").encode()
    return f"cascade-{action}:v2:{hotkey}:{ref}:{timestamp}:{key_digest}".encode()


LABEL_HEADER = "X-Submission-Label"


def submission_label(headers) -> tuple[str, dict | None]:
    """``(label, None)`` from the optional label header, or ``("", error)``.

    The label is OUTSIDE the signed canonical message on purpose: it is a
    display name with no security role (a replayed or tampered label can
    at worst mislabel the replayer's own row — the row is keyed by the
    signed hotkey), and folding it in would bump the message version for
    every miner tool. Validation is the queue's :func:`normalize_label`, so
    the intake and a direct queue write agree on what a label may be.
    """
    from .queue import normalize_label

    try:
        return normalize_label(headers.get(LABEL_HEADER) or ""), None
    except ValueError as e:
        return "", {"code": "bad_label", "message": str(e)}


def header_key_digest(headers) -> str:
    """The canonical-message key field for a request's headers."""
    import hashlib

    from .providers import header_credentials

    api_key = header_credentials(headers)
    if not api_key:
        return "-"
    return hashlib.sha256(api_key.encode()).hexdigest()


def compute_key_digest(provider: str, api_key: str) -> str:
    """The canonical-message key field for a direct credential deposit: binds
    WHICH marketplace as well as which key, so an on-path replay can neither
    swap the key nor re-file it under another provider."""
    import hashlib

    return hashlib.sha256(f"{provider}:{api_key}".encode()).hexdigest()


def verify_hotkey_signature(hotkey: str, message: bytes, signature_hex: str) -> bool:
    """sr25519 verification against the hotkey's SS58 address.

    A malformed or mismatched signature is False. A missing/broken bittensor
    SDK is NOT a False: it raises ``SignatureVerifierUnavailable`` (the
    request handler turns it into a logged 500), because answering False
    would reject every valid miner signature with a misleading
    ``bad_signature`` (DEC-CA-0051).
    """
    from ..shared.sr25519 import verify_hotkey

    return verify_hotkey(hotkey, message, signature_hex)


class PerIpLimiter:
    """Sliding-window request cap per client IP (in-app backstop).

    ``rpm`` requests per rolling 60s per IP; ``max_connections`` concurrent
    connections per IP. Both are BACKSTOPS for a proxy-less intake — behind
    the mandated TLS proxy every connection arrives from the proxy's IP, so
    the server honours ``X-Forwarded-For`` only when the peer IS the
    configured ``trusted_proxy`` (otherwise the header is attacker-supplied
    and ignored). ``rpm = 0`` disables the request cap.
    """

    WINDOW = 60.0
    MAX_TRACKED = 20_000    # bound the table; idle IPs are evicted on pressure

    def __init__(self, rpm: int = 60, max_connections: int = 8,
                 clock: Callable[[], float] = time.time) -> None:
        self.rpm = int(rpm)
        self.max_connections = int(max_connections)
        self.clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}
        self._open: dict[str, int] = {}

    def allow_request(self, ip: str) -> bool:
        if self.rpm <= 0:
            return True
        now = self.clock()
        with self._lock:
            if len(self._hits) > self.MAX_TRACKED:
                cutoff = now - self.WINDOW
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] >= cutoff}
            hits = [t for t in self._hits.get(ip, ()) if t > now - self.WINDOW]
            if len(hits) >= self.rpm:
                self._hits[ip] = hits
                return False
            hits.append(now)
            self._hits[ip] = hits
            return True

    def connection_open(self, ip: str) -> bool:
        """Claim a connection slot for ``ip``; False when it is at its cap."""
        if self.max_connections <= 0:
            return True
        with self._lock:
            n = self._open.get(ip, 0)
            if n >= self.max_connections:
                return False
            self._open[ip] = n + 1
            return True

    def connection_close(self, ip: str) -> None:
        with self._lock:
            n = self._open.get(ip, 0) - 1
            if n <= 0:
                self._open.pop(ip, None)
            else:
                self._open[ip] = n


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
        direct_intake: object | None = None,
        compute_vaults: dict | None = None,
        require_signature: bool = True,
        verify: Callable[[str, bytes, str], bool] = verify_hotkey_signature,
        clock: Callable[[], float] = time.time,
        store: object | None = None,   # SubmissionStore; None = submissions off
        resolve_registered: Callable[[str], bool | None] | None = None,
        lane_c_queue: object | None = None,       # LaneCQueue; None = lane C off
        resolve_lane_c_reveal: Callable[[str, str, str], int | None] | None = None,
        lane_c_targets: tuple[str, ...] | None = None,
        mkt_suites: tuple[str, ...] = (),
        lane_c_closed: Callable[[str], str | None] | None = None,
        gpu_catalog: GpuCatalog | None = None,
        lane_b_queue: object | None = None,       # funding LaneBQueue; None = lane B off
        resolve_lane_b_reveal: Callable[[str, tuple, str], int | None] | None = None,
        lane_b_targets: tuple[str, ...] | None = None,
        lane_b_max_targets: int | None = None,
    ) -> None:
        self.direct_intake = direct_intake
        # DEC-CA-0059 item 8 lane B: the domain-expert funding queue + its
        # reveal oracle ``(hotkey, targets, ref) -> reveal block | None`` (the
        # hotkey's revealed ``metro-v1:expert:`` commitment must declare
        # exactly these targets and this ref), the configured slot list and
        # the per-leg cap (None = accept any well-formed list; the trainer
        # re-checks).
        self.lane_b_queue = lane_b_queue
        self.resolve_lane_b_reveal = resolve_lane_b_reveal
        self.lane_b_targets = tuple(lane_b_targets) if lane_b_targets is not None else None
        self.lane_b_max_targets = (int(lane_b_max_targets) if lane_b_max_targets is not None
                                   else None)
        # DEC-CA-0060: the SKUs a fund request may choose (``[round]
        # funded_pod_skus``) and the walls behind the fund-time warning. None
        # = no chain config (dev): any well-formed SKU, no warning.
        self.gpu_catalog = gpu_catalog
        # DEC-CA-0056 lane C: the fine-tune queue + its reveal oracle
        # ``(hotkey, target, ref) -> reveal block | None`` (the hotkey's
        # revealed ``metro-v1:ft:`` commitment must name exactly them), and
        # the configured target list (None = accept any well-formed target;
        # the trainer skips unknown ones).
        self.lane_c_queue = lane_c_queue
        self.resolve_lane_c_reveal = resolve_lane_c_reveal
        self.lane_c_targets = tuple(lane_c_targets) if lane_c_targets is not None else None
        # DEC-CA-0064: MarketCast suites admitted as mkt:<suite> lane-C
        # targets (``[marketcast] target_suites``; none while it is off)
        self.mkt_suites = tuple(mkt_suites)
        # concurrent competitions (owner 2026-10-20): ``target -> refusal |
        # None`` at the current chain block — a competition past its
        # ``until_block`` admits nothing new (None = no end check: dev)
        self.lane_c_closed = lane_c_closed
        # marketplace → PayerKeyVault for direct-MoE gangs (Lium shares ``vault``).
        self.compute_vaults = compute_vaults or {}
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

    def _auth(self, action: str, headers, key_digest: str | None = None,
              signed_ref: str | None = None, gpu_segment: str = "",
              ) -> tuple[int, dict] | tuple[None, dict]:
        """Shared identity/signature gate; (None, ctx) when the request passes.
        ``signed_ref`` (lane C) replaces ``X-Commit-Ref`` in the signed
        message; ``ctx["ref"]`` stays the header's ref. ``gpu_segment``
        (DEC-CA-0060) selects the v3 message when non-empty."""
        from .providers import header_credentials

        try:
            header_credentials(headers)
        except ValueError as exc:
            return 400, {"code": "bad_provider_credentials", "message": str(exc)}
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
            msg = canonical_fund_message(action, hotkey,
                                         ref if signed_ref is None else signed_ref, ts,
                                         key_digest or header_key_digest(headers),
                                         gpu_segment)
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
        from .providers import header_credentials

        try:
            header_credentials(headers)
        except ValueError as exc:
            return 400, {"code": "bad_provider_credentials", "message": str(exc)}
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
        from .store import DigestOwned, SubmissionQuotaExceeded, SubmissionTooLarge, vault_ref

        label, label_err = submission_label(headers)
        if label_err is not None:
            return 400, label_err

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
        from .providers import header_credentials

        try:
            api_key = header_credentials(headers)
        except ValueError as exc:
            return 400, {"code": "bad_provider_credentials", "message": str(exc)}
        if api_key:
            # One-request flow: the key vaults now, the entry parks until the
            # miner's chain reveal resolves (the sweep promotes it). Vault
            # AFTER the store accepted, mirroring fund(). The response tells
            # the truth about what parked: a live entry for a DIFFERENT ref
            # is never silently replaced — the miner re-funds the new ref
            # deliberately (or withdraws the old one first).
            outcome = self.queue.add_pending(hotkey, ref, label=label)
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

    def _gpu_choice(self, headers) -> tuple[GpuChoice, str, dict | None]:
        """The request's GPU choice (DEC-CA-0060), checked header-only BEFORE
        the signature (an unknown SKU is refused without a chain read)."""
        return parse_gpu_headers(headers, self.gpu_catalog)

    def _gpu_reply(self, body: dict, choice: GpuChoice) -> dict:
        """``body`` plus the chosen GPU and, when its measured wall cannot fit
        the budget, the advisory ``warning`` — never a refusal. The default
        choice adds nothing (today's reply bytes)."""
        if not choice.chosen:
            return body
        out = {**body, **choice.as_dict()}
        warn = wall_warning(choice, self.gpu_catalog)
        if warn:
            out["warning"] = warn
        return out

    def fund(self, headers) -> tuple[int, dict]:
        self.sweep_pending()
        choice, segment, gpu_err = self._gpu_choice(headers)
        if gpu_err is not None:
            return 400, gpu_err
        status, ctx = self._auth("fund", headers, gpu_segment=segment)
        if status is not None:
            return status, ctx
        # Belt over the reveal check's braces: a reveal implies the hotkey was
        # registered when it committed, but a since-deregistered hotkey has no
        # seat to fund.
        gate = self._registration_gate(ctx["hotkey"])
        if gate is not None:
            return gate
        from .providers import header_credentials

        try:
            api_key = header_credentials(headers)
        except ValueError as exc:
            return 400, {"code": "bad_provider_credentials", "message": str(exc)}
        if not api_key:
            return 400, {"code": "missing_lium_api_key",
                         "message": "a funded entry needs your Lium API key "
                                    "as X-Lium-Api-Key"}
        label, label_err = submission_label(headers)
        if label_err is not None:
            return 400, label_err
        hotkey, ref = ctx["hotkey"], ctx["ref"]
        reveal_block = self.resolve_reveal(hotkey, ref)
        if reveal_block is None:
            return 403, {"code": "not_revealed",
                         "message": "no revealed commitment for this hotkey matches "
                                    "X-Commit-Ref — reveal first, then fund"}
        outcome = self.queue.add(hotkey, ref, int(reveal_block), label=label,
                                 gpu_choice=choice)
        # Vault AFTER the queue accepts (mirrors PRISM: creds stored only once
        # the row is real) — and on every accepted outcome, so a re-fund of a
        # queued entry refreshes a key that may be nearing its TTL.
        self.vault.insert(hotkey, api_key)
        log.info("fund %s: %s (reveal_block=%d%s)", hotkey, outcome, reveal_block,
                 f", gpu={choice.gpu}" if choice.chosen else "")
        if outcome == "already-queued":
            return 200, self._gpu_reply({"status": "already-queued"}, choice)
        return 202, self._gpu_reply({"status": outcome}, choice)

    def credentials(self, headers) -> tuple[int, dict]:
        """DEC-CA-0055 key rotation: replace a hotkey's Lium key in the payer
        vault at any time — the v2 king keeps its own leg funded (a failing
        key slashes it to 1 % of lane A). The ``/v1/fund`` identity gate
        (signature binding the key digest, fresh strictly-increasing
        timestamp, subnet registration, a revealed commitment matching
        ``X-Commit-Ref``); NOTHING is queued and the lifetime-submission rule
        is never consulted — a rotation is custody, not a submission."""
        from .providers import header_credentials

        try:
            api_key = header_credentials(headers)
        except ValueError as exc:
            return 400, {"code": "bad_provider_credentials", "message": str(exc)}
        if not api_key:
            return 400, {"code": "missing_lium_api_key",
                         "message": "send the new Lium API key as X-Lium-Api-Key"}
        status, ctx = self._auth("credentials", headers)
        if status is not None:
            return status, ctx
        gate = self._registration_gate(ctx["hotkey"])
        if gate is not None:
            return gate
        if self.resolve_reveal(ctx["hotkey"], ctx["ref"]) is None:
            return 403, {"code": "not_revealed",
                         "message": "no revealed commitment for this hotkey matches "
                                    "X-Commit-Ref — reveal first, then rotate the key"}
        self.vault.insert(ctx["hotkey"], api_key)
        log.info("credentials %s: provider credentials rotated in the payer vault", ctx["hotkey"])
        return 200, {"status": "stored"}

    def direct_credentials(self, headers) -> tuple[int, dict]:
        """Deposit (or rotate) the key a direct-MoE gang is rented with, on the
        marketplace the miner chose. Same identity gate as ``/v1/fund``; the
        key lands only in that marketplace's vault and nothing is queued —
        admission is a separate signed ``/v2/direct/fund`` naming the provider.
        """
        provider = (headers.get("X-Compute-Provider") or "").strip().lower()
        api_key = (headers.get("X-Compute-Api-Key") or "").strip()
        if provider not in self.compute_vaults:
            return 400, {"code": "unsupported_compute_provider",
                         "message": "X-Compute-Provider must be one of "
                                    + ", ".join(sorted(self.compute_vaults))}
        if not api_key:
            return 400, {"code": "missing_compute_api_key",
                         "message": "send the marketplace key as X-Compute-Api-Key"}
        status, ctx = self._auth("credentials", headers,
                                 key_digest=compute_key_digest(provider, api_key))
        if status is not None:
            return status, ctx
        gate = self._registration_gate(ctx["hotkey"])
        if gate is not None:
            return gate
        if self.resolve_reveal(ctx["hotkey"], ctx["ref"]) is None:
            return 403, {"code": "not_revealed",
                         "message": "no revealed commitment for this hotkey matches "
                                    "X-Commit-Ref — reveal first, then deposit a key"}
        self.compute_vaults[provider].insert(ctx["hotkey"], api_key)
        log.info("direct credentials %s: %s key stored", ctx["hotkey"], provider)
        return 200, {"status": "stored", "provider": provider}

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

    # ── lane C (DEC-CA-0056) ─────────────────────────────────────────────────

    def _lane_c_request(self, action: str, headers, gpu_segment: str = "",
                        ) -> tuple[int, dict] | tuple[None, dict]:
        """The lane-C identity gate: lane C on, a well-formed configured
        target, then the ``/v1/fund`` gate with the signature over
        :func:`lane_c_signed_ref` (target AND ref) and the key digest."""
        if self.lane_c_queue is None:
            return 503, {"code": "lane_c_disabled",
                         "message": "this intake does not run the lane-C (fine-tune) queue"}
        target = (headers.get(LANE_C_TARGET_HEADER) or "").strip()
        ref = (headers.get("X-Commit-Ref") or "").strip()
        if not target:
            return 400, {"code": "missing_target",
                         "message": f"send the declared lane-C target as {LANE_C_TARGET_HEADER}"}
        try:
            from ..shared.config.lane_c import check_target, is_mkt_target

            if is_mkt_target(target) and not self.mkt_suites:
                return 400, {"code": "marketcast_inactive",
                             "message": f"{target!r}: MarketCast (DEC-CA-0064) is not active "
                                        "on this subnet — mkt:<suite> targets are refused"}
            target = check_target(target, mkt_suites=self.mkt_suites)
        except ValueError as e:
            return 400, {"code": "bad_target", "message": str(e)}
        except ImportError:   # pragma: no cover — shared lane-C config not shipped
            pass
        if self.lane_c_targets is not None and target not in self.lane_c_targets:
            return 400, {"code": "unknown_target",
                         "message": f"{target!r} is not a lane-C target; choose one of "
                                    + ", ".join(self.lane_c_targets)}
        if action == "lane-c-fund" and self.lane_c_closed is not None:
            why = self.lane_c_closed(target)
            if why:
                return 409, {"code": "competition_ended", "message": why}
        try:
            signed = lane_c_signed_ref(target, ref) if ref else None
        except ValueError as e:
            return 400, {"code": "bad_ref", "message": str(e)}
        status, ctx = self._auth(action, headers, signed_ref=signed, gpu_segment=gpu_segment)
        if status is not None:
            return status, ctx
        ctx["target"] = target
        return None, ctx

    def lane_c_fund(self, headers) -> tuple[int, dict]:
        """Fund one lane-C fine-tune leg (``POST /v1/lane-c/fund``): the
        ``/v1/fund`` gate (signature binding key digest + target + ref, fresh
        strictly-increasing timestamp, registered hotkey) plus a REVEALED
        ``metro-v1:ft:<target>:hippius:<ref>`` commitment naming exactly this
        target and ref. Queues into the lane-C queue (never the challenger
        queue) and vaults the key in the shared payer vault — the same key a
        hotkey's other legs rent on (one key per hotkey, rotated by
        ``/v1/credentials``). Consumes NO lane-A lifetime submission."""
        try:
            self.lane_c_queue.expire_stale()      # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — housekeeping never fails a request
            log.exception("lane-C queue TTL sweep failed")
        choice, segment, gpu_err = self._gpu_choice(headers)
        if gpu_err is not None:
            return 400, gpu_err
        status, ctx = self._lane_c_request("lane-c-fund", headers, gpu_segment=segment)
        if status is not None:
            return status, ctx
        gate = self._registration_gate(ctx["hotkey"])
        if gate is not None:
            return gate
        from .providers import header_credentials

        try:
            api_key = header_credentials(headers)
        except ValueError as exc:
            return 400, {"code": "bad_provider_credentials", "message": str(exc)}
        if not api_key:
            return 400, {"code": "missing_lium_api_key",
                         "message": "a funded fine-tune leg needs your Lium API key "
                                    "as X-Lium-Api-Key"}
        label, label_err = submission_label(headers)
        if label_err is not None:
            return 400, label_err
        hotkey, ref, target = ctx["hotkey"], ctx["ref"], ctx["target"]
        resolve = self.resolve_lane_c_reveal
        reveal_block = resolve(hotkey, target, ref) if resolve is not None else None
        if reveal_block is None:
            return 403, {"code": "not_revealed",
                         "message": "no revealed lane-C commitment "
                                    "(metro-v1:ft:<target>:hippius:<ref>) for this hotkey "
                                    "matches the target and X-Commit-Ref — commit with "
                                    "`cascade lane-c commit`, wait for the reveal, then fund"}
        outcome = self.lane_c_queue.add_lane_c(  # type: ignore[union-attr]
            hotkey, target, ref, int(reveal_block), label=label, gpu_choice=choice)
        self.vault.insert(hotkey, api_key)
        log.info("lane-C fund %s: %s target=%s (reveal_block=%d%s)", hotkey, outcome, target,
                 reveal_block, f", gpu={choice.gpu}" if choice.chosen else "")
        if outcome == "already-queued":
            return 200, self._gpu_reply(
                {"status": "already-queued", "lane": "c", "target": target}, choice)
        return 202, self._gpu_reply({"status": outcome, "lane": "c", "target": target}, choice)

    def lane_c_withdraw(self, headers) -> tuple[int, dict]:
        """Withdraw a still-queued lane-C entry (``POST /v1/lane-c/withdraw``).
        The vaulted key is NOT forgotten here: it is shared with the hotkey's
        other legs (use ``/v1/withdraw`` to forget it)."""
        status, ctx = self._lane_c_request("lane-c-withdraw", headers)
        if status is not None:
            return status, ctx
        if self.lane_c_queue.withdraw(ctx["hotkey"]):  # type: ignore[union-attr]
            log.info("lane-C withdraw %s", ctx["hotkey"])
            return 200, {"status": "withdrawn", "lane": "c"}
        return 409, {"code": "not_queued",
                     "message": "no queued lane-C entry for this hotkey (an in-flight leg "
                                "runs to its end)"}

    # ── lane B (DEC-CA-0059 item 8) ──────────────────────────────────────────

    def _lane_b_request(self, action: str, headers, gpu_segment: str = "",
                        ) -> tuple[int, dict] | tuple[None, dict]:
        """The lane-B identity gate: lane B on, a well-formed declared target
        list (configured slots, at most the per-leg cap, no repeats), then the
        ``/v1/fund`` gate with the signature over :func:`lane_b_signed_ref`
        (targets as sent AND ref) and the key digest. ``ctx["targets"]`` is
        the list in the ``[lane_b]`` slot order (as sent when unconfigured)."""
        if self.lane_b_queue is None:
            return 503, {"code": "lane_b_disabled",
                         "message": "this intake does not run the lane-B (domain-expert) queue"}
        raw = (headers.get(LANE_B_TARGETS_HEADER) or "").strip()
        sent = lane_b_targets(raw)
        ref = (headers.get("X-Commit-Ref") or "").strip()
        if not sent:
            return 400, {"code": "missing_targets",
                         "message": f"send the declared lane-B targets as {LANE_B_TARGETS_HEADER} "
                                    "(comma-separated, e.g. energy or energy,horizon:720)"}
        if len(set(sent)) != len(sent) or any("|" in t for t in sent):
            return 400, {"code": "bad_targets",
                         "message": f"{LANE_B_TARGETS_HEADER} must list distinct targets"}
        if self.lane_b_targets is not None:
            unknown = [t for t in sent if t not in self.lane_b_targets]
            if unknown:
                return 400, {"code": "unknown_target",
                             "message": f"{unknown} not lane-B targets; choose from "
                                        + ", ".join(self.lane_b_targets)}
        if self.lane_b_max_targets is not None and len(sent) > self.lane_b_max_targets:
            return 400, {"code": "too_many_targets",
                         "message": f"{len(sent)} targets > [lane_b] max_domains_per_leg = "
                                    f"{self.lane_b_max_targets}"}
        try:
            signed = lane_b_signed_ref(sent, ref) if ref else None
        except ValueError as e:
            return 400, {"code": "bad_ref", "message": str(e)}
        status, ctx = self._auth(action, headers, signed_ref=signed, gpu_segment=gpu_segment)
        if status is not None:
            return status, ctx
        order = self.lane_b_targets
        ctx["targets"] = (tuple(t for t in order if t in sent) if order is not None else sent)
        return None, ctx

    def lane_b_fund(self, headers) -> tuple[int, dict]:
        """Fund one lane-B domain-expert leg (``POST /v1/lane-b/fund``): the
        ``/v1/fund`` gate (signature binding key digest + targets + ref, fresh
        strictly-increasing timestamp, registered hotkey) plus a REVEALED
        ``metro-v1:expert:`` commitment declaring exactly these targets and
        this ref. Queues into the lane-B queue (never the challenger queue)
        and vaults the key in the shared payer vault — the same key a
        hotkey's other legs rent on. Consumes NO lane-A lifetime submission."""
        try:
            self.lane_b_queue.expire_stale()      # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — housekeeping never fails a request
            log.exception("lane-B queue TTL sweep failed")
        choice, segment, gpu_err = self._gpu_choice(headers)
        if gpu_err is not None:
            return 400, gpu_err
        status, ctx = self._lane_b_request("lane-b-fund", headers, gpu_segment=segment)
        if status is not None:
            return status, ctx
        gate = self._registration_gate(ctx["hotkey"])
        if gate is not None:
            return gate
        from .providers import header_credentials

        try:
            api_key = header_credentials(headers)
        except ValueError as exc:
            return 400, {"code": "bad_provider_credentials", "message": str(exc)}
        if not api_key:
            return 400, {"code": "missing_lium_api_key",
                         "message": "a funded domain-expert leg needs your Lium API key "
                                    "as X-Lium-Api-Key"}
        label, label_err = submission_label(headers)
        if label_err is not None:
            return 400, label_err
        hotkey, ref, targets = ctx["hotkey"], ctx["ref"], ctx["targets"]
        resolve = self.resolve_lane_b_reveal
        reveal_block = resolve(hotkey, targets, ref) if resolve is not None else None
        if reveal_block is None:
            return 403, {"code": "not_revealed",
                         "message": "no revealed lane-B commitment (metro-v1:expert:…) for "
                                    "this hotkey declares these targets and X-Commit-Ref — "
                                    "commit with `cascade deploy --expert-domain`, wait for "
                                    "the reveal, then fund"}
        outcome = self.lane_b_queue.add_lane_b(  # type: ignore[union-attr]
            hotkey, targets, ref, int(reveal_block), label=label, gpu_choice=choice)
        self.vault.insert(hotkey, api_key)
        log.info("lane-B fund %s: %s targets=%s (reveal_block=%d%s)", hotkey, outcome,
                 ",".join(targets), reveal_block, f", gpu={choice.gpu}" if choice.chosen else "")
        body = {"status": outcome, "lane": "b", "targets": list(targets)}
        if outcome == "already-queued":
            return 200, self._gpu_reply(body, choice)
        return 202, self._gpu_reply(body, choice)

    def lane_b_withdraw(self, headers) -> tuple[int, dict]:
        """Withdraw a still-queued lane-B entry (``POST /v1/lane-b/withdraw``).
        The vaulted key is NOT forgotten here: it is shared with the hotkey's
        other legs (use ``/v1/withdraw`` to forget it)."""
        status, ctx = self._lane_b_request("lane-b-withdraw", headers)
        if status is not None:
            return status, ctx
        if self.lane_b_queue.withdraw(ctx["hotkey"]):  # type: ignore[union-attr]
            log.info("lane-B withdraw %s", ctx["hotkey"])
            return 200, {"status": "withdrawn", "lane": "b"}
        return 409, {"code": "not_queued",
                     "message": "no queued lane-B entry for this hotkey (an in-flight leg "
                                "runs to its end)"}

    def lane_b_queue_view(self) -> tuple[int, dict]:
        if self.lane_b_queue is None:
            return 503, {"code": "lane_b_disabled"}
        return 200, self.lane_b_queue.public_view()  # type: ignore[union-attr]

    def lane_c_queue_view(self) -> tuple[int, dict]:
        if self.lane_c_queue is None:
            return 503, {"code": "lane_c_disabled"}
        return 200, self.lane_c_queue.public_view()  # type: ignore[union-attr]

    def queue_view(self) -> tuple[int, dict]:
        self.sweep_pending()
        return 200, self.queue.public_view()

    # ── HTTP server ──────────────────────────────────────────────────────────

    def make_server(self, host: str, port: int, *,
                    max_connections: int = 64,
                    request_timeout_s: float = 30.0,
                    request_deadline_s: float = 120.0,
                    max_concurrent_uploads: int = 4,
                    per_ip_rpm: int = 60, per_ip_connections: int = 8,
                    trusted_proxy: str = "") -> ThreadingHTTPServer:
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
        * ``per_ip_rpm`` / ``per_ip_connections`` (:class:`PerIpLimiter`): a
          single source cannot take every slot or hammer the chain oracle.
          Behind the proxy, set ``trusted_proxy`` to its IP so the client is
          read from ``X-Forwarded-For`` (never trusted from anyone else).
        """
        intake = self
        upload_slots = threading.BoundedSemaphore(max_concurrent_uploads)
        limiter = PerIpLimiter(rpm=per_ip_rpm, max_connections=per_ip_connections)
        too_many = json.dumps({"code": "rate_limited",
                               "message": "too many requests from this address; "
                                          "retry shortly"}).encode()
        reject_ip = (b"HTTP/1.1 429 Too Many Requests\r\n"
                     b"Content-Type: application/json\r\n"
                     b"Content-Length: " + str(len(too_many)).encode() + b"\r\n"
                     b"Retry-After: 10\r\n"
                     b"Connection: close\r\n"
                     b"\r\n" + too_many)

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
                self.send_header("Cache-Control", "no-store")
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

            def _client_ip(self) -> str:
                peer = self.client_address[0]
                if trusted_proxy and peer == trusted_proxy:
                    xff = (self.headers.get("X-Forwarded-For") or "").split(",")
                    last = xff[-1].strip() if xff and xff[-1].strip() else ""
                    return last or peer
                return peer

            def _rate_limited(self) -> bool:
                if limiter.allow_request(self._client_ip()):
                    return False
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(too_many)))
                self.send_header("Retry-After", "10")
                self.end_headers()
                self.wfile.write(too_many)
                return True

            def do_GET(self) -> None:  # noqa: N802 — http.server API
                if self.path != "/health" and self._rate_limited():
                    return
                if self.path == "/health":
                    self._reply(200, {"status": "ok"})
                elif self.path == "/v1/queue":
                    self._dispatch(intake.queue_view)
                elif self.path == "/v1/lane-c/queue":
                    self._dispatch(intake.lane_c_queue_view)
                elif self.path == "/v1/lane-b/queue":
                    self._dispatch(intake.lane_b_queue_view)
                else:
                    self._reply(404, {"code": "not_found"})

            def do_POST(self) -> None:  # noqa: N802 — http.server API
                if self._rate_limited():
                    return
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
                if self.path == "/v2/direct/fund":
                    if intake.direct_intake is None:
                        self._reply(503, {"code": "direct_protocol_disabled"})
                        return
                    if not 0 < length <= 65536:
                        self._reply(413, {"code": "invalid_direct_body_size"})
                        return
                    try:
                        envelope = json.loads(self.rfile.read(length))
                    except (ValueError, UnicodeError):
                        self._reply(400, {"code": "invalid_json"})
                        return
                    self._dispatch(lambda: intake.direct_intake.fund(envelope, now=intake.clock()))
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
                elif self.path == "/v1/credentials":
                    self._dispatch(lambda: intake.credentials(self.headers))
                elif self.path == "/v2/direct/credentials":
                    self._dispatch(lambda: intake.direct_credentials(self.headers))
                elif self.path == "/v1/withdraw":
                    self._dispatch(lambda: intake.withdraw(self.headers))
                elif self.path == "/v1/lane-c/fund":
                    self._dispatch(lambda: intake.lane_c_fund(self.headers))
                elif self.path == "/v1/lane-c/withdraw":
                    self._dispatch(lambda: intake.lane_c_withdraw(self.headers))
                elif self.path == "/v1/lane-b/fund":
                    self._dispatch(lambda: intake.lane_b_fund(self.headers))
                elif self.path == "/v1/lane-b/withdraw":
                    self._dispatch(lambda: intake.lane_b_withdraw(self.headers))
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
                        stuck = [(key, sock) for key, (sock, t0, _ip) in
                                 self._live.items()
                                 if now - t0 > request_deadline_s]
                    for key, sock in stuck:
                        log.warning("closing connection past the %.0fs "
                                    "deadline", request_deadline_s)
                        with contextlib.suppress(OSError):
                            sock.shutdown(_socket.SHUT_RDWR)
                        with contextlib.suppress(OSError):
                            sock.close()
                        with self._live_lock:
                            self._live.pop(key, None)

            def server_close(self) -> None:
                self._reaper_stop.set()
                super().server_close()

            def process_request(self, request, client_address) -> None:
                ip = client_address[0]
                if not limiter.connection_open(ip):
                    try:
                        request.settimeout(request_timeout_s)
                        request.sendall(reject_ip)
                    except OSError:
                        pass
                    self.shutdown_request(request)
                    return
                if not self._slots.acquire(blocking=False):
                    limiter.connection_close(ip)
                    try:
                        request.settimeout(request_timeout_s)
                        request.sendall(reject)
                    except OSError:
                        pass  # client already gone — the close below is enough
                    self.shutdown_request(request)
                    return
                with self._live_lock:
                    self._live[id(request)] = (request, time.monotonic(), ip)
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    # Thread never started (spawn failure) → the slot would
                    # leak; on success process_request_thread owns the release.
                    self._release(request)
                    raise

            def _release(self, request) -> None:
                with self._live_lock:
                    live = self._live.pop(id(request), None)
                self._slots.release()
                if live is not None:
                    limiter.connection_close(live[2])

            def process_request_thread(self, request, client_address) -> None:
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self._release(request)

        return BoundedServer((host, port), Handler)
