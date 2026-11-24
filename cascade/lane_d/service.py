"""The lane D intake service ``cascade-laned-intake`` (LANE_D_PLAN §1.2 "Flow",
"Ownership and dedup"; §1.5 B10; §11.2 B13; W3).

A WSGI app behind Caddy on the box with three routes:

* ``POST /v1/lane-d/submit``: the signed manifest of a checkpoint. The intake
  stores the canonical manifest in the vault itself and answers with upload
  grants for ``<vault_prefix>/<hotkey>/<digest>/`` minted by the one
  :class:`GrantMinter` of the W3a-chosen upload path
  (``decisions/STORAGE-LANE-D.md``: :data:`UPLOAD_PATH`).
* ``POST /v1/lane-d/ready``: the signed licence attestation (B13). The minter
  finalises the upload (multipart: the intake completes it), the object list
  and sizes are checked against the manifest without downloading anything but
  ``manifest.json``, ``coldkey_at_ready`` is read from the chain at ready (A8)
  and the entry gets ``ready_block`` and its ``commit_payload``.
* ``GET /v1/lane-d/queue``: every entry's public row (never a grant).

Every refusal is a typed ``{"code", "message"}`` with a code from
:data:`REFUSAL_CODES`. Storage, chain and grant errors fail closed as a typed
503 that changes nothing: a transient error is never read as "absent". The
queue is one JSON file written atomically, persisted before any accepted
submit or ready answers, and a corrupt file refuses to start.

Grants (presigned URLs) appear only in the submit response: never in a log
record, a file under the state dir, or the queue view.

Torch-free and validator-free: stdlib, the lane D intake helpers and the
shared sr25519 verifier.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import OrderedDict, deque
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path

from ..shared import sr25519
from .entry_config import (
    ENTRY_CONFIG_CODES,
    ENTRY_CONFIG_FILE,
    MAX_ENTRY_CONFIG_BYTES,
    check_entry_config,
)
from .intake import ALLOWED_FILES, MANIFEST_OBJECT, IntakeError, canonical_manifest
from .storage import LaneDStorage

log = logging.getLogger("cascade.lane_d.service")

# ── the upload path (W3a decision record) ───────────────────────────────────

# The path ids of decisions/STORAGE-LANE-D.md, in the record's table order.
UPLOAD_PATH_IDS = ("presigned_multipart", "single_presigned_put", "upload_relay")
# The record's ``Chosen path:``; the intake builds and offers only this one.
UPLOAD_PATH = "presigned_multipart"
OFFERED_UPLOAD_PATHS = (UPLOAD_PATH,)

# ── the wire contract ────────────────────────────────────────────────────────

SUBMIT_PATH = "/v1/lane-d/submit"
READY_PATH = "/v1/lane-d/ready"
QUEUE_PATH = "/v1/lane-d/queue"
_ROUTES = {SUBMIT_PATH: ("POST", "submit"), READY_PATH: ("POST", "ready"),
           QUEUE_PATH: ("GET", "queue")}

# A signed timestamp must be within this many seconds of the intake's clock.
FRESHNESS_SECONDS = 300
# D13 (owner 2026-11-24): the one open licence a lane D checkpoint is published under
# (the exact SPDX id; a modified MIT is not MIT).
LICENCES = ("MIT",)
# B13 (plan §11.2): the one statement a miner attests at ready.
LICENCE_ATTESTATION = (
    "I attest that this checkpoint's weights, code and config, and the training data "
    "used to produce them, are licensed for publication under the MIT licence, "
    "and that the training data contains no outputs of any teacher model released under "
    "a restricted licence.")

# RAM budget (the box has 7 GB and the validator scores on its CPU).
MAX_BODY_BYTES = 16384
RATE_LIMIT_MAX_KEYS = 4096
RATE_WINDOW_SECONDS = 60.0
DEFAULT_PER_IP_RPM = 60
DEFAULT_PER_HOTKEY_RPM = 20

QUEUE_FILENAME = "lane_d_intake_queue.json"
QUEUE_SCHEMA = "lane-d-intake-queue/1"

# Entry states (plan §1.5). "Open" = ready and not yet judged.
SUBMITTED, READY, REVEALED, READY_EXPIRED, CLOSED = (
    "submitted", "ready", "revealed", "ready_expired", "closed")
ENTRY_STATES = (SUBMITTED, READY, REVEALED, READY_EXPIRED, CLOSED)
OPEN_STATES = (READY, REVEALED)

# The commitment a readied entry reveals: ``metro-v1:weights:c<ctx>:hippius:<ref>``
# where ``ref`` is the entry's vault prefix without its trailing ``/``.
COMMIT_PREFIX = "metro-v1:weights:"
_COMMIT_RE = re.compile(r"^metro-v1:weights:c(?P<ctx>[0-9]{1,7}):hippius:(?P<ref>[^\s:]+)$")

REFUSAL_CODES = (
    # transport
    "rate_limited", "not_found", "method_not_allowed", "body_too_large",
    "bad_content_length", "upload_path_not_offered",
    # ownership
    "missing_hotkey", "bad_hotkey", "bad_digest", "bad_context", "stale_timestamp",
    "bad_signature", "replayed_timestamp", "signature_unavailable", "not_registered",
    "not_owner", "digest_owned",
    # submit
    "bad_manifest", "digest_mismatch", "file_not_allowed", "context_not_armed",
    "hotkey_entry_open", "duplicate_weights", "file_too_large",
    # ready
    "unknown_entry", "bad_body", "licence_not_open", "licence_attestation_missing",
    "context_mismatch", "missing_object", "size_mismatch", "stray_object",
    "upload_incomplete", "coldkey_cap_open", "queue_full", "ready_expired", "entry_closed",
    # ready: the whole config.json pinned (cascade.lane_d.entry_config)
    "config_unreadable", "config_key_not_allowed", "config_value_not_allowed",
    # fail closed
    "storage_unavailable", "chain_unavailable", "grants_unavailable", "dedup_unavailable",
    "state_unavailable", "internal_error",
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_SS58 = re.compile(r"[1-9A-HJ-NP-Za-km-z]{40,60}")
_DECIMAL = re.compile(r"[0-9]{1,7}")
_UNIX_SECONDS = re.compile(r"-?[0-9]{1,12}")
_LOOPBACK = re.compile(r"^(127\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|::1|::ffff:127\.[0-9.]+)$")

# The public row of an entry (README "Queue"): ready-time fields are null before ready.
ROW_FIELDS = ("hotkey", "digest", "state", "context", "coldkey_at_ready", "ready_block",
              "ready_expires_block", "weights_sha256", "licence",
              "licence_attestation_sha256")
_READY_FIELDS = ("coldkey_at_ready", "ready_block", "ready_expires_block", "weights_sha256",
                 "licence", "licence_attestation_sha256", "commit_payload")


# ── the external services ────────────────────────────────────────────────────


class LaneDReads(abc.ABC):
    """The chain and registry reads the intake makes. Any failure raises; a
    failure is never an answer (``None`` / empty)."""

    @abc.abstractmethod
    def current_block(self) -> int:
        """The chain head."""

    @abc.abstractmethod
    def coldkey_of(self, hotkey: str) -> str | None:
        """The owning coldkey ss58, or ``None`` iff ``hotkey`` is not
        registered on the netuid."""

    @abc.abstractmethod
    def reveal_block(self, hotkey: str, digest: str) -> int | None:
        """The block of ``hotkey``'s lane D reveal of ``digest``, else ``None``."""

    @abc.abstractmethod
    def published_weights(self) -> frozenset[str]:
        """The weights sha256 set of every published checkpoint (lane A legs
        and kings, lane D kings) and the current house."""


class GrantMinter(abc.ABC):
    """Mints the upload grants of ONE upload path (:attr:`upload_path`)."""

    upload_path: str = ""

    @abc.abstractmethod
    def mint(self, prefix: str, files: dict[str, int]) -> dict:
        """JSON-serialisable grants for ``files`` (``{name: size}``) under
        ``prefix``."""

    @abc.abstractmethod
    def finalize(self, prefix: str, files: dict[str, int]) -> None:
        """Called once at a successful ``ready`` before the object-list check
        (multipart: completes the upload). May raise
        ``intake.IntakeError(code)``, answered as ``409 <code>``."""


# ── helpers ──────────────────────────────────────────────────────────────────


def canonical_message(action: str, hotkey: str, digest: str, context, timestamp,
                      body: bytes) -> bytes:
    """The bytes the hotkey signs for ``action`` (``submit`` or ``ready``)."""
    return (f"cascade-lane-d-{action}:v1:{hotkey}:{digest}:{context}:{timestamp}:"
            f"{hashlib.sha256(bytes(body)).hexdigest()}").encode()


def entry_prefix(vault_prefix: str, hotkey: str, digest: str) -> str:
    """The vault prefix of one entry (trailing ``/``)."""
    return f"{vault_prefix}/{hotkey}/{digest}/"


def commit_payload(context: int, prefix: str) -> str:
    """The commitment payload a readied entry reveals."""
    return f"{COMMIT_PREFIX}c{int(context)}:hippius:{prefix.rstrip('/')}"


def parse_commit_payload(payload: str) -> tuple[int, str] | None:
    """``(context, ref)`` of a lane D commitment payload, else ``None``."""
    m = _COMMIT_RE.match(payload or "")
    return (int(m.group("ctx")), m.group("ref")) if m else None


class _Refusal(Exception):
    """A typed answer: ``status`` and ``code`` (in :data:`REFUSAL_CODES`)."""

    def __init__(self, status: int, code: str, message: str = "", *, extra: dict | None = None,
                 headers: list[tuple[str, str]] | None = None):
        if code not in REFUSAL_CODES:
            raise ValueError(f"unknown refusal code {code!r}")
        super().__init__(f"{status} {code}: {message}")
        self.status = int(status)
        self.code = code
        self.message = message or code
        self.extra = dict(extra or {})
        self.headers = list(headers or [])


class _RateLimiter:
    """Sliding-window request counts per key, tracking at most ``max_keys``
    keys (the least recently seen key is dropped first)."""

    def __init__(self, max_keys: int, window: float = RATE_WINDOW_SECONDS):
        self.max_keys = int(max_keys)
        self.window = float(window)
        self._hits: OrderedDict[tuple[str, str], deque] = OrderedDict()

    def __len__(self) -> int:
        return len(self._hits)

    def _make_room(self, now: float) -> None:
        stale = [k for k, dq in self._hits.items() if not dq or dq[-1] <= now - self.window]
        for k in stale:
            del self._hits[k]
        while len(self._hits) >= self.max_keys:
            self._hits.popitem(last=False)

    def hit(self, kind: str, key: str, limit: int, now: float) -> int:
        """Count one request; 0 if allowed, else the Retry-After seconds."""
        k = (kind, key)
        dq = self._hits.get(k)
        if dq is None:
            if len(self._hits) >= self.max_keys:
                self._make_room(now)
            dq = self._hits[k] = deque()
        else:
            self._hits.move_to_end(k)
        while dq and dq[0] <= now - self.window:
            dq.popleft()
        if len(dq) >= int(limit):
            return max(1, int(math.ceil(dq[0] + self.window - now)))
        dq.append(now)
        return 0


def _header(environ: dict, name: str) -> str:
    return str(environ.get("HTTP_" + name.upper().replace("-", "_"), "")).strip()


def client_ip(environ: dict) -> str:
    """The last ``X-Forwarded-For`` hop when the peer is loopback (Caddy),
    else the peer address (the header is ignored)."""
    remote = str(environ.get("REMOTE_ADDR", "")).strip()
    if _LOOPBACK.match(remote):
        hops = [h.strip() for h in str(environ.get("HTTP_X_FORWARDED_FOR", "")).split(",")]
        hops = [h for h in hops if h]
        if hops:
            return hops[-1]
    return remote


def _read_body(environ: dict) -> bytes:
    """At most ``Content-Length`` bytes; refused unread when the length is
    missing-but-garbage, negative or over :data:`MAX_BODY_BYTES`."""
    raw = str(environ.get("CONTENT_LENGTH", "") or "").strip()
    if raw == "":
        return b""
    if not raw.isdigit():
        raise _Refusal(400, "bad_content_length", "Content-Length must be a decimal integer")
    n = int(raw)
    if n > MAX_BODY_BYTES:
        raise _Refusal(413, "body_too_large", f"body over {MAX_BODY_BYTES} bytes")
    stream = environ["wsgi.input"]
    chunks, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _parse_manifest(body: bytes) -> dict:
    """The submitted manifest ``{"files": {name: {"sha256", "size"}}}``."""
    try:
        doc = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _Refusal(400, "bad_manifest", "the manifest is not JSON") from None
    if not isinstance(doc, dict) or set(doc) != {"files"} or not isinstance(doc["files"], dict) \
            or not doc["files"]:
        raise _Refusal(400, "bad_manifest", 'the manifest must be {"files": {name: ...}}')
    for name, meta in doc["files"].items():
        if not isinstance(meta, dict) or set(meta) != {"sha256", "size"}:
            raise _Refusal(400, "bad_manifest", f"{name!r}: needs exactly sha256 and size")
        size, sha = meta["size"], meta["sha256"]
        if type(size) is not int or size <= 0:
            raise _Refusal(400, "bad_manifest", f"{name!r}: size must be a positive integer")
        if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
            raise _Refusal(400, "bad_manifest", f"{name!r}: sha256 must be 64 lowercase hex")
    return doc


def _status_line(status: int) -> str:
    try:
        return f"{status} {HTTPStatus(status).phrase}"
    except ValueError:
        return f"{status} Error"


# ── the app ──────────────────────────────────────────────────────────────────


class LaneDIntake:
    """The intake as a WSGI callable (see the module docstring)."""

    def __init__(self, *, storage: LaneDStorage, reads: LaneDReads, minter: GrantMinter, cfg,
                 ops, state_dir, now: Callable[[], float],
                 per_ip_rpm: int = DEFAULT_PER_IP_RPM,
                 per_hotkey_rpm: int = DEFAULT_PER_HOTKEY_RPM):
        if not isinstance(minter, GrantMinter):
            raise TypeError(f"minter must be a GrantMinter; got {type(minter).__name__}")
        if getattr(minter, "upload_path", None) != UPLOAD_PATH:
            raise ValueError(f"the minter serves upload path {minter.upload_path!r}; this "
                             f"intake offers only {UPLOAD_PATH!r} (decisions/STORAGE-LANE-D.md)")
        if not isinstance(storage, LaneDStorage):
            raise TypeError(f"storage must be a LaneDStorage; got {type(storage).__name__}")
        if not isinstance(reads, LaneDReads):
            raise TypeError(f"reads must be a LaneDReads; got {type(reads).__name__}")
        for name, v in (("per_ip_rpm", per_ip_rpm), ("per_hotkey_rpm", per_hotkey_rpm)):
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be an integer >= 1; got {v!r}")
        self.storage = storage
        self.reads = reads
        self.minter = minter
        self.cfg = cfg
        self.ops = ops
        self.state_dir = Path(state_dir)
        self.queue_path = self.state_dir / QUEUE_FILENAME
        self._now = now
        self.per_ip_rpm = per_ip_rpm
        self.per_hotkey_rpm = per_hotkey_rpm
        self._lock = threading.RLock()
        self._limiter = _RateLimiter(RATE_LIMIT_MAX_KEYS)
        self._last_ts: dict[tuple[str, str], int] = {}
        self._refreshed_block: int | None = None
        self._entries: dict[str, dict] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, dict]:
        """The persisted entries; an absent file is an empty queue, an
        unreadable one refuses to start (it never starts empty)."""
        if not self.queue_path.exists():
            return {}
        try:
            doc = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            raise RuntimeError(f"lane D intake queue {self.queue_path} is unreadable "
                               f"({type(e).__name__}); refusing to start with an empty "
                               f"queue") from None
        if not isinstance(doc, dict) or doc.get("schema") != QUEUE_SCHEMA \
                or not isinstance(doc.get("entries"), list):
            raise RuntimeError(f"lane D intake queue {self.queue_path} is not a "
                               f"{QUEUE_SCHEMA} file; refusing to start")
        out: dict[str, dict] = {}
        for rec in doc["entries"]:
            if (not isinstance(rec, dict) or not isinstance(rec.get("hotkey"), str)
                    or not isinstance(rec.get("digest"), str)
                    or rec.get("state") not in ENTRY_STATES
                    or not isinstance(rec.get("files"), dict)
                    or rec["hotkey"] in out):
                raise RuntimeError(f"lane D intake queue {self.queue_path} holds a malformed "
                                   f"entry; refusing to start")
            out[rec["hotkey"]] = rec
        return out

    def _write_queue(self) -> None:
        """Write the queue atomically (temp file, fsync, ``os.replace``)."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"schema": QUEUE_SCHEMA,
                           "entries": [self._entries[k] for k in sorted(self._entries)]},
                          sort_keys=True, indent=1)
        tmp = self.queue_path.with_name(f".{self.queue_path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "wb") as f:
                f.write(text.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.queue_path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                log.warning("lane_d_intake_unavailable could not remove the temp queue file %s",
                            tmp.name)
            raise
        fd = os.open(str(self.state_dir), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _commit(self, before: dict[str, dict]) -> None:
        """Persist the in-memory change, or roll it back and refuse."""
        try:
            self._write_queue()
        except Exception as e:  # noqa: BLE001 — any write failure fails closed
            self._entries = before
            log.error("lane_d_intake_unavailable state write failed (%s); change rolled back",
                      type(e).__name__)
            raise _Refusal(503, "state_unavailable", "the queue could not be persisted") from None

    def _snapshot(self) -> dict[str, dict]:
        return copy.deepcopy(self._entries)

    # ── views ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row(rec: dict) -> dict:
        """The public row: the ready-time fields stay null until ready."""
        return {k: rec.get(k) for k in ROW_FIELDS}

    def _ready_answer(self, rec: dict) -> dict:
        out = {"status": "ready", "hotkey": rec["hotkey"], "digest": rec["digest"],
               "context": rec["context"]}
        out.update({k: rec.get(k) for k in _READY_FIELDS})
        return out

    def rate_limit_keys(self) -> int:
        """How many keys the rate limiter tracks (bounded by
        :data:`RATE_LIMIT_MAX_KEYS`)."""
        with self._lock:
            return len(self._limiter)

    # ── chain reads (fail closed) ────────────────────────────────────────────

    def _chain(self, fn, *args):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 — never read as an answer
            log.warning("lane_d_intake_unavailable chain read %s failed (%s)",
                        getattr(fn, "__name__", "read"), type(e).__name__)
            raise _Refusal(503, "chain_unavailable", "the chain read failed; retry") from None

    def _published(self) -> frozenset[str]:
        try:
            return frozenset(self.reads.published_weights())
        except Exception as e:  # noqa: BLE001
            log.warning("lane_d_intake_unavailable published-weights read failed (%s)",
                        type(e).__name__)
            raise _Refusal(503, "dedup_unavailable", "the dedup source is unavailable") from None

    def _refresh(self) -> int:
        """Evaluate TTL (B10) at the chain head and return the head. A ready
        entry becomes ``revealed`` (reveal inside its window) or
        ``ready_expired`` (head past the window, no reveal inside it). A
        failed reveal read never expires an entry."""
        block = int(self._chain(self.reads.current_block))
        if block == self._refreshed_block:
            return block
        before = self._snapshot()
        changed, complete = [], True
        for hk in sorted(self._entries):
            rec = self._entries[hk]
            if rec["state"] != READY:
                continue
            try:
                rb = self.reads.reveal_block(rec["hotkey"], rec["digest"])
            except Exception as e:  # noqa: BLE001 — the entry stays open
                complete = False
                log.warning("lane_d_intake_unavailable reveal read failed for %s (%s); "
                            "the entry stays open", rec["digest"], type(e).__name__)
                continue
            if rb is not None and int(rb) <= int(rec["ready_expires_block"]):
                rec["state"], rec["reveal_block"] = REVEALED, int(rb)
                changed.append(rec)
            elif block > int(rec["ready_expires_block"]):
                rec["state"] = READY_EXPIRED
                changed.append(rec)
        if changed:
            self._commit(before)
            for rec in changed:
                if rec["state"] == READY_EXPIRED:
                    log.info("lane_d_intake_expired hotkey=%s digest=%s ready_block=%s "
                             "expires=%s head=%s", rec["hotkey"], rec["digest"],
                             rec["ready_block"], rec["ready_expires_block"], block)
        if complete:
            self._refreshed_block = block
        return block

    def tick(self) -> int | None:
        """One background refresh (#842): the same TTL / reveal evaluation a
        request runs, under the request lock, persisted with the same atomic
        write. Returns the head, or ``None`` when a chain read or the state
        write failed (logged; the next tick retries; nothing changes)."""
        with self._lock:
            try:
                return self._refresh()
            except _Refusal as r:
                log.warning("lane_d_intake_unavailable background refresh skipped (%s)", r.code)
                return None

    def refresh_timer(self, interval_s: float | None = None, *,
                      wait: Callable[[float], bool] | None = None) -> RefreshTimer:
        """A :class:`RefreshTimer` ticking every ``interval_s`` (default
        ``[lane_d_ops] queue_poll_seconds``)."""
        if interval_s is None:
            interval_s = getattr(self.ops, "queue_poll_seconds", None)
        return RefreshTimer(self, interval_s, wait=wait)

    # ── WSGI ─────────────────────────────────────────────────────────────────

    def __call__(self, environ, start_response):
        headers: list[tuple[str, str]] = []
        try:
            status, doc = self._handle(environ)
        except _Refusal as r:
            status, headers = r.status, r.headers
            doc = {"code": r.code, "message": r.message, **r.extra}
            if r.status >= 500:
                log.warning("lane_d_intake_unavailable %s %s", r.status, r.code)
            else:
                log.info("lane_d_intake_refused %s %s", r.status, r.code)
        except Exception as e:  # noqa: BLE001 — a typed 500, never a traceback to the client
            status, doc = 500, {"code": "internal_error", "message": "internal error"}
            log.error("lane_d_intake_unavailable unhandled %s", type(e).__name__)
        body = json.dumps(doc, sort_keys=True).encode("utf-8")
        start_response(_status_line(status),
                       [("Content-Type", "application/json"),
                        ("Content-Length", str(len(body))),
                        ("Cache-Control", "no-store"), *headers])
        return [body]

    def _handle(self, environ) -> tuple[int, dict]:
        with self._lock:
            now = float(self._now())
            retry = self._limiter.hit("ip", client_ip(environ), self.per_ip_rpm, now)
            if retry:
                raise _Refusal(429, "rate_limited", "too many requests from this address",
                               headers=[("Retry-After", str(retry))])
            path = str(environ.get("PATH_INFO", ""))
            method = str(environ.get("REQUEST_METHOD", "")).upper()
            route = _ROUTES.get(path)
            if route is None:
                raise _Refusal(404, "not_found", "unknown route")
            want, action = route
            if method != want:
                raise _Refusal(405, "method_not_allowed", f"{path} takes {want}",
                               headers=[("Allow", want)])
            if action == "queue":
                return 200, self._queue_view()
            body = _read_body(environ)
            return self._signed(action, environ, body, now)

    def _queue_view(self) -> dict:
        try:
            self._refresh()
        except _Refusal:
            log.warning("lane_d_intake_unavailable queue view served without a TTL refresh")
        return {"entries": [self._row(self._entries[k]) for k in sorted(self._entries)]}

    # ── ownership ────────────────────────────────────────────────────────────

    def _signed(self, action: str, environ: dict, body: bytes, now: float) -> tuple[int, dict]:
        declared = _header(environ, "X-Upload-Path")
        if declared and declared not in OFFERED_UPLOAD_PATHS:
            raise _Refusal(400, "upload_path_not_offered",
                           f"this intake offers only {list(OFFERED_UPLOAD_PATHS)}")
        hotkey = _header(environ, "X-Miner-Hotkey")
        if not hotkey:
            raise _Refusal(400, "missing_hotkey", "X-Miner-Hotkey is required")
        if not _SS58.fullmatch(hotkey):
            raise _Refusal(400, "bad_hotkey", "X-Miner-Hotkey is not an ss58 address")
        digest = _header(environ, "X-Content-Digest")
        if not _HEX64.fullmatch(digest):
            raise _Refusal(400, "bad_digest", "X-Content-Digest must be 64 lowercase hex")
        raw_ctx = _header(environ, "X-Context-Length")
        if not _DECIMAL.fullmatch(raw_ctx):
            raise _Refusal(400, "bad_context", "X-Context-Length must be a decimal integer")
        context = int(raw_ctx)
        raw_ts = _header(environ, "X-Timestamp")
        if not _UNIX_SECONDS.fullmatch(raw_ts) or abs(int(raw_ts) - now) > FRESHNESS_SECONDS:
            raise _Refusal(400, "stale_timestamp",
                           f"X-Timestamp must be unix seconds within {FRESHNESS_SECONDS} s")
        ts = int(raw_ts)
        message = canonical_message(action, hotkey, digest, raw_ctx, raw_ts, body)
        try:
            ok = sr25519.verify_hotkey(hotkey, message, _header(environ, "X-Signature"))
        except sr25519.SignatureVerifierUnavailable:
            raise _Refusal(503, "signature_unavailable",
                           "the signature verifier is unavailable; retry") from None
        if not ok:
            raise _Refusal(400, "bad_signature", "the signature does not verify for this hotkey")
        key = (action, hotkey)
        last = self._last_ts.get(key)
        if last is not None and ts <= last:
            raise _Refusal(409, "replayed_timestamp",
                           "X-Timestamp must be later than the last accepted one")
        self._last_ts[key] = ts
        self._prune_timestamps(now)
        retry = self._limiter.hit("hotkey", hotkey, self.per_hotkey_rpm, now)
        if retry:
            raise _Refusal(429, "rate_limited", "too many requests for this hotkey",
                           headers=[("Retry-After", str(retry))])
        if action == "submit":
            return self._submit(hotkey, digest, context, body)
        return self._ready(hotkey, digest, context, body)

    def _prune_timestamps(self, now: float) -> None:
        """Forget timestamps too old to pass the freshness check anyway."""
        if len(self._last_ts) <= RATE_LIMIT_MAX_KEYS:
            return
        floor = now - FRESHNESS_SECONDS
        self._last_ts = {k: v for k, v in self._last_ts.items() if v >= floor}

    def _owner_of(self, digest: str, besides: str) -> str | None:
        for hk, rec in self._entries.items():
            if hk != besides and rec["digest"] == digest:
                return hk
        return None

    def _open_weights_elsewhere(self, weights: str, hotkey: str) -> bool:
        return any(rec["state"] in OPEN_STATES and rec.get("weights") == weights
                   for hk, rec in self._entries.items() if hk != hotkey)

    # ── submit ───────────────────────────────────────────────────────────────

    def _submit(self, hotkey: str, digest: str, context: int, body: bytes) -> tuple[int, dict]:
        manifest = _parse_manifest(body)
        canonical = canonical_manifest(manifest)
        if hashlib.sha256(canonical).hexdigest() != digest:
            raise _Refusal(400, "digest_mismatch",
                           "the manifest does not hash to X-Content-Digest")
        names = set(manifest["files"])
        if names != set(ALLOWED_FILES):
            raise _Refusal(400, "file_not_allowed",
                           f"a lane D checkpoint is exactly {list(ALLOWED_FILES)}")
        if context not in tuple(self.cfg.contexts):
            raise _Refusal(400, "context_not_armed",
                           f"context {context} is not in {list(self.cfg.contexts)}")
        if self._chain(self.reads.coldkey_of, hotkey) is None:
            raise _Refusal(403, "not_registered", "the hotkey is not registered on the netuid")
        self._refresh()
        if self._owner_of(digest, hotkey) is not None:
            raise _Refusal(409, "digest_owned", "another hotkey already submitted this digest")
        own = self._entries.get(hotkey)
        if own is not None and own["state"] in OPEN_STATES:
            raise _Refusal(409, "hotkey_entry_open",
                           "this hotkey already has an open entry")
        weights = manifest["files"]["weights.safetensors"]["sha256"]
        if weights in self._published() or self._open_weights_elsewhere(weights, hotkey):
            raise _Refusal(409, "duplicate_weights",
                           "these weights are a published checkpoint or an open entry")
        prefix = entry_prefix(self.ops.vault_prefix, hotkey, digest)
        try:
            self.storage.put(prefix + MANIFEST_OBJECT, canonical)
        except Exception as e:  # noqa: BLE001
            log.warning("lane_d_intake_unavailable vault put failed (%s)", type(e).__name__)
            raise _Refusal(503, "storage_unavailable", "the vault is unavailable; retry") from None
        sizes = {n: int(manifest["files"][n]["size"]) for n in sorted(names)}
        grants = self._mint(prefix, sizes)
        before = self._snapshot()
        self._entries[hotkey] = {
            "hotkey": hotkey, "digest": digest, "context": int(context), "state": SUBMITTED,
            "files": sizes, "weights": weights, "reveal_block": None, "closed_reason": None,
            **{k: None for k in _READY_FIELDS},
        }
        self._commit(before)
        log.info("lane_d_intake_submit hotkey=%s digest=%s context=%d", hotkey, digest, context)
        return 201, {"status": "granted", "upload_path": UPLOAD_PATH, "hotkey": hotkey,
                     "digest": digest, "context": int(context), "prefix": prefix,
                     "grants": grants}

    def _mint(self, prefix: str, sizes: dict[str, int]) -> dict:
        try:
            grants = self.minter.mint(prefix, dict(sizes))
            json.dumps(grants)
        except IntakeError as e:
            code = e.code if e.code in REFUSAL_CODES else "grants_unavailable"
            raise _Refusal(400 if code != "grants_unavailable" else 503, code,
                           "the upload cannot be granted") from None
        except Exception as e:  # noqa: BLE001
            log.warning("lane_d_intake_unavailable grant mint failed (%s)", type(e).__name__)
            raise _Refusal(503, "grants_unavailable", "upload grants are unavailable; retry") \
                from None
        return grants

    # ── ready ────────────────────────────────────────────────────────────────

    def _ready(self, hotkey: str, digest: str, context: int, body: bytes) -> tuple[int, dict]:
        try:
            doc = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise _Refusal(400, "bad_body", "the ready body is not JSON") from None
        if not isinstance(doc, dict):
            raise _Refusal(400, "bad_body", 'the ready body is {"licence", "attestation"}')
        licence = doc.get("licence")
        if not isinstance(licence, str) or licence not in LICENCES:
            raise _Refusal(400, "licence_not_open", f"the licence must be one of {list(LICENCES)}")
        if doc.get("attestation") != LICENCE_ATTESTATION:
            raise _Refusal(400, "licence_attestation_missing",
                           "the attestation must be the exact licence statement")
        block = self._refresh()
        rec = self._entries.get(hotkey)
        if rec is None or rec["digest"] != digest:
            if self._owner_of(digest, hotkey) is not None:
                raise _Refusal(403, "not_owner", "the uploader must be the committer")
            raise _Refusal(404, "unknown_entry", "no submitted entry for this hotkey and digest")
        if rec["state"] in OPEN_STATES:
            return 200, self._ready_answer(rec)
        if rec["state"] == READY_EXPIRED:
            raise _Refusal(409, "ready_expired", "the entry was not revealed inside its TTL")
        if rec["state"] == CLOSED:
            raise _Refusal(409, "entry_closed", "the entry is closed")
        if int(context) != int(rec["context"]):
            raise _Refusal(400, "context_mismatch", "the context differs from the submit's")
        if rec["weights"] in self._published() or self._open_weights_elsewhere(rec["weights"],
                                                                               hotkey):
            raise _Refusal(409, "duplicate_weights",
                           "these weights are a published checkpoint or an open entry")
        prefix = entry_prefix(self.ops.vault_prefix, hotkey, digest)
        files = {n: int(s) for n, s in rec["files"].items()}
        self._finalize(prefix, files)
        weights = self._check_objects(prefix, digest, files)
        coldkey = self._chain(self.reads.coldkey_of, hotkey)
        if coldkey is None:
            raise _Refusal(403, "not_registered", "the hotkey is not registered on the netuid")
        n_open = sum(1 for r in self._entries.values()
                     if r["state"] in OPEN_STATES and r.get("coldkey_at_ready") == coldkey)
        cap = int(self.cfg.max_open_per_coldkey)
        if n_open >= cap:
            raise _Refusal(409, "coldkey_cap_open",
                           "this coldkey already has its cap of open entries (advisory: the "
                           "validator's queue selection is the authority)",
                           extra={"advisory": True, "open": n_open, "cap": cap})
        if sum(1 for r in self._entries.values() if r["state"] in OPEN_STATES) \
                >= int(self.ops.queue_max_pending):
            raise _Refusal(503, "queue_full", "the lane D queue is full; retry later")
        before = self._snapshot()
        rec = self._entries[hotkey]
        rec.update({
            "state": READY, "coldkey_at_ready": str(coldkey), "ready_block": int(block),
            "ready_expires_block": int(block) + int(self.cfg.ready_ttl_blocks),
            "weights_sha256": weights, "licence": licence,
            "licence_attestation_sha256":
                hashlib.sha256(LICENCE_ATTESTATION.encode("utf-8")).hexdigest(),
            "commit_payload": commit_payload(rec["context"], prefix),
        })
        self._commit(before)
        log.info("lane_d_intake_ready hotkey=%s digest=%s ready_block=%d expires=%d",
                 hotkey, digest, rec["ready_block"], rec["ready_expires_block"])
        return 200, self._ready_answer(rec)

    def _finalize(self, prefix: str, files: dict[str, int]) -> None:
        try:
            self.minter.finalize(prefix, dict(files))
        except IntakeError as e:
            code = e.code if e.code in REFUSAL_CODES else "upload_incomplete"
            raise _Refusal(409, code, "the upload is not complete") from None
        except Exception as e:  # noqa: BLE001
            log.warning("lane_d_intake_unavailable upload finalize failed (%s)",
                        type(e).__name__)
            raise _Refusal(503, "grants_unavailable", "the upload could not be finalised; "
                           "retry") from None

    def _check_objects(self, prefix: str, digest: str, files: dict[str, int]) -> str:
        """The object list and sizes against the stored manifest, then the
        whole ``config.json`` against the pinned allowlist
        (:func:`cascade.lane_d.entry_config.check_entry_config`); nothing but
        ``manifest.json`` and that small JSON file is read, never the weights.
        Returns the weights sha256."""
        try:
            try:
                raw = self.storage.get(prefix + MANIFEST_OBJECT)
            except KeyError:
                raise _Refusal(409, "missing_object", MANIFEST_OBJECT) from None
            try:
                manifest = json.loads(bytes(raw).decode("utf-8"))
                canonical = canonical_manifest(manifest)
            except (UnicodeDecodeError, ValueError, KeyError, TypeError):
                raise _Refusal(409, "digest_mismatch", "the stored manifest is unreadable") \
                    from None
            if hashlib.sha256(canonical).hexdigest() != digest:
                raise _Refusal(409, "digest_mismatch",
                               "the stored manifest does not hash to the entry digest")
            listed = set(self.storage.list(prefix))
            expected = {prefix + n for n in files} | {prefix + MANIFEST_OBJECT}
            stray = sorted(listed - expected)
            if stray:
                raise _Refusal(409, "stray_object", f"{len(stray)} object(s) outside the manifest")
            for name in sorted(files):
                try:
                    size = int(self.storage.size(prefix + name))
                except KeyError:
                    raise _Refusal(409, "missing_object", name) from None
                if size != int(files[name]):
                    raise _Refusal(409, "size_mismatch", f"{name}: {size} != {files[name]}")
            if int(files.get(ENTRY_CONFIG_FILE, 0)) > MAX_ENTRY_CONFIG_BYTES:
                raise _Refusal(409, "config_unreadable",
                               f"{ENTRY_CONFIG_FILE} is larger than {MAX_ENTRY_CONFIG_BYTES} bytes")
            try:
                raw_config = self.storage.get(prefix + ENTRY_CONFIG_FILE)
            except KeyError:
                raise _Refusal(409, "missing_object", ENTRY_CONFIG_FILE) from None
            if hashlib.sha256(bytes(raw_config)).hexdigest() \
                    != str(manifest["files"][ENTRY_CONFIG_FILE]["sha256"]):
                raise _Refusal(409, "digest_mismatch",
                               f"{ENTRY_CONFIG_FILE} does not hash to its manifest entry")
            try:
                check_entry_config(bytes(raw_config))
            except IntakeError as e:
                code = e.code if e.code in ENTRY_CONFIG_CODES else "config_unreadable"
                raise _Refusal(409, code, e.detail[:300]) from None
        except _Refusal:
            raise
        except Exception as e:  # noqa: BLE001 — a transient error is never "absent"
            log.warning("lane_d_intake_unavailable vault read failed (%s)", type(e).__name__)
            raise _Refusal(503, "storage_unavailable", "the vault is unavailable; retry") from None
        return str(manifest["files"]["weights.safetensors"]["sha256"])

    # ── close (W5h/W17 wire it to the judged status) ─────────────────────────

    def close(self, hotkey: str, digest: str, reason: str) -> dict:
        """Move an entry to ``closed`` and free its slots; returns its row."""
        with self._lock:
            rec = self._entries.get(hotkey)
            if rec is None or rec["digest"] != digest:
                raise IntakeError("unknown_entry", f"{hotkey} {digest}")
            before = self._snapshot()
            rec = self._entries[hotkey]
            rec["state"], rec["closed_reason"] = CLOSED, str(reason)
            try:
                self._commit(before)
            except _Refusal:
                raise OSError(f"lane D intake queue {self.queue_path} could not be "
                              f"written") from None
            log.info("lane_d_intake_closed hotkey=%s digest=%s reason=%s", hotkey, digest, reason)
            return self._row(rec)


class RefreshTimer:
    """Refresh the intake queue without traffic (#842).

    The validator reads ``lane_d_intake_queue.json`` directly, so ``ready →
    revealed`` and ``ready → ready_expired`` must land in the file in quiet
    periods too. :meth:`run` ticks once at once (the startup refresh), then
    every ``interval_s`` until :meth:`stop`. ``wait(timeout) -> bool`` is the
    sleep (default: the stop event's ``wait``; ``True`` ends the loop), so a
    test drives ticks without real sleeps. A tick never raises: an unexpected
    error is logged and the loop keeps going."""

    def __init__(self, app: LaneDIntake, interval_s, *,
                 wait: Callable[[float], bool] | None = None):
        if isinstance(interval_s, bool) or not isinstance(interval_s, int | float) \
                or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError(f"refresh interval must be a positive number of seconds; "
                             f"got {interval_s!r}")
        self.app = app
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._wait = wait or self._stop.wait
        self._thread: threading.Thread | None = None
        self.ticks = 0

    def step(self) -> int | None:
        """One tick; returns the head the refresh saw (``None`` = skipped)."""
        self.ticks += 1
        try:
            return self.app.tick()
        except Exception as e:  # noqa: BLE001 — the timer thread must never die
            log.error("lane_d_intake_unavailable background refresh failed (%s)",
                      type(e).__name__)
            return None

    def run(self) -> None:
        while not self._stop.is_set():
            self.step()
            if self._wait(self.interval_s) or self._stop.is_set():
                return

    def start(self) -> RefreshTimer:
        if self._thread is not None:
            raise RuntimeError("refresh timer already started")
        self._thread = threading.Thread(target=self.run, name="lane-d-intake-refresh",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)


__all__ = [
    "DEFAULT_PER_HOTKEY_RPM", "DEFAULT_PER_IP_RPM", "ENTRY_STATES", "FRESHNESS_SECONDS",
    "GrantMinter", "LICENCES", "LICENCE_ATTESTATION", "LaneDIntake", "LaneDReads", "RefreshTimer",
    "MAX_BODY_BYTES", "OFFERED_UPLOAD_PATHS", "OPEN_STATES", "QUEUE_FILENAME", "QUEUE_PATH",
    "RATE_LIMIT_MAX_KEYS", "READY_PATH", "REFUSAL_CODES", "SUBMIT_PATH", "UPLOAD_PATH",
    "UPLOAD_PATH_IDS", "canonical_message", "client_ip", "commit_payload", "entry_prefix",
    "parse_commit_payload",
]
