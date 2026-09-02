"""Rent-failure taxonomy for miner-funded pods: whose fault, and does it burn?

A near-literal port of PRISM's ``lium-rent-pool`` crate (``base`` repo), which
classifies Lium rent failures from their error text. The classes and their
consequences:

* ``"auth"`` — the miner's key is invalid/unauthorized/revoked. Miner-fixable:
  the entry releases (they re-fund with a working key); it never burns, and it
  is never retried on the operator's key.
* ``"rate_limited"`` — a 429 on the key. Requeue without burning an attempt,
  but only within :data:`RECOVERY_WINDOW_SECONDS` — an entry that has been
  429ing for six hours is not going to clear by waiting.
* ``"no_capacity"`` — the market is sold out of the pinned SKU. Requeue
  without burning, with NO time bound (PRISM: "sold out is not Score(0)");
  the entry waits for capacity however long that takes.
* ``"infra"`` — everything else (launch failure, dud pod, transport). Bounded
  auto-retry on the miner's kept key, per the queue's attempt counter.

Order matters exactly as in the source: auth markers short-circuit the
no-capacity check, because Lium's permission errors mention resources too and
an unauthorized key must never be read as "sold out" (it would wait forever).
"""

from __future__ import annotations

import re

__all__ = [
    "RECOVERY_WINDOW_SECONDS",
    "classify_rent_failure",
    "is_auth_or_permission",
    "is_no_capacity",
    "is_rate_limited",
    "parse_retry_secs",
    "should_recover",
]

RECOVERY_WINDOW_SECONDS = 6 * 3600.0

# Retry-After style hints are advisory and occasionally absurd; cap them so a
# hostile/buggy header cannot park an entry for a day.
_RETRY_SECS_CAP = 7200

_AUTH_MARKERS = (
    "permission", "unauthorized", "forbidden", "invalid api key",
    "missing_lium_api_key", "api key missing",
)
_RATE_MARKERS = ("too many requests", "rate limit")
_CAPACITY_MARKERS = (
    "no_capacity", "no lium offer", "no offer matches", "no matching offer",
    "sold out", "out of capacity", "only 0 ",
)

# The numeric codes match only as standalone tokens: error text routinely
# embeds pod names, and pod names embed a slug of the payer's HOTKEY — a
# vanity hotkey containing "401"/"429" would otherwise steer every failure on
# its pods into the auth/rate classes, dodging the bounded infra-attempt
# budget forever (review 2026-09-02). No boundary between alphanumerics, so
# "ab401cd" never matches while "HTTP 401" and "(401)" do.
_AUTH_CODE_RE = re.compile(r"(?<![0-9a-z])401(?![0-9a-z])")
_RATE_CODE_RE = re.compile(r"(?<![0-9a-z])429(?![0-9a-z])")


def is_rate_limited(msg: str) -> bool:
    low = (msg or "").lower()
    return any(m in low for m in _RATE_MARKERS) or bool(_RATE_CODE_RE.search(low))


def is_auth_or_permission(msg: str) -> bool:
    low = (msg or "").lower()
    return any(m in low for m in _AUTH_MARKERS) or bool(_AUTH_CODE_RE.search(low))


def is_no_capacity(msg: str) -> bool:
    """Sold-out signal — but never for a message that is really an auth error."""
    if is_auth_or_permission(msg):
        return False
    low = (msg or "").lower()
    return any(m in low for m in _CAPACITY_MARKERS)


def classify_rent_failure(msg: str) -> str:
    """``"auth" | "rate_limited" | "no_capacity" | "infra"`` for a rent error."""
    if is_auth_or_permission(msg):
        return "auth"
    if is_rate_limited(msg):
        return "rate_limited"
    if is_no_capacity(msg):
        return "no_capacity"
    return "infra"


def parse_retry_secs(text: str) -> int | None:
    """Seconds to wait, parsed from rate-limit prose; ``None`` when it has none.

    Handles the shapes Lium actually emits: ``"try again in 37 seconds"``,
    a bare number, ``"per 1 hour"`` (→ a courteous 120s — the window rolls),
    ``"per 5 seconds"``. Capped at :data:`_RETRY_SECS_CAP`.
    """
    low = (text or "").lower()
    m = re.search(r"try again in\s+(\d+)", low)
    if m:
        return min(int(m.group(1)), _RETRY_SECS_CAP)
    m = re.search(r"per\s+(\d+)\s+hour", low)
    if m:
        return 120
    m = re.search(r"per\s+(\d+)\s+second", low)
    if m:
        return min(int(m.group(1)), _RETRY_SECS_CAP)
    m = re.search(r"\b(\d+)\b", low)
    if m and ("retry" in low or "wait" in low or "second" in low):
        return min(int(m.group(1)), _RETRY_SECS_CAP)
    return None


def should_recover(error_detail: str, failed_at: float, now: float) -> bool:
    """Requeue-without-burn decision for a failed funded entry.

    Sold-out recovers unconditionally (no time bound). Rate-limited recovers
    only within :data:`RECOVERY_WINDOW_SECONDS` of the failure. Auth never
    recovers automatically — the miner holds the fix. Storage-layer throttles
    that merely *mention* rate limiting (dataset CDN, hub mirrors) are
    excluded first: re-renting a pod does not help a throttled download.
    """
    low = (error_detail or "").lower()
    if any(m in low for m in ("huggingface", "hippius", "hf_hub", '"stage": "dataset"')):
        return False
    if is_no_capacity(low):
        return True
    if is_rate_limited(low):
        return (now - failed_at) <= RECOVERY_WINDOW_SECONDS
    return False
