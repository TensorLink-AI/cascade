"""Promotion record — the trainer's signed warm-start promotion declaration.

Cascade promotion is **propose-and-verify** (DEC-CA-0013): the trainer *selects*
which reign checkpoints become the next warm-start init set (it authored the
bench numbers selection runs on, so fleet re-derivation only ever re-checked the
trainer's arithmetic on the trainer's own data), and every validator *verifies*
the declaration against a small deterministic envelope — provenance, quality
floor, reign-clock ripeness, and set size — instead of re-deriving the choice.
The selection policy itself (top-k, diversity, rotation/allocation across
members) is trainer-side policy, deliberately NOT consensus-critical.

This module is the declaration's wire format: a :class:`PromotionRecord` names
the promoted *generation* (1-based; generation 0 is the random-init era), the
member checkpoints, and the round/block the promotion fired at. Each round then
trains from ONE member (the manifest's ``warm_start_ckpt``); validators accept
any member of the verified live set, which is what leaves the trainer free to
rotate — or adaptively allocate — rounds across members.

Conventions follow :mod:`cascade.shared.bench_report` exactly: frozen
dataclasses, an explicit ``record_version``, a canonical sorted-key JSON body,
and sign/verify over :meth:`PromotionRecord.canonical_body` with the trainer's
bittensor hotkey (``[manifest] trainer_hotkey`` — the trust anchor validators
already hold). Records publish to the manifest bucket
(``promotions/gen-<n>.json``) plus a tiny unsigned index
(``promotions/index.json``) so a validator joining mid-history can find the
latest generation without walking keys. The index is a locator only — trust
always comes from the record's signature, never from the index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PROMOTION_RECORD_VERSION = 1


def promotion_record_key(generation: int) -> str:
    return f"promotions/gen-{int(generation)}.json"


def promotion_index_key() -> str:
    return "promotions/index.json"


@dataclass(frozen=True)
class PromotedMember:
    """One checkpoint of the promoted warm-start set.

    ``checkpoint_id`` is the exact ``metro-v1:trained:…`` trained-pointer (the
    join key everywhere — never a UID, which recycles). ``source_round`` is the
    round whose signed bench report scored it, which is what lets a validator
    verify the member's provenance and quality on demand even when its own
    reign log missed that round's report. ``score`` is the trainer's
    :func:`cascade.validator.cascade.cascade_score` at selection time
    (observability; verification re-reads the signed report, never this copy).
    """

    checkpoint_id: str
    size: str
    source_round: str
    score: float


def member_to_json(m: PromotedMember) -> dict:
    """The ONE member JSON shape — shared by the record's canonical body, the
    trainer engine's persisted state, and the warm-start pointer file, so a
    field added to :class:`PromotedMember` changes exactly one encoder."""
    return {"checkpoint_id": m.checkpoint_id, "size": m.size,
            "source_round": m.source_round, "score": m.score}


def member_from_json(m: dict) -> PromotedMember:
    return PromotedMember(
        checkpoint_id=str(m["checkpoint_id"]), size=str(m.get("size", "")),
        source_round=str(m.get("source_round", "")),
        score=float(m.get("score", float("nan"))),
    )


@dataclass(frozen=True)
class PromotionRecord:
    """A fired promotion: the declared warm-start member set for a generation.

    ``generation`` is 1-based and strictly monotonic — validators only ever
    accept ``generation == accepted + 1`` (or a catch-up jump on bootstrap), so
    a replayed old record can never roll the live set back. ``fired_round`` /
    ``fired_block`` record when the trainer's reign clock fired (the block is
    the epoch-start block ripeness was judged at). ``king_hotkey`` is the king
    whose reign produced the candidates, for observability.
    """

    generation: int
    king_hotkey: str
    fired_round: str
    fired_block: int
    members: tuple[PromotedMember, ...] = ()
    record_version: int = PROMOTION_RECORD_VERSION
    signature: str | None = None  # trainer_hotkey signature over canonical_body()

    def member_ids(self) -> tuple[str, ...]:
        return tuple(m.checkpoint_id for m in self.members)

    def canonical_body(self) -> bytes:
        """Deterministic byte serialisation of everything except the signature —
        the signed payload, mirroring
        :meth:`cascade.shared.bench_report.BenchReport.canonical_body`."""
        body = {
            "record_version": self.record_version,
            "generation": self.generation,
            "king_hotkey": self.king_hotkey,
            "fired_round": self.fired_round,
            "fired_block": self.fired_block,
            "members": [member_to_json(m) for m in self.members],
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dump_promotion_record(record: PromotionRecord) -> str:
    """Serialise a record (including signature) to a JSON string."""
    body = json.loads(record.canonical_body().decode("utf-8"))
    body["signature"] = record.signature
    return json.dumps(body, indent=2, sort_keys=True)


def load_promotion_record(text: str) -> PromotionRecord:
    """Parse a promotion-record JSON string. Raises ``ValueError`` on schema
    problems."""
    obj = json.loads(text)
    version = int(obj.get("record_version", 0))
    if version != PROMOTION_RECORD_VERSION:
        raise ValueError(f"unsupported record_version {version}; need {PROMOTION_RECORD_VERSION}")
    members = tuple(member_from_json(m) for m in (obj.get("members") or ()))
    return PromotionRecord(
        generation=int(obj["generation"]),
        king_hotkey=str(obj.get("king_hotkey", "")),
        fired_round=str(obj.get("fired_round", "")),
        fired_block=int(obj.get("fired_block", 0)),
        members=members,
        record_version=version,
        signature=obj.get("signature"),
    )


def sign_promotion_record(record: PromotionRecord, wallet: object) -> PromotionRecord:
    """Sign ``canonical_body()`` with the trainer's bittensor hotkey — the exact
    scheme (and wallet duck-type) of :func:`cascade.shared.manifest.sign_manifest`."""
    from dataclasses import replace

    hotkey = getattr(wallet, "hotkey", wallet)
    try:
        sig = hotkey.sign(record.canonical_body())
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"promotion_record_signing_failed: {type(e).__name__}: {e}") from e
    return replace(record, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_promotion_record_signature(record: PromotionRecord, trainer_hotkey: str) -> bool:
    """Verify the record was signed by ``trainer_hotkey`` (an ss58 address).

    Mirrors :func:`cascade.shared.bench_report.verify_bench_report_signature`:
    False on a missing signature or any mismatch; raises only when ``bittensor``
    is unavailable so a caller never silently accepts an unverified record."""
    if not record.signature or not trainer_hotkey:
        return False
    try:
        from bittensor import Keypair  # type: ignore
    except ImportError as e:  # pragma: no cover - validator has bittensor
        raise RuntimeError(
            "bittensor required to verify promotion-record signatures; install the [chain] extra"
        ) from e
    try:
        kp = Keypair(ss58_address=trainer_hotkey)
        return bool(kp.verify(record.canonical_body(), bytes.fromhex(record.signature)))
    except Exception:  # noqa: BLE001 — any malformed sig/address ⇒ untrusted
        return False


def publish_promotion_record(store: object, record_text: str, generation: int) -> str:
    """Write a generation's promotion record to the manifest-bucket store (the
    R2 dual-write / HF failover the manifest enjoys covers it too) and refresh
    the unsigned locator index. Returns the record's key.

    Both objects publish public-read — the dashboard renders the rotation
    roster from the record (inferring the cycle from receipts breaks at every
    generation boundary), and trust comes from the record's signature, not the
    ACL. Same fallback as the receipt publisher: a backend without canned ACLs
    publishes private rather than not at all."""
    from .hippius import StorageError

    key = promotion_record_key(generation)
    index_text = json.dumps({"latest_generation": int(generation)}, sort_keys=True)
    for k, text in ((key, record_text), (promotion_index_key(), index_text)):
        try:
            store.put_text(k, text, content_type="application/json", acl="public-read")
        except StorageError:
            store.put_text(k, text, content_type="application/json")
    return key


def load_promotion_index(text: str) -> int:
    """The latest published generation out of ``promotions/index.json`` (0 on
    ANY malformed content — the locator is best-effort by design; a non-dict
    JSON body, a partial write, an error page all read as "nothing located")."""
    try:
        obj = json.loads(text)
        return int(obj.get("latest_generation", 0)) if isinstance(obj, dict) else 0
    except Exception:  # noqa: BLE001 — best-effort locator, never raises
        return 0


# ── all-time leaderboard (DEC-CA-0044) — unsigned, presentational ────────────

LEADERBOARD_KEY = "promotions/leaderboard.json"
LEADERBOARD_SCHEMA = 1


def build_leaderboard_doc(
    *,
    as_of: str,
    rule_active: bool,
    k: int,
    weights: tuple[float, float, float],
    notice_blocks: int,
    entries: list[dict],
    live_generation: int,
    live_members: list[str],
    upcoming: dict | None,
    activation_block: int = 0,
) -> dict:
    """Assemble the public all-time leaderboard document (pure).

    The trainer publishes it at every round boundary so dashboards and the
    miner CLI can show the fixed population the warm-start init is drawn
    from, and — the part miners act on — the ANNOUNCED change: ``upcoming``
    carries the next generation's member set, the round/block it was
    announced at and the block it takes effect (``announced_block +
    notice_blocks``), so a miner has the notice period to prepare against
    the exact checkpoint (``cascade score --warm-start upcoming``).

    ``entries`` are rank-ordered ``{rank, checkpoint_id, size, source_round,
    hotkey, role, score, gifteval_crps, …, time_mase}`` rows (the trainer's
    :class:`~cascade.trainer.promotion.LeaderEntry` shape); ``live_members``
    is the generation the field trains from NOW. Unsigned and
    presentational like the status docs — the signed promotion record stays
    the fleet's ground truth; a consumer must survive it stale or absent.
    """
    doc: dict = {
        "schema": LEADERBOARD_SCHEMA,
        "as_of": str(as_of),
        "rule": "alltime_top_k" if rule_active else "reign_scoped",
        "rule_active": bool(rule_active),
        "activation_block": int(activation_block),
        "k": int(k),
        "weights": {"gifteval": float(weights[0]), "boom": float(weights[1]),
                    "time": float(weights[2])},
        "notice_blocks": int(notice_blocks),
        "entries": [dict(e) for e in entries],
        "live_generation": int(live_generation),
        "live_members": [str(m) for m in live_members],
    }
    if upcoming:
        doc["upcoming"] = dict(upcoming)
    return doc


def publish_leaderboard(store: object, doc: dict) -> str:
    """Write the leaderboard doc public-read (same ACL fallback as the
    promotion record). Returns the key."""
    from .hippius import StorageError

    text = json.dumps(doc, indent=2, sort_keys=True)
    try:
        store.put_text(LEADERBOARD_KEY, text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        store.put_text(LEADERBOARD_KEY, text, content_type="application/json")
    return LEADERBOARD_KEY


def upcoming_from_doc(doc: object) -> dict | None:
    """The ``upcoming`` block out of a round/heat status doc's ``warm_start``
    or out of the leaderboard doc — ``None`` unless it is a well-formed
    announcement (a member list and an effective block). Shared by every
    consumer so a malformed announcement renders as nothing, never as a
    wrong checkpoint."""
    if not isinstance(doc, dict):
        return None
    up = doc.get("upcoming")
    if not isinstance(up, dict):
        ws = doc.get("warm_start")
        up = ws.get("upcoming") if isinstance(ws, dict) else None
    if not isinstance(up, dict):
        return None
    members = [m for m in (up.get("members") or ())
               if isinstance(m, dict) and m.get("checkpoint_id")]
    try:
        effective = int(up.get("effective_block"))
    except (TypeError, ValueError):
        return None
    if not members or effective <= 0:
        return None
    return {**up, "members": members, "effective_block": effective}


def annotate_upcoming(
    up: dict, *, now_block: int, seconds_per_block: float, now_s: float,
) -> dict:
    """Add the wall-clock ESTIMATE consumers show beside an announcement:
    ``blocks_remaining`` and ``effective_at`` (ISO-8601 UTC) derived from the
    chain cadence. The block is exact; the time is an estimate (labelled so
    by every renderer), recomputed by each consumer from its own block read."""
    from datetime import UTC, datetime

    remaining = max(0, int(up.get("effective_block", 0)) - int(now_block))
    eta = float(now_s) + remaining * float(seconds_per_block)
    return {**up, "blocks_remaining": remaining,
            "effective_at": datetime.fromtimestamp(eta, tz=UTC).isoformat(timespec="seconds")}
