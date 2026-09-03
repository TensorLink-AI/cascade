"""Public heat standings — ``status/heat.json`` and the ``heats/`` mirror.

The heat screen trains every eligible challenger cheaply, ranks them, advances
the top ``[round] finalists`` and throws the rest of the checkpoints away. Those
standings are the ONLY feedback a non-finalist miner ever gets about the round
it entered — and until this module they reached the public only as the
presentational ``heat`` block of the manifest a validator embeds in its signed
receipt. That receipt is published after the duel trained at the full budget
*and* a validator scored it: hours after the heat itself settled, and not at all
if the round is later rejected at a gate. A miner iterating between rounds was
therefore reading its heat placement long after the next submission deadline had
passed.

So the trainer publishes the standings the moment the heat settles — screened,
burned, finalists chosen, before a single duel pod is dispatched:

* ``status/heat.json`` — the latest heat, whichever round it belongs to. The
  dashboards' live panel: it carries ``epoch_start_block`` as the join key, so a
  consumer can tell "this round's heat, duel still running" from "last round's".
* ``heats/round-<id>.json`` — the same document under its own round key, so a
  standing never disappears when the next round overwrites the live pointer.
* ``heats/index.json`` — a rolling window of compact per-round summaries. A
  static page (or a miner's script) cannot LIST a bucket, so this is how the
  per-round documents are discoverable at all.

Everything here is presentational and unsigned, exactly like the manifest's heat
block: an auditor cannot cheaply reproduce a discarded heat checkpoint, so these
numbers are a *view*, never part of the signed claim or of weights. The signed
receipts remain the audit record, and a publish failure must never disturb the
round it is describing (callers wrap accordingly). The document is written with
the same field shape as the manifest block (:func:`cascade.shared.manifest.
heat_to_json`) so the dashboards render live and settled standings identically.
"""

from __future__ import annotations

import json

HEAT_STATUS_KEY = "status/heat.json"
HEAT_INDEX_KEY = "heats/index.json"
HEAT_STATUS_SCHEMA = 1
HEAT_INDEX_SCHEMA = 1

# Rolling window of per-round summaries kept in the index — the per-round
# documents themselves are never pruned (they are tiny and immutable).
HEAT_INDEX_MAX_KEEP = 400

# Cap on the per-hotkey rows inside the document's ``skipped`` block. The
# aggregate counts are always complete; the row list is a sample so a round
# with hundreds of skips (r48: 328 burned re-commits) stays a small document.
SKIPPED_ENTRIES_MAX = 25

# A live doc older than this describes a round whose heat is long over; a
# consumer wanting "the heat of the round in flight" ignores it (the per-round
# mirror is how you read an old heat). Generous: the heat settles early in the
# round and the duel that follows it runs for hours.
HEAT_STATUS_FRESH_SECONDS = 86_400.0
# Tolerated forward clock skew between the trainer and a consumer.
HEAT_STATUS_SKEW_SECONDS = 300.0


def heat_round_key(round_id: str) -> str:
    """Per-round key of the immutable heat mirror."""
    return f"heats/round-{round_id}.json"


def build_heat_status(
    heat: object,
    *,
    round_id: str,
    epoch_start_block: int,
    as_of: str,
    screened: int | None = None,
    network: str = "",
    netuid: int | None = None,
    no_screen_reason: str = "",
    finalists: int | None = None,
    warm_start: dict | None = None,
    skipped: list[dict] | None = None,
    duel_only: bool = False,
) -> dict:
    """Assemble the public heat document (pure — storage I/O stays with the caller).

    ``heat`` is the round's :class:`~cascade.shared.manifest.HeatResult`, or
    ``None`` when no screen ran (the field fit within the finalist slots, no
    screener was wired, or nobody was eligible). A no-screen round still
    publishes: without it the live pointer would keep showing the PREVIOUS
    round's standings as if they were this round's. ``no_screen_reason`` is the
    one-line human explanation the dashboards show in that case, and
    ``finalists`` (the configured slot count) fills the field the absent heat
    block would otherwise have carried.

    ``epoch_start_block`` is the consumer's join key — dashboards derive the
    current epoch start from the chain grid and match it against this field.

    ``warm_start`` is the round's promoted-init descriptor
    (``{init_checkpoint, size[, generation[, next_scheduled_init]]}``) once a
    cascade promotion is live, ``None`` in the random-init era. Every entrant
    this round trained from that one init, so it rides the round document, not
    the entrant rows. ``next_scheduled_init`` is the rotation's pick for the
    FOLLOWING round — a schedule, not a promise: a promotion firing at the
    boundary replaces the member set, and dashboards should caption it so.

    ``skipped`` — ``[{hotkey, uid, reason}]`` for hotkeys with a standing
    commitment that never became entrants (today: ``already_submitted``, a
    burned hotkey re-committing its one used submission). ADDITIVE: the field
    is only present when there are skips, existing fields are untouched, and
    the block is bounded — complete ``by_reason`` counts plus at most
    :data:`SKIPPED_ENTRIES_MAX` per-hotkey rows — so 300+ skips (r48) cannot
    bloat the document. Without it an all-burned field publishes as an
    unexplained ``0 entrants``.
    """
    from .manifest import heat_to_json

    doc: dict = {
        "schema": HEAT_STATUS_SCHEMA,
        "as_of": str(as_of),
        "round_id": str(round_id),
        "epoch_start_block": int(epoch_start_block),
        "screen_size": "",
        "finalists": 0,
        "entrants": [],
        "leader_lcb": None,
        "n_windows": None,
        "n_clusters": None,
    }
    body = heat_to_json(heat)
    if body is None:
        doc["no_screen"] = True
        doc["no_screen_reason"] = str(no_screen_reason)
        if finalists is not None:
            doc["finalists"] = int(finalists)
    else:
        doc.update(body)
    if duel_only:
        # A duel-only round: the entrants are ``seated`` (duelling the king
        # directly) or ``waiting`` (carried to a later round); no screen ran,
        # the reason line says how the seats were filled.
        doc["duel_only"] = True
        if no_screen_reason:
            doc["no_screen_reason"] = str(no_screen_reason)
    if screened is not None:
        doc["screened"] = int(screened)
    if skipped:
        doc["skipped"] = skipped_block(skipped)
    if warm_start:
        doc["warm_start"] = dict(warm_start)
    if network:
        doc["network"] = str(network)
    if netuid is not None:
        doc["netuid"] = int(netuid)
    return doc


def skipped_block(
    skipped: list[dict],
    *,
    max_entries: int = SKIPPED_ENTRIES_MAX,
) -> dict:
    """The bounded ``skipped`` block of the heat document.

    ``total`` and ``by_reason`` always cover EVERY skip; ``entries`` is the
    first ``max_entries`` rows (input order — the trainer records them in
    commitment order) with ``entries_truncated`` flagging the cut, so a
    consumer can tell "these are all of them" from "this is a sample".
    """
    rows = [r for r in skipped if isinstance(r, dict)]
    by_reason: dict[str, int] = {}
    for r in rows:
        key = str(r.get("reason", "unknown"))
        by_reason[key] = by_reason.get(key, 0) + 1
    return {
        "total": len(rows),
        "by_reason": dict(sorted(by_reason.items())),
        "entries": [{"hotkey": str(r.get("hotkey", "")),
                     "uid": r.get("uid"),
                     "reason": str(r.get("reason", "unknown"))}
                    for r in rows[:max_entries]],
        "entries_truncated": len(rows) > max_entries,
    }


def heat_summary(doc: dict) -> dict:
    """One compact ``heats/index.json`` entry for a published heat document.

    Carries what a listing needs — round context, field size, who led, how
    decisive the screen was — plus ``heat_key``, the pointer back to the full
    per-round document.
    """
    ents = [e for e in doc.get("entrants", ()) if isinstance(e, dict)]
    leader = next((e for e in ents if e.get("rank") == 1), None)
    advanced = [e for e in ents if str(e.get("status")) in ("advanced", "seated")]
    skipped = doc.get("skipped") if isinstance(doc.get("skipped"), dict) else {}
    return {
        "duel_only": bool(doc.get("duel_only", False)),
        "round_id": str(doc.get("round_id", "")),
        "epoch_start_block": int(doc.get("epoch_start_block", 0)),
        "as_of": str(doc.get("as_of", "")),
        "screen_size": str(doc.get("screen_size", "")),
        "finalists": doc.get("finalists"),
        "n_entrants": len(ents),
        "n_advanced": len(advanced),
        "n_skipped": int(skipped.get("total", 0) or 0),
        "no_screen": bool(doc.get("no_screen", False)),
        "leader_uid": (leader or {}).get("uid"),
        "leader_hotkey": (leader or {}).get("hotkey"),
        "leader_lcb": doc.get("leader_lcb"),
        "n_windows": doc.get("n_windows"),
        "n_clusters": doc.get("n_clusters"),
        "heat_key": heat_round_key(str(doc.get("round_id", ""))),
    }


def publish_heat_status(store: object, doc: dict) -> list[str]:
    """Write the heat document to its per-round key AND the live pointer.

    Per-round key first: if the second write fails the standing is still
    readable (and indexable), whereas a live pointer without its round mirror
    would vanish at the next round. Returns the keys written.
    """
    round_key = heat_round_key(str(doc.get("round_id", "")))
    keys = [round_key, HEAT_STATUS_KEY]
    for key in keys:
        _publish_public_json(store, key, doc)
    return keys


def read_heat_index(store: object) -> dict:
    """Read ``heats/index.json``; an empty index if absent or malformed."""
    from .hippius import StorageError

    empty = {"schema": HEAT_INDEX_SCHEMA, "heats": []}
    try:
        text = store.get_text(HEAT_INDEX_KEY)
    except StorageError:
        return empty
    except Exception:  # noqa: BLE001 — any store hiccup reads as "no index yet"
        return empty
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return empty
    if not isinstance(obj, dict) or not isinstance(obj.get("heats"), list):
        return empty
    return obj


def update_heat_index(
    store: object,
    doc: dict,
    *,
    max_keep: int = HEAT_INDEX_MAX_KEEP,
) -> dict:
    """Append/replace this round's entry in ``heats/index.json``, public-read.

    Idempotent per ``round_id`` (a re-run round replaces its own entry), sorted
    by ``epoch_start_block`` then ``round_id`` — round ids are block-hash seeds,
    not monotonic — and capped at ``max_keep`` most recent entries. The trainer
    is the only writer, so unlike the receipts index there is no cross-writer
    merge to lose. Returns the stored entry.
    """
    entry = heat_summary(doc)
    rid = entry["round_id"]
    index = read_heat_index(store)
    heats = [h for h in index.get("heats", [])
             if isinstance(h, dict) and str(h.get("round_id")) != rid]
    heats.append(entry)
    heats.sort(key=lambda h: (int(h.get("epoch_start_block", 0)), str(h.get("round_id", ""))))
    out = {
        "schema": HEAT_INDEX_SCHEMA,
        "updated_at": str(doc.get("as_of", "")),
        "heats": heats[-max_keep:],
    }
    _publish_public_json(store, HEAT_INDEX_KEY, out)
    return entry


def live_heat(
    doc: object,
    *,
    epoch_start_block: int,
    now_s: float,
    max_age_s: float = HEAT_STATUS_FRESH_SECONDS,
) -> dict | None:
    """Validate a fetched ``status/heat.json`` as THIS round's heat.

    Returns the doc when it is well-formed, matches ``epoch_start_block``, and
    its ``as_of`` is within ``max_age_s`` of ``now_s`` (epoch seconds; small
    forward skew tolerated). Anything else — another round's standings, stale,
    malformed — returns None, and the caller falls back to the settled round's
    manifest block (or shows nothing).
    """
    from datetime import datetime

    if not isinstance(doc, dict):
        return None
    if not isinstance(doc.get("entrants"), list):
        return None
    try:
        if int(doc.get("epoch_start_block", -1)) != int(epoch_start_block):
            return None
        as_of = datetime.fromisoformat(str(doc.get("as_of", "")))
    except (TypeError, ValueError):
        return None
    if as_of.tzinfo is None:
        return None  # naive timestamps are ambiguous across hosts; reject
    age = float(now_s) - as_of.timestamp()
    if age > max_age_s or age < -HEAT_STATUS_SKEW_SECONDS:
        return None
    return doc


def _publish_public_json(store: object, key: str, doc: dict) -> str:
    """Write ``doc`` public-read, retried private on a backend that rejects
    canned object ACLs (mirrors the receipt-index publish)."""
    from .hippius import StorageError

    text = json.dumps(doc, indent=2, sort_keys=True)
    try:
        store.put_text(key, text, content_type="application/json", acl="public-read")
    except StorageError:
        store.put_text(key, text, content_type="application/json")
    return key
