"""Live chain status for the dashboard — ``status/chain.json``.

The web dashboard is a static page reading public-read JSON from the manifest
bucket; it cannot poll the Bittensor chain itself. Receipts settle only at the
END of a round, so between receipts the page had no live view of the chain —
no fresh block anchor, and no way to show a submission the moment it is
revealed. This module publishes that missing view: a small public-read
``status/chain.json`` written on the validator's poll cadence, carrying

* a fresh chain anchor (``as_of``, ``current_block``, the epoch grid) for the
  round-stage strip and the next-round countdown, and
* every currently revealed generator commitment (uid, hotkey, ref, commit
  block) — the dashboard's live submissions feed.

It also owns the round-stage window derivation (`stage_windows`) shared by the
``cascade round`` CLI and stamped into the status doc, so the terminal and web
dashboards estimate heat/duel/validation off the same numbers.

Everything here is presentational and best-effort: nothing is signed, nothing
feeds weights, and a publish failure must never disturb a round (callers wrap
it accordingly). The signed receipts remain the audit record.
"""

from __future__ import annotations

import json

CHAIN_STATUS_KEY = "status/chain.json"
CHAIN_STATUS_SCHEMA = 1

# Trainer-reported round stage — ``status/round.json``. The wall-clock stage
# ESTIMATE below (`stage_windows`) models the heat as ONE competitor's budget,
# so on a large field (65 entrants across 4 lanes ≈ 8h of heats) it calls
# "duel" while the trainer is genuinely mid-heat. This doc is the trainer
# saying where the round actually is: written at every stage transition (and
# throttled heat progress), consumed by the dashboards when fresh, with the
# estimate as the fallback. Presentational and unsigned like the rest of this
# module — it never feeds weights, and consumers must survive it being stale,
# absent, or malformed.
ROUND_STATUS_KEY = "status/round.json"
ROUND_STATUS_SCHEMA = 1
ROUND_STAGES = ("heat", "duel", "validation")
# A doc older than this is ignored (trainer restarted/paused/crashed — fall
# back to the estimate rather than pinning the strip on a dead stage).
ROUND_STATUS_FRESH_SECONDS = 3600.0
# Tolerated forward clock skew between the trainer and a consumer.
ROUND_STATUS_SKEW_SECONDS = 300.0

# Fixed per-stage overhead the stage estimate absorbs on top of the training
# budgets: generator fetch, sandbox boot, screening eval, checkpoint upload,
# manifest publish. Rough by design — the pre-settle stages are estimates
# until the round's public receipt confirms it settled.
STAGE_OVERHEAD_SECONDS = 900.0


def stage_windows(cfg: object) -> tuple[float, float]:
    """Rough wall-clock ``(heat_seconds, duel_seconds)`` for one round.

    Derived from the same budgets the trainer enforces: the heat's wall-clock
    cap (``[round] heat_*``, mirroring ``TrainingContractConfig.for_hours``)
    and the final duel's per-size ``max_train_seconds`` (summed — sizes train
    sequentially), each padded with :data:`STAGE_OVERHEAD_SECONDS`. Anything
    past ``heat + duel`` is presumed duel validation until the receipt lands.
    """
    rnd = cfg.round
    guard = max(
        rnd.heat_guard_factor * rnd.heat_train_hours * 3600.0,
        float(rnd.heat_guard_floor_seconds),
    )
    heat_wall = min(guard, float(cfg.screen_contract().max_train_seconds))
    duel_wall = float(sum(c.max_train_seconds for c in cfg.throne_contracts()))
    return heat_wall + STAGE_OVERHEAD_SECONDS, duel_wall + STAGE_OVERHEAD_SECONDS


def build_chain_status(
    cfg: object,
    *,
    current_block: int,
    commitments: list,
    network: str = "",
    as_of: str = "",
) -> dict:
    """Assemble the status document (pure — chain I/O stays with the caller).

    ``commitments`` is the latest revealed commitment per hotkey (what
    ``ChainClient.poll_commitments`` returns). Malformed payloads and
    pre-``commit_floor_block`` commits are dropped, mirroring the trainer's
    eligibility rules; the dashboard splits this-round vs next-round itself
    from each entry's ``commit_block`` against the epoch grid.
    """
    from ..interface.validation import parse_commit

    epoch_blocks = max(1, int(cfg.round.epoch_blocks))
    floor = int(cfg.round.commit_floor_block)
    block_time = (
        cfg.round.round_hours * 3600.0 / epoch_blocks
        if cfg.round.round_hours > 0 and epoch_blocks > 0 else 12.0
    )
    subs = []
    for c in commitments:
        if floor and c.commit_block < floor:
            continue
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        subs.append({
            "uid": int(c.uid),
            "hotkey": str(c.hotkey),
            "gen_ref": parsed.ref,
            "commit_block": int(c.commit_block),
        })
    subs.sort(key=lambda s: (-s["commit_block"], s["uid"]))
    heat_s, duel_s = stage_windows(cfg)
    return {
        "schema": CHAIN_STATUS_SCHEMA,
        "as_of": str(as_of),
        "network": str(network),
        "netuid": int(cfg.subnet.netuid),
        "current_block": int(current_block),
        "epoch_blocks": epoch_blocks,
        "epoch_start_block": (int(current_block) // epoch_blocks) * epoch_blocks,
        "block_time_s": block_time,
        "stage_windows": {"heat_seconds": heat_s, "duel_seconds": duel_s},
        "submissions": subs,
    }


def publish_chain_status(store: object, status: dict) -> str:
    """Write the status doc public-read (mirrors the receipt-index publish:
    retried without the ACL on backends that reject canned object ACLs).
    Returns the key."""
    return _publish_public_json(store, CHAIN_STATUS_KEY, status)


def build_round_status(
    *,
    round_id: str,
    epoch_start_block: int,
    stage: str,
    as_of: str,
    heat_done: int | None = None,
    heat_total: int | None = None,
    finalists: int | None = None,
) -> dict:
    """Assemble the trainer-reported round-stage doc (pure).

    ``epoch_start_block`` is the consumer's join key: dashboards derive the
    current epoch start from the chain grid and only trust a doc for THAT
    round (a doc left behind by a previous round simply doesn't match).
    ``heat_done``/``heat_total`` give the heat a real progress number.
    """
    if stage not in ROUND_STAGES:
        raise ValueError(f"unknown round stage: {stage!r}")
    doc: dict = {
        "schema": ROUND_STATUS_SCHEMA,
        "as_of": str(as_of),
        "round_id": str(round_id),
        "epoch_start_block": int(epoch_start_block),
        "stage": str(stage),
    }
    if heat_total is not None:
        doc["heat_total"] = int(heat_total)
    if heat_done is not None:
        doc["heat_done"] = int(heat_done)
    if finalists is not None:
        doc["finalists"] = int(finalists)
    return doc


def publish_round_status(store: object, status: dict) -> str:
    """Write the trainer-reported round-stage doc public-read."""
    return _publish_public_json(store, ROUND_STATUS_KEY, status)


def live_round_stage(
    doc: object,
    *,
    epoch_start_block: int,
    now_s: float,
    max_age_s: float = ROUND_STATUS_FRESH_SECONDS,
) -> dict | None:
    """Validate a fetched ``status/round.json`` against the current round.

    Returns the doc (as a plain dict) when it is well-formed, reports a known
    stage, matches ``epoch_start_block``, and its ``as_of`` is within
    ``max_age_s`` of ``now_s`` (epoch seconds; small forward skew tolerated).
    Anything else — stale, another round's leftover, malformed, wrong types —
    returns None and the caller falls back to the wall-clock estimate.
    """
    from datetime import datetime

    if not isinstance(doc, dict):
        return None
    if doc.get("stage") not in ROUND_STAGES:
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
    if age > max_age_s or age < -ROUND_STATUS_SKEW_SECONDS:
        return None
    return doc


def _publish_public_json(store: object, key: str, doc: dict) -> str:
    from .hippius import StorageError

    text = json.dumps(doc, indent=2, sort_keys=True)
    try:
        store.put_text(key, text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        store.put_text(key, text, content_type="application/json")
    return key
