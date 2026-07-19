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
    from .hippius import StorageError

    text = json.dumps(status, indent=2, sort_keys=True)
    try:
        store.put_text(CHAIN_STATUS_KEY, text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        store.put_text(CHAIN_STATUS_KEY, text, content_type="application/json")
    return CHAIN_STATUS_KEY
