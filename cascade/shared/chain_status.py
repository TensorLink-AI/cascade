"""Live chain status for the dashboard — ``status/chain.json``.

The web dashboard is a static page reading public-read JSON from the manifest
bucket; it cannot poll the Bittensor chain itself. Receipts settle only at the
END of a round, so between receipts the page had no live view of the chain —
no fresh block anchor, and no way to show a submission the moment it is
revealed. This module publishes that missing view: a small public-read
``status/chain.json`` written on the validator's poll cadence, carrying

* a fresh chain anchor (``as_of``, ``current_block``, the epoch grid) for the
  next-settlement countdown, and
* every currently revealed generator commitment (uid, hotkey, ref, commit
  block) — the dashboard's live submissions feed.

Everything here is presentational and best-effort: nothing is signed, nothing
feeds weights, and a publish failure must never disturb a round (callers wrap
it accordingly). The signed receipts remain the audit record.
"""

from __future__ import annotations

from .config import BLOCK_SECONDS
from .hippius import publish_public_json

CHAIN_STATUS_KEY = "status/chain.json"
CHAIN_STATUS_SCHEMA = 1


def build_chain_status(
    cfg: object,
    *,
    current_block: int,
    commitments: list,
    network: str = "",
    as_of: str = "",
    economics: dict | None = None,
    activation: dict | None = None,
) -> dict:
    """Assemble the status document (pure — chain I/O stays with the caller).

    ``commitments`` is the latest revealed commitment per hotkey (what
    ``ChainClient.poll_commitments`` returns). Malformed payloads and
    pre-``commit_floor_block`` commits are dropped, mirroring the trainer's
    eligibility rules; the dashboard splits this-round vs next-round itself
    from each entry's ``commit_block`` against the epoch grid.

    ``economics`` is ``ChainClient.subnet_economics()`` output (alpha price,
    TAO emissions/day) and is included verbatim when provided — the
    stakeholder scoreboard's Economics cell reads it. ``activation`` is the
    validator's stake-weighted activation view (DEC-CA-0045,
    ``cascade.shared.activation.summary``): feature, threshold, the last
    boundary tally, lock-in and rollover blocks. Like everything else here
    it is presentational and unsigned.
    """
    from ..interface.validation import parse_commit
    from .config import effective_epoch_blocks
    from .era import era_length_blocks

    # The grid IN FORCE at the current block: while a switch is scheduled
    # (typed in or resolved by activation) the raw field is the post-switch
    # length and the strip would count boundaries that do not exist yet.
    epoch_blocks = max(1, int(effective_epoch_blocks(cfg.round, int(current_block))))
    floor = int(cfg.round.commit_floor_block)
    # The chain's fixed block time, NOT round_hours / epoch_blocks (that ratio
    # is 48 s/block once the rollover shrinks the grid to 900 blocks).
    block_time = BLOCK_SECONDS
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
    return {
        "schema": CHAIN_STATUS_SCHEMA,
        "as_of": str(as_of),
        "network": str(network),
        "netuid": int(cfg.subnet.netuid),
        "current_block": int(current_block),
        "epoch_blocks": epoch_blocks,
        "era_blocks": era_length_blocks(cfg.round, int(current_block)),
        "epoch_start_block": (int(current_block) // epoch_blocks) * epoch_blocks,
        "block_time_s": block_time,
        "submissions": subs,
        **({"economics": economics} if economics else {}),
        **({"activation": activation} if activation else {}),
    }


def publish_chain_status(store: object, status: dict) -> str:
    """Write the status doc public-read (mirrors the receipt-index publish:
    retried without the ACL on backends that reject canned object ACLs).
    Returns the key."""
    return publish_public_json(store, CHAIN_STATUS_KEY, status)
