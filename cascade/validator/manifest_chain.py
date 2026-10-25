"""Pure manifest-chain and commitment helpers for the validator loop."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..shared.manifest import load_manifest
from ..shared.receipt import Participant

log = logging.getLogger("cascade.validator")


def participants_from_commitments(commitments: list, cutoff_block: int,
                                  floor_block: int = 0) -> tuple[Participant, ...]:
    """The round's eligible participant set, for the public receipt.

    The receipt's eligibility rule (unchanged from the boundary-round trainer,
    so archived receipts replay): parseable generator pointers revealed STRICTLY BEFORE the epoch boundary and
    at/after the go-live ``floor_block``, latest commit per hotkey among the
    eligible ones — but keeps ``commit_block`` so an auditor can re-check every
    entrant against the cutoff. Sorted by UID for a deterministic receipt body.

    ``commitments`` should carry each hotkey's FULL reveal history
    (``poll_commitments(include_history=True)``): with only the latest reveal, a
    miner who re-committed after the boundary vanishes from the record even
    though their eligible pre-cutoff entry fielded the round.
    """
    from ..interface.validation import parse_commit

    best: dict[str, Participant] = {}
    for c in commitments:
        if floor_block and c.commit_block < floor_block:
            continue
        if c.commit_block >= cutoff_block:
            continue
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        prev = best.get(c.hotkey)
        if prev is None or c.commit_block >= prev.commit_block:
            best[c.hotkey] = Participant(
                hotkey=c.hotkey, uid=c.uid, gen_ref=parsed.ref, commit_block=c.commit_block
            )
    return tuple(sorted(best.values(), key=lambda p: p.uid))


def ref_as_of(commitments: list, hotkey: str, block: int) -> str | None:
    """The generator ref ``hotkey`` had revealed AS OF ``block`` (DEC-CA-0043
    ref binding at ``train_block``): the latest parseable commitment with
    ``commit_block <= block``. ``commitments`` is the full reveal history
    (``poll_commitments(include_history=True)``). None when nothing was
    revealed by then."""
    from ..interface.validation import parse_commit

    best_block, best_ref = -1, None
    for c in commitments:
        if c.hotkey != hotkey or int(c.commit_block) > int(block):
            continue
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        if int(c.commit_block) >= best_block:
            best_block, best_ref = int(c.commit_block), parsed.ref
    return best_ref


CHAIN_WALK_DEPTH = 16          # manifests per poll on the first attempt
CHAIN_WALK_MAX_DEPTH = 1024    # doubled per unreached poll up to this


@dataclass(frozen=True)
class ManifestChain:
    """:func:`chain_manifests` result: the manifests to handle OLDEST FIRST,
    and whether the walk reached the last handled settlement (or the chain's
    root). ``reached=False`` — a hole (unreadable round) or the depth cap —
    means NOTHING may be handled: judging the oldest collected manifest would
    fail its chain check and latch past every settlement in the gap, a
    dethrone included, and the validator's throne would silently diverge
    from the fleet's (the wedge the walk exists to prevent, one level
    down)."""

    manifests: tuple[tuple[str, str], ...]
    reached: bool


def chain_manifests(store: object, raw_latest: str, last_handled_round_id: str | None,
                    *, max_depth: int = CHAIN_WALK_DEPTH,
                    read_round: Callable[[object, str], str] | None = None,
                    ) -> ManifestChain:
    """The manifests to handle, OLDEST FIRST, walking ``prev_round_id`` back
    from ``latest.json`` to ``last_handled_round_id`` (DEC-CA-0043: never jump
    to latest — a validator that missed settlements catches up through every
    one, dethrones included, before the era-king envelope is checked).

    ``manifests`` is ``[(raw_text, sha256)]``. With no chain link on the
    latest manifest (pre-rollover), no persisted position, or a latest that
    already follows the last handled settlement, this is just ``[latest]``.
    A readable manifest with no ``prev_round_id`` is the chain's ROOT (the
    first settlement links to the last legacy round, which is unchained):
    reaching it counts as reached — everything before it was never chained.
    The walk is bounded by ``max_depth``; an unreadable round or the cap
    yields ``reached=False`` and the caller retries next poll, deeper.
    """
    from ..shared.hippius import manifest_round_key

    def _digest(raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    latest = load_manifest(raw_latest)
    chain: list[tuple[str, str]] = [(raw_latest, _digest(raw_latest))]
    prev = str(latest.prev_round_id or "")
    if (not prev or last_handled_round_id is None or prev == last_handled_round_id
            or str(latest.round_id) == last_handled_round_id):
        return ManifestChain(tuple(chain), True)
    reader = read_round or (lambda s, rid: s.get_text(manifest_round_key(rid)))
    seen = {str(latest.round_id)}
    reached = False
    while prev and prev != last_handled_round_id:
        if prev in seen:
            log.error("manifest chain: round %s links back into itself; walking stops", prev)
            break
        if len(chain) >= max_depth:
            log.warning("manifest chain: %d manifests walked without reaching %s (depth cap)",
                        len(chain), last_handled_round_id)
            break
        seen.add(prev)
        try:
            raw = reader(store, prev)
        except Exception as e:  # noqa: BLE001 — a hole in the chain: nothing is handled
            log.warning("manifest chain: round %s unreadable (%s); walking stops", prev, e)
            break
        m = load_manifest(raw)
        chain.append((raw, _digest(raw)))
        prev = str(m.prev_round_id or "")
    else:
        reached = True            # prev == last handled, or the root ("" — unchained before it)
    chain.reverse()
    return ManifestChain(tuple(chain), reached)
