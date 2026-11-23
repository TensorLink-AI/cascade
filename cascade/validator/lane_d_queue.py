"""Lane D queue selection (LANE_D_PLAN §1.2 "Enforcement" / "Coldkey
resolution", §1.5 "Maximum reveal delay"; W5h).

:func:`select_queue` is what :func:`~cascade.validator.lane_d.settle_lane_d`
calls, after registration, to pick the entries judged this settlement:

1. **Reveal-delay close.** An entry still ``pending`` at a boundary at or past
   its reveal deadline (``reveal_block + max_reveal_delay_days * 7200``)
   closes unjudged, ``closed`` / ``reveal_delay_exceeded``, logged, before
   anything is selected (A9).
2. **The coldkey re-check at reveal** (D22 option b). The validator's re-read
   of the hotkey's coldkey reaches the settlement as the optional
   ``coldkey_at_reveal`` key of an ``entries`` item. Absent: no re-check.
   Equal to ``coldkey_at_ready``: resolved, selected as usual. A different
   coldkey: ``closed`` / ``coldkey_changed``. ``None`` (unreadable):
   ``pending`` / ``coldkey_unresolved``, alerted (``lane_d_coldkey_unresolved``),
   never selected and never charged; at its
   ``max_coldkey_unresolved_settlements``-th unresolved settlement it closes,
   ``closed`` / ``coldkey_unresolved``. A later listing with a readable value
   resolves it. The count and the resolved value are kept on the entry record
   (``coldkey_unresolved_settlements``, ``coldkey_at_reveal``) so they survive
   between settlements.
3. **FIFO by T under the caps.** The remaining open entries are walked by
   ``(reveal_ts, digest)``; one over ``max_entries_per_settlement`` or
   ``max_judged_per_coldkey_per_settlement`` (per ``coldkey_at_ready``) is
   skipped ``coldkey_cap_settlement``, carried and not charged.

Entries closed or held unresolved here never count against a cap.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# A9: the reveal delay is configured in days of 12 s blocks.
BLOCKS_PER_DAY = 7200
# The optional re-check key of an ``entries`` item (and of the resolved record).
RECHECK_KEY = "coldkey_at_reveal"
UNRESOLVED_COUNT_KEY = "coldkey_unresolved_settlements"
OPEN_STATUSES = ("pending", "confirm_pending")

CAP_REASON = "coldkey_cap_settlement"
CHANGED_REASON = "coldkey_changed"
UNRESOLVED_REASON = "coldkey_unresolved"
DELAY_REASON = "reveal_delay_exceeded"
UNRESOLVED_ALERT = "lane_d_coldkey_unresolved"
DELAY_LOG_TAG = "lane_d_reveal_delay_exceeded"


def reveal_deadline_block(rec: dict, cfg) -> int:
    """The block at which an entry still pending closes (inclusive)."""
    return int(rec["reveal_block"]) + int(cfg.max_reveal_delay_days) * BLOCKS_PER_DAY


def _close_overdue(state: dict, cfg, boundary: int, exclude: frozenset[str]) -> list[dict]:
    skipped = []
    for dg, rec in sorted(state["entries"].items()):
        if rec["status"] != "pending" or dg in exclude:
            continue
        deadline = reveal_deadline_block(rec, cfg)
        if int(boundary) >= deadline:
            rec.update(status="closed", reason=DELAY_REASON)
            skipped.append({"digest": dg, "reason": DELAY_REASON})
            logger.warning("%s %s: still pending at boundary %d >= reveal deadline %d "
                           "(max_reveal_delay_days %d); closed unjudged, %s",
                           DELAY_LOG_TAG, dg[:12], int(boundary), deadline,
                           int(cfg.max_reveal_delay_days), DELAY_REASON)
    return skipped


def _recheck(state: dict, cfg, boundary: int, listed: dict[str, dict],
             exclude: frozenset[str]) -> tuple[list[dict], set[str]]:
    """Apply the re-check; returns the skips and the digests held unresolved."""
    skipped: list[dict] = []
    unresolved: set[str] = set()
    bound = int(cfg.max_coldkey_unresolved_settlements)
    for dg, rec in sorted(state["entries"].items()):
        if rec["status"] != "pending" or dg in exclude or RECHECK_KEY in rec:
            continue
        item = listed.get(dg)
        if item is not None and RECHECK_KEY in item:
            seen = item[RECHECK_KEY]
        elif UNRESOLVED_COUNT_KEY in rec:
            # Still unresolved: no readable value has reached the validator.
            seen = None
        else:
            continue
        if seen is not None and str(seen):
            rec.pop(UNRESOLVED_COUNT_KEY, None)
            if str(seen) == rec["coldkey_at_ready"]:
                rec[RECHECK_KEY] = str(seen)
                if rec["reason"] == UNRESOLVED_REASON:
                    rec["reason"] = "pending_floor"
            else:
                rec.update(status="closed", reason=CHANGED_REASON)
                skipped.append({"digest": dg, "reason": CHANGED_REASON})
            continue
        count = int(rec.get(UNRESOLVED_COUNT_KEY, 0)) + 1
        rec[UNRESOLVED_COUNT_KEY] = count
        closing = count >= bound
        rec.update(status="closed" if closing else "pending", reason=UNRESOLVED_REASON)
        skipped.append({"digest": dg, "reason": UNRESOLVED_REASON})
        unresolved.add(dg)
        logger.warning("%s %s: coldkey re-check at reveal unreadable at boundary %d "
                       "(%d of %d settlements); %s", UNRESOLVED_ALERT, dg[:12], int(boundary),
                       count, bound, "closed unjudged" if closing else "held, not selected")
    return skipped, unresolved


def _fifo_caps(state: dict, cfg, exclude: frozenset[str]) -> tuple[list[str], list[dict]]:
    open_ = sorted((dg for dg, r in state["entries"].items()
                    if r["status"] in OPEN_STATUSES and dg not in exclude),
                   key=lambda dg: (state["entries"][dg]["reveal_ts"], dg))
    chosen, skipped, per_cold = [], [], {}
    for dg in open_:
        cold = state["entries"][dg]["coldkey_at_ready"]
        if (len(chosen) >= int(cfg.max_entries_per_settlement)
                or per_cold.get(cold, 0) >= int(cfg.max_judged_per_coldkey_per_settlement)):
            state["entries"][dg]["reason"] = CAP_REASON
            skipped.append({"digest": dg, "reason": CAP_REASON})
            continue
        per_cold[cold] = per_cold.get(cold, 0) + 1
        chosen.append(dg)
    return chosen, skipped


def select_queue(state: dict, cfg, boundary: int, entries: list[dict],
                 exclude: frozenset[str] = frozenset()) -> tuple[list[str], list[dict]]:
    """This settlement's selected digests and the typed skips the queue
    produced (reveal-delay closes, re-check refusals, unresolved holds and
    cap skips). ``exclude`` (held gate candidates, re-gated without a duel)
    is never touched, selected or counted against a cap."""
    listed = {str(e["digest"]): e for e in entries}
    skipped = _close_overdue(state, cfg, boundary, exclude)
    checked, unresolved = _recheck(state, cfg, boundary, listed, exclude)
    skipped += checked
    chosen, capped = _fifo_caps(state, cfg, exclude | frozenset(unresolved))
    return chosen, skipped + capped


__all__ = ["BLOCKS_PER_DAY", "RECHECK_KEY", "UNRESOLVED_ALERT", "reveal_deadline_block",
           "select_queue"]
