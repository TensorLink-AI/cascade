"""Lane kill switches at the funding intake (#931).

A lane-wide ``[lane_b] until_block`` / ``[lane_c] until_block`` (the #889
containment kill switches, usually set by an adopted owner-signed overlay) or
a lane-C competition's own ``until_block`` ends that lane (or competition):
the settlement AT ``until_block`` is the last one judged, so a fund accepted
at chain head ``h`` settles at the earliest at the first boundary after
``h`` — never judged once ``h >= until_block``. The intake therefore:

* refuses a new fund with a typed ``409 lane_ended`` (lane-wide end) or
  ``409 competition_ended`` (the competition's own end), carrying the end
  block, BEFORE the provider key is vaulted (:func:`fund_refusal`);
* releases the already-queued, not yet started entries of an ended lane or
  competition: status ``failed``, ``last_error_class`` ``lane_ended`` /
  ``competition_ended`` — no pod was rented, nothing was charged, it is not
  the miner's fault (:func:`release_ended`). An in-flight leg is never
  touched (the trainer runs it to its verdict and keeps its key);
* accepts a fund for a lane whose ``share`` (or a competition whose
  ``weight``) is 0 in the config in force — a paused lane can resume — but
  says so in the reply (:func:`pay_warning`).

W-LAUNCH adds lane A's own cutoff, ``[round] lane_a_until_block`` (trainer +
intake policy, never an overlay key): :func:`lane_a_refusal` refuses a new
lane A fund with the same ``409 lane_ended`` (``lane`` = ``"a"``) and
:func:`lane_a_released_why` releases its not yet started entries.

Pure functions of the config in force (``EffectiveLanes`` view) and the head
block: no chain read, no consensus input. Nothing has ended (every
``until_block`` 0 or still ahead) ⇒ every function here is a no-op.
"""

from __future__ import annotations

LANE_ENDED = "lane_ended"
COMPETITION_ENDED = "competition_ended"


def end_block(cfg, head: int | None) -> int | None:
    """``cfg.until_block`` when it is set and ``head`` has reached it, else
    None (open, no end, or the head unknown)."""
    if cfg is None or head is None:
        return None
    until = int(getattr(cfg, "until_block", 0) or 0)
    if until > 0 and int(head) >= until:
        return until
    return None


def lane_a_end_block(until_block: int | None, head: int | None) -> int | None:
    """W-LAUNCH: ``[round] lane_a_until_block`` when it is set and ``head``
    has reached it, else None (open, no cutoff, or the head unknown)."""
    until = int(until_block or 0)
    if until > 0 and head is not None and int(head) >= until:
        return until
    return None


def lane_a_refusal(until_block: int | None, head: int | None) -> dict | None:
    """W-LAUNCH (owner 2026-11-25, "lane A should shut down when the other
    lanes start"): the typed 409 body refusing a new lane A fund (``/v1/fund``
    or a funded ``/v1/submit``) once the head reached ``[round]
    lane_a_until_block``, else None (admit). Same ``lane_ended`` shape as the
    lane B / C kill switches, ``lane`` = ``"a"``. Trainer + intake policy: no
    overlay key, no consensus input."""
    until = lane_a_end_block(until_block, head)
    if until is None:
        return None
    return {"code": LANE_ENDED, "lane": "a", "until_block": int(until), "block": int(head),  # type: ignore[arg-type]
            "message": f"lane A ended at block {until} ([round] lane_a_until_block; head "
                       f"{head}): no new lane A funds are admitted — nothing was queued, "
                       "your key was not kept. Lanes B, C and D are open."}


def lane_a_released_why(until_block: int | None, head: int | None):
    """``why(entry)`` for :meth:`FundedQueue.release_unstarted`: every not yet
    started lane A entry is released once the cutoff is in force (no pod was
    rented, nothing charged), else None (keep it)."""
    until = lane_a_end_block(until_block, head)
    if until is None:
        return None
    reason = (f"lane A ended at block {until}: released before training — no pod "
              "rented, nothing charged (not the miner's fault)", LANE_ENDED)
    return lambda entry: reason


def _lane_refusal(lane: str, until: int, head: int) -> dict:
    return {"code": LANE_ENDED, "lane": lane, "until_block": int(until), "block": int(head),
            "message": f"lane {lane.upper()} ended at block {until} ([lane_{lane}] "
                       f"until_block in the config in force; head {head}): no new funds "
                       "are admitted — nothing was queued, your key was not kept"}


def _competition(lc, target: str):
    try:
        return lc.competition_of(target)
    except Exception:  # noqa: BLE001 — an unknown kind has no competition end
        return None


def fund_refusal(lane: str, cfg, head: int | None, *, target: str | None = None) -> dict | None:
    """The typed 409 body refusing a new lane-``lane`` fund at ``head``, or
    None (admit). Lane C with a ``target``: the lane-wide end first, then
    the target's competition (a ``[lane_c.competitions.<kind>]`` table or a
    DEC-CA-0075 spec instance) end."""
    until = end_block(cfg, head)
    if until is not None:
        return _lane_refusal(lane, until, int(head))  # type: ignore[arg-type]
    if lane == "c" and target is not None and cfg is not None:
        comp = _competition(cfg, target)
        until = end_block(comp, head)
        if until is not None:
            return {"code": COMPETITION_ENDED, "lane": "c", "competition": str(comp.kind),
                    "until_block": int(until), "block": int(head),  # type: ignore[arg-type]
                    "message": f"lane-C competition {comp.kind!r} ended at block {until} — "
                               f"no new {target!r} entries are admitted; nothing was "
                               "queued, your key was not kept"}
    return None


def pay_warning(lane: str, cfg, *, target: str | None = None) -> str:
    """A reply warning when the fund is accepted into a lane (or lane-C
    competition) that pays nothing in the config in force; "" otherwise."""
    if cfg is None:
        return ""
    share = getattr(cfg, "share", None)
    if share is not None and float(share) <= 0.0:
        return (f"lane {lane.upper()} share is 0 in the config in force: the fund is "
                "accepted (a paused lane can resume) but a win pays nothing while the "
                "share stays 0")
    if lane == "c" and target is not None:
        comp = _competition(cfg, target)
        weight = getattr(comp, "weight", None) if comp is not None else None
        if weight is not None and float(weight) <= 0.0:
            return (f"lane-C competition {comp.kind!r} has weight 0 in the config in force: "
                    "the fund is accepted (it can resume) but a win pays nothing while "
                    "the weight stays 0")
    return ""


def _released_why(lane: str, cfg, head: int, entry) -> tuple[str, str] | None:
    until = end_block(cfg, head)
    if until is not None:
        return (f"lane {lane.upper()} ended at block {until}: released before training — "
                "no pod rented, nothing charged (not the miner's fault)", LANE_ENDED)
    if lane != "c":
        return None
    from .queue import split_lane_c_ref

    try:
        target, _ = split_lane_c_ref(entry.ref)
    except ValueError:
        return None
    comp = _competition(cfg, target)
    until = end_block(comp, head)
    if until is not None:
        return (f"lane-C competition {comp.kind!r} ended at block {until}: released before "
                "training — no pod rented, nothing charged (not the miner's fault)",
                COMPETITION_ENDED)
    return None


def release_ended(lane: str, queue, cfg, head: int | None) -> list[str]:
    """Release (terminally fail, nothing charged) every not yet started entry
    of ``queue`` whose lane or competition has ended at ``head``; returns the
    released hotkeys. A no-op when nothing has ended."""
    if queue is None or cfg is None or head is None:
        return []
    h = int(head)
    lane_until = end_block(cfg, h)
    if lane_until is None and lane != "c":
        return []
    if lane_until is None and not any(
            end_block(c, h) is not None for c in getattr(cfg, "competition_list", ()) or ()):
        return []
    return queue.release_unstarted(lambda e: _released_why(lane, cfg, h, e))


__all__ = ["COMPETITION_ENDED", "LANE_ENDED", "end_block", "fund_refusal", "lane_a_end_block",
           "lane_a_refusal", "lane_a_released_why", "pay_warning", "release_ended"]
