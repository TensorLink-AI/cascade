"""Lane B row claims and overlap deferral (plan §12b.2 LB2 "Overlaps", WB15).

A funded lane B entry may declare the expert rows its leg trains (WB14,
``declared_rows``). This module is the pure planning half of honouring them:

* **claim order** — the earliest reveal holds: the funded entry's
  ``reveal_block``, then the hotkey (the reveal-seniority rule of
  ``cascade.funding.queue`` and ``expert_lane.LaneBQueue.next_dispatchable``);
* **holders** — rows are exclusive against open legs, finished-but-unsettled
  legs (verdict pending) and pending merges (until the era init consumes
  them), whichever mode (declared or ESFT) planned them (DEC-CA-0059);
* **deferral** — a declared claim overlapping a holder waits, uncharged, and
  is reported as ``deferred: rows_held`` with the holder's entry and the
  expected release (the holder's verdict or merge era);
* **sentinel targets** (WB18, §12b.5 LB5, D35) — a suite-target entry on the
  era's sentinel waits, uncharged, as ``deferred: sentinel_target``, in at
  most ``sentinel_max_defer_eras`` distinct eras; after that it is judged on
  its target regardless.

Torch-free: the driver calls it under its lock with plain data.
"""

from __future__ import annotations

from dataclasses import dataclass

ROWS_HELD = "rows_held"
RELEASE_VERDICT = "verdict"
RELEASE_MERGE = "merge"


def normalise_rows(rows) -> dict[int, list[int]]:
    """``{layer: sorted unique rows}`` with int keys and empty layers dropped."""
    out: dict[int, list[int]] = {}
    for li, rs in (rows or {}).items():
        vals = sorted({int(r) for r in rs})
        if vals:
            out[int(li)] = vals
    return dict(sorted(out.items()))


def row_cells(rows) -> set[tuple[int, int]]:
    """The ``(layer, row)`` cells of a selection."""
    return {(int(li), int(r)) for li, rs in (rows or {}).items() for r in rs}


def claim_key(reveal_block: int, hotkey: str) -> tuple[int, str]:
    """Seniority of a claim: earliest reveal, then the lower hotkey."""
    return int(reveal_block), str(hotkey)


@dataclass(frozen=True)
class Holder:
    """Rows some leg or merge holds, and when they are expected back."""

    hotkey: str
    entry_ref: str
    rows: frozenset                       # {(layer, row)}
    kind: str                             # RELEASE_VERDICT | RELEASE_MERGE
    era: int

    def public(self) -> dict:
        return {"hotkey": self.hotkey, "entry_ref": self.entry_ref}


def verdict_release_era(base_era: int, era_index: int, max_base_age_eras: int) -> int:
    """The era an open or finished-but-unsettled leg's rows are expected
    back: its verdict at this era's settlement at the earliest, and at the
    latest the era past which settlement drops it unjudged
    (``base_era + max_base_age_eras``, ``LaneBQueue.stale``)."""
    lo, hi = int(base_era), int(base_era) + int(max_base_age_eras)
    return min(max(int(era_index), lo), hi)


def leg_holder(hotkey: str, entry_ref: str, selection, base_era: int, era_index: int,
               max_base_age_eras: int) -> Holder:
    return Holder(hotkey=str(hotkey), entry_ref=str(entry_ref),
                  rows=frozenset(row_cells(selection)), kind=RELEASE_VERDICT,
                  era=verdict_release_era(base_era, era_index, max_base_age_eras))


def merge_holder(hotkey: str, entry_ref: str, selection, era_index: int) -> Holder:
    """A pending merge holds its rows until the NEXT era's init applies it."""
    return Holder(hotkey=str(hotkey), entry_ref=str(entry_ref),
                  rows=frozenset(row_cells(selection)), kind=RELEASE_MERGE,
                  era=int(era_index) + 1)


def first_overlap(rows, holders) -> Holder | None:
    """The first holder (in the given order) any of ``rows`` overlaps."""
    cells = row_cells(rows)
    for h in holders:
        if cells & h.rows:
            return h
    return None


def deferred_row(hotkey: str, entry_ref: str, holder: Holder, rows) -> dict:
    """The public ``status/lane_b.json`` row of a deferred claim."""
    cells = row_cells(rows)
    overlap: dict[int, list[int]] = {}
    for li, r in sorted(cells & holder.rows):
        overlap.setdefault(li, []).append(r)
    return {"hotkey": str(hotkey), "entry_ref": str(entry_ref), "deferred": ROWS_HELD,
            "holder": holder.public(),
            "expected_release": {"kind": holder.kind, "era": int(holder.era)},
            "overlap": {str(k): v for k, v in overlap.items()}}


# ── sentinel targets (WB18; §12b.5 LB5, D35) ─────────────────────────────────

SENTINEL_TARGET = "sentinel_target"
SENTINEL_HOLD, SENTINEL_PAST_BOUND = "hold", "past_bound"


def sentinel_hit(declared, sentinel_target: str | None) -> bool:
    """Whether an entry declaring ``declared`` targets the era's sentinel
    suite target (``suite:<sentinel>``; ``None`` = no sentinel this era)."""
    return sentinel_target is not None and str(sentinel_target) in {str(t) for t in declared}


def sentinel_decision(eras_deferred, era_index: int, max_defer_eras: int) -> str:
    """D35 (a) for an entry on the era's sentinel, given the distinct eras it
    was already deferred in: ``hold`` (deferred this era: already counted, or
    still under the bound) or ``past_bound`` (deferred in
    ``sentinel_max_defer_eras`` other eras — it is judged on its target)."""
    seen = {int(e) for e in eras_deferred}
    if int(era_index) in seen or len(seen) < int(max_defer_eras):
        return SENTINEL_HOLD
    return SENTINEL_PAST_BOUND


def sentinel_row(hotkey: str, entry_ref: str, suite: str, eras_deferred,
                 max_defer_eras: int, era_index: int) -> dict:
    """The public ``status/lane_b.json`` row of a sentinel-deferred entry: the
    era's sentinel suite, the eras it has been deferred in and the last era
    it can be deferred in before it is judged regardless."""
    eras = sorted({int(e) for e in eras_deferred})
    left = max(0, int(max_defer_eras) - len(eras))
    return {"hotkey": str(hotkey), "entry_ref": str(entry_ref), "deferred": SENTINEL_TARGET,
            "suite": str(suite), "eras_deferred": eras,
            "sentinel_max_defer_eras": int(max_defer_eras),
            "expected_release": {"kind": "sentinel_rotation", "era": int(era_index) + 1,
                                 "latest_era": int(era_index) + left + 1}}


def has_capacity(declared, busy: dict[str, int], legs_per_domain: int) -> bool:
    """A leg holds one ``legs_per_domain`` slot on EVERY target it declared."""
    return all(busy.get(t, 0) < int(legs_per_domain) for t in declared)


def target_load(open_legs) -> dict[str, int]:
    """Open legs per target (``expert_lane.LaneBQueue.next_dispatchable``)."""
    busy: dict[str, int] = {}
    for leg in open_legs:
        for t in leg.sub.declared:
            busy[t] = busy.get(t, 0) + 1
    return busy
