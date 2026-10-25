"""Pair two checkpoints' public-benchmark item rows into the lane-B guard input
(DEC-CA-0050 P5).

:func:`cascade.eval.expert_lane.judge_expert_leg` takes
``public={"gift_crps": (king_items, chal_items), …}`` — paired per-item scores
that :func:`~cascade.eval.expert_lane.paired_geomean_lcb` bootstraps. This
module builds that mapping from two :func:`cascade.eval.benchmarks.
item_rows_from_report` results (king, candidate):

* inner join on the item id, in sorted id order (deterministic — every
  validator/audit replay builds the same arrays from the same rows);
* an item is dropped from a metric when either side's value is missing,
  non-finite or ≤ 0 (the bootstrap works in log space), or when its id
  repeats within one side's rows (ambiguous — never guess which row counts);
  every drop is counted per guard key;
* a benchmark whose suite is not ``"ok"`` on BOTH sides is reported in
  ``missing`` and gets NO key. ``judge_expert_leg`` treats an absent key as
  unguarded, so a caller that requires a benchmark must check
  :attr:`PairedPublic.missing` (``require=`` below does it for you).

Pure stdlib + math, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

METRICS = ("crps", "mase")


class MissingPublicGuard(ValueError):
    """A required public benchmark could not be paired."""


@dataclass
class PairedPublic:
    public: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    items: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _usable(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)) and float(v) > 0)


def _index(rows: Iterable[Mapping]) -> tuple[dict[str, Mapping], set[str]]:
    by_id: dict[str, Mapping] = {}
    dup: set[str] = set()
    for r in rows:
        item = r.get("item")
        if not isinstance(item, str):
            continue
        if item in by_id:
            dup.add(item)
        by_id[item] = r
    return by_id, dup


def pair_public_rows(
    king: Mapping[str, Mapping] | None,
    chal: Mapping[str, Mapping] | None,
    *,
    benchmarks: Iterable[str] | None = None,
    require: Iterable[str] = (),
    min_items: int = 1,
) -> PairedPublic:
    """Build the paired ``public`` mapping from two item-row results.

    ``benchmarks`` restricts which guard keys are built (default: every
    benchmark present on either side). ``require`` lists benchmark keys that
    MUST pair with at least ``min_items`` usable values per metric; missing
    or insufficient evidence raises :class:`MissingPublicGuard`.
    """
    king = king or {}
    chal = chal or {}
    require = tuple(require)
    names = sorted(set(benchmarks) if benchmarks is not None else set(king) | set(chal))
    out = PairedPublic()
    for b in names:
        k, c = king.get(b), chal.get(b)
        if not k or not c or k.get("status") != "ok" or c.get("status") != "ok":
            out.missing.append(b)
            continue
        # Preserve legacy unpinned evidence, but never compare known different
        # datasets or silently drop the pin from one side of a comparison.
        kr, cr = k.get("data_revision"), c.get("data_revision")
        if (kr or cr) and kr != cr:
            out.missing.append(b)
            continue
        k_by, k_dup = _index(k.get("rows") or [])
        c_by, c_dup = _index(c.get("rows") or [])
        union = set(k_by) | set(c_by)
        for m in METRICS:
            key = f"{b}_{m}"
            ids, kv, cv = [], [], []
            for item in sorted(union):
                if item in k_dup or item in c_dup or item not in k_by or item not in c_by:
                    continue
                a, z = k_by[item].get(m), c_by[item].get(m)
                if _usable(a) and _usable(z):
                    ids.append(item)
                    kv.append(float(a))
                    cv.append(float(z))
            out.public[key] = (kv, cv)
            out.items[key] = ids
            out.dropped[key] = len(union) - len(ids)
    missing_required = sorted(set(require) & set(out.missing)
                              | (set(require) - set(names)))
    missing_required += [b for b in require if b not in missing_required and any(
        len(out.items.get(f"{b}_{m}", [])) < min_items for m in METRICS)]
    if missing_required:
        raise MissingPublicGuard(f"required public benchmarks not pairable: {missing_required}")
    return out
