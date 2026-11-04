"""Archetype-aware pool selection (2026-11-04, with the tsbench-forge team).

The forge catalog tags every entry with a statistical archetype
(``stat_archetype``, twelve classes, re-assigned from fresh data on every
sync; entries without enough history are untagged) and ships a model file,
``archetypes.json``, next to the catalog with the benchmark reference shares
of each archetype. The balanced picker can STEER the pool's archetype mix
toward those shares — ``archetype_steering`` S in [0, 1]:

    target_a = S * ref_share_a + (1 - S) * natural_share_a

``natural_share`` is the archetype mix of the eligible candidates, so S = 0
reproduces the unsteered pool exactly and S = 1 matches the benchmark mix as
far as supply and the existing caps allow. Untagged series form their own
``unknown`` bucket at its natural share — steering never trades them away
and never invents a reference share for them.

Steering changes only WHICH source family is served next inside a pass of
the round-robin (families whose head series' archetype is furthest under its
target go first). One series per family per pass, the per-domain and
per-(domain, freq) caps, the capacity floor and the seeded rotation are all
untouched, and the source family stays the scoring (bootstrap) key.

Overlap (the forge team's ``stat_clusters.alignment``)::

    overlap = sum_a min(pool_share_a, ref_share_a)

over the tagged archetypes, pool shares counted in SERIES over tagged series
only; the unknown fraction is reported next to it. ``ref_share_by_bench``
gives the same number per public benchmark (GIFT-Eval, BOOM, TIME) — the
problem being fixed shows up per benchmark.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

UNKNOWN = "unknown"
ARCHETYPES_FILE = "archetypes.json"
TAG_KEY = "stat_archetype"


class ArchetypeModelError(ValueError):
    """``archetypes.json`` is present but unusable."""


@dataclass(frozen=True)
class ArchetypeModel:
    version: str
    ref_share: dict[str, float]                       # ref_share_daily_or_faster, sums to 1
    ref_share_by_bench: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = ""                                  # where it was read from
    sha256: str = ""

    @property
    def archetypes(self) -> tuple[str, ...]:
        return tuple(sorted(self.ref_share))


def _normalise(shares: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (shares or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError) as e:
            raise ArchetypeModelError(f"share of {k!r} is not a number: {v!r}") from e
        if f < 0:
            raise ArchetypeModelError(f"share of {k!r} is negative: {f}")
        if str(k).lower() == UNKNOWN:
            continue                   # never a reference share for the untagged
        out[str(k)] = f
    total = sum(out.values())
    if total <= 0:
        raise ArchetypeModelError("reference shares sum to zero")
    return {k: v / total for k, v in sorted(out.items())}


def parse_archetype_model(doc: dict, *, source: str = "", sha256: str = "") -> ArchetypeModel:
    """Tolerant reader of the forge model file.

    Flat form::

        {"model_version": "...", "ref_share_daily_or_faster": {"A01": 0.1, ...},
         "ref_share_by_bench": {"gift": {...}, "boom": {...}, "time": {...}}}

    or a list form, ``{"archetypes": [{"id": "A01", "ref_share_daily_or_faster":
    0.1, "ref_share_by_bench": {"gift": 0.1, ...}}, ...]}``.
    """
    if not isinstance(doc, dict):
        raise ArchetypeModelError("archetypes.json must be a JSON object")
    version = str(doc.get("model_version") or doc.get("version") or doc.get("archetype_model_version") or "")
    if not version:
        raise ArchetypeModelError("archetypes.json carries no model_version")
    ref = doc.get("ref_share_daily_or_faster")
    by_bench_raw = doc.get("ref_share_by_bench") or {}
    if ref is None and isinstance(doc.get("archetypes"), list):
        ref, by_bench_raw = {}, {}
        for row in doc["archetypes"]:
            if not isinstance(row, dict):
                continue
            aid = str(row.get("id") or row.get("archetype") or "")
            if not aid:
                continue
            ref[aid] = row.get("ref_share_daily_or_faster", row.get("ref_share", 0.0))
            for bench, share in (row.get("ref_share_by_bench") or {}).items():
                by_bench_raw.setdefault(str(bench), {})[aid] = share
    if not isinstance(ref, dict) or not ref:
        raise ArchetypeModelError("archetypes.json carries no ref_share_daily_or_faster")
    by_bench = {str(b): _normalise(s) for b, s in by_bench_raw.items() if isinstance(s, dict) and s}
    return ArchetypeModel(version=version, ref_share=_normalise(ref), ref_share_by_bench=by_bench,
                          source=source, sha256=sha256)


def load_archetype_model(path: Path | str) -> ArchetypeModel | None:
    """The model at ``path``, or None when the file is absent (steering then
    has nothing to steer toward and the build is the unsteered one)."""
    p = Path(path)
    if not p.is_file():
        return None
    raw = p.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ArchetypeModelError(f"{p}: not valid JSON ({e})") from e
    return parse_archetype_model(doc, source=str(p), sha256=hashlib.sha256(raw).hexdigest())


def archetype_of(metadata: dict | None) -> str:
    """The archetype bucket of one series' metadata (``unknown`` when untagged)."""
    tag = str((metadata or {}).get(TAG_KEY) or "").strip()
    return tag or UNKNOWN


def shares(counts: Counter | dict, *, tagged_only: bool = True) -> dict[str, float]:
    """Counts → shares. ``tagged_only`` leaves the unknown bucket out of the
    denominator (the overlap convention)."""
    items = {k: int(v) for k, v in counts.items() if v and (not tagged_only or k != UNKNOWN)}
    total = sum(items.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted(items.items())}


def overlap(pool_counts: Counter | dict, ref_share: dict[str, float]) -> float:
    """``sum_a min(pool_share_a, ref_share_a)`` over the reference archetypes;
    pool shares in series over TAGGED series only. 0 when nothing is tagged."""
    ps = shares(pool_counts, tagged_only=True)
    if not ps:
        return 0.0
    return round(sum(min(ps.get(a, 0.0), r) for a, r in ref_share.items()), 6)


def unknown_fraction(counts: Counter | dict) -> float:
    total = sum(int(v) for v in counts.values())
    return round(int(counts.get(UNKNOWN, 0)) / total, 6) if total else 0.0


def steering_targets(natural_counts: Counter | dict, model: ArchetypeModel, steering: float) -> dict[str, float]:
    """Target share per bucket (unknown included): ``S * ref + (1 - S) *
    natural`` over the tagged buckets, scaled to the tagged mass, with the
    unknown bucket kept at its natural share."""
    s = min(1.0, max(0.0, float(steering)))
    nat_all = shares(natural_counts, tagged_only=False)
    unk = nat_all.get(UNKNOWN, 0.0)
    nat_tagged = shares(natural_counts, tagged_only=True)
    buckets = set(nat_tagged) | set(model.ref_share)
    targets: dict[str, float] = {}
    for a in buckets:
        mixed = s * model.ref_share.get(a, 0.0) + (1.0 - s) * nat_tagged.get(a, 0.0)
        targets[a] = mixed * (1.0 - unk)
    if unk > 0:
        targets[UNKNOWN] = unk
    return targets


def archetype_report(chosen_counts: Counter | dict, eligible_counts: Counter | dict,
                     model: ArchetypeModel | None, *, steering: float, n_sources: int,
                     n_series: int) -> dict:
    """The dry-run / provenance report: mix before (eligible) and after
    (chosen), overlaps, unknown fraction, supply per archetype."""
    chosen = Counter({k: int(v) for k, v in chosen_counts.items()})
    eligible = Counter({k: int(v) for k, v in eligible_counts.items()})
    rep: dict = {
        "steering": float(steering),
        "model_version": model.version if model else None,
        "n_series": int(n_series),
        "n_sources": int(n_sources),
        "unknown_fraction": unknown_fraction(chosen),
        "unknown_fraction_eligible": unknown_fraction(eligible),
        "chosen_by_archetype": dict(sorted(chosen.items())),
        "supply_by_archetype": dict(sorted(eligible.items())),
        "pool_share": shares(chosen),
        "eligible_share": shares(eligible),
    }
    if model is not None:
        rep["ref_share"] = dict(model.ref_share)
        rep["overlap_daily_or_faster"] = overlap(chosen, model.ref_share)
        rep["overlap_daily_or_faster_eligible"] = overlap(eligible, model.ref_share)
        rep["overlap_by_bench"] = {b: overlap(chosen, r) for b, r in sorted(model.ref_share_by_bench.items())}
        rep["overlap_by_bench_eligible"] = {b: overlap(eligible, r)
                                           for b, r in sorted(model.ref_share_by_bench.items())}
        # how far supply limits each archetype at this pool size: series the
        # target asks for vs series that exist
        n_tagged = sum(v for k, v in chosen.items() if k != UNKNOWN)
        rep["target_series_by_archetype"] = {
            a: round(r * n_tagged) for a, r in model.ref_share.items()}
        rep["supply_limited"] = sorted(a for a, want in rep["target_series_by_archetype"].items()
                                       if eligible.get(a, 0) < want)
    return rep


def render_report(rep: dict) -> str:
    lines = [f"archetype steering S={rep['steering']:.2f}  model={rep.get('model_version') or '-'}  "
             f"series={rep['n_series']} sources={rep['n_sources']}  "
             f"unknown={rep['unknown_fraction']:.1%} (eligible {rep['unknown_fraction_eligible']:.1%})"]
    if "overlap_daily_or_faster" in rep:
        lines.append(f"  overlap daily-or-faster: {rep['overlap_daily_or_faster']:.3f} "
                     f"(eligible pool {rep['overlap_daily_or_faster_eligible']:.3f})")
        for b, v in rep.get("overlap_by_bench", {}).items():
            lines.append(f"  overlap {b}: {v:.3f} (eligible {rep['overlap_by_bench_eligible'][b]:.3f})")
    keys = sorted(set(rep["supply_by_archetype"]) | set(rep["chosen_by_archetype"]) | set(rep.get("ref_share", {})))
    lines.append("  archetype  chosen  share   ref     target  supply")
    for a in keys:
        lines.append(f"  {a:<9}  {rep['chosen_by_archetype'].get(a, 0):>6}  "
                     f"{rep['pool_share'].get(a, 0.0):6.3f}  {rep.get('ref_share', {}).get(a, 0.0):6.3f}  "
                     f"{rep.get('target_series_by_archetype', {}).get(a, 0):>6}  "
                     f"{rep['supply_by_archetype'].get(a, 0):>6}")
    if rep.get("supply_limited"):
        lines.append("  supply-limited: " + ", ".join(rep["supply_limited"]))
    return "\n".join(lines)
