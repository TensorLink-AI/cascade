"""Deterministic eval-pool builder — clean, validate, dedup, write.

This is the production-critical, network-free core. Given an iterable of
:class:`~cascade.pool.source.HarvestedSeries` (from any source), it produces a
directory in the exact layout :mod:`cascade.validator.pool` reads back:

* one ``<series_id>.npy`` per series (float32; the loader upcasts to float64),
* ``metadata.json`` mapping ``series_id -> {freq, seasonal_period, domain}``,
* ``provenance.json`` (ignored by the loader) recording how the pool was built.

Every step is pure and deterministic: same harvested inputs ⇒ byte-identical
directory ⇒ identical Hippius CID (the registry packs a sorted, zeroed-metadata
tar). That reproducibility is the audit hook — anyone can re-build and compare.

Cleaning rules (a series is **dropped**, never silently corrupted, if it fails):

* gaps (NaN/None) are linearly interpolated per channel, but a series with more
  than ``max_missing_frac`` missing is dropped (too unreliable to score on);
* it is truncated to the freshest ``context_length + horizon`` points (older
  history is unused by the scorer, so dropping it shrinks the pool for free);
* it must still be at least ``min_length`` long after that;
* series with more than ``max_channels`` variates are rejected, and degenerate
  (constant / ~zero-variance) series are dropped — a flat target makes MASE/CRPS
  uninformative;
* exact-duplicate series (same float bytes) are de-duplicated.

Cap selection (``PoolBuildConfig.selection``): ``"first"`` keeps series in
harvest order until a cap fills — the catalog order decides who gets the slots,
so a capped domain is starved of every source listed after the fill.
``"balanced"`` gathers every valid series first, then deals the slots
round-robin across SOURCES (one series per source per pass) in an order drawn
from ``selection_seed``: a capped domain holds as many distinct sources as its
caps allow (sources are the KOTH cluster-bootstrap key), and which series fill
them rotates from snapshot to snapshot. Same inputs + same seed ⇒ same pool.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..eval.seasonality import get_seasonality
from .source import DataSource, FetchJson, HarvestContext, HarvestedSeries, HttpFetcher

# cascade-pool/2 (2026-11-04): archetype-aware selection — provenance.json carries
# an ``archetypes`` block (model version, steering, catalog hash, tags used) and
# PoolBuildConfig grew ``archetype_steering`` / ``min_clusters``; a /1 snapshot
# cannot be rebuilt byte-identically by this builder, which is what the bump says.
BUILDER_VERSION = "cascade-pool/2"
SELECTION_MODES = ("first", "balanced")

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_id(series_id: str) -> str:
    s = _SAFE_ID.sub("_", series_id.strip()).strip("._-")
    return s or "series"


# Domain labels that mean "no domain" (the forge source falls back to
# "unknown" when a catalog entry carries none).
UNLABELED_DOMAINS = frozenset({"", "unknown", "none"})


class DomainFloorError(ValueError):
    """A pool build left some domain under ``min_series_per_domain``."""


class ClusterFloorError(ValueError):
    """The pool holds fewer scoring clusters than ``min_clusters`` allows."""


def cluster_count(records: Iterable[SeriesRecord]) -> int:
    """Distinct scoring clusters (``metadata.source``) in ``records``."""
    return len({_source_key(r) for r in records})


def check_cluster_floor(records: list[SeriesRecord], floor: int | None) -> None:
    """Raise :class:`ClusterFloorError` when fewer than ``floor`` distinct
    clusters remain — the breadth the KOTH bootstrap resamples over. Archetype
    steering may never buy its mix with fewer clusters (2026-11-04)."""
    if not floor:
        return
    n = cluster_count(records)
    if n < int(floor):
        raise ClusterFloorError(f"pool holds {n} scoring clusters < min_clusters={floor}")


def check_domain_floor(per_domain: dict, floor: int | None) -> None:
    """Raise :class:`DomainFloorError` naming every domain under ``floor``
    (DEC-CA-0032 Part A). ``None``/0 disables the check."""
    if not floor:
        return
    short = {d: n for d, n in sorted(per_domain.items())
             if str(d).strip().lower() not in UNLABELED_DOMAINS and n < floor}
    if short:
        raise DomainFloorError(
            f"domains under the {floor}-series floor: "
            + ", ".join(f"{d}={n}" for d, n in short.items())
            + " — add sources/feeds for them or lower --min-series-per-domain")


@dataclass(frozen=True)
class PoolBuildConfig:
    """Knobs for cleaning/validation. Defaults are sized for ``[eval]`` at
    ``context_length = 4096`` / ``horizon = 64``."""

    context_length: int = 4096
    horizon: int = 64
    # Minimum context a kept window must afford. Below context_length on purpose:
    # daily sources (e.g. web traffic) can't reach 4096 steps of history, but a
    # window with a few hundred steps of context is still a valid eval task. The
    # backbone (hourly) sources fill the full context.
    min_context: int = 256
    max_missing_frac: float = 0.2
    max_channels: int = 1
    min_std: float = 1e-8
    keep_tail: bool = True
    max_series_per_domain: int | None = None
    # Cap per (domain, freq) CELL — the granularity-aware balance knob: an
    # hourly flood inside a domain can't crowd out that domain's daily series.
    max_series_per_domain_freq: int | None = None
    max_series_total: int | None = None
    # Require an upstream feed id on every kept series (DEC-CA-0026 item 3):
    # ``source`` is the KOTH cluster-bootstrap key, and multivariate windows
    # must never enter a pool without it (their channels would otherwise lean
    # on the series_id fallback alone). Off by default — existing sources that
    # predate source stamping keep building — but flip it on before any pool
    # carries multichannel windows.
    require_source: bool = False
    # Producer-only mode; recorded source keys keep archived verdicts unchanged.
    cluster_by: str = "source"
    # v2 primary pool (DEC-CA-0050 P2): every kept series must carry a real
    # domain label — lane B judges per domain, so an "unknown"/empty label is
    # a window no domain verdict can use. Off by default (legacy pools keep
    # building byte-identically).
    require_domain: bool = False
    # DEC-CA-0032 Part A capacity floor: refuse to write a pool in which any
    # labeled domain holds fewer than this many series (each series is one
    # window candidate per round). None = no floor (legacy behaviour).
    min_series_per_domain: int | None = None
    # Which series win a capped slot: "first" (harvest order, the historical
    # behaviour) or "balanced" (seeded round-robin across sources — see the
    # module docstring). ``selection_seed`` is recorded in provenance.json so
    # the choice is reproducible from the published snapshot alone.
    selection: str = "first"
    selection_seed: str = ""
    # Archetype steering (cascade.pool.archetypes): 0 = off (the unsteered
    # pool, byte-identical in selection), 1 = match the benchmark reference
    # shares as far as supply and the caps allow. Needs a model file; with
    # none the value is recorded but nothing steers.
    archetype_steering: float = 0.0
    # Scoring-breadth floor: refuse a pool with fewer distinct clusters
    # (source families) than this. None = no floor.
    min_clusters: int | None = None

    @property
    def min_length(self) -> int:
        return self.horizon + self.min_context

    @property
    def keep_length(self) -> int:
        return self.context_length + self.horizon


@dataclass(frozen=True)
class SeriesRecord:
    """A cleaned, validated series ready to write."""

    series_id: str
    values: np.ndarray  # float32, (L,) univariate or (C, L)
    metadata: dict
    domain: str
    content_hash: str


@dataclass
class BuildSummary:
    """Outcome of a pool build (printed by the CLI, embedded in provenance)."""

    out_dir: str
    as_of: str
    context_length: int
    horizon: int
    n_series: int = 0
    n_points: int = 0
    per_domain: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"eval pool built at {self.out_dir}",
            f"  as_of={self.as_of} context_length={self.context_length} horizon={self.horizon}",
            f"  series={self.n_series} points={self.n_points:,}",
        ]
        if self.per_domain:
            by_dom = ", ".join(f"{k}={v}" for k, v in sorted(self.per_domain.items()))
            lines.append(f"  by domain: {by_dom}")
        if self.dropped:
            dr = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items()))
            lines.append(f"  dropped: {dr}")
        return "\n".join(lines)


def _to_2d_float(values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[-1] < 2:
        return None
    return arr


def _interp_channel(ch: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Linear-interpolate non-finite entries; constant-extrapolate the edges.

    Returns ``(filled, missing_frac)``, or ``(None, 1.0)`` if the whole channel
    is non-finite (nothing to interpolate from)."""
    finite = np.isfinite(ch)
    if finite.all():
        return ch, 0.0
    if not finite.any():
        return None, 1.0
    idx = np.arange(len(ch))
    out = ch.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], ch[finite])
    return out, float(1.0 - finite.mean())


def prepare_series(
    hs: HarvestedSeries, cfg: PoolBuildConfig
) -> tuple[SeriesRecord | None, str | None]:
    """Clean and validate one harvested series.

    Returns ``(record, None)`` on success or ``(None, reason)`` if dropped, where
    ``reason`` is a stable string for the provenance drop counters.
    """
    arr = _to_2d_float(hs.values)
    if arr is None:
        return None, "bad_shape"
    family = hs.attrs.get("source_family") if hs.attrs else None
    if cfg.cluster_by not in ("source", "source_family"):
        raise ValueError("cluster_by must be source or source_family")
    if cfg.cluster_by == "source_family" and (not isinstance(family, str)
                                               or not family or family.startswith("unresolved:")):
        return None, "missing_source_family"
    cluster = family if cfg.cluster_by == "source_family" else hs.source
    if cfg.require_source and not cluster:
        return None, "missing_source"
    if cfg.require_domain and str(hs.domain or "").strip().lower() in UNLABELED_DOMAINS:
        return None, "missing_domain"
    n_channels, length = arr.shape
    if n_channels > cfg.max_channels:
        return None, "too_many_channels"

    cleaned = np.empty_like(arr)
    worst_missing = 0.0
    for c in range(n_channels):
        filled, missing = _interp_channel(arr[c])
        if filled is None:
            return None, "all_missing"
        cleaned[c] = filled
        worst_missing = max(worst_missing, missing)
    if worst_missing > cfg.max_missing_frac:
        return None, "too_much_missing"

    if cfg.keep_tail and length > cfg.keep_length:
        cleaned = cleaned[:, -cfg.keep_length :]
        length = cleaned.shape[-1]
    if length < cfg.min_length:
        return None, "too_short"

    for c in range(n_channels):
        if np.ptp(cleaned[c]) == 0.0 or float(np.std(cleaned[c])) < cfg.min_std:
            return None, "degenerate"

    seasonal = (
        int(hs.seasonal_period)
        if hs.seasonal_period is not None
        else get_seasonality(hs.freq)
    )
    metadata = {"freq": hs.freq, "seasonal_period": seasonal, "domain": hs.domain}
    if cluster:
        metadata["source"] = cluster
    if cfg.cluster_by == "source_family":
        metadata["source_family"] = family
        if hs.source:
            metadata["catalog_source"] = hs.source
    # DGP class label (tsbench-forge catalog) — drives the jittered mix's
    # class-rotation tier (cascade.validator.windows). Absent for sources
    # without one; the sampler degrades to domain-level jitter only.
    if hs.attrs and hs.attrs.get("dgp_class"):
        metadata["dgp_class"] = str(hs.attrs["dgp_class"])
    # Statistical archetype (tsbench-forge catalog ``stat_archetype``): the
    # bucket archetype steering balances; absent = the "unknown" bucket.
    if hs.attrs and hs.attrs.get("stat_archetype"):
        metadata["stat_archetype"] = str(hs.attrs["stat_archetype"])

    values = cleaned.astype(np.float32)
    if values.shape[0] == 1:  # store univariate as 1-D; loader promotes to (1, L)
        values = values[0]
    content_hash = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    return (
        SeriesRecord(
            series_id=_sanitize_id(hs.series_id),
            values=values,
            metadata=metadata,
            domain=hs.domain,
            content_hash=content_hash,
        ),
        None,
    )


def collect_records(
    sources: Iterable[DataSource],
    ctx: HarvestContext,
    cfg: PoolBuildConfig,
    fetch: FetchJson,
    archetypes=None,
) -> tuple[list[SeriesRecord], Counter]:
    """Harvest from every source, clean/validate, dedup, and apply caps.

    De-duplicates by content hash across all sources, enforces per-domain and
    total caps, and disambiguates any sanitised-id collision with a numeric
    suffix so every record maps to a unique filename / metadata key.
    """
    if cfg.selection not in SELECTION_MODES:
        raise ValueError(f"selection must be one of {SELECTION_MODES}; got {cfg.selection!r}")
    if cfg.selection == "balanced":
        return _collect_balanced(sources, ctx, cfg, fetch, archetypes=archetypes)

    records: list[SeriesRecord] = []
    drops: Counter = Counter()
    seen_hash: set[str] = set()
    used_counts: dict[str, int] = {}
    assigned: set[str] = set()
    per_domain: Counter = Counter()
    per_cell: Counter = Counter()  # (domain, freq) — granularity-aware cap

    for src in sources:
        for hs in src.harvest(fetch, ctx):
            if cfg.max_series_total is not None and len(records) >= cfg.max_series_total:
                drops["total_cap"] += 1
                continue
            rec, reason = prepare_series(hs, cfg)
            if rec is None:
                drops[reason or "unknown"] += 1
                continue
            if rec.content_hash in seen_hash:
                drops["duplicate"] += 1
                continue
            if (
                cfg.max_series_per_domain is not None
                and per_domain[rec.domain] >= cfg.max_series_per_domain
            ):
                drops["domain_cap"] += 1
                continue
            cell = (rec.domain, rec.metadata.get("freq"))
            if (
                cfg.max_series_per_domain_freq is not None
                and per_cell[cell] >= cfg.max_series_per_domain_freq
            ):
                drops["domain_freq_cap"] += 1
                continue

            # Disambiguate any sanitised-id collision so every record maps to a
            # unique filename / metadata key.
            base = rec.series_id
            count = used_counts.get(base, 0)
            sid = base if count == 0 else f"{base}-{count + 1}"
            while sid in assigned:
                count += 1
                sid = f"{base}-{count + 1}"
            used_counts[base] = count + 1
            assigned.add(sid)
            if sid != base:
                rec = SeriesRecord(sid, rec.values, rec.metadata, rec.domain, rec.content_hash)

            seen_hash.add(rec.content_hash)
            per_domain[rec.domain] += 1
            per_cell[cell] += 1
            records.append(rec)

    records.sort(key=lambda r: r.series_id)  # deterministic on-disk order
    return records, drops


def _source_key(rec: SeriesRecord) -> str:
    """The cluster a record belongs to — the same key the KOTH bootstrap
    resamples on (``source``, falling back to the series id)."""
    return str(rec.metadata.get("source") or rec.series_id)


def seeded_rng(seed: str, *salt: str) -> np.random.Generator:
    """A generator fixed by ``seed`` (+ optional salt) — stable across
    platforms and Python hash randomisation."""
    h = hashlib.sha256("\x1f".join((seed, *salt)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _collect_balanced(
    sources: Iterable[DataSource],
    ctx: HarvestContext,
    cfg: PoolBuildConfig,
    fetch: FetchJson,
    archetypes=None,
) -> tuple[list[SeriesRecord], Counter]:
    """``selection = "balanced"``: clean + dedup EVERYTHING, then deal capped
    slots round-robin across sources in a seeded order.

    Candidates are put in a canonical order first (sources by key, series by
    content hash), so the harvest order has no say — only the seed does.
    """
    by_source, drops = gather_candidates(sources, ctx, cfg, fetch)
    return select_balanced(by_source, cfg, drops, archetypes=archetypes)


def gather_candidates(
    sources: Iterable[DataSource],
    ctx: HarvestContext,
    cfg: PoolBuildConfig,
    fetch: FetchJson,
) -> tuple[dict[str, list[SeriesRecord]], Counter]:
    """Harvest + clean + dedup every candidate, grouped by cluster key. The
    expensive half of a balanced build; :func:`select_balanced` can then be
    run several times (a steering sweep) over the same candidates."""
    drops: Counter = Counter()
    seen_hash: set[str] = set()
    by_source: dict[str, list[SeriesRecord]] = {}
    for src in sources:
        for hs in src.harvest(fetch, ctx):
            rec, reason = prepare_series(hs, cfg)
            if rec is None:
                drops[reason or "unknown"] += 1
                continue
            if rec.content_hash in seen_hash:
                drops["duplicate"] += 1
                continue
            seen_hash.add(rec.content_hash)
            by_source.setdefault(_source_key(rec), []).append(rec)
    return by_source, drops


def eligible_archetype_counts(by_source: dict[str, list[SeriesRecord]]) -> Counter:
    """Supply per archetype bucket among the deduplicated candidates."""
    from .archetypes import archetype_of

    return Counter(archetype_of(r.metadata) for recs in by_source.values() for r in recs)


def select_balanced(
    by_source: dict[str, list[SeriesRecord]],
    cfg: PoolBuildConfig,
    drops: Counter | None = None,
    *,
    archetypes=None,
) -> tuple[list[SeriesRecord], Counter]:
    """Deal the capped slots round-robin across clusters in the seeded order.

    With ``cfg.archetype_steering > 0`` and an :class:`ArchetypeModel`, each
    pass serves first the clusters whose head series' archetype is furthest
    below its target share (``steering_targets``); the pass still takes at
    most one series per cluster, so rotation, caps and floors are unchanged.
    At 0 the pass order is the seeded order exactly (byte-identical pools).
    """
    from .archetypes import archetype_of, steering_targets

    drops = Counter() if drops is None else drops
    steer = float(cfg.archetype_steering or 0.0) if archetypes is not None else 0.0
    targets = (steering_targets(eligible_archetype_counts(by_source), archetypes, steer)
               if steer > 0 else None)
    rng = seeded_rng(cfg.selection_seed, "sources")
    keys = sorted(by_source)
    order = [keys[i] for i in rng.permutation(len(keys))]
    queues: dict[str, list[SeriesRecord]] = {}
    for k in order:
        recs = sorted(by_source[k], key=lambda r: r.content_hash)
        perm = seeded_rng(cfg.selection_seed, "series", k).permutation(len(recs))
        queues[k] = [recs[i] for i in perm]

    chosen: list[SeriesRecord] = []
    per_domain: Counter = Counter()
    per_cell: Counter = Counter()
    per_arch: Counter = Counter()
    progressed = True
    while progressed:
        progressed = False
        pass_order = order
        if targets is not None:
            # deficit of a cluster = how far its head series' archetype sits
            # under its target share right now (on the first pass: the target
            # itself); largest deficit served first, ties (and the unsteered
            # case) keep the seeded order
            n = max(1, len(chosen))
            deficit = {}
            for k in order:
                if queues[k]:
                    a = archetype_of(queues[k][0].metadata)
                    deficit[k] = targets.get(a, 0.0) - per_arch[a] / n
            pass_order = sorted((k for k in order if queues[k]),
                                key=lambda k: -deficit[k])
        for k in pass_order:
            q = queues[k]
            if not q:
                continue
            rec = q[0]
            if cfg.max_series_total is not None and len(chosen) >= cfg.max_series_total:
                drops["total_cap"] += len(q)
                q.clear()
                continue
            if (cfg.max_series_per_domain is not None
                    and per_domain[rec.domain] >= cfg.max_series_per_domain):
                drops["domain_cap"] += 1
                q.pop(0)
                progressed = True
                continue
            cell = (rec.domain, rec.metadata.get("freq"))
            if (cfg.max_series_per_domain_freq is not None
                    and per_cell[cell] >= cfg.max_series_per_domain_freq):
                drops["domain_freq_cap"] += 1
                q.pop(0)
                progressed = True
                continue
            q.pop(0)
            chosen.append(rec)
            per_domain[rec.domain] += 1
            per_cell[cell] += 1
            per_arch[archetype_of(rec.metadata)] += 1
            progressed = True

    # Id disambiguation over a canonical order, so it is seed- and
    # harvest-order-independent for any given chosen set.
    chosen.sort(key=lambda r: (r.series_id, r.content_hash))
    used_counts: dict[str, int] = {}
    assigned: set[str] = set()
    records: list[SeriesRecord] = []
    for rec in chosen:
        base = rec.series_id
        count = used_counts.get(base, 0)
        sid = base if count == 0 else f"{base}-{count + 1}"
        while sid in assigned:
            count += 1
            sid = f"{base}-{count + 1}"
        used_counts[base] = count + 1
        assigned.add(sid)
        if sid != base:
            rec = SeriesRecord(sid, rec.values, rec.metadata, rec.domain, rec.content_hash)
        records.append(rec)
    records.sort(key=lambda r: r.series_id)
    return records, drops


def write_pool(
    records: list[SeriesRecord],
    out_dir: Path | str,
    *,
    as_of: str,
    cfg: PoolBuildConfig,
    drops: Counter | None = None,
    overwrite: bool = False,
    archetypes=None,
    archetype_report: dict | None = None,
    catalog_sha256: str = "",
) -> BuildSummary:
    """Write records to ``out_dir`` in the validator-pool layout.

    Raises if ``out_dir`` already holds ``.npy`` series unless ``overwrite`` is
    set (a re-build should be explicit, not an accidental merge with a stale pool).
    """
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.glob("*.npy"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{dest} already contains {len(existing)} .npy series; pass overwrite=True to replace"
        )
    for p in existing:
        p.unlink()

    if not records:
        raise ValueError("refusing to write an empty pool (no series survived validation)")

    metadata: dict[str, dict] = {}
    per_domain: Counter = Counter()
    n_points = 0
    for rec in records:
        np.save(dest / f"{rec.series_id}.npy", rec.values, allow_pickle=False)
        metadata[rec.series_id] = rec.metadata
        per_domain[rec.domain] += 1
        n_points += int(rec.values.shape[-1])

    (dest / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = BuildSummary(
        out_dir=str(dest),
        as_of=as_of,
        context_length=cfg.context_length,
        horizon=cfg.horizon,
        n_series=len(records),
        n_points=n_points,
        per_domain=dict(per_domain),
        dropped=dict(drops or {}),
    )
    from .archetypes import TAG_KEY, UNKNOWN, archetype_of

    # Archetype provenance (cascade-pool/2): the model version and file hash,
    # the catalog hash and the tag of every chosen cluster AS USED — the forge
    # re-assigns tags from fresh data, so the snapshot must say which tags it
    # was built from, not just which model.
    tags_used: dict[str, str] = {}
    for rec in records:
        tag = str(rec.metadata.get(TAG_KEY) or "")
        if tag:
            tags_used[_source_key(rec)] = tag
    arch_counts = Counter(archetype_of(r.metadata) for r in records)
    provenance = {
        "builder_version": BUILDER_VERSION,
        "config": asdict(cfg),
        "summary": asdict(summary),
        "archetypes": {
            "model_version": getattr(archetypes, "version", None),
            "model_sha256": getattr(archetypes, "sha256", "") or None,
            "steering": float(cfg.archetype_steering or 0.0),
            "catalog_sha256": catalog_sha256 or None,
            "n_clusters": cluster_count(records),
            "by_archetype": dict(sorted(arch_counts.items())),
            "unknown_series": int(arch_counts.get(UNKNOWN, 0)),
            "tags_used": dict(sorted(tags_used.items())),
            "report": archetype_report,
        },
    }
    (dest / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def build_pool(
    sources: Iterable[DataSource],
    out_dir: Path | str,
    ctx: HarvestContext,
    cfg: PoolBuildConfig,
    *,
    fetch: FetchJson | None = None,
    overwrite: bool = False,
    archetypes=None,
    catalog_sha256: str = "",
) -> BuildSummary:
    """Harvest → clean/validate → dedup → write. The one-call entry point."""
    fetch = fetch or HttpFetcher()
    report = None
    if cfg.selection == "balanced":
        # gather once; the report needs the eligible (pre-cap) supply too
        from .archetypes import archetype_of, archetype_report

        by_source, drops = gather_candidates(sources, ctx, cfg, fetch)
        records, drops = select_balanced(by_source, cfg, drops, archetypes=archetypes)
        report = archetype_report(Counter(archetype_of(r.metadata) for r in records),
                                  eligible_archetype_counts(by_source), archetypes,
                                  steering=float(cfg.archetype_steering or 0.0),
                                  n_sources=cluster_count(records), n_series=len(records))
    else:
        records, drops = collect_records(sources, ctx, cfg, fetch)
    from collections import Counter as _Counter

    check_domain_floor(_Counter(r.domain for r in records), cfg.min_series_per_domain)
    check_cluster_floor(records, cfg.min_clusters)
    return write_pool(
        records, out_dir, as_of=ctx.as_of.isoformat(), cfg=cfg, drops=drops, overwrite=overwrite,
        archetypes=archetypes, archetype_report=report, catalog_sha256=catalog_sha256,
    )


def steering_sweep(
    sources: Iterable[DataSource],
    ctx: HarvestContext,
    cfg: PoolBuildConfig,
    *,
    archetypes,
    steerings: Iterable[float] = (0.0, 0.5, 1.0),
    fetch: FetchJson | None = None,
) -> list[dict]:
    """DRY RUN: gather the candidates once, select at each steering value and
    return one :func:`archetype_report` per value (plus per-domain counts,
    drops and the cluster count). Nothing is written. The "before" numbers
    (S = 0) come from the same code and the same candidates as the "after"."""
    from dataclasses import replace

    from .archetypes import archetype_of, archetype_report

    fetch = fetch or HttpFetcher()
    base = replace(cfg, selection="balanced")
    by_source, drops0 = gather_candidates(sources, ctx, base, fetch)
    supply = eligible_archetype_counts(by_source)
    out: list[dict] = []
    for s in steerings:
        c = replace(base, archetype_steering=float(s))
        queues = {k: list(v) for k, v in by_source.items()}
        records, drops = select_balanced(queues, c, Counter(drops0), archetypes=archetypes)
        rep = archetype_report(Counter(archetype_of(r.metadata) for r in records), supply,
                               archetypes, steering=float(s), n_sources=cluster_count(records),
                               n_series=len(records))
        rep["per_domain"] = dict(sorted(Counter(r.domain for r in records).items()))
        rep["dropped"] = dict(sorted(drops.items()))
        rep["series_ids"] = [r.series_id for r in records]
        out.append(rep)
    return out
