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

BUILDER_VERSION = "cascade-pool/1"

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_id(series_id: str) -> str:
    s = _SAFE_ID.sub("_", series_id.strip()).strip("._-")
    return s or "series"


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
    max_series_total: int | None = None

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
) -> tuple[list[SeriesRecord], Counter]:
    """Harvest from every source, clean/validate, dedup, and apply caps.

    De-duplicates by content hash across all sources, enforces per-domain and
    total caps, and disambiguates any sanitised-id collision with a numeric
    suffix so every record maps to a unique filename / metadata key.
    """
    records: list[SeriesRecord] = []
    drops: Counter = Counter()
    seen_hash: set[str] = set()
    used_counts: dict[str, int] = {}
    assigned: set[str] = set()
    per_domain: Counter = Counter()

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
            records.append(rec)

    records.sort(key=lambda r: r.series_id)  # deterministic on-disk order
    return records, drops


def write_pool(
    records: list[SeriesRecord],
    out_dir: Path | str,
    *,
    as_of: str,
    cfg: PoolBuildConfig,
    drops: Counter | None = None,
    overwrite: bool = False,
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
    provenance = {
        "builder_version": BUILDER_VERSION,
        "config": asdict(cfg),
        "summary": asdict(summary),
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
) -> BuildSummary:
    """Harvest → clean/validate → dedup → write. The one-call entry point."""
    fetch = fetch or HttpFetcher()
    records, drops = collect_records(sources, ctx, cfg, fetch)
    return write_pool(
        records, out_dir, as_of=ctx.as_of.isoformat(), cfg=cfg, drops=drops, overwrite=overwrite
    )
