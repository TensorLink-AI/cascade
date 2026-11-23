"""Lane D ``pool_d`` selection: ``pool``'s selected series, joined (plan W6a).

``pool_d`` (LANE_D_PLAN §1.7, D12) carries **the same selected series as
lane A's ``pool``**, joined, never re-selected. This module runs lane A's
own :func:`~cascade.pool.builder.collect_records` unchanged over one harvest
and maps every record it kept to a stable join key::

    key = (source, pre-suffix source series key, channel)
        = (str(hs.source), builder._sanitize_id(hs.series_id), channel)

Why not lane A's emitted id or its content hash: ``collect_records`` assigns
collision suffixes ``-2``, ``-3``… in sort order, so an emitted id moves when
another colliding series appears or vanishes, and ``content_hash`` is the
sha256 of the truncated values, so it changes whenever the tail moves. The
emitted id and the content hash are used here only *inside one builder run*,
to find which harvested item produced each lane A record; they are never the
join key and never compared across snapshots.

Lane A is untouched: :func:`build_pool_with_selection` writes the exact bytes
:func:`~cascade.pool.builder.build_pool` writes for the same inputs (it calls
the same ``collect_records``, ``check_domain_floor`` and ``write_pool``).

A lane A record is refused from ``pool_d`` (logged, typed) when its key is
not determined: ``key_collision`` (two harvested series of ONE source share
``(source, base)``), ``no_source`` (no upstream feed id), ``ambiguous_record``
(lane A de-duplicated harvested series with different keys into it) or
``id_collision`` (two keys map to one ``pool_d`` series id).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .builder import (
    BuildSummary,
    PoolBuildConfig,
    SeriesRecord,
    _sanitize_id,
    check_domain_floor,
    collect_records,
    prepare_series,
    write_pool,
)
from .source import DataSource, FetchJson, HarvestContext, HarvestedSeries, HttpFetcher

log = logging.getLogger("cascade.pool")

SELECTION_SCHEMA = "pool-d-selection/1"
REFUSAL_REASONS = ("key_collision", "no_source", "ambiguous_record", "id_collision")

JoinKey = tuple[str, str, int]


def join_key(hs: HarvestedSeries, channel: int = 0) -> JoinKey:
    """``(source, pre-suffix source series key, channel)`` of a harvested series."""
    return (str(hs.source), _sanitize_id(hs.series_id), int(channel))


def pool_d_series_id(key: JoinKey) -> str:
    """The stable ``pool_d`` series id of a join key:
    ``<base>~<sanitised source>.c<channel>``. It holds only
    ``[A-Za-z0-9._~-]``, so it never contains the ``:`` of a window id."""
    source, base, channel = key
    return f"{base}~{_sanitize_id(source)}.c{int(channel)}"


@dataclass(frozen=True)
class SelectedSeries:
    """One channel of one lane A record, with its join key."""

    key: JoinKey
    pool_series_id: str
    pool_d_series_id: str
    feed: str
    cluster: str
    channel: int
    n_channels: int
    freq: str
    domain: str
    attrs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "key": [self.key[0], self.key[1], int(self.key[2])],
            "pool_series_id": self.pool_series_id,
            "pool_d_series_id": self.pool_d_series_id,
            "feed": self.feed,
            "cluster": self.cluster,
            "channel": int(self.channel),
            "n_channels": int(self.n_channels),
            "freq": self.freq,
            "domain": self.domain,
            "attrs": {str(k): _plain(v) for k, v in sorted(self.attrs.items())},
        }


def _plain(value):
    """A JSON-safe copy of an attrs value (numpy scalars and arrays included)."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class PoolDSelection:
    """The joined selection: every lane A record is selected (each channel)
    or refused with a typed reason, never both."""

    series: tuple[SelectedSeries, ...]
    refused: tuple[tuple[str, str], ...]

    def join_keys(self) -> frozenset[JoinKey]:
        return frozenset(s.key for s in self.series)

    def by_key(self) -> dict[JoinKey, SelectedSeries]:
        return {s.key: s for s in self.series}

    def to_json(self) -> dict:
        return {
            "schema": SELECTION_SCHEMA,
            "series": [s.to_json() for s in self.series],
            "refused": [[sid, reason] for sid, reason in self.refused],
        }

    def sha256(self) -> str:
        doc = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(doc.encode()).hexdigest()


def _channels(rec: SeriesRecord) -> int:
    return int(rec.values.shape[0]) if rec.values.ndim == 2 else 1


def _refuse(refused: list, rec: SeriesRecord, reason: str, detail: str) -> None:
    refused.append((rec.series_id, reason))
    log.warning("pool_d_join_refused %s: %s (%s)", rec.series_id, reason, detail)


def select_pool_d(
    harvested: Sequence[HarvestedSeries],
    records: Sequence[SeriesRecord],
    cfg: PoolBuildConfig,
) -> PoolDSelection:
    """Join lane A's ``records`` (``collect_records`` over exactly
    ``harvested``, in harvest order, with ``cfg``) to their join keys."""
    per_source_key = Counter(
        (str(h.source), _sanitize_id(h.series_id)) for h in harvested if h.source)
    by_hash: dict[str, list[int]] = {}
    for i, h in enumerate(harvested):
        rec, _reason = prepare_series(h, cfg)
        if rec is not None:
            by_hash.setdefault(rec.content_hash, []).append(i)

    chosen: list[SelectedSeries] = []
    refused: list[tuple[str, str]] = []
    for rec in records:
        cands = by_hash.get(rec.content_hash, [])
        idents = {(harvested[i].source, _sanitize_id(harvested[i].series_id)) for i in cands}
        if len(idents) != 1:
            _refuse(refused, rec, "ambiguous_record",
                    f"{len(idents)} harvested identities carry its values")
            continue
        h = harvested[cands[0]]
        if not h.source:
            _refuse(refused, rec, "no_source", "the harvested series has no source")
            continue
        base_key = (str(h.source), _sanitize_id(h.series_id))
        if per_source_key[base_key] > 1:
            _refuse(refused, rec, "key_collision",
                    f"{per_source_key[base_key]} series of source {h.source!r} share key "
                    f"{base_key[1]!r}")
            continue
        n_ch = _channels(rec)
        for c in range(n_ch):
            key = join_key(h, c)
            chosen.append(SelectedSeries(
                key=key,
                pool_series_id=rec.series_id,
                pool_d_series_id=pool_d_series_id(key),
                feed=str(h.source),
                cluster=str(rec.metadata.get("source") or rec.series_id),
                channel=c,
                n_channels=n_ch,
                freq=str(rec.metadata.get("freq")),
                domain=str(rec.domain),
                attrs=dict(h.attrs or {}),
            ))

    ids = Counter(s.pool_d_series_id for s in chosen)
    clashing = {s.pool_series_id for s in chosen if ids[s.pool_d_series_id] > 1}
    if clashing:
        by_id = {r.series_id: r for r in records}
        for sid in sorted(clashing):
            _refuse(refused, by_id[sid], "id_collision", "its pool_d series id is not unique")
        chosen = [s for s in chosen if s.pool_series_id not in clashing]

    chosen.sort(key=lambda s: s.pool_d_series_id)
    return PoolDSelection(series=tuple(chosen), refused=tuple(sorted(refused)))


@dataclass
class _Replay:
    """One source's already-harvested series, re-yielded in harvest order."""

    name: str
    items: list[HarvestedSeries]

    def harvest(self, fetch: FetchJson, ctx: HarvestContext) -> Iterable[HarvestedSeries]:
        yield from self.items


def collect_with_selection(
    sources: Iterable[DataSource],
    ctx: HarvestContext,
    cfg: PoolBuildConfig,
    fetch: FetchJson,
) -> tuple[list[SeriesRecord], Counter, PoolDSelection]:
    """Harvest each source ONCE, run lane A's unchanged ``collect_records``
    over the replayed series, then join its records (:func:`select_pool_d`)."""
    replays: list[_Replay] = []
    for src in sources:
        replays.append(_Replay(str(getattr(src, "name", "")), list(src.harvest(fetch, ctx))))
    records, drops = collect_records(replays, ctx, cfg, fetch)
    harvested = [h for r in replays for h in r.items]
    return records, drops, select_pool_d(harvested, records, cfg)


def build_pool_with_selection(
    sources: Iterable[DataSource],
    out_dir: Path | str,
    ctx: HarvestContext,
    cfg: PoolBuildConfig,
    *,
    fetch: FetchJson | None = None,
    overwrite: bool = False,
) -> tuple[BuildSummary, PoolDSelection]:
    """:func:`~cascade.pool.builder.build_pool` (same bytes) plus the joined
    ``pool_d`` selection of the same run."""
    fetch = fetch or HttpFetcher()
    records, drops, selection = collect_with_selection(sources, ctx, cfg, fetch)
    check_domain_floor(Counter(r.domain for r in records), cfg.min_series_per_domain)
    summary = write_pool(
        records, out_dir, as_of=ctx.as_of.isoformat(), cfg=cfg, drops=drops, overwrite=overwrite
    )
    return summary, selection


__all__ = [
    "REFUSAL_REASONS",
    "SELECTION_SCHEMA",
    "PoolDSelection",
    "SelectedSeries",
    "build_pool_with_selection",
    "collect_with_selection",
    "join_key",
    "pool_d_series_id",
    "select_pool_d",
]
