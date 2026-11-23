"""tsbench-forge source — the live-data catalog as the held-out pool.

Reads the on-disk output of the tsbench-forge scraper (a curated catalog of
~90 daily-or-faster real public feeds across the 7 GIFT-Eval domains):

* ``<forge_dir>/sources.yaml``          — catalog: domain / dgp_class / frequency / panel
* ``<forge_dir>/data/<source_id>/<YYYY-MM-DD>.parquet`` — dated scrape snapshots

The coupling is the parquet contract only — no import of tsbench-forge code.
In the bucket-relay deployment the forge scrape host syncs ``data/`` +
``sources.yaml`` to a private bucket and the owner orchestrator syncs it down
before ``cascade-pool publish --sources tsbench_forge``; this source then reads
the local mirror. Point it at the mirror with ``TSFORGE_DIR`` (a directory
holding ``sources.yaml`` and ``data/``) or the constructor args.

Design notes:

* **Freshness/determinism** — only parquet files dated ``<= ctx.as_of`` are
  read and rows stamped after ``as_of`` are dropped, so re-building for a past
  ``as_of`` against the same mirror is reproducible (the audit hook).
* **Staleness guard** — a daily-or-faster source is fresh iff its newest
  snapshot file is at most ``max_stale_days`` before ``as_of`` (unchanged
  file-date rule). Slower-than-daily cadences and every rollup series are
  judged by their newest observation too, within ``max_stale_days ×
  cadence`` (see :func:`is_fresh`). Stale sources are skipped, and if *every* catalog source
  is stale the harvest raises instead of silently republishing old data (the
  decoupled-cron failure mode).
* **Rollups** — a catalog entry tagged ``rollup: sum|mean|last`` additionally
  yields that feed aggregated to UTC-day (``.rollup_D``) and Monday-start
  UTC-week (``.rollup_W``) cadence; see :func:`rollup_buckets` for the
  complete-bucket rule. Untagged entries are unaffected.
* **Clustering** — each series carries ``source=<catalog id>`` so the pool
  metadata records the cluster key the KOTH cluster bootstrap resamples
  (panel siblings and windows from one feed are correlated, not independent).
* Requires the ``pool-forge`` extra (pyarrow, pyyaml, pandas) — producer-side
  only; validators never import this module.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..source import FetchJson, HarvestContext, HarvestedSeries, HarvestError

log = logging.getLogger("cascade.pool")

# Directory holding `sources.yaml` + `data/` (the synced forge mirror).
ENV_FORGE_DIR = "TSFORGE_DIR"
DEFAULT_FORGE_DIR = "./tsforge"
# DEC-CA-0041 mv_channels contract cap (docs/EVAL_POOL.md): ordered value-columns
# within one (source, panel_row), at most this many, packed into one (C, L) window.
MAX_MV_CHANNELS = 32

# ISO-8601 duration (forge `frequency`) → (pandas-style freq, seasonal period).
# Explicit periods because cascade's gluonts-style mapping gives D → 1 while
# daily series want the weekly cycle, and sub-hourly multipliers need scaling.
# Sub-daily gets its natural daily cycle where it fits a 4096-step context;
# minute-and-faster uses the hourly cycle so the period stays well inside even
# short (min_context) histories. Weekly-and-slower catalog entries are skipped:
# they cannot reach the builder's min_length at their cadence.
#
# This table pins today's catalog; cadences NOT listed here are handled by
# :func:`resolve_frequency`, which derives the same convention from the parsed
# ISO duration — so a source added to the forge catalog at a brand-new cadence
# is picked up without a cascade change (the catalog grows on its own schedule).
FREQ_MAP: dict[str, tuple[str, int]] = {
    "PT30S": ("30S", 120),
    "PT1M": ("min", 60),
    "PT2M30S": ("150S", 24),
    "PT5M": ("5min", 288),
    "PT6M": ("6min", 240),
    "PT10M": ("10min", 144),
    "PT15M": ("15min", 96),
    "PT30M": ("30min", 48),
    "PT1H": ("H", 24),
    "PT8H": ("8H", 3),
    "P1D": ("D", 7),
}

_ISO_DURATION = re.compile(
    r"^P(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def _iso_duration_seconds(freq: str) -> float | None:
    """Nominal seconds of an ISO-8601 duration, or ``None`` if unparseable.
    Months/years use nominal lengths — only coarse skip decisions need them."""
    m = _ISO_DURATION.match(freq.strip())
    if not m or not any(m.groups()):
        return None
    g = {k: int(v) for k, v in m.groupdict().items() if v}
    return (
        g.get("years", 0) * 31_557_600
        + g.get("months", 0) * 2_629_800
        + g.get("weeks", 0) * 604_800
        + g.get("days", 0) * 86_400
        + g.get("hours", 0) * 3_600
        + g.get("minutes", 0) * 60
        + g.get("seconds", 0)
    ) or None


def resolve_frequency(freq: str) -> tuple[str, int] | None:
    """``(pandas freq, seasonal period)`` for a catalog frequency, or ``None``
    when the cadence can't fill an eval window (slower than daily, or
    unparseable).

    Unknown-but-parseable cadences derive the FREQ_MAP convention: daily data
    gets the weekly cycle; 5-minute-and-slower sub-daily gets its daily cycle;
    faster-than-5-minute gets its hourly cycle (the daily one would not fit a
    short ``min_context`` history). The derivation reproduces every FREQ_MAP
    entry exactly (pinned by test), so the table is a readability/canonical
    layer, not divergent behaviour.
    """
    known = FREQ_MAP.get(freq)
    if known is not None:
        return known
    sec = _iso_duration_seconds(str(freq))
    if sec is None or sec > 86_400:
        return None
    sec = int(sec)
    pandas_freq = f"{sec}S"
    if sec == 86_400:
        return "D", 7
    cycle = 86_400 if sec >= 300 else 3_600
    period = max(2, round(cycle / sec))
    return pandas_freq, period

# A window straddling a scrape gap is a fake level shift; segment at gaps
# larger than this multiple of the median sampling interval (mirrors the
# forge serving layer's contiguity rule).
GAP_FACTOR = 8.0
# Preferred minimum segment length: horizon + the builder's default min_context.
MIN_CONTEXT = 256

# Timestamp formats "mixed" inference can't get: wikimedia's YYYYMMDDHH and
# NDBC's space-separated stamps.
_TS_FALLBACK_FORMATS = ("%Y%m%d%H", "%Y %m %d %H %M", "%Y %m %d")

# ── rollups (catalog tag ``rollup: sum|mean|last``) ─────────────────────────
# Aggregations a catalog entry may request. Any other value is ignored with a
# warning (the tag is a cross-repo contract; cascade never guesses one).
ROLLUP_AGGS = ("sum", "mean", "last")
# Target cadences: (series-id suffix, bucket seconds, pandas freq, seasonal
# period). Days are calendar UTC days; weeks are Monday-start UTC weeks
# (Mon 00:00 → next Mon 00:00), which pandas spells "W" (= W-SUN, the week
# ENDING Sunday). Daily keeps the reader's weekly cycle (resolve_frequency
# "P1D"); weekly uses the annual cycle, which fits the builder's min_length.
# A target is emitted only when it is strictly slower than the raw cadence.
ROLLUP_TARGETS: tuple[tuple[str, int, str, int], ...] = (
    ("D", 86_400, "D", 7),
    ("W", 604_800, "W", 52),
)
# A bucket is complete iff its finite raw observations number at least
# (1 - this) × the bucket's expected count (bucket seconds / raw cadence).
# Mirrors the builder's default PoolBuildConfig.max_missing_frac (0.2): the
# same share of missing points the builder tolerates per series is tolerated
# per bucket; beyond it the bucket is NaN, never a partial aggregate.
ROLLUP_MAX_MISSING_FRAC = 0.2
_EPOCH_DOW_TO_MONDAY = 3  # 1970-01-01 was a Thursday: (day + 3) // 7 = Monday-start week


def rollup_agg(entry: dict) -> str | None:
    """The catalog entry's ``rollup`` aggregation, or ``None`` when untagged
    (or tagged with a value outside :data:`ROLLUP_AGGS`, which is logged)."""
    tag = entry.get("rollup")
    if tag is None:
        return None
    if tag not in ROLLUP_AGGS:
        log.warning("tsbench_forge: ignoring %s rollup=%r (allowed: %s)",
                    entry.get("id"), tag, "|".join(ROLLUP_AGGS))
        return None
    return str(tag)


def rollup_buckets(
    sec: np.ndarray, values: np.ndarray, *, agg: str, raw_sec: float, bucket_sec: int,
    max_missing_frac: float = ROLLUP_MAX_MISSING_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate one channel of raw observations into calendar UTC buckets.

    ``sec``: int64 UTC epoch seconds, ascending; ``values``: float, same length.
    Returns ``(bucket_start_sec, aggregated)`` over the CONTIGUOUS bucket grid
    from the first to the last bucket holding an observation.

    Complete-bucket rule (pinned by test): with ``expected = bucket_sec /
    raw_sec`` raw points per bucket, a bucket whose finite observations number
    fewer than ``(1 - max_missing_frac) × expected`` is NaN — including the
    leading/trailing partial days of a feed. A complete bucket aggregates its
    finite observations:

    * ``mean`` — their mean;
    * ``last`` — the latest-timestamped one;
    * ``sum``  — their sum; if the bucket is complete but short of
      ``expected`` points (within tolerance) the sum is pro-rated to the full
      bucket (``mean × expected``) — the bucket-level analogue of the
      builder's gap interpolation, so a tolerated gap is never a fake dip.
    """
    sec = np.asarray(sec, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    day = sec // 86_400
    if bucket_sec == 86_400:
        key = day
    elif bucket_sec == 604_800:
        key = (day + _EPOCH_DOW_TO_MONDAY) // 7
    else:  # pragma: no cover - only the two calendar targets exist
        raise ValueError(f"unsupported rollup bucket {bucket_sec}s")
    if not key.size:
        return np.empty(0, dtype=np.int64), np.empty(0)
    k0 = int(key.min())
    n_buckets = int(key.max()) - k0 + 1
    if bucket_sec == 86_400:
        starts = (np.arange(n_buckets, dtype=np.int64) + k0) * 86_400
    else:
        starts = ((np.arange(n_buckets, dtype=np.int64) + k0) * 7 - _EPOCH_DOW_TO_MONDAY) * 86_400

    fin = np.isfinite(vals)
    kf, vf = key[fin] - k0, vals[fin]
    counts = np.bincount(kf, minlength=n_buckets).astype(np.float64)
    out = np.full(n_buckets, np.nan)
    expected = float(bucket_sec) / float(raw_sec)
    complete = counts >= (1.0 - max_missing_frac) * expected
    if agg == "last":
        # Observations are ascending, so the last occurrence of each key is
        # the latest one: first occurrence in the reversed array.
        uniq, first_rev = np.unique(kf[::-1], return_index=True)
        last = np.full(n_buckets, np.nan)
        last[uniq] = vf[::-1][first_rev]
        agg_vals = last
    else:
        sums = np.bincount(kf, weights=vf, minlength=n_buckets)
        with np.errstate(invalid="ignore", divide="ignore"):
            if agg == "mean":
                agg_vals = sums / counts
            elif agg == "sum":
                agg_vals = np.where(counts < expected, sums / counts * expected, sums)
            else:
                raise ValueError(f"unknown rollup aggregation {agg!r}")
    out[complete] = agg_vals[complete]
    return starts, out


def freshness_window(cadence_sec: float | None, max_stale_days: int) -> dt.timedelta:
    """How far behind ``as_of`` a series' newest point may lag: ``max_stale_days``
    missed scrape cycles, a cycle being the slower of one day (the scrape
    cron) and the series' own cadence. Daily-and-faster: exactly
    ``max_stale_days`` days (the historical rule); weekly: ``max_stale_days``
    weeks; monthly: ``max_stale_days`` nominal months."""
    cycle = max(86_400.0, float(cadence_sec or 0.0))
    return dt.timedelta(seconds=max_stale_days * cycle)


def observation_freshness_applies(cadence_sec: float | None) -> bool:
    """True iff a series' newest OBSERVATION is checked (not only its file
    date): cadences slower than daily, plus every rollup series (checked in
    ``_rollup_values``). Daily-or-faster raw feeds keep the file-date-only rule
    so publication-lagged feeds stay admitted (owner 2026-11-23)."""
    return cadence_sec is not None and float(cadence_sec) > 86_400


def is_fresh(
    *, as_of: dt.date, cadence_sec: float | None, max_stale_days: int,
    newest_file: dt.date | None, newest_obs: dt.datetime | None,
) -> bool:
    """Freshness by the newest OBSERVATION relative to cadence.

    Fresh iff both the newest snapshot-file date and the newest observation
    timestamp (UTC; ``None`` = unknown, e.g. no parseable timestamps) are at or
    after ``as_of 00:00 UTC − freshness_window(cadence)``. Requiring the file
    too keeps every stale-file skip of the old rule (an observation can't be
    fresher than the scrape that delivered it). Callers pass ``newest_obs``
    only where :func:`observation_freshness_applies` (slower than daily) and
    for rollup series; daily-or-faster raw feeds pass ``None`` (file date
    only, the historical rule)."""
    floor = dt.datetime.combine(as_of, dt.time(), tzinfo=dt.UTC) - freshness_window(
        cadence_sec, max_stale_days)
    if newest_file is not None and dt.datetime.combine(
            newest_file, dt.time(), tzinfo=dt.UTC) < floor:
        return False
    return newest_obs is None or newest_obs >= floor


class TsbenchForgeSource:
    name = "tsbench_forge"

    def __init__(
        self,
        forge_dir: str | Path | None = None,
        *,
        max_stale_days: int = 4,
        max_series_per_source: int = 200,
        mv_pack: bool = False,
        panel_sample_seed: str | None = None,
    ) -> None:
        self.forge_dir = Path(forge_dir or os.environ.get(ENV_FORGE_DIR, DEFAULT_FORGE_DIR))
        self.max_stale_days = int(max_stale_days)
        self.max_series_per_source = int(max_series_per_source)
        # DEC-CA-0041: when True, a catalog entry's ``mv_channels`` columns are
        # packed into ONE (C, L) series (multivariate scoring). When False
        # (default, until mv_score is armed), a tagged source is PROJECTED to its
        # first mv_channel — one univariate window, so the pool stays inert.
        self.mv_pack = bool(mv_pack)
        # Balanced selection (cascade-pool --selection balanced): when set, the
        # per-feed cap keeps a SEEDED random sample of the feed's panel rows
        # (and counts only rows that yield values) instead of the first N in
        # sort order — which pinned the same alphabetically-first stations
        # into every snapshot. ``None`` = the historical first-N behaviour.
        self.panel_sample_seed = panel_sample_seed

    # ------------------------------------------------------------------ deps

    @staticmethod
    def _require_deps():
        try:
            import pandas as pd
            import pyarrow.parquet as pq
            import yaml
        except ImportError as e:  # pragma: no cover - exercised only without extra
            raise HarvestError(
                "tsbench_forge source needs pyarrow, pyyaml, and pandas; "
                'install the producer extra: pip install "cascade[pool-forge]"'
            ) from e
        return pd, pq, yaml

    # --------------------------------------------------------------- harvest

    def harvest(self, fetch: FetchJson, ctx: HarvestContext) -> Iterable[HarvestedSeries]:
        """Yield one series per (catalog source, panel row). ``fetch`` is unused
        — all I/O is the local forge mirror."""
        pd, pq, yaml = self._require_deps()

        catalog_path = self.forge_dir / "sources.yaml"
        data_dir = self.forge_dir / "data"
        if not catalog_path.is_file():
            raise HarvestError(
                f"forge catalog not found at {catalog_path}; sync the forge mirror "
                f"and/or set {ENV_FORGE_DIR}"
            )
        with open(catalog_path, encoding="utf-8") as f:
            catalog = yaml.safe_load(f) or []

        from ..families import catalog_families

        families = catalog_families(catalog)
        emitted = 0
        usable_sources = 0
        for entry in catalog:
            sid = entry.get("id")
            if not sid or entry.get("disabled"):
                continue
            mapped = resolve_frequency(str(entry.get("frequency", "")))
            if mapped is None:
                # Weekly-and-slower can't fill a window (expected); anything
                # unparseable is a catalog-schema surprise worth surfacing.
                if _iso_duration_seconds(str(entry.get("frequency", ""))) is None:
                    log.warning(
                        "tsbench_forge: skipping %s — unparseable frequency %r",
                        sid, entry.get("frequency"),
                    )
                continue
            freq, seasonal_period = mapped

            raw_sec = _iso_duration_seconds(str(entry.get("frequency", "")))

            files = self._snapshot_files(data_dir / sid, as_of=ctx.as_of)
            if not files:
                continue
            newest_file = _stem_date(files[-1])
            if not is_fresh(as_of=ctx.as_of, cadence_sec=raw_sec,
                            max_stale_days=self.max_stale_days,
                            newest_file=newest_file, newest_obs=None):
                continue  # cron fell behind for this feed; don't serve stale data
            # Daily-or-faster feeds keep the historical file-date-only rule (owner
            # 2026-11-23: publication-lagged feeds stay admitted). The newest-
            # observation rule applies only to slower-than-daily cadences.
            slow = observation_freshness_applies(raw_sec)
            if not slow:
                usable_sources += 1

            df = self._load_frame(pd, pq, files)
            if df is None or df.empty:
                continue
            if slow:
                newest_obs = self._newest_observation(pd, df, ctx)
                if not is_fresh(as_of=ctx.as_of, cadence_sec=raw_sec,
                                max_stale_days=self.max_stale_days,
                                newest_file=newest_file, newest_obs=newest_obs):
                    log.warning("tsbench_forge: skipping %s — newest observation %s is "
                                "stale for its cadence (as_of=%s)", sid, newest_obs, ctx.as_of)
                    continue
                usable_sources += 1

            agg = rollup_agg(entry)
            targets = [t for t in ROLLUP_TARGETS if agg and raw_sec and raw_sec < t[1]]
            domain = str(entry.get("domain", "unknown"))
            dgp_class = str(entry.get("dgp_class", ""))
            # DEC-CA-0041 mv_channels contract (docs/EVAL_POOL.md): an ordered
            # list of value-columns within one (source, panel_row). Pack them to
            # (C, L) when armed; else project to the first (one univariate row).
            mvc = entry.get("mv_channels") or None
            if mvc:
                mvc = [str(c) for c in mvc][:MAX_MV_CHANNELS]
                columns = mvc if self.mv_pack else [mvc[0]]
            else:
                columns = None
            for panel_key, values, sub in self._iter_panel_series(
                    pd, df, ctx, columns=columns, feed=sid, with_frame=True):
                suffix = "__".join(f"{k}_{v}" for k, v in panel_key.items())
                series_id = f"tsforge__{sid}" + (f"__{suffix}" if suffix else "")
                attrs = {"dgp_class": dgp_class, **panel_key, "source_family": families[str(sid)]}
                out = [HarvestedSeries(
                    series_id=series_id,
                    values=values,
                    freq=freq,
                    domain=domain,
                    seasonal_period=seasonal_period,
                    source=sid,
                    attrs=attrs,
                )]
                for tag, bucket_sec, r_freq, r_period in targets:
                    rolled = self._rollup_values(
                        pd, sub, [c for c in sub.columns if c.startswith("_panel_")], ctx,
                        columns=columns, agg=agg, raw_sec=raw_sec, bucket_sec=bucket_sec,
                    )
                    if rolled is None:
                        continue
                    # Same source/family: a rollup is the SAME feed at another
                    # cadence, so it adds no cluster to the KOTH bootstrap.
                    out.append(HarvestedSeries(
                        series_id=f"{series_id}.rollup_{tag}",
                        values=rolled,
                        freq=r_freq,
                        domain=domain,
                        seasonal_period=r_period,
                        source=sid,
                        attrs={**attrs, "rollup": agg},
                    ))
                for hs in out:
                    if emitted >= ctx.max_series:
                        log.warning(
                            "tsbench_forge: max_series=%d reached at catalog entry %s; "
                            "later catalog entries are dropped (catalog-order bias) — "
                            "raise HarvestContext.max_series to cover the full catalog",
                            ctx.max_series, sid,
                        )
                        return
                    yield hs
                    emitted += 1

        if usable_sources == 0:
            raise HarvestError(
                f"no fresh tsbench-forge data under {data_dir} "
                f"(as_of={ctx.as_of}, max_stale_days={self.max_stale_days}); "
                "is the forge scrape cron / bucket sync running?"
            )

    # ----------------------------------------------------------------- files

    @staticmethod
    def _snapshot_files(src_dir: Path, *, as_of: dt.date) -> list[Path]:
        """Dated parquet snapshots for one source, oldest → newest, capped at
        ``as_of`` so a rebuild for a past date sees the same inputs."""
        if not src_dir.is_dir():
            return []
        out = []
        for p in sorted(src_dir.glob("*.parquet")):
            stamp = _stem_date(p)
            if stamp is None or stamp <= as_of:
                out.append(p)
        return out

    def _load_frame(self, pd, pq, files: list[Path]):
        """Concatenate snapshots, dedup on timestamp (keep newest scrape), and
        sort chronologically.

        Reading *all* dated files (not just the newest) covers both feed shapes:
        history-rich feeds re-deliver deep history every scrape (dedup collapses
        it) while snapshot-style feeds only accumulate history across files.
        """
        frames = []
        for p in files:
            try:
                frames.append(pq.read_table(p).to_pandas())
            except Exception:  # noqa: BLE001 — one corrupt snapshot must not kill the source
                continue
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        if "timestamp" in df.columns:
            # Scope the dedup per panel row — panel siblings share timestamps.
            keys = ["timestamp", *[c for c in df.columns if c.startswith("_panel_")]]
            df = df.drop_duplicates(subset=keys, keep="last")
        return df

    # ---------------------------------------------------------------- series

    def _iter_panel_series(self, pd, df, ctx: HarvestContext, *, columns=None, feed: str = "",
                           with_frame: bool = False):
        """Yield ``(panel_key, values)`` per panel row (or once when unpaneled);
        ``(panel_key, values, row_frame)`` when ``with_frame`` (rollups re-read
        the row's raw observations).

        ``columns`` (DEC-CA-0041): the specific value-columns to emit — a
        length-1 list projects to one univariate channel, a longer list packs an
        aligned ``(C, L)`` group. ``None`` keeps the densest-column default."""
        def item(panel_key, values, sub):
            return (panel_key, values, sub) if with_frame else (panel_key, values)

        panel_cols = [c for c in df.columns if c.startswith("_panel_")]
        if not panel_cols:
            values = self._extract_values(pd, df, panel_cols, ctx, columns=columns)
            if values is not None:
                yield item({}, values, df)
            return
        groups = df.groupby(panel_cols, sort=True)
        if self.panel_sample_seed is not None:
            for panel_key, values, sub in self._iter_sampled_panel(
                    pd, groups, panel_cols, ctx, columns=columns, feed=feed):
                yield item(panel_key, values, sub)
            return
        for n_done, (key, sub) in enumerate(groups):
            if n_done >= self.max_series_per_source:
                return
            key_t = key if isinstance(key, tuple) else (key,)
            panel_key = {
                c.removeprefix("_panel_"): str(v) for c, v in zip(panel_cols, key_t, strict=False)
            }
            values = self._extract_values(pd, sub, panel_cols, ctx, columns=columns)
            if values is not None:
                yield item(panel_key, values, sub)

    def _iter_sampled_panel(self, pd, groups, panel_cols, ctx: HarvestContext, *,
                            columns=None, feed: str = ""):
        """Seeded per-feed sample: visit the feed's panel rows in an order drawn
        from ``(panel_sample_seed, feed)`` and yield the first
        ``max_series_per_source`` that produce values."""
        from ..builder import seeded_rng

        items = list(groups)          # sorted by key (groupby sort=True) — canonical
        order = seeded_rng(str(self.panel_sample_seed), "panel", feed).permutation(len(items))
        kept = 0
        for i in order:
            if kept >= self.max_series_per_source:
                return
            key, sub = items[i]
            key_t = key if isinstance(key, tuple) else (key,)
            panel_key = {
                c.removeprefix("_panel_"): str(v) for c, v in zip(panel_cols, key_t, strict=False)
            }
            values = self._extract_values(pd, sub, panel_cols, ctx, columns=columns)
            if values is not None:
                kept += 1
                yield panel_key, values, sub

    @staticmethod
    def _sorted_frame(pd, df, ctx: HarvestContext):
        """``(df, ts)`` sorted by parsed UTC timestamp with rows on/after
        ``as_of + 1 day`` dropped; ``ts`` is ``None`` (``df`` untouched) when
        the frame has no usable timestamps."""
        ts = None
        if "timestamp" in df.columns and len(df) > 1:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
            if ts.notna().mean() <= 0.9:
                for fmt in _TS_FALLBACK_FORMATS:
                    alt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format=fmt)
                    if alt.notna().mean() > 0.9:
                        ts = alt
                        break
            if ts.notna().mean() > 0.9:
                keep = ts.notna()
                df, ts = df[keep], ts[keep]
                order = np.argsort(ts.to_numpy(), kind="stable")
                df, ts = df.iloc[order], ts.iloc[order]
                # Freshness cutoff: nothing scraped after as_of enters the pool.
                cutoff = pd.Timestamp(ctx.as_of, tz="UTC") + pd.Timedelta(days=1)
                keep = ts < cutoff
                df, ts = df[keep], ts[keep]
            else:
                ts = None
        return df, ts

    @staticmethod
    def _named_channels(df, columns, excluded):
        """Exactly the named value-columns as float arrays, or ``None`` if any
        is missing / non-numeric (a tagged channel must be present)."""
        chans = []
        for c in columns:
            if c in excluded or c not in df.columns:
                return None
            try:
                chans.append(df[c].astype(float).to_numpy())
            except (TypeError, ValueError):
                return None
        return chans

    @staticmethod
    def _densest_column(df, excluded):
        """``(finite_count, values)`` of the numeric column with the most finite
        values (first wins ties), or ``(-1, None)``."""
        best_finite, best = -1, None
        for c in [c for c in df.columns if c not in excluded]:
            try:
                v = df[c].astype(float).to_numpy()
            except (TypeError, ValueError):
                continue
            finite = int(np.isfinite(v).sum())
            if finite > best_finite:
                best_finite, best = finite, v
        return best_finite, best

    def _newest_observation(self, pd, df, ctx: HarvestContext) -> dt.datetime | None:
        """Newest observation timestamp (UTC) at or before ``as_of`` across the
        whole source frame, or ``None`` when timestamps are unusable."""
        _, ts = self._sorted_frame(pd, df, ctx)
        if ts is None or not len(ts):
            return None
        return ts.iloc[-1].to_pydatetime()

    def _extract_values(self, pd, df, panel_cols: list[str], ctx: HarvestContext,
                        *, columns=None):
        """One panel row's frame → a clean-enough float array, or ``None``.

        Sorts by parsed timestamp, drops post-``as_of`` rows, and keeps the
        freshest contiguous segment. With ``columns=None`` it picks the densest
        numeric value column and returns 1-D (categorical feeds skipped — rank
        codes are meaningless under MASE/CRPS). With ``columns`` given
        (DEC-CA-0041) it takes exactly those, aligned on the shared segment:
        one column → 1-D, several → ``(C, L)``; a missing/non-numeric named
        column drops the whole group (a tagged channel must be present).
        NaN gaps are left in place; the builder interpolates or drops.
        """
        df, ts = self._sorted_frame(pd, df, ctx)
        if len(df) < 2:
            return None

        excluded = {"timestamp", *panel_cols}
        ts_arr = ts.to_numpy() if ts is not None else None

        if columns is not None:
            # DEC-CA-0041: pack exactly the named channels, aligned on ONE shared
            # freshest segment (they are columns of the same panel row).
            chans = self._named_channels(df, columns, excluded)
            if chans is None:
                return None
            width = min(len(v) for v in chans)
            if width < 2:
                return None
            lo, hi = self._freshest_segment(
                ts_arr[:width] if ts_arr is not None else None, width,
                min_len=ctx.horizon + MIN_CONTEXT,
            )
            packed = np.stack([v[:width][lo:hi] for v in chans], axis=0)   # (C, L)
            return packed[0] if packed.shape[0] == 1 else packed

        best_finite, best = self._densest_column(df, excluded)
        if best is None or best_finite < 2:
            return None

        lo, hi = self._freshest_segment(
            ts_arr, len(best), min_len=ctx.horizon + MIN_CONTEXT,
        )
        return best[lo:hi]

    def _rollup_values(self, pd, df, panel_cols: list[str], ctx: HarvestContext, *,
                       columns, agg: str, raw_sec: float, bucket_sec: int):
        """One panel row's raw observations → its calendar-bucket rollup.

        Same channel choice as :meth:`_extract_values` (named ``columns`` or the
        densest column), bucketed by :func:`rollup_buckets` over the row's full
        ``<= as_of`` history. The kept span is the freshest contiguous run of
        COMPLETE buckets (all channels complete), by the same
        :meth:`_freshest_segment` gap rule applied at the bucket cadence, from
        its first to its last complete bucket: leading/trailing partial
        buckets never enter, interior incomplete ones stay NaN for the
        builder. ``None`` without usable timestamps, with < 2 complete
        buckets, or when the newest complete bucket is stale for its cadence.
        """
        df, ts = self._sorted_frame(pd, df, ctx)
        if ts is None or len(df) < 2:
            return None
        excluded = {"timestamp", *panel_cols}
        if columns is not None:
            chans = self._named_channels(df, columns, excluded)
            if chans is None:
                return None
        else:
            best_finite, best = self._densest_column(df, excluded)
            if best is None or best_finite < 2:
                return None
            chans = [best]
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        sec = ((ts - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)

        starts, rolled = None, []
        for v in chans:
            starts, r = rollup_buckets(sec, v, agg=agg, raw_sec=raw_sec, bucket_sec=bucket_sec)
            rolled.append(r)
        grid = np.stack(rolled, axis=0)                       # (C, n_buckets)
        complete = np.flatnonzero(np.isfinite(grid).all(axis=0))
        if complete.size < 2:
            return None
        lo, hi = self._freshest_segment(
            starts[complete].astype("datetime64[s]"), int(complete.size),
            min_len=ctx.horizon + MIN_CONTEXT,
        )
        g_lo, g_hi = int(complete[lo]), int(complete[hi - 1]) + 1
        newest_end = dt.datetime.fromtimestamp(int(starts[g_hi - 1]) + bucket_sec,
                                               tz=dt.UTC)
        if not is_fresh(as_of=ctx.as_of, cadence_sec=bucket_sec,
                        max_stale_days=self.max_stale_days,
                        newest_file=None, newest_obs=newest_end):
            return None
        out = grid[:, g_lo:g_hi]
        return out[0] if out.shape[0] == 1 else out

    @staticmethod
    def _freshest_segment(
        ts_arr: np.ndarray | None, n: int, *, min_len: int
    ) -> tuple[int, int]:
        """Bounds of the newest contiguous segment that is at least ``min_len``
        long, falling back to the longest segment; whole series without usable
        timestamps."""
        if ts_arr is None or n < 3:
            return 0, n
        d = np.diff(ts_arr).astype("timedelta64[s]").astype(float)
        pos = d[d > 0]
        if not pos.size:
            return 0, n
        med = float(np.median(pos))
        cuts = (np.flatnonzero(d > GAP_FACTOR * med) + 1).tolist()
        bounds = [0, *cuts, n]
        segments = list(zip(bounds[:-1], bounds[1:], strict=False))
        for lo, hi in reversed(segments):  # newest first
            if hi - lo >= min_len:
                return lo, hi
        return max(segments, key=lambda p: p[1] - p[0])


def _stem_date(p: Path) -> dt.date | None:
    try:
        return dt.date.fromisoformat(p.stem)
    except ValueError:
        return None
