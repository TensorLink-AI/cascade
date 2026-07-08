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
* **Staleness guard** — a source whose newest snapshot is older than
  ``max_stale_days`` before ``as_of`` is skipped; if *every* catalog source is
  stale the harvest raises instead of silently republishing old data (the
  decoupled-cron failure mode).
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


class TsbenchForgeSource:
    name = "tsbench_forge"

    def __init__(
        self,
        forge_dir: str | Path | None = None,
        *,
        max_stale_days: int = 4,
        max_series_per_source: int = 200,
    ) -> None:
        self.forge_dir = Path(forge_dir or os.environ.get(ENV_FORGE_DIR, DEFAULT_FORGE_DIR))
        self.max_stale_days = int(max_stale_days)
        self.max_series_per_source = int(max_series_per_source)

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

        stale_cutoff = ctx.as_of - dt.timedelta(days=self.max_stale_days)
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

            files = self._snapshot_files(data_dir / sid, as_of=ctx.as_of)
            if not files:
                continue
            newest = _stem_date(files[-1])
            if newest is not None and newest < stale_cutoff:
                continue  # cron fell behind for this feed; don't serve stale data
            usable_sources += 1

            df = self._load_frame(pd, pq, files)
            if df is None or df.empty:
                continue
            domain = str(entry.get("domain", "unknown"))
            dgp_class = str(entry.get("dgp_class", ""))
            for panel_key, values in self._iter_panel_series(pd, df, ctx):
                if emitted >= ctx.max_series:
                    log.warning(
                        "tsbench_forge: max_series=%d reached at catalog entry %s; "
                        "later catalog entries are dropped (catalog-order bias) — "
                        "raise HarvestContext.max_series to cover the full catalog",
                        ctx.max_series, sid,
                    )
                    return
                suffix = "__".join(f"{k}_{v}" for k, v in panel_key.items())
                series_id = f"tsforge__{sid}" + (f"__{suffix}" if suffix else "")
                yield HarvestedSeries(
                    series_id=series_id,
                    values=values,
                    freq=freq,
                    domain=domain,
                    seasonal_period=seasonal_period,
                    source=sid,
                    attrs={"dgp_class": dgp_class, **panel_key},
                )
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

    def _iter_panel_series(self, pd, df, ctx: HarvestContext):
        """Yield ``(panel_key, values)`` per panel row (or once when unpaneled)."""
        panel_cols = [c for c in df.columns if c.startswith("_panel_")]
        if not panel_cols:
            values = self._extract_values(pd, df, panel_cols, ctx)
            if values is not None:
                yield {}, values
            return
        groups = df.groupby(panel_cols, sort=True)
        for n_done, (key, sub) in enumerate(groups):
            if n_done >= self.max_series_per_source:
                return
            key_t = key if isinstance(key, tuple) else (key,)
            panel_key = {
                c.removeprefix("_panel_"): str(v) for c, v in zip(panel_cols, key_t, strict=False)
            }
            values = self._extract_values(pd, sub, panel_cols, ctx)
            if values is not None:
                yield panel_key, values

    def _extract_values(self, pd, df, panel_cols: list[str], ctx: HarvestContext):
        """One panel row's frame → a clean-enough 1-D float array, or ``None``.

        Sorts by parsed timestamp, drops post-``as_of`` rows, picks the densest
        numeric value column (categorical feeds are skipped — rank codes are
        meaningless under MASE/CRPS), and keeps the freshest contiguous segment.
        NaN gaps are left in place; the builder interpolates or drops.
        """
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
        if len(df) < 2:
            return None

        excluded = {"timestamp", *panel_cols}
        candidates = [c for c in df.columns if c not in excluded]
        best_finite, best = -1, None
        for c in candidates:
            try:
                v = df[c].astype(float).to_numpy()
            except (TypeError, ValueError):
                continue
            finite = int(np.isfinite(v).sum())
            if finite > best_finite:
                best_finite, best = finite, v
        if best is None or best_finite < 2:
            return None

        lo, hi = self._freshest_segment(
            ts.to_numpy() if ts is not None else None,
            len(best),
            min_len=ctx.horizon + MIN_CONTEXT,
        )
        return best[lo:hi]

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
