"""MarketCast forward bar recorder (DEC-CA-0064 review M15, 2026-10-19;
native intervals 2026-11-10).

Hyperliquid's ``candleSnapshot`` serves only its most recent ~5000 candles
per interval (measured 2026-10-19: 1m ≈ 3.6 days, 5m ≈ 17, 15m ≈ 52, 1h ≈
208), so a window's realized bars must be captured before the API drops
them. ``cascade-pool mkt-record`` (cron'd every few minutes) RECORDS the
native 1m candles and the candles of every configured native interval
(``[marketcast] native_intervals``) plus 1h, forward, as they close, and
``cascade-pool publish --lane mkt --bar-store DIR`` builds the snapshot's
banks from the recording (the 1-minute bank from the recording alone; the
5m / 15m / 1h banks from the API with the recording filling what it no
longer serves).

**No lead time (owner 2026-11-10).** A series is cut from its own native
bank, so at arming only each series' CONTEXT must exist, which the API depth
covers; the FIRST pass of an empty (asset, interval) backfills the API's full
depth, so a recorder started at arming already holds every bar a window's
context needs. Later passes re-fetch ``lookback``.

* **Resumable.** Every run re-fetches the last ``lookback`` (default 48 h,
  well inside the API depth) and merges: a recorder down for less than the
  API depth loses nothing.
* **Gap-tolerant.** A minute nobody recorded stays NaN — never
  interpolated; the windows over it are simply not cut.
* **Closed bars only.** A candle is written once its minute has ended
  (``open + 60 s ≤ now``), so an in-progress candle never lands.

Layout::

    <root>/<class>/<asset>/<YYYY-MM-DD>/<field>.npy        1m: float64[1440], NaN = missing
    <root>/<class>/<asset>/<iv>/<YYYY-MM-DD>/<field>.npy   native <iv> (5m / 15m / 1h):
                                                           float64[1440 / minutes(iv)]
    <root>/<class>/<asset>/funding.jsonl                   {"t", "rate", "premium"} samples

Only Hyperliquid primaries (``hyperliquid`` / ``hyperliquid_xyz``) are
recorded here; Binance (the crypto cross-check) serves long history and is
still fetched at publish; dTAO alpha keeps its own recorder
(``mkt-record-alpha``).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .marketcast_feeds import FIELDS, HL_INTERVAL_MINUTES, MINUTE_MS

DAY = 86400
DEFAULT_LOOKBACK_MINUTES = 48 * 60
HL_1M_DEPTH_MINUTES = 5000          # measured 2026-10-19: candleSnapshot's 1m depth
HL_DEPTH_CANDLES = 5000             # the same ~5000-candle depth at every interval
RECORD_COARSE = "1h"                # the hourly bank is always recorded too


def _day_key(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=UTC).strftime("%Y-%m-%d")


class MktBarStore:
    """The recorder's on-disk bank (see the module docstring)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, cls: str, asset: str, interval: str = "1m") -> Path:
        d = self.root / cls / asset
        return d if interval == "1m" else d / interval

    def has_bars(self, cls: str, asset: str, interval: str = "1m") -> bool:
        """Whether anything was recorded for ``(cls, asset, interval)``."""
        d = self._dir(cls, asset, interval)
        return d.is_dir() and any(p.is_dir() and p.name[:1].isdigit() for p in d.iterdir())

    def write_bars(self, cls: str, asset: str, candles, *, now: float,
                   interval: str = "1m") -> int:
        """Merge closed ``interval`` candles ``[(open_ms, o, h, l, c, v),
        ...]``; returns the candles written. Atomic per day file."""
        step = 60 * HL_INTERVAL_MINUTES[interval]
        per_day = DAY // step
        by_day: dict[str, list] = {}
        for t, o, h, lo, c, v in candles:
            t_s = int(t) // 1000
            if t_s % step or t_s + step > float(now):
                continue                                   # off-grid / still open
            by_day.setdefault(_day_key(t_s), []).append((t_s, (o, h, lo, c, v)))
        n = 0
        for day, rows in sorted(by_day.items()):
            d = self._dir(cls, asset, interval) / day
            d.mkdir(parents=True, exist_ok=True)
            day0 = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
            arrs = {f: self._load(d / f"{f}.npy", per_day) for f in FIELDS}
            for t_s, vals in rows:
                k = (t_s - day0) // step
                for f, x in zip(FIELDS, vals, strict=True):
                    arrs[f][k] = float(x)
                n += 1
            for f, a in arrs.items():
                tmp = d / f".{f}.npy.tmp"
                with open(tmp, "wb") as fh:
                    np.save(fh, a, allow_pickle=False)
                os.replace(tmp, d / f"{f}.npy")
        return n

    def write_funding(self, cls: str, asset: str, rows) -> int:
        """Merge hourly funding samples ``[(time_ms, rate, premium), ...]``
        (dedup by time)."""
        p = self._dir(cls, asset) / "funding.jsonl"
        have = {int(r["t"]): r for r in self._funding_rows(p)}
        new = 0
        for t, rate, prem in rows:
            if int(t) not in have:
                new += 1
            have[int(t)] = {"t": int(t), "rate": float(rate),
                            "premium": None if not np.isfinite(prem) else float(prem)}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(have[k], sort_keys=True) + "\n" for k in sorted(have)),
                       encoding="utf-8")
        os.replace(tmp, p)
        return new

    @staticmethod
    def _load(p: Path, n: int = 1440) -> np.ndarray:
        if p.is_file():
            a = np.load(p, allow_pickle=False)
            if a.shape == (int(n),):
                return a.astype(np.float64)
        return np.full(int(n), np.nan)

    @staticmethod
    def _funding_rows(p: Path) -> list[dict]:
        if not p.is_file():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def read(self, cls: str, asset: str, t0: int, minutes: int, *,
             interval: str = "1m") -> dict[str, np.ndarray]:
        """``{open, high, low, close, volume, funding, premium}`` on the
        ``interval`` grid ``[t0, t0 + minutes·step)`` (``minutes`` = the
        number of grid steps) — NaN wherever nothing was recorded (funding:
        the latest recorded sample at or before each step)."""
        from .marketcast_feeds import funding_on_grid

        step = 60 * HL_INTERVAL_MINUTES[interval]
        per_day = DAY // step
        n = int(minutes)
        out = {f: np.full(n, np.nan) for f in FIELDS}
        if int(t0) % step:
            raise ValueError(f"t0 {t0} is not on the {interval} grid")
        d0 = int(t0) // DAY * DAY
        for day0 in range(d0, int(t0) + step * n, DAY):
            d = self._dir(cls, asset, interval) / _day_key(day0)
            if not d.is_dir():
                continue
            lo = max(int(t0), day0)
            hi = min(int(t0) + step * n, day0 + DAY)
            a, b = (lo - day0) // step, (hi - day0) // step
            o0 = (lo - int(t0)) // step
            for f in FIELDS:
                out[f][o0:o0 + (b - a)] = self._load(d / f"{f}.npy", per_day)[a:b]
        rows = [(int(r["t"]), float(r["rate"]),
                 float("nan") if r.get("premium") is None else float(r["premium"]))
                for r in self._funding_rows(self._dir(cls, asset) / "funding.jsonl")]
        out["funding"], out["premium"] = funding_on_grid(rows, int(t0), n,
                                                         step_minutes=step // 60)
        return out

    def coverage(self, cls: str, asset: str, t0: int, minutes: int) -> float:
        """Share of the grid's minutes with a recorded close."""
        c = self.read(cls, asset, t0, minutes)["close"]
        return float(np.isfinite(c).mean()) if c.size else 0.0


def record_intervals(mc) -> tuple[str, ...]:
    """The candle intervals a recorder pass records: 1m, every configured
    native interval and 1h (in that order, deduplicated)."""
    out = ["1m", *tuple(getattr(mc, "native_intervals", ()) or ()), RECORD_COARSE]
    return tuple(dict.fromkeys(out))


def record_once(store: MktBarStore, mc, *, post, now: float,
                lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
                backfill: bool = True) -> dict:
    """One recorder pass: for every asset of an armed class whose PRIMARY is
    a Hyperliquid source, per interval of :func:`record_intervals` fetch
    the last ``lookback_minutes`` of candles (capped at the API's
    ~5000-candle depth; with ``backfill`` an (asset, interval) nothing was
    recorded for yet fetches that FULL depth — owner 2026-11-10: no lead
    time) and the funding samples, and merge the CLOSED candles. A failing
    asset is skipped (its candles stay NaN until the next pass). Returns
    ``{"<class>:<asset>": 1m candles written | "error: …"}``."""
    from . import marketcast_feeds as F

    if not 0 < int(lookback_minutes) <= HL_1M_DEPTH_MINUTES:
        raise ValueError(f"lookback must be in (0, {HL_1M_DEPTH_MINUTES}] minutes (the API's 1m "
                         "depth)")
    end_ms = int(now) // 60 * 60 * 1000
    out: dict = {}
    am = mc.assets_map
    for cls in mc.armed_classes:
        srcs = mc.sources_for(cls)
        if not srcs or srcs[0] not in ("hyperliquid", "hyperliquid_xyz"):
            continue
        for asset in am.get(cls, ()):
            coin = F.hl_coin(srcs[0], asset)
            try:
                n1 = 0
                for iv in record_intervals(mc):
                    im = HL_INTERVAL_MINUTES[iv]
                    depth = HL_DEPTH_CANDLES * im
                    look = (depth if backfill and not store.has_bars(cls, asset, iv)
                            else min(depth, max(int(lookback_minutes), 2 * im)))
                    n = store.write_bars(cls, asset, F.fetch_hl_candles(
                        post, coin, end_ms - look * MINUTE_MS, end_ms, interval=iv),
                        now=now, interval=iv)
                    if iv == "1m":
                        n1 = n
                start_ms = end_ms - int(lookback_minutes) * MINUTE_MS
                store.write_funding(cls, asset, F.fetch_hl_funding(post, coin, start_ms, end_ms))
                out[f"{cls}:{asset}"] = n1
            except Exception as e:  # noqa: BLE001 — one asset never stops the recorder
                out[f"{cls}:{asset}"] = f"error: {type(e).__name__}: {e}"[:200]
    return out


__all__ = ["DEFAULT_LOOKBACK_MINUTES", "HL_1M_DEPTH_MINUTES", "HL_DEPTH_CANDLES", "MktBarStore",
           "RECORD_COARSE", "record_intervals", "record_once"]
