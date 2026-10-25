"""MarketCast forward bar recorder (DEC-CA-0064 review M15, 2026-10-19).

Hyperliquid's ``candleSnapshot`` serves only its most recent ~5000 candles
per interval (measured 2026-10-19: 1m ≈ 3.6 days), so the 21-day minute grid
of an eval snapshot can NOT be fetched after the fact. The main validator
therefore RECORDS bars forward, as they close, with ``cascade-pool
mkt-record`` (cron'd every few minutes), and ``cascade-pool publish --lane
mkt --bar-store DIR`` builds the snapshot's minute bank from the recording:
``snapshot_days`` then means recorded history.

* **Resumable.** Every run re-fetches the last ``lookback`` (default 48 h,
  well inside the API depth) and merges: a recorder down for less than the
  API depth loses nothing.
* **Gap-tolerant.** A minute nobody recorded stays NaN — never
  interpolated; the windows over it are simply not cut.
* **Closed bars only.** A candle is written once its minute has ended
  (``open + 60 s ≤ now``), so an in-progress candle never lands.

Layout::

    <root>/<class>/<asset>/<YYYY-MM-DD>/<field>.npy   float64[1440], NaN = missing
    <root>/<class>/<asset>/funding.jsonl              {"t", "rate", "premium"} samples

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

from .marketcast_feeds import FIELDS, MINUTE_MS

DAY = 86400
DEFAULT_LOOKBACK_MINUTES = 48 * 60
HL_1M_DEPTH_MINUTES = 5000          # measured 2026-10-19: candleSnapshot's 1m depth


def _day_key(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=UTC).strftime("%Y-%m-%d")


class MktBarStore:
    """The recorder's on-disk bank (see the module docstring)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, cls: str, asset: str) -> Path:
        return self.root / cls / asset

    def write_bars(self, cls: str, asset: str, candles, *, now: float) -> int:
        """Merge closed 1m candles ``[(open_ms, o, h, l, c, v), ...]``;
        returns the minutes written. Atomic per day file."""
        by_day: dict[str, list] = {}
        for t, o, h, lo, c, v in candles:
            t_s = int(t) // 1000
            if t_s % 60 or t_s + 60 > float(now):
                continue                                   # off-grid / still open
            by_day.setdefault(_day_key(t_s), []).append((t_s, (o, h, lo, c, v)))
        n = 0
        for day, rows in sorted(by_day.items()):
            d = self._dir(cls, asset) / day
            d.mkdir(parents=True, exist_ok=True)
            day0 = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
            arrs = {f: self._load(d / f"{f}.npy") for f in FIELDS}
            for t_s, vals in rows:
                k = (t_s - day0) // 60
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
    def _load(p: Path) -> np.ndarray:
        if p.is_file():
            a = np.load(p, allow_pickle=False)
            if a.shape == (1440,):
                return a.astype(np.float64)
        return np.full(1440, np.nan)

    @staticmethod
    def _funding_rows(p: Path) -> list[dict]:
        if not p.is_file():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def read(self, cls: str, asset: str, t0: int, minutes: int) -> dict[str, np.ndarray]:
        """``{open, high, low, close, volume, funding, premium}`` on the minute
        grid ``[t0, t0 + 60·minutes)`` — NaN wherever nothing was recorded
        (funding: the latest recorded sample at or before each minute)."""
        from .marketcast_feeds import funding_on_grid

        out = {f: np.full(int(minutes), np.nan) for f in FIELDS}
        if int(t0) % 60:
            raise ValueError(f"t0 {t0} is not a minute boundary")
        d0 = int(t0) // DAY * DAY
        for day0 in range(d0, int(t0) + 60 * int(minutes), DAY):
            d = self._dir(cls, asset) / _day_key(day0)
            if not d.is_dir():
                continue
            lo = max(int(t0), day0)
            hi = min(int(t0) + 60 * int(minutes), day0 + DAY)
            a, b = (lo - day0) // 60, (hi - day0) // 60
            o0 = (lo - int(t0)) // 60
            for f in FIELDS:
                out[f][o0:o0 + (b - a)] = self._load(d / f"{f}.npy")[a:b]
        rows = [(int(r["t"]), float(r["rate"]),
                 float("nan") if r.get("premium") is None else float(r["premium"]))
                for r in self._funding_rows(self._dir(cls, asset) / "funding.jsonl")]
        out["funding"], out["premium"] = funding_on_grid(rows, int(t0), int(minutes))
        return out

    def coverage(self, cls: str, asset: str, t0: int, minutes: int) -> float:
        """Share of the grid's minutes with a recorded close."""
        c = self.read(cls, asset, t0, minutes)["close"]
        return float(np.isfinite(c).mean()) if c.size else 0.0


def record_once(store: MktBarStore, mc, *, post, now: float,
                lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES) -> dict:
    """One recorder pass: for every asset of an armed class whose PRIMARY is
    a Hyperliquid source, fetch the last ``lookback_minutes`` of 1m candles
    and the funding samples, and merge the CLOSED bars. A failing asset is
    skipped (its minutes stay NaN until the next pass). Returns
    ``{"<class>:<asset>": minutes written | "error: …"}``."""
    from . import marketcast_feeds as F

    if not 0 < int(lookback_minutes) <= HL_1M_DEPTH_MINUTES:
        raise ValueError(f"lookback must be in (0, {HL_1M_DEPTH_MINUTES}] minutes (the API's 1m "
                         "depth)")
    end_ms = int(now) // 60 * 60 * 1000
    start_ms = end_ms - int(lookback_minutes) * MINUTE_MS
    out: dict = {}
    am = mc.assets_map
    for cls in mc.armed_classes:
        srcs = mc.sources_for(cls)
        if not srcs or srcs[0] not in ("hyperliquid", "hyperliquid_xyz"):
            continue
        for asset in am.get(cls, ()):
            coin = F.hl_coin(srcs[0], asset)
            try:
                n = store.write_bars(cls, asset, F.fetch_hl_candles(post, coin, start_ms, end_ms),
                                     now=now)
                store.write_funding(cls, asset, F.fetch_hl_funding(post, coin, start_ms, end_ms))
                out[f"{cls}:{asset}"] = n
            except Exception as e:  # noqa: BLE001 — one asset never stops the recorder
                out[f"{cls}:{asset}"] = f"error: {type(e).__name__}: {e}"[:200]
    return out


__all__ = ["DEFAULT_LOOKBACK_MINUTES", "HL_1M_DEPTH_MINUTES", "MktBarStore", "record_once"]
