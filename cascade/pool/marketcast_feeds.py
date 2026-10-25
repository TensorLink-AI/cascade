"""MarketCast feed adapters (DEC-CA-0064) — public market data, no keys.

Pure request builders and response parsers; the network call is an
injectable ``post(url, body) -> json`` / ``get(url) -> json`` so everything
here is unit-tested against hand-written fixtures. Only the pool publisher
(the owner orchestrator) and the MarketCast training leg call these:
validators NEVER call a feed — they read the signed snapshot.

Sources (``[marketcast] data_sources``):

* ``hyperliquid``      Hyperliquid main-dex perps (crypto). ``POST
  https://api.hyperliquid.xyz/info`` with ``{"type": "candleSnapshot",
  "req": {"coin", "interval": "1m", "startTime", "endTime"}}`` → rows
  ``{"t", "T", "s", "i", "o", "c", "h", "l", "v", "n"}`` (prices / volume as
  decimal strings, ``t`` the open time in ms) and ``{"type":
  "fundingHistory", "coin", "startTime", "endTime"}`` → rows ``{"coin",
  "fundingRate", "premium", "time"}`` (hourly).
* ``hyperliquid_xyz``  the same endpoint for the ``xyz`` builder-deployed
  dex: the coin is ``"xyz:<SYMBOL>"`` (FX, metals, commodities, indices /
  ETFs, equities). These are PERP prices — they carry a basis to the cash
  market and drift off-hours; acceptable for a forecasting eval.
* ``binance_spot``     Binance spot 1m klines, the crypto CROSS-CHECK:
  ``GET https://api.binance.com/api/v3/klines?symbol=<SYM>USDT&interval=1m
  &startTime=&endTime=&limit=1000`` → ``[open_ms, o, h, l, c, v, close_ms,
  …]``.
* ``bittensor_alpha``  dTAO subnet pool state sampled by the publisher
  (:func:`read_alpha_state`, the pinned SDK read path
  :meth:`cascade.shared.chain.ChainClient.subnet_economics` uses) and kept as
  rows ``(time, block, netuid, price, tao_in, alpha_in)``.

Response shapes were read 2026-10-19 from the live public endpoints; the
fixtures in ``tests/unit/test_marketcast_pool.py`` are hand-written in
those shapes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("cascade.pool.marketcast")

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
XYZ_DEX = "xyz"
BINANCE_LIMIT = 1000
MINUTE_MS = 60_000
FIELDS = ("open", "high", "low", "close", "volume")

PostFn = Callable[[str, dict], object]
GetFn = Callable[[str], object]


class FeedError(RuntimeError):
    """A feed answered with something we cannot use (shape drift, HTTP)."""


# ── symbols ──────────────────────────────────────────────────────────────────


def hl_coin(source: str, asset: str) -> str:
    """The Hyperliquid coin name: the bare symbol on the main dex, ``xyz:<S>``
    on the ``xyz`` dex."""
    if source == "hyperliquid":
        return str(asset)
    if source == "hyperliquid_xyz":
        return f"{XYZ_DEX}:{asset}"
    raise ValueError(f"{source!r} is not a Hyperliquid source")


def binance_symbol(asset: str) -> str:
    return f"{asset}USDT"


# ── request builders ─────────────────────────────────────────────────────────


def hl_candle_request(coin: str, start_ms: int, end_ms: int, interval: str = "1m") -> dict:
    return {"type": "candleSnapshot",
            "req": {"coin": str(coin), "interval": interval, "startTime": int(start_ms),
                    "endTime": int(end_ms)}}


def hl_funding_request(coin: str, start_ms: int, end_ms: int) -> dict:
    return {"type": "fundingHistory", "coin": str(coin), "startTime": int(start_ms),
            "endTime": int(end_ms)}


def binance_klines_url(symbol: str, start_ms: int, end_ms: int, interval: str = "1m",
                       limit: int = BINANCE_LIMIT) -> str:
    return (f"{BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}"
            f"&startTime={int(start_ms)}&endTime={int(end_ms)}&limit={int(limit)}")


# ── parsers ──────────────────────────────────────────────────────────────────


def _num(x) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise FeedError(f"non-finite value {x!r}")
    return v


def parse_hl_candles(rows) -> list[tuple[int, float, float, float, float, float]]:
    """``[(open_ms, o, h, l, c, v), ...]`` sorted by time (duplicates: the
    last wins). Raises :class:`FeedError` on shape drift."""
    if not isinstance(rows, list):
        raise FeedError(f"candleSnapshot answered {type(rows).__name__}, not a list")
    out: dict[int, tuple] = {}
    for r in rows:
        try:
            t = int(r["t"])
            out[t] = (t, _num(r["o"]), _num(r["h"]), _num(r["l"]), _num(r["c"]), _num(r["v"]))
        except (KeyError, TypeError, ValueError) as e:
            raise FeedError(f"candleSnapshot row {r!r}: {e}") from e
    return [out[k] for k in sorted(out)]


def parse_hl_funding(rows) -> list[tuple[int, float, float]]:
    """``[(time_ms, funding_rate, premium), ...]`` sorted by time."""
    if not isinstance(rows, list):
        raise FeedError(f"fundingHistory answered {type(rows).__name__}, not a list")
    out: dict[int, tuple] = {}
    for r in rows:
        try:
            t = int(r["time"])
            prem = r.get("premium")
            out[t] = (t, _num(r["fundingRate"]), _num(prem) if prem is not None else float("nan"))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise FeedError(f"fundingHistory row {r!r}: {e}") from e
    return [out[k] for k in sorted(out)]


def parse_binance_klines(rows) -> list[tuple[int, float, float, float, float, float]]:
    """``[(open_ms, o, h, l, c, v), ...]`` sorted by time."""
    if not isinstance(rows, list):
        raise FeedError(f"klines answered {type(rows).__name__}, not a list")
    out: dict[int, tuple] = {}
    for r in rows:
        try:
            t = int(r[0])
            out[t] = (t, _num(r[1]), _num(r[2]), _num(r[3]), _num(r[4]), _num(r[5]))
        except (IndexError, TypeError, ValueError) as e:
            raise FeedError(f"klines row {r!r}: {e}") from e
    return [out[k] for k in sorted(out)]


# ── paginated fetches (network injected) ─────────────────────────────────────


HL_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def fetch_hl_candles(post: PostFn, coin: str, start_ms: int, end_ms: int, *,
                     max_pages: int = 10_000, interval: str = "1m") -> list[tuple]:
    """All ``interval`` candles of ``coin`` with open time in ``[start_ms,
    end_ms)``, paging forward from the last returned open time (no
    assumption about a server page size). Stops on an empty or non-advancing
    page. Hyperliquid serves only the most recent ~5000 candles of an
    interval (measured 2026-10-19), so older bars come back missing."""
    out: list[tuple] = []
    cur = int(start_ms)
    step_ms = MINUTE_MS * HL_INTERVAL_MINUTES[interval]
    for _ in range(max_pages):
        if cur >= end_ms:
            break
        page = parse_hl_candles(post(HL_INFO_URL, hl_candle_request(coin, cur, end_ms - 1,
                                                                    interval)))
        page = [r for r in page if cur <= r[0] < end_ms]
        if not page:
            break
        out.extend(page)
        nxt = page[-1][0] + step_ms
        if nxt <= cur:
            break
        cur = nxt
    return out


def fetch_hl_funding(post: PostFn, coin: str, start_ms: int, end_ms: int, *,
                     max_pages: int = 10_000) -> list[tuple]:
    out: list[tuple] = []
    cur = int(start_ms)
    for _ in range(max_pages):
        if cur >= end_ms:
            break
        page = parse_hl_funding(post(HL_INFO_URL, hl_funding_request(coin, cur, end_ms - 1)))
        page = [r for r in page if cur <= r[0] < end_ms]
        if not page:
            break
        out.extend(page)
        nxt = page[-1][0] + 1
        if nxt <= cur:
            break
        cur = nxt
    return out


def fetch_binance_klines(get: GetFn, symbol: str, start_ms: int, end_ms: int, *,
                         max_pages: int = 10_000) -> list[tuple]:
    out: list[tuple] = []
    cur = int(start_ms)
    for _ in range(max_pages):
        if cur >= end_ms:
            break
        page = parse_binance_klines(get(binance_klines_url(symbol, cur, end_ms - 1)))
        page = [r for r in page if cur <= r[0] < end_ms]
        if not page:
            break
        out.extend(page)
        nxt = page[-1][0] + MINUTE_MS
        if nxt <= cur:
            break
        cur = nxt
    return out


# ── onto the 1-minute grid ───────────────────────────────────────────────────


def bars_on_grid(candles: Sequence[tuple], t0: int, minutes: int, *,
                 step_minutes: int = 1) -> dict[str, np.ndarray]:
    """``{open, high, low, close, volume}`` arrays of length ``minutes`` (the
    number of grid steps) on the grid of ``step_minutes`` starting at unix
    second ``t0``. A step with no candle is NaN (missing to the base).
    Candles off the grid are dropped."""
    out = {f: np.full(int(minutes), np.nan) for f in FIELDS}
    base = int(t0) * 1000
    for t, o, h, lo, c, v in candles:
        k, rem = divmod(int(t) - base, MINUTE_MS * int(step_minutes))
        if rem or not 0 <= k < minutes:
            continue
        out["open"][k], out["high"][k], out["low"][k] = o, h, lo
        out["close"][k], out["volume"][k] = c, v
    return out


def funding_on_grid(rows: Sequence[tuple], t0: int, minutes: int, *, step_minutes: int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    """``(funding, premium)`` per grid step: the latest funding sample at
    or before the step (forward-filled; NaN before the first sample)."""
    fu = np.full(int(minutes), np.nan)
    pr = np.full(int(minutes), np.nan)
    base = int(t0) * 1000
    step = MINUTE_MS * int(step_minutes)
    for t, rate, prem in sorted(rows):
        k = (int(t) - base + step - 1) // step              # first step at/after the sample
        if k >= minutes:
            continue
        k = max(0, k)
        fu[k:] = rate
        pr[k:] = prem
    return fu, pr


@dataclass(frozen=True)
class AlphaRow:
    time: int          # unix seconds
    block: int
    netuid: int
    price: float       # TAO per alpha
    tao_in: float
    alpha_in: float


def alpha_on_grid(rows: Sequence[AlphaRow], t0: int, minutes: int, *,
                  step_minutes: int = 1) -> dict[str, np.ndarray]:
    """``{price, tao_in, alpha_in}`` per grid step: a sample lands in the
    step containing its time (the last sample in a step wins); steps with no
    sample are NaN."""
    out = {k: np.full(int(minutes), np.nan) for k in ("price", "tao_in", "alpha_in")}
    for r in sorted(rows, key=lambda r: (r.time, r.block)):
        k = (int(r.time) - int(t0)) // (60 * int(step_minutes))
        if 0 <= k < minutes:
            out["price"][k], out["tao_in"][k], out["alpha_in"][k] = (
                r.price, r.tao_in, r.alpha_in)
    return out


def cross_check(primary_close, secondary_close, *, tol_bp: float = 50.0) -> dict:
    """Primary vs cross-check closes on the same grid: the minutes both have,
    the share deviating by more than ``tol_bp`` and the median |deviation|
    in bp. A report, never a gate (perp vs spot carries a basis)."""
    a = np.asarray(primary_close, dtype=np.float64)
    b = np.asarray(secondary_close, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    n = int(ok.sum())
    if not n:
        return {"n": 0, "share_over_tol": None, "median_abs_bp": None, "tol_bp": tol_bp}
    dev = np.abs(1e4 * np.log(a[ok] / b[ok]))
    return {"n": n, "share_over_tol": float((dev > tol_bp).mean()),
            "median_abs_bp": float(np.median(dev)), "tol_bp": float(tol_bp)}


# ── dTAO alpha (pinned SDK read path) ────────────────────────────────────────


def read_alpha_state(client, netuid: int) -> AlphaRow | None:  # pragma: no cover — chain
    """One live sample of subnet ``netuid``'s pool through the pinned SDK:
    the spot price from ``subtensor.prices.alpha_price(netuid)["tao_per_alpha"]``
    and the reserves from the ``SubnetInfoRuntimeApi.get_dynamic_info``
    runtime call (the calls :meth:`cascade.shared.chain.ChainClient.
    subnet_economics` makes). The reserve keys ``tao_in`` / ``alpha_in`` are
    NOT verified against SDK 11.3.0 here: a missing key reads NaN. None when
    the read fails."""
    import time as _time

    from ..shared.chain import _import_bittensor

    rao = 1e9
    try:
        bt = _import_bittensor()
        sub = client.subtensor()
        block = int(sub.block)
        price = float(sub.prices.alpha_price(int(netuid))["tao_per_alpha"])
        try:
            dyn = sub.runtime(bt.runtime_api.SubnetInfoRuntimeApi.get_dynamic_info,
                              [int(netuid)])
            tao_in = float(int(dyn["tao_in"]) / rao) if "tao_in" in dyn else float("nan")
            alpha_in = float(int(dyn["alpha_in"]) / rao) if "alpha_in" in dyn else float("nan")
        except Exception:  # noqa: BLE001
            tao_in = alpha_in = float("nan")
        return AlphaRow(int(_time.time()), block, int(netuid), price, tao_in, alpha_in)
    except Exception as e:  # noqa: BLE001
        log.warning("alpha sample SN%d failed: %s", int(netuid), e)
        return None


def alpha_netuid(asset: str) -> int:
    """``"SN64"`` → 64."""
    s = str(asset)
    if not (s.upper().startswith("SN") and s[2:].isdigit()):
        raise ValueError(f"alpha asset {asset!r} must be SN<netuid>")
    return int(s[2:])


def load_alpha_rows(path) -> list[AlphaRow]:
    """Read an alpha recorder file (JSON lines of :class:`AlphaRow`)."""
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(AlphaRow(**json.loads(line)))
    except FileNotFoundError:
        return []
    return out


def append_alpha_row(path, row: AlphaRow) -> None:
    from dataclasses import asdict

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(row), sort_keys=True) + "\n")


# ── production network (publisher / training leg only) ──────────────────────


def http_post_json(url: str, body: dict, *, timeout: float = 30.0):  # pragma: no cover — network
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:     # noqa: S310 — fixed https URLs
        return json.loads(r.read().decode("utf-8"))


def http_get_json(url: str, *, timeout: float = 30.0):  # pragma: no cover — network
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as r:     # noqa: S310 — fixed https URLs
        return json.loads(r.read().decode("utf-8"))


__all__ = [
    "AlphaRow", "BINANCE_KLINES_URL", "FIELDS", "FeedError", "HL_INFO_URL",
    "HL_INTERVAL_MINUTES", "alpha_netuid",
    "alpha_on_grid", "append_alpha_row", "bars_on_grid", "binance_klines_url", "binance_symbol",
    "cross_check", "fetch_binance_klines", "fetch_hl_candles", "fetch_hl_funding",
    "funding_on_grid", "hl_candle_request", "hl_coin", "hl_funding_request", "http_get_json",
    "http_post_json", "load_alpha_rows", "parse_binance_klines", "parse_hl_candles",
    "parse_hl_funding", "read_alpha_state",
]
