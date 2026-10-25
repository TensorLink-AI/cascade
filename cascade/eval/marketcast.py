"""MarketCast (DEC-CA-0064) — the pure scoring pieces: series construction
from 1-minute bars, quantile metrics, the coverage gate, knot mapping, the
forward-only window rule and the per-suite paired verdict.

numpy only, no I/O, deterministic in ``seed`` (validators, satellites and
``cascade-audit`` replay it bit-for-bit). Nothing here is reachable from a
live verdict while ``[marketcast] from_block`` is 0 (the shipped value).

Conventions:

* Inputs are 1-minute bars in time order; NaN marks a missing bar (Timestral
  reads NaN as missing). Prices must be > 0 where finite.
* Path quantities (per-block realized vol, range, excursions) are their OWN
  series, never derived from marginal quantiles of returns (the base outputs
  marginals; long horizons autoregress on the median).
* Quantile forecasts are ``(..., K)`` arrays over ascending ``levels``
  (the base's 21 knots by default).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..shared.config.marketcast import COVERAGE_GATE_SUITES, MIN_CONTEXT, TIMESTRAL_KNOTS

BP = 1e4


def _prices(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got {a.shape}")
    fin = np.isfinite(a)
    if np.any(a[fin] <= 0.0):
        raise ValueError(f"{name} must be > 0 where finite")
    return a


def _step(h: int, name: str = "horizon") -> int:
    if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or int(h) < 1:
        raise ValueError(f"{name} must be a positive integer number of bars; got {h!r}")
    return int(h)


# ── series construction (from 1-minute bars) ────────────────────────────────


def log_returns_bp(close, horizon: int = 1) -> np.ndarray:
    """Non-overlapping log returns over ``horizon`` bars, in basis points:
    ``1e4 · (ln c[(i+1)h] − ln c[ih])`` for the closes sampled every
    ``horizon`` bars. NaN where either end is missing."""
    c = _prices(close, "close")
    h = _step(horizon)
    return BP * np.diff(np.log(c[::h]))


def realized_vol_bp(close, block: int) -> np.ndarray:
    """Realized vol per consecutive block of ``block`` one-minute returns:
    ``std(ddof=1)`` of the finite 1m log returns (bp) in the block. A block
    with fewer than 2 finite returns is NaN (skipped — missing to the base);
    a trailing partial block is dropped. One value per full block."""
    c = _prices(close, "close")
    b = _step(block, "block")
    r = BP * np.diff(np.log(c))
    n = r.size // b
    out = np.full(n, np.nan)
    for i in range(n):
        seg = r[i * b:(i + 1) * b]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2:
            out[i] = float(np.std(seg, ddof=1))
    return out


def range_excursions(open_, high, low, horizon: int) -> dict[str, np.ndarray]:
    """Per consecutive block of ``horizon`` bars (trailing partial dropped),
    in natural-log units relative to the block's first open ``o``:

    * ``range`` = ``ln(max high / min low)``      (≥ 0)
    * ``up``    = ``ln(max high / o)``            max up excursion
    * ``down``  = ``ln(o / min low)``             max down excursion (≥ 0 magnitude)

    Missing bars are ignored inside a block; a block with no finite bar (or
    a missing first open) is NaN. CRPS is taken on ``log(range)``."""
    o = _prices(open_, "open")
    hi = _prices(high, "high")
    lo = _prices(low, "low")
    if not (o.shape == hi.shape == lo.shape):
        raise ValueError("open/high/low must have the same length")
    h = _step(horizon)
    n = o.size // h
    rng, up, dn = (np.full(n, np.nan) for _ in range(3))
    for i in range(n):
        s = slice(i * h, (i + 1) * h)
        hh, ll, o0 = hi[s], lo[s], o[i * h]
        hh, ll = hh[np.isfinite(hh)], ll[np.isfinite(ll)]
        if not (hh.size and ll.size and np.isfinite(o0)):
            continue
        mh, ml = float(hh.max()), float(ll.min())
        if mh < ml:
            raise ValueError(f"block {i}: max high {mh} < min low {ml}")
        rng[i] = np.log(mh / ml)
        up[i] = max(0.0, np.log(mh / o0))
        dn[i] = max(0.0, np.log(o0 / ml))
    return {"range": rng, "up": up, "down": dn}


def log_volume(volume, block: int) -> np.ndarray:
    """``ln(sum volume)`` per consecutive block of ``block`` bars (missing
    bars ignored, trailing partial dropped). A block with zero or no volume
    is NaN (log undefined; missing to the base)."""
    v = np.asarray(volume, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"volume must be 1-D; got {v.shape}")
    if np.any(v[np.isfinite(v)] < 0.0):
        raise ValueError("volume must be >= 0 where finite")
    b = _step(block, "block")
    n = v.size // b
    out = np.full(n, np.nan)
    for i in range(n):
        seg = v[i * b:(i + 1) * b]
        seg = seg[np.isfinite(seg)]
        tot = float(seg.sum()) if seg.size else 0.0
        if tot > 0.0:
            out[i] = np.log(tot)
    return out


def funding_change_bp(funding, horizon: int) -> np.ndarray:
    """Change in the perp funding rate over ``horizon`` samples, in bp:
    ``1e4 · (f[(i+1)h] − f[ih])`` (non-overlapping). NaN where missing."""
    f = np.asarray(funding, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"funding must be 1-D; got {f.shape}")
    return BP * np.diff(f[::_step(horizon)])


def funding_bp(funding) -> np.ndarray:
    """A funding-rate series (fraction per funding interval) in bp."""
    f = np.asarray(funding, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"funding must be 1-D; got {f.shape}")
    return BP * f


def basis_bp(premium) -> np.ndarray:
    """Perp basis in bp from Hyperliquid's funding ``premium`` (the
    mark-vs-oracle premium the funding formula samples): ``1e4 · premium``.
    The ``mkt:funding`` basis diagnostic; NaN where missing."""
    p = np.asarray(premium, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"premium must be 1-D; got {p.shape}")
    return BP * p


_CS_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread_bp(high, low, block: int) -> np.ndarray:
    """Effective bid-ask spread per block of ``block`` bars from high/low
    prices (Corwin & Schultz 2012, J. Finance 67(2)): for each pair of
    consecutive bars ``β = ln(H₁/L₁)² + ln(H₂/L₂)²``, ``γ = ln(max H / min L)²``,
    ``α = (√(2β) − √β)/k − √(γ/k)`` with ``k = 3 − 2√2``,
    ``S = 2(eᵅ − 1)/(1 + eᵅ)`` floored at 0; the block value is the mean of
    the estimates of the pairs INSIDE the block (``block ≥ 2``), in bp. NaN
    for a block with no finite pair (trailing partial block dropped). The
    ``mkt:liquidity`` spread series."""
    hi = _prices(high, "high")
    lo = _prices(low, "low")
    if hi.shape != lo.shape:
        raise ValueError("high/low must have the same length")
    b = _step(block, "block")
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.log(hi / lo) ** 2
        beta = hl[:-1] + hl[1:]
        gam = np.log(np.maximum(hi[:-1], hi[1:]) / np.minimum(lo[:-1], lo[1:])) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_K - np.sqrt(gam / _CS_K)
        spread = np.maximum(0.0, 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha)))
    n = hi.size // b
    out = np.full(n, np.nan)
    for i in range(n):
        seg = spread[i * b:(i + 1) * b - 1]          # the pairs inside the block
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = BP * float(seg.mean())
    return out


def dollar_volume(close, volume, block: int) -> np.ndarray:
    """``Σ close · volume`` per block of ``block`` bars (missing bars
    ignored; NaN for a block with none)."""
    c = _prices(close, "close")
    v = np.asarray(volume, dtype=np.float64)
    if c.shape != v.shape:
        raise ValueError("close/volume must have the same length")
    b = _step(block, "block")
    dv = c * v
    n = c.size // b
    out = np.full(n, np.nan)
    for i in range(n):
        seg = dv[i * b:(i + 1) * b]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(seg.sum())
    return out


def adv(daily_dollar_volume, window: int) -> np.ndarray:
    """Average daily dollar volume: the trailing mean of the last ``window``
    finite daily values (inclusive), NaN until ``window`` days exist."""
    d = np.asarray(daily_dollar_volume, dtype=np.float64)
    w = _step(window, "window")
    out = np.full(d.size, np.nan)
    for i in range(w - 1, d.size):
        seg = d[i - w + 1:i + 1]
        if np.isfinite(seg).all():
            out[i] = float(seg.mean())
    return out


def alpha_price(tao_in, alpha_in) -> np.ndarray:
    """dTAO subnet alpha price in TAO from the pool reserves:
    ``tao_in / alpha_in`` (the constant-product spot price). NaN where a
    reserve is missing or not > 0."""
    t = np.asarray(tao_in, dtype=np.float64)
    a = np.asarray(alpha_in, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = t / a
    return np.where(np.isfinite(p) & (t > 0) & (a > 0), p, np.nan)


def log_depth(tao_in) -> np.ndarray:
    """Pool depth ``ln(tao_in)`` (TAO side of the subnet pool); NaN ≤ 0."""
    t = np.asarray(tao_in, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, np.log(t), np.nan)


# ── per-suite series at a frequency (from aligned 1-minute bars) ────────────
#
# ``bars`` maps a field to a 1-D array on a 1-minute grid starting at a UTC
# minute ``t0``: open / high / low / close / volume (perps and spot),
# funding / premium (forward-filled hourly funding samples, perps), price
# (alpha). Block ``i`` of a frequency of ``f`` minutes covers minutes
# ``[i·f, (i+1)·f)``; its value is known at the block's END.


def _block_end(x: np.ndarray, f: int) -> np.ndarray:
    n = x.size // f
    return x[f - 1:n * f:f] if n else np.empty(0)


def _lead_nan_diff(x: np.ndarray) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if x.size > 1:
        out[1:] = np.diff(x)
    return out


def suite_series(suite: str, bars: Mapping[str, np.ndarray], freq_minutes: int, *,
                 bar_minutes: int = 1) -> np.ndarray:
    """The series a suite forecasts, one value per ``freq_minutes`` block
    of ``bars`` sampled every ``bar_minutes`` (1 = the 1-minute bank; 60 =
    the coarse hourly bank, where ``vol`` is the std of the block's HOURLY bp
    returns — the same construction for context and target):

    * ``returns`` — log return of the block-end close vs the previous
      block-end close, bp (block 0 NaN);
    * ``vol``     — ``ln`` realized vol (std of the block's 1m bp returns);
    * ``range``   — ``ln`` of ``ln(max H / min L)`` over the block;
    * ``funding`` — change of the block-end funding rate, bp (block 0 NaN);
    * ``liquidity`` — ``ln Σ volume`` over the block;
    * ``alpha``   — log return of the block-end alpha price, bp.

    NaN marks missing. Log targets are NaN where the level is not > 0."""
    f = _step(freq_minutes, "freq_minutes")
    bm = _step(bar_minutes, "bar_minutes")
    if f % bm:
        raise ValueError(f"freq {f}m is not a multiple of the {bm}m bars")
    f //= bm
    with np.errstate(divide="ignore", invalid="ignore"):
        if suite in ("returns", "alpha"):
            px = np.asarray(bars["close" if suite == "returns" else "price"], dtype=np.float64)
            px = np.where(px > 0, px, np.nan)
            return BP * _lead_nan_diff(np.log(_block_end(px, f)))
        if suite == "vol":
            rv = realized_vol_bp(bars["close"], f)
            return np.where(rv > 0, np.log(rv), np.nan)
        if suite == "range":
            r = range_excursions(bars["open"], bars["high"], bars["low"], f)["range"]
            return np.where(r > 0, np.log(r), np.nan)
        if suite == "funding":
            fu = np.asarray(bars["funding"], dtype=np.float64)
            return BP * _lead_nan_diff(_block_end(fu, f))
        if suite == "liquidity":
            return log_volume(bars["volume"], f)
    raise ValueError(f"unknown MarketCast suite {suite!r}")


def window_geometry(freqs: Sequence[str], horizon: str) -> tuple[int, int]:
    """``(freq_minutes, steps)`` a horizon is forecast at: the COARSEST
    configured frequency ``f`` with ``f ≤ h`` that divides ``h``, and
    ``steps = h / f`` (≤ :data:`TIMESTRAL_MAX_HORIZON`, one pass)."""
    from ..shared.config.marketcast import TIMESTRAL_MAX_HORIZON, duration_minutes

    h = duration_minutes(horizon)
    fit = [duration_minutes(f) for f in freqs if duration_minutes(f) <= h
           and h % duration_minutes(f) == 0]
    if not fit:
        raise ValueError(f"no frequency in {list(freqs)} divides horizon {horizon!r}")
    f = max(fit)
    steps = h // f
    if steps > TIMESTRAL_MAX_HORIZON:
        raise ValueError(f"horizon {horizon} at {f}m is {steps} steps > "
                         f"{TIMESTRAL_MAX_HORIZON} (one pass)")
    return f, steps


# ── covariates (inputs, never targets) ───────────────────────────────────────


def _us_eastern_offset_hours(t: float) -> int:
    """UTC offset of US Eastern time at unix ``t``: -4 (EDT) from the second
    Sunday of March 07:00 UTC to the first Sunday of November 06:00 UTC
    (the 2007 US rule), else -5 (EST). Pure — no tz database."""
    from datetime import UTC, datetime

    y = datetime.fromtimestamp(float(t), tz=UTC).year

    def _nth_sunday(month: int, n: int) -> float:
        d1 = datetime(y, month, 1, tzinfo=UTC)
        first = (6 - d1.weekday()) % 7          # weekday(): Mon=0 … Sun=6
        return datetime(y, month, 1 + first + 7 * (n - 1), tzinfo=UTC).timestamp()

    start = _nth_sunday(3, 2) + 7 * 3600
    end = _nth_sunday(11, 1) + 6 * 3600
    return -4 if start <= float(t) < end else -5


def calendar_rows(times) -> np.ndarray:
    """``(4, N)`` known-future calendar covariates at unix ``times`` (block
    starts): sin / cos of the UTC hour of day and of the day of week."""
    t = np.asarray(times, dtype=np.float64)
    hod = (t % 86400.0) / 86400.0
    dow = ((np.floor(t / 86400.0) + 3.0) % 7.0) / 7.0      # 1970-01-01 was a Thursday
    tau = 2.0 * np.pi
    return np.stack([np.sin(tau * hod), np.cos(tau * hod), np.sin(tau * dow), np.cos(tau * dow)])


def session_rows(times) -> np.ndarray:
    """``(3, N)`` known-future session flags at unix ``times`` (block
    starts), US Eastern time with the US DST rule: weekend (Sat / Sun ET),
    the FX rollover hour (17:00–18:00 ET), the US cash-equity session
    (09:30–16:00 ET, Mon–Fri). Exchange holidays are not modelled."""
    out = np.zeros((3, len(times)))
    for j, t in enumerate(np.asarray(times, dtype=np.float64)):
        et = float(t) + 3600.0 * _us_eastern_offset_hours(t)
        dow = int((np.floor(et / 86400.0) + 3) % 7)           # Mon=0 … Sun=6
        mins = (et % 86400.0) / 60.0
        out[0, j] = float(dow >= 5)
        out[1, j] = float(17 * 60 <= mins < 18 * 60)
        out[2, j] = float(dow < 5 and 9 * 60 + 30 <= mins < 16 * 60)
    return out


PAST_COVARIATES = ("cross_asset", "funding", "volume")
KNOWN_COVARIATES = ("calendar", "session")


def build_covariates(names: Sequence[str], *, block_starts, n_context: int,
                     volume_series=None, funding_series=None,
                     cross_series=None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(past, known)`` covariate rows for one window: ``past`` is
    ``(C_p, n_context)``, ``known`` ``(C_f, len(block_starts))`` where
    ``block_starts`` are the unix starts of the context + horizon blocks.
    Rows come in the order of :data:`PAST_COVARIATES` / :data:`KNOWN_COVARIATES`
    (never the recipe's order), so a fine-tune reads its rows identically in
    training and eval. A past series missing for the asset is a NaN row
    (missing to the base). ``None`` when no row of that kind is asked."""
    want = set(names)
    bad = sorted(want - set(PAST_COVARIATES) - set(KNOWN_COVARIATES))
    if bad:
        raise ValueError(f"unknown covariates {bad}")
    past = []
    for name, src in (("cross_asset", cross_series), ("funding", funding_series),
                      ("volume", volume_series)):
        if name in want:
            row = (np.full(n_context, np.nan) if src is None
                   else np.asarray(src, dtype=np.float64))
            if row.shape != (n_context,):
                raise ValueError(f"covariate {name} has {row.shape}, expected ({n_context},)")
            past.append(row)
    known = []
    if "calendar" in want:
        known.append(calendar_rows(block_starts))
    if "session" in want:
        known.append(session_rows(block_starts))
    return (np.stack(past) if past else None,
            np.concatenate(known) if known else None)


# ── forward-only windows over a bar bank ─────────────────────────────────────


@dataclass(frozen=True)
class MktWindow:
    """One forecast window: ``context`` → ``target`` (``steps`` values) of a
    suite series at ``freq_minutes``, forecast from unix ``start`` (the first
    target block's start) to ``end`` (the last target block's end)."""
    wid: str
    cls: str
    asset: str
    suite: str
    horizon: str
    freq_minutes: int
    steps: int
    start: float
    end: float
    index: int                      # first target block on the series
    context: np.ndarray = field(repr=False)
    target: np.ndarray = field(repr=False)


def window_id(cls: str, asset: str, suite: str, horizon: str, start: float) -> str:
    return f"{cls}:{asset}|{suite}|{horizon}|{int(start)}"


def spread_select(n: int, k: int) -> list[int]:
    """``k`` indices of ``range(n)`` spread EVENLY over it (deterministic:
    ``round(i·(n−1)/(k−1))``; all of them when ``n ≤ k``) — review M10: a
    verdict sample spanning the whole eligible period, not its last hours."""
    n, k = int(n), int(k)
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [n - 1] if k == 1 else []
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


def cut_windows(series, *, cls: str, asset: str, suite: str, horizon: str,
                freq_minutes: int, steps: int, t0: float, context_length: int,
                after: float, now: float, min_context: int = MIN_CONTEXT,
                limit: int | None = None, eligible=None,
                stale_context: bool = False) -> list[MktWindow]:
    """Every forward-only window of one series: origins on the ABSOLUTE
    UTC grid (block index ``k`` with ``(t0/60/f + k) % steps == 0``, so a
    window id is stable across snapshots), start strictly after ``after``,
    end ≤ ``now``, at least ``min_context`` finite context values and one
    finite target. ``eligible(start)`` is a further filter on what is known
    at the window's START (the liquidity floor); ``stale_context`` drops a
    window whose last ``min_context`` finite CONTEXT values are all equal (a
    market that stopped printing — known before the window starts). Nothing here ever reads the
    REALISED target's values beyond finiteness (red-team F3, 2026-10-20: a
    filter on the outcome conditions the test set on it — a forecaster that
    knows the filter moves its mass where the filter left the outcomes; a
    zero-scale cell is masked in the verdict instead, :func:`clustered_gain`).
    ``limit`` keeps that many windows spread EVENLY over the eligible span
    (:func:`spread_select`). ``t0`` is the series' block-0 start (a multiple
    of ``freq_minutes`` minutes)."""
    s = np.asarray(series, dtype=np.float64)
    f = _step(freq_minutes, "freq_minutes")
    h = _step(steps, "steps")
    if int(t0) % (60 * f):
        raise ValueError(f"t0 {t0} is not on the {f}-minute grid")
    base = int(t0) // (60 * f)
    out: list[MktWindow] = []
    first = (-base) % h
    for k in range(first, s.size - h + 1, h):
        start = float(t0) + 60.0 * f * k
        end = start + 60.0 * f * h
        if not (start > float(after) and end <= float(now)):
            continue
        ctx = s[max(0, k - int(context_length)):k]
        tgt = s[k:k + h]
        if int(np.isfinite(ctx).sum()) < int(min_context) or not np.isfinite(tgt).any():
            continue
        if stale_context:
            tail = ctx[np.isfinite(ctx)][-max(1, int(min_context)):]
            if float(tail.max() - tail.min()) == 0.0:
                continue
        if eligible is not None and not eligible(start):
            continue
        out.append(MktWindow(window_id(cls, asset, suite, horizon, start), cls, asset, suite,
                             horizon, f, h, start, end, k, ctx, tgt))
    if limit is not None:
        out = [out[i] for i in spread_select(len(out), int(limit))]
    return out


# ── quantile-knot mapping ────────────────────────────────────────────────────


def _levels(levels) -> np.ndarray:
    q = np.asarray(levels, dtype=np.float64)
    if q.ndim != 1 or q.size < 2 or np.any(np.diff(q) <= 0) or q[0] <= 0 or q[-1] >= 1:
        raise ValueError(f"levels must be >= 2 strictly increasing values in (0, 1); got {q}")
    return q


def map_quantiles(q_pred, src_levels, dst_levels) -> np.ndarray:
    """Map quantile forecasts ``(..., K_src)`` on ``src_levels`` onto
    ``dst_levels`` ``(..., K_dst)``: crossing values are first sorted (the
    monotone rearrangement), then linearly interpolated in level space. A
    destination level outside ``[src_levels[0], src_levels[-1]]`` is clamped
    to the end knot (flat — no tail extrapolation). Exact on shared levels,
    so mapping onto a superset and back is the identity."""
    src = _levels(src_levels)
    dst = np.asarray(dst_levels, dtype=np.float64)
    q = np.sort(np.asarray(q_pred, dtype=np.float64), axis=-1)
    if q.shape[-1] != src.size:
        raise ValueError(f"q_pred last axis {q.shape[-1]} != {src.size} source levels")
    flat = q.reshape(-1, src.size)
    out = np.empty((flat.shape[0], dst.size))
    for i, row in enumerate(flat):
        out[i] = np.interp(dst, src, row)
    return out.reshape(*q.shape[:-1], dst.size)


def to_repo_levels(q_timestral) -> np.ndarray:
    """Timestral's 21 knots → the repo's 9-level grid (0.1 … 0.9, a subset:
    exact)."""
    from .crps import DEFAULT_QUANTILE_LEVELS

    return map_quantiles(q_timestral, TIMESTRAL_KNOTS, DEFAULT_QUANTILE_LEVELS)


# ── metrics ─────────────────────────────────────────────────────────────────


def pinball(q_pred, y, levels) -> np.ndarray:
    """Pinball loss per knot: ``q_pred`` ``(..., K)``, ``y`` ``(...)`` →
    ``(..., K)``. NaN where ``y`` is NaN."""
    q = _levels(levels)
    qp = np.asarray(q_pred, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)[..., None]
    d = yy - qp
    return np.maximum(q * d, (q - 1.0) * d)


def crps_quantile(q_pred, y, levels=TIMESTRAL_KNOTS) -> np.ndarray:
    """CRPS approximated from quantiles: ``2 · mean_k pinball_k`` (the
    repo's GIFT-Eval convention in :mod:`.crps`, unnormalised — the target
    keeps its units, so returns in bp give CRPS in bp). Uniform knot weights:
    on Timestral's uneven grid the 0.01 / 0.99 tails carry 1/21 each. Shape
    ``(...)``; NaN where ``y`` is NaN."""
    return 2.0 * pinball(q_pred, y, levels).mean(axis=-1)


def window_crps(q_pred, y, levels=TIMESTRAL_KNOTS) -> np.ndarray:
    """Per-window CRPS: ``q_pred`` ``(N, H, K)``, ``y`` ``(N, H)`` → ``(N,)``,
    the mean over the finite horizon steps (NaN for a window with none)."""
    c = crps_quantile(q_pred, y, levels)
    with np.errstate(invalid="ignore"):
        fin = np.isfinite(c)
        tot = np.where(fin, c, 0.0).sum(axis=-1)
        cnt = fin.sum(axis=-1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def qlike(vol_true, vol_pred) -> np.ndarray:
    """QLIKE on variances from vol inputs (Patton 2011, the robust-to-noise
    loss for vol forecasts): with ``x = σ²_true / σ²_pred``,
    ``QLIKE = x − ln x − 1`` ≥ 0, zero iff equal. NaN where either side is
    missing or not > 0. Scored on the forecast MEDIAN of realized vol."""
    t = np.asarray(vol_true, dtype=np.float64)
    p = np.asarray(vol_pred, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (t * t) / (p * p)
        out = x - np.log(x) - 1.0
    ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    return np.where(ok, out, np.nan)


def interval_coverage(q_pred, y, levels=TIMESTRAL_KNOTS,
                      coverage_levels: Sequence[float] = (0.5, 0.8, 0.98)) -> dict[float, float]:
    """Empirical coverage of the central intervals: for nominal ``c`` the
    bounds are the forecast quantiles at ``(1 − c)/2`` and ``(1 + c)/2``
    (interpolated from the knots by :func:`map_quantiles`), coverage = the
    share of finite ``y`` with ``lo ≤ y ≤ hi``. NaN when no finite ``y``."""
    yy = np.asarray(y, dtype=np.float64)
    out: dict[float, float] = {}
    fin = np.isfinite(yy)
    for c in coverage_levels:
        b = map_quantiles(q_pred, levels, ((1.0 - c) / 2.0, (1.0 + c) / 2.0))
        lo, hi = b[..., 0], b[..., 1]
        inside = (yy >= lo) & (yy <= hi) & fin
        out[float(c)] = float(inside.sum() / fin.sum()) if fin.any() else float("nan")
    return out


def window_coverage_counts(q_pred, y, levels=TIMESTRAL_KNOTS,
                           coverage_levels: Sequence[float] = (0.5, 0.8, 0.98)
                           ) -> tuple[list[int], int]:
    """Per-window coverage tallies ``([inside per level], finite steps)`` —
    the integers the signed rows carry, so coverage re-aggregates exactly
    across windows (:func:`coverage_from_counts`)."""
    yy = np.asarray(y, dtype=np.float64)
    fin = np.isfinite(yy)
    ins = []
    for c in coverage_levels:
        b = map_quantiles(q_pred, levels, ((1.0 - c) / 2.0, (1.0 + c) / 2.0))
        ins.append(int(((yy >= b[..., 0]) & (yy <= b[..., 1]) & fin).sum()))
    return ins, int(fin.sum())


def window_tail_counts(q_pred, y, levels=TIMESTRAL_KNOTS,
                       tail_levels: Sequence[float] = (0.05, 0.95)) -> list[int]:
    """Per-window tail tallies (red-team F1b): per tail level ``l`` the
    finite steps with ``y < q_l`` (``l < 0.5``) or ``y > q_l`` (``l > 0.5``)
    — a calibrated forecaster hits each with rate ``min(l, 1 − l)``."""
    yy = np.asarray(y, dtype=np.float64)
    fin = np.isfinite(yy)
    out = []
    for lv in tail_levels:
        q = map_quantiles(q_pred, levels, (float(lv),))[..., 0]
        hit = (yy < q) if float(lv) < 0.5 else (yy > q)
        out.append(int((hit & fin).sum()))
    return out


def coverage_from_counts(inside: Sequence[Sequence[int]], n: Sequence[int],
                         coverage_levels: Sequence[float]) -> dict[float, float]:
    """Empirical coverage per level from per-window tallies (NaN when no
    finite step)."""
    tot = int(sum(int(x) for x in n))
    out = {}
    for j, c in enumerate(coverage_levels):
        # a row without tallies (an excluded window: cov_n = 0) adds nothing
        k = sum(int(row[j]) for row in inside if len(row) > j)
        out[float(c)] = float(k / tot) if tot else float("nan")
    return out


@dataclass(frozen=True)
class CoverageRow:
    nominal: float
    empirical: float
    under: bool          # empirical < nominal − tolerance  (fails a gated level)
    over: bool           # empirical > nominal + tolerance_over (fails a gated level when
    #                      tolerance_over is set — review M14; else reported only)
    horizon: str = ""    # "" = pooled over horizons
    gated: bool = True   # False = a report-only level (e.g. 98 %)
    kind: str = "central"   # "central" (interval coverage) | "tail" (hit rate beyond q_l)
    slack: float = 0.0   # calibration_z × standard error: the tolerance actually used is
    #                      max(tolerance, slack) (red-team F2: thin classes never false-fail)

    @property
    def key(self) -> str:
        """``"<horizon>@<nominal>"`` (central) / ``"<horizon>@tail<level>"``;
        ``"all"`` for a pooled row."""
        return (f"{self.horizon or 'all'}@{'tail' if self.kind == 'tail' else ''}"
                f"{self.nominal}")


@dataclass(frozen=True)
class CoverageGate:
    passed: bool
    tolerance: float
    rows: tuple[CoverageRow, ...]


def coverage_gate(empirical: Mapping[float, float], *, tolerance: float,
                  tolerance_over: float | None = None, horizon: str = "",
                  gate_levels: Sequence[float] | None = None,
                  slack: Mapping[float, float] | None = None, kind: str = "central",
                  expected: Mapping[float, float] | None = None) -> CoverageGate:
    """The calibration gate over one set of levels: a GATED level FAILS when
    its empirical rate is below ``expected − tolerance`` (or missing / NaN)
    or — with ``tolerance_over`` (review M14: intervals too wide are useless
    downstream) — above ``expected + tolerance_over``. ``expected`` defaults
    to the nominal level (central coverage); a tail row expects
    ``min(l, 1 − l)``. ``slack[level]`` (red-team F2: ``calibration_z ×``
    the standard error at the expected rate) widens BOTH tolerances to
    ``max(tolerance, slack)``. Levels outside ``gate_levels`` (default: all)
    are reported only. Without ``tolerance_over`` over-coverage is reported
    only."""
    if not 0.0 < float(tolerance) < 0.5:
        raise ValueError(f"tolerance must be in (0, 0.5); got {tolerance}")
    over_tol = float(tolerance if tolerance_over is None else tolerance_over)
    gl = None if gate_levels is None else {float(x) for x in gate_levels}
    rows = []
    ok = bool(empirical)
    for nom, emp in sorted(empirical.items()):
        e = float(emp)
        want = float((expected or {}).get(nom, nom))
        sl = float((slack or {}).get(nom, 0.0))
        gated = gl is None or float(nom) in gl
        under = not np.isfinite(e) or e < want - max(float(tolerance), sl)
        over = bool(np.isfinite(e) and e > want + max(over_tol, sl))
        rows.append(CoverageRow(float(nom), e, bool(under), over, str(horizon), bool(gated),
                                str(kind), sl))
        if gated:
            ok = ok and not under and not (tolerance_over is not None and over)
    return CoverageGate(bool(ok), float(tolerance), tuple(rows))


def combine_gates(gates: Sequence[CoverageGate]) -> CoverageGate:
    """One gate from per-horizon gates: passes iff every one passes."""
    gates = list(gates)
    return CoverageGate(bool(gates) and all(g.passed for g in gates),
                        float(gates[0].tolerance) if gates else 0.0,
                        tuple(r for g in gates for r in g.rows))


def mase_median(median, y, history, seasonal_period: int = 1) -> float:
    """MASE of the forecast median (used for ``mkt:funding``), via the
    repo's :func:`cascade.eval.mase.mase` on the finite steps."""
    from .mase import mase

    m = np.asarray(median, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    hist = np.asarray(history, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(yy)
    if not ok.any():
        return float("nan")
    return mase(m[ok], yy[ok], hist[np.isfinite(hist)], seasonal_period)


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[order] = np.arange(1, x.size + 1, dtype=np.float64)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def rank_ic(pred, realized) -> float:
    """Cross-sectional Spearman rank IC between predicted medians and
    realized values (DIAGNOSTIC only). ``(T, A)`` panels give the mean over
    rows with ≥ 3 finite assets; a 1-D input is one cross-section."""
    p = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    r = np.atleast_2d(np.asarray(realized, dtype=np.float64))
    if p.shape != r.shape:
        raise ValueError(f"pred {p.shape} != realized {r.shape}")
    ics = []
    for pp, rr in zip(p, r, strict=True):
        ok = np.isfinite(pp) & np.isfinite(rr)
        if ok.sum() < 3:
            continue
        a, b = _avg_ranks(pp[ok]), _avg_ranks(rr[ok])
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            continue
        ics.append(float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)))
    return float(np.mean(ics)) if ics else float("nan")


def rank_auc(pred, realized) -> float:
    """AUC of the predicted median ranking assets above the realized
    cross-sectional median (DIAGNOSTIC only; Mann-Whitney with average ranks
    over the pooled cross-sections). NaN when a class is empty."""
    p = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    r = np.atleast_2d(np.asarray(realized, dtype=np.float64))
    if p.shape != r.shape:
        raise ValueError(f"pred {p.shape} != realized {r.shape}")
    scores, labels = [], []
    for pp, rr in zip(p, r, strict=True):
        ok = np.isfinite(pp) & np.isfinite(rr)
        if ok.sum() < 2:
            continue
        scores.append(pp[ok])
        labels.append(rr[ok] > np.median(rr[ok]))
    if not scores:
        return float("nan")
    s, lab = np.concatenate(scores), np.concatenate(labels)
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _avg_ranks(s)
    return float((ranks[lab].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ── forward-only windows ────────────────────────────────────────────────────


def window_eligible(submission_time: float, start: float, end: float,
                    now: float | None = None) -> bool:
    """A window is eligible for a submission iff its forecast START is
    strictly after the submission's block time (no look-ahead on public
    market data) and, when ``now`` is given, its END has been observed
    (``end ≤ now``) so it can be scored."""
    if end < start:
        raise ValueError(f"window end {end} < start {start}")
    return bool(start > submission_time and (now is None or end <= now))


def eligible_mask(submission_time: float, starts, ends, now: float | None = None) -> np.ndarray:
    """Vectorised :func:`window_eligible`."""
    s = np.asarray(starts, dtype=np.float64)
    e = np.asarray(ends, dtype=np.float64)
    if s.shape != e.shape:
        raise ValueError("starts / ends shape mismatch")
    if np.any(e < s):
        raise ValueError("a window ends before it starts")
    m = s > float(submission_time)
    if now is not None:
        m &= e <= float(now)
    return m


# ── the paired verdict ──────────────────────────────────────────────────────


@dataclass
class SuiteVerdict:
    suite: str
    passed: bool
    reason: str
    n: int
    n_clusters: int
    gain: float = float("nan")          # point 1 − geomean_(class, horizon) mean ft / mean base
    gain_lcb: float = float("nan")
    coverage: CoverageGate | None = None
    diagnostics: dict = field(default_factory=dict)


def cell_sums(base_loss, ft_loss, *, clusters: Sequence, horizons: Sequence, weights=None,
              zero_scale_frac: float = 1e-3) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    """Red-team F1/F3 (2026-10-20): per CELL (one bootstrap cluster = asset
    class × time block, × one horizon) the weighted SUMS of the base's and
    the fine-tune's losses, ``(Sb, Sf)`` of shape ``(clusters, horizons)``.
    A cell whose base mean loss is ≤ ``zero_scale_frac`` × the median base
    mean of its horizon's cells (or not > 0) is MASKED — zeroed and returned
    by key (``"<cluster>|<horizon>"``): a ratio over ~0 is noise (the
    DEC-CA-0009 lesson), and masking reads only the BASE's loss, never the
    candidate's forecast. Returns ``(Sb, Sf, cluster labels, horizon labels,
    masked keys)``."""
    b = np.asarray(base_loss, dtype=np.float64)
    f = np.asarray(ft_loss, dtype=np.float64)
    w = np.ones(b.size) if weights is None else np.asarray(weights, dtype=np.float64)
    cl: dict = {}
    hl: dict = {}
    cc = np.array([cl.setdefault(c, len(cl)) for c in clusters], dtype=np.int64)
    hc = np.array([hl.setdefault(h, len(hl)) for h in horizons], dtype=np.int64)
    g, nh = len(cl), len(hl)
    Sb, Sf, W = (np.zeros((g, nh)) for _ in range(3))
    if b.size:
        np.add.at(Sb, (cc, hc), w * b)
        np.add.at(Sf, (cc, hc), w * f)
        np.add.at(W, (cc, hc), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_b = np.where(W > 0, Sb / np.where(W > 0, W, 1.0), np.nan)
    keep = W > 0
    for j in range(nh):
        col = mean_b[:, j][keep[:, j]]
        med = float(np.median(col)) if col.size else 0.0
        keep[:, j] &= (Sb[:, j] > 0) & (mean_b[:, j] > float(zero_scale_frac) * med)
    labels_c, labels_h = list(cl), list(hl)
    masked = sorted(f"{labels_c[i]}|{labels_h[j]}" for i in range(g) for j in range(nh)
                    if W[i, j] > 0 and not keep[i, j])
    return np.where(keep, Sb, 0.0), np.where(keep, Sf, 0.0), labels_c, labels_h, masked


def _class_of(cluster) -> str:
    return str(cluster).split("|", 1)[0]


def clustered_gain(base_loss, ft_loss, *, clusters: Sequence, horizons: Sequence,
                   weights=None, alpha: float, B: int, seed: str,
                   zero_scale_frac: float = 1e-3, ratio_cap: float = 4.0
                   ) -> tuple[float, float]:
    """``(point gain, LCB_α)`` — see :func:`clustered_gain_detail`."""
    return clustered_gain_detail(base_loss, ft_loss, clusters=clusters, horizons=horizons,
                                 weights=weights, alpha=alpha, B=B, seed=seed,
                                 zero_scale_frac=zero_scale_frac, ratio_cap=ratio_cap)[:2]


def clustered_gain_detail(base_loss, ft_loss, *, clusters: Sequence, horizons: Sequence,
                          weights=None, alpha: float, B: int, seed: str,
                          zero_scale_frac: float = 1e-3, ratio_cap: float = 4.0
                          ) -> tuple[float, float, list[str], int]:
    """Red-team F1 (2026-10-20): ``(point gain, LCB_α, masked cell keys,
    kept cells)`` of ``1 − exp(mean_h mean_c ln r_ch)`` where ``r_ch = Σ ft /
    Σ base`` is the ratio of MEAN losses of asset class ``c`` (the cluster
    label's prefix before ``|``; one class when unlabelled) at horizon ``h``,
    over the class's (unmasked, :func:`cell_sums`) clusters — classes equal
    inside a horizon, horizons equal (review M10/M13), ``r`` clipped to
    ``[1/ratio_cap, ratio_cap]``.

    Mean CRPS is a proper score, so inside ``(c, h)`` a forecaster's best
    play is its true belief; only the class's pooled ratio is logged (the
    DEC-CA-0009 shape — never the log of one window's CRPS, which rewarded a
    too-narrow forecast against an oracle base: +2.7–4.5 %). The ratio pools
    every time block of the class: per-(class × block) ratios over ~3 windows
    (thin classes, sparse horizons) kept a small-sample Jensen bias that
    still favoured sharpening (+0.4 % at 24 windows / asset / horizon).

    LCB: a cluster bootstrap — whole clusters (class × time block) resampled
    with replacement, the SAME bag for every horizon, each ``r_ch``
    recomputed as a ratio of the bag's sums. Percentile LCB. NaN when
    empty."""
    from .bootstrap import _seed_to_int

    if not np.asarray(base_loss).size:
        return float("nan"), float("nan"), [], 0
    Sb, Sf, labels, _h, masked = cell_sums(base_loss, ft_loss, clusters=clusters,
                                           horizons=horizons, weights=weights,
                                           zero_scale_frac=zero_scale_frac)
    g, nh = Sb.shape
    by_cls: dict[str, list[int]] = {}
    for i, c in enumerate(labels):
        by_cls.setdefault(_class_of(c), []).append(i)
    cap = float(np.log(float(ratio_cap)))

    def _stat(Mw: np.ndarray) -> np.ndarray:
        per_h = []
        for j in range(nh):
            rows = []
            for idx in by_cls.values():
                sb = Mw[:, idx] @ Sb[idx, j]
                sf = Mw[:, idx] @ Sf[idx, j]
                with np.errstate(invalid="ignore", divide="ignore"):
                    lr = np.log(np.maximum(sf, 0.0)) - np.log(np.where(sb > 0, sb, 1.0))
                rows.append(np.where(sb > 0, np.clip(lr, -cap, cap), np.nan))
            r = np.stack(rows)
            ok = np.isfinite(r)
            per_h.append(np.where(ok.any(axis=0), np.where(ok, r, 0.0).sum(axis=0)
                                  / np.maximum(ok.sum(axis=0), 1), np.nan))
        h = np.stack(per_h)
        ok = np.isfinite(h)
        return np.where(ok.any(axis=0), np.where(ok, h, 0.0).sum(axis=0)
                        / np.maximum(ok.sum(axis=0), 1), np.nan)

    point = float(1.0 - np.exp(_stat(np.ones((1, g)))[0]))
    rng = np.random.default_rng(_seed_to_int(seed))
    idx = rng.integers(0, g, size=(int(B), g))
    Mw = np.zeros((int(B), g))
    np.add.at(Mw, (np.repeat(np.arange(int(B)), g), idx.ravel()), 1.0)
    boot = 1.0 - np.exp(_stat(Mw))
    boot = boot[np.isfinite(boot)]
    lcb = float(np.quantile(boot, alpha)) if boot.size else float("nan")
    return point, lcb, masked, int((Sb > 0).sum())


def judge_suite(suite: str, base_loss, ft_loss, *, alpha: float, B: int, gain_margin: float,
                seed: str, clusters: Sequence | None = None, min_windows: int = 1,
                min_clusters: int = 1, coverage: CoverageGate | None = None,
                diagnostics: dict | None = None, horizons: Sequence | None = None,
                weights=None, time_blocks: Sequence | None = None,
                min_time_blocks: int = 1, zero_scale_frac: float = 1e-3,
                ratio_cap: float = 4.0) -> SuiteVerdict:
    """Paired verdict for one suite: the fine-tune vs the untouched base on
    the SAME eligible windows (per-window losses, lower better — CRPS in bp
    for returns, CRPS on log RV / log range / log volume, …).

    Statistic (red-team F1, 2026-10-20): :func:`clustered_gain_detail` — per
    (asset class × horizon) the ratio of MEAN losses, geomean over classes
    (equal inside a horizon) and horizons (equal), its one-sided ``LCB_α``
    from a bootstrap over whole clusters. ``clusters`` None = one cluster,
    ``horizons`` None = one horizon. Masked zero-scale cells are recorded in
    ``diagnostics["masked_cells"]``.

    Pass iff ``LCB ≥ gain_margin`` (> 0 — an untouched base never passes),
    the point gain is > 0, there are ≥ ``min_windows`` paired finite windows
    over ≥ ``min_clusters`` clusters, and — for a calibration-gated suite
    (:data:`COVERAGE_GATE_SUITES`) — the fine-tune's :class:`CoverageGate`
    passed."""
    if not 0.0 < float(gain_margin) < 1.0:
        raise ValueError(f"gain_margin must be in (0, 1); got {gain_margin}")
    b = np.asarray(base_loss, dtype=np.float64)
    f = np.asarray(ft_loss, dtype=np.float64)
    if b.shape != f.shape or b.ndim != 1:
        raise ValueError(f"paired 1-D losses required; got {b.shape} vs {f.shape}")
    ok = np.isfinite(b) & np.isfinite(f) & (b >= 0) & (f >= 0)
    cl = None if clusters is None else [c for c, m in zip(clusters, ok, strict=True) if m]
    hz = None if horizons is None else [h for h, m in zip(horizons, ok, strict=True) if m]
    tb = None if time_blocks is None else {t for t, m in zip(time_blocks, ok, strict=True) if m}
    wt = None if weights is None else np.asarray(weights, dtype=np.float64)[ok]
    b, f = b[ok], f[ok]
    n = int(b.size)
    gated = suite in COVERAGE_GATE_SUITES
    diag = dict(diagnostics or {})
    g = 1 if cl is None else len(set(cl))

    def _v(passed, reason, **kw):
        return SuiteVerdict(suite, bool(passed), reason, n, int(g), coverage=coverage,
                            diagnostics=diag, **kw)

    if n < max(1, int(min_windows)):
        return _v(False, f"inconclusive: {n} windows < {min_windows}")
    if g < max(1, int(min_clusters)):
        return _v(False, f"inconclusive: {g} clusters < {min_clusters}")
    if tb is not None and len(tb) < max(1, int(min_time_blocks)):
        return _v(False, f"inconclusive: {len(tb)} time blocks < {min_time_blocks}")
    gain, lcb, masked, kept = clustered_gain_detail(
        b, f, clusters=cl if cl is not None else ["all"] * n,
        horizons=hz if hz is not None else [""] * n, weights=wt, alpha=alpha, B=B,
        seed=f"{seed}|mkt:{suite}", zero_scale_frac=zero_scale_frac, ratio_cap=ratio_cap)
    diag["cells"] = int(kept)
    diag["masked_cells"] = len(masked)
    if masked:
        diag["masked"] = masked[:8]
    if gated and coverage is None:
        return _v(False, "coverage gate missing", gain=gain, gain_lcb=lcb)
    if not (np.isfinite(lcb) and lcb >= gain_margin):
        return _v(False, f"gain LCB {lcb:.4g} < margin {gain_margin}", gain=gain, gain_lcb=lcb)
    if not gain > 0.0:
        return _v(False, "non-positive point gain", gain=gain, gain_lcb=lcb)
    if gated and not coverage.passed:
        bad = [f"{r.key}{'<' if r.under else '>'}" for r in coverage.rows
               if r.gated and (r.under or r.over)]
        return _v(False, f"coverage gate failed at {bad}", gain=gain, gain_lcb=lcb)
    return _v(True, "win", gain=gain, gain_lcb=lcb)


__all__ = [
    "MIN_CONTEXT", "KNOWN_COVARIATES", "MktWindow", "PAST_COVARIATES", "adv", "alpha_price", "basis_bp",
    "build_covariates", "calendar_rows", "corwin_schultz_spread_bp", "cut_windows",
    "dollar_volume", "funding_bp", "log_depth", "session_rows", "suite_series",
    "window_geometry", "window_id", "window_coverage_counts", "coverage_from_counts",
    "window_tail_counts", "cell_sums", "clustered_gain_detail",
    "BP", "CoverageGate", "CoverageRow", "SuiteVerdict", "clustered_gain", "combine_gates",
    "coverage_gate", "crps_quantile", "spread_select",
    "eligible_mask", "funding_change_bp", "interval_coverage", "judge_suite", "log_returns_bp",
    "log_volume", "map_quantiles", "mase_median", "pinball", "qlike", "range_excursions",
    "rank_auc", "rank_ic", "realized_vol_bp", "to_repo_levels", "window_crps",
    "window_eligible",
]
