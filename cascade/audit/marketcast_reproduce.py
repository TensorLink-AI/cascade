"""``cascade-audit marketcast --reproduce`` (DEC-CA-0064 review fixes): check
the main validator's MarketCast inputs and forecasts against the world,
not against its own signed bytes. Reports only — it never changes
consensus (the verdict replay is :func:`cascade.audit.checks.
check_marketcast`).

Two independent re-derivations:

* **Bars.** A deterministic sample of the snapshot's bars is re-fetched from
  Hyperliquid's public ``candleSnapshot`` (no key) and compared on the close
  within ``bar_tol_bp`` basis points. Hyperliquid serves only its most recent
  ~5000 candles per interval (measured 2026-10-19: 1m ≈ 3.6 days, 1h ≈ 208
  days), so the sample reads the coarse HOURLY bank (always within reach for
  half a year) and the minute bank only where the feed still serves it; a
  bar the feed no longer serves is counted, never failed.
* **Forecasts** (torch + the ``timestral`` extra installed). A deterministic
  sample of each entry's windows is re-forecast from the PUBLIC checkpoints
  (the pinned base and the fine-tune) and compared with the published float32
  sidecar: ``|recorded − reproduced| ≤ atol + rtol · |recorded|`` per knot.
  The defaults (``rtol`` 2e-3, ``atol`` 1e-3 in the series' units) absorb
  float32 rounding, cross-GPU nondeterminism and the different NaN
  left-padding of a one-window batch; a fine-tune that does not reproduce
  well beyond that FAILs.

The pure parts (sampling, comparisons, orchestration over injected network /
forecasters) are tested; the network and GPU glue is not covered.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .checks import FAIL, PASS, WARN, CheckResult

DEFAULT_RTOL = 2e-3
DEFAULT_ATOL = 1e-3


def sample_ids(ids, n: int, seed: str) -> list:
    """A deterministic sample of ``n`` of ``ids`` (by the sha256 of
    ``seed|id``), returned in their original order."""
    keyed = sorted(ids, key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())
    keep = set(keyed[:max(0, int(n))])
    return [x for x in ids if x in keep]


def bar_samples(snap, n: int, seed: str) -> list[tuple]:
    """``[(cls, asset, interval, open_s, close), ...]``: up to ``n`` finite
    closes per bank sampled from the snapshot — hourly (``"1h"``) from the
    coarse bank, minute (``"1m"``) from the newest 5000 minute bars."""
    out = []
    banks = [("bars", "1m", snap.t0, 60)]
    c = snap.meta.get("coarse")
    if c:
        banks.append(("coarse", "1h", int(c["t0"]), 3600))
    for bank, interval, t0, step in banks:
        cand = []
        for cls, syms in sorted(snap.meta.get("assets", {}).items()):
            for a in syms:
                close = snap.bars(cls, a, bank=bank).get("close")
                if close is None:
                    continue
                idx = np.flatnonzero(np.isfinite(close))
                if bank == "bars":
                    idx = idx[idx >= close.size - 5000]
                cand += [(cls, a, interval, int(t0 + step * int(i)), float(close[int(i)]))
                         for i in idx]
        out += sample_ids(cand, n, f"{seed}|{bank}")
    return out


def compare_bars(samples, fetch: Callable, *, tol_bp: float) -> CheckResult:
    """``fetch(cls, asset, interval, open_s) -> close | None`` (None = the
    feed no longer serves that bar). FAIL on any close more than ``tol_bp``
    off; WARN when nothing could be compared."""
    worst, n, gone = 0.0, 0, 0
    bad = []
    for cls, a, interval, t, close in samples:
        got = fetch(cls, a, interval, t)
        if got is None or not (got > 0 and close > 0):
            gone += 1
            continue
        n += 1
        dev = abs(1e4 * math.log(float(got) / float(close)))
        worst = max(worst, dev)
        if dev > tol_bp:
            bad.append(f"{cls}:{a} {interval}@{t} {close} vs {got} ({dev:.2f} bp)")
    if bad:
        return CheckResult("marketcast_bars", FAIL, f"{len(bad)}/{n} sampled close(s) off the "
                           f"public feed by > {tol_bp} bp: {bad[:3]}")
    if not n:
        return CheckResult("marketcast_bars", WARN, f"no sampled bar is still served by the "
                           f"feed ({gone} not served)")
    return CheckResult("marketcast_bars", PASS, f"{n} sampled close(s) within {tol_bp} bp "
                       f"(worst {worst:.3f} bp); {gone} no longer served")


def compare_quantiles(recorded, reproduced, *, rtol: float = DEFAULT_RTOL,
                      atol: float = DEFAULT_ATOL) -> tuple[float, bool]:
    """``(worst |Δ| / (atol + rtol·|recorded|), ok)`` over every knot; NaN
    in both is equal, NaN in one is a mismatch."""
    a = np.asarray(recorded, dtype=np.float64)
    b = np.asarray(reproduced, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf"), False
    both = np.isnan(a) & np.isnan(b)
    one = np.isnan(a) ^ np.isnan(b)
    if one.any():
        return float("inf"), False
    with np.errstate(invalid="ignore"):
        r = np.where(both, 0.0, np.abs(a - b) / (atol + rtol * np.abs(a)))
    worst = float(np.nanmax(r)) if r.size else 0.0
    return worst, worst <= 1.0


def hl_fetch(post, snap) -> Callable:  # pragma: no cover — thin glue over the feed parsers
    """The ``fetch`` of :func:`compare_bars` over Hyperliquid (``post`` =
    :func:`cascade.pool.marketcast_feeds.http_post_json`)."""
    from ..pool import marketcast_feeds as F
    from ..shared.config.marketcast import split_sources

    def _fetch(cls, asset, interval, t):
        src = (split_sources(snap.meta.get("sources", {}).get(cls, "")) or ("",))[0]
        if src not in ("hyperliquid", "hyperliquid_xyz"):
            return None
        step = 60 * F.HL_INTERVAL_MINUTES[interval]
        rows = F.parse_hl_candles(post(F.HL_INFO_URL, F.hl_candle_request(
            F.hl_coin(src, asset), t * 1000, (t + step) * 1000 - 1, interval)))
        hit = [r for r in rows if r[0] == t * 1000]
        return hit[0][4] if hit else None

    return _fetch


def reproduce_marketcast(receipt, mc, *, reports: dict, quantiles: dict, snapshots: dict,
                         post=None, forecaster_fn: Callable | None = None, sample: int = 16,
                         rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                         bar_tol_bp: float = 1.0, bar_fetch: Callable | None = None
                         ) -> list[CheckResult]:
    """The ``--reproduce`` checks of one receipt. ``forecaster_fn(pointer |
    "base") -> (forecaster, covariates)`` (None = torch / timestral not
    installed: WARN); ``bar_fetch`` overrides the Hyperliquid fetch (tests)."""
    from ..eval.timestral_adapter import batched_forecasts
    from ..pool.marketcast import snapshot_windows, window_covariates
    from ..shared.marketcast_report import decode_quantiles, load_mkt_report

    out: list[CheckResult] = []
    reps = []
    for key, text in sorted((reports or {}).items()):
        try:
            reps.append((key, load_mkt_report(text)))
        except (ValueError, KeyError, TypeError):
            continue
    snaps = {}
    for _k, rep in reps:
        d = str(rep.snapshot.get("digest", ""))
        if d in (snapshots or {}):
            snaps[d] = snapshots[d]
    if not snaps:
        return [CheckResult("marketcast_reproduce", WARN, "no published snapshot supplied "
                            "(--mkt-snapshots): nothing to reproduce")]
    for d, snap in sorted(snaps.items()):
        fetch = bar_fetch(snap) if bar_fetch is not None else hl_fetch(post, snap)
        out.append(compare_bars(bar_samples(snap, sample, f"bars|{d}"), fetch,
                                tol_bp=bar_tol_bp))
    if forecaster_fn is None:
        out.append(CheckResult("marketcast_reproduce", WARN, "torch / the timestral extra not "
                               "installed: forecasts not re-run"))
        return out
    base = forecaster_fn("base")[0]
    knots = tuple(mc.quantile_knots)
    worst, n, bad = 0.0, 0, []
    for _key, rep in reps:
        snap = snaps.get(str(rep.snapshot.get("digest", "")))
        if snap is None:
            continue
        for e in rep.entries:
            data = quantiles.get(e.quantiles_key) if e.quantiles_key else None
            if e.deferred or e.fault or data is None:
                continue
            ws = snapshot_windows(snap, mc, e.target.split(":", 1)[1], after=float(e.after))
            qs = decode_quantiles(data, [w.steps for w in ws], len(knots))
            ft, covs = forecaster_fn(e.trained_pointer)
            pick = set(sample_ids([w.wid for w in ws], sample, f"{rep.round_id}|{e.hotkey}"))
            for w, (qb, qf) in zip(ws, qs, strict=True):
                if w.wid not in pick:
                    continue
                p, kn = window_covariates(snap, w, tuple(covs),
                                          context_length=int(mc.eval_context_length))
                rb = batched_forecasts(base, [w.context], w.steps, levels=knots)[0]
                rf = batched_forecasts(ft, [w.context], w.steps, levels=knots,
                                       past=None if p is None else [p],
                                       known=None if kn is None else [kn])[0]
                for who, rec, got in (("base", qb, rb), ("fine-tune", qf, rf)):
                    x, ok = compare_quantiles(rec, got, rtol=rtol, atol=atol)
                    worst = max(worst, x)
                    n += 1
                    if not ok:
                        bad.append(f"{e.hotkey[:10]} {w.wid} {who} ({x:.2f}× tolerance)")
    if bad:
        out.append(CheckResult("marketcast_reproduce", FAIL, f"{len(bad)}/{n} re-forecast(s) "
                               f"off the published quantiles: {bad[:3]}"))
    elif not n:
        out.append(CheckResult("marketcast_reproduce", WARN, "no computed entry with a "
                               "sidecar to re-forecast"))
    else:
        out.append(CheckResult("marketcast_reproduce", PASS, f"{n} re-forecast(s) within "
                               f"rtol={rtol} atol={atol} (worst {worst:.2f}× tolerance)"))
    return out


def timestral_forecaster_fn(cfg, *, workdir: Path, device: str = "cpu"
                            ) -> Callable | None:  # pragma: no cover — GPU + the extra
    """``pointer | "base" -> (forecaster, covariates)`` over the PUBLIC
    checkpoints (anonymous Hub pull), or None when torch / timestral are
    not installed."""
    try:
        import torch  # noqa: F401

        from ..eval.timestral_adapter import load_timestral

        load_timestral()
    except Exception:  # noqa: BLE001 — not installed: the caller WARNs
        return None
    from ..eval.timestral_adapter import TimestralQuantileForecaster, load_pinned_base, read_meta
    from ..shared.hippius import fetch_from_hub
    from ..shared.manifest import parse_trained_pointer

    mc = cfg.marketcast
    cache: dict = {}

    def _fn(pointer: str):
        hit = cache.get(pointer)
        if hit is not None:
            return hit
        if pointer == "base":
            fc, _d = load_pinned_base(mc.base_ref, mc.base_sha256, cache_dir=workdir / "_hf",
                                      device=device)
            cache[pointer] = (fc, ())
        else:
            ref = parse_trained_pointer(pointer)
            dest = Path(workdir) / hashlib.sha256(str(ref).encode()).hexdigest()[:16]
            fetch_from_hub(ref, dest)
            cache[pointer] = (TimestralQuantileForecaster.from_dir(dest, device=device),
                              tuple(read_meta(dest).get("covariates", ()) or ()))
        return cache[pointer]

    return _fn


__all__ = ["DEFAULT_ATOL", "DEFAULT_RTOL", "bar_samples", "compare_bars", "compare_quantiles",
           "reproduce_marketcast", "sample_ids", "timestral_forecaster_fn"]
