"""MarketCast pool (DEC-CA-0064): the signed, content-addressed snapshot of
realized 1-minute bars that every MarketCast verdict is cut from.

Validators never call a market feed. The owner orchestrator (``cascade-pool
publish --lane mkt``) fetches the last ``[marketcast] snapshot_days`` of bars
for every configured asset from the class's primary source (cross-checks
reported), writes them onto one UTC-minute grid and signs the snapshot with
the main validator's hotkey (``[netconfig] owner_hotkey``). The tar goes to
``<pool_prefix>/snapshots/block-<N>.tar`` + ``<pool_prefix>/index.json`` in
the pool bucket (the lane-C publish path) and retired snapshots are mirrored
by ``cascade-pool reveal --lane mkt``.

Layout of a snapshot directory::

    snapshot.json                       signed body (below) + signature
    bars/<class>/<asset>/<field>.npy    float64, length ``minutes`` (NaN = missing)
    coarse/<class>/<asset>/<field>.npy  float64 HOURLY bars, length ``coarse.steps``

The coarse (hourly) bank (review 2026-10-19) carries ``[marketcast]
coarse_history_days`` of history for the series forecast at >= 1h
(:func:`~cascade.shared.config.marketcast.coarse_frequency`): Hyperliquid
serves only its most recent ~5000 candles per interval (measured 2026-10-19:
1m ≈ 3.6 days, 1h ≈ 208 days), so a 1d series cut from 21 days of minute
bars could never hold a scorable window. The body's ``coarse`` record
(``t0``, ``steps``, ``bar_minutes``) is present only when the bank is.

``snapshot.json`` body: ``version``, ``t0`` (unix seconds, a UTC day
boundary), ``minutes``, ``end`` (= ``t0 + 60·minutes``, the last observed
instant), ``anchor_block`` / ``anchor_time`` / ``block_seconds`` (the block →
time map the forward-only rule uses), ``sources`` (class → source spec),
``assets`` (class → symbols present), ``files`` (relative path → sha256),
``cross_check`` (per asset, a report), ``signer``. The digest is the sha256 of
the canonical body; the signature covers the same bytes.

Windows (:func:`snapshot_windows`) are cut with the pure rule in
:mod:`cascade.eval.marketcast` — forward-only: start strictly after the
entry's train-block time + ``start_margin_seconds``, end at or before the
snapshot's ``end``; the most recent ``windows_per_cell`` per (asset,
horizon).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..eval.marketcast import MktWindow, build_covariates, cut_windows, suite_series
from ..eval.marketcast import window_geometry as _geometry
from ..shared.config.marketcast import (
    BLOCK_SECONDS,
    COARSE_BAR_MINUTES,
    MarketCastConfig,
    coarse_frequency,
)

log = logging.getLogger("cascade.pool.marketcast")

MKT_SNAPSHOT_VERSION = 1
SNAPSHOT_FILE = "snapshot.json"
DAY = 86400


class SnapshotError(ValueError):
    """A MarketCast snapshot is malformed, tampered or unsigned."""


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_body(meta: dict) -> bytes:
    """The signed bytes: the snapshot body without ``signature``."""
    d = {k: v for k, v in meta.items() if k != "signature"}
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


def snapshot_digest(meta: dict) -> str:
    return hashlib.sha256(canonical_body(meta)).hexdigest()


def sign_snapshot_meta(meta: dict, wallet) -> dict:
    hotkey = getattr(wallet, "hotkey", wallet)
    sig = hotkey.sign(canonical_body(meta))
    return {**meta, "signature": sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)}


def verify_snapshot_signature(meta: dict, signer_hotkey: str) -> bool:
    sig = str(meta.get("signature", "") or "")
    if not sig or not signer_hotkey:
        return False
    from ..shared.sr25519 import verify_hotkey

    return verify_hotkey(signer_hotkey, canonical_body(meta), sig)


# ── writing ──────────────────────────────────────────────────────────────────


def write_snapshot(out_dir: Path | str, *, t0: int, minutes: int,
                   bars: dict[tuple[str, str], dict[str, np.ndarray]],
                   sources: dict[str, str], anchor_block: int, anchor_time: float,
                   cross: dict | None = None, signer: str = "", wallet=None,
                   block_seconds: float = BLOCK_SECONDS,
                   coarse: dict[tuple[str, str], dict[str, np.ndarray]] | None = None,
                   coarse_t0: int = 0, coarse_steps: int = 0) -> dict:
    """Write a snapshot directory from ``bars`` (``{(class, asset): {field:
    array}}``, every array of length ``minutes`` on the grid at ``t0``) and,
    optionally, the coarse hourly bank ``coarse`` (arrays of length
    ``coarse_steps`` on the hourly grid at ``coarse_t0``, a UTC day
    boundary), and return its (signed when ``wallet`` is given) meta.
    Deterministic: the same arrays give the same bytes (``np.save`` of
    float64); without ``coarse`` the body is exactly the pre-bank one."""
    out = Path(out_dir)
    if int(t0) % 60:
        raise SnapshotError(f"t0 {t0} is not a minute boundary")
    if coarse and int(coarse_t0) % 86400:
        raise SnapshotError(f"coarse t0 {coarse_t0} is not a UTC day boundary")
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    assets: dict[str, list[str]] = {}
    for bank, n, data in (("bars", int(minutes), bars),
                          ("coarse", int(coarse_steps), coarse or {})):
        for (cls, asset), fields_ in sorted(data.items()):
            for name, arr in sorted(fields_.items()):
                a = np.asarray(arr, dtype=np.float64)
                if a.shape != (n,):
                    raise SnapshotError(f"{bank}/{cls}/{asset}/{name}: shape {a.shape} != "
                                        f"({n},)")
                rel = f"{bank}/{cls}/{asset}/{name}.npy"
                p = out / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, a, allow_pickle=False)
                files[rel] = _sha256_file(p)
            if bank == "bars" or asset not in assets.get(cls, ()):
                assets.setdefault(cls, []).append(asset)
    meta = {"version": MKT_SNAPSHOT_VERSION, "t0": int(t0), "minutes": int(minutes),
            "end": int(t0) + 60 * int(minutes), "anchor_block": int(anchor_block),
            "anchor_time": float(anchor_time), "block_seconds": float(block_seconds),
            "sources": dict(sorted(sources.items())),
            "assets": {c: sorted(v) for c, v in sorted(assets.items())},
            "files": dict(sorted(files.items())), "cross_check": dict(cross or {}),
            "signer": str(signer or "")}
    if coarse:
        meta["coarse"] = {"t0": int(coarse_t0), "steps": int(coarse_steps),
                          "bar_minutes": COARSE_BAR_MINUTES}
    if wallet is not None:
        meta = sign_snapshot_meta(meta, wallet)
    (out / SNAPSHOT_FILE).write_text(json.dumps(meta, sort_keys=True, indent=1),
                                     encoding="utf-8")
    return meta


def snapshot_grid(end_time: float, days: int) -> tuple[int, int]:
    """``(t0, minutes)`` of an eval snapshot ending at the last full minute
    at or before ``end_time``, starting on the UTC day boundary ``days``
    days earlier."""
    end_min = int(end_time) // 60 * 60
    t0 = (end_min - int(days) * DAY) // DAY * DAY
    return t0, (end_min - t0) // 60


def coarse_grid(end_time: float, days: int) -> tuple[int, int]:
    """``(t0, hours)`` of the coarse bank: complete hours up to the last
    full hour at or before ``end_time``, from the UTC day boundary ``days``
    days earlier (an in-progress candle never lands on the grid)."""
    end_h = int(end_time) // 3600 * 3600
    t0 = (end_h - int(days) * DAY) // DAY * DAY
    return t0, (end_h - t0) // 3600


def fetch_coarse_bars(cls: str, asset: str, sources: tuple[str, ...], t0: int, steps: int, *,
                      post, alpha_rows=None) -> dict[str, np.ndarray]:
    """Hourly bars of one asset for the coarse bank from its class's
    PRIMARY source: Hyperliquid ``1h`` candles (+ hourly funding / premium
    for a perp), or the recorded alpha rows bucketed hourly. Binance is a
    1-minute cross-check only and has no coarse leg."""
    from . import marketcast_feeds as F

    if not sources:
        raise SnapshotError(f"{cls}: no data source configured")
    hm = COARSE_BAR_MINUTES
    start_ms, end_ms = int(t0) * 1000, (int(t0) + 60 * hm * int(steps)) * 1000
    primary = sources[0]
    if primary in ("hyperliquid", "hyperliquid_xyz"):
        coin = F.hl_coin(primary, asset)
        bars = F.bars_on_grid(F.fetch_hl_candles(post, coin, start_ms, end_ms, interval="1h"),
                              t0, steps, step_minutes=hm)
        fu, pr = F.funding_on_grid(F.fetch_hl_funding(post, coin, start_ms, end_ms),
                                   t0, steps, step_minutes=hm)
        bars["funding"], bars["premium"] = fu, pr
        return bars
    if primary == "bittensor_alpha":
        rows = [r for r in (alpha_rows or {}).get(F.alpha_netuid(asset), ())]
        return F.alpha_on_grid(rows, t0, steps, step_minutes=hm)
    raise SnapshotError(f"{cls}/{asset}: source {primary!r} has no coarse adapter")


def needs_coarse(mc: MarketCastConfig) -> bool:
    """Whether any enabled (suite, horizon) is forecast from the coarse bank."""
    for s in mc.suites:
        for h in mc.horizons_map.get(s, ()):
            try:
                f, _steps = _geometry(mc.frequencies_map.get(s, ()), h)
            except ValueError:
                continue
            if coarse_frequency(s, f):
                return True
    return False


def fetch_class_bars(cls: str, asset: str, sources: tuple[str, ...], t0: int, minutes: int, *,
                     post, get, alpha_rows=None, store=None) -> tuple[dict[str, np.ndarray], dict]:
    """Bars of one asset from its class's PRIMARY source (funding / premium
    too for a Hyperliquid perp), plus a cross-check report against each
    further source that can serve it. Network calls are ``post`` / ``get``.
    With ``store`` (a :class:`~cascade.pool.marketcast_store.MktBarStore`,
    review M15) a Hyperliquid primary's minute bars come from the forward
    recording instead — the API serves only its last ~3.6 days of 1m."""
    from . import marketcast_feeds as F

    if not sources:
        raise SnapshotError(f"{cls}: no data source configured")
    start_ms, end_ms = int(t0) * 1000, (int(t0) + 60 * int(minutes)) * 1000
    primary = sources[0]
    report: dict = {}
    if primary in ("hyperliquid", "hyperliquid_xyz") and store is not None:
        bars = store.read(cls, asset, t0, minutes)
    elif primary in ("hyperliquid", "hyperliquid_xyz"):
        coin = F.hl_coin(primary, asset)
        bars = F.bars_on_grid(F.fetch_hl_candles(post, coin, start_ms, end_ms), t0, minutes)
        fu, pr = F.funding_on_grid(F.fetch_hl_funding(post, coin, start_ms, end_ms),
                                   t0, minutes)
        bars["funding"], bars["premium"] = fu, pr
    elif primary == "binance_spot":
        bars = F.bars_on_grid(F.fetch_binance_klines(get, F.binance_symbol(asset),
                                                     start_ms, end_ms), t0, minutes)
    elif primary == "bittensor_alpha":
        n = F.alpha_netuid(asset)
        rows = [r for r in (alpha_rows or {}).get(n, ())]
        bars = F.alpha_on_grid(rows, t0, minutes)
    else:
        raise SnapshotError(f"{cls}/{asset}: source {primary!r} has no adapter")
    for extra in sources[1:]:
        try:
            if extra == "binance_spot":
                other = F.bars_on_grid(F.fetch_binance_klines(
                    get, F.binance_symbol(asset), start_ms, end_ms), t0, minutes)
            elif extra in ("hyperliquid", "hyperliquid_xyz"):
                other = F.bars_on_grid(F.fetch_hl_candles(
                    post, F.hl_coin(extra, asset), start_ms, end_ms), t0, minutes)
            else:
                continue
            report[extra] = F.cross_check(bars.get("close", bars.get("price")), other["close"])
        except Exception as e:  # noqa: BLE001 — a cross-check never fails the publish
            report[extra] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return bars, report


def build_snapshot(out_dir: Path | str, mc: MarketCastConfig, *, end_time: float,
                   anchor_block: int, anchor_time: float, post, get, alpha_rows=None,
                   wallet=None, signer: str = "", days: int | None = None,
                   store=None) -> dict:
    """Fetch every configured asset of every class an enabled suite reads
    and write the snapshot. An asset its source does not answer is left
    out (logged); the classes' sources are recorded."""
    t0, minutes = snapshot_grid(end_time, int(days or mc.snapshot_days))
    ct0, csteps = coarse_grid(end_time, int(mc.coarse_history_days))
    want_coarse = needs_coarse(mc)
    bars: dict = {}
    coarse: dict = {}
    cross: dict = {}
    am = mc.assets_map
    for cls in mc.armed_classes:
        srcs = mc.sources_for(cls)
        for asset in am.get(cls, ()):
            try:
                b, rep = fetch_class_bars(cls, asset, srcs, t0, minutes, post=post, get=get,
                                          alpha_rows=alpha_rows, store=store)
            except Exception as e:  # noqa: BLE001 — one asset never sinks the snapshot
                log.warning("marketcast snapshot: %s/%s skipped (%s)", cls, asset, e)
                continue
            if not any(np.isfinite(v).any() for v in b.values()):
                log.warning("marketcast snapshot: %s/%s has no data in the window", cls, asset)
                continue
            bars[(cls, asset)] = b
            if rep:
                cross[f"{cls}:{asset}"] = rep
            if want_coarse:
                try:
                    cb = fetch_coarse_bars(cls, asset, srcs, ct0, csteps, post=post,
                                           alpha_rows=alpha_rows)
                except Exception as e:  # noqa: BLE001 — its >= 1h series then go missing
                    log.warning("marketcast snapshot: %s/%s coarse bars skipped (%s)", cls,
                                asset, e)
                    continue
                if any(np.isfinite(v).any() for v in cb.values()):
                    coarse[(cls, asset)] = cb
    return write_snapshot(out_dir, t0=t0, minutes=minutes, bars=bars,
                          sources={c: mc.data_sources_map.get(c, "") for c in mc.armed_classes},
                          anchor_block=anchor_block, anchor_time=anchor_time, cross=cross,
                          signer=signer, wallet=wallet, coarse=coarse or None,
                          coarse_t0=ct0, coarse_steps=csteps)


# ── reading ──────────────────────────────────────────────────────────────────


@dataclass
class MktSnapshot:
    """A verified snapshot directory."""
    root: Path
    meta: dict
    digest: str
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def t0(self) -> int:
        return int(self.meta["t0"])

    @property
    def end(self) -> int:
        return int(self.meta["end"])

    def block_time(self, block: int) -> float:
        """Unix time of ``block`` by the snapshot's anchor (the forward-only
        rule's clock; deterministic from the signed body)."""
        m = self.meta
        return float(m["anchor_time"]) + (int(block) - int(m["anchor_block"])) * float(
            m["block_seconds"])

    def assets(self, cls: str) -> list[str]:
        return list(self.meta.get("assets", {}).get(cls, ()))

    def bars(self, cls: str, asset: str, *, bank: str = "bars") -> dict[str, np.ndarray]:
        key = (bank, cls, asset)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = {}
        prefix = f"{bank}/{cls}/{asset}/"
        for rel in self.meta["files"]:
            if rel.startswith(prefix):
                out[rel[len(prefix):-4]] = np.load(self.root / rel, allow_pickle=False)
        self._cache[key] = out
        return out

    def bank(self, cls: str, asset: str, suite: str, freq_minutes: int
             ) -> tuple[dict[str, np.ndarray], int, int]:
        """``(bars, t0, bar_minutes)`` a suite series at ``freq_minutes`` is
        cut from: the coarse hourly bank for a coarse frequency when the
        snapshot carries one, else the 1-minute bars."""
        c = self.meta.get("coarse")
        if c and coarse_frequency(suite, freq_minutes):
            return (self.bars(cls, asset, bank="coarse"), int(c["t0"]),
                    int(c.get("bar_minutes", COARSE_BAR_MINUTES)))
        return self.bars(cls, asset), self.t0, 1


def load_snapshot(root: Path | str, *, signer_hotkey: str = "",
                  verify_signature: bool = True) -> MktSnapshot:
    """Read + verify a snapshot directory: every listed file's sha256, no
    unlisted bar file, and (``verify_signature``) the body's signature by
    ``signer_hotkey``. Raises :class:`SnapshotError`."""
    root = Path(root)
    try:
        meta = json.loads((root / SNAPSHOT_FILE).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SnapshotError(f"{root}/{SNAPSHOT_FILE} unreadable: {e}") from e
    if not isinstance(meta, dict) or int(meta.get("version", 0)) != MKT_SNAPSHOT_VERSION:
        raise SnapshotError("unsupported MarketCast snapshot version")
    for rel, sha in meta.get("files", {}).items():
        p = root / rel
        if ".." in Path(rel).parts or not p.is_file() or _sha256_file(p) != sha:
            raise SnapshotError(f"snapshot file {rel} missing or sha256 mismatch")
    listed = set(meta.get("files", {}))
    extra = sorted(str(p.relative_to(root)) for bank in ("bars", "coarse")
                   if (root / bank).is_dir() for p in (root / bank).rglob("*")
                   if p.is_file() and str(p.relative_to(root)) not in listed)
    if extra:
        raise SnapshotError(f"unlisted snapshot files {extra[:3]}")
    if verify_signature and not verify_snapshot_signature(meta, signer_hotkey):
        raise SnapshotError("snapshot signature does not verify against the main validator "
                            f"({signer_hotkey or 'owner_hotkey unset'})")
    return MktSnapshot(root, meta, snapshot_digest(meta))


# ── windows ──────────────────────────────────────────────────────────────────


def asset_series(snap: MktSnapshot, cls: str, asset: str, suite: str,
                 freq_minutes: int) -> np.ndarray | None:
    """The suite series of one asset, block 0 at ``snap.bank(...)``'s ``t0``
    (None when the snapshot lacks the fields it needs, e.g. funding for a
    spot-only asset)."""
    b, _t0, bm = snap.bank(cls, asset, suite, freq_minutes)
    need = {"returns": ("close",), "vol": ("close",), "range": ("open", "high", "low"),
            "funding": ("funding",), "liquidity": ("volume",), "alpha": ("price",)}[suite]
    if not all(k in b for k in need):
        return None
    return suite_series(suite, b, freq_minutes, bar_minutes=bm)


def liquidity_filter(snap: MktSnapshot, mc: MarketCastConfig, cls: str, asset: str, suite: str,
                     freq_minutes: int):
    """Review M13 — the liquidity floor, as ``eligible(start)`` for
    :func:`~cascade.eval.marketcast.cut_windows`: over the 24 h before a
    window (clipped to the bank's start, pro-rated), the share of live bars
    (finite close / price) must reach ``min_bar_coverage`` and the traded
    dollar volume (Σ close × volume) ``min_dollar_volume_24h`` (skipped for a
    series without volume, e.g. dTAO alpha). Read from the same bank the
    series is cut from, so the audit re-derives it from the snapshot."""
    b, t0, bm = snap.bank(cls, asset, suite, freq_minutes)
    px = np.asarray(b.get("close", b.get("price")), dtype=np.float64)
    if px.ndim != 1 or not px.size:
        return lambda _start: False
    live = np.concatenate([[0.0], np.cumsum(np.isfinite(px))])
    vol = b.get("volume")
    dv = None
    if vol is not None and float(mc.min_dollar_volume_24h) > 0:
        x = np.asarray(px) * np.asarray(vol, dtype=np.float64)
        dv = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(x), x, 0.0))])
    lb = 1440 // int(bm)

    def _ok(start: float) -> bool:
        kb = min(int((float(start) - t0) // (60 * bm)), px.size)
        a = max(0, kb - lb)
        n = kb - a
        if n <= 0:
            return False
        if (live[kb] - live[a]) / n < float(mc.min_bar_coverage):
            return False
        return dv is None or (dv[kb] - dv[a]) * lb / n >= float(mc.min_dollar_volume_24h)

    return _ok


# price-return suites: a window whose recent CONTEXT is exactly flat (the
# market stopped printing) is out — read from the context, never the target
# (red-team F3). Funding changes are legitimately flat for long spans.
STALE_CONTEXT_SUITES = ("returns", "alpha")


def snapshot_windows(snap: MktSnapshot, mc: MarketCastConfig, suite: str, *,
                     after: float) -> list[MktWindow]:
    """Every scorable window of ``suite`` for a submission whose windows
    must start after ``after`` (unix): per configured (class, asset) present
    in the snapshot and per suite horizon, ``windows_per_cell`` forward-only
    windows spread EVENLY over the eligible span (review M10), each past the
    liquidity floor (:func:`liquidity_filter`, read from what is known at the
    window's START). Never filtered on the realised target (red-team F3,
    2026-10-20 — the old ``min_target_abs_bp`` floor conditioned the test set
    on the outcome: a lattice forecaster moving mass off zero gained 18.7 %);
    a zero-scale cell is masked in the verdict instead. Sorted by window id."""
    out: list[MktWindow] = []
    freqs = mc.frequencies_map[suite]
    for h in mc.horizons_map[suite]:
        f, steps = _geometry(freqs, h)
        for cls, asset in mc.suite_assets(suite):
            if asset not in snap.assets(cls):
                continue
            s = asset_series(snap, cls, asset, suite, f)
            if s is None:
                continue
            out.extend(cut_windows(
                s, cls=cls, asset=asset, suite=suite, horizon=h, freq_minutes=f, steps=steps,
                t0=snap.bank(cls, asset, suite, f)[1],
                context_length=int(mc.eval_context_length), after=float(after),
                now=float(snap.end), limit=int(mc.windows_per_cell),
                eligible=liquidity_filter(snap, mc, cls, asset, suite, f),
                stale_context=suite in STALE_CONTEXT_SUITES))
    return sorted(out, key=lambda w: w.wid)


CROSS_ASSET = ("crypto", "BTC")


def window_covariates(snap: MktSnapshot, w: MktWindow, names, *, context_length: int
                      ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The fine-tune's covariate rows for window ``w`` (the order of
    :data:`cascade.eval.marketcast.PAST_COVARIATES` / ``KNOWN_COVARIATES``):
    past rows aligned with ``w.context``, known rows over context + horizon."""
    if not names:
        return None, None
    n_ctx = int(w.context.size)
    k0 = w.index - n_ctx
    sl = slice(k0, w.index)
    f = w.freq_minutes
    b, t0, bm = snap.bank(w.cls, w.asset, w.suite, f)

    def _past(series):
        return None if series is None else np.asarray(series)[sl]

    vol = _past(suite_series("liquidity", b, f, bar_minutes=bm)) if "volume" in b else None
    fund = None
    if "funding" in b:
        from ..eval.marketcast import _block_end, funding_bp

        fund = _past(funding_bp(_block_end(np.asarray(b["funding"]), f // bm)))
    cross = None
    if CROSS_ASSET[1] in snap.assets(CROSS_ASSET[0]):
        cb, ct0, cbm = snap.bank(*CROSS_ASSET, w.suite, f)
        if "close" in cb and (ct0, cbm) == (t0, bm):
            cross = _past(suite_series("returns", cb, f, bar_minutes=cbm))
    starts = t0 + 60.0 * f * np.arange(k0, w.index + w.steps)
    return build_covariates(names, block_starts=starts, n_context=n_ctx, volume_series=vol,
                            funding_series=fund, cross_series=cross)


# ── the bucket source (validator side) ───────────────────────────────────────


class MktSnapshotSource:
    """The MarketCast snapshot active at a settlement block, fetched from the
    pool bucket under ``prefix`` (greatest ``effective_block <= block``),
    sha256-verified against the index, then verified as a snapshot."""

    def __init__(self, store, *, prefix: str, signer_hotkey: str, cache_dir: Path | str,
                 verify_signature: bool = True) -> None:
        self.store, self.prefix = store, str(prefix)
        self.signer, self.verify = str(signer_hotkey or ""), bool(verify_signature)
        self.cache_dir = Path(cache_dir) / self.prefix
        self._cache: dict[str, MktSnapshot] = {}

    def select(self, block: int):
        from ..shared.hippius import read_pool_index, select_snapshot

        index = read_pool_index(self.store, self.prefix)
        return select_snapshot(index, int(block)) if index else None

    def snapshot_for(self, block: int) -> tuple[MktSnapshot, str, str] | None:
        """``(snapshot, key, tar_sha256)`` active at ``block`` or None."""
        meta = self.select(block)
        if meta is None:
            return None
        hit = self._cache.get(meta.sha256)
        if hit is None:
            from ..shared.hippius import fetch_pool_snapshot

            dest = self.cache_dir / f"snapshot-{meta.effective_block}-{meta.sha256[:12]}"
            fetch_pool_snapshot(self.store, meta, dest)
            hit = load_snapshot(dest, signer_hotkey=self.signer, verify_signature=self.verify)
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[meta.sha256] = hit
        return hit, meta.key, meta.sha256


__all__ = [
    "CROSS_ASSET", "MKT_SNAPSHOT_VERSION", "STALE_CONTEXT_SUITES", "liquidity_filter", "MktSnapshot", "MktSnapshotSource", "SNAPSHOT_FILE",
    "SnapshotError", "asset_series", "build_snapshot", "canonical_body", "coarse_grid",
    "fetch_class_bars", "fetch_coarse_bars", "needs_coarse",
    "load_snapshot", "sign_snapshot_meta", "snapshot_digest", "snapshot_grid",
    "snapshot_windows", "verify_snapshot_signature", "window_covariates", "write_snapshot",
]
