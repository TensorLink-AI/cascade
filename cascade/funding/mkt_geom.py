"""MarketCast sample-geometry check (2026-11-28): can a recipe's leg cut its
training samples from the history its data range actually holds?

The 2026-11-27 23:24 failure: a funded leg (5Cu2sxndK8Dw, mkt:market,
``data.start`` 2024-01-01 .. ``data.end`` 2026-10-25, context 4096, batch 6,
``synthetic_fraction`` 0) ended as a miner fault "could not cut a
high/15mx96 training sample" (``marketcast_leg.py`` ``MktSampler.batch``)
after ~32 min of CPU prep on a paid pod. The cause is NOT the context length
(``MktSampler._real`` cuts the context as ``s[max(0, k - ctx):k]`` and needs
only ``MIN_CONTEXT`` = 64 finite steps in it): it is the DATE RANGE. Every
series spans ``[data.start, data.end + 1 day)`` on the bar grid, but the
15-minute bank (the recorder's native 15m candles, Hyperliquid serves ~52
days of them) begins 2026-09-25, so each 15m series is ~98,900 blocks of
which ~2,900 (~3 %) are finite. ``_real`` draws the sample end ``k``
uniformly over the whole span, 16 tries per sample: a sample misses with
probability ~0.62, a batch of 6 with ~0.057, and a leg with ~4,000 15m
batch draws is certain to hit one; whichever suite does first ("high")
names the error. A shorter context does not help; a later ``data.start``
(or no 15m in ``data.frequencies``) does.

This module re-derives, stdlib only, what the pod's sampler does with a
recipe over the recorded history (the public post-embargo bar index,
:mod:`cascade.shared.marketcast_public` ``index.json``: per class / symbol /
timeframe the first and last recorded bar — the same recorder the leg's
bar relay serves, and a recipe's range ends ``embargo_days`` before the
fund, i.e. inside the public cut):

* per (suite, horizon) of the recipe: bar ``f`` and ``steps``
  (``window_geometry``), the bank it is cut from (``bank_minutes``: 1h for
  whole hours, else the coarsest native interval dividing ``f``, else 1m);
* per asset: the series length ``L`` (blocks of ``f`` over the range) and
  its finite span ``[a, b]``; a series exists only with > 64 finite blocks;
  the share ``p`` of sample ends ``k ∈ [64, L - steps]`` with ``k ≥ a + 64``
  and ``k ≤ b`` (64 finite context steps, a finite target);
* a sample misses with ``q = Σ_class w · mean_asset (1 - p)^16`` (the
  class-weighted pick, 16 tries on one key), times ``1 - synthetic_fraction``
  (a synthetic draw is assumed to succeed); a batch fails with ``q^batch``;
* (suite, horizon) is drawn ``steps · P(suite) · P(horizon | suite)`` times
  (the sampler's weights), so the leg dies with
  ``1 - Π (1 - q^batch)^n``.

Gaps inside a recorded span are ignored, so the estimate is optimistic: a
refusal is never a false alarm from it. A recipe at or over :data:`MAX_RISK`
is refused at fund time with the earliest ``data.start`` that brings it under.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

__all__ = ["MAX_RISK", "GeomVerdict", "geom_verdict", "history_spans", "leg_risk"]

MIN_CONTEXT = 64              # marketcast_leg.MIN_CONTEXT
TRIES = 16                    # MktSampler._real: draws of k per sample
MAX_RISK = 0.01               # refuse a recipe whose leg dies with >= 1 %
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
DAY_S = 86400


def _ts(s: str) -> int:
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return int(d.timestamp())


def history_spans(index: Mapping) -> dict[tuple[str, str, str], tuple[int, int]]:
    """``{(class, SYMBOL, tf): (first, last)}`` bar open times (unix s) of
    a public bar index (``cascade-marketcast-public-bars/1``), over all its
    month files."""
    out: dict[tuple[str, str, str], tuple[int, int]] = {}
    for rec in dict(index.get("files") or {}).values():
        if not isinstance(rec, Mapping) or rec.get("timeframe") not in TF_MINUTES:
            continue
        try:
            k = (str(rec["class"]), str(rec["symbol"]).upper(), str(rec["timeframe"]))
            lo, hi = _ts(rec["first"]), _ts(rec["last"])
        except (KeyError, ValueError, TypeError):
            continue
        if k in out:
            lo, hi = min(lo, out[k][0]), max(hi, out[k][1])
        out[k] = (lo, hi)
    return out


@dataclass(frozen=True)
class _Recipe:
    target_suite: str
    assets: dict[str, tuple[str, ...]]
    class_weights: dict[str, float]
    freqs: tuple[str, ...]
    start: date
    end: date
    horizon_mix: dict[str, float]
    suite_weights: dict[str, float]
    synthetic_fraction: float
    synthetic_channels: bool
    batch: int
    steps: int


def _recipe(obj) -> _Recipe | None:
    """The geometry-relevant fields of a ``marketcast.json`` object, read
    leniently (None when malformed: the trainer's full parse refuses it)."""
    try:
        data, op, ob = obj["data"], obj["optimizer"], obj["objective"]
        target = str(obj.get("target", ""))
        return _Recipe(
            target_suite=target.split(":", 1)[1] if ":" in target else target,
            assets={str(c): tuple(str(a) for a in v) for c, v in dict(data["assets"]).items()},
            class_weights={str(c): float(w) for c, w in dict(data["class_weights"]).items()},
            freqs=tuple(str(f) for f in data["frequencies"]),
            start=date.fromisoformat(str(data["start"])),
            end=date.fromisoformat(str(data["end"])),
            horizon_mix={str(h): float(w) for h, w in dict(obj.get("horizon_mix") or {}).items()},
            suite_weights={str(s): float(w) for s, w in dict(ob["suite_weights"]).items()},
            synthetic_fraction=float(obj.get("synthetic_fraction", 0.0) or 0.0),
            synthetic_channels=bool(obj.get("synthetic_channels")),
            batch=int(op["batch"]), steps=int(op["steps"]))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return None


def _geometry(freqs, horizon: str) -> tuple[int, int] | None:
    """``window_geometry`` (cascade.eval.marketcast), None where it raises."""
    from ..shared.config.marketcast import TIMESTRAL_MAX_HORIZON, duration_minutes, horizon_path

    try:
        p = horizon_path(horizon)
        if p is not None:
            if p[0] not in tuple(freqs) or p[1] > TIMESTRAL_MAX_HORIZON:
                return None
            return duration_minutes(p[0]), int(p[1])
        h = duration_minutes(horizon)
        fit = [duration_minutes(f) for f in freqs
               if duration_minutes(f) <= h and h % duration_minutes(f) == 0]
        if not fit or h // max(fit) > TIMESTRAL_MAX_HORIZON:
            return None
        return max(fit), h // max(fit)
    except ValueError:
        return None


def _bank_tf(mc, f: int) -> str:
    bm = int(mc.bank_minutes(f))
    return {v: k for k, v in TF_MINUTES.items()}.get(bm, "1m")


def _p_cut(span, t0: int, t1: int, f: int, steps: int) -> float | None:
    """Share of the sampler's draws of ``k`` that cut a sample from one
    series, None when the series is not built (<= 64 finite blocks)."""
    blk = 60 * f
    n = (t1 - t0) // blk
    if span is None or n <= 0:
        return None
    a = max(0, -(-(span[0] - t0) // blk))
    b = min(n - 1, (span[1] - t0) // blk)
    if b - a + 1 <= MIN_CONTEXT:
        return None
    draws = n - steps + 1 - MIN_CONTEXT
    if draws <= 0:
        return 0.0
    ok = min(b, n - steps) - max(MIN_CONTEXT, a + MIN_CONTEXT) + 1
    return max(0, ok) / draws


@dataclass(frozen=True)
class _Cell:
    suite: str
    horizon: str
    bar: str                 # the bank timeframe
    batch_fail: float        # P(one batch of this cell cannot be cut)
    draws: float             # expected batches of this cell over the leg


def _cells(r: _Recipe, mc, spans, start: date) -> list[_Cell]:
    from ..shared.config.marketcast import SUITE_CLASSES

    t0 = int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp())
    t1 = int(datetime.combine(r.end, datetime.min.time(), tzinfo=UTC).timestamp()) + DAY_S
    hmap, fmap = dict(mc.horizons_map), dict(mc.frequencies_map)
    composite = r.target_suite not in hmap
    synth = r.synthetic_fraction > 0
    pure = r.synthetic_fraction >= 1.0
    per_suite: dict[str, tuple[float, dict[str, tuple[float, float, str]]]] = {}
    for base, w in r.suite_weights.items():
        if w <= 0 or base not in hmap:
            continue
        freqs = [f for f in fmap.get(base, ()) if f in r.freqs]
        if base == r.target_suite or composite:
            hz = {h: x for h, x in r.horizon_mix.items() if h in hmap[base]} \
                or {h: 1.0 for h in hmap[base]}
        else:
            hz = {h: 1.0 for h in hmap[base]}
        classes = (mc.suite_classes(base) if hasattr(mc, "suite_classes")
                   else SUITE_CLASSES.get(base, ()))
        cells: dict[str, tuple[float, float, str]] = {}   # h -> (q_sample, mix, tf)
        have = False
        for h, x in hz.items():
            g = _geometry(freqs, h)
            if g is None:
                continue
            f, steps = g
            tf = _bank_tf(mc, f)
            q_cls, w_cls = 0.0, 0.0
            for cls, syms in r.assets.items():
                if cls not in classes:
                    continue
                ps = [p for p in (_p_cut(spans.get((cls, a.upper(), tf)), t0, t1, f, steps)
                                  for a in syms) if p is not None]
                if not ps:
                    continue
                have = True
                cw = float(r.class_weights.get(cls, 0.0))
                if cw <= 0:
                    continue
                q_cls += cw * sum((1.0 - p) ** TRIES for p in ps) / len(ps)
                w_cls += cw
            if w_cls <= 0:
                # no real series at this bar: without a generator the sampler
                # drops the horizon; with named synthetic channels it drops it
                # on the first miss; else only synthetic draws can cut it
                if not synth or r.synthetic_channels:
                    continue
                q = 1.0
            else:
                q = q_cls / w_cls
            cells[h] = (q * (1.0 - min(1.0, r.synthetic_fraction)), float(x), tf)
        if cells and (have or pure):
            per_suite[base] = (float(w), cells)
    tot = sum(w for w, _c in per_suite.values())
    out = []
    for base, (w, cells) in per_suite.items():
        own = base == r.target_suite or composite
        mix = {h: (c[1] if own else 1.0) for h, c in cells.items()}
        mt = sum(mix.values()) or 1.0
        for h, (q, _x, tf) in cells.items():
            out.append(_Cell(base, h, tf, q ** max(1, r.batch),
                             r.steps * (w / tot) * (mix[h] / mt)))
    return out


def leg_risk(cells: list[_Cell]) -> float:
    """P(the leg hits a batch it cannot cut) over its ``steps`` batches."""
    log_ok = 0.0
    for c in cells:
        if c.batch_fail >= 1.0 and c.draws > 0:
            return 1.0
        log_ok += c.draws * math.log1p(-c.batch_fail)
    return 1.0 - math.exp(log_ok)


@dataclass(frozen=True)
class GeomVerdict:
    risk: float
    suite: str               # the worst (suite, horizon) cell
    horizon: str
    bar: str
    bar_first: str           # first recorded day of that bank over the recipe's assets
    start: str               # the recipe's data.start
    fix_start: str           # earliest data.start under MAX_RISK ("" = none)

    @property
    def refused(self) -> bool:
        return self.risk >= MAX_RISK

    def message(self) -> str:
        fix = (f"set data.start to {self.fix_start} or later" if self.fix_start
               else "narrow data.start/data.end to where the history is recorded")
        return (f"your recipe's leg would fail \"could not cut a {self.suite}/{self.horizon} "
                f"training sample\" (estimated {100 * self.risk:.0f} % chance over its steps): "
                f"{self.horizon} is cut from the {self.bar} bars, whose recorded history "
                f"starts {self.bar_first or 'after your data.end'}, but data.start is "
                f"{self.start}, so almost every training draw over the range lands where "
                f"there are no {self.bar} bars. context_length is not the limit (the context "
                f"is cut to the history before each sample). {fix}, or drop that bar from "
                f"data.frequencies / horizon_mix; `cascade lane-c data` shows the recorded "
                f"history")


def geom_verdict(obj, mc, index: Mapping) -> GeomVerdict | None:
    """The sample-geometry verdict of recipe ``obj`` under ``[marketcast]``
    ``mc`` over the public bar ``index``; None when the recipe cannot be read."""
    r = _recipe(obj)
    if r is None or r.end < r.start:
        return None
    spans = history_spans(index)
    try:
        cells = _cells(r, mc, spans, r.start)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not cells:
        return None
    risk = leg_risk(cells)
    worst = max(cells, key=lambda c: c.draws * c.batch_fail if c.batch_fail < 1 else math.inf)
    firsts = sorted(lo for (cls, a, tf), (lo, _hi) in spans.items()
                    if tf == worst.bar and a in {s.upper() for s in r.assets.get(cls, ())})
    bar_first = (datetime.fromtimestamp(firsts[0], tz=UTC).date().isoformat()
                 if firsts else "")
    fix = ""
    if risk >= MAX_RISK:
        # the earliest later start (each recorded first day of any trained
        # bank, then day by day after the latest) that brings the risk under
        cand = sorted({datetime.fromtimestamp(lo, tz=UTC).date() + timedelta(days=1)
                       for (_c, _a, _tf), (lo, _hi) in spans.items()})
        for d in (d for d in cand if r.start < d <= r.end):
            c2 = _cells(r, mc, spans, d)
            if c2 and leg_risk(c2) < MAX_RISK:
                fix = d.isoformat()
                break
    return GeomVerdict(round(risk, 4), worst.suite, worst.horizon, worst.bar, bar_first,
                       r.start.isoformat(), fix)
