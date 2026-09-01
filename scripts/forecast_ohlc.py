#!/usr/bin/env python3
"""Forecast OHLC candles with per-candle quantile bands from a cascade checkpoint.

The live contract is univariate — every checkpoint's ``forecast_wrapper.py``
decodes one series at a time (the variate-attention layers run at ``C = 1``
under the current pin). An OHLC candle is four aligned series, so this tool
forecasts ``open``/``high``/``low``/``close`` as four channels of ONE batched
call to the wrapper's quantile head::

    forecast_quantiles_batch([open, high, low, close], horizon)
        -> (4, horizon, num_q)

and re-assembles the result per candle: for every future candle t and every
quantile level tau you get a full OHLC tuple. Because the four marginals are
decoded independently they can disagree (a decoded ``high`` can sit below the
decoded ``close`` at the same level), so the assembly applies a coherence
repair before anything is published:

* per channel, levels are sorted (no quantile crossing — the wrapper already
  guarantees this, re-asserted here because the repair depends on it);
* per candle and level, ``high`` is raised to cover ``open``/``close`` and
  ``low`` lowered to cover them, so ``low <= open, close <= high`` always
  holds (and hence ``low <= high``, through ``open``).

The repair is a presentational projection of four marginal quantile curves
onto candle geometry — it is NOT a joint distribution over candles, and the
numbers stay marginal per channel. Pre-CPM checkpoints (no quantile head) fall
back to the validator's sample-path contract and take empirical quantiles.

Usage::

    python scripts/forecast_ohlc.py --checkpoint /path/to/ckpt \\
        --candles sn84.csv --horizon 64 --holdout 64 \\
        --out-json forecast.json --plot forecast.png

``--candles`` is a CSV with (case-insensitive) ``open,high,low,close`` columns
(remap with ``--columns open=o,high=hi,...``). ``--holdout N`` keeps the last
N candles out of the model's context and overlays them on the plot as ground
truth. Plotting needs matplotlib (not a core dependency); the JSON path does
not. The checkpoint wrapper itself imports torch.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CHANNELS = ("open", "high", "low", "close")

# Fallback grid for pre-CPM checkpoints that declare no quantile_levels —
# the same 9-level grid every CPM checkpoint trains (trainer's default).
DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ── candle IO ────────────────────────────────────────────────────────────────


def load_candles(path: str | Path, column_map: dict[str, str] | None = None) -> np.ndarray:
    """Read a CSV of candles into ``(n, 4)`` float64 ``[open, high, low, close]``.

    Header matching is case-insensitive; ``column_map`` remaps channel name →
    CSV column name (e.g. ``{"open": "o"}``). Rows must be complete and finite.
    """
    wanted = {ch: (column_map or {}).get(ch, ch).lower() for ch in CHANNELS}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty file") from None
        lookup = {name.strip().lower(): i for i, name in enumerate(header)}
        missing = [wanted[ch] for ch in CHANNELS if wanted[ch] not in lookup]
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {missing}; header has {sorted(lookup)} "
                f"(remap with --columns)"
            )
        idx = [lookup[wanted[ch]] for ch in CHANNELS]
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if not row or all(not c.strip() for c in row):
                continue
            try:
                rows.append([float(row[i]) for i in idx])
            except (ValueError, IndexError) as e:
                raise ValueError(f"{path}:{line_no}: bad candle row: {e}") from None
    if not rows:
        raise ValueError(f"{path}: no candle rows")
    candles = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(candles).all():
        raise ValueError(f"{path}: non-finite values in candle data")
    return candles


# ── quantile assembly ────────────────────────────────────────────────────────


def repair_ohlc_coherence(q: np.ndarray) -> np.ndarray:
    """Project per-channel quantiles onto valid candle geometry.

    ``q`` is ``(horizon, 4, num_q)`` in channel order ``CHANNELS``. Returns a
    copy where, per channel, levels are non-decreasing, and per (candle,
    level), ``low <= open, close <= high``. Open/close are never moved — the
    body is the model's word; only the wicks are widened to contain it.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 3 or q.shape[1] != len(CHANNELS):
        raise ValueError(f"expected (horizon, 4, num_q), got {q.shape}")
    q = np.sort(q, axis=-1)  # monotone in the level, per channel
    out = q.copy()
    # Envelopes over {open, high, close} / {open, low, close}: a degenerate
    # high (low) marginal is repaired without contaminating the other wick,
    # and low <= open <= high gives low <= high for free.
    out[:, 1, :] = q[:, (0, 1, 3), :].max(axis=1)
    out[:, 2, :] = q[:, (0, 2, 3), :].min(axis=1)
    return out


def forecast_ohlc_quantiles(
    wrapper, candles: np.ndarray, horizon: int, *, num_samples: int = 256
) -> tuple[np.ndarray, list[float]]:
    """Forecast ``horizon`` future candles from ``(n, 4)`` OHLC history.

    Returns ``(q, levels)`` with ``q`` of shape ``(horizon, 4, num_q)``
    (coherence-repaired, channel order ``CHANNELS``) and ``levels`` the
    ascending quantile grid. Uses the batched quantile head when the
    checkpoint has one; otherwise draws sample paths per channel via the
    validator contract and takes empirical quantiles.
    """
    candles = np.asarray(candles, dtype=np.float64)
    if candles.ndim != 2 or candles.shape[1] != len(CHANNELS):
        raise ValueError(f"candles must be (n, 4) [open, high, low, close], got {candles.shape}")
    if candles.shape[0] < 2:
        raise ValueError("need at least 2 candles of history")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    levels = [float(v) for v in getattr(wrapper, "quantile_levels", None) or DEFAULT_QUANTILE_LEVELS]
    if sorted(levels) != levels or len(set(levels)) != len(levels):
        raise ValueError(f"wrapper.quantile_levels not strictly ascending: {levels}")
    histories = [candles[:, i] for i in range(len(CHANNELS))]

    if hasattr(wrapper, "forecast_quantiles_batch"):
        q = np.asarray(wrapper.forecast_quantiles_batch(histories, horizon), dtype=np.float64)
        expected = (len(CHANNELS), horizon, len(levels))
        if q.shape != expected:
            raise ValueError(f"forecast_quantiles_batch returned {q.shape}; expected {expected}")
    else:
        # Pre-CPM checkpoint: validator contract only. One seeded sample-path
        # draw per channel, then empirical quantiles of the marginals.
        per_channel = []
        for h in histories:
            samples = np.asarray(wrapper.forecast(h, horizon, num_samples), dtype=np.float64)
            if samples.shape != (1, num_samples, horizon):
                raise ValueError(
                    f"wrapper.forecast returned {samples.shape}; "
                    f"expected (1, {num_samples}, {horizon})"
                )
            per_channel.append(np.quantile(samples[0], levels, axis=0).T)  # (horizon, num_q)
        q = np.stack(per_channel)  # (4, horizon, num_q)

    return repair_ohlc_coherence(np.moveaxis(q, 0, 1)), levels


def quantiles_to_records(q: np.ndarray, levels: list[float]) -> list[dict]:
    """JSON-able per-candle records: ``[{"step": 1, "open": {"0.1": ...}, ...}]``.

    ``step`` is 1-based — step 1 is the first candle after the end of context.
    """
    keys = [f"{lv:g}" for lv in levels]
    return [
        {
            "step": t + 1,
            **{
                ch: dict(zip(keys, (float(v) for v in q[t, c]), strict=True))
                for c, ch in enumerate(CHANNELS)
            },
        }
        for t in range(q.shape[0])
    ]


# ── checkpoint loading (mirrors benchmarks/cascade_benchmark/predictor.py) ───


def load_wrapper(checkpoint_dir: str | Path, device: str = "cpu"):
    """Import ``forecast_wrapper.Wrapper`` from the checkpoint and instantiate it.

    The wrapper is owner-produced and trusted, so no sandboxing is applied —
    do not point this at an unverified generator artifact.
    """
    wrapper_py = Path(checkpoint_dir) / "forecast_wrapper.py"
    if not wrapper_py.is_file():
        raise FileNotFoundError(f"missing forecast_wrapper.py in {checkpoint_dir}")
    spec = importlib.util.spec_from_file_location("cascade_ohlc_wrapper", wrapper_py)
    if spec is None or spec.loader is None:
        raise ImportError("could not load forecast_wrapper spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    Wrapper = getattr(module, "Wrapper", None)
    if Wrapper is None:
        raise AttributeError("forecast_wrapper.py defines no `Wrapper` class")
    return Wrapper(str(checkpoint_dir), device=device)


# ── plotting ─────────────────────────────────────────────────────────────────


def _nearest_level(levels: list[float], target: float) -> int:
    return int(np.argmin([abs(lv - target) for lv in levels]))


def _draw_candles(ax, xs, candles, *, alpha=1.0, width=0.6):
    for x, (o, h, lo, c) in zip(xs, candles, strict=True):
        color = "#2e7d32" if c >= o else "#c62828"
        ax.vlines(x, lo, h, color=color, linewidth=0.9, alpha=alpha)
        body_lo, body_hi = min(o, c), max(o, c)
        if body_hi == body_lo:  # doji: keep the body visible
            body_hi = body_lo + 1e-12
        ax.add_patch(
            _mpl_patches().Rectangle(
                (x - width / 2, body_lo), width, body_hi - body_lo,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.5,
            )
        )


def _mpl_patches():
    import matplotlib.patches as patches

    return patches


def plot_ohlc_forecast(
    history: np.ndarray,
    q: np.ndarray,
    levels: list[float],
    out_path: str | Path,
    *,
    actual_future: np.ndarray | None = None,
    history_bars: int = 96,
    title: str | None = None,
) -> None:
    """Render history candles + per-candle quantile forecast to ``out_path``.

    Each forecast candle shows: outer wick = [low @ ~q10, high @ ~q90], inner
    band = [low @ ~q30, high @ ~q70], body = median open → median close
    (green/red by median direction), dot = median close. ``actual_future``
    (``(m, 4)`` OHLC) overlays ground-truth candles on top for comparison.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "plotting requires matplotlib (`uv pip install matplotlib`); "
            "use --out-json for the numbers alone"
        ) from e

    history = np.asarray(history, dtype=np.float64)[-int(history_bars):]
    horizon = q.shape[0]
    i_lo, i_hi = _nearest_level(levels, 0.1), _nearest_level(levels, 0.9)
    i_q1, i_q3 = _nearest_level(levels, 0.3), _nearest_level(levels, 0.7)
    i_med = _nearest_level(levels, 0.5)

    fig, ax = plt.subplots(figsize=(14, 5))
    hist_x = np.arange(-history.shape[0] + 1, 1)
    _draw_candles(ax, hist_x, history)

    fx = np.arange(1, horizon + 1)
    op, hi, lo, cl = (q[:, c, :] for c in range(4))
    # per-candle quantile glyphs: wide band, narrow band, median body
    ax.vlines(fx, lo[:, i_lo], hi[:, i_hi], color="#7986cb", linewidth=1.0, alpha=0.55,
              label=f"low q{levels[i_lo]:g} – high q{levels[i_hi]:g}")
    ax.vlines(fx, lo[:, i_q1], hi[:, i_q3], color="#5c6bc0", linewidth=3.0, alpha=0.35,
              label=f"low q{levels[i_q1]:g} – high q{levels[i_q3]:g}")
    for x, o, c in zip(fx, op[:, i_med], cl[:, i_med], strict=True):
        color = "#2e7d32" if c >= o else "#c62828"
        ax.vlines(x, min(o, c), max(o, c), color=color, linewidth=3.2, alpha=0.45)
    ax.plot(fx, cl[:, i_med], ".", color="#3949ab", markersize=3, label="median close")

    if actual_future is not None and len(actual_future):
        _draw_candles(ax, np.arange(1, len(actual_future) + 1), actual_future, width=0.4)

    ax.axvline(0.5, color="0.6", linewidth=0.8)
    ax.set_xlabel("candles (0 = end of context)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_columns(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in spec.split(","):
        ch, _, col = part.partition("=")
        ch = ch.strip().lower()
        if ch not in CHANNELS or not col.strip():
            raise argparse.ArgumentTypeError(
                f"--columns wants comma-separated ch=col with ch in {CHANNELS}, got {part!r}"
            )
        out[ch] = col.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir (with forecast_wrapper.py)")
    ap.add_argument("--candles", required=True, help="CSV with open,high,low,close columns")
    ap.add_argument("--horizon", type=int, default=64, help="future candles to forecast")
    ap.add_argument("--holdout", type=int, default=0,
                    help="keep the last N candles out of context and overlay them as actuals")
    ap.add_argument("--columns", type=_parse_columns, default=None,
                    help="remap channels, e.g. open=o,high=hi,low=lo,close=last")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-samples", type=int, default=256,
                    help="sample paths per channel for pre-CPM checkpoints")
    ap.add_argument("--out-json", default=None, help="write per-candle quantiles as JSON")
    ap.add_argument("--plot", default=None, help="write a PNG of candles + quantile bands")
    ap.add_argument("--history-bars", type=int, default=96, help="context candles to draw")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    if not (args.out_json or args.plot):
        ap.error("nothing to do: pass --out-json and/or --plot")

    candles = load_candles(args.candles, args.columns)
    holdout = max(0, int(args.holdout))
    if holdout >= candles.shape[0] - 1:
        ap.error(f"--holdout {holdout} leaves no context ({candles.shape[0]} candles total)")
    context, actual = (candles[:-holdout], candles[-holdout:]) if holdout else (candles, None)

    wrapper = load_wrapper(args.checkpoint, device=args.device)
    q, levels = forecast_ohlc_quantiles(
        wrapper, context, args.horizon, num_samples=args.num_samples
    )

    if args.out_json:
        payload = {
            "checkpoint": str(args.checkpoint),
            "horizon": int(args.horizon),
            "context_candles": int(context.shape[0]),
            "quantile_levels": levels,
            "channels": list(CHANNELS),
            "candles": quantiles_to_records(q, levels),
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out_json}")
    if args.plot:
        plot_ohlc_forecast(
            context, q, levels, args.plot,
            actual_future=actual, history_bars=args.history_bars, title=args.title,
        )
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
