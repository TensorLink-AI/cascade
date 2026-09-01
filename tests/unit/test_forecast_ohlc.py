"""Tests for scripts/forecast_ohlc.py — OHLC per-candle quantile forecasting.

The script batches the four candle channels through a checkpoint wrapper's
quantile head and projects the marginals onto valid candle geometry. These
tests drive it with fake wrappers (no torch, no checkpoint): channel routing,
the coherence repair, the pre-CPM sample fallback, CSV IO, and the JSON shape.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "forecast_ohlc.py"
_spec = importlib.util.spec_from_file_location("forecast_ohlc", _SCRIPT)
forecast_ohlc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = forecast_ohlc
_spec.loader.exec_module(forecast_ohlc)

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def make_candles(n: int, base: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    close = base + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    return np.stack([open_, high, low, close], axis=1)


class QuantileWrapper:
    """CPM-style fake: quantiles = last value + level offset + channel offset."""

    quantile_levels = LEVELS

    def __init__(self):
        self.calls = []

    def forecast_quantiles_batch(self, histories, horizon):
        self.calls.append(([np.asarray(h).copy() for h in histories], horizon))
        out = np.empty((len(histories), horizon, len(LEVELS)))
        for b, h in enumerate(histories):
            anchor = float(np.asarray(h)[-1])
            for t in range(horizon):
                out[b, t] = anchor + np.linspace(-1, 1, len(LEVELS)) + 0.01 * b
        return out


class SampleWrapper:
    """Pre-CPM fake: validator contract only, deterministic spread of paths."""

    def forecast(self, history, horizon, num_samples):
        anchor = float(np.asarray(history)[-1])
        spread = np.linspace(-1, 1, num_samples)[:, None]  # (S, 1)
        return (anchor + np.tile(spread, (1, horizon)))[None]  # (1, S, horizon)


def test_channels_routed_in_ohlc_order_one_batched_call():
    candles = make_candles(50)
    w = QuantileWrapper()
    q, levels = forecast_ohlc.forecast_ohlc_quantiles(w, candles, 8)
    assert len(w.calls) == 1
    histories, horizon = w.calls[0]
    assert horizon == 8
    for i, h in enumerate(histories):
        np.testing.assert_array_equal(h, candles[:, i])
    assert q.shape == (8, 4, 9)
    assert levels == LEVELS
    # open/close bodies pass through the repair untouched
    np.testing.assert_allclose(q[:, 0, 4], candles[-1, 0], atol=1e-12)
    np.testing.assert_allclose(q[:, 3, 4], candles[-1, 3] + 0.03, atol=1e-12)


def test_coherence_repair_envelopes_and_sorts():
    # high channel below everything, low channel above everything, and one
    # channel with deliberately crossed (descending) levels
    q = np.zeros((2, 4, 3))
    q[:, 0] = [1.0, 2.0, 3.0]  # open
    q[:, 1] = [0.0, 0.0, 0.0]  # high (too low)
    q[:, 2] = [9.0, 9.0, 9.0]  # low (too high)
    q[:, 3] = [4.0, 2.5, 0.5]  # close, crossed levels
    out = forecast_ohlc.repair_ohlc_coherence(q)
    assert (np.diff(out, axis=-1) >= 0).all()  # monotone per channel
    hi, lo = out[:, 1], out[:, 2]
    for c in (0, 3):
        assert (hi >= out[:, c]).all()
        assert (lo <= out[:, c]).all()
    assert (lo <= hi).all()
    # close was sorted before the envelope was taken
    np.testing.assert_array_equal(out[:, 3], [[0.5, 2.5, 4.0], [0.5, 2.5, 4.0]])
    # high = max(open, high, close); low = min(open, low, close) — the bogus
    # low marginal (9.0) is repaired without inflating the high wick
    np.testing.assert_array_equal(hi, [[1.0, 2.5, 4.0], [1.0, 2.5, 4.0]])
    np.testing.assert_array_equal(lo, [[0.5, 2.0, 3.0], [0.5, 2.0, 3.0]])


def test_repair_rejects_bad_shape():
    with pytest.raises(ValueError, match="horizon, 4, num_q"):
        forecast_ohlc.repair_ohlc_coherence(np.zeros((3, 5, 9)))


def test_sample_fallback_takes_empirical_quantiles_at_default_grid():
    candles = make_candles(30)
    q, levels = forecast_ohlc.forecast_ohlc_quantiles(
        SampleWrapper(), candles, 4, num_samples=101
    )
    assert levels == list(forecast_ohlc.DEFAULT_QUANTILE_LEVELS)
    assert q.shape == (4, 4, 9)
    # paths are uniform on anchor±1 → q0.5 ≈ anchor, q0.9 ≈ anchor+0.8 (close ch.)
    anchor = candles[-1, 3]
    np.testing.assert_allclose(q[:, 3, 4], anchor, atol=0.05)
    np.testing.assert_allclose(q[:, 3, 8], anchor + 0.8, atol=0.05)
    # candle geometry holds even though channels were sampled independently
    assert (q[:, 1] >= q[:, 0]).all() and (q[:, 1] >= q[:, 3]).all()
    assert (q[:, 2] <= q[:, 0]).all() and (q[:, 2] <= q[:, 3]).all()


def test_forecast_input_validation():
    w = QuantileWrapper()
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        forecast_ohlc.forecast_ohlc_quantiles(w, np.zeros((10, 3)), 4)
    with pytest.raises(ValueError, match="at least 2"):
        forecast_ohlc.forecast_ohlc_quantiles(w, np.zeros((1, 4)), 4)
    with pytest.raises(ValueError, match="horizon"):
        forecast_ohlc.forecast_ohlc_quantiles(w, make_candles(10), 0)

    class BadShapeWrapper(QuantileWrapper):
        def forecast_quantiles_batch(self, histories, horizon):
            return np.zeros((4, horizon + 1, len(LEVELS)))

    with pytest.raises(ValueError, match="forecast_quantiles_batch returned"):
        forecast_ohlc.forecast_ohlc_quantiles(BadShapeWrapper(), make_candles(10), 4)


def test_load_candles_case_insensitive_and_remapped(tmp_path):
    p = tmp_path / "candles.csv"
    p.write_text("time,Open,HIGH,lo,Close\n1,10,11,9,10.5\n2,10.5,12,10,11\n\n")
    got = forecast_ohlc.load_candles(p, {"low": "lo"})
    np.testing.assert_array_equal(got, [[10, 11, 9, 10.5], [10.5, 12, 10, 11]])


def test_load_candles_errors(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("open,high,close\n1,2,3\n")
    with pytest.raises(ValueError, match="missing column"):
        forecast_ohlc.load_candles(p)
    p.write_text("open,high,low,close\n1,2,nan,3\n")
    with pytest.raises(ValueError, match="non-finite"):
        forecast_ohlc.load_candles(p)
    p.write_text("open,high,low,close\n1,2,x,3\n")
    with pytest.raises(ValueError, match="bad candle row"):
        forecast_ohlc.load_candles(p)
    p.write_text("open,high,low,close\n")
    with pytest.raises(ValueError, match="no candle rows"):
        forecast_ohlc.load_candles(p)


def test_quantile_records_shape():
    q, levels = forecast_ohlc.forecast_ohlc_quantiles(QuantileWrapper(), make_candles(20), 3)
    recs = forecast_ohlc.quantiles_to_records(q, levels)
    assert [r["step"] for r in recs] == [1, 2, 3]
    assert set(recs[0]) == {"step", "open", "high", "low", "close"}
    assert list(recs[0]["close"]) == [f"{lv:g}" for lv in LEVELS]
    assert recs[0]["close"]["0.5"] == pytest.approx(q[0, 3, 4])
    json.dumps(recs)  # JSON-serializable end to end


def test_cli_json_end_to_end(tmp_path, monkeypatch):
    csv_path = tmp_path / "candles.csv"
    candles = make_candles(40)
    lines = ["open,high,low,close"] + [",".join(f"{v:.6f}" for v in row) for row in candles]
    csv_path.write_text("\n".join(lines) + "\n")
    out = tmp_path / "out.json"
    monkeypatch.setattr(forecast_ohlc, "load_wrapper", lambda d, device="cpu": QuantileWrapper())
    rc = forecast_ohlc.main([
        "--checkpoint", str(tmp_path), "--candles", str(csv_path),
        "--horizon", "5", "--holdout", "8", "--out-json", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["horizon"] == 5
    assert payload["context_candles"] == 32  # holdout removed from context
    assert payload["quantile_levels"] == LEVELS
    assert len(payload["candles"]) == 5


def test_cli_holdout_must_leave_context(tmp_path, monkeypatch):
    csv_path = tmp_path / "candles.csv"
    lines = ["open,high,low,close"] + ["1,2,0.5,1.5"] * 5
    csv_path.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(forecast_ohlc, "load_wrapper", lambda d, device="cpu": QuantileWrapper())
    with pytest.raises(SystemExit):
        forecast_ohlc.main([
            "--checkpoint", str(tmp_path), "--candles", str(csv_path),
            "--holdout", "5", "--out-json", str(tmp_path / "o.json"),
        ])
