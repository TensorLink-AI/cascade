"""Owner 2026-11-17 "do gift eval style": ``[marketcast] crps_scale =
"seasonal_naive"`` divides every window's CRPS by the GIFT-Eval (gluonts MASE)
seasonal-naive error of its own context, and the training loss uses the same
scale. ``"level"`` (the default) keeps today's divisors."""
from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.marketcast import cut_windows, gift_season, level_scale, seasonal_naive_scale
from cascade.shared.config.marketcast import parse_marketcast


def test_gift_seasonality_matches_gluonts():
    assert [gift_season(f) for f in (1, 5, 15, 60, 240, 1440, 10080)] == \
        [1440, 288, 96, 24, 6, 1, 1]
    assert gift_season(7) == 1                          # a day is not whole steps


def test_seasonal_naive_scale_is_the_mase_denominator():
    x = np.arange(100, dtype=float) % 24                # pure daily cycle at 1h
    assert seasonal_naive_scale(x, 60) == pytest.approx(np.abs(np.diff(x)).mean())  # → lag 1
    assert np.isnan(seasonal_naive_scale(np.zeros(100), 60))   # flat: no lag gives a scale
    y = np.cumsum(np.ones(60))
    assert seasonal_naive_scale(y, 60) == pytest.approx(24.0)
    assert seasonal_naive_scale(y[:10], 60) == pytest.approx(1.0)   # shorter than a season → m=1
    z = y.copy()
    z[:5] = np.nan
    assert seasonal_naive_scale(z, 60) == pytest.approx(24.0)       # NaN pairs skipped


def test_config_option():
    assert parse_marketcast({}).crps_scale == "seasonal_naive"
    assert parse_marketcast({"crps_scale": "level"}).crps_scale == "level"
    assert parse_marketcast({"crps_scale": "seasonal_naive"}).crps_scale == "seasonal_naive"
    with pytest.raises(ValueError, match="crps_scale"):
        parse_marketcast({"crps_scale": "mase"})


def _cut(mode, suite="ohlcv.c"):
    s = 100.0 + np.cumsum(np.random.default_rng(0).standard_normal(400))
    return cut_windows(s, cls="crypto", asset="BTC", suite=suite, horizon="1h", freq_minutes=60,
                       steps=1, t0=0, context_length=200, after=-1, now=400 * 3600,
                       crps_scale=mode)


def test_cut_windows_uses_the_chosen_divisor():
    lv, sn = _cut("level"), _cut("seasonal_naive")
    assert [w.wid for w in lv] == [w.wid for w in sn]
    for a, b in zip(lv, sn, strict=True):
        assert a.scale == level_scale("ohlcv.c", a.context)
        assert b.scale == pytest.approx(seasonal_naive_scale(b.context, 60))
    assert all(w.scale == 1.0 for w in _cut("level", "returns"))
    assert all(w.scale != 1.0 for w in _cut("seasonal_naive", "returns"))


def test_training_loss_default_unchanged_and_seasonal_rescales():
    torch = pytest.importorskip("torch")
    from cascade.trainer.marketcast_leg import quantile_loss
    rng = np.random.default_rng(1)
    ctx = np.cumsum(rng.standard_normal((3, 100)), 1)
    y = ctx[:, -1:] + rng.standard_normal((3, 2))
    pred = torch.zeros(3, 2, 3) + torch.as_tensor(ctx[:, -1])[:, None, None]
    lv, w = (0.1, 0.5, 0.9), (1.0, 1.0, 1.0)
    a = quantile_loss(pred, y, ctx, lv, w)
    assert torch.equal(a, quantile_loss(pred, y, ctx, lv, w, freq_minutes=None))
    b = quantile_loss(pred, y, ctx, lv, w, freq_minutes=60)
    assert torch.isfinite(b) and not torch.isclose(a, b)


def test_flat_context_keeps_its_window_on_the_level_scale():
    # red-team F3: the realised series never drops a window; a flat context
    # has no seasonal-naive error, so it falls back to the level scale
    ws = cut_windows(np.zeros(400), cls="crypto", asset="BTC", suite="returns", horizon="1h",
                     freq_minutes=60, steps=1, t0=0, context_length=200, after=-1,
                     now=400 * 3600, crps_scale="seasonal_naive")
    assert ws and all(w.scale == 1.0 for w in ws)
