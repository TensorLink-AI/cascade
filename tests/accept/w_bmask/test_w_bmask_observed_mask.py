"""W-BMASK: the bench sidecar's GIFT-Eval / GluonTS context convention.

Tests without the ``torch`` mark never import torch (production box:
``-m 'not torch'``). The ``torch`` tests run in CI on a random-init 2m
pyramid checkpoint.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import SIDECAR, load_observed, load_pinned_wrapper, series, tiny_checkpoint

torch_mark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)

CONTEXT = 4096
HORIZON = 48


# ── non-torch: the observed mask itself ──────────────────────────────────────


def test_short_history_padding_is_unobserved_and_filled_like_gluonts():
    obs = load_observed()
    h = series(100, seed=1)
    values, observed, counts = obs.observed_context([h], CONTEXT)
    assert values.shape == observed.shape == (1, CONTEXT)
    # InstanceSplitter: left pad, dummy_value 0.0, observed indicator 0.
    assert not observed[0, :CONTEXT - 100].any()
    assert (values[0, :CONTEXT - 100] == 0.0).all()
    assert observed[0, CONTEXT - 100:].all()
    np.testing.assert_array_equal(values[0, CONTEXT - 100:], h)
    assert counts == {"histories": 1, "padded": 1, "with_missing": 0,
                      "below_min_obs": 0, "all_missing": 0}


def test_nan_and_inf_are_unobserved_not_raised():
    obs = load_observed()
    h = series(CONTEXT + 10, seed=2)
    h[[CONTEXT - 5, CONTEXT, CONTEXT + 3]] = [np.nan, np.inf, -np.inf]
    values, observed, counts = obs.observed_context([h], CONTEXT)
    kept = h[-CONTEXT:]
    bad = ~np.isfinite(kept)
    assert bad.sum() == 3
    np.testing.assert_array_equal(observed[0], ~bad)
    assert (values[0][bad] == 0.0).all() and np.isfinite(values).all()
    assert counts["padded"] == 0 and counts["with_missing"] == 1


def test_all_missing_and_empty_histories_are_counted():
    obs = load_observed()
    _, observed, counts = obs.observed_context(
        [np.full(50, np.nan), np.zeros(0), np.r_[np.full(10, np.nan), 1.0, 2.0]], CONTEXT)
    assert not observed[:2].any() and observed[2].sum() == 2
    assert counts["all_missing"] == 2 and counts["below_min_obs"] == 3
    assert counts["padded"] == 3 and counts["histories"] == 3


def test_sidecar_loader_and_fev_hand_raw_history_to_the_adapter():
    pred = (SIDECAR / "predictor.py").read_text()
    assert "from .observed import adapt" in pred
    assert "return adapt(Wrapper(str(checkpoint_dir), device=device))" in pred
    fev = (SIDECAR / "suites" / "fev_bench.py").read_text()
    assert 'raw if getattr(inner[k], "handles_missing", False) else chunk' in fev
    res = (SIDECAR / "results.py").read_text()
    assert "gluonts-observed/1" in res


# ── torch: the adapter on a real pyramid checkpoint ──────────────────────────


@lru_cache(maxsize=1)
def _ckpt(root: str) -> Path:
    return tiny_checkpoint(Path(root) / "pyramid")


@pytest.fixture
def inner(tmp_path_factory):
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield load_pinned_wrapper(_ckpt(str(tmp_path_factory.getbasetemp() / "w_bmask_ckpt")))
    finally:
        torch.set_num_threads(threads)


@torch_mark
def test_adapt_wraps_the_pinned_pyramid_wrapper_only(inner):
    obs = load_observed()
    masked = obs.adapt(inner)
    assert isinstance(masked, obs.ObservedMaskWrapper)
    assert masked.handles_missing is True and masked.inner is inner
    assert masked.quantile_levels == list(inner.quantile_levels)

    class Other:
        quantile_levels = (0.5,)
    other = Other()
    assert obs.adapt(other) is other


@torch_mark
def test_full_length_finite_history_is_identical_to_the_pinned_wrapper(inner):
    masked = load_observed().adapt(inner)
    hs = [series(CONTEXT, seed=3), series(CONTEXT + 77, seed=4)]
    np.testing.assert_array_equal(masked.forecast_quantiles_batch(hs, HORIZON),
                                  inner.forecast_quantiles_batch(hs, HORIZON))


@torch_mark
def test_nan_history_forecasts_where_the_pinned_wrapper_raises(inner):
    masked = load_observed().adapt(inner)
    h = series(3000, seed=5)
    h[[10, 500, 2999]] = np.nan
    with pytest.raises(ValueError, match="non-finite forecast history"):
        inner.forecast_quantiles_batch([h], HORIZON)
    q = masked.forecast_quantiles_batch([h], HORIZON)
    assert q.shape == (1, HORIZON, len(masked.quantile_levels)) and np.isfinite(q).all()
    s = masked.forecast(h, HORIZON, 4)
    assert s.shape == (1, 4, HORIZON) and np.isfinite(s).all()
    a = masked.forecast_quantiles_batch([np.full(200, np.nan)], HORIZON)
    assert np.isfinite(a).all()
    assert masked.counts["all_missing"] >= 1 and masked.counts["with_missing"] >= 2


def _anchor(masked, history):
    """The (loc, scale) the adapter de-standardises the horizon with."""
    import torch

    with torch.no_grad():
        context, observed = masked._prep([history])
        _, loc, scale = masked._parameters(context[:, None], observed[:, None], HORIZON)
    return float(loc[0, 0, 0]), float(scale[0, 0, 0])


@torch_mark
@pytest.mark.parametrize("n_points, nan_at", [(300, ()), (1500, (7, 400, 1499)), (CONTEXT, (0, 99))])
def test_scaler_stats_equal_the_unpadded_observed_series(inner, n_points, nan_at):
    import torch

    masked = load_observed().adapt(inner)
    h = series(n_points, seed=6 + n_points)
    h[list(nan_at)] = np.nan
    kept = h[np.isfinite(h)]
    _, loc_ref, scale_ref = masked.m.causal_standardize(
        torch.as_tensor(kept[None], dtype=torch.float64))
    loc, scale = _anchor(masked, h)
    assert loc == pytest.approx(float(loc_ref[0, -1]), rel=1e-6, abs=1e-6)
    assert scale == pytest.approx(float(scale_ref[0, -1]), rel=1e-6, abs=1e-6)


@torch_mark
def test_first_value_padding_moves_the_pinned_wrappers_scaler(inner):
    """Control: the pinned wrapper's first-value pad counts as observed, so
    its scale is NOT the observed series' (which is why the adapter exists)."""
    import torch

    masked = load_observed().adapt(inner)
    h = series(300, seed=9)
    with torch.no_grad():
        ctx = inner._prep([h])[:, None]
        _, _, scale_pinned = inner._parameters(ctx, HORIZON)
    _, scale = _anchor(masked, h)
    assert abs(float(scale_pinned[0, 0, 0]) - scale) > 1e-3 * scale
