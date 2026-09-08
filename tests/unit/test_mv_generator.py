"""Reference multivariate generator (DEC-CA-0041): shape/determinism, and the
load-bearing property — channels are genuinely CROSS-PREDICTIVE (a parent's
lagged past lowers a child's forecast error), which is what earns reward under
per-variate GIFT scoring. A merely-correlated generator would not pass the last.
"""
from __future__ import annotations

import numpy as np

from cascade.interface.generator import check_series
from cascade.interface.mv_reference_generator import Generator


def _gen(seed=0, **over):
    g = Generator("", seed=seed)
    for k, v in over.items():
        setattr(g, f"_{k}", v)
    return g


def test_shape_channels_and_finite():
    g = _gen(length=300, min_channels=2, max_channels=4)
    out = list(g.generate(6))
    assert len(out) == 6
    for s in out:
        assert s.ndim == 2 and 2 <= s.shape[0] <= 4 and s.shape[1] == 300
        assert np.isfinite(s).all()


def test_deterministic_in_seed_and_index():
    a = list(_gen(seed=7, length=200).generate(3))
    b = list(_gen(seed=7, length=200).generate(3))
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[0], _gen(seed=8, length=200).generate(1).__next__())


def test_passes_check_series_at_a_raised_cap():
    s = next(iter(_gen(length=300, max_channels=4).generate(1)))
    # C>1 only passes once max_channels is raised (the Phase-4 gate); the array
    # itself is otherwise valid (finite, in-band length).
    check_series(s, min_length=64, max_length=4096, max_channels=8)


def test_supports_the_ablation_channel_sweep_up_to_32():
    """Toto2's variate layers train at up to 32 variates, so the generator must
    produce valid coupled corpora across the whole sweep. (Distinct from the
    eval-pool mv_channels <= 8 contract, which caps REAL harvested columns.)"""
    for c in (4, 8, 16, 32):
        g = _gen(seed=c, length=256, min_channels=c, max_channels=c)
        s = next(iter(g.generate(1)))
        assert s.shape == (c, 256), f"C={c} produced {s.shape}"
        assert np.isfinite(s).all()


def test_coupling_survives_at_wide_C():
    """Cross-predictiveness must hold at the wide end of the sweep too, not just
    at C=4 — a wide corpus of independent channels would earn no MV reward."""
    g = _gen(seed=11, length=800, min_channels=16, max_channels=16,
             coupling_strength=1.5, max_delay=12)
    h = 15
    gains = []
    for s in g.generate(2):
        c = s.shape[0]
        for k in range(1, c, 4):          # sample children across the width
            tgt = s[k][h:]
            own = _lag_matrix(s[k], h)
            allc = np.concatenate([_lag_matrix(s[j], h) for j in range(c)], axis=1)
            gains.append(_ridge_oos_mse(own, tgt) - _ridge_oos_mse(allc, tgt))
    assert np.mean(gains) > 0.0, f"no cross-predictive lift at C=16 ({np.mean(gains):.4g})"


def _lag_matrix(x, h):
    """Rows t=h..L-1, cols [x[t-1], …, x[t-h]]."""
    return np.stack([x[t - h:t][::-1] for t in range(h, len(x))], axis=0)


def _ridge_oos_mse(x_feats, y, alpha=1.0):
    n = len(y)
    cut = int(n * 0.7)
    xtr, ytr, xte, yte = x_feats[:cut], y[:cut], x_feats[cut:], y[cut:]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    a = xtr.T @ xtr + alpha * np.eye(xtr.shape[1])
    w = np.linalg.solve(a, xtr.T @ (ytr - ytr.mean()))
    return float(np.mean((yte - (xte @ w + ytr.mean())) ** 2))


def test_channels_are_cross_predictive_not_just_correlated():
    """Including sibling lags lowers a child channel's 1-step OOS MSE vs its own
    lags alone — averaged over children and series. This is exactly A1's
    admission test, applied to the generator's own output."""
    g = _gen(seed=3, length=800, min_channels=3, max_channels=3,
             coupling_strength=1.5, max_delay=12)
    h = 15  # > max_delay, so a parent's driving lag is in the feature window
    gains = []
    for s in g.generate(8):
        c = s.shape[0]
        for k in range(1, c):
            tgt = s[k][h:]
            own = _lag_matrix(s[k], h)
            allc = np.concatenate([_lag_matrix(s[j], h) for j in range(c)], axis=1)
            gains.append(_ridge_oos_mse(own, tgt) - _ridge_oos_mse(allc, tgt))
    assert np.mean(gains) > 0.0, f"sibling lags did not help (mean gain {np.mean(gains):.4g})"
