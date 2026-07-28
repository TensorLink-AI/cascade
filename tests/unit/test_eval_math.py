"""CRPS (MWSQL) + MASE sanity and the paired bootstrap."""

from __future__ import annotations

import numpy as np

from cascade.eval.bootstrap import paired_bootstrap_lcb, paired_bootstrap_lcb_aggregated
from cascade.eval.crps import (
    geomean_wql_from_components,
    mwsql_components,
    mwsql_from_components,
    wql_per_window,
)
from cascade.eval.mase import in_sample_naive_mae, mase


def test_mwsql_perfect_forecast_is_zero():
    obs = np.array([[1.0, 2.0, 3.0]])
    samples = np.broadcast_to(obs[:, None, :], (1, 50, 3)).copy()
    qloss, abs_t = mwsql_components(samples, obs)
    assert mwsql_from_components(qloss, abs_t) < 1e-9


def test_mwsql_worse_forecast_scores_higher():
    obs = np.ones((1, 4))
    good = np.broadcast_to(obs[:, None, :], (1, 50, 4)).copy() + 0.05 * np.random.default_rng(0).standard_normal((1, 50, 4))
    bad = good + 5.0
    g = mwsql_from_components(*mwsql_components(good, obs))
    b = mwsql_from_components(*mwsql_components(bad, obs))
    assert b > g


def test_mase_perfect_is_zero_and_scale_floor():
    hist = np.arange(50, dtype=np.float64)
    obs = np.array([50.0, 51.0, 52.0])
    assert mase(obs.copy(), obs, hist, seasonal_period=1) == 0.0
    assert in_sample_naive_mae(np.zeros(50), 1) >= 1e-9


def test_paired_bootstrap_lcb_detects_clear_winner():
    rng = np.random.default_rng(0)
    king = rng.uniform(1.0, 2.0, size=400)
    chal = king * 0.7  # challenger 30% better on every window
    lcb = paired_bootstrap_lcb(king, chal, alpha=0.05, B=2000, seed="x")
    assert lcb > 0.2


def test_paired_bootstrap_lcb_no_improvement_is_nonpositive():
    rng = np.random.default_rng(1)
    king = rng.uniform(1.0, 2.0, size=400)
    chal = king.copy()
    lcb = paired_bootstrap_lcb(king, chal, alpha=0.05, B=2000, seed="x")
    assert lcb <= 1e-6


def test_paired_bootstrap_is_deterministic_in_seed():
    rng = np.random.default_rng(2)
    king = rng.uniform(1.0, 2.0, size=200)
    chal = king * 0.9
    a = paired_bootstrap_lcb(king, chal, seed="block-hash-abc", B=1000)
    b = paired_bootstrap_lcb(king, chal, seed="block-hash-abc", B=1000)
    assert a == b


def _components(n, num_q, scale, seed):
    rng = np.random.default_rng(seed)
    qloss = rng.uniform(0.1, 1.0, size=(n, num_q)) * scale
    abs_t = rng.uniform(5.0, 10.0, size=n)
    mase_a = rng.uniform(0.5, 1.5, size=n) * scale
    return qloss, abs_t, mase_a


def test_aggregated_lcb_requires_paired_abs_target():
    k_q, k_a, k_m = _components(50, 9, 1.0, 0)
    c_q, _, c_m = _components(50, 9, 0.5, 1)
    # Same abs_target (paired windows) → fine.
    lcb = paired_bootstrap_lcb_aggregated(k_q, k_a, k_m, c_q, k_a, c_m, B=500, seed="s")
    assert lcb > 0.0  # challenger has lower qloss/mase scale


def test_aggregated_lcb_rejects_unpaired_windows():
    import pytest

    k_q, k_a, k_m = _components(50, 9, 1.0, 0)
    c_q, c_a, c_m = _components(50, 9, 0.5, 1)  # different abs_target
    with pytest.raises(ValueError):
        paired_bootstrap_lcb_aggregated(k_q, k_a, k_m, c_q, c_a, c_m, B=200, seed="s")


def test_singleton_clusters_match_default_bootstrap():
    rng = np.random.default_rng(7)
    n, nq = 60, 9
    k_q = rng.uniform(0.1, 1.0, size=(n, nq))
    abs_t = rng.uniform(5.0, 10.0, size=n)
    k_m = rng.uniform(0.5, 1.5, size=n)
    c_q, c_m = k_q * 0.9, k_m * 0.9
    base = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="s")
    singletons = paired_bootstrap_lcb_aggregated(
        k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="s", clusters=list(range(n))
    )
    assert base == singletons


def test_cluster_bootstrap_is_wider_under_cluster_correlation():
    """Windows that move together must not be counted as independent: with a
    strong per-cluster effect the cluster LCB sits below the i.i.d. LCB."""
    rng = np.random.default_rng(11)
    n_clusters, per = 8, 40
    n = n_clusters * per
    labels = [f"feed{i // per}" for i in range(n)]
    k_q = rng.uniform(0.4, 0.6, size=(n, 9))
    abs_t = rng.uniform(5.0, 10.0, size=n)
    k_m = rng.uniform(0.9, 1.1, size=n)
    # Challenger improvement varies BY CLUSTER (correlated), not by window.
    effect = np.repeat(rng.uniform(0.7, 1.1, size=n_clusters), per)
    c_q = k_q * effect[:, None]
    c_m = k_m * effect
    iid = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=2000, seed="s")
    clustered = paired_bootstrap_lcb_aggregated(
        k_q, abs_t, k_m, c_q, abs_t, c_m, B=2000, seed="s", clusters=labels
    )
    assert clustered < iid


def test_cluster_bootstrap_deterministic_in_seed_and_labels():
    rng = np.random.default_rng(3)
    n = 40
    k_q = rng.uniform(0.1, 1.0, size=(n, 9))
    abs_t = rng.uniform(5.0, 10.0, size=n)
    k_m = rng.uniform(0.5, 1.5, size=n)
    # Heterogeneous improvement so the bag distribution isn't a point mass
    # (a constant factor cancels in every bag, making all seeds agree).
    factor = rng.uniform(0.6, 1.0, size=n)
    c_q, c_m = k_q * factor[:, None], k_m * factor
    labels = [f"s{i % 5}" for i in range(n)]
    a = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="x", clusters=labels)
    b = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="x", clusters=labels)
    assert a == b
    c = paired_bootstrap_lcb_aggregated(k_q, abs_t, k_m, c_q, abs_t, c_m, B=500, seed="y", clusters=labels)
    assert a != c


# --- scale-invariant CRPS aggregation (2026-07-28) --------------------------
# Regression cover for the defect that let three high-magnitude windows become
# 100% of the CRPS denominator on a 2000-window round. See
# cascade.eval.crps.wql_per_window.


def test_wql_per_window_masks_zero_target_windows():
    qloss = np.array([[1.0, 1.0], [1.0, 1.0]])
    abs_t = np.array([4.0, 0.0])
    wql, valid = wql_per_window(qloss, abs_t)
    assert valid.tolist() == [True, False]
    assert wql[0] == 0.5 and wql[1] == 0.0


def test_geomean_wql_ignores_undefined_windows():
    """A zero-|y| window must not enter the aggregate — flooring its denominator
    would inject ~1e11 into a log-space mean and swamp it."""
    qloss = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    abs_t = np.array([4.0, 4.0, 0.0])
    assert geomean_wql_from_components(qloss, abs_t) == geomean_wql_from_components(
        qloss[:2], abs_t[:2]
    )
    assert np.isnan(geomean_wql_from_components(qloss[2:], abs_t[2:]))


def test_geomean_wql_is_scale_invariant_but_pooled_is_not():
    """Rescaling ONE window's magnitude must not move a scale-invariant
    aggregate. The pooled MWSQL moves a long way — that is the bug."""
    # Two windows the model does DIFFERENTLY well on (wql 0.2 vs 0.6) — if the
    # ratios matched there would be nothing for a reweighting to shift.
    qloss = np.array([[1.0, 1.0], [6.0, 6.0]])
    abs_t = np.array([10.0, 20.0])
    big_qloss = qloss.copy()
    big_abs = abs_t.copy()
    big_qloss[1] *= 1e6          # same window, same ratio, 1e6x the magnitude
    big_abs[1] *= 1e6
    assert np.isclose(
        geomean_wql_from_components(qloss, abs_t),
        geomean_wql_from_components(big_qloss, big_abs),
    )
    assert not np.isclose(
        mwsql_from_components(qloss, abs_t),
        mwsql_from_components(big_qloss, big_abs),
        rtol=0.05,
    )


def test_one_huge_window_cannot_decide_the_duel():
    """Challenger is better on 99 windows and worse on one 1e9x-magnitude
    window. Pooled MWSQL hands the round to the king on that single window;
    the scale-invariant aggregate does not."""
    n = 100
    nq = 2
    abs_t = np.full(n, 10.0)
    abs_t[0] = 1e10                                   # one giant-magnitude series
    king_q = np.full((n, nq), 0.30) * abs_t[:, None]
    chal_q = np.full((n, nq), 0.20) * abs_t[:, None]  # challenger better everywhere
    chal_q[0] = 0.90 * abs_t[0]                       # ...except on the giant one
    king_mase = np.full(n, 1.0)
    chal_mase = np.full(n, 1.0)

    pooled = paired_bootstrap_lcb_aggregated(
        king_q, abs_t, king_mase, chal_q, abs_t, chal_mase,
        B=2000, seed=7, wql_mode="pooled",
    )
    geo = paired_bootstrap_lcb_aggregated(
        king_q, abs_t, king_mase, chal_q, abs_t, chal_mase,
        B=2000, seed=7, wql_mode="geomean",
    )
    assert pooled < 0.0 < geo


def test_bootstrap_default_is_the_scale_invariant_rule():
    rng = np.random.default_rng(3)
    n, nq = 40, 2
    abs_t = rng.uniform(1.0, 5.0, size=n)
    king_q = rng.uniform(0.2, 0.4, size=(n, nq)) * abs_t[:, None]
    chal_q = rng.uniform(0.1, 0.3, size=(n, nq)) * abs_t[:, None]
    mase_a = np.full(n, 1.0)
    kw = dict(B=500, seed=11)
    default = paired_bootstrap_lcb_aggregated(
        king_q, abs_t, mase_a, chal_q, abs_t, mase_a, **kw
    )
    explicit = paired_bootstrap_lcb_aggregated(
        king_q, abs_t, mase_a, chal_q, abs_t, mase_a, wql_mode="geomean", **kw
    )
    assert default == explicit
