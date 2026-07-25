"""Heat-cut separation — the observational read on the screen's finalist boundary."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.heat_cut import measure_heat_cut
from cascade.eval.scoring import WindowScore

ALPHA = 0.05
B = 1000
MARGIN = 0.02


def _scores(n, scale, seed, source=None):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(
            WindowScore(
                series_id=str(i),
                mase=float(rng.uniform(0.5, 1.5) * scale),
                qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
                abs_target=float(rng.uniform(5.0, 10.0)),
                source=(f"feed{i % source}" if source else None),
            )
        )
    return out


def _scaled(base, factor):
    """A paired entrant: same windows (same abs_target), uniformly better/worse."""
    return [
        WindowScore(s.series_id, s.mase * factor, s.qloss_per_q * factor,
                    s.abs_target, source=s.source)
        for s in base
    ]


def _measure(advanced, rejected, margin=MARGIN):
    return measure_heat_cut(advanced, rejected, alpha=ALPHA, B=B, seed=7, margin=margin)


def test_wide_cut_is_separated():
    rejected = _scores(256, 1.0, 0)
    advanced = _scaled(rejected, 0.6)          # 40% better — well clear of the margin
    cut = _measure(advanced, rejected)
    assert cut.separated
    assert cut.lcb >= cut.margin
    assert cut.observed > 0                    # advanced entrant scored lower
    assert cut.n_windows == 256


def test_coin_flip_cut_is_not_separated():
    """The case worth catching: two entrants the screen ordered on noise. The
    ranking still put one ahead; the diagnostic says that order isn't real."""
    rejected = _scores(256, 1.0, 0)
    advanced = _scaled(rejected, 0.999)        # a hair better, far inside the margin
    cut = _measure(advanced, rejected)
    assert not cut.separated
    assert cut.lcb < cut.margin


def test_lcb_p50_p95_are_ordered():
    rejected = _scores(200, 1.0, 3)
    advanced = _scaled(rejected, 0.8)
    cut = _measure(advanced, rejected)
    assert cut.lcb <= cut.p50 <= cut.p95


def test_clustered_windows_widen_the_bound():
    """Windows from a handful of feeds are correlated; the cluster bootstrap must
    not read them as independent evidence. Identical data both ways — only the
    ``source`` labels differ — and the improvement varies *by feed*, so treating
    240 rows as independent is exactly the overconfidence being tested for."""
    n, n_feeds = 240, 4
    rejected = _scores(n, 1.0, 5, source=n_feeds)
    # Per-feed improvement: 0.8 on one feed, ~1.0 on another. A row bootstrap
    # averages this away; a cluster bootstrap keeps whole feeds in or out.
    advanced = [
        WindowScore(s.series_id, s.mase * f, s.qloss_per_q * f, s.abs_target,
                    source=s.source)
        for s, f in ((s, 0.8 + 0.06 * (i % n_feeds)) for i, s in enumerate(rejected))
    ]
    clustered = _measure(advanced, rejected)
    # Same rows, cluster labels stripped ⇒ every row its own cluster.
    unlabelled = [
        WindowScore(s.series_id, s.mase, s.qloss_per_q, s.abs_target) for s in advanced
    ]
    flat = _measure(unlabelled,
                    [WindowScore(s.series_id, s.mase, s.qloss_per_q, s.abs_target)
                     for s in rejected])
    assert clustered.n_clusters == n_feeds
    assert flat.n_clusters == n
    assert clustered.lcb < flat.lcb


def test_margin_is_recorded_as_given():
    rejected = _scores(200, 1.0, 1)
    advanced = _scaled(rejected, 0.95)
    assert _measure(advanced, rejected, margin=0.5).margin == 0.5
    # Same cut, a bar it cannot clear.
    assert not _measure(advanced, rejected, margin=0.5).separated


def test_unpaired_lengths_raise():
    with pytest.raises(ValueError, match="unpaired heat scores"):
        _measure(_scores(10, 1.0, 0), _scores(11, 1.0, 0))


def test_mismatched_windows_raise():
    """Two entrants that somehow screened on different windows must not produce
    a number — the abs_target guard in the bootstrap catches it."""
    advanced = _scores(50, 1.0, 0)
    rejected = _scores(50, 1.0, 99)            # different abs_target draws
    with pytest.raises(ValueError, match="not paired"):
        _measure(advanced, rejected)


def test_deterministic_for_a_fixed_seed():
    rejected = _scores(200, 1.0, 2)
    advanced = _scaled(rejected, 0.85)
    a = measure_heat_cut(advanced, rejected, alpha=ALPHA, B=B, seed=11, margin=MARGIN)
    b = measure_heat_cut(advanced, rejected, alpha=ALPHA, B=B, seed=11, margin=MARGIN)
    assert (a.lcb, a.p50, a.p95) == (b.lcb, b.p50, b.p95)
