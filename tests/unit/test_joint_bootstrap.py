"""The joint bootstrap the cohort max-T correction rides on.

Covers the claim the bound rests on — that the duel's ranking metric is the
same functional form a bootstrap bag computes — and that the joint resample is
paired across entrants (one shared draw, same windows or it refuses).
"""

from __future__ import annotations

import numpy as np

from cascade.eval.bootstrap import (
    _bag_geomeans,
    _cluster_sums,
    joint_bag_geomeans,
)
from cascade.eval.scoring import WindowScore, global_geomean, stack_components

N_WINDOWS = 60


def _entrant(scale, seed):
    """One entrant's scores on a FIXED window set: ``abs_target`` is seeded off
    the windows, not the model, so every entrant is paired by construction —
    exactly how a duel scores (one slice, king and every challenger)."""
    rng = np.random.default_rng(seed)
    win = np.random.default_rng(1234)  # window-level draw: identical per entrant
    out = []
    for i in range(N_WINDOWS):
        out.append(
            WindowScore(
                series_id=str(i),
                mase=float(rng.uniform(0.5, 1.5) * scale),
                qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
                abs_target=float(win.uniform(5.0, 10.0)),
            )
        )
    return out


def test_global_geomean_is_one_bootstrap_bag_on_the_identity_resample():
    """The ranking metric and the resampled metric must be the same function —
    otherwise the duel ranks on one thing and the LCB bounds another."""
    scores = _entrant(1.0, seed=7)
    qloss, abs_t, mase = stack_components(scores)
    codes = np.arange(len(scores))
    identity = codes[None, :]  # one bag containing every window exactly once
    bag = _bag_geomeans(*_cluster_sums(qloss, abs_t, mase, codes, len(scores)), identity)
    assert np.isclose(bag[0], global_geomean(scores), rtol=1e-12)


def test_joint_bags_share_one_draw_across_entrants():
    """The shared resample is what makes within-bag comparison paired: feeding
    the SAME entrant twice must produce two identical rows."""
    scores = _entrant(1.0, seed=71)
    comps = stack_components(scores)
    bags = joint_bag_geomeans([comps, comps], B=200, seed=3, clusters=None)
    assert bags.shape == (2, 200)
    assert np.array_equal(bags[0], bags[1])


def test_joint_bags_reject_entrants_scored_on_different_windows():
    a = stack_components(_entrant(1.0, seed=81))
    q, abs_t, m = stack_components(_entrant(1.0, seed=82))
    try:
        joint_bag_geomeans([a, (q, abs_t + 1.0, m)], B=50, seed=3)
    except ValueError as e:
        assert "same windows" in str(e)
    else:
        raise AssertionError("expected unpaired entrants to raise")
