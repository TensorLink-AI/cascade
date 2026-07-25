"""Heat-screen shadow diagnostics: the joint bootstrap and what it reports.

These cover the two claims the diagnostics rest on — that the heat's ranking
metric is the same functional form the duel's LCB bounds, and that the
leader-vs-runner-up bound is literally the duel's statistic applied to the top
two entrants — plus the behaviour on decisive vs. coin-flip fields.
"""

from __future__ import annotations

import numpy as np

from cascade.eval.bootstrap import (
    _bag_geomeans,
    _cluster_sums,
    joint_bag_geomeans,
    paired_bootstrap_lcb_aggregated,
)
from cascade.eval.heat import screen_diagnostics
from cascade.eval.scoring import WindowScore, global_geomean, stack_components

N_WINDOWS = 60
ALPHA = 0.05
B = 2000


def _entrant(scale, seed):
    """One entrant's scores on a FIXED window set: ``abs_target`` is seeded off
    the windows, not the model, so every entrant is paired by construction —
    exactly how the heat screens (one slice, every challenger)."""
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
    otherwise the heat selects on one thing and the LCB bounds another."""
    scores = _entrant(1.0, seed=7)
    qloss, abs_t, mase = stack_components(scores)
    codes = np.arange(len(scores))
    identity = codes[None, :]  # one bag containing every window exactly once
    bag = _bag_geomeans(*_cluster_sums(qloss, abs_t, mase, codes, len(scores)), identity)
    assert np.isclose(bag[0], global_geomean(scores), rtol=1e-12)


def test_leader_bound_is_the_duel_statistic_on_the_top_two():
    """screen_diagnostics' leader_lcb must equal paired_bootstrap_lcb_aggregated
    with the runner-up in the king's slot — same draws, same number."""
    leader, runner = _entrant(1.0, seed=1), _entrant(1.35, seed=2)
    diag = screen_diagnostics(
        [("leader", leader), ("runner", runner)], seed=99, B=B, alpha=ALPHA
    )
    k_q, k_a, k_m = stack_components(runner)   # runner-up plays the king
    c_q, c_a, c_m = stack_components(leader)
    expected = paired_bootstrap_lcb_aggregated(
        k_q, k_a, k_m, c_q, c_a, c_m, alpha=ALPHA, B=B, seed=99,
        clusters=[f"__row{i}" for i in range(N_WINDOWS)],
    )
    assert np.isclose(diag.leader_lcb, expected, rtol=1e-12)


def test_decisive_field_separates_leader_from_the_rest():
    ranked = [
        ("a", _entrant(1.0, seed=11)),
        ("b", _entrant(1.6, seed=12)),
        ("c", _entrant(1.9, seed=13)),
    ]
    diag = screen_diagnostics(ranked, seed=5, B=B, alpha=ALPHA)
    assert diag.leader_key == "a" and diag.runner_up_key == "b"
    assert diag.p_best["a"] > 0.99          # wins essentially every bag
    assert diag.leader_lcb > 0.0            # reliably ahead of second
    assert diag.n_windows == N_WINDOWS
    assert diag.n_clusters == N_WINDOWS     # no source labels ⇒ singleton clusters


def test_coin_flip_field_is_reported_as_undecided():
    """Two entrants of the same quality: P(best) near half and a bound that does
    NOT clear zero. This is the case the screen currently cannot see."""
    ranked = [("a", _entrant(1.0, seed=21)), ("b", _entrant(1.0, seed=22))]
    diag = screen_diagnostics(ranked, seed=5, B=B, alpha=ALPHA)
    assert 0.2 < diag.p_best["a"] < 0.8
    assert diag.leader_lcb <= 0.0


def test_p_best_is_a_distribution_over_entrants():
    ranked = [("a", _entrant(1.0, seed=31)), ("b", _entrant(1.05, seed=32)),
              ("c", _entrant(1.1, seed=33))]
    diag = screen_diagnostics(ranked, seed=5, B=B, alpha=ALPHA)
    assert np.isclose(sum(diag.p_best.values()), 1.0)
    assert set(diag.p_best) == {"a", "b", "c"}


def _feed_correlated_pair(*, labelled):
    """Two entrants whose relative standing is driven by which FEED a window came
    from: within a feed the leader's advantage is constant, across feeds it swings
    hard. That is the correlation structure clustering exists for — the honest
    sample size is 4 feeds, not 60 windows.

    ``labelled`` controls only whether the ``source`` metadata is carried, so the
    two runs differ in the bootstrap's granularity and nothing else.
    """
    per_feed = 15
    advantage = [0.6, 0.9, 1.0, 1.4]  # runner-up's scale, by feed
    win = np.random.default_rng(1234)
    a, b = [], []
    for i in range(4 * per_feed):
        feed = i // per_feed
        abs_t = float(win.uniform(5.0, 10.0))
        src = f"feed{feed}" if labelled else None
        base_q = win.uniform(0.1, 1.0, size=9)
        base_m = float(win.uniform(0.5, 1.5))
        a.append(WindowScore(series_id=str(i), mase=base_m, qloss_per_q=base_q,
                             abs_target=abs_t, source=src))
        b.append(WindowScore(series_id=str(i), mase=base_m * advantage[feed],
                             qloss_per_q=base_q * advantage[feed],
                             abs_target=abs_t, source=src))
    return [("a", a), ("b", b)]


def test_clustered_windows_widen_the_bound():
    """Windows from one upstream feed are correlated in which model they favour;
    resampling feeds rather than windows shrinks the effective sample size, so
    the bound must loosen."""
    d_flat = screen_diagnostics(_feed_correlated_pair(labelled=False),
                                seed=5, B=B, alpha=ALPHA)
    d_clustered = screen_diagnostics(_feed_correlated_pair(labelled=True),
                                     seed=5, B=B, alpha=ALPHA)
    assert d_flat.n_clusters == 60 and d_clustered.n_clusters == 4
    assert d_clustered.leader_lcb < d_flat.leader_lcb


def test_single_entrant_has_no_runner_up_to_compare_against():
    diag = screen_diagnostics([("solo", _entrant(1.0, seed=51))], seed=5, B=B)
    assert diag.p_best == {"solo": 1.0}
    assert diag.leader_lcb is None and diag.runner_up_key is None


def test_unpaired_entrants_report_no_diagnostics_rather_than_raising():
    """Different window sets ⇒ the comparison is meaningless, but a diagnostic
    must never take down a heat."""
    a = _entrant(1.0, seed=61)
    b = [WindowScore(series_id=s.series_id, mase=s.mase, qloss_per_q=s.qloss_per_q,
                     abs_target=s.abs_target + 1.0) for s in _entrant(1.0, seed=62)]
    assert screen_diagnostics([("a", a), ("b", b)], seed=5, B=B) is None


def test_empty_and_scoreless_fields_are_none():
    assert screen_diagnostics([], seed=5, B=B) is None
    assert screen_diagnostics([("a", [])], seed=5, B=B) is None


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
