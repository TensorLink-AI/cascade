"""DEC-CA-0032 two-tier domain split.

The flat capped Dirichlet (alpha=4) realizes wildly uneven mixes (r40: nature
30.5% vs web_cloudops 4.0% with ~300 spare windows) and can starve scarce
domains below even their caps. The tier split pins scarce domains AT capacity
and divides the remainder evenly (tight alpha) over the rest, degenerating to
a tight even split over all domains once capacity floors land.
"""

from __future__ import annotations

import numpy as np

from cascade.validator.windows import MixParams, _capped_jittered_split, _tiered_split

# r40-era production capacities after the 0.7 series bag:
# econ_fin, energy, healthcare, nature, sales, transport, web_cloudops
R40_CAPS = [65, 290, 106, 345, 81, 420, 303]
N, BLOCK = 1200, 8


def test_scarce_domains_always_at_capacity() -> None:
    for trial in range(50):
        rng = np.random.default_rng(trial)
        counts = _tiered_split(N, R40_CAPS, 75.0, BLOCK, rng)
        assert counts[0] == 65    # econ_fin
        assert counts[4] == 81    # sales
        assert counts[2] == 106   # healthcare
        assert sum(counts) == N


def test_abundant_tier_stays_near_even() -> None:
    shares = []
    for trial in range(200):
        rng = np.random.default_rng(trial)
        counts = _tiered_split(N, R40_CAPS, 75.0, BLOCK, rng)
        residual = N - (65 + 81 + 106)
        for i in (1, 3, 5, 6):  # energy, nature, transport, web
            shares.append(counts[i] / residual)
    shares = np.asarray(shares)
    # Even share of the residual tier is 0.25; the tight alpha keeps realized
    # shares in a band the flat alpha=4 split blows through routinely.
    assert shares.min() > 0.14 and shares.max() < 0.36
    assert abs(shares.mean() - 0.25) < 0.01


def test_flat_split_can_starve_uncapped_domains_tier_cannot() -> None:
    # Document the failure mode the tier fixes: under the flat split some
    # trial gives an abundant domain (<cap) less than half its even share.
    def worst_abundant_share(split_fn, alpha):
        worst = 1.0
        for trial in range(300):
            rng = np.random.default_rng(trial)
            counts = split_fn(N, R40_CAPS, alpha, BLOCK, rng)
            for i in (1, 3, 5, 6):
                if counts[i] < R40_CAPS[i]:  # not cap-bound
                    worst = min(worst, counts[i] / N)
        return worst
    flat_worst = worst_abundant_share(_capped_jittered_split, 4.0)
    tier_worst = worst_abundant_share(_tiered_split, 75.0)
    assert flat_worst < 0.07     # flat split starves (r40 web: 4%)
    assert tier_worst > 0.10     # tier floor holds


def test_determinism_same_seed_same_counts() -> None:
    a = _tiered_split(N, R40_CAPS, 75.0, BLOCK, np.random.default_rng(7))
    b = _tiered_split(N, R40_CAPS, 75.0, BLOCK, np.random.default_rng(7))
    assert a == b


def test_degenerates_to_even_when_capacity_ample() -> None:
    caps = [400] * 7  # every domain clears the even share -> no scarce tier
    for trial in range(100):
        rng = np.random.default_rng(trial)
        counts = _tiered_split(N, caps, 75.0, BLOCK, rng)
        assert sum(counts) == N
        for c in counts:
            assert abs(c / N - 1 / 7) < 0.05  # uniform +/- a few points


def test_n_exceeding_total_capacity_clamps() -> None:
    caps = [10, 20, 30]
    counts = _tiered_split(10_000, caps, 75.0, 8, np.random.default_rng(0))
    assert counts == caps


def test_gating_defaults_off() -> None:
    mix = MixParams(from_block=100)
    assert mix.active(200)
    assert not mix.tier_active(200)          # tier_from_block defaults 0
    tiered = MixParams(from_block=100, tier_from_block=150)
    assert not tiered.tier_active(120)       # active but pre-tier boundary
    assert tiered.tier_active(150)
