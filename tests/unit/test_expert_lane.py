"""Lane B (DEC-CA-0050): gain + non-inferiority guards, decay, emission split."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.expert_lane import (
    REST_GUARD,
    ExpertLaneParams,
    LaneBWin,
    judge_expert_leg,
    paired_geomean_lcb,
    reward_multiplier,
    split_emissions,
)

P = ExpertLaneParams(B=2000, min_target_windows=40, min_guard_windows=30, min_public_items=5)


def _pool(rng, domains_n: dict[str, int], effect: dict[str, float]):
    """King/candidate window arrays; ``effect[d]`` scales the candidate's loss on
    domain d (0.9 = 10% better, 1.05 = 5% worse)."""
    doms, kq, km = [], [], []
    for d, n in domains_n.items():
        doms += [d] * n
        kq.append(rng.lognormal(0.0, 0.3, size=(n, 9)))
        km.append(rng.lognormal(0.0, 0.3, size=n))
    kq, km = np.concatenate(kq), np.concatenate(km)
    f = np.array([effect.get(d, 1.0) for d in doms])
    noise = rng.lognormal(0.0, 0.02, size=len(doms))
    cq = kq * (f * noise)[:, None]
    cm = km * f * noise
    a = np.ones(len(doms)) * 50.0
    return dict(domains=doms, king_qloss=kq, king_abs_target=a, king_mase=km,
                chal_qloss=cq, chal_abs_target=a, chal_mase=cm)


def test_clean_domain_gain_merges():
    rng = np.random.default_rng(0)
    pool = _pool(rng, {"energy": 120, "web": 80, "nature": 80}, {"energy": 0.9})
    v = judge_expert_leg(target_domain="energy", params=P, seed="s", **pool)
    assert v.passed, v.reason
    assert v.target_lcb > 0.05
    assert set(v.guard_lcbs) == {"web", "nature"}


def test_regression_elsewhere_blocks():
    rng = np.random.default_rng(1)
    pool = _pool(rng, {"energy": 120, "web": 80, "nature": 80}, {"energy": 0.9, "web": 1.05})
    v = judge_expert_leg(target_domain="energy", params=P, seed="s", **pool)
    assert not v.passed and "domain web" in v.reason


def test_no_gain_blocks_and_small_target_is_inconclusive():
    rng = np.random.default_rng(2)
    pool = _pool(rng, {"energy": 120, "web": 80}, {})
    v = judge_expert_leg(target_domain="energy", params=P, seed="s", **pool)
    assert not v.passed and v.reason.startswith("no gain")
    pool = _pool(rng, {"energy": 10, "web": 80}, {"energy": 0.5})
    v = judge_expert_leg(target_domain="energy", params=P, seed="s", **pool)
    assert v.reason.startswith("inconclusive")


def test_small_domains_are_pooled_not_skipped():
    rng = np.random.default_rng(3)
    pool = _pool(rng, {"energy": 120, "sales": 20, "econ": 20}, {"energy": 0.9, "sales": 1.08,
                                                                 "econ": 1.08})
    v = judge_expert_leg(target_domain="energy", params=P, seed="s", **pool)
    assert REST_GUARD in v.guard_lcbs and v.unguarded == []
    assert not v.passed and REST_GUARD in v.reason


def test_public_benchmark_guard():
    rng = np.random.default_rng(4)
    pool = _pool(rng, {"energy": 120, "web": 80}, {"energy": 0.9})
    k = rng.lognormal(-0.3, 0.2, size=97)
    ok = {"gift_crps": (k, k * 0.999)}
    bad = {"gift_crps": (k, k * 1.04), "boom_mase": (k, k)}
    assert judge_expert_leg(target_domain="energy", public=ok, params=P, seed="s", **pool).passed
    v = judge_expert_leg(target_domain="energy", public=bad, params=P, seed="s", **pool)
    assert not v.passed and "gift_crps" in v.reason


def test_verdict_is_deterministic():
    rng = np.random.default_rng(5)
    pool = _pool(rng, {"energy": 120, "web": 80}, {"energy": 0.95})
    a = judge_expert_leg(target_domain="energy", params=P, seed="blk", **pool)
    b = judge_expert_leg(target_domain="energy", params=P, seed="blk", **pool)
    assert a == b


def test_paired_geomean_lcb_sign():
    k = np.full(50, 2.0)
    assert paired_geomean_lcb(k, k * 0.9, alpha=0.05, B=500, seed=1) == pytest.approx(0.1)
    assert paired_geomean_lcb(k, k * 1.1, alpha=0.05, B=500, seed=1) == pytest.approx(-0.1)


def test_reward_multiplier_schedule():
    m = [reward_multiplier(r, full_eras=5) for r in range(10)]
    assert m[:5] == [1.0] * 5
    assert m[5] == 0.5 and m[6] == 0.25 and m[7] == 0.125
    assert m[8:] == [0.1, 0.1]
    assert reward_multiplier(6, full_eras=5, half_life_rounds=2.0) == pytest.approx(0.5)


def test_split_emissions_burn_and_lane_a_remainder():
    lane_a = {7: 2.0, 8: 1.0}
    wins = [LaneBWin(uid=11, domain="energy", multiplier=1.0),
            LaneBWin(uid=12, domain="web", multiplier=0.25)]
    w = split_emissions(lane_a, wins, share_b=0.5, n_domain_slots=7, burn_uid=0)
    slot = 0.5 / 7
    assert w[11] == pytest.approx(slot) and w[12] == pytest.approx(slot * 0.25)
    assert w[7] == pytest.approx(0.5 * 2 / 3) and w[8] == pytest.approx(0.5 / 3)
    assert w[0] == pytest.approx(0.5 - slot * 1.25)
    w2 = split_emissions(lane_a, wins, share_b=0.5, n_domain_slots=7, burn_uid=0,
                         remainder="lane_a")
    assert 0 not in w2 and sum(w2.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        split_emissions(lane_a, wins + [LaneBWin(13, "energy", 1.0)], share_b=0.5,
                        n_domain_slots=7, burn_uid=0)
