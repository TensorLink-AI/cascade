"""Lane B (DEC-CA-0050) margin + cluster floor: an unchanged expert never
merges, a one-source target is inconclusive, the loader refuses a zero bar."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.expert_lane import ExpertLaneParams, judge_expert_leg, paired_point_rel
from cascade.shared.config import load_chain_config, parse_lane_b

P = ExpertLaneParams(B=1000, min_target_windows=40, min_guard_windows=30)


def _pool(n_target=120, n_other=60, effect=1.0, n_sources=None, seed=0):
    rng = np.random.default_rng(seed)
    n = n_target + n_other
    doms = ["energy"] * n_target + ["web"] * n_other
    kq = rng.lognormal(0.0, 0.3, size=(n, 9))
    km = rng.lognormal(0.0, 0.3, size=n)
    f = np.array([effect if d == "energy" else 1.0 for d in doms])
    a = np.full(n, 50.0)
    clusters = None
    if n_sources is not None:
        clusters = [f"src{i % n_sources}" if d == "energy" else f"w{i}"
                    for i, d in enumerate(doms)]
    return dict(domains=doms, king_qloss=kq, king_abs_target=a, king_mase=km,
                chal_qloss=kq * f[:, None], chal_abs_target=a, chal_mase=km * f,
                clusters=clusters)


def test_identical_scores_are_rejected_even_at_zero_margin():
    for margin in (0.0, 0.01):
        v = judge_expert_leg(target_domain="energy", seed="s",
                             params=ExpertLaneParams(B=1000, min_target_windows=40,
                                                     min_guard_windows=30,
                                                     gain_margin=margin), **_pool())
        assert not v.passed and v.reason.startswith("no gain")
        assert v.target_gain == pytest.approx(0.0)


def test_one_source_hundred_windows_is_inconclusive():
    v = judge_expert_leg(target_domain="energy", params=P, seed="s",
                         **_pool(effect=0.5, n_sources=1))
    assert not v.passed and v.inconclusive == "clusters" and v.target_clusters == 1
    assert v.reason.startswith("inconclusive")
    ok = judge_expert_leg(target_domain="energy", params=P, seed="s",
                          **_pool(effect=0.9, n_sources=12))
    assert ok.passed, ok.reason
    assert ok.target_clusters == 12 and ok.target_gain == pytest.approx(0.1)


def test_point_rel_matches_a_uniform_effect():
    pool = _pool(effect=0.8)
    m = np.asarray(pool["domains"]) == "energy"
    rel = paired_point_rel(pool["king_qloss"][m], pool["king_abs_target"][m],
                           pool["king_mase"][m], pool["chal_qloss"][m],
                           pool["chal_abs_target"][m], pool["chal_mase"][m])
    assert rel == pytest.approx(0.2)


def test_lane_b_config_margin_and_cluster_floor():
    lb = parse_lane_b({})
    assert lb.gain_margin == 0.01 and lb.min_target_clusters == 5
    assert lb.verdict_params().min_target_clusters == 5
    assert lb.verdict_params().gain_margin == 0.01
    for bad in (0.0, -0.01, 1.0):
        with pytest.raises(ValueError, match="gain_margin"):
            parse_lane_b({"gain_margin": bad})
    with pytest.raises(ValueError, match="min_target_clusters"):
        parse_lane_b({"min_target_clusters": 0})


@pytest.mark.parametrize("name", ["chain.toml", "chain.testnet.toml"])
def test_shipped_tomls_carry_the_lane_b_margin(name):
    from pathlib import Path

    cfg = load_chain_config(Path(__file__).resolve().parents[2] / name)
    assert cfg.lane_b.gain_margin == 0.01 and cfg.lane_b.min_target_clusters == 5
    assert cfg.lane_c.min_target_clusters == 5
