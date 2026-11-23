"""W5i: the clamped combined statistic in the verdict (``judge_cohort`` and
the max-T cohort LCB of ``cohort_maxt_lcbs``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from .helpers import judge, lane_d_cfg, ref_gain

N_FWD, N_BT = 100, 5


def _king(n, seed=1):
    rng = np.random.default_rng(seed)
    return (1.0 + 0.1 * rng.random(n), 1.0 + 0.1 * rng.random(n))


def _clusters(n):
    return [f"src{i % 20}" for i in range(n)]


def _kinds(n_fwd=N_FWD, n_bt=N_BT):
    return ["forward"] * n_fwd + ["backtest"] * n_bt


def test_w5i_all_forward_is_byte_identical_to_today():
    """c = 0 with no backtest windows: today's gain formula and the same LCB."""
    from cascade.eval.bootstrap import cohort_maxt_lcbs

    cfg = lane_d_cfg()
    kc, km = _king(N_FWD)
    member = (kc * 0.97, km * 0.98)
    cl = _clusters(N_FWD)
    out = judge((kc, km), [member], clusters=cl, cfg=cfg, seed="w5i", kinds=["forward"] * N_FWD)
    base = judge((kc, km), [member], clusters=cl, cfg=cfg, seed="w5i")
    assert out[0]["lcb"] == base[0]["lcb"] and out[0]["gain"] == base[0]["gain"]

    def g(c, m):
        return math.sqrt(math.exp(float(np.mean(np.log(np.maximum(c, 1e-12)))))
                         * math.exp(float(np.mean(np.log(np.maximum(m, 1e-9))))))

    kg = g(kc, km)
    assert out[0]["gain"] == (kg - g(*member)) / kg

    def comp(c, m):
        return ((np.asarray(c) / 2.0)[:, None], np.ones(len(c)), np.asarray(m))

    lcb = cohort_maxt_lcbs(comp(kc, km), [comp(*member)], alpha=float(cfg.alpha),
                           B=int(cfg.bootstrap_B), seed="w5i", clusters=cl)[0]
    assert out[0]["lcb"] == float(lcb)


def test_w5i_combined_gain_formula():
    cfg = lane_d_cfg()
    kc, km = _king(N_FWD + N_BT)
    cc, cm = kc * 0.96, km * 0.97
    cc[N_FWD:] = kc[N_FWD:] * 0.9
    cm[N_FWD:] = km[N_FWD:] * 0.9
    out = judge((kc, km), [(cc, cm)], clusters=_clusters(N_FWD + N_BT), cfg=cfg, seed="w5i",
                kinds=_kinds())
    want = ref_gain((kc, km), (cc, cm), _kinds(), cap=0.05, floor=0.85)
    assert out[0]["gain"] == pytest.approx(want, rel=1e-10, abs=1e-12)
    assert out[0]["forward"]["n"] == N_FWD and out[0]["backtest"]["n"] == N_BT


def test_w5i_cap_0_gives_backtest_windows_no_weight():
    cfg = lane_d_cfg(backtest_weight_cap=0.0)
    kc, km = _king(N_FWD + N_BT)
    cc, cm = kc * 0.99, km * 0.99
    cc[N_FWD:] = 1e-12
    out = judge((kc, km), [(cc, cm)], clusters=_clusters(N_FWD + N_BT), cfg=cfg, seed="w5i",
                kinds=_kinds())
    want = ref_gain((kc, km), (cc, cm), _kinds(), cap=0.0, floor=0.85)
    assert out[0]["gain"] == pytest.approx(want, rel=1e-10, abs=1e-12)


def test_w5i_perfect_memorisation_never_clears():
    """Forward identical, backtest memorised (CRPS at the floor): the gain and
    the LCB are bounded by 1 - floor ** c < d_margin."""
    cfg = lane_d_cfg()
    kc, km = _king(N_FWD + N_BT)
    cc, cm = kc.copy(), km.copy()
    cc[N_FWD:] = 1e-12
    cm[N_FWD:] = 1e-9
    out = judge((kc, km), [(cc, cm)], clusters=_clusters(N_FWD + N_BT), cfg=cfg, seed="w5i",
                kinds=_kinds())
    c = min(0.05, N_BT / (N_FWD + N_BT))
    bound = 1 - 0.85 ** c
    assert out[0]["gain"] <= bound + 1e-12
    # A replicate's own c never exceeds the cap, so no LCB passes 1 - floor ** cap.
    assert out[0]["lcb"] <= 1 - 0.85 ** 0.05 + 1e-9
    assert out[0]["cleared"] is False


def test_w5i_clamp_applies_inside_every_bootstrap_replicate():
    """A uniform 5% forward gain plus wildly mixed backtest windows: clamped
    in each replicate the LCB stays near the point gain; an unclamped
    replicate spread would push it far below."""
    cfg = lane_d_cfg()
    kc, km = _king(N_FWD + N_BT)
    cc, cm = kc * 0.95, km * 0.95
    for j in range(N_BT):
        cc[N_FWD + j] = 1e-12 if j % 2 == 0 else 1e6
        cm[N_FWD + j] = 1e-9 if j % 2 == 0 else 1e6
    cl = [f"src{i}" for i in range(N_FWD + N_BT)]
    out = judge((kc, km), [(cc, cm)], clusters=cl, cfg=cfg, seed="w5i", kinds=_kinds())
    t = out[0]["gain"]
    assert t == pytest.approx(ref_gain((kc, km), (cc, cm), _kinds(), cap=0.05, floor=0.85),
                              rel=1e-10, abs=1e-12)
    assert out[0]["lcb"] >= t - 0.02, (out[0]["lcb"], t)


def test_w5i_verdict_is_deterministic():
    cfg = lane_d_cfg()
    kc, km = _king(N_FWD + N_BT)
    member = (kc * 0.96, km * 0.97)
    args = dict(clusters=_clusters(N_FWD + N_BT), cfg=cfg, seed="w5i", kinds=_kinds())
    assert judge((kc, km), [member], **args) == judge((kc, km), [member], **args)
