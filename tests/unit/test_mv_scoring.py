"""GIFT-Eval multivariate weighting for the private-pool duel (DEC-CA-0041).

A multivariate window's channels are averaged into ONE per-window contribution
so the window counts once (GIFT convention), matching the source-cluster
bootstrap that already treats it as one resampling unit. These tests pin the
load-bearing properties: it is BIT-IDENTICAL on any univariate pool (the whole
pool until MV data lands), it collapses a C-channel window to a single
per-window row (arithmetic mean over variates, zero-target channels masked from
the WQL half only), and it is block-gated so validator and audit resolve the
same rule from the round's block.
"""
from __future__ import annotations

import numpy as np

from cascade.eval.crps import DEFAULT_QUANTILE_LEVELS
from cascade.eval.koth import KothParams, cohort_maxt_lcb_map, evaluate_round
from cascade.eval.scoring import (
    WindowScore,
    collapse_channels_by_window,
)
from cascade.shared.config import ScoringConfig, mv_score_active

NQ = len(DEFAULT_QUANTILE_LEVELS)


def ws(sid, mase, wql, *, abs_t=1.0, channel=0, source="s"):
    """A WindowScore whose per-window WQL is exactly ``wql`` (encoded via a flat
    qloss against ``abs_t``; ``2*mean_q(qloss)/abs_t == wql``)."""
    q = np.full(NQ, wql * abs_t / 2.0, dtype=np.float64)
    return WindowScore(series_id=sid, mase=mase, qloss_per_q=q,
                       abs_target=abs_t, channel=channel, source=source)


def _params(**kw):
    base = dict(win_margin_start=0.005, win_margin_end=0.005,
                margin_warmup_rounds=8, min_windows=4, bootstrap_B=2000,
                bootstrap_alpha=0.05, dethrone_cp=1, min_clusters=0)
    base.update(kw)
    return KothParams(**base)


def _uni_pool(n=12, seed=0):
    rng = np.random.default_rng(seed)
    king, chal = [], []
    for i in range(n):
        diff = float(rng.uniform(0.6, 1.4))
        king.append(ws(f"w{i}", 0.9 * diff, 0.2 * diff, source=f"src{i % 4}"))
        chal.append(ws(f"w{i}", 0.9 * diff * 0.9, 0.2 * diff * 0.9,
                       source=f"src{i % 4}"))
    return king, chal


# --------------------------------------------------------------------------- #
# 1. BIT-IDENTICAL on a univariate pool, mv_score on vs off (the guard rail)
# --------------------------------------------------------------------------- #
def test_univariate_pool_is_bit_identical_mv_on_vs_off():
    king, chal = _uni_pool()
    off = evaluate_round(king, chal, _params(mv_score=False), seed="r")
    on = evaluate_round(king, chal, _params(mv_score=True), seed="r")
    assert on.lcb == off.lcb                     # exact, not approx
    assert on.king_geomean == off.king_geomean
    assert on.chal_geomean == off.chal_geomean
    assert on.n_windows == off.n_windows == len(king)


def test_collapse_leaves_univariate_scores_untouched():
    king, _ = _uni_pool(n=6)
    out = collapse_channels_by_window(king)
    assert len(out) == len(king)
    for a, b in zip(out, king, strict=True):
        assert a is b                            # singleton groups pass through


# --------------------------------------------------------------------------- #
# 2. A C-channel window collapses to ONE per-window row
# --------------------------------------------------------------------------- #
def test_multichannel_window_collapses_to_one_row():
    scores = [
        ws("A", 0.8, 0.10, channel=0), ws("A", 1.2, 0.30, channel=1),
        ws("A", 1.0, 0.20, channel=2),
        ws("B", 0.5, 0.05, channel=0), ws("B", 1.5, 0.15, channel=1),
    ]
    out = collapse_channels_by_window(scores)
    assert [s.series_id for s in out] == ["A", "B"]
    assert [s.channel for s in out] == [0, 0]
    # MASE is the arithmetic mean over channels
    assert out[0].mase == (0.8 + 1.2 + 1.0) / 3
    assert out[1].mase == (0.5 + 1.5) / 2
    # WQL (recovered from the encoded row) is the arithmetic mean over channels
    wqlA = 2.0 * out[0].qloss_per_q.mean() / out[0].abs_target
    assert wqlA == (0.10 + 0.30 + 0.20) / 3


def test_zero_target_channels_masked_from_wql_only():
    # window C: one valid channel (|y|>0) + one zero-target channel
    scores = [ws("C", 1.0, 0.20, abs_t=10.0, channel=0),
              ws("C", 2.0, 0.0, abs_t=0.0, channel=1)]
    (row,) = collapse_channels_by_window(scores)
    assert row.abs_target == 1.0                 # WQL defined (>=1 valid channel)
    wql = 2.0 * row.qloss_per_q.mean() / row.abs_target
    assert np.isclose(wql, 0.20)                 # only the valid channel's WQL
    assert row.mase == (1.0 + 2.0) / 2           # MASE still counts both

    # window D: every channel zero-target -> WQL undefined, MASE still counts
    dscores = [ws("D", 1.0, 0.0, abs_t=0.0, channel=0),
               ws("D", 3.0, 0.0, abs_t=0.0, channel=1)]
    (drow,) = collapse_channels_by_window(dscores)
    assert drow.abs_target == 0.0                # masked from the WQL half
    assert drow.mase == 2.0


# --------------------------------------------------------------------------- #
# 3. MV weighting actually changes for C>1: a coupled window votes once, not C×
# --------------------------------------------------------------------------- #
def test_mv_window_votes_once_not_per_channel():
    # one univariate source + one 4-channel window where the challenger is much
    # better. Per-row (mv off) weights it 4×; averaged (mv on) weights it once.
    king = [ws("u", 1.0, 0.20, source="u")]
    chal = [ws("u", 1.0, 0.20, source="u")]
    for c in range(4):
        king.append(ws("m", 1.0, 0.20, channel=c, source="m"))
        chal.append(ws("m", 0.5, 0.10, channel=c, source="m"))  # 2x better
    p_off = _params(min_windows=1, mv_score=False)
    p_on = _params(min_windows=1, mv_score=True)
    # point estimates (king/chal geomeans) differ between the two weightings
    off = evaluate_round(king, chal, p_off, seed="x")
    on = evaluate_round(king, chal, p_on, seed="x")
    # under averaging the MV window is 1 of 2 windows; under per-row it is 4 of 5
    assert on.n_windows == 2 and off.n_windows == 5
    # so the challenger's aggregate improvement is LARGER under per-row weighting
    off_impr = (off.king_geomean - off.chal_geomean) / off.king_geomean
    on_impr = (on.king_geomean - on.chal_geomean) / on.king_geomean
    assert off_impr > on_impr > 0


# --------------------------------------------------------------------------- #
# 4. block-gate resolver
# --------------------------------------------------------------------------- #
def _paired_king_chals(n=10, seed=0):
    rng = np.random.default_rng(seed)
    king, c1, c2 = [], [], []
    for i in range(n):
        diff = float(rng.uniform(0.6, 1.4))
        src = f"src{i % 3}"
        king.append(ws(f"w{i}", 0.9 * diff, 0.2 * diff, source=src))
        c1.append(ws(f"w{i}", 0.9 * diff * 0.92, 0.2 * diff * 0.92, source=src))
        c2.append(ws(f"w{i}", 0.9 * diff * 0.96, 0.2 * diff * 0.96, source=src))
    return king, c1, c2


def test_cohort_maxt_univariate_bit_identical_mv_on_vs_off():
    """The k>=2 max-T path must also be a bit-exact no-op on a univariate pool."""
    king, c1, c2 = _paired_king_chals()
    cohort = [("hkA", c1), ("hkB", c2)]
    off = cohort_maxt_lcb_map(king, cohort, _params(mv_score=False), seed="c")
    on = cohort_maxt_lcb_map(king, cohort, _params(mv_score=True), seed="c")
    assert off == on                              # exact dict equality


def test_cohort_maxt_collapses_channels_when_armed():
    """A multivariate window must be re-weighted (channels averaged to one) in
    the cohort max-T LCB when mv_score is armed — not voted per channel."""
    king, c1, c2 = _paired_king_chals(n=6)
    for scores, f in ((king, 1.0), (c1, 0.80), (c2, 0.85)):
        for c in range(3):
            scores.append(ws("mv", 0.9 * f, 0.2 * f, channel=c, source="mvsrc"))
    cohort = [("A", c1), ("B", c2)]
    off = cohort_maxt_lcb_map(king, cohort, _params(mv_score=False), seed="c")
    on = cohort_maxt_lcb_map(king, cohort, _params(mv_score=True), seed="c")
    assert off != on                              # collapse changed the weighting


def test_mv_score_active_block_gate():
    s0 = ScoringConfig.__new__(ScoringConfig)   # only need mv_score_from_block
    object.__setattr__(s0, "mv_score_from_block", 0)
    assert mv_score_active(s0, 10_000) is False  # 0 = off forever

    s = ScoringConfig.__new__(ScoringConfig)
    object.__setattr__(s, "mv_score_from_block", 1000)
    assert mv_score_active(s, None) is False
    assert mv_score_active(s, 999) is False
    assert mv_score_active(s, 1000) is True
    assert mv_score_active(s, 5000) is True
