"""The baseline-referenced (%-of-increment) margin — DEC-CA-0023's E2 scoring
rule behind [scoring] margin_mode."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.bootstrap import increment_bootstrap_rel
from cascade.eval.koth import KothParams, evaluate_round
from cascade.eval.scoring import WindowScore

Q = 9


def _params(**kw) -> KothParams:
    base = dict(
        win_margin_start=0.02, win_margin_end=0.02, margin_warmup_rounds=0,
        min_windows=5, bootstrap_B=500, bootstrap_alpha=0.05, dethrone_cp=1,
    )
    base.update(kw)
    return KothParams(**base)


def _scores(level: float, n: int = 40, seed: int = 0) -> list[WindowScore]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(WindowScore(
            series_id=f"w{i}",
            mase=level * float(rng.uniform(0.9, 1.1)),
            qloss_per_q=level * rng.uniform(0.9, 1.1, Q),
            abs_target=10.0 + i,       # shared across competitors (paired)
            channel=0,
        ))
    return out


def _paired(king_level, chal_level, base_level, n=40):
    # abs_target must be identical across competitors; build from one target set.
    king = _scores(king_level, n, seed=1)
    chal = _scores(chal_level, n, seed=2)
    base = _scores(base_level, n, seed=3)
    return king, chal, base


# ── the statistic ────────────────────────────────────────────────────────────


def _stack(scores):
    from cascade.eval.scoring import stack_components

    return stack_components(scores)


def test_increment_rel_positive_when_challenger_improves_more():
    king, chal, base = _paired(0.9, 0.8, 1.0)   # chal's increment 2x the king's
    rel = increment_bootstrap_rel(
        _stack(king), _stack(chal), _stack(base), B=400, seed=7
    )
    assert rel.shape == (400,)
    assert np.quantile(rel, 0.05) > 0.0


def test_increment_rel_symmetric_zero_when_equal():
    king, chal, base = _paired(0.9, 0.9, 1.0)
    # Identical levels but different noise draws — the median should sit near
    # zero, well inside (-1, 1) increments.
    rel = increment_bootstrap_rel(
        _stack(king), _stack(chal), _stack(base), B=400, seed=7
    )
    assert abs(np.median(rel)) < 0.75


def test_floor_prevents_explosion_on_null_increments():
    # Both models ≈ the baseline: unfloored units would divide noise by noise.
    king, chal, base = _paired(1.0, 1.0, 1.0)
    rel_floored = increment_bootstrap_rel(
        _stack(king), _stack(chal), _stack(base), B=400, seed=7, floor_frac=0.05
    )
    rel_tiny_floor = increment_bootstrap_rel(
        _stack(king), _stack(chal), _stack(base), B=400, seed=7, floor_frac=1e-9
    )
    assert np.abs(rel_floored).max() < np.abs(rel_tiny_floor).max()
    assert np.isfinite(rel_floored).all()


def test_level_statistic_strangles_where_increment_does_not():
    # The E2 motivation: a compounding lineage converges toward the shared
    # init, so (king − chal)/king shrinks with the increments while
    # (king − chal)/unit stays scaled to them.
    n = 40
    for scale in (1.0, 0.1, 0.01):
        king, chal, base = _paired(1.0 - 0.10 * scale, 1.0 - 0.20 * scale, 1.0, n)
        rel = increment_bootstrap_rel(
            _stack(king), _stack(chal), _stack(base), B=300, seed=7,
            floor_frac=0.001,
        )
        # The increment statistic keeps the same order of magnitude at every
        # scale — the challenger's edge is ~the same fraction of an increment.
        assert 0.05 < abs(np.median(rel)) < 2.0


# ── evaluate_round wiring ────────────────────────────────────────────────────


def test_level_mode_rejects_stray_baseline_and_increment_requires_it():
    king, chal, base = _paired(0.9, 0.8, 1.0)
    with pytest.raises(ValueError, match="margin_mode is 'level'"):
        evaluate_round(king, chal, _params(), seed=1, baseline_scores=base)
    with pytest.raises(ValueError, match="needs baseline_scores"):
        evaluate_round(king, chal, _params(margin_mode="increment"), seed=1)


def test_increment_round_result_records_mode_and_baseline():
    king, chal, base = _paired(0.9, 0.7, 1.0)
    res = evaluate_round(
        king, chal, _params(margin_mode="increment"), seed=1,
        baseline_scores=base,
    )
    assert res.margin_mode == "increment"
    assert res.baseline_geomean is not None and res.baseline_geomean > 0
    assert res.challenger_wins_round      # 3x the king's increment clears 2%
    assert np.isfinite(res.lcb)


def test_level_mode_is_bit_identical_to_before():
    # margin_mode default("level") + no baseline: same call path, same numbers.
    king, chal, _ = _paired(1.0, 0.9, 1.0)
    a = evaluate_round(king, chal, _params(), seed=42)
    b = evaluate_round(king, chal, _params(margin_mode="level"), seed=42)
    assert a.lcb == b.lcb and a.margin == b.margin
    assert a.margin_mode == "level" and a.baseline_geomean is None


def test_unpaired_baseline_rejected():
    king, chal, base = _paired(0.9, 0.8, 1.0)
    with pytest.raises(ValueError, match="unpaired baseline"):
        evaluate_round(
            king, chal, _params(margin_mode="increment"), seed=1,
            baseline_scores=base[:-1],
        )


def test_inconclusive_still_reports_mode():
    king, chal, base = _paired(0.9, 0.8, 1.0, n=3)   # < min_windows
    res = evaluate_round(
        king, chal, _params(margin_mode="increment"), seed=1,
        baseline_scores=base,
    )
    assert res.inconclusive and res.margin_mode == "increment"


# ── config plumbing ──────────────────────────────────────────────────────────


def test_margin_mode_validated_at_load():
    from cascade.shared.config import _margin_mode

    assert _margin_mode("level") == "level"
    assert _margin_mode("increment") == "increment"
    with pytest.raises(ValueError, match="margin_mode"):
        _margin_mode("incremant")


def test_koth_params_carry_margin_fields():
    p = _params(margin_mode="increment", margin_increment_floor=0.02)
    assert p.margin_mode == "increment" and p.margin_increment_floor == 0.02
    # Defaults are the exact pre-field behaviour.
    d = _params()
    assert d.margin_mode == "level" and d.margin_increment_floor == 0.01


# ── audit: judged-mode detection from baseline rows ──────────────────────────


def test_baseline_pooled_detects_judged_mode():
    from cascade.audit.checks import _baseline_pooled
    from cascade.shared.receipt import EntryScores, WindowScoreRecord

    base = _scores(1.0, 6, seed=3)
    rows = tuple(WindowScoreRecord.from_score(s) for s in base)

    class _R:  # minimal receipt stand-in: only entry_scores is read
        entry_scores = (
            EntryScores(role="king", size="", hotkey="k", uid=1, scores=rows),
            EntryScores(role="baseline", size="", hotkey="__warm_start_init__",
                        uid=-1, scores=rows),
        )

    out = _baseline_pooled(_R, [""])
    assert out is not None and len(out) == 6

    class _R2:
        entry_scores = (
            EntryScores(role="king", size="", hotkey="k", uid=1, scores=rows),
        )

    assert _baseline_pooled(_R2, [""]) is None


def test_baseline_pooled_missing_size_raises():
    import pytest as _pytest

    from cascade.audit.checks import _baseline_pooled
    from cascade.shared.receipt import EntryScores, WindowScoreRecord

    rows = tuple(WindowScoreRecord.from_score(s) for s in _scores(1.0, 4, seed=3))

    class _R:
        entry_scores = (
            EntryScores(role="baseline", size="toto2-4m",
                        hotkey="__warm_start_init__", uid=-1, scores=rows),
        )

    with _pytest.raises(KeyError):
        _baseline_pooled(_R, ["toto2-22m"])
