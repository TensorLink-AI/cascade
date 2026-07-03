"""Public-benchmark no-regression gate — the paired bootstrap over gift-eval
ratio rows and the KOTH combiner that folds it into a round result.

The gate can only BLOCK a private-pool win, never grant one; these tests pin
that asymmetry and the uncomputable → inconclusive semantics.
"""

from __future__ import annotations

from cascade.eval.gift_gate import evaluate_gift_gate, uncomputable_gate
from cascade.eval.koth import RoundResult, apply_gift_gate


def _rows(mult: float, n: int = 20, start: int = 0) -> list[dict]:
    """n configs whose ratios are a constant ``mult`` — so every bootstrap bag
    is identical and the LCB is deterministic regardless of resampling."""
    return [
        {"full": f"ds/{i}", "crps_ratio": mult, "mase_ratio": mult}
        for i in range(start, start + n)
    ]


PARAMS = dict(alpha=0.05, B=2000, seed=7, min_configs=15)


# ── gate math ────────────────────────────────────────────────────────────────


def test_identical_models_pass_at_zero_tolerance():
    g = evaluate_gift_gate(_rows(1.0), _rows(1.0), tolerance=0.0, **PARAMS)
    assert g.computed and g.passed
    assert abs(g.lcb) < 1e-9  # rel = 0 in every bag


def test_clearly_regressed_challenger_fails():
    # challenger ratios 20% larger (worse) everywhere ⇒ rel ≈ -0.20 ≪ -tol
    g = evaluate_gift_gate(_rows(1.0), _rows(1.2), tolerance=0.03, **PARAMS)
    assert g.computed and not g.passed
    assert g.lcb < -0.1


def test_noise_level_deficit_within_tolerance_passes():
    # challenger a flat 1.5% worse — worse, but inside the 3% tolerance
    g = evaluate_gift_gate(_rows(1.0), _rows(1.015), tolerance=0.03, **PARAMS)
    assert g.computed and g.passed
    assert -0.03 < g.lcb < 0.0


def test_challenger_better_passes_with_positive_lcb():
    g = evaluate_gift_gate(_rows(1.0), _rows(0.8), tolerance=0.03, **PARAMS)
    assert g.computed and g.passed and g.lcb > 0.1


def test_join_keeps_only_shared_configs():
    king = _rows(1.0, n=20, start=0)   # ds/0..19
    chal = _rows(1.0, n=20, start=15)  # ds/15..34 → 5 shared
    g = evaluate_gift_gate(king, chal, tolerance=0.03, alpha=0.05, B=100, seed=1, min_configs=3)
    assert g.computed and g.n_configs == 5


def test_below_min_configs_is_uncomputable():
    g = evaluate_gift_gate(_rows(1.0, n=10), _rows(1.0, n=10), tolerance=0.03, **PARAMS)
    assert not g.computed and g.passed is None and g.n_configs == 10


def test_no_shared_configs_is_uncomputable():
    g = evaluate_gift_gate(_rows(1.0, n=20, start=0), _rows(1.0, n=20, start=100),
                           tolerance=0.03, **PARAMS)
    assert not g.computed and g.n_configs == 0


def test_none_ratios_filled_not_crashed():
    king = _rows(1.0)
    chal = _rows(1.0)
    chal[3]["crps_ratio"] = None  # invalid → filled with the column mean
    g = evaluate_gift_gate(king, chal, tolerance=0.03, **PARAMS)
    assert g.computed and g.passed


def test_seed_determinism():
    a = evaluate_gift_gate(_rows(1.0), _rows(1.05), tolerance=0.03, alpha=0.05, B=2000,
                           seed="block-hash-abc", min_configs=15)
    b = evaluate_gift_gate(_rows(1.0), _rows(1.05), tolerance=0.03, alpha=0.05, B=2000,
                           seed="block-hash-abc", min_configs=15)
    assert a.lcb == b.lcb


# ── combiner truth table ───────────────────────────────────────────────────────


def _win() -> RoundResult:
    return RoundResult(
        challenger_wins_round=True, lcb=0.1, margin=0.02, n_windows=300,
        king_geomean=1.0, chal_geomean=0.6, inconclusive=False,
    )


def _passing():
    return evaluate_gift_gate(_rows(1.0), _rows(1.0), tolerance=0.03, **PARAMS)


def _failing():
    return evaluate_gift_gate(_rows(1.0), _rows(1.5), tolerance=0.03, **PARAMS)


def test_enforce_win_and_pass_stays_a_win():
    out = apply_gift_gate(_win(), _passing(), mode="enforce")
    assert out.challenger_wins_round and not out.inconclusive
    assert out.gift_gate_passed is True


def test_enforce_win_and_fail_becomes_a_loss():
    out = apply_gift_gate(_win(), _failing(), mode="enforce")
    assert not out.challenger_wins_round and not out.inconclusive  # loss → streak resets
    assert out.gift_gate_passed is False


def test_enforce_win_uncomputable_becomes_inconclusive():
    out = apply_gift_gate(_win(), uncomputable_gate(0.03, "sidecar down"), mode="enforce")
    assert not out.challenger_wins_round and out.inconclusive        # king holds, streak untouched
    assert out.gift_gate_passed is None and out.gift_lcb is None


def test_shadow_never_changes_the_verdict():
    out = apply_gift_gate(_win(), _failing(), mode="shadow")
    assert out.challenger_wins_round and not out.inconclusive        # unchanged…
    assert out.gift_gate_passed is False                            # …but recorded for logging


def test_off_mode_is_inert_but_records_diagnostics():
    out = apply_gift_gate(_win(), _passing(), mode="off")
    assert out.challenger_wins_round
    assert out.gift_gate_passed is True


def test_gate_is_ignored_on_a_non_win():
    loss = RoundResult(
        challenger_wins_round=False, lcb=-0.01, margin=0.02, n_windows=300,
        king_geomean=1.0, chal_geomean=1.1, inconclusive=False,
    )
    out = apply_gift_gate(loss, _failing(), mode="enforce")
    assert not out.challenger_wins_round and not out.inconclusive  # unchanged
