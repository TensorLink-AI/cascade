"""Init-baseline floor (DEC-CA-0034), the validator duel side — RETIRED
(rev6.18 B12, owner 2026-11-23).

It ran in SHADOW on mainnet from 2026-08-27 (floor computed and recorded on
the receipt, verdict unchanged) and never enforced. Release-then-activate:
``[scoring] init_gate_retired_from_block`` is the typed block; a round before
it still runs the legacy shadow floor (so archived receipts replay), from it
the floor is off. ``init_gate_mode`` / ``init_gate_tolerance`` are retired
keys (ignored), "enforce" and the tolerance are gone.

Off must be byte-identical to the pre-gate code, and the receipt tests pin
signature stability. (The trainer's heat SHADOW row retired with the heat.)
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.koth import KothParams, evaluate_round
from cascade.eval.scoring import WindowScore
from cascade.shared.manifest import HeatResult, _heat_from_json, heat_to_json

WS_REF = "metro-v1:trained:hippius:cascade/ckpt-init@sha256:" + "a" * 64

# heat entrant scores: b beats the init (0.30), c and d are worse
SCORES = {"b": 0.25, "c": 0.35, "d": 0.40}
INIT_SCORE = 0.30


def test_default_config_keeps_the_legacy_shadow_until_the_retire_block(cfg):
    # The shipped file types a placeholder block; until it every round runs
    # the legacy shadow floor exactly as before (never enforce), from it off.
    gate = cfg.scoring.init_gate_retired_from_block
    assert gate > 0
    assert cfg.koth_params(block=gate - 1).init_gate_mode == "shadow"
    assert cfg.koth_params(block=None).init_gate_mode == "shadow"
    assert cfg.koth_params(block=gate).init_gate_mode == "off"
    assert not hasattr(cfg.scoring, "init_gate_mode")
    assert not hasattr(cfg.scoring, "init_gate_tolerance")


def test_retire_block_unset_is_off_at_every_block():
    from cascade.shared.config import ScoringConfig, init_gate_mode_at

    s = ScoringConfig(win_margin_start=0.02, win_margin_end=0.02,
                      margin_warmup_rounds=0, min_windows=10, bootstrap_B=200,
                      bootstrap_alpha=0.05, dethrone_cp=1)
    assert s.init_gate_retired_from_block == 0
    assert {init_gate_mode_at(s, b) for b in (None, 0, 1, 9_560_700)} == {"off"}


def test_heat_json_round_trips_init_baseline():
    heat = HeatResult(screen_size="toto2-4m", finalists=1, init_baseline=0.2086)
    assert _heat_from_json(heat_to_json(heat)).init_baseline == pytest.approx(0.2086)
    bare = _heat_from_json(heat_to_json(HeatResult(screen_size="x", finalists=1)))
    assert bare.init_baseline is None


# ── duel-side floor ───────────────────────────────────────────────────────────


def _pw(scale: float, n: int = 80) -> list[WindowScore]:
    """Paired per-window scores at a controlled quality ``scale`` (lower is
    better): abs_target fixed per window index, losses scaled — every list is
    paired with every other by construction."""
    win = np.random.default_rng(9000)
    rng = np.random.default_rng(7)
    return [
        WindowScore(
            series_id=f"w{i}",
            mase=float(rng.uniform(0.8, 1.2)) * scale,
            qloss_per_q=rng.uniform(0.3, 0.7, size=9) * scale,
            abs_target=float(win.uniform(5.0, 10.0)),
        )
        for i in range(n)
    ]


def _params(**kw) -> KothParams:
    base = dict(win_margin_start=0.02, win_margin_end=0.02,
                margin_warmup_rounds=0, min_windows=10, bootstrap_B=200,
                bootstrap_alpha=0.05, dethrone_cp=1, min_clusters=0)
    base.update(kw)
    return KothParams(**base)


KING = _pw(1.00)
GOOD_CHAL = _pw(0.80)     # beats the king decisively AND beats a 0.90 init
BAD_CHAL = _pw(0.85)      # beats the king decisively but is worse than 0.82 init


def test_floor_off_rejects_stray_baseline():
    with pytest.raises(ValueError, match="init-baseline gate is off"):
        evaluate_round(KING, GOOD_CHAL, _params(), seed=1,
                       baseline_scores=_pw(0.9))


def test_floor_off_verdict_unchanged():
    r = evaluate_round(KING, GOOD_CHAL, _params(), seed=1)
    assert r.challenger_wins_round is True
    assert r.init_floor_passed is None and r.baseline_geomean is None


def test_floor_shadow_records_but_never_gates():
    r = evaluate_round(KING, BAD_CHAL, _params(init_gate_mode="shadow"),
                       seed=1, baseline_scores=_pw(0.82))
    assert r.challenger_wins_round is True     # shadow: win stands
    assert r.init_floor_passed is False        # …but the floor verdict is recorded
    assert r.baseline_geomean is not None


def test_floor_shadow_never_grants_a_win():
    # A challenger that does NOT clear the margin stays a loser even when it
    # beats the init soundly — shadow only records.
    near_king = _pw(0.995)
    r = evaluate_round(KING, near_king, _params(init_gate_mode="shadow"),
                       seed=1, baseline_scores=_pw(1.5))
    assert r.init_floor_passed is True
    assert r.challenger_wins_round is False


def test_floor_without_baseline_cannot_run():
    r = evaluate_round(KING, GOOD_CHAL, _params(init_gate_mode="shadow"), seed=1)
    assert r.init_floor_passed is None
    assert r.challenger_wins_round is True     # judged as if the gate were off


def test_enforce_and_tolerance_are_gone():
    import dataclasses

    names = {f.name for f in dataclasses.fields(KothParams)}
    assert "init_gate_mode" in names           # archived params replay
    assert "init_gate_tolerance" not in names
    # a recorded "enforce" (never shipped) would not block a win either
    r = evaluate_round(KING, BAD_CHAL, _params(init_gate_mode="enforce"),
                       seed=1, baseline_scores=_pw(0.82))
    assert r.init_floor_passed is False and r.challenger_wins_round is True


# ── receipt signature stability ───────────────────────────────────────────────


def test_receipt_params_drop_gate_defaults():
    from cascade.shared.receipt import _PARAMS_DROP_WHEN_DEFAULT

    assert _PARAMS_DROP_WHEN_DEFAULT["init_gate_mode"] == "off"
    assert _PARAMS_DROP_WHEN_DEFAULT["init_gate_tolerance"] == 0.0


def test_verdict_body_drops_absent_floor_fields():
    from cascade.shared.receipt import VerdictRecord, _verdict_body

    base = dict(params={}, bootstrap_seed="1", king_tenure_rounds=0, lcb=0.1,
                margin=0.02, challenger_wins_round=True, inconclusive=False,
                n_windows=10, king_geomean=1.0, chal_geomean=0.9,
                gift_lcb=None, gift_gate_passed=None, dethroned=True,
                note="", king_hotkey="k", king_uid=1)
    off = _verdict_body(VerdictRecord(**base))
    assert "init_baseline_geomean" not in off and "init_floor_passed" not in off
    on = _verdict_body(VerdictRecord(**base, init_baseline_geomean=0.82,
                                     init_floor_passed=False))
    assert on["init_baseline_geomean"] == 0.82
    assert on["init_floor_passed"] is False


# ── loader round-trip (the DEC-CA-0035 sweep's regression class) ─────────────


def test_loader_round_trips_the_retire_block(tmp_path):
    """Config knobs need loader parsing: the typed block must reach
    koth_params, and the retired keys must load and change nothing."""
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "chain.toml").read_text()
    line = next(x for x in text.splitlines()
                if x.startswith("init_gate_retired_from_block"))
    patched = text.replace(
        line, 'init_gate_retired_from_block = 9600300\n'
              'init_gate_mode = "enforce"\ninit_gate_tolerance = 0.05', 1)
    p = tmp_path / "chain.toml"
    p.write_text(patched)
    c = load_chain_config(p)
    assert c.scoring.init_gate_retired_from_block == 9600300
    assert c.koth_params(block=9600299).init_gate_mode == "shadow"
    assert c.koth_params(block=9600300).init_gate_mode == "off"
