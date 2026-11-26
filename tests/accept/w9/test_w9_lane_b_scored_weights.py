"""Plan rev6.20 (owner 2026-11-24, lane B only: EMA, no cooldown): the lane B
judge's new refusal stage ``scored_weights`` is a known stage of cascade-audit's
lane B row check (``checks.LANE_B_STAGES``): a refused row is consistent, a
row that "passed" at that stage is flagged like any non-verdict stage."""

from __future__ import annotations


def test_audit_knows_the_scored_weights_stage():
    from cascade.audit import checks
    from cascade.shared.config.v2 import parse_lane_b

    assert "scored_weights" in checks.LANE_B_STAGES, checks.LANE_B_STAGES
    lb = parse_lane_b({"from_block": 1, "domains": ["energy"]})
    row = {"hotkey": "hk", "domain": "energy", "stage": "scored_weights", "passed": False,
           "reason": "entry scored 'raw', the rule says 'ema010'", "target_lcb": None,
           "guard_lcbs": {}, "public_lcbs": {}}
    assert checks._lane_b_row_verdict(row, lb, ("energy",)) is None
    bad = checks._lane_b_row_verdict({**row, "passed": True}, lb, ("energy",))
    assert bad and "scored_weights-stage row passed" in bad, bad
