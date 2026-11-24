"""Plan rev6.20 (owner 2026-11-24, lane B only: "let's just do EMA, no
cooldown") on the box (no torch): the pure lane B schedule surface of
pyramid_trainer.py, read from source like the W-EMA gate tests, plus the
worker's use of the shared leg-contract rule. Lane A keeps W-EMA: the
existing gate tests (``power_ema_active`` is False for every input) and the
torch byte-identity tests stay as they are. Contract: README.md "rev6.20"."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from . import helpers as h

SELECTION = {0: [1], 1: [0, 2]}
SCHEDULE = "warmup_constant_ema"
REPO = Path(__file__).resolve().parents[3]


def _ns() -> dict:
    ns = h.namespace()
    for name in ("lane_b_ema_leg", "lane_b_schedule_factor"):
        assert name in ns, f"pyramid_trainer.{name} must be a module-level def (rev6.20)"
    return ns


def _wsd(step: int, total: int) -> float:
    """pyramid_optimizer.schedule_factor, copied (that module imports torch)."""
    cooldown = max(1, round(total / 11))
    start = total - cooldown
    warmup = min(80, max(1, start))
    if step < warmup:
        return step / warmup
    if step >= start:
        return 0.0 if cooldown == 1 else max(0, 1 - (step - start) / (cooldown - 1))
    return 1.0


def test_lane_b_ema_schedule_is_recognised_on_expert_legs_only():
    gate = _ns()["lane_b_ema_leg"]
    assert gate("wsd", None) is False            # every lane A leg
    assert gate("wsd", SELECTION) is False       # a lane B leg before the key
    assert gate(SCHEDULE, SELECTION) is True     # a lane B leg from the key
    with pytest.raises(ValueError):
        gate(SCHEDULE, None)                     # never a lane A leg
    with pytest.raises(ValueError):
        gate("warmup_cosine", SELECTION)


def test_lane_b_schedule_has_wsd_warmup_and_no_cooldown():
    factor = _ns()["lane_b_schedule_factor"]
    for total in (3, 12, 23_841, 27_500):
        cooldown = max(1, round(total / 11))
        start = total - cooldown
        for step in range(0, min(total, 200)):
            if step < start:
                assert factor(step, total) == _wsd(step, total), (total, step)
        # the WSD decay tail is gone: peak LR to the last step
        tail = range(max(start, min(80, max(1, start))), total)
        assert all(factor(s, total) == 1.0 for s in tail), total
    assert [factor(s, 27_500) for s in (0, 40, 80, 24_999, 25_000, 27_499)] == [0, .5, 1, 1, 1, 1]


def test_train_pyramid_uses_the_lane_b_schedule_and_scores_the_ema():
    """train_pyramid derives the leg kind from its contract (lane_b_ema_leg),
    steps the LR with lane_b_schedule_factor on such a leg, and scores the
    EMA there (the scored decision names ``no_cooldown``)."""
    _ns()
    fn = next(n for n in h.trainer_tree().body
              if isinstance(n, ast.FunctionDef) and n.name == "train_pyramid")
    calls = {c.func.id for c in ast.walk(fn)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert {"lane_b_ema_leg", "lane_b_schedule_factor", "schedule_factor"} <= calls, calls
    src = ast.unparse(fn)
    assert "no_cooldown or power_ema_active(" in src
    assert "'schedule': contract.lr_schedule" in src


def test_scored_ema_width_is_the_owners_sigma_0_10():
    """The lane B scored width is the image's POWER_EMA_SIGMA_RELS[POWER_EMA_EVAL]
    (owner 2026-11-01: score the 0.10 average) and the shared constant agrees."""
    from cascade.shared.config import lane_b

    assert getattr(lane_b, "LANE_B_EMA_SCHEDULE", None) == SCHEDULE
    c = h.constants()
    assert c["POWER_EMA_SIGMA_RELS"][c["POWER_EMA_EVAL"]] == lane_b.LANE_B_EMA_SIGMA_REL == 0.10
    assert lane_b.LANE_B_SCORED_EMA == "ema010"


def test_worker_expert_role_applies_the_shared_leg_contract_rule():
    """worker.py turns the expert leg's for_hours contract into the lane B
    leg contract with the one shared helper, keyed on the leg's --block."""
    src = (REPO / "cascade" / "trainer" / "worker.py").read_text(encoding="utf-8")
    assert "lane_b_leg_schedule(cfg.lane_b, contract, int(args.block))" in src
