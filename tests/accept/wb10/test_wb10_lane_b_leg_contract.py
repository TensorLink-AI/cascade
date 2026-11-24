"""WB10, plan rev6.20 (owner 2026-11-24, lane B only: "let's just do EMA, no
cooldown"; no activation block, lanes B/C/D are not rolled out): every lane B
leg contract is the era contract at the leg's hours with ``lr_schedule =
"warmup_constant_ema"`` (``lane_b_leg_contract``, the one helper the trainer
and the worker share). No ``[lane_b]`` key arms it, and lane A's contracts
and contract_digest never move. Contract: README.md "rev6.20"."""

from __future__ import annotations

from dataclasses import replace

from . import helpers as h

SCHEDULE = "warmup_constant_ema"


def _lane_b():
    from cascade.shared.config import lane_b

    for name in ("lane_b_leg_contract", "LANE_B_EMA_SCHEDULE", "LANE_B_SCORED_EMA"):
        assert hasattr(lane_b, name), f"cascade.shared.config.lane_b.{name} (plan rev6.20)"
    return lane_b


def test_lane_b_leg_contract_is_the_ema_schedule():
    lane_b = _lane_b()
    from cascade.shared.manifest import contract_digest

    cfg = h.load_ok(h.shipped_raw("chain.toml"))
    lane_a = cfg.throne_contracts_at(h.EFFECTIVE)[0]
    before = contract_digest(lane_a)
    leg = lane_a.for_hours(3.0)
    got = lane_b.lane_b_leg_contract(leg)
    assert lane_b.LANE_B_EMA_SCHEDULE == SCHEDULE and lane_b.LANE_B_SCORED_EMA == "ema010"
    assert got == replace(leg, lr_schedule=SCHEDULE)
    assert contract_digest(got) != contract_digest(leg)        # the leg contract records it
    assert leg.lr_schedule == lane_a.lr_schedule == "wsd"      # lane A untouched
    assert contract_digest(cfg.throne_contracts_at(h.EFFECTIVE)[0]) == before


def test_lane_b_leg_contract_leaves_a_non_pyramid_contract_alone():
    """Only the pyramid trainer implements the schedule; lane B runs only on
    v2 (pyramid) eras, so a pre-v2 contract passes through unchanged."""
    lane_b = _lane_b()
    cfg = h.load_ok(h.shipped_raw("chain.toml"))
    leg = replace(cfg.throne_contracts_at(h.EFFECTIVE)[0].for_hours(3.0), base_arch="toto2")
    assert lane_b.lane_b_leg_contract(leg) is leg


def test_no_lane_b_key_arms_it():
    """Owner 2026-11-24: no activation block. The rule is lane B itself."""
    lane_b = _lane_b()
    from cascade.shared.config.v2 import LaneBConfig

    names = set(lane_b.LANE_B_TRACK_DEFAULTS) | set(LaneBConfig.__dataclass_fields__)
    assert not {n for n in names if "cooldown" in n or n.startswith("ema") or "_ema" in n}, \
        sorted(names)
