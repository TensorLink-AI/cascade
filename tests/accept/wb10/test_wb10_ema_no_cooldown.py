"""WB10, plan rev6.20 (owner 2026-11-24, lane B only: "let's just do EMA, no
cooldown"): the typed lane B activation key ``[lane_b]
ema_no_cooldown_from_block`` (0 = never, today's behaviour), its scored-weights
rule and the lane B leg-contract helper. Lane A's contracts and
contract_digest never move. Contract: README.md "rev6.20"."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from . import helpers as h

KEY = "ema_no_cooldown_from_block"
SCHEDULE = "warmup_constant_ema"


def _track_has_key():
    from cascade.shared.config.lane_b import LANE_B_TRACK_DEFAULTS

    assert KEY in LANE_B_TRACK_DEFAULTS, f"[lane_b] {KEY} is not a lane B track key"
    assert LANE_B_TRACK_DEFAULTS[KEY] == 0, "the default must be 0 = never"


def test_ema_no_cooldown_key_defaults_to_never():
    _track_has_key()
    from cascade.shared.config.v2 import LaneBConfig, parse_lane_b

    assert KEY in {f.name for f in fields(LaneBConfig)}
    lb = parse_lane_b({})
    assert lb.ema_no_cooldown_from_block == 0
    assert [lb.scored_weights_at(b) for b in (None, 0, 1, h.EFFECTIVE, 10**12)] == [""] * 5
    for name in h.SHIPPED:
        assert h.load_ok(h.shipped_raw(name)).lane_b.ema_no_cooldown_from_block == 0, name


def test_ema_no_cooldown_key_is_a_typed_block():
    _track_has_key()
    from cascade.shared.config.v2 import parse_lane_b

    lb = parse_lane_b({KEY: h.EFFECTIVE})
    assert lb.ema_no_cooldown_from_block == h.EFFECTIVE
    assert lb.scored_weights_at(h.EFFECTIVE - 1) == ""
    assert lb.scored_weights_at(h.EFFECTIVE) == "ema010"
    assert lb.scored_weights_at(h.EFFECTIVE + h.SETTLEMENT) == "ema010"
    for bad in (-1, True, "v2", 1.5, "9712800"):
        with pytest.raises(ValueError, match=KEY):
            parse_lane_b({KEY: bad})


def test_ema_no_cooldown_loader_wants_an_era_boundary_after_v2():
    _track_has_key()
    v2_from = int(h.shipped_raw("chain.toml")["v2"]["from_block"])
    assert v2_from < h.EFFECTIVE
    cfg = h.load_ok(h.raw_with("chain.toml", {KEY: h.EFFECTIVE}))
    assert cfg.lane_b.ema_no_cooldown_from_block == h.EFFECTIVE
    assert h.refused(h.raw_with("chain.toml", {KEY: h.EFFECTIVE + h.SETTLEMENT}))
    assert h.refused(h.raw_with("chain.toml", {KEY: v2_from - h.ERA_MAIN * 10}))


def test_leg_contract_takes_the_ema_schedule_only_from_the_block():
    _track_has_key()
    from cascade.shared.config.lane_b import lane_b_leg_schedule
    from cascade.shared.manifest import contract_digest

    off = h.load_ok(h.shipped_raw("chain.toml"))
    on = h.load_ok(h.raw_with("chain.toml", {KEY: h.EFFECTIVE}))
    lane_a = on.throne_contracts_at(h.EFFECTIVE)[0]
    assert contract_digest(lane_a) == contract_digest(off.throne_contracts_at(h.EFFECTIVE)[0])
    leg = lane_a.for_hours(3.0)
    assert lane_b_leg_schedule(on.lane_b, leg, h.EFFECTIVE - 1) is leg
    assert lane_b_leg_schedule(off.lane_b, leg, h.EFFECTIVE) is leg
    armed = lane_b_leg_schedule(on.lane_b, leg, h.EFFECTIVE)
    assert armed == replace(leg, lr_schedule=SCHEDULE)
    assert contract_digest(armed) != contract_digest(leg)       # the leg contract records it
    assert leg.lr_schedule == lane_a.lr_schedule == "wsd"       # lane A untouched
