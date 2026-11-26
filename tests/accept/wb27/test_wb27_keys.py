"""WB27 keys (plan rev6.21, owner 2026-11-24): ``lr_scale_min`` /
``lr_scale_max`` and the 12 h leg caps. No activation block (lane B is not
rolled out): the levers and the caps are the default lane B behaviour."""

from __future__ import annotations

from dataclasses import fields

import pytest

from . import helpers as h

KEYS = {"lr_scale_min": 0.5, "lr_scale_max": 2.0, "max_leg_hours": 12.0,
        "max_leg_seconds": 54_000, "sigma_rel_choices": (0.03, 0.05, 0.10, 0.15, 0.20)}


def test_keys_ship_at_their_defaults():
    h.require_built()
    from cascade.shared.config import load_chain_config
    from cascade.shared.config.lane_b import LANE_B_TRACK_DEFAULTS
    from cascade.shared.config.v2 import LaneBConfig, parse_lane_b

    names = {f.name for f in fields(LaneBConfig)}
    assert "miner_levers_from_block" not in names
    assert "miner_levers_from_block" not in LANE_B_TRACK_DEFAULTS
    for k, v in KEYS.items():
        assert LANE_B_TRACK_DEFAULTS[k] == v and k in names
        assert getattr(parse_lane_b({}), k) == v
    # TESTNET-FAST (owner 2026-11-24): testnet caps a lane B leg at 1 h (x 1.25)
    caps = {"chain.toml": (12.0, 54_000), "chain.testnet.toml": (1.0, 4500)}
    for name in ("chain.toml", "chain.testnet.toml"):
        lb = load_chain_config(str(h.REPO / name)).lane_b
        assert lb.leg_caps_at(None) == caps[name], name
        assert (lb.lr_scale_min, lb.lr_scale_max) == (0.5, 2.0), name


def test_lr_scale_bounds_refusals():
    h.require_built()
    from cascade.shared.config.v2 import parse_lane_b

    lb = parse_lane_b({"lr_scale_min": 0.75, "lr_scale_max": 1.5})
    assert (lb.lr_scale_min, lb.lr_scale_max) == (0.75, 1.5)
    for lo, hi in ((0.0, 2.0), (1.1, 2.0), (0.5, 0.9), (0.5, 4.5)):
        with pytest.raises(ValueError, match="lr_scale"):
            parse_lane_b({"lr_scale_min": lo, "lr_scale_max": hi})
    for bad in (True, "1", None):
        with pytest.raises(ValueError, match="lr_scale"):
            parse_lane_b({"lr_scale_min": bad})


def test_ema_set_bounds():
    h.require_built()
    from cascade.shared.config.v2 import parse_lane_b

    assert parse_lane_b({"sigma_rel_choices": [0.01, 0.1, 0.3]}).sigma_rel_choices == (0.01, 0.1, 0.3)
    for bad in ([], [0.05], [0.1, 0.31], [0.005, 0.1], [0.1, 0.1], [0.1, 0.125], "0.1"):
        with pytest.raises(ValueError, match="sigma_rel_choices"):
            parse_lane_b({"sigma_rel_choices": bad})


def test_twelve_hour_caps_bind_everywhere():
    h.require_built()
    cfg = h.chain_cfg()
    lb = cfg.lane_b
    assert lb.leg_caps_at(0) == lb.leg_caps_at(10 ** 9) == (12.0, 54_000)
    m = h.leg()
    c = h.base_contract(cfg)
    ok = dict(steps=100, lb_cfg=lb, contract=c, block=0)
    assert m.check_leg_record(train_hours=12.0, guard_seconds=54_000, **ok) is None
    assert m.check_leg_record(train_hours=12.5, guard_seconds=54_000,
                              **ok) == "train_hours_out_of_range"
    assert m.check_leg_record(train_hours=12.0, guard_seconds=54_001,
                              **ok) == "guard_out_of_range"


def test_no_lane_a_contract_moves():
    h.require_built()
    from cascade.shared.manifest import contract_payload

    cfg = h.chain_cfg()
    for c in cfg.throne_contracts_at(h.effective_block(cfg)):
        p = contract_payload(c)
        assert not {"lane_b_lr_scale", "curriculum", "lane_b_stop_record"} & set(p)
        assert c.lane_b_lr_scale == 1.0 and c.curriculum == () and not c.lane_b_stop_record
