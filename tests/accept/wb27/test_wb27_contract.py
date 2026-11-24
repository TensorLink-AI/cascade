"""WB27 leg contract (plan rev6.21): the trainer and the worker derive the
lane B leg contract with the same three steps — the WB23 length (or the
limits leg), rev6.20's schedule, the levers effective at the dispatch block
— and the receipt binds it; lane A contracts never move."""

from __future__ import annotations

import inspect

from . import helpers as h

LEVERS = {"lr_scale": 1.5, "curriculum": [["easy", "0.4"], ["hard", "0.6"]], "ema": 0.05}


def test_empty_spec_is_todays_leg_contract():
    lv = h.levers()
    cfg = h.chain_cfg()
    base = h.base_contract(cfg)
    b = h.effective_block(cfg)
    want = base.for_hours(cfg.lane_b.leg_hours, guard_factor=cfg.round.heat_guard_factor,
                          guard_floor_seconds=cfg.round.heat_guard_floor_seconds)
    from dataclasses import replace

    # every lane B leg: rev6.20's EMA / no cooldown, ONE EMA (0.10 by default)
    want = replace(want, lr_schedule="warmup_constant_ema", power_ema_sigma_rels=(0.10,),
                   power_ema_eval=0)
    assert lv.lane_b_leg_contract(cfg, base, block=b, spec={}) == want
    # default levers in a spec change nothing
    assert lv.lane_b_leg_contract(cfg, base, block=b, spec={"levers": {"lr_scale": 1.0}}) == want


def test_armed_levers_fold_into_the_leg_contract():
    lv = h.levers()
    from cascade.shared.manifest import contract_digest, contract_payload

    cfg, b = h.armed_cfg()
    base = h.base_contract(cfg)
    plain = lv.lane_b_leg_contract(cfg, base, block=b, spec={})
    c = lv.lane_b_leg_contract(cfg, base, block=b, spec={"levers": LEVERS})
    assert c.lr_schedule == "warmup_constant_ema"
    assert c.lane_b_lr_scale == 1.5 and c.curriculum == (("easy", 40), ("hard", 60))
    assert c.power_ema_sigma_rels == (0.05,) and c.power_ema_eval == 0   # ONE EMA
    assert contract_digest(c) != contract_digest(plain)
    p = contract_payload(c)
    assert p["lane_b_lr_scale"] == 1.5 and p["power_ema_eval"] == 0
    # the default width is the one EMA of a plain EMA leg too (only it computed)
    assert (plain.power_ema_sigma_rels, plain.power_ema_eval) == ((0.10,), 0)
    # no activation block: any dispatch block gives the same contract
    assert lv.lane_b_leg_contract(cfg, base, block=0, spec={"levers": LEVERS}) == c
    assert lv.lane_b_scored_weights_for(cfg.lane_b, 0, lv.LegLevers(ema=0.15)) == "ema015"
    assert lv.lane_b_scored_weights_for(cfg.lane_b, 0, None) == "ema010"
    wide = lv.lane_b_leg_contract(cfg, base, block=b, spec={"levers": {"context_length": 16384}})
    assert wide.context_length == 16384 and base.context_length == 4096


def test_limits_leg_contract_records_the_stop():
    lv = h.levers()
    cfg, b = h.armed_cfg()
    base = h.base_contract(cfg)
    spec = {"train_hours": 12.0, "leg_steps": 9536, "leg_guard_seconds": 43_200,
            "train_tokens": 20 * 10**9}
    c = lv.lane_b_leg_contract(cfg, base, block=b, spec=spec)
    assert (c.training_steps, c.max_train_seconds, c.target_train_hours) == (9536, 43_200, 12.0)
    assert c.lane_b_stop_record is True
    hours_only = lv.lane_b_leg_contract(cfg, base, block=b, spec={
        "train_hours": 2.0, "leg_steps": 100, "leg_guard_seconds": 9600})
    assert hours_only.lane_b_stop_record is False        # WB23's leg, as before


def test_lane_a_contracts_are_byte_identical():
    from cascade.shared.manifest import contract_digest

    off = h.chain_cfg()
    on, b = h.armed_cfg()
    assert [contract_digest(c) for c in on.throne_contracts_at(b)] == \
        [contract_digest(c) for c in off.throne_contracts_at(b)]


def test_ema_choice_applies_only_to_an_ema_leg_contract():
    lv = h.levers()
    base = h.base_contract(h.chain_cfg())
    off = lv.apply_levers(base.for_hours(1.0), lv.LegLevers(ema=0.05))   # a wsd contract
    assert off.power_ema_eval == -1 and off.power_ema_sigma_rels == ()
    assert lv.apply_levers(base.for_hours(1.0), lv.LegLevers(context_length=8192)).context_length \
        == 8192
    rec = lv.leg_context_contract(base, {"context_length": 8192})
    assert rec.context_length == 8192 and lv.leg_context_contract(base, {}) is base


def test_trainer_and_worker_make_the_same_three_steps():
    from cascade.trainer import lane_b_driver

    src = inspect.getsource(lane_b_driver.RunnerLaneBOps._dispatch_funded)
    for call in ("leg_length_contract(self.r.cfg, base, spec)",
                 "lane_b_leg_contract(contract)",
                 "apply_spec_levers(self.r.cfg.lane_b, contract, spec, int(block))",
                 "finetune_recipe"):
        assert call in src, call
    worker = (h.REPO / "cascade" / "trainer" / "worker.py").read_text(encoding="utf-8")
    for call in ("leg_length_contract(cfg, contract, lane_b_spec)",
                 "lane_b_leg_contract(contract)",
                 "apply_spec_levers(cfg.lane_b, contract, lane_b_spec, int(args.block))"):
        assert call in worker, call


def test_remote_argv_carries_the_spec_only_when_set():
    from types import SimpleNamespace

    from cascade.trainer.remote import worker_argv

    host = SimpleNamespace(remote_python="py", chain_toml="")
    kw = dict(gen_ref="g", uid=1, hotkey="hk", role="expert", base_seed=1, block=2,
              trainer_spec="t", domain="energy", expert_selection={0: [1]},
              warm_start_ref="w")
    plain = worker_argv(host, **kw)
    assert "--recipe-json" not in plain
    spec = worker_argv(host, finetune_recipe='{"levers":{"lr_scale":1.5}}', **kw)
    assert spec[:len(plain)] == plain
    assert spec[len(plain):] == ["--recipe-json", '{"levers":{"lr_scale":1.5}}']


def test_driver_leg_spec_fixes_levers_at_dispatch():
    from cascade.trainer.lane_b_driver import _leg_spec

    cfg, b = h.armed_cfg()
    f = {"funded_levers": dict(LEVERS)}
    assert _leg_spec(cfg.lane_b, f, b) == {"levers": LEVERS} and f["levers"] == LEVERS
    f = {"levers": {"x": 1}}
    assert _leg_spec(cfg.lane_b, f, b) == {} and "levers" not in f
    f = {"train_hours": 12.0, "leg_steps": 9, "leg_guard_seconds": 43_200,
         "train_tokens": 99, "gpu_measured": False}
    assert _leg_spec(cfg.lane_b, f, b) == {"train_hours": 12.0, "leg_steps": 9,
                                            "leg_guard_seconds": 43_200, "train_tokens": 99,
                                            "gpu_measured": False}
