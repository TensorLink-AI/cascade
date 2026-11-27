"""WB23 trainer (TB25, TB26): the funding record carries the leg length, a
leg with ``train_hours`` trains the lane B leg contract (steps fixed at
admission, guard capped at ``max_leg_seconds``), a leg that cannot land in
the current era is held to the next era's pre-train window, and the leg
length reaches status and the signed expert manifest entry. The external
rent (``_run_funded_leg``) is the only stand-in."""

from __future__ import annotations

import dataclasses
import inspect
import threading
from types import SimpleNamespace

import pytest

from .helpers import (
    SKU,
    STEP_RATES_ENV,
    base_contract,
    chain_cfg,
    lane_b_cfg,
    leg,
    step_rate,
    table,
)

GEN = "alice/metro-gen@sha256:" + "a" * 64
PTR = "metro-v1:trained:hippius:org/ckpt@sha256:" + "b" * 64


def _runner(tmp_path, monkeypatch, run):
    import json

    from cascade.trainer.loop import TrainerRunner

    p = tmp_path / "t.json"
    p.write_text(json.dumps(table([SKU])), encoding="utf-8")
    monkeypatch.setenv(STEP_RATES_ENV, str(p))
    cfg = chain_cfg()
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    era = SimpleNamespace(base_seed=12345, start_block=int(cfg.training.v2_from_block), index=3)
    r.cfg = dataclasses.replace(cfg, round=dataclasses.replace(cfg.round,
                                                               funded_mode="required"))
    r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(current=era))
    monkeypatch.setattr(TrainerRunner, "_dispatcher", lambda self: object())
    monkeypatch.setattr(TrainerRunner, "_rolling_lane_pool",
                        lambda self: pytest.fail("operator pool used"))
    monkeypatch.setattr(TrainerRunner, "_run_funded_leg", run)
    return r


def _leg(**length):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg

    funding = {"entry_ref": "x", "gpu": SKU, "gpu_allow_fallback": False,
               "gpu_max_price_per_hour": 30.0, **length}
    return OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5),
                   selection={0: [1, 2]}, base_king_ref="kingref", base_era=3,
                   funding=funding)


def _dispatch(r, open_leg):
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    ops = RunnerLaneBOps(r, client=None, window_source=None)
    ev, results = threading.Event(), []

    def done(leg, ptr, *a, failure=None, **kw):
        results.append((ptr, failure))
        ev.set()

    ops.dispatch_leg(open_leg, 3, 7300, done)
    assert ev.wait(10), "the leg never returned"
    return results[0]


def _capture(seen):
    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        seen["contract"] = contract
        return SimpleNamespace(trained_pointer=PTR, corpus_digest="cd", train_block=block,
                               gpu_name="NVIDIA H100 80GB HBM3")
    return run


def _lane_b_ema(c):
    """rev6.20 (#845, no key) + WB27 (owner 2026-11-24; tests-correction):
    every lane B leg contract is the EMA / no-cooldown one with its ONE EMA
    width (0.10 by default); the WB23 length terms are unchanged."""
    return dataclasses.replace(c, lr_schedule="warmup_constant_ema",
                               power_ema_sigma_rels=(0.10,), power_ema_eval=0)


def test_dispatch_contract_unchanged_when_unarmed(tmp_path, monkeypatch):
    """No train_hours on the funding record: today's for_hours(leg_hours)."""
    seen = {}
    r = _runner(tmp_path, monkeypatch, _capture(seen))
    ptr, failure = _dispatch(r, _leg())
    assert failure is None and ptr == PTR
    cfg = r.cfg
    want = base_contract(cfg).for_hours(cfg.lane_b.leg_hours,
                                        guard_factor=cfg.round.heat_guard_factor,
                                        guard_floor_seconds=cfg.round.heat_guard_floor_seconds)
    # no spec (no train_hours, no levers): the worker folds no levers, so the
    # trainer expects the EMA leg contract with the image's EMAs (WB27
    # 2026-11-27 owner fix: the receipt digest the worker reports)
    assert seen["contract"] == dataclasses.replace(want, lr_schedule="warmup_constant_ema")


def test_dispatch_contract_armed_nonempty_lane_d_effect(tmp_path, monkeypatch):
    """TB25: a 6 h leg trains the steps fixed at admission under the lane B cap."""
    m = leg()
    lb = lane_b_cfg()
    length = m.admit_leg_length(lb, step_rate(), 6.0, contract=base_contract())
    seen = {}
    r = _runner(tmp_path, monkeypatch, _capture(seen))
    ptr, failure = _dispatch(r, _leg(train_hours=6.0, leg_steps=length.steps,
                                     leg_guard_seconds=length.guard_seconds))
    assert failure is None and ptr == PTR
    c = seen["contract"]
    assert c.training_steps == length.steps
    assert c.max_train_seconds == length.guard_seconds <= int(lb.max_leg_seconds)
    assert c == _lane_b_ema(m.leg_contract(base_contract(r.cfg), length))


def test_mark_funded_carries_the_leg_length():
    from cascade.trainer.expert_lane import LaneBSubmission
    from cascade.trainer.lane_b_driver import LaneBDriver

    entry = SimpleNamespace(ref="x", gpu=SKU, gpu_allow_fallback=False,
                            gpu_max_price_per_hour=30.0, train_hours=2.0, leg_steps=900,
                            leg_guard_seconds=9600)
    fq = SimpleNamespace(mark_in_flight=lambda *a, **k: True, get=lambda hk: None,
                         entries=lambda: [])
    drv = object.__new__(LaneBDriver)
    sub = LaneBSubmission("alice", 7, GEN, "energy", 5)
    import cascade.trainer.progress as progress

    orig = progress.funded_attempt_id
    progress.funded_attempt_id = lambda *a, **k: "att"
    try:
        rec = LaneBDriver._mark_funded(drv, fq, {"alice": entry}, sub, 3, 100)
        legacy = LaneBDriver._mark_funded(
            drv, fq, {"alice": SimpleNamespace(**{**vars(entry), "train_hours": 0.0,
                                                  "leg_steps": 0, "leg_guard_seconds": 0})},
            sub, 3, 100)
    finally:
        progress.funded_attempt_id = orig
    assert (rec.get("train_hours"), rec.get("leg_steps"), rec.get("leg_guard_seconds")) == (
        2.0, 900, 9600), rec
    assert not {"train_hours", "leg_steps", "leg_guard_seconds"} & set(legacy), legacy


# ── TB26: cross-era admission of a long leg ──────────────────────────────────

def test_tb26_leg_length_admission_is_rolling_admit():
    from cascade.trainer import lane_b_driver, rolling

    assert hasattr(lane_b_driver, "leg_length_admission"), (
        "lane_b_driver has no leg_length_admission (WB23 / TB26)")
    cfg = chain_cfg()
    rc, b = cfg.round, int(cfg.training.v2_from_block)
    crossed = []
    for off in (5, 3550):   # early in an era, then late in it
        kw = dict(block_now=b + off, now=1_700_000_000.0, margin_seconds=900.0,
                  current_era=rolling.settlement_era(rc, b + off).index)
        want = rolling.admit(rc, wall_seconds=27_000, **kw)
        assert lane_b_driver.leg_length_admission(rc, guard_seconds=27_000, **kw) == want
        crossed.append(want.cross_era)
    assert crossed == [False, True], "the fixture must cover both an in-era and a held leg"


def test_tb26_long_leg_late_in_era_is_held_source():
    from cascade.trainer import lane_b_driver

    src = inspect.getsource(lane_b_driver)
    assert "def leg_length_admission" in src, "lane_b_driver has no leg_length_admission"
    assert src.count("leg_length_admission(") >= 2, (
        "the driver never calls leg_length_admission for a funded leg")
    assert '"leg_length"' in src, "a held leg must be reported deferred: leg_length"


def test_status_open_rows_show_the_leg_length():
    from cascade.trainer.lane_b_driver import LaneBDriver

    src = inspect.getsource(LaneBDriver.status)
    for k in ('"train_hours"', '"steps"', '"guard_seconds"'):
        assert k in src, f"status()['open'] rows carry no {k}"


def test_settle_writes_the_leg_length_into_the_entry():
    from cascade.trainer.lane_b_driver import LaneBDriver

    settle = inspect.getsource(LaneBDriver.settle)
    for k in ("train_hours=", "leg_steps=", "leg_guard_seconds="):
        assert k in settle, f"settle passes no {k} into ExpertEntry"
