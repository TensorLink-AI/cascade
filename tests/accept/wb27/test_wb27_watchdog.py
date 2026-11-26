"""2026-11-26 first mainnet lane B leg (5GMzsP…, energy, 8xH200): THROUGHPUT
ABORT twice, on two hosts, at "step 1/23841 after ~600s". Not a hang: a leg
funded without ``train_hours`` trains ``[lane_b] leg_hours`` via ``for_hours``
— lane A's 23,841 steps behind a 3 h wall — while the expert-only step runs
at the WB19-measured 0.055 steps/s on 8xH200 (~18 s/step). The worker writes
its progress record at step 1 and every 50 steps, so at the 600 s early judge
the record still said step 1, and 23,841 steps projected to ~80 h. Such a leg
is wall-capped by design (``deadline_hit``); the watchdog now judges it on its
wall's worth of steps at the measured rate of the rented type. No torch."""

from __future__ import annotations

import dataclasses
import importlib
import threading
from types import SimpleNamespace

import pytest

from . import helpers as h

STEPS = 23841            # the v2 contract's training_steps (lane A)
WALL = 10800             # [lane_b] leg_hours 3.0 → for_hours guard (factor 1.0)
H200_RATE = 0.05547751688587357   # evidence/wb19.json, 8xH200 expert run


def _guards():
    with h.torch_blocked():
        cg = importlib.import_module("cascade.trainer.cost_guards")
    # Fails by assertion (not AttributeError) on a base without the fix.
    assert hasattr(cg, "wall_bounded_steps") and hasattr(cg, "wall_bounded_leg"), \
        "WB27: cost_guards lacks the wall-bounded expert-leg watchdog"
    return cg


def _runner():
    rnd = SimpleNamespace(progress_watchdog_grace_seconds=1200,
                          progress_watchdog_early_seconds=600,
                          progress_first_step_seconds=2700, payer_key_check_seconds=0,
                          progress_watchdog_tolerance=1.5)
    return SimpleNamespace(cfg=SimpleNamespace(round=rnd))


def _watchdog(runner, *, clock, sku="8xH200"):
    cg = _guards()
    w = cg.leg_watchdog(runner, gen=SimpleNamespace(hotkey="5GMzsPiUhy6gtqU5"),
                        pod=SimpleNamespace(instance_id="pod-0", sku=sku),
                        contract=SimpleNamespace(training_steps=STEPS, max_train_seconds=WALL))
    w.clock = clock
    w._started = None
    return w


def _run(w, rate, *, until, step1_at=40.0, poll=30.0):
    """Feed ``w`` the worker's records (step 1, then every 50 steps) for a leg
    training at ``rate`` steps/s; returns ``(verdict, t)``."""
    t = [0.0]
    w.clock = lambda: t[0]
    while t[0] <= until:
        if t[0] < step1_at:
            rec = None
        else:
            done = 1 + int((t[0] - step1_at) * rate)
            rec = {"step": 1 if done < 50 else done - done % 50, "tokens": 0}
        v = w(rec)
        if v is not None:
            return v, t[0]
        t[0] += poll
    return None, t[0]


def test_wall_bounded_steps():
    cg = _guards()
    assert cg.wall_bounded_steps(STEPS, WALL, H200_RATE) == 599
    assert cg.wall_bounded_steps(500, WALL, H200_RATE) == 500       # never above the contract
    assert cg.wall_bounded_steps(STEPS, WALL, 100.0) == STEPS
    for bad in (None, 0, -1.0, float("nan"), "x"):
        assert cg.wall_bounded_steps(STEPS, WALL, bad) == 0         # judge off
    assert cg.wall_bounded_steps(STEPS, 0, H200_RATE) == 0


def test_the_incident_aborts_without_the_scope_and_trains_with_it(caplog):
    cg = _guards()
    # before: the contract's lane A steps — aborted at the early judge, any host
    w = _watchdog(_runner(), clock=lambda: 0.0)
    assert w.training_steps == STEPS
    verdict, t = _run(w, H200_RATE, until=3600)
    assert verdict and cg.THROUGHPUT_ABORT in verdict and "step 1/23841" in verdict
    assert 600 <= t < 700
    # after: the wall-bounded judge on the measured 8xH200 rate
    runner = _runner()
    with cg.wall_bounded_leg(runner, lambda sku: H200_RATE if sku == "8xH200" else None):
        caplog.set_level("INFO", logger="cascade.trainer")
        w = _watchdog(runner, clock=lambda: 0.0)
    assert runner._leg_local_obj.wall_bounded_rate_of is None       # scope closed
    assert w.training_steps == 599 and w.max_train_seconds == WALL
    assert "wall-bounded lane B leg on 8xH200" in caplog.text
    assert _run(w, H200_RATE, until=WALL) == (None, pytest.approx(WALL, abs=30))


def test_a_slow_host_and_a_stall_still_abort():
    cg = _guards()
    rate_of = {"8xH200": H200_RATE}.get
    for rate in (H200_RATE / 3, 0.0):
        runner = _runner()
        with cg.wall_bounded_leg(runner, rate_of):
            w = _watchdog(runner, clock=lambda: 0.0)
        verdict, t = _run(w, rate, until=WALL)
        assert verdict and cg.THROUGHPUT_ABORT in verdict and "/599 " in verdict
        assert t <= 1500                     # just past the grace judge (one record credited)


def test_an_unmeasured_type_turns_only_the_throughput_judge_off():
    cg = _guards()
    runner = _runner()
    with cg.wall_bounded_leg(runner, lambda sku: None):
        w = _watchdog(runner, clock=lambda: 0.0, sku="4xL40S")
    assert w.training_steps == 0 and w.first_step == 2700
    assert _run(w, 0.0, until=WALL)[0] is None                      # wall still caps the leg
    w2 = _watchdog(runner, clock=lambda: 0.0)                       # outside the scope
    assert w2.training_steps == STEPS


def _ops(cfg, tmp_path, monkeypatch, seen):
    import cascade.funding.lane_b_gpus as lbg
    from cascade.trainer.lane_b_driver import RunnerLaneBOps
    from cascade.trainer.loop import TrainerRunner

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    era = SimpleNamespace(base_seed=12345, start_block=7200, index=3)
    r.cfg = dataclasses.replace(cfg, round=dataclasses.replace(cfg.round,
                                                               funded_mode="required"))
    r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(current=era))
    monkeypatch.setattr(TrainerRunner, "_dispatcher", lambda self: object())

    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        rate_of = getattr(self._leg_local_obj, "wall_bounded_rate_of", None)
        seen.append((contract, None if rate_of is None else
                     (rate_of("8xH200"), rate_of("8xH100"))))
        return SimpleNamespace(trained_pointer="ptr", corpus_digest="cd", train_block=block,
                               gpu_name="H200")

    monkeypatch.setattr(TrainerRunner, "_run_funded_leg", run)
    gpus = SimpleNamespace(
        rent_skus=lambda: ("8xH200",),
        rate=lambda sku: SimpleNamespace(steps_per_s=H200_RATE) if sku == "8xH200" else None)
    monkeypatch.setattr(lbg, "lane_b_gpus_from_env", lambda cfg, **kw: gpus)
    return RunnerLaneBOps(r, client=None, window_source=None), r


def test_the_driver_opens_the_scope_only_for_a_leg_without_train_hours(cfg, tmp_path,
                                                                       monkeypatch):
    _guards()                       # by assertion on a base without the fix
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg

    seen: list = []
    ops, r = _ops(cfg, tmp_path, monkeypatch, seen)
    sub = LaneBSubmission("alice", 7, h.REF, "energy", 5, targets=["energy"])
    base = {"entry_ref": "x", "gpu": "8xH200"}
    for funding in (base, {**base, "train_hours": 3.0, "leg_steps": 599,
                           "leg_guard_seconds": 11400}):
        leg = OpenLeg(sub=sub, selection={0: [1, 2]}, base_king_ref="kingref", base_era=3,
                      funding=dict(funding))
        ev = threading.Event()
        ops.dispatch_leg(leg, 3, 7300, lambda *a, ev=ev, **k: ev.set())
        assert ev.wait(5)
    (legacy, rates), (wb23, none) = seen
    assert rates[0] == pytest.approx(H200_RATE * 4096 / legacy.context_length)
    assert rates[1] is None and none is None
    assert wb23.training_steps <= 599          # WB23: the fund already fixed (clamped) steps
    assert getattr(r._leg_local_obj, "wall_bounded_rate_of", None) is None
