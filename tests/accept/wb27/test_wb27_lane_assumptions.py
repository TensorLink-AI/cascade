"""2026-11-26 lane B / lane C audit ("make sure there are no similar bugs for
lane B"): after the first mainnet lane B leg was aborted at "step 1/23841",
every lane-A-tuned guard a lane B / MarketCast leg passes was re-checked. The
same class — a lane A number or assumption applied to a lane B / C leg —
held in five more places; each test below fails on the base by assertion.

1. a LIMITS leg (token limit / own-risk unmeasured GPU) keeps lane A's
   reference steps behind its hour wall and was judged on them (the incident);
2. a WB23 fixed-steps leg made good on the operator account could rent any
   measured type, not the one its steps were fixed on;
3. an operator-rented lane B leg never recorded its layout (gpu_n = 0 signed,
   the wall-bounded judge off);
4. lane A's full-budget SKU wall table timed lane B legs (cost cap, approval,
   balance check) — a 12 h leg looked like 3.4 h;
5. the detached dispatch killed every leg at a fixed 6 h (12 h lane B / C
   legs), and the first-step deadline aborted every MarketCast leg (its
   worker writes no progress record). No torch."""

from __future__ import annotations

import dataclasses
import importlib
import threading
from types import SimpleNamespace

import pytest

from . import helpers as h
from .test_wb27_watchdog import H200_RATE, _ops, _runner


def _guards():
    with h.torch_blocked():
        cg = importlib.import_module("cascade.trainer.cost_guards")
    assert hasattr(cg, "leg_dispatch_timeout") and hasattr(cg, "with_leg_timeout"), \
        "audit: cost_guards lacks the per-leg dispatch timeout"
    return cg


# ── 1 + 2: the driver's scopes ───────────────────────────────────────────────

def _dispatch_seen(cfg, tmp_path, monkeypatch, funding):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg

    seen: list = []
    ops, r = _ops(cfg, tmp_path, monkeypatch, seen)
    import cascade.funding.lane_b_gpus as lbg

    rates = {"8xH200": H200_RATE, "8xRTX4090": 0.1366}
    gpus = SimpleNamespace(
        rent_skus=lambda: ("8xH200", "8xRTX4090"),
        rate=lambda sku: (SimpleNamespace(steps_per_s=rates[sku]) if sku in rates else None))
    monkeypatch.setattr(lbg, "lane_b_gpus_from_env", lambda cfg, **kw: gpus)
    scopes: list = []
    inner = type(r)._run_funded_leg

    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        scopes.append(getattr(self._leg_local_obj, "lane_b_rent_skus", None))
        return inner(self, disp, gen, seeds, block, contract, suffix, **kw)

    monkeypatch.setattr(type(r), "_run_funded_leg", run)
    sub = LaneBSubmission("alice", 7, h.REF, "energy", 5, targets=["energy"])
    leg = OpenLeg(sub=sub, selection={0: [1, 2]}, base_king_ref="kingref", base_era=3,
                  funding=dict(funding))
    ev = threading.Event()
    ops.dispatch_leg(leg, 3, 7300, lambda *a, **k: ev.set())
    assert ev.wait(5)
    (contract, rates), = seen
    return contract, rates, scopes[0]


def test_a_limits_leg_is_judged_wall_bounded(cfg, tmp_path, monkeypatch):
    _guards()
    # a token-limit leg on the measured 8xH200: steps from the tokens (capped at
    # lane A's reference steps), wall = the hour limit
    contract, rates, _ = _dispatch_seen(cfg, tmp_path, monkeypatch, {
        "entry_ref": "x", "gpu": "8xH200", "train_hours": 3.0, "leg_steps": 9536,
        "leg_guard_seconds": 10800, "train_tokens": 20_000_000_000})
    assert rates is not None, "a limits leg must open the wall-bounded watchdog scope"
    assert rates[0] == pytest.approx(H200_RATE * 4096 / contract.context_length)
    # an own-risk unmeasured type: the scope opens, the rate is unknown (judge off)
    _, rates, scope = _dispatch_seen(cfg, tmp_path, monkeypatch, {
        "entry_ref": "x", "gpu": "8xH100", "gpu_measured": False, "train_hours": 3.0,
        "leg_steps": 1, "leg_guard_seconds": 10800})
    assert rates is not None and rates[1] is None
    assert scope == ("8xH100",)


def test_a_fixed_steps_leg_rents_only_its_chosen_type(cfg, tmp_path, monkeypatch):
    _guards()
    _, rates, scope = _dispatch_seen(cfg, tmp_path, monkeypatch, {
        "entry_ref": "x", "gpu": "8xH200", "train_hours": 3.0, "leg_steps": 599,
        "leg_guard_seconds": 13603})
    assert rates is None                       # WB23 steps: the plain judge
    assert scope == ("8xH200",)
    # a leg without train_hours keeps lane B's whole measured list
    _, rates, scope = _dispatch_seen(cfg, tmp_path, monkeypatch,
                                     {"entry_ref": "x", "gpu": "8xH200"})
    assert rates is not None and scope == ("8xH200", "8xRTX4090")


# ── 3: the operator path records the rented layout ───────────────────────────

def test_an_operator_rented_leg_records_its_sku(monkeypatch):
    _guards()
    from cascade.trainer.loop import TrainerRunner

    monkeypatch.setattr("cascade.trainer.runner_operator_pods.leg_corpus",
                        lambda *a, **k: {})
    host = SimpleNamespace(sku="8xH100", host="1.2.3.4")
    r = SimpleNamespace(
        _rent_operator_leg_host=lambda rid, gen: (host, "op-pod-0"),
        _teardown_operator_pod=lambda pod: None, _funded_queue=lambda: None,
        _record_funded_failure=lambda *a, **k: None,
        _harvest_funded_checkpoint=lambda *a: SimpleNamespace(miner_hotkey="hkA"))
    r.__dict__["_leg_local_obj"] = loc = threading.local()
    loc.funded_pod_sku = ""

    class Disp:
        def dispatch(self, host, **kw):
            return SimpleNamespace(miner_hotkey="hkA")

    TrainerRunner._run_operator_rented_leg(
        r, Disp(), SimpleNamespace(hotkey="hkA", uid=1, ref="org/gen@sha256:" + "a" * 64),
        SimpleNamespace(base_seed=7), 1,
        SimpleNamespace(arch_preset="x", corpus_mode="stream_cpu"), "",
        warm_start_ref=None, role="expert")
    assert loc.funded_pod_sku == "8xH100", \
        "an operator-rented leg must record its layout (gpu_n, the watchdog's rate)"


# ── 4: lane A's SKU wall table never times a lane B / MarketCast leg ─────────

def _scoped_runner(cfg, tmp_path, *, wall, lane_b=True):
    from cascade.trainer.loop import TrainerRunner

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    r.cfg = dataclasses.replace(cfg, round=dataclasses.replace(
        cfg.round, funded_sku_wall_seconds=(("8xH200", 12270), ("8xH100", 13600)),
        funded_sku_wall_measured_image="", funded_max_leg_cost_usd=110.0))
    loc = r.__dict__.setdefault("_leg_local_obj", threading.local())
    loc.funded_contract = SimpleNamespace(max_train_seconds=wall)
    if lane_b:
        loc.lane_b_rent_skus = ("8xH200", "8xH100")
    return r


def test_a_lane_b_leg_is_timed_on_its_own_wall(cfg, tmp_path):
    _guards()
    r = _scoped_runner(cfg, tmp_path, wall=54000)            # a 12 h lane B leg
    assert r._leg_wall_seconds() == 54000.0
    assert r._leg_wall_seconds("8xH200") == 54000.0
    assert r._funded_price_caps()["sku_wall_seconds"] == ()
    assert r._funded_price_caps()["default_wall_seconds"] == 54000.0
    # lane A (no scope) keeps the measured table
    a = _scoped_runner(cfg, tmp_path / "a", wall=18000, lane_b=False)
    assert a._leg_wall_seconds("8xH200") == 12270.0
    assert dict(a._funded_price_caps()["sku_wall_seconds"])["8xH200"] == 12270


# ── 5: dispatch wall and the MarketCast watchdog ─────────────────────────────

def test_the_dispatch_wall_follows_the_leg():
    cg = _guards()
    six_h = 6 * 3600
    lane_a = SimpleNamespace(max_train_seconds=18000)
    assert cg.leg_dispatch_timeout(six_h, lane_a, first_step_seconds=2700) == six_h
    assert cg.leg_dispatch_timeout(six_h, SimpleNamespace(max_train_seconds=10800)) == six_h
    assert cg.leg_dispatch_timeout(six_h, SimpleNamespace(max_train_seconds=54000),
                                   first_step_seconds=2700) == 54000 + 3600
    assert cg.leg_dispatch_timeout(six_h, SimpleNamespace(max_train_seconds=43200),
                                   first_step_seconds=5400) == 43200 + 5400
    from cascade.trainer.remote import RemoteDispatcher

    disp = RemoteDispatcher(trainer_spec="m:C")
    out = cg.with_leg_timeout(_runner(), disp, SimpleNamespace(max_train_seconds=54000))
    assert out.timeout_seconds == 54000 + 3600 and disp.timeout_seconds == six_h
    assert cg.with_leg_timeout(_runner(), disp, lane_a) is disp
    assert cg.with_leg_timeout(_runner(), object(), lane_a) is not None


def test_a_marketcast_leg_has_no_first_step_deadline():
    cg = _guards()
    runner = _runner()
    loc = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    contract = SimpleNamespace(training_steps=7947, max_train_seconds=3600)
    gen, pod = SimpleNamespace(hotkey="5Mkt"), SimpleNamespace(instance_id="p", sku="1xRTX4090")
    lane_a = cg.leg_watchdog(runner, gen=gen, pod=pod, contract=contract)
    assert lane_a.first_step == 2700 and lane_a.training_steps == 7947
    loc.mkt_rent_skus = ("1xRTX4090",)
    w = cg.leg_watchdog(runner, gen=gen, pod=pod, contract=contract)
    assert w.first_step == 0 and w.training_steps == 0, \
        "the MarketCast worker writes no progress record: no first-step/throughput judge"
    t = [0.0]
    w.clock = lambda: t[0]
    while t[0] < 3600:                       # every poll: no record at all
        assert w(None) is None
        t[0] += 30
