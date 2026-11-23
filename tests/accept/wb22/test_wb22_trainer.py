"""WB22 trainer: a lane B leg rents from lane B's list (TB23, rent half), lane
A's rent path is unchanged, the leg's GPU count is recorded (``gpu_n``) and
carried into the signed expert manifest entry. The external rent
(``_run_funded_leg``: Lium) is the only stand-in."""

from __future__ import annotations

import dataclasses
import inspect
import json
import threading
from types import SimpleNamespace

import pytest

from .helpers import (
    MAIN_ROUND_SKUS,
    REPO,
    STEP_RATES_ENV,
    chain_cfg,
    gpus,
    norm,
    write_table,
)

GEN = "alice/metro-gen@sha256:" + "a" * 64
PTR = "metro-v1:trained:hippius:org/ckpt@sha256:" + "b" * 64


def _choice(gpu="", price=0.0):
    from cascade.funding.gpu_choice import GpuChoice

    return GpuChoice(gpu=gpu, max_price_per_hour=price)


class _Cap:
    """The real rent-capacity mixin on the shipped config."""

    def __new__(cls, cfg):
        from cascade.trainer.runner_capacity import _FundedCapacityMixin

        klass = type("R", (_FundedCapacityMixin,), {})
        obj = object.__new__(klass)
        obj.cfg = cfg
        return obj


def _lane_b(rates_for):
    from cascade.funding.step_rates import _row

    from .helpers import table

    rates = {s: _row(s, r) for s, r in table(list(rates_for))["rows"].items()}
    return gpus().lane_b_gpus(chain_cfg(), rates)


# ── the rent path (runner_capacity) ──────────────────────────────────────────

def test_lane_a_rent_list_unchanged_when_unarmed():
    """Outside the lane B scope (every lane A leg) the rent path is main's."""
    r = _Cap(chain_cfg())
    assert norm(r._funded_skus_for_rent()) == MAIN_ROUND_SKUS
    assert r._chosen_gpu_rent(_choice("8xRTX4090")) is None
    got = r._chosen_gpu_rent(_choice("8xh100", 30.0))
    assert got is not None and got[0] == ("8xH100",)
    assert got[1]["max_price_per_hour"] == 30.0 and got[1]["max_leg_cost_usd"] == 0.0


def test_lane_a_rent_list_armed_nonempty_lane_d_effect():
    """Inside the lane B scope the rent path reads lane B's measured list."""
    m = gpus()
    r = _Cap(chain_cfg())
    with m.rent_scope(r, _lane_b(["8xRTX4090", "8xH100"])):
        assert norm(r._funded_skus_for_rent()) == ("8xH100", "8xRTX4090")
        assert r._sku_per_leg_active() is True
        got = r._chosen_gpu_rent(_choice("8XRTX4090", 12.0))
        assert got is not None and got[0] == ("8xRTX4090",)
        assert got[1]["max_price_per_hour"] == 12.0 and got[1]["max_leg_cost_usd"] == 0.0
        # listed for lane B but unmeasured: not rentable, the default path
        assert r._chosen_gpu_rent(_choice("8xH200")) is None
    assert norm(r._funded_skus_for_rent()) == MAIN_ROUND_SKUS


def test_lane_b_scope_makes_per_leg_choice_even_with_round_locked():
    m = gpus()
    cfg = chain_cfg()
    cfg = dataclasses.replace(cfg, round=dataclasses.replace(cfg.round, funded_sku_per_leg=False))
    r = _Cap(cfg)
    assert r._sku_per_leg_active() is False
    with m.rent_scope(r, _lane_b(["8xH100"])):
        assert r._sku_per_leg_active() is True
        assert norm(r._funded_skus_for_rent()) == ("8xH100",)


# ── the driver (RunnerLaneBOps._dispatch_funded) ─────────────────────────────

def _runner(tmp_path, monkeypatch, run):
    from cascade.trainer.loop import TrainerRunner

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


def _leg(gpu="8xRTX4090"):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg

    return OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5),
                   selection={0: [1, 2]}, base_king_ref="kingref", base_era=3,
                   funding={"entry_ref": "x", "gpu": gpu, "gpu_allow_fallback": False,
                            "gpu_max_price_per_hour": 12.0})


def _dispatch(r, leg):
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    ops = RunnerLaneBOps(r, client=None, window_source=None)
    ev, results = threading.Event(), []

    def done(leg, ptr, *a, failure=None, **kw):
        results.append((ptr, a, failure, kw))
        ev.set()

    ops.dispatch_leg(leg, 3, 7300, done)
    assert ev.wait(10), "the leg never returned"
    return results[0]


def test_dispatch_rents_from_lane_b_list_and_records_gpu_n(tmp_path, monkeypatch):
    seen = {}

    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        seen["skus"] = tuple(self._funded_skus_for_rent())
        seen["chosen"] = self._chosen_gpu_rent(kw["gpu_choice"])
        # what the real rent records on the leg's thread (runner_funded_legs)
        self.__dict__.setdefault("_leg_local_obj", threading.local()).funded_pod_sku = \
            "8xRTX4090"
        return SimpleNamespace(trained_pointer=PTR, corpus_digest="cd", train_block=block,
                               gpu_name="NVIDIA GeForce RTX 4090")

    monkeypatch.setenv(STEP_RATES_ENV, str(write_table(tmp_path / "t.json",
                                                       ["8xRTX4090", "8xH200"])))
    r = _runner(tmp_path, monkeypatch, run)
    ptr, args, failure, kw = _dispatch(r, _leg())
    assert failure is None and ptr == PTR
    assert seen["skus"] == ("8xH200", "8xRTX4090"), seen
    assert seen["chosen"] is not None and seen["chosen"][0] == ("8xRTX4090",)
    assert kw.get("gpu_n") == 8, f"done() got no gpu_n=8 (kwargs {kw})"


def test_dispatch_with_nothing_measured_never_rents(tmp_path, monkeypatch):
    calls = []

    def run(self, *a, **kw):
        calls.append(1)
        return SimpleNamespace(trained_pointer=PTR, corpus_digest="cd", train_block=1,
                               gpu_name="")

    monkeypatch.setenv(STEP_RATES_ENV, str(tmp_path / "absent.json"))
    r = _runner(tmp_path, monkeypatch, run)
    ptr, _, failure, _ = _dispatch(r, _leg())
    assert calls == [], "an unmeasured lane B list must never reach the rent"
    assert ptr is None and failure is not None
    assert (failure.error_class, failure.burn, failure.miner_fault) == ("no_capacity", False,
                                                                        False)


def test_rent_records_the_pod_sku_on_the_leg_thread():
    src = (REPO / "cascade" / "trainer" / "runner_funded_legs.py").read_text(encoding="utf-8")
    body = src[src.index("def _run_funded_leg_scoped("):]
    rent = body.index("self._rent_funded_host(")
    assert "funded_pod_sku" in body[rent:rent + 2500], (
        "_run_funded_leg_scoped must set _loc.funded_pod_sku from the rented pod's sku "
        "right after _rent_funded_host")


# ── gpu_n in the driver state and the signed expert manifest ─────────────────

def test_finished_leg_and_leg_done_carry_gpu_n():
    from cascade.trainer.lane_b_driver import FinishedExpertLeg, LaneBDriver

    names = {f.name: f for f in dataclasses.fields(FinishedExpertLeg)}
    assert "gpu_n" in names and names["gpu_n"].default == 0
    params = inspect.signature(LaneBDriver._leg_done).parameters
    assert "gpu_n" in params and params["gpu_n"].default == 0
    settle = inspect.getsource(LaneBDriver.settle)
    assert "gpu_n=" in settle, "settle must pass the recorded gpu_n into ExpertEntry"


def _entry(**kw):
    from cascade.shared.expert_manifest import ExpertEntry

    return ExpertEntry(miner_hotkey="hk", miner_uid=1, gen_ref=GEN, domain="energy",
                       selection={0: [1, 2]}, exclude={}, base_king_pointer=PTR,
                       rebased_pointer=PTR, corpus_digest="cd", train_block=5,
                       commit_block=4, gpu_name="NVIDIA H100 80GB HBM3", **kw)


def _manifest(entry):
    from cascade.shared.expert_manifest import ExpertManifest

    return ExpertManifest(round_id="r1", created_block=10, king_pointer=PTR,
                          lineage_tokens=1, entries=(entry,))


def test_expert_entry_gpu_n_round_trips_and_is_dropped_at_zero():
    from cascade.shared.expert_manifest import ExpertEntry, load_expert_manifest

    assert "gpu_n" in {f.name for f in dataclasses.fields(ExpertEntry)}, (
        "ExpertEntry has no gpu_n (WB22/WB25: the leg's GPU count)")
    legacy = _manifest(_entry())
    body = json.loads(legacy.canonical_body())
    assert "gpu_n" not in body["entries"][0], "gpu_n = 0 must keep main's bytes"
    m = _manifest(_entry(gpu_n=8))
    assert json.loads(m.canonical_body())["entries"][0]["gpu_n"] == 8
    back = load_expert_manifest(m.to_json())
    assert back.entries[0].gpu_n == 8 and back.digest() == m.digest()
    assert load_expert_manifest(legacy.to_json()).entries[0].gpu_n == 0


@pytest.mark.parametrize("bad", [-1, True])
def test_expert_entry_refuses_a_bad_gpu_n(bad):
    from cascade.shared.expert_manifest import ExpertEntry

    assert "gpu_n" in {f.name for f in dataclasses.fields(ExpertEntry)}
    with pytest.raises(ValueError):
        _entry(gpu_n=bad)
