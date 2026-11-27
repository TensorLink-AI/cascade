"""WB27 2026-11-27 (owner: "fix lane B now"): two lane B record defects.

1. A lane B leg funded WITHOUT levers or ``train_hours`` (no spec, no
   ``--recipe-json``): the worker folds no levers (``if lane_b_spec:``), so
   the trainer must not either — its expected receipt digest is the
   worker's (live: a3ea79b2... at contract block 9594000, not ebfed637...).
2. A re-upload of a pending harvest records the leg with the selection,
   contract block and exclude the checkpoint was TRAINED with (carried on
   the pending record), never the selection the current tick plans.

Torch-free: nothing here imports torch or loads a checkpoint."""

from __future__ import annotations

import dataclasses
import inspect
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import helpers as h

GEN = "alice/metro-gen@sha256:" + "a" * 64
PTR = "metro-v1:trained:hippius:org/ckpt@sha256:" + "b" * 64
LIVE_CFG = Path("/root/cascade/chain.validator.toml")


def _live_cfg_readable() -> bool:
    try:
        return LIVE_CFG.is_file()
    except OSError:            # CI: /root is not readable
        return False


def _fields():
    """``runner_common.lane_b_dispatch_fields`` (the pending-record fields)."""
    from cascade.trainer import runner_common

    f = getattr(runner_common, "lane_b_dispatch_fields", None)
    assert f is not None, "the pending record carries no lane B dispatch fields"
    return f


def _rows():
    from cascade.trainer import runner_common

    f = getattr(runner_common, "lane_b_record_rows", None)
    assert f is not None, "no lane B record rows reader"
    return f
LIVE_CONTRACT_BLOCK = 9594000
LIVE_DISPATCH_BLOCK = 9596663
LIVE_RECEIPT = "a3ea79b2b236eea6c01b7c41bbcf4323f026f65166db6e4cf85d5bb6fc770533"
TRAINED = {0: [23, 31], 6: [61, 82, 84]}
TICK = {1: [9]}


def _worker_leg_contract(cfg, contract_block: int, spec: dict | None):
    """The worker's ``--role expert`` contract steps (worker.py), torch-free:
    the era contract at ``--contract-block``, the WB23 length or
    ``for_hours(leg_hours)``, rev6.20's EMA leg, the levers ONLY with a spec."""
    from cascade.shared.config.lane_b import lane_b_leg_contract
    from cascade.shared.lane_b_levers import apply_spec_levers, leg_length_contract

    contract = cfg.training.at_block(contract_block).primary_size
    if spec and float(spec.get("train_hours", 0.0) or 0.0) > 0:
        contract = leg_length_contract(cfg, contract, spec)
    else:
        contract = contract.for_hours(float(cfg.lane_b.leg_hours),
                                      guard_factor=cfg.round.heat_guard_factor,
                                      guard_floor_seconds=cfg.round.heat_guard_floor_seconds)
    contract = lane_b_leg_contract(contract)
    if spec:
        contract = apply_spec_levers(cfg.lane_b, contract, spec, contract_block)
    return contract


def test_no_spec_leg_contract_is_the_workers():
    # the rule this fix mirrors lives in the worker image: pin it
    src = (h.REPO / "cascade" / "trainer" / "worker.py").read_text(encoding="utf-8")
    i = src.index("apply_spec_levers(cfg.lane_b, contract, lane_b_spec")
    assert "if lane_b_spec:" in src[src.rindex("lane_b_leg_contract(contract)", 0, i):i]
    with h.torch_blocked():
        from cascade.shared.lane_b_levers import lane_b_leg_contract
        from cascade.shared.manifest import contract_digest

        cfg = h.chain_cfg()
        b = h.effective_block(cfg)
        base = cfg.throne_contracts_at(b)[0]
        trainer = lane_b_leg_contract(cfg, base, block=b, spec={})
        worker = _worker_leg_contract(cfg, b, None)
        assert (trainer.power_ema_sigma_rels, trainer.power_ema_eval) == ((), -1)
        assert contract_digest(trainer) == contract_digest(worker)
        # a spec (here only a length) still folds the default levers, both sides
        spec = {"train_hours": 2.0, "leg_steps": 100, "leg_guard_seconds": 9600}
        t2 = lane_b_leg_contract(cfg, base, block=b, spec=spec)
        assert (t2.power_ema_sigma_rels, t2.power_ema_eval) == ((0.10,), 0)
        assert contract_digest(t2) == contract_digest(_worker_leg_contract(cfg, b, spec))
        if _live_cfg_readable():
            # on the box: the 2026-11-27 leg — the trainer now expects the
            # digest its receipt carries, at the leg's contract block
            from cascade.shared.config import load_chain_config

            live = load_chain_config(str(LIVE_CFG))
            lbase = live.throne_contracts_at(LIVE_CONTRACT_BLOCK)[0]
            got = lane_b_leg_contract(live, lbase, block=LIVE_DISPATCH_BLOCK, spec={})
            assert contract_digest(got) == LIVE_RECEIPT
            assert contract_digest(_worker_leg_contract(
                live, LIVE_CONTRACT_BLOCK, None)) == LIVE_RECEIPT


# ── the dispatch (RunnerLaneBOps → _run_funded_leg stand-in) ─────────────────


def _runner(tmp_path, monkeypatch, run):
    from cascade.trainer.loop import TrainerRunner

    cfg = h.chain_cfg()
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    era = SimpleNamespace(base_seed=12345, start_block=int(cfg.training.v2_from_block), index=3)
    r.cfg = dataclasses.replace(cfg, round=dataclasses.replace(cfg.round, funded_mode="required"))
    r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(current=era))
    monkeypatch.setattr(TrainerRunner, "_dispatcher", lambda self: object())
    monkeypatch.setattr(TrainerRunner, "_rolling_lane_pool",
                        lambda self: pytest.fail("operator pool used"))
    monkeypatch.setattr(TrainerRunner, "_era_contract_block", lambda self, era: 4242)
    monkeypatch.setattr(TrainerRunner, "_run_funded_leg", run)
    # the WB19 step-rate table lane B rents from (WB23's stand-in)
    from tests.accept.wb23.helpers import SKU, STEP_RATES_ENV, table

    p = tmp_path / "rates.json"
    p.write_text(json.dumps(table([SKU])), encoding="utf-8")
    monkeypatch.setenv(STEP_RATES_ENV, str(p))
    return r


def _leg(selection=None, exclude=None):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from tests.accept.wb23.helpers import SKU

    funding = {"entry_ref": "x", "gpu": SKU, "gpu_allow_fallback": False,
               "gpu_max_price_per_hour": 30.0}
    return OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5),
                   selection=dict(selection or TICK), base_king_ref="kingref", base_era=3,
                   exclude=dict(exclude or {}), funding=funding)


def _dispatch(r, open_leg):
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    ops = RunnerLaneBOps(r, client=None, window_source=None)
    ev, results = threading.Event(), []

    def done(leg, ptr, *a, failure=None, **kw):
        loc = r.__dict__.get("_leg_local_obj")          # this leg's thread
        kw["_left_dispatch"] = getattr(loc, "lane_b_dispatch", None)
        results.append((ptr, failure, kw))
        ev.set()

    ops.dispatch_leg(open_leg, 3, 7300, done)
    assert ev.wait(10), "the leg never returned"
    return results[0]


def _entry(block=7300):
    return SimpleNamespace(trained_pointer=PTR, corpus_digest="cd", train_block=block,
                           gpu_name="NVIDIA H200")


def test_dispatch_trains_the_workers_contract_for_a_no_spec_leg(tmp_path, monkeypatch):
    from cascade.shared.manifest import contract_digest

    seen = {}

    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        seen.update(contract=contract, kw=kw)
        return _entry(block)

    r = _runner(tmp_path, monkeypatch, run)
    ptr, failure, _ = _dispatch(r, _leg())
    assert failure is None and ptr == PTR, failure
    assert "finetune_recipe" not in seen["kw"]            # no --recipe-json: no spec
    era_block = int(r.cfg.training.v2_from_block)
    worker = _worker_leg_contract(r.cfg, era_block, None)
    assert contract_digest(seen["contract"]) == contract_digest(worker)


def test_reuploaded_harvest_is_recorded_with_its_trained_selection(tmp_path, monkeypatch):
    lane_b_dispatch_fields = _fields()

    seen = {}

    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        loc = self.__dict__["_leg_local_obj"]
        seen["dispatch"] = dict(loc.lane_b_dispatch)
        # what _reupload_pending leaves after re-uploading an earlier harvest
        loc.lane_b_reused = lane_b_dispatch_fields(TRAINED, 4242, {})
        return _entry(9596663)

    r = _runner(tmp_path, monkeypatch, run)
    ptr, failure, kw = _dispatch(r, _leg(selection=TICK))
    assert failure is None and ptr == PTR, failure
    assert seen["dispatch"] == {"expert_selection": {"1": [9]}, "contract_block": 4242,
                                "expert_exclude": {}}
    assert kw["selection"] == TRAINED and kw["exclude"] == {}
    assert kw["_left_dispatch"] is None                  # never leaks past the leg

    # a fresh leg (no re-upload) is recorded with its own selection
    def fresh(self, disp, gen, seeds, block, contract, suffix, **kw):
        return _entry(block)

    monkeypatch.setattr(type(r), "_run_funded_leg", fresh)
    ptr, failure, kw = _dispatch(r, _leg(selection=TRAINED))
    assert failure is None and ptr == PTR and "selection" not in kw


def test_leg_done_records_the_trained_selection():
    from cascade.trainer.expert_lane import LaneBQueue
    from cascade.trainer.lane_b_driver import LaneBDriver, LaneBDriverState

    assert "selection" in inspect.signature(LaneBDriver._leg_done).parameters, (
        "_leg_done cannot record a re-uploaded leg's trained selection")
    drv = LaneBDriver.__new__(LaneBDriver)
    leg = _leg(selection=TICK, exclude={2: [5]})
    drv._lock, drv._running, drv.st = threading.RLock(), {}, LaneBDriverState()
    drv.queue = LaneBQueue(open=[leg])
    drv.ops = SimpleNamespace()
    drv.save = lambda: None
    drv._leg_done(leg, PTR, "cd", 9596663, "NVIDIA H200", selection=TRAINED, exclude={})
    (fin,) = drv.st.finished
    assert {int(k): v for k, v in fin["leg"]["selection"].items()} == TRAINED
    assert fin["leg"]["exclude"] == {} and fin["train_block"] == 9596663


# ── the pending record (write at harvest, bind at re-upload) ─────────────────


def _record(tmp_path, **extra):
    dest = tmp_path / "harvest"
    dest.mkdir(exist_ok=True)
    return {"dest": str(dest), "harvest_job_id": "job1", "ckpt_repo": "org/ckpt",
            "suffix": "-xenergy-u7", "size": "cascade-moe-v2-120m", "miner_hotkey": "alice",
            "miner_uid": 7, "role": "challenger", "gen_ref": GEN, "corpus_digest": "cd",
            "train_block": 9596663, "gpu_name": "NVIDIA H200",
            "receipt_size": "cascade-moe-v2-120m", "warm_start_ref": "kingref",
            "contract_digest": "", **extra}


def test_failed_upload_records_the_dispatch_fields(tmp_path, monkeypatch):
    from cascade.trainer import loop
    from cascade.trainer.runner_common import read_pending_upload

    lane_b_dispatch_fields = _fields()

    monkeypatch.setattr(loop, "upload_dir_to_hub_or_hf",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("hub down")))
    loc = threading.local()
    loc.lane_b_dispatch = lane_b_dispatch_fields(TRAINED, 9594000, {3: [1]})
    runner = SimpleNamespace(work_root=tmp_path, _upload_retry_sleep=lambda s: None,
                             hub=lambda: None, _hf_ckpt_repo=lambda repo: None)
    runner.__dict__["_leg_local_obj"] = loc
    with pytest.raises(Exception, match="upload failed"):
        loop.upload_harvested_checkpoint(runner, _record(tmp_path), base_seed=1)
    rec = read_pending_upload(tmp_path, 1, "alice")
    assert rec["expert_selection"] == {"0": [23, 31], "6": [61, 82, 84]}
    assert rec["contract_block"] == 9594000 and rec["expert_exclude"] == {"3": [1]}
    # a re-upload's failure keeps the TRAINED fields, never the current tick's
    loc.lane_b_dispatch = lane_b_dispatch_fields(TICK, 9594000, {})
    with pytest.raises(Exception, match="upload failed"):
        loop.upload_harvested_checkpoint(runner, rec, base_seed=1)
    assert read_pending_upload(tmp_path, 1, "alice")["expert_selection"] == rec[
        "expert_selection"]


def _reupload(tmp_path, monkeypatch, rec, dispatch):
    from cascade.shared import manifest
    from cascade.trainer import loop
    from cascade.trainer import runner_funded_legs as rfl
    from cascade.trainer.runner_common import write_pending_upload

    write_pending_upload(tmp_path, 1, "alice", rec)
    monkeypatch.setattr(manifest, "contract_digest", lambda c: "")
    uploaded = []
    monkeypatch.setattr(loop, "upload_harvested_checkpoint",
                        lambda runner, r, base_seed: uploaded.append(r) or "entry")
    loc = threading.local()
    loc.lane_b_dispatch = dispatch
    runner = SimpleNamespace(work_root=tmp_path, _record_funded_failure=lambda *a, **k: None)
    runner.__dict__["_leg_local_obj"] = loc
    gen = SimpleNamespace(hotkey="alice", ref=GEN)
    contract = SimpleNamespace(arch_preset="cascade-moe-v2-120m", base_arch="cascade_moe")
    got = rfl._reupload_pending(runner, gen, SimpleNamespace(base_seed=1), contract,
                                "-xenergy-u7", "kingref")
    return got, uploaded, loc


def test_reupload_binds_the_recorded_selection(tmp_path, monkeypatch):
    lane_b_dispatch_fields = _fields()

    fields = lane_b_dispatch_fields(TRAINED, 9594000, {})
    got, uploaded, loc = _reupload(tmp_path, monkeypatch, _record(tmp_path, **fields),
                                   lane_b_dispatch_fields(TICK, 9594000, {}))
    assert got == "entry" and len(uploaded) == 1
    assert loc.lane_b_reused == fields


def test_reupload_refuses_an_unattributable_record(tmp_path, monkeypatch):
    lane_b_dispatch_fields = _fields()

    tick = lane_b_dispatch_fields(TICK, 9597600, {})
    got, uploaded, _ = _reupload(tmp_path, monkeypatch, _record(tmp_path), tick)
    assert got is None and uploaded == []                 # legacy: selection unknown


def test_reupload_across_an_era_keeps_the_trained_contract_block(tmp_path, monkeypatch):
    """The 2026-11-27 leg: trained at 9594000, re-dispatched in the next era
    (9597600) — the digest binds the contract; the record's fields win."""
    lane_b_dispatch_fields = _fields()

    fields = lane_b_dispatch_fields(TRAINED, 9594000, {})
    got, uploaded, loc = _reupload(tmp_path, monkeypatch, _record(tmp_path, **fields),
                                   lane_b_dispatch_fields(TICK, 9597600, {}))
    assert got == "entry" and loc.lane_b_reused == fields


def test_pending_fields_are_json_canonical():
    lane_b_dispatch_fields, lane_b_record_rows = _fields(), _rows()

    f = lane_b_dispatch_fields({6: [84, 61], 0: [31, 23]}, None, {1: [], 2: [7]})
    assert json.dumps(f, sort_keys=True) == (
        '{"contract_block": null, "expert_exclude": {"2": [7]}, '
        '"expert_selection": {"0": [23, 31], "6": [61, 84]}}')
    assert lane_b_record_rows(f["expert_selection"]) == {0: [23, 31], 6: [61, 84]}
